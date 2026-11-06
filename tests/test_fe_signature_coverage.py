"""The feature signature must cover every module that can change a feature.

``fe_signature()`` is the project's guard against serving features built by
different code than the ones a model was trained on. It hashes the modules
listed in ``_FE_SOURCE_FILES`` — so a module that affects feature *values* but
is missing from that list makes the guard silently pass while the store holds
stale rows. That failure mode is not hypothetical: the 2026-06 "+15% place
market edge" was a stale-store artifact that vanished once the store was
rebuilt correctly (and the missing module then was the ATR sectional merge).

This test walks the import graph from the FE entry points and requires every
reachable ``src`` module to be either registered in the signature or listed in
``FE_SIGNATURE_EXEMPT`` with a reason. Adding an FE dependency therefore forces
a deliberate choice instead of a silent omission.
"""

import ast
import os
from pathlib import Path

from src.featured_dataset import FE_SIGNATURE_EXEMPT, _FE_SOURCE_FILES

SRC = Path(__file__).resolve().parent.parent / "src"

# Where feature building starts. Anything these can reach may run during a
# build and is therefore in scope.
FE_ENTRY_POINTS = (
    "featured_dataset",
    "feature_store",
    "feature_engineer",
    "data_processor",
    "ratings",
    "fe_incremental",
)


def _module_path(module: str) -> Path | None:
    path = SRC.joinpath(*module.split("."))
    if path.with_suffix(".py").is_file():
        return path.with_suffix(".py")
    if (path / "__init__.py").is_file():
        return path / "__init__.py"
    return None


def _src_imports(module: str) -> set[str]:
    """``src.*`` modules imported by *module*, including function-local imports."""
    path = _module_path(module)
    if path is None:
        return set()
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if node.module == "src" or (node.module and node.module.startswith("src.")):
                prefix = node.module.removeprefix("src").lstrip(".")
                if prefix:
                    found.add(prefix)
                for alias in node.names:
                    found.add(".".join(filter(None, (prefix, alias.name))))
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith("src."):
                    found.add(alias.name.split(".", 1)[1])
    found = {m for m in found if _module_path(m) is not None}
    # Importing a leaf runs its package initializers too.
    for leaf in list(found):
        parts = leaf.split(".")
        found.update(".".join(parts[:n]) for n in range(1, len(parts)))
    return found


def _reachable() -> set[str]:
    seen: set[str] = set()
    stack = list(FE_ENTRY_POINTS)
    while stack:
        module = stack.pop()
        if module in seen:
            continue
        seen.add(module)
        stack.extend(_src_imports(module) - seen)
    return seen


def _registered() -> set[str]:
    return {f.removeprefix("src/").removesuffix(".py").replace("/", ".")
            for f in _FE_SOURCE_FILES if f.startswith("src/") and f.endswith(".py")}


def test_every_fe_dependency_is_hashed_or_explicitly_exempt():
    unclassified = sorted(_reachable() - _registered() - set(FE_SIGNATURE_EXEMPT))
    assert not unclassified, (
        "These modules are reachable from the feature-engineering entry points "
        "but are neither hashed by fe_signature() nor exempt, so editing them "
        "would not invalidate the feature store:\n  "
        + "\n  ".join(unclassified)
        + "\n\nAdd each to _FE_SOURCE_FILES (if it can change a feature value) "
        "or to FE_SIGNATURE_EXEMPT with a reason (if it cannot)."
    )


def test_exemptions_are_justified_and_not_also_hashed():
    """An exemption must name a real module, carry a reason, and not be hashed."""
    for module, reason in FE_SIGNATURE_EXEMPT.items():
        assert _module_path(module) is not None, f"stale exemption for {module!r}"
        assert reason.strip(), f"exemption for {module!r} needs a reason"
    overlap = sorted(set(FE_SIGNATURE_EXEMPT) & _registered())
    assert not overlap, f"listed as both hashed and exempt: {overlap}"


def test_registered_sources_exist():
    """A typo'd path would hash as 'missing' forever and never invalidate."""
    root = SRC.parent
    missing = [rel for rel in _FE_SOURCE_FILES if not (root / rel).is_file()]
    assert not missing, f"_FE_SOURCE_FILES names files that do not exist: {missing}"


def test_import_walk_follows_collection_package_and_compatibility_paths():
    assert "ingestion.atr_tabs" in _src_imports("atr_features")
    assert "ingestion" in _src_imports("atr_features")
    assert "ingestion.racing_tv" in _src_imports("ingestion.pipeline")
    assert "ingestion.storage" in _src_imports("ingestion.sporting_life")


def _relocate_project(monkeypatch, destination: Path) -> None:
    """Point config at *destination* as if the whole checkout lived there.

    A real move relocates ``BASE_DIR`` and every path derived from it
    together, so the test must move them together too.
    """
    import config

    monkeypatch.setattr(config, "BASE_DIR", str(destination))
    for key in ("DATA_DIR", "RAW_DATA_DIR", "PROCESSED_DATA_DIR", "MODELS_DIR"):
        current = getattr(config, key, None)
        if isinstance(current, str):
            rel = os.path.relpath(current, SRC.parent)
            monkeypatch.setattr(config, key, str(destination / rel))


def test_signature_survives_a_relocated_checkout(tmp_path, monkeypatch):
    """Moving the project must not invalidate a multi-GB store.

    The signature answers "was this built by the same code?", not "where does
    it live?" — absolute paths in the hash would force a spurious rebuild on
    every clone, move or second machine.
    """
    from src import featured_dataset as fd

    original = fd.fe_signature()

    scratch = tmp_path / "proj"
    (scratch / "src").mkdir(parents=True)
    # Copy bytes, not text: the hash is over raw bytes and Windows text mode
    # would rewrite the line endings.
    for rel in _FE_SOURCE_FILES:
        (scratch / rel).parent.mkdir(parents=True, exist_ok=True)
        (scratch / rel).write_bytes((SRC.parent / rel).read_bytes())
    _relocate_project(monkeypatch, scratch)

    assert fd.fe_signature() == original, "relocating the checkout must not invalidate"


def test_signature_changes_when_a_registered_source_changes(tmp_path, monkeypatch):
    """Guard the mechanism itself, not just the list."""
    from src import featured_dataset as fd

    scratch = tmp_path / "proj"
    (scratch / "src").mkdir(parents=True)
    for rel in _FE_SOURCE_FILES:
        (scratch / rel).parent.mkdir(parents=True, exist_ok=True)
        (scratch / rel).write_bytes((SRC.parent / rel).read_bytes())
    _relocate_project(monkeypatch, scratch)

    before = fd.fe_signature()
    target = scratch / "src" / "atr_features.py"
    target.write_bytes(target.read_bytes() + b"\n_EDIT = 1\n")
    assert fd.fe_signature() != before, "editing a hashed source must invalidate"
    target.write_bytes(target.read_bytes().replace(b"_EDIT = 1", b"_EDIT = 2"))
    assert fd.fe_signature() != before, "changing a constant must invalidate"


def test_signature_ignores_comments_docstrings_and_formatting(tmp_path, monkeypatch):
    """None of these can change a feature value, and each used to cost a rebuild."""
    from src import featured_dataset as fd

    scratch = tmp_path / "proj"
    (scratch / "src").mkdir(parents=True)
    for rel in _FE_SOURCE_FILES:
        (scratch / rel).parent.mkdir(parents=True, exist_ok=True)
        (scratch / rel).write_bytes((SRC.parent / rel).read_bytes())
    _relocate_project(monkeypatch, scratch)

    before = fd.fe_signature()
    target = scratch / "src" / "utils.py"
    source = target.read_bytes()
    target.write_bytes(b'"""Reworded module docstring."""\n# a new comment\n\n\n'
                       + source.split(b'"""', 2)[2] + b"\n\n# trailing note\n")
    assert fd.fe_signature() == before, "comments, docstrings and blank lines must not invalidate"
    target.write_bytes(source + b"\nimport config\n_READ = config.ELO_K_BASE\n")
    assert fd.fe_signature() != before


def test_signature_ignores_line_endings(tmp_path, monkeypatch):
    """git autocrlf rewrites line endings on checkout, stash and worktree add.

    The working tree holds a mix of LF and CRLF sources, so a raw-byte hash
    changed whenever git touched a file — forcing a ~1 hour rebuild for a
    change that cannot alter any feature value.
    """
    from src import featured_dataset as fd

    scratch = tmp_path / "proj"
    (scratch / "src").mkdir(parents=True)
    for rel in _FE_SOURCE_FILES:
        (scratch / rel).parent.mkdir(parents=True, exist_ok=True)
        (scratch / rel).write_bytes((SRC.parent / rel).read_bytes())
    _relocate_project(monkeypatch, scratch)

    def rewrite(eol: bytes) -> str:
        for rel in _FE_SOURCE_FILES:
            raw = (scratch / rel).read_bytes().replace(b"\r\n", b"\n")
            (scratch / rel).write_bytes(raw.replace(b"\n", eol))
        return fd.fe_signature()

    assert rewrite(b"\n") == rewrite(b"\r\n"), "line endings must not invalidate"
