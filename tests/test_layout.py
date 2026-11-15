"""Guard: imports between the top-level folders run one way only.

Research lives in ``research/`` and may import ``src``; nothing in ``src/``,
``scripts/``, ``tests/`` or ``config.py`` may import ``research``.
Before 2026-11-07 research and pipeline code shared ``src/`` and ``scripts/``,
and a pipeline import of a research helper made neither side removable.

``scripts/`` holds command wrappers over ``src``, so ``src`` never imports
``scripts``. Until 2026-11-07 the store builder lived in a script and
``src/store_refresh.py`` imported it from there.
"""
import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CORE = ("src", "scripts", "tests")


def _python_files(*folders):
    for name in folders:
        yield from (p for p in sorted((ROOT / name).rglob("*.py")) if "__pycache__" not in p.parts)


def _core_files():
    yield from _python_files(*CORE)
    yield ROOT / "config.py"


def _imports_of(package: str, source: str, label: str) -> list[str]:
    """Absolute imports of *package* (or its submodules) in *source*."""
    found = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            names = [node.module]
        elif isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        else:
            continue
        found += [f"{label}:{node.lineno} imports {name}"
                  for name in names if name == package or name.startswith(package + ".")]
    return found


def _offenders(files, package: str) -> list[str]:
    return [hit for path in files
            for hit in _imports_of(package, path.read_text(encoding="utf-8"),
                                   path.relative_to(ROOT).as_posix())]


def test_core_code_never_imports_research():
    offenders = _offenders(_core_files(), "research")
    assert not offenders, "Pipeline code imports research code:\n" + "\n".join(offenders)


def test_pipeline_never_imports_commands():
    offenders = _offenders([*_python_files("src"), ROOT / "config.py"], "scripts")
    assert not offenders, ("Pipeline code imports a command from scripts/; move the shared "
                           "code into src/:\n" + "\n".join(offenders))


def test_guard_recognises_package_imports_only():
    source = ("import researcher\nfrom src import research_notes\n"
              "from research.pace_shape_064 import run\nimport research\n"
              "from scripts.build_feature_store_batched import main\n")
    assert _imports_of("research", source, "probe.py") == [
        "probe.py:3 imports research.pace_shape_064", "probe.py:4 imports research"]
    assert _imports_of("scripts", source, "probe.py") == [
        "probe.py:5 imports scripts.build_feature_store_batched"]
