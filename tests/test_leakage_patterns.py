"""Static guard against the documented catastrophic leakage patterns.

CLAUDE.md rule 1: ``df.groupby(key)[col].cummax().shift(1)`` is forbidden —
the ``.shift(1)`` operates on the flattened Series, not per group, so each row
receives a value from the *previous row's* horse. With raw results data ordered
by finish position within a race, that leaks the outcome. The safe form is
``groupby(key)[col].transform(lambda x: x.cummax().shift(1))``, where the
shift happens inside the per-group callable.

This bug class was found live (``horse_max_prize_contested`` inflated
no-market top-1 accuracy from 0.29 to 0.52), so it stays under test.

Three things matter beyond the single documented spelling:

* **Any grouped-then-globally-shifted expression leaks the same way.**
  ``.transform("cummax").shift(1)`` and ``.rolling(3).mean().shift(1)`` are
  exactly as broken as ``.cummax().shift(1)``, so they are all forbidden.
  The previous version of this lint skipped every line containing the word
  ``transform``, which made the first of those invisible to it.
* **Research scripts count.** Every phantom edge this project has chased was
  measured by a research script, not by ``src/``. A leak there does not
  ship, but it does burn weeks and can get a dead idea promoted, so the scan
  covers ``scripts/`` and every file under ``research/`` too.
* **The check reads the syntax tree, not the text.** Telling the safe form
  from the broken one means knowing whether the ``.shift`` sits inside the
  per-group callable — which is a question about nesting that no regex over a
  line can answer. Walking the AST receiver chain answers it exactly: the
  inner shift of ``transform(lambda x: x.cummax().shift(1))`` is applied to
  ``x``, with no ``groupby`` in its chain, so it is accepted; the shift in
  ``transform(lambda x: x.cummax()).shift(1)`` is applied to the grouped
  result, so it is flagged.
"""

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

#: Directories scanned, in addition to the project-root ``*.py`` files.
SCAN_DIRS = ("src", "scripts")
#: Scanned with subfolders: experiments live in their own folders under research/.
RECURSIVE_SCAN_DIRS = ("research",)

#: Per-group cumulative/window operations. A global ``.shift`` applied to the
#: result of any of these, on a grouped object, crosses group boundaries.
CUMULATIVE = {"cumsum", "cummax", "cummin", "cumprod", "cumcount", "cumany", "cumall"}
WINDOWED = {"rolling", "expanding", "ewm"}


def _receiver_chain(node: ast.AST) -> list[tuple[str, ast.AST]]:
    """``(method name, call node)`` pairs for the chain *node* was built from.

    Walks outward-in through attributes, calls and subscripts and stops at the
    first plain name/expression. It never descends into a call's *arguments*,
    so a ``lambda`` body is deliberately out of scope: code inside the lambda
    runs per group and is analysed on its own terms.
    """
    chain: list[tuple[str, ast.AST]] = []
    while True:
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Attribute):
                chain.append((node.func.attr, node))
                node = node.func.value
            else:
                return chain
        elif isinstance(node, ast.Attribute):
            chain.append((node.attr, node))
            node = node.value
        elif isinstance(node, ast.Subscript):
            node = node.value
        else:
            return chain


def _leak_reason(shift_call: ast.Call) -> str | None:
    """Why *shift_call* leaks across groups, or ``None`` if it is fine."""
    chain = _receiver_chain(shift_call.func.value)
    names = {name for name, _ in chain}
    if "groupby" not in names:
        return None  # not a grouped object: a plain shift is the caller's business
    for name, node in chain:
        if name in CUMULATIVE:
            return f"grouped .{name}() then global .shift()"
        if name in WINDOWED:
            return f"grouped .{name}() then global .shift()"
        if name == "transform":
            # transform(lambda …) still returns a group-aligned Series, so a
            # shift applied to *it* is global just the same.
            return "grouped .transform() then global .shift()"
    return None


def _offenders_in_source(source: str, label: str) -> list[str]:
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:  # pragma: no cover - a broken file is its own error
        return [f"{label}: could not parse ({exc})"]
    found = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "shift"
        ):
            reason = _leak_reason(node)
            if reason:
                found.append(f"{label}:{node.lineno}: {reason}")
    return found


def _scanned_files(root: Path):
    for name in SCAN_DIRS:
        yield from sorted((root / name).glob("*.py"))
    for name in RECURSIVE_SCAN_DIRS:
        yield from sorted((root / name).rglob("*.py"))
    yield from sorted(root.glob("*.py"))


def _offenders(root: Path = ROOT) -> list[str]:
    found = []
    for path in _scanned_files(root):
        found += _offenders_in_source(
            path.read_text(encoding="utf-8"), path.relative_to(root).as_posix(),
        )
    return found


def test_no_grouped_history_feature_shifts_globally():
    offenders = _offenders()
    assert not offenders, (
        "Forbidden grouped-then-global-shift pattern — the shift crosses "
        "horses/races and leaks the outcome. Move the shift inside the group, "
        "e.g. transform(lambda x: x.cummax().shift(1)):\n  "
        + "\n  ".join(offenders)
    )


def test_lint_flags_every_leaky_spelling():
    """A lint that cannot fail is not a guard — prove each shape bites."""
    leaky = (
        "out = df.groupby('horse')['prize'].cummax().shift(1)",
        "out = df.groupby('horse')['prize'].cumsum().shift(1)",
        "out = df.groupby('horse')['prize'].transform('cummax').shift(1)",
        "out = df.groupby('horse')['prize'].rolling(3).mean().shift(1)",
        "out = df.groupby('horse')['prize'].expanding().mean().shift(1)",
        # The shift escapes the per-group callable by sitting outside it.
        "out = df.groupby('horse')['prize'].transform(lambda x: x.cummax()).shift(1)",
        # Chain split across lines, as it would really be written.
        "out = (\n    df.groupby(['horse', 'track'])['prize']\n"
        "    .cummax()\n    .shift(1)\n)",
    )
    for source in leaky:
        assert _offenders_in_source(source, "t.py"), f"not flagged: {source}"


def test_lint_accepts_correct_history_features():
    """Guard against false positives that would block the correct idioms."""
    safe = (
        # The documented safe form: the shift runs inside the group.
        "a = df.groupby('horse')['prize'].transform(lambda x: x.cummax().shift(1))",
        "b = df.groupby('horse')['prize'].transform(lambda x: x.rolling(3).mean().shift(1))",
        # groupby().shift() is already per-group.
        "c = df.groupby('horse')['prize'].shift(1)",
        # Ungrouped shifts are not this lint's business.
        "d = df['prize'].cummax().shift(1)",
        "e = series.shift(1)",
    )
    for source in safe:
        assert not _offenders_in_source(source, "t.py"), f"false positive: {source}"
