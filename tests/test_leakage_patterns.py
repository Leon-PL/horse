"""Static guard against the documented catastrophic leakage pattern.

CLAUDE.md rule 1: ``df.groupby(key)[col].cummax().shift(1)`` is forbidden
— the ``.shift(1)`` operates on the flattened Series, not per group, so
each row receives a value from the *previous row's* horse. With raw
results data ordered by finish position within a race, that leaks the
outcome. The safe form is
``groupby(key)[col].transform(lambda x: x.cummax().shift(1))``.

This bug class was found live (horse_max_prize_contested inflated
no-market top-1 accuracy from 0.29 to 0.52), so it stays under test.
"""

import re
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "src"

# groupby(...)[...].cum*( ... ).shift(   — on one logical line, without
# transform in the chain.
FORBIDDEN = re.compile(
    r"groupby\([^)]*\)(?:\[[^\]]*\])?\.cum\w+\([^)]*\)\.shift\("
)


def _logical_lines(text: str):
    """Join physical lines so simple chained calls split over lines are seen."""
    joined = re.sub(r"\(\s*\n\s*", "(", text)
    joined = re.sub(r"\n\s*\.", ".", joined)
    return joined.splitlines()


def test_no_global_shift_after_grouped_cumulative():
    offenders = []
    for path in SRC.glob("*.py"):
        for lineno, line in enumerate(_logical_lines(path.read_text(encoding="utf-8")), 1):
            if "transform" in line or line.lstrip().startswith("#"):
                continue
            if FORBIDDEN.search(line):
                offenders.append(f"{path.name}:{lineno}: {line.strip()[:120]}")
    assert not offenders, (
        "Forbidden groupby().cum*().shift() pattern (global shift leaks "
        "across horses — use transform(lambda x: x.cum*().shift(1))):\n"
        + "\n".join(offenders)
    )
