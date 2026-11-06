"""Guards the invariant that makes "build once, append forever" valid.

A settled featured row must never change as later races arrive. That holds only
while every feature is backward-or-guarded (taxonomy classes R/W/L/B/C/F in
docs/architecture/FE_TAXONOMY.md). Two things can break it and are tested here:

1. an *un-guarded* forward reference (``shift(-1)`` outside the one date-guarded
   collateral-form stage), and
2. the FE signature failing to change when FE code/config changes (which would
   let the FeatureStore serve stale features after an FE edit).
"""

import re
from pathlib import Path

import config
from src.featured_dataset import fe_signature

FE_SRC = Path(__file__).resolve().parent.parent / "src" / "feature_engineer.py"

# The only stage allowed a forward shift — it date-guards (next_date < cur_date),
# so it stays a pure function of the past (taxonomy class F).
_ALLOWED_FORWARD_SHIFT_FN = "add_collateral_form_features"


def _function_span(text: str, fn_name: str) -> tuple[int, int]:
    """(start, end) line numbers of a top-level ``def fn_name`` block."""
    lines = text.splitlines()
    start = None
    for i, ln in enumerate(lines):
        if re.match(rf"^def {re.escape(fn_name)}\b", ln):
            start = i
            break
    assert start is not None, f"{fn_name} not found"
    end = len(lines)
    for j in range(start + 1, len(lines)):
        if re.match(r"^def \w", lines[j]):       # next top-level def
            end = j
            break
    return start + 1, end  # 1-based inclusive start


def test_no_unguarded_forward_shift():
    text = FE_SRC.read_text(encoding="utf-8")
    lo, hi = _function_span(text, _ALLOWED_FORWARD_SHIFT_FN)
    offenders = []
    for n, line in enumerate(text.splitlines(), 1):
        if line.lstrip().startswith("#"):
            continue
        if ".shift(-1)" in line and not (lo <= n <= hi):
            offenders.append(f"feature_engineer.py:{n}: {line.strip()[:100]}")
    assert not offenders, (
        "Un-guarded forward reference (shift(-1)) outside "
        f"{_ALLOWED_FORWARD_SHIFT_FN} — this breaks append-only/leakage. "
        "Add a strict date guard and allowlist the stage:\n" + "\n".join(offenders)
    )


class TestFeSignature:
    def test_stable_when_unchanged(self):
        assert fe_signature() == fe_signature()

    def test_changes_with_config(self, monkeypatch):
        before = fe_signature()
        monkeypatch.setattr(config, "TE_WINDOW_DAYS", config.TE_WINDOW_DAYS + 1)
        assert fe_signature() != before

    def test_short_hex(self):
        sig = fe_signature()
        assert isinstance(sig, str) and len(sig) == 16
