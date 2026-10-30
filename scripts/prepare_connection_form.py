"""Audit a fixed market-adjusted connection-form proposal on frozen history.

No canonical writes, live-store access, collection, scoring or learner fitting.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd

from src.connection_market_research import FEATURES, PRIOR_STARTS, WINDOW_DAYS, complete_market_residuals, connection_market_features
from src.deployment_validation import file_sha256 as sha
from src.prior_market_research import past_market_observations

OUTPUT = ROOT / "data/audit/connection_form_039"
BASELINE = ROOT / "data/audit/market_transfer_034/baseline"
RAW = ROOT / "data/audit/market_transfer_034/source_inputs.parquet"
KEY = ["race_id", "horse_name", "race_date"]
SOURCES = ["src/connection_market_research.py", "src/forecast_evaluation.py", "src/result_status.py",
           "src/prior_market_research.py", "scripts/prepare_connection_form.py",
           "scripts/screen_frozen_win_candidate.py", "tests/test_connection_market_research.py"]


def write(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)


def main():
    started = time.monotonic()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    if (OUTPUT / "plan.json").exists():
        raise ValueError("Preserve the registered case; choose a new research case")
    baseline = json.loads((BASELINE / "manifest.json").read_text())
    if len(baseline["baseline_features"]) != 154 or sha(BASELINE / "inputs.parquet") != baseline["inputs_sha256"]:
        raise ValueError("Use the frozen corrected154-input baseline")
    source_hashes = {name: sha(ROOT / name) for name in SOURCES}
    plan = {"registered_at_utc": datetime.now(timezone.utc).isoformat(),
        "hypothesis": "Recent connection win rates confound the quality of assigned runners; test wins minus earlier complete-book market expectations instead.",
        "window_days": WINDOW_DAYS, "prior_neutral_starts": PRIOR_STARTS, "added_features": list(FEATURES),
        "training_dates": ["2023-09-24", "2025-09-23"], "screen_dates": ["2025-10-01", "2025-12-31"],
        "advance_rule": "Both overall and established-handicap race log loss lower, neither runner Brier worse, and overall date-cluster95%LL upper bound below zero before any costly native follow-up. No tuning or subset rescue.",
        "baseline": str(BASELINE.relative_to(ROOT)), "raw": str(RAW.relative_to(ROOT)),
        "source_sha256": source_hashes,
        "input_sha256": {str(p.relative_to(ROOT)): sha(p) for p in [RAW, BASELINE / "inputs.parquet", BASELINE / "manifest.json"]},
        "fit_budget": "Exact cached154-input400-tree baseline; at most one158-input400-tree win candidate; no place/calibration/current-data/quarter fits.",
        "scope": "Previously inspected development. No current-race prices as inputs; no production or forward-recorder writes. Other agent's feature migration may proceed independently."}
    write(OUTPUT / "plan.json", plan)
    frame = pd.read_parquet(RAW)
    connections = pd.read_parquet(BASELINE / "inputs.parquet", columns=KEY + ["trainer", "jockey", "handicap", "horse_prev_races"])
    frame = frame.merge(connections, on=KEY, how="left", validate="one_to_one", indicator=True)
    if not frame.pop("_merge").eq("both").all():
        raise ValueError("Frozen connection identities do not cover raw inputs")
    frame.to_parquet(OUTPUT / "source_inputs.parquet", index=False)
    book = complete_market_residuals(frame)
    reference = past_market_observations(frame).logrel
    expected = np.exp(reference) / pd.to_numeric(frame.num_runners)
    np.testing.assert_allclose(book.probability, expected, atol=1e-15, rtol=0, equal_nan=True)
    valid = book.probability.notna()
    np.testing.assert_allclose(book.loc[valid].groupby(frame.loc[valid, "race_id"]).probability.sum(), 1., atol=1e-12, rtol=0)
    np.testing.assert_allclose(book.loc[valid].groupby(frame.loc[valid, "race_id"]).residual.sum(), 0., atol=1e-12, rtol=0)
    full = connection_market_features(frame)
    earlier = frame.race_date.le("2025-09-23")
    prefix, seed = connection_market_features(frame.loc[earlier], return_state=True)
    pd.testing.assert_frame_equal(prefix, full.loc[earlier])
    pd.testing.assert_frame_equal(connection_market_features(frame.loc[~earlier], state=seed), full.loc[~earlier])
    # Real-field permutation and future mutation checks use a bounded source
    # prefix, not a fit or selection on favourable races.
    pilot = frame.loc[frame.race_date.between("2025-08-01", "2025-10-07")].copy()
    original = connection_market_features(pilot)
    pd.testing.assert_frame_equal(connection_market_features(pilot.sample(frac=1, random_state=42)).sort_index(), original.sort_index(), atol=1e-14, rtol=0)
    changed = pilot.copy()
    later = changed.race_date.ge("2025-10-01")
    changed.loc[later, "odds"] = np.nan
    changed.loc[later, "won"] = 0
    pd.testing.assert_frame_equal(connection_market_features(changed).loc[~later], original.loc[~later])
    projection = pd.concat([frame[KEY], full], axis=1)
    projection.to_parquet(OUTPUT / "projection.parquet", index=False)
    candidate = OUTPUT / "features.txt"
    candidate.write_text("# Research only: fixed connection-market form039\n" + "\n".join(baseline["baseline_features"] + list(FEATURES)) + "\n", encoding="utf-8")
    q4 = frame.race_date.between("2025-10-01", "2025-12-31")
    established = frame.groupby("race_id").horse_prev_races.transform("min").ge(5)
    handicap = frame.groupby("race_id").handicap.transform("min").eq(1)
    coverage = {}
    for label, mask in [("all_Q4", q4), ("established_handicaps_Q4", q4 & established & handicap)]:
        part = full.loc[mask]
        coverage[label] = {"runners": len(part), "fields": int(frame.loc[mask, "race_id"].nunique()),
            "known_excess_runners": {name: int(part[name].notna().sum()) for name in FEATURES[:2]},
            "median_priced_starts": {name: float(np.expm1(part[name]).median()) for name in FEATURES[2:]}}
    if any(sha(ROOT / name) != digest for name, digest in source_hashes.items()):
        raise ValueError("Research source changed during preparation")
    with zipfile.ZipFile(OUTPUT / "source_snapshot.zip", "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for name in SOURCES:
            archive.write(ROOT / name, name)
    report = {"prepared_at_utc": datetime.now(timezone.utc).isoformat(), "rows": len(frame),
        "fields": int(frame.race_id.nunique()), "complete_priced_rows": int(valid.sum()),
        "canonical_price_math_matches": True, "full_prefix_seed_exact": True,
        "real_permutation_equal_at_1e_14": True, "future_independent": True,
        "source_sha256": source_hashes, "coverage": coverage,
        "output_sha256": {name: sha(OUTPUT / name) for name in ["source_inputs.parquet", "projection.parquet", "features.txt", "source_snapshot.zip", "plan.json"]},
        "elapsed_seconds": time.monotonic() - started,
        "limitations": ["Connection identity uses conservative exact normalized source names; aliases are not guessed.",
                        "Earlier SP expectations are approximate market beliefs; excess wins are descriptors, not causal ability or profits.",
                        "Support counts priced starts only. Missing books and unknown connections do not become neutral evidence."]}
    write(OUTPUT / "preparation.json", report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
