"""Build and inspect one fixed historical performance proposal before fitting."""
from pathlib import Path
import json
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd

from src.deployment_validation import file_sha256
from src.established_form_features import race_discipline
from src.handicap_performance_research import FEATURES, handicap_performance_features, performance_observations

OUTPUT = ROOT / "data/audit/handicap_performance_031"
REFERENCE = ROOT / "data/audit/established_form_029/native_canonical"


def main():
    began = time.monotonic()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    columns = ["race_id", "horse_id", "horse_name", "race_date", "race_name", "race_type", "surface",
        "age", "weight_lbs", "has_observed_weight", "official_rating", "has_official_rating", "handicap",
        "distance_furlongs", "finish_position", "finish_pos_label", "num_runners", "won", "beaten_lengths"]
    frame = pd.read_parquet(ROOT / "data/processed/feature_store/processed.parquet", columns=columns,
        filters=[("race_date", "<=", pd.Timestamp("2026-03-31"))])
    features = handicap_performance_features(frame)
    projection = pd.concat([frame[["race_id", "horse_name", "race_date"]], features], axis=1)
    if projection.duplicated(["race_id", "horse_name"]).any():
        raise ValueError("Historical projection has duplicate runners")
    projection.to_parquet(OUTPUT / "projection.parquet", index=False)
    obs = performance_observations(frame)
    context = pd.concat([frame, obs, features], axis=1)
    context["discipline"] = race_discipline(frame)
    # No fitting/evaluation here. Inspect past-figure support and raw magnitudes.
    groups = {"all": np.ones(len(frame), bool),
        "Q4_2025": frame.race_date.between("2025-10-01", "2025-12-31"),
        "Q1_2026": frame.race_date.between("2026-01-01", "2026-03-31")}
    coverage = {}
    for name, mask in groups.items():
        selected = context.loc[mask]
        coverage[name] = {"runners": len(selected), "observations": int(selected.performance.notna().sum()),
            "features": {c: {"known": int(selected[c].notna().sum()),
                "quantiles": selected[c].quantile([0, .01, .5, .99, 1]).to_dict()} for c in FEATURES},
            "by_discipline": selected.groupby("discipline", observed=True).performance.agg(["size", "count"]).to_dict()}
    context.loc[frame.race_date.between("2025-10-01", "2026-03-31")].to_parquet(OUTPUT / "context.parquet", index=False)
    context.loc[context.performance.notna()].sort_values("performance").iloc[np.r_[0:10, -10:0]].to_csv(
        OUTPUT / "extreme_observations.csv", index=False)
    # Prefix/append equality is checked on real data without any model fit.
    cut = pd.Timestamp("2025-12-31")
    mask = frame.race_date.le(cut)
    prior, state = handicap_performance_features(frame.loc[mask], return_state=True)
    later = handicap_performance_features(frame.loc[~mask], state=state)
    pd.testing.assert_frame_equal(features.loc[mask], prior)
    pd.testing.assert_frame_equal(features.loc[~mask], later)
    baseline = json.loads((REFERENCE / "manifest.json").read_text())["baseline_features"]
    candidate = ROOT / "feature_sets/handicap_performance_031.txt"
    candidate.write_text("# Research only: preset historical handicap performance proposal031\n"+
        "\n".join(baseline+list(FEATURES))+"\n", encoding="utf-8")
    report = {"scope": "No-fit feature audit; original store and production model untouched",
        "features": list(FEATURES), "coverage": coverage, "prior_day_and_seeded_append_exact": True,
        "performance_quantiles": obs.performance.quantile([0, .001, .01, .5, .99, .999, 1]).to_dict(),
        "sources": {p: file_sha256(ROOT/p) for p in ("src/handicap_performance_research.py", "scripts/prepare_handicap_performance.py")},
        "projection_sha256": file_sha256(OUTPUT / "projection.parquet"), "elapsed_seconds": time.monotonic()-began,
        "limitations": ["Approximate distance-only poundage; no pace, ground, time or discretionary adjustments.",
            "Only mature handicap performances with three fully observed mature top finishers.",
            "Anchor assumes the median of the first three finishers ran to its observed mark.",
            "Declared weight, before apprentice claims; no reconstruction from current profile ratings."]}
    (OUTPUT / "audit.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
