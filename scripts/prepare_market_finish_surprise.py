"""Source audit of one market-relative finishing-history proposal; no fitting."""
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd

from src.deployment_validation import file_sha256 as sha
from src.featured_dataset import fe_signature
from src.forecast_evaluation import valid_race_mask
from src.market_finish_surprise_research import FEATURES, finish_surprise_observations, market_finish_surprise_features
from src.run_store import get_production_run_id

OUTPUT = ROOT / "data/audit/market_finish_surprise_035"
BASE = ROOT / "data/audit/market_transfer_034"
KEY = ["race_id", "horse_name", "race_date"]
SOURCES = ["src/market_finish_surprise_research.py", "src/prior_market_research.py",
    "src/established_form_features.py", "src/result_status.py", "src/forecast_evaluation.py",
    "scripts/prepare_market_finish_surprise.py", "scripts/screen_market_finish_surprise.py"]


def main():
    started = time.monotonic()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    if (OUTPUT / "preparation.json").exists():
        raise ValueError("Preserve the completed proposal and its registered rule")
    before = {"production": get_production_run_id(), "fe_signature": fe_signature()}
    sources = {p: sha(ROOT / p) for p in SOURCES}
    reference = json.loads((BASE / "baseline/manifest.json").read_text())
    if sha(BASE / "baseline/inputs.parquet") != reference["inputs_sha256"]:
        raise ValueError("Frozen corrected154 inputs changed")
    frame = pd.read_parquet(BASE / "source_inputs.parquet")
    observations = finish_surprise_observations(frame)
    finite = observations.finish_surprise.notna()
    if not observations.loc[finite, "finish_surprise"].between(-1, 1).all():
        raise ValueError("Pairwise residual exceeds its mathematical bounds")
    sums = observations.finish_surprise.groupby(frame.race_id).sum()
    np.testing.assert_allclose(sums, 0, atol=1e-10, rtol=0)
    if observations.loc[pd.to_numeric(frame.finish_position).le(0), "finish_surprise"].notna().any():
        raise ValueError("A nonfinisher received an ability descriptor")
    added = market_finish_surprise_features(frame)
    past = frame.race_date.le("2025-09-23")
    prefix, state = market_finish_surprise_features(frame.loc[past], return_state=True)
    pd.testing.assert_frame_equal(prefix, added.loc[past])
    pd.testing.assert_frame_equal(market_finish_surprise_features(frame.loc[~past], state=state), added.loc[~past])
    pd.concat([frame[KEY], added], axis=1).to_parquet(OUTPUT / "projection.parquet", index=False)
    context = pd.read_parquet(BASE / "baseline/inputs.parquet", columns=KEY + ["horse_prev_races", "handicap"])
    combined = frame.merge(context, on=KEY, validate="one_to_one")
    pd.testing.assert_frame_equal(combined[KEY], frame[KEY])
    combined = pd.concat([combined, added], axis=1)
    valid = valid_race_mask(combined)
    established = combined.groupby("race_id").horse_prev_races.transform("min").ge(5)
    handicap = combined.groupby("race_id").handicap.transform("min").eq(1)
    cohorts = {}
    for name, begin, end in (("Q4_screen", "2025-10-01", "2025-12-31"),
                              ("Q1_description", "2026-01-01", "2026-03-31")):
        part = combined.loc[valid & established & handicap & combined.race_date.between(begin, end)]
        cohorts[name] = {"races": int(part.race_id.nunique()), "runners": len(part),
            "mean_history_known": int(part[FEATURES[0]].notna().sum()),
            "latest_history_known": int(part[FEATURES[1]].notna().sum()),
            "zero_support": int(part[FEATURES[2]].eq(0).sum())}
    cases = pd.read_csv(ROOT / "data/audit/prior_market_feasibility_033/error_case_inputs.csv", parse_dates=["race_date"])
    cases[KEY + ["case_group"]].merge(combined[KEY + list(FEATURES)], on=KEY,
        validate="one_to_one").to_csv(OUTPUT / "fixed_case_inputs.csv", index=False)
    observed = pd.concat([frame[KEY], observations], axis=1)
    observed.to_parquet(OUTPUT / "past_observations.parquet", index=False)
    candidate = ROOT / "feature_sets/market_finish_surprise_035.txt"
    candidate.write_text("# Research only: past finishing performance relative to price expectations035\n" +
                         "\n".join(reference["baseline_features"] + list(FEATURES)) + "\n", encoding="utf-8")
    after = {"production": get_production_run_id(), "fe_signature": fe_signature()}
    if before != after or any(sha(ROOT / p) != digest for p, digest in sources.items()):
        raise ValueError("Source or production changed during the audit")
    report = {"completed_at_utc": datetime.now(timezone.utc).isoformat(), "rows": len(frame),
        "features": list(FEATURES), "baseline_features": 154, "candidate_features": 157,
        "source_sha256": sources, "input_sha256": {
            "data/audit/market_transfer_034/source_inputs.parquet": sha(BASE / "source_inputs.parquet"),
            "data/audit/market_transfer_034/baseline/manifest.json": sha(BASE / "baseline/manifest.json"),
            "data/audit/market_finish_surprise_035/projection.parquet": sha(OUTPUT / "projection.parquet")},
        "candidate_file_sha256": sha(candidate), "real_prefix_and_seeded_append_exact": True,
        "observation_rows": int(finite.sum()), "maximum_field_residual_sum_error": float(sums.abs().max()),
        "coverage": {c: int(added[c].notna().sum()) for c in FEATURES},
        "quantiles": added.quantile([0, .01, .5, .99, 1]).to_dict(), "target_cohorts": cohorts,
        "installed": after, "elapsed_seconds": time.monotonic() - started,
        "screen_rule": "One400-tree candidate against the exact cached corrected154 baseline. Lower overall and established-handicap race log loss, with neither runner Brier worse. No test-driven retuning.",
        "joint_rule": "If both034 and035 pass their individual point-score rules, one joint cheap screen is allowed. Only its fixed union can advance to one native check if it also passes; if the union fails, keep the pre-existing034 shortlist. No separate native runs for both families.",
        "method_reference": "https://projecteuclid.org/journals/annals-of-statistics/volume-32/issue-1/MM-algorithms-for-generalized-Bradley-Terry-models/10.1214/aos/1079120141.pdf",
        "scope": "Source/feature audit only. Development periods already inspected; no fresh confirmation or market-parity claim.",
        "limitations": ["Win-price probabilities used as pairwise strengths are an approximation, not a model-free expected finishing order.",
            "Removing nonfinishers defines a conditional performance descriptor, not an unbiased market-forecast residual.",
            "Three actual same-scale starts/365days; missing books and nonfinishes remain unknown and advance the window.",
            "Official ratings and cross-scale pound conversions do not enter the descriptor."]}
    (OUTPUT / "preparation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
