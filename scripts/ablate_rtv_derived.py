"""Fixed LightGBM ablation: RTV-derived families added to the promoted recipe's inputs.

Runs ``scripts/quick_win_feature_screen.py`` (400 trees, two-year window, one
seed, simple race normalisation, cached baseline) over four development
quarters, for three fixed candidates chosen before any fit:

  all        Timeform card + sectional-time figures + within-race stride and jump metrics
  rtv_only   the same without the Timeform card
  timeform   the Timeform card alone

Baseline inputs are the promoted run's validation-reference features, read
from the current feature store. Candidate inputs come from
``scripts/rtv_derived_screen.py`` (run it first). Each quarter trains on the
two years ending eight days before it; the golden hold-out is never touched.

Usage::

    python scripts/ablate_rtv_derived.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import quick_win_feature_screen as screen  # noqa: E402
from benchmark_independent_features import score_pair  # noqa: E402
from rtv_derived_screen import OUT as RESEARCH, attach  # noqa: E402
from src.deployment_validation import file_sha256  # noqa: E402
from src.feature_store import FeatureStore  # noqa: E402
from src.featured_dataset import _model_load_columns  # noqa: E402
from src.wf_eval import development_mask  # noqa: E402

CASE = ROOT / "data/audit/rtv_derived_ablation"
PROMOTED = ROOT / "data/runs/20261030_135650_092286/meta.json"
FOLDS = {"2025Q3": ("2025-07-01", "2025-09-30"), "2025Q4": ("2025-10-01", "2025-12-31"),
         "2026Q1": ("2026-01-01", "2026-03-31"), "2026Q2": ("2026-04-01", "2026-06-03")}
TIMEFORM = ["tf_rating", "tf_stars", "tf_tipped", "tf_improver", "tf_doubtful", "tf_focus"]
RTV = ["time_fig_mean3", "time_fig_best3", "time_fig_last",
       *[f"{m}_rel_mean3" for m in ("acceleration", "avg_length", "avg_frequency", "max_frequency", "stride_length",
                                    "exit_speed", "time_in_envelope", "speed_lost", "recovery_time", "exit_entry")]]
CANDIDATES = {"all": TIMEFORM + RTV, "rtv_only": RTV, "timeform": TIMEFORM}


def build_reference(start: str, end: str) -> list[str]:
    reference = CASE / "reference"
    reference.mkdir(parents=True, exist_ok=True)
    baseline = json.loads(PROMOTED.read_text())["training_config"]["validation_reference"]["features"]
    if not (reference / "manifest.json").exists():
        store = FeatureStore()
        columns = list(dict.fromkeys([*(_model_load_columns(store) or []), *baseline, "horse_prev_races",
                                      "horse_id", "track", "off_time"]))
        frame = pd.read_parquet(store.featured_path, columns=columns,
                                filters=[("race_date", ">=", pd.Timestamp(start)), ("race_date", "<=", pd.Timestamp(end))])
        if not development_mask(frame.race_date).all():
            raise ValueError("Reference window reaches the golden hold-out")
        frame.to_parquet(reference / "inputs.parquet", index=False)
        runs = pd.read_parquet(RESEARCH / "rtv_runs.parquet")
        keys = frame[["race_id", "horse_id", "horse_name", "race_date", "track", "off_time"]].copy()
        projection = attach(keys, runs)
        added = list(dict.fromkeys(TIMEFORM + RTV))
        projection = projection[["race_id", "horse_name", "race_date", *added]]
        projection[added] = projection[added].apply(pd.to_numeric, errors="coerce").astype("float64")
        if len(projection) != len(frame) or projection.duplicated(["race_id", "horse_name", "race_date"]).any():
            raise ValueError("Projection must cover every input row exactly once")
        projection.to_parquet(reference / "projection.parquet", index=False)
        manifest = {"baseline_features": baseline, "inputs_sha256": file_sha256(reference / "inputs.parquet"),
                    "teacher_sha256": file_sha256(ROOT / "data/processed/betfair_bsp.parquet"),
                    "projection_sha256": file_sha256(reference / "projection.parquet"),
                    "store_meta": json.loads((Path(store.featured_path).parent / "meta.json").read_text()),
                    "promoted_run": PROMOTED.parent.name}
        (reference / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        coverage = projection[added].notna().mean().round(3).to_dict()
        (reference / "coverage.json").write_text(json.dumps(coverage, indent=2), encoding="utf-8")
    for name, added in CANDIDATES.items():
        (CASE / f"{name}.txt").write_text("\n".join(baseline + added) + "\n", encoding="utf-8")
    return baseline


def main() -> None:
    first = pd.Timestamp(min(s for s, _ in FOLDS.values())) - pd.DateOffset(years=2)
    build_reference(str(first.date()), max(e for _, e in FOLDS.values()))
    screen.REFERENCE = CASE / "reference"
    pooled = {name: [] for name in CANDIDATES}
    rows = []
    for fold, (test_start, test_end) in FOLDS.items():
        start = pd.Timestamp(test_start)
        screen.START = str((start - pd.DateOffset(years=2)).date())
        screen.CUTOFF = str((start - pd.Timedelta(days=8)).date())
        screen.TEST_START, screen.TEST_END = test_start, test_end
        for name in CANDIDATES:
            screen.OUTPUT = CASE / fold / name
            if not (screen.OUTPUT / "report.json").exists():
                screen.run(CASE / f"{name}.txt", CASE / "reference" / "projection.parquet", allow_new_baseline=True)
            report = json.loads((screen.OUTPUT / "report.json").read_text())
            delta = report["comparison"]["paired_delta_candidate_minus_baseline"]["date_cluster_race_log_loss"]
            rows.append({"fold": fold, "candidate": name, "races": report["comparison"]["races"],
                         "baseline_ll": report["comparison"]["baseline"]["race_log_loss"],
                         "delta_ll": delta["mean"], "ci_lo": delta["ci95"][0], "ci_hi": delta["ci95"][1]})
            pooled[name].append(pd.read_parquet(screen.OUTPUT / "predictions.parquet"))
    for name, parts in pooled.items():
        comparison = score_pair(pd.concat(parts, ignore_index=True))
        delta = comparison["paired_delta_candidate_minus_baseline"]["date_cluster_race_log_loss"]
        rows.append({"fold": "pooled", "candidate": name, "races": comparison["races"],
                     "baseline_ll": comparison["baseline"]["race_log_loss"],
                     "delta_ll": delta["mean"], "ci_lo": delta["ci95"][0], "ci_hi": delta["ci95"][1]})
    summary = pd.DataFrame(rows)
    summary.to_csv(CASE / "summary.csv", index=False)
    print("\nRace log loss, candidate minus baseline (negative = better), date-cluster 95% interval:")
    print(summary.to_string(index=False, float_format=lambda v: f"{v:+.5f}" if abs(v) < 1 else f"{v:.5f}"))


if __name__ == "__main__":
    main()
