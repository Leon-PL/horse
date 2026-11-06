"""Fixed LightGBM ablation of input-implementation repairs on the promoted 154 inputs.

Same screen as scripts/ablate_rtv_derived.py (400 trees, two-year window, one
seed, simple race normalisation) over four development quarters. Each
candidate replaces baseline inputs with repaired versions in place; nothing
is added. Candidates were fixed before any fit.
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
from src.deployment_validation import file_sha256  # noqa: E402
from src.feature_store import FeatureStore  # noqa: E402
from src.featured_dataset import _model_load_columns  # noqa: E402
from src.wf_eval import development_mask  # noqa: E402

CASE = ROOT / "data/audit/feature_impl_review_047"
PROMOTED = ROOT / "data/runs/20261030_135650_092286/meta.json"
FOLDS = {"2025Q3": ("2025-07-01", "2025-09-30"), "2025Q4": ("2025-10-01", "2025-12-31"),
         "2026Q1": ("2026-01-01", "2026-03-31"), "2026Q2": ("2026-04-01", "2026-06-03")}

VS_BEST_BASES = ["horse_dist_cat_wr_shrunk", "jockey_elo", "jockey_win_rate_shrunk", "jockey_place_rate_shrunk",
                 "jockey_track_win_rate", "jt_prev_runs", "trainer_freq", "trainer_elo_vs_field",
                 "trainer_place_rate_shrunk", "trainer_track_win_rate", "trainer_class_wr", "trainer_dist_cat_wr",
                 "trainer_going_wr", "field_avg_elo_10", "field_avg_elo_ewma", "horse_elo_momentum",
                 "horse_elo_momentum_vs_field", "horse_margin_elo_momentum", "combined_elo_vs_field",
                 "horse_avg_position", "days_since_last_win", "jockey_te_norm_pos", "jockey_te_ewma",
                 "trainer_te_norm_pos", "trainer_te_ewma", "jt_te_norm_pos", "jockey_te_norm_pos_365d",
                 "trainer_te_norm_pos_365d"]
NAN_FIX = ["field_avg_elo_10_vs_best_in_race", "field_avg_elo_ewma_vs_best_in_race", "speed_fig_best_5_vs_field",
           "speed_fig_adj_best_5_vs_field", "field_avg_elo_10_vs_field"]
COLD = ["days_since_last_race", "days_log", "days_since_last_race_z_in_race", "days_log_vs_field",
        "freshness_relative_score", "days_since_last_win", "days_since_last_win_vs_best_in_race"]
FORM = ["horse_avg_pos_3", "horse_best_pos_5", "form_slope", "elo_x_recent_form", "horse_avg_position_vs_best_in_race"]
MISC = ["prize_money_change_vs_avg", "draw_x_field_size"]

LOO = {f"{b}_vs_best_in_race": f"{b}_vs_best_in_race__loo" for b in VS_BEST_BASES}
CANDIDATES = {
    "relative_nan": {c: c + "__fix" for c in NAN_FIX},
    "relative_loo": {**{c: c + "__fix" for c in NAN_FIX if c.endswith("_vs_field")}, **LOO},
    "cold_start": {c: c + "__fix" for c in COLD},
    "form_npos": {c: c + "__fix" for c in FORM},
    "prize_draw": {c: c + "__fix" for c in MISC},
    "all": {**{c: c + "__fix" for c in NAN_FIX + COLD + FORM + MISC}, **LOO,
            "days_since_last_win_vs_best_in_race": "days_since_last_win_vs_best_in_race__fixloo",
            "horse_avg_position_vs_best_in_race": "horse_avg_position_vs_best_in_race__fixloo"},
}


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
        added = sorted({v for c in CANDIDATES.values() for v in c.values() if v is not None})
        projection = pd.read_parquet(CASE / "projection.parquet", columns=["race_id", "horse_name", "race_date", *added])
        keys = frame[["race_id", "horse_name", "race_date"]]
        projection = keys.merge(projection, on=["race_id", "horse_name", "race_date"], how="left",
                                validate="one_to_one", indicator=True)
        if not projection.pop("_merge").eq("both").all():
            raise ValueError("Projection must cover every input row exactly once")
        projection[added] = projection[added].astype("float64")
        projection.to_parquet(reference / "projection.parquet", index=False)
        manifest = {"baseline_features": baseline, "inputs_sha256": file_sha256(reference / "inputs.parquet"),
                    "teacher_sha256": file_sha256(ROOT / "data/processed/betfair_bsp.parquet"),
                    "projection_sha256": file_sha256(reference / "projection.parquet"),
                    "store_meta": json.loads((Path(store.featured_path).parent / "meta.json").read_text()),
                    "promoted_run": PROMOTED.parent.name, "candidates": CANDIDATES}
        (reference / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    for name, swaps in CANDIDATES.items():
        missing = set(swaps) - set(baseline)
        if missing:
            raise ValueError(f"{name}: not baseline inputs {sorted(missing)}")
        kept = [swaps.get(c, c) for c in baseline if swaps.get(c, c) is not None]  # None drops an input
        (CASE / f"{name}.txt").write_text("\n".join(kept) + "\n", encoding="utf-8")
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
            cmp = report["comparison"]
            delta = cmp["paired_delta_candidate_minus_baseline"]["date_cluster_race_log_loss"]
            rows.append({"fold": fold, "candidate": name, "races": cmp["races"],
                         "baseline_ll": cmp["baseline"]["race_log_loss"],
                         "delta_ll": delta["mean"], "ci_lo": delta["ci95"][0], "ci_hi": delta["ci95"][1]})
            pooled[name].append(pd.read_parquet(screen.OUTPUT / "predictions.parquet"))
    for name, parts in pooled.items():
        comparison = score_pair(pd.concat(parts, ignore_index=True))
        delta = comparison["paired_delta_candidate_minus_baseline"]["date_cluster_race_log_loss"]
        rows.append({"fold": "pooled", "candidate": name, "races": comparison["races"],
                     "baseline_ll": comparison["baseline"]["race_log_loss"],
                     "delta_ll": delta["mean"], "ci_lo": delta["ci95"][0], "ci_hi": delta["ci95"][1]})
        (CASE / f"pooled_{name}.json").write_text(json.dumps(comparison, indent=2), encoding="utf-8")
    summary = pd.DataFrame(rows)
    summary.to_csv(CASE / "summary.csv", index=False)
    print("\nRace log loss, candidate minus baseline (negative = better), date-cluster 95% interval:")
    print(summary.to_string(index=False, float_format=lambda v: f"{v:+.5f}" if abs(v) < 1 else f"{v:.5f}"))


if __name__ == "__main__":
    main()
