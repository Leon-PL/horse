"""Date/row/result invariance of the existing history-support family for012."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd

from scripts.audit_atr_invariance import differences, fixture
from scripts.benchmark_independent_features import canonical_hash
from scripts.replay_market_snapshots import digest
from src import feature_engineer as fe
from src.ratings import compute_elo_features, compute_glicko_features

ADDED = ["horse_prev_races", "jockey_prev_rides", "trainer_prev_runs", "jt_prev_runs",
         "horse_win_rate_shrunk", "horse_place_rate_shrunk", "jockey_win_rate_shrunk",
         "trainer_win_rate_shrunk", "jt_win_rate_shrunk", "horse_ts_sigma",
         "has_horse_elo", "has_jockey_elo", "has_trainer_elo",
         "has_hidden_form", "hidden_form_runs", "experience_with_hidden"]
SOURCES = ["scripts/audit_history_invariance.py", "scripts/audit_atr_invariance.py",
           "src/feature_engineer.py", "src/ratings.py", "src/fe_incremental.py", "config.py",
           "src/live_prediction.py"]


def raw_fixture(days=24):
    frame = fixture(days)
    horse = frame.horse_name.str.extract(r"(\d+)")[0].astype(int)
    frame["jockey"] = "J" + (horse % 4).astype(str)
    frame["trainer"] = "T" + (horse % 3).astype(str)
    frame["form"] = np.where(horse % 3 == 0, "", "321")
    frame["days_since_last_run"] = np.where(horse % 3 == 0, 0, 10)
    frame["horse_runs"] = 9999  # Retroactive raw career totals must be ignored.
    frame["distance_furlongs"] = 8.
    frame["race_type"], frame["going"], frame["surface"] = "Flat", "Good", "Turf"
    return frame


def derive(raw):
    out = raw.sort_values(["race_date", "off_time", "race_id"], kind="stable").reset_index(drop=True)
    out = fe._add_result_history_flags(out)
    with fe.counts_state_scope(None):
        for stage in (fe.add_horse_features, fe.add_hidden_form_features, fe.add_jockey_features,
                      fe.add_trainer_features, fe.add_jockey_trainer_features,
                      compute_elo_features, compute_glicko_features):
            out = stage(out)
    return out.set_index(["race_id", "horse_id"])


def audit(registration):
    added = set(registration["candidate_features"]) - set(registration["baseline_features"])
    if added != set(ADDED):
        raise ValueError("Audit exactly the registered sixteen-column compact family")
    raw = raw_fixture()
    original = derive(raw)
    if set(ADDED) - set(original):
        raise ValueError("Fixture must exercise all registered history features")
    boundary = pd.Timestamp("2024-03-25")
    earlier = raw.loc[raw.race_date < boundary].copy()
    prefix = derive(earlier)
    changed = raw.copy()
    future = changed.race_date.ge(boundary)
    changed.loc[future, "finish_position"] = 5 - changed.loc[future, "finish_position"]
    changed.loc[future, "won"] = changed.loc[future, "finish_position"].eq(1).astype(int)
    changed.loc[future, "lengths_behind"] = changed.loc[future, "finish_position"] - 1.
    future_changed = derive(changed)
    extended = derive(raw_fixture(days=30))
    day = raw.race_date.unique()[15]
    cards = raw.copy()
    today = cards.race_date.eq(day)
    # Match _prepare_live_card: pending labels are numeric zero, never NaN won.
    cards.loc[today, ["finish_position", "won"]] = 0
    cards.loc[today, "lengths_behind"] = np.nan
    live = derive(cards)
    swapped = raw.copy()
    swapped.loc[today, "finish_position"] = 5 - swapped.loc[today, "finish_position"]
    swapped.loc[today, "won"] = swapped.loc[today, "finish_position"].eq(1).astype(int)
    keys = pd.MultiIndex.from_frame(raw.loc[today, ["race_id", "horse_id"]])
    poisoned = raw.copy()
    poisoned["horse_runs"] = np.arange(len(poisoned)) * 10000
    checks = {
        "prefix_vs_full_history": differences(prefix, original, ADDED),
        "changed_future_outcomes": differences(original.loc[prefix.index], future_changed, ADDED),
        "appended_future_races": differences(original, extended, ADDED),
        "current_day_outcomes_removed": differences(original.loc[keys], live, ADDED),
        "current_day_results_swapped": differences(original.loc[keys], derive(swapped), ADDED),
        "input_row_permutation": differences(original, derive(raw.sample(frac=1, random_state=13)), ADDED),
        "retroactive_career_totals_ignored": differences(original, derive(poisoned), ADDED),
    }
    sensitivity = differences(original, future_changed, ADDED)
    debut = original.loc[original.index.get_level_values("race_id").isin(["r0-0", "r0-1"])]
    semantics = {
        "first_recorded_start_has_zero_horse_history": bool(debut.horse_prev_races.eq(0).all()),
        "first_start_has_no_rating_availability": bool(debut[["has_horse_elo", "has_horse_glicko"]].eq(0).all().all()),
        "hidden_external_form_is_distinguished": bool(debut.has_hidden_form.eq(1).any() and debut.has_hidden_form.eq(0).any()),
        "pooled_rate_is_positive_for_cold_start": bool(debut.horse_win_rate_shrunk.gt(0).all()),
    }
    return {"passed": all(not v for v in checks.values()) and all(semantics.values()) and bool(sensitivity),
            "tested_columns": sorted(added), "checks": checks, "semantics": semantics,
            "prior_outcome_sensitivity_columns": sorted(sensitivity),
            "registration_content_sha256": canonical_hash(registration),
            "source_sha256": {p: digest(ROOT / p) for p in SOURCES},
            "scope": "Actual native feature/rating functions, synthetic histories, complete fields and same-day shared entities; no model-score selection"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registration", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(json.loads(args.registration.read_text()))
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["passed"] else 1)
