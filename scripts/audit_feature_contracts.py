"""Outcome-free semantic checks of selected features and overseas-history gaps."""
from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from src.audited_class_features import uk_class_number
from src.data_processor import _GOING_MAP
from src.track_config import TRACK_CONFIG

OUT = ROOT / "data/audit/feature_review_018"


def audit():
    OUT.mkdir(parents=True, exist_ok=True)
    path = ROOT / "data/processed/feature_store/featured.parquet"
    wanted = ["race_id", "race_date", "horse_id", "horse_name", "race_type", "track",
              "race_class", "official_rating", "has_official_rating", "class_change",
              "or_change", "or_change_vs_field", "distance_change", "dist_x_class_change",
              "horse_prev_races", "has_hidden_form", "hidden_form_runs", "prior_form_runs",
              "days_since_last_run", "days_since_last_race", "form", "age", "weight_lbs",
              "surface", "num_runners", "base_place_rate", "trainer", "jockey", "sire",
              "horse_glicko_rd", "horse_elo", "years_from_peak", "draw_x_field_size"]
    schema = pq.read_schema(path).names
    frame = pd.read_parquet(path, columns=[c for c in wanted if c in schema])
    frame = frame.sort_values(["horse_id", "race_date", "race_id"], kind="stable")
    group = frame.groupby("horse_id", sort=False)
    previous_real = group.has_official_rating.shift()
    previous_type = group.race_type.shift()
    previous_class = group.race_class.shift()
    previous_date = group.race_date.shift()
    class_num = uk_class_number(frame.race_class)
    prev_class_num = uk_class_number(previous_class)
    observed_transition = previous_date.notna()
    invalid_class = observed_transition & (class_num.isna() | prev_class_num.isna())
    fabricated_mark_change = observed_transition & (frame.has_official_rating.eq(0) | previous_real.eq(0))
    code_switch = observed_transition & frame.race_type.ne(previous_type)
    first_local = frame.horse_prev_races.eq(0)
    source_experience = frame.days_since_last_run.gt(0) | frame.prior_form_runs.ge(2)
    # Restrict sampling to pre-race characteristics. Never select on result/odds.
    cheltenham = frame.track.str.casefold().eq("cheltenham") & first_local & source_experience
    jumps = frame.race_type.isin(["Hurdle", "Chase", "Nh Flat", "NH Flat"])
    frame.loc[cheltenham, wanted_columns(frame, ["horse_id", "horse_name", "race_id", "race_date",
        "race_type", "age", "form", "prior_form_runs", "horse_prev_races", "days_since_last_run",
        "days_since_last_race", "trainer", "official_rating", "has_official_rating", "horse_glicko_rd"])
    ].sort_values(["race_date", "horse_id"]).to_csv(OUT / "cheltenham_missing_local_history.csv", index=False)
    examples = frame.loc[fabricated_mark_change & frame.or_change.ne(0), wanted_columns(frame,
        ["horse_id", "horse_name", "race_date", "race_type", "official_rating", "has_official_rating", "or_change"])]
    examples.head(100).to_csv(OUT / "unsupported_mark_change_examples.csv", index=False)
    raw = pd.read_csv(ROOT / "data/raw/race_results.csv", usecols=["going", "horse_id", "race_date"])
    raw_going = raw.going.astype("string").str.strip().str.lower()
    unmatched = raw_going.loc[~raw_going.isin(_GOING_MAP)].value_counts(dropna=False)
    unmatched.rename_axis("source_going").reset_index(name="rows").to_csv(OUT / "unmapped_going.csv", index=False)
    tracks = frame.track.astype("string").str.casefold()
    known_tracks = {t.casefold() for t in TRACK_CONFIG}
    frame.loc[~tracks.isin(known_tracks), "track"].value_counts().to_csv(OUT / "unknown_track_config.csv")
    selected = [c.strip() for c in (ROOT / "feature_sets/win.txt").read_text().splitlines()
                if c.strip() and not c.startswith("#")]
    report = {
        "scope": "Source and predictor semantics only; no model fitting or outcome-based selection",
        "rows": len(frame), "races": int(frame.race_id.nunique()), "selected_features": len(selected),
        "class": {"missing_current": int(class_num.isna().sum()),
            "observed_transitions_with_unknown_class": int(invalid_class.sum()),
            "unknown_transitions_with_nonzero_class_change": int((invalid_class & frame.class_change.ne(0)).sum()),
            "racing_code_switches": int(code_switch.sum()),
            "code_switches_with_nonzero_mark_change": int((code_switch & frame.or_change.ne(0)).sum())},
        "official_rating": {"transitions_with_imputed_current_or_previous": int(fabricated_mark_change.sum()),
            "imputed_transitions_with_nonzero_change": int((fabricated_mark_change & frame.or_change.ne(0)).sum())},
        "experience": {"first_local_rows": int(first_local.sum()),
            "first_local_with_source_experience": int((first_local & source_experience).sum()),
            "jumps_first_local_with_source_experience": int((jumps & first_local & source_experience).sum()),
            "cheltenham_first_local_with_source_experience": int(cheltenham.sum()),
            "hidden_history_flag_selected": "has_hidden_form" in selected,
            "external_history_inputs_selected": [c for c in selected if c.startswith("external_")],
            "source_gap_positive_but_local_recency_default60": int((first_local & frame.days_since_last_run.gt(0) & frame.days_since_last_race.eq(60)).sum())},
        "place_baseline": {"four_runner_rows_with_prior_one": int((frame.num_runners.eq(4) & frame.base_place_rate.eq(1)).sum())},
        "going": {"unmapped_rows": int((~raw_going.isin(_GOING_MAP)).sum()), "unmapped_values": len(unmatched)},
        "identity": {"missing_horse_ids": int(frame.horse_id.isna().sum()),
            "nonpositive_horse_ids": int(pd.to_numeric(frame.horse_id, errors="coerce").le(0).sum())},
        "weights": {"outside_broad_50_220lb": int((frame.weight_lbs.lt(50) | frame.weight_lbs.gt(220)).sum())},
        "track": {"rows_using_default_config": int((~tracks.isin(known_tracks)).sum())},
    }
    (OUT / "contract_audit.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


def wanted_columns(frame, columns):
    return [c for c in columns if c in frame]


if __name__ == "__main__":
    audit()
