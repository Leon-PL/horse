"""Exercise installed forecasts on recent settled fields and one upcoming card.

No fitting, betting, formal forward-pair publication or performance selection.
Store a local operational receipt with the exact model and input fingerprints.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd

from src.ingestion.sporting_life import get_scraped_racecards
from src.deployment_validation import file_sha256
from src.feature_store import FeatureStore
from src.forecast_evaluation import valid_race_mask
from src.live_prediction import features_for_live_cards
from src.run_store import get_production_run_id, load_production_predictor, production_model_path

def check_outputs(result, predictor=None):
    for name in ("win_probability", "place_probability"):
        values = result[name].to_numpy()
        if not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
            raise ValueError(f"Invalid {name}")
    np.testing.assert_allclose(result.groupby("race_id").win_probability.sum(), 1., atol=1e-12, rtol=0)
    if (result.place_probability+1e-12 < result.win_probability).any():
        raise ValueError("A place forecast cannot be below its win probability")
    groups = result.groupby("race_id", sort=False)
    counts = groups.size()
    declared = groups.num_runners.first()
    if not counts.eq(declared).all() or not groups.num_runners.nunique().eq(1).all():
        raise ValueError("Probability check requires whole race fields")
    handicap = groups.handicap.first().astype(bool)
    slots = np.where(counts<=4, 3, np.where(counts<=7, 2, np.where(counts<=15, 3, np.where(handicap, 4, 3))))
    expected = np.minimum(slots, counts)
    if predictor is not None and getattr(predictor,"place_refinement",None) is not None:
        codes = groups.race_type.first().astype(str).to_numpy() if "race_type" in result else np.full(len(counts),"Unknown")
        expected = predictor.place_refinement.expected_slots(codes,expected)
    np.testing.assert_allclose(groups.place_probability.sum(), expected, atol=1e-9, rtol=0)


def run(output, date_str=None):
    output = Path(output).resolve()
    if (output / "report.json").exists():
        raise ValueError("Preserve the completed receipt; choose a new output directory")
    output.mkdir(parents=True, exist_ok=True)
    predictor = load_production_predictor()
    if predictor is None:
        raise ValueError("No installed model")
    run_id = get_production_run_id()
    model_hash = file_sha256(production_model_path())
    store = FeatureStore()
    meta = store.meta()
    columns = list(dict.fromkeys([*predictor.feature_cols, "race_id", "horse_id", "horse_name", "race_date",
        "odds", "num_runners", "handicap", "track", "off_time", "race_type"]))
    with store._access():
        recent = pd.read_parquet(store.featured_path, columns=columns+["won", "finish_position"],
            filters=[("race_date", ">=", pd.Timestamp(meta["last_date"])-pd.Timedelta(days=2))])
    eligible = valid_race_mask(recent)
    excluded_races = int(recent.loc[~eligible, "race_id"].nunique())
    recent = recent.loc[eligible].copy()
    if recent.empty:
        raise ValueError("No complete recent fields available for the operational check")
    batch = predictor.predict_races(recent)
    check_outputs(batch,predictor)
    maximum = 0.
    for race_id in recent.race_id.drop_duplicates().head(5):
        frame = recent.loc[recent.race_id.eq(race_id)]
        single = predictor.predict_race(frame).set_index("horse_name")
        grouped = batch.loc[batch.race_id.eq(race_id)].set_index("horse_name").reindex(single.index)
        for name in ("win_probability", "place_probability"):
            difference = float(np.max(np.abs(single[name]-grouped[name])))
            maximum = max(maximum, difference)
            np.testing.assert_allclose(single[name], grouped[name], atol=1e-12, rtol=0)
    now = pd.Timestamp.now(tz="Europe/London")
    day = pd.Timestamp(date_str).date().isoformat() if date_str else now.tz_convert("Europe/London").date().isoformat()
    cards = get_scraped_racecards(day, force_refresh=True)
    if cards is None or cards.empty:
        raise ValueError("No current public card available for the live-path smoke check")
    start = pd.to_datetime(cards.race_date.astype(str).str[:10]+" "+cards.off_time.astype(str),
                           format="%Y-%m-%d %H:%M", errors="coerce").dt.tz_localize("Europe/London", ambiguous="NaT", nonexistent="NaT")
    upcoming = cards.loc[start.gt(now+pd.Timedelta(minutes=30))].copy()
    if upcoming.empty:
        raise ValueError("No race remains safely before its start for this operational check")
    fields = upcoming.groupby("race_id", sort=False).size()
    upcoming = upcoming.loc[upcoming.race_id.isin(fields[fields.gt(1)].index)]
    if upcoming.empty:
        raise ValueError("No multi-runner upcoming card")
    identity = upcoming.sort_values("off_time", kind="stable").race_id.iloc[0]
    card = upcoming.loc[upcoming.race_id.eq(identity)].copy()
    if card.duplicated("horse_name").any():
        raise ValueError("Ambiguous live runner names")
    card.to_parquet(output / "raw_upcoming_card.parquet", index=False)
    featured = features_for_live_cards(card, day, store=store, use_cache=False)
    forecasts = predictor.predict_races(featured)
    check_outputs(forecasts,predictor)
    featured[columns].to_parquet(output / "upcoming_model_inputs.parquet", index=False)
    forecasts.to_csv(output / "upcoming_predictions.csv", index=False)
    source_ids = pd.to_numeric(card.horse_id, errors="raise")
    model_ids = pd.to_numeric(featured.horse_id, errors="raise")
    for ids in (source_ids, model_ids):
        if ids.isna().any() or ids.le(0).any() or ids.mod(1).ne(0).any() or ids.duplicated().any():
            raise ValueError("Ambiguous live horse identities")
    np.testing.assert_array_equal(np.sort(source_ids), np.sort(model_ids))
    if len(forecasts) != len(featured) or set(forecasts.horse_name) != set(featured.horse_name):
        raise ValueError("Forecasts changed the prepared runner population")
    source_names = dict(zip(source_ids, card.horse_name))
    name_changes = [{"horse_id": int(identity), "source_name": source_names[identity], "model_name": name}
                    for identity, name in zip(model_ids, featured.horse_name) if source_names[identity] != name]
    if get_production_run_id() != run_id or file_sha256(production_model_path()) != model_hash:
        raise ValueError("Installed model changed during the live check")
    report = {"passed": True, "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_id": run_id, "model_sha256": model_hash, "history_last_date": meta["last_date"],
        "training_as_of": predictor.eval_split_info.get("as_of"), "refit_rounds": predictor.eval_split_info.get("refit_rounds"),
        "features": len(predictor.feature_cols), "recent_races": int(recent.race_id.nunique()), "recent_runners": len(recent),
        "incomplete_recent_fields_excluded": excluded_races,
        "native_single_batch_max_error": maximum, "upcoming_race_id": str(identity),
        "upcoming_track": str(card.track.iloc[0]), "upcoming_off_time": str(card.off_time.iloc[0]),
        "upcoming_runners": len(card), "normalised_name_changes": name_changes,
        "input_sha256": {name: file_sha256(output / name)
            for name in ("raw_upcoming_card.parquet", "upcoming_model_inputs.parquet", "upcoming_predictions.csv")},
        "source_sha256": file_sha256(Path(__file__)),
        "scope": "Operational probability/input check; no test-score, betting or eligible forward-pair claim"}
    (output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path, help="New operational receipt directory")
    parser.add_argument("--date", help="Card date, e.g. tomorrow when today's racing has finished")
    args = parser.parse_args()
    run(args.output, args.date)
