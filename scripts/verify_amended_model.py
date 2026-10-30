"""Check corrected model inputs and serving math, without accuracy scoring."""
from datetime import datetime, timezone
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd

from scripts.verify_live_upgrade import check_outputs
from src.data_scraper import SportingLifeScraper
from src.deployment_validation import file_sha256 as sha
from src.featured_dataset import data_signature, fe_signature
from src.feature_store import FeatureStore
from src.forecast_evaluation import valid_race_mask
from src.live_prediction import features_for_live_cards
from src.model import RacePredictor
from src.run_store import RUNS_DIR, get_production_run_id

CASE = ROOT / "data/audit/result_amendments_042/consolidated"
INSTALL = CASE / "installation"
OLD_STORE = ROOT / "data/audit/live_feature_speed_20261029/installation/previous_store"


def align(frame):
    return frame.set_index(["race_id", "horse_name"]).sort_index()


def run(installed=False):
    fit = json.loads((CASE / "refit/fitted.json").read_text())
    plan = json.loads((CASE / "refit/final_plan.json").read_text())
    output = INSTALL / ("qa_after" if installed else "qa_before")
    output.mkdir(exist_ok=False)
    if installed and get_production_run_id() != fit["run_id"]:
        raise ValueError("The corrected current model is not installed")
    artifact = Path(RUNS_DIR) / fit["run_id"] / "model.joblib"
    if sha(artifact) != fit["model_sha256"]:
        raise ValueError("The fitted artifact changed")
    model = RacePredictor()
    model.load(str(artifact))
    store = FeatureStore() if installed else FeatureStore(str(INSTALL / "ready_store"))
    meta = store.meta()
    if meta["fe_signature"] != fe_signature() or meta["fe_signature"] != fit["fe_signature"]:
        raise ValueError("Model/store/code generations differ")
    processed = pd.read_parquet(store.processed_path,
        filters=[("race_date", "<=", pd.Timestamp(plan["config"]["fit_as_of"]))])
    if data_signature(processed) != fit["signatures"]["data_signature"]:
        raise ValueError("Store training-prefix inputs differ from the fitted model")
    del processed
    columns = list(dict.fromkeys([*model.feature_cols, "race_id", "horse_id", "horse_name", "race_date",
        "num_runners", "handicap", "race_type", "odds", "finish_position", "finish_pos_label", "won"]))
    recent = pd.read_parquet(store.featured_path, columns=columns,
        filters=[("race_date", ">=", pd.Timestamp(meta["last_date"]) - pd.Timedelta(days=2))])
    recent = recent.loc[valid_race_mask(recent)].copy()
    if recent.empty:
        raise ValueError("No complete recent fields for serving checks")
    forecasts = model.predict_races(recent)
    check_outputs(forecasts, model)
    maximum = 0.
    for race_id in recent.race_id.drop_duplicates().head(5):
        single = align(model.predict_race(recent.loc[recent.race_id.eq(race_id)]))
        batch = align(forecasts.loc[forecasts.race_id.eq(race_id)]).reindex(single.index)
        for col in ("win_probability", "place_probability"):
            maximum = max(maximum, float(np.max(np.abs(single[col] - batch[col]))))
            np.testing.assert_allclose(single[col], batch[col], atol=1e-12, rtol=0)
    poisoned = recent.copy()
    poisoned["won"] = 1 - poisoned.won
    poisoned["finish_position"] = 99
    poisoned["finish_pos_label"] = "DNF"
    poisoned["odds"] = 1000000.
    altered = align(model.predict_races(poisoned))
    ordinary = align(forecasts)
    for col in ("win_probability", "place_probability"):
        np.testing.assert_allclose(altered[col], ordinary[col], atol=1e-12, rtol=0)
    # Later source corrections must never alter earlier selected histories.
    repaired = pd.read_parquet(CASE / "proposed_repairs.parquet", columns=["race_date"])
    first = pd.to_datetime(repaired.race_date).min()
    prefix_columns = [*model.feature_cols, "race_id", "horse_name"]
    old_prefix = pd.read_parquet(OLD_STORE / "featured.parquet", columns=prefix_columns,
                                filters=[("race_date", "<", first)])
    new_prefix = pd.read_parquet(store.featured_path, columns=prefix_columns,
                                filters=[("race_date", "<", first)])
    pd.testing.assert_frame_equal(align(old_prefix), align(new_prefix), check_exact=True)
    if installed:
        card = pd.read_parquet(INSTALL / "qa_before/raw_card.parquet")
        day = str(card.race_date.iloc[0])[:10]
    else:
        today = pd.Timestamp.now(tz="Europe/London").date()
        day = max(pd.Timestamp(today), pd.Timestamp(meta["last_date"]) + pd.Timedelta(days=1)).date().isoformat()
        scraper = SportingLifeScraper()
        links = scraper.get_racecard_urls(day, uk_only=True)
        if not links:
            raise ValueError("No public upcoming card listing available")
        card = pd.DataFrame(scraper.scrape_racecard(links[0]["url"], day))
        if card.empty or card.horse_name.duplicated().any() or len(card) < 2:
            raise ValueError("The upcoming public card has no unambiguous multi-runner field")
    card.to_parquet(output / "raw_card.parquet", index=False)
    featured = features_for_live_cards(card, day, store=store, use_cache=False)
    live = model.predict_races(featured)
    check_outputs(live, model)
    single = align(model.predict_race(featured))
    batch = align(live)
    for col in ("win_probability", "place_probability"):
        np.testing.assert_allclose(single[col], batch[col], atol=1e-12, rtol=0)
    np.testing.assert_array_equal(np.sort(pd.to_numeric(card.horse_id)), np.sort(pd.to_numeric(featured.horse_id)))
    featured.to_parquet(output / "live_features.parquet", index=False)
    live.to_parquet(output / "live_predictions.parquet", index=False)
    if installed:
        pd.testing.assert_frame_equal(featured.reset_index(drop=True),
            pd.read_parquet(INSTALL / "qa_before/live_features.parquet"), check_exact=True)
        pd.testing.assert_frame_equal(live.reset_index(drop=True),
            pd.read_parquet(INSTALL / "qa_before/live_predictions.parquet"), check_exact=True)
    if sha(artifact) != fit["model_sha256"] or fe_signature() != fit["fe_signature"]:
        raise ValueError("Model or source changed during serving QA")
    report = {"passed": True, "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "installed": installed, "run_id": fit["run_id"], "model_sha256": fit["model_sha256"],
        "fe_signature": fit["fe_signature"], "features": len(model.feature_cols),
        "recent_fields": int(recent.race_id.nunique()), "recent_runners": len(recent),
        "single_batch_max_error": maximum, "current_label_price_poison_independent": True,
        "unchanged_causal_prefix_rows": len(old_prefix), "unchanged_causal_prefix_exact": True,
        "live_card_date": day, "live_race_id": str(card.race_id.iloc[0]), "live_runners": len(card),
        "installed_live_inputs_and_probabilities_exact": bool(installed),
        "input_sha256": {name: sha(output / name) for name in ("raw_card.parquet", "live_features.parquet", "live_predictions.parquet")},
        "script_sha256": sha(Path(__file__)), "scope": "Input/serving constraints and path consistency only; no prediction accuracy score or eligible forward observation"}
    with (output / "report.json").open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--installed", action="store_true")
    run(parser.parse_args().installed)
