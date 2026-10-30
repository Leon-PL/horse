"""Native win/place serving QA for the staged or installed 044 generation."""
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
CASE = ROOT/"data/audit/rtv_repair_features_044"
PROJECT = CASE/"project"
INSTALLED = "--installed" in sys.argv
NORMALISED = "--normalised-contract" in sys.argv
sys.path.insert(0, str(ROOT if INSTALLED else PROJECT))

import numpy as np
import pandas as pd
from src.deployment_validation import file_sha256 as sha
from src.featured_dataset import data_signature, fe_signature
from src.feature_store import FeatureStore
from src.forecast_evaluation import valid_race_mask
from src.live_prediction import features_for_live_cards
from src.model import RacePredictor
from src.run_store import get_production_run_id
from src.source_pipeline import _publish
from scripts.verify_live_upgrade import check_outputs


def main():
    fit = json.loads((CASE/"refit/fitted.json").read_text())
    plan = json.loads((CASE/"refit/plan.json").read_text())
    output = CASE/"installation"/("qa_contract_after" if NORMALISED else "qa_after" if INSTALLED else "qa_before")
    output.mkdir(parents=True, exist_ok=True)
    if (output/"report.json").exists():
        raise ValueError("Preserve the completed serving QA")
    if INSTALLED and get_production_run_id() != fit["run_id"]:
        raise ValueError("The corrected generation is not installed")
    base = ROOT if INSTALLED else PROJECT
    artifact = base/"data/runs"/fit["run_id"]/"model.joblib"
    if sha(artifact) != fit["model_sha256"]:
        raise ValueError("The corrected model changed")
    model = RacePredictor()
    model.load(str(artifact))
    store = FeatureStore()
    metadata = store.meta()
    if metadata["fe_signature"] != fe_signature() or fe_signature() != fit["fe_signature"]:
        raise ValueError("Serving code/store/model generations differ")
    # Training records the complete loaded source history separately from its
    # earlier fitting cutoff. Bind that exact source snapshot, then fingerprint
    # the fitted prefix explicitly rather than conflating the two scopes.
    source_cutoff = json.loads((CASE/"rebuilt.json").read_text())["meta"]["last_date"]
    processed = pd.read_parquet(store.processed_path, filters=[("race_date", "<=", pd.Timestamp(source_cutoff))])
    expected_signature = (json.loads((CASE/"source_contract/published.json").read_text())["migration"]["normalized_source_signature"]
                          if NORMALISED else fit["signatures"]["data_signature"])
    if data_signature(processed) != expected_signature:
        raise ValueError("The refit's loaded source snapshot and served history disagree")
    training_prefix_signature = data_signature(processed, cutoff=plan["config"]["fit_as_of"])
    del processed
    columns = list(dict.fromkeys([*model.feature_cols, "race_id", "horse_id", "horse_name", "race_date", "race_type",
        "num_runners", "handicap", "won", "finish_position", "odds", "track", "off_time"]))
    recent = pd.read_parquet(store.featured_path, columns=columns,
        filters=[("race_date", ">=", pd.Timestamp(metadata["last_date"])-pd.Timedelta(days=2))])
    recent = recent.loc[valid_race_mask(recent)].copy()
    predictions = model.predict_races(recent)
    check_outputs(predictions, model)
    maximum = 0.
    for identity in recent.race_id.drop_duplicates().head(5):
        single = model.predict_race(recent.loc[recent.race_id.eq(identity)]).set_index("horse_name")
        batch = predictions.loc[predictions.race_id.eq(identity)].set_index("horse_name").reindex(single.index)
        for name in ("win_probability", "place_probability"):
            difference = float(np.max(np.abs(single[name]-batch[name])))
            maximum = max(maximum, difference)
            np.testing.assert_allclose(single[name], batch[name], atol=1e-12, rtol=0)
    poisoned = model.predict_races(recent.assign(won=1-recent.won, finish_position=999, odds=np.nan))
    for name in ("win_probability", "place_probability"):
        np.testing.assert_array_equal(predictions[name], poisoned[name])
    if INSTALLED:
        previous = json.loads((CASE/"installation/qa_before/report.json").read_text())
        source = Path(previous["saved_card_source"])
        card = pd.read_parquet(CASE/"installation/qa_before/saved_card.parquet")
    else:
        public_cards = sorted((ROOT/"data/racecards_cache").glob("racecards_*.csv"))
        if not public_cards:
            raise ValueError("No saved public card is available for native replay")
        source = public_cards[-1]
        cards = pd.read_csv(source)
        size = cards.groupby("race_id").race_id.transform("size")
        suitable = cards.loc[size.between(5, 16) & pd.to_numeric(cards.num_runners).eq(size)]
        if suitable.empty:
            raise ValueError("No complete saved card")
        identity = suitable.race_id.iloc[0]
        card = suitable.loc[suitable.race_id.eq(identity)].copy()
        card.to_parquet(output/"saved_card.parquet", index=False)
    day = str(card.race_date.iloc[0])[:10]
    features = features_for_live_cards(card, day, store=store, use_cache=False)
    forecasts = model.predict_races(features)
    check_outputs(forecasts, model)
    key = ["race_id", "horse_name"]
    features[key+model.feature_cols].to_parquet(output/"live_inputs.parquet", index=False)
    forecasts.to_parquet(output/"live_forecasts.parquet", index=False)
    if INSTALLED:
        for name, current in (("live_inputs.parquet", features[key+model.feature_cols]), ("live_forecasts.parquet", forecasts)):
            prior = pd.read_parquet(CASE/"installation/qa_before"/name)
            left = prior.sort_values(key).reset_index(drop=True)
            right = current.sort_values(key).reset_index(drop=True)
            if NORMALISED and name == "live_forecasts.parquet":
                core = key+["win_probability", "place_probability"]
                left, right = left[core], right[core]
            pd.testing.assert_frame_equal(left, right[left.columns], check_dtype=False, check_exact=True)
    _publish(output/"report.json", {"completed_at": datetime.now(timezone.utc).isoformat(), "passed": True,
        "run_id": fit["run_id"], "fe_signature": fe_signature(), "model_sha256": fit["model_sha256"],
        "recent_fields": int(recent.race_id.nunique()), "recent_runners": len(recent),
        "single_batch_max_error": maximum, "current_label_price_independence_exact": True,
        "loaded_source_cutoff": source_cutoff, "loaded_source_signature_exact": True,
        "training_cutoff": plan["config"]["fit_as_of"], "training_prefix_signature": training_prefix_signature,
        "saved_card_source": str(source), "saved_card_sha256": sha(source), "saved_card_runners": len(card),
        "saved_card_is_operational_replay_not_prospective_forecast": True, "installed": INSTALLED,
        "normalised_operational_timestamp_contract": NORMALISED,
        "staged_installed_inputs_and_probabilities_exact": INSTALLED})
    print({"passed": True, "installed": INSTALLED, "fields": recent.race_id.nunique(), "saved_card_runners": len(card)}, flush=True)


if __name__ == "__main__":
    main()
