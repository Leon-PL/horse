"""Exercise the installed public serving path against preserved baseline outputs."""
from pathlib import Path
from datetime import datetime, timezone
import json
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from src.deployment_validation import file_sha256
from src.feature_store import FeatureStore
from src.featured_dataset import fe_signature
from src.forecast_evaluation import valid_race_mask
from src.forward_store import ForwardStudy
from src.live_prediction import features_for_live_cards
from src.run_store import get_production_run_id, load_production_predictor, production_model_path, feature_generation_status
from scripts.verify_live_upgrade import check_outputs

OUTPUT = ROOT / "data/audit/live_feature_speed_20261029/installation"
activation = json.loads((OUTPUT / "activation.json").read_text())
plan = json.loads((OUTPUT / "plan.json").read_text())
if get_production_run_id() != activation["run_id"]:
    raise ValueError("Qualified migrated model is not installed")
if file_sha256(production_model_path()) != plan["parent_model_sha256"]:
    raise ValueError("Installed model bytes changed")
if feature_generation_status()["state"] != "match" or fe_signature() != activation["fe_signature"]:
    raise ValueError("Installed generation does not match code/store/model")
ForwardStudy(activation["study"]).verify()
model = load_production_predictor()
store = FeatureStore()
cards = pd.read_parquet(ROOT / "data/audit/predictive_review_038/live/raw_upcoming_card.parquet")
started = time.perf_counter()
features = features_for_live_cards(cards, str(cards.race_date.iloc[0])[:10], store=store, use_cache=False)
feature_seconds = time.perf_counter() - started
baseline_dir = OUTPUT.parent / "native_card"
pd.testing.assert_frame_equal(features.reset_index(drop=True),
    pd.read_parquet(baseline_dir / "baseline_features.parquet"), check_exact=True)
predictions = model.predict_races(features)
before = pd.read_parquet(baseline_dir / "baseline_predictions.parquet")
pd.testing.assert_frame_equal(predictions.reset_index(drop=True), before, check_exact=True)
check_outputs(predictions, model)
columns = list(dict.fromkeys([*model.feature_cols, "race_id", "horse_name", "race_date", "num_runners",
                            "finish_position", "finish_pos_label", "won", "handicap", "race_type", "odds"]))
import pyarrow.parquet as pq
available = set(pq.read_schema(store.featured_path).names)
columns = [column for column in columns if column in available]
recent = pd.read_parquet(store.featured_path, columns=columns,
                        filters=[("race_date", ">=", pd.Timestamp(store.last_date()) - pd.Timedelta(days=2))])
recent = recent.loc[valid_race_mask(recent)].copy()
batch = model.predict_races(recent)
check_outputs(batch, model)
maximum = 0.0
for identity in recent.race_id.drop_duplicates().head(5):
    single = model.predict_race(recent.loc[recent.race_id.eq(identity)]).set_index("horse_name")
    grouped = batch.loc[batch.race_id.eq(identity)].set_index("horse_name").reindex(single.index)
    for name in ("win_probability", "place_probability"):
        maximum = max(maximum, float(np.max(np.abs(single[name] - grouped[name]))))
        np.testing.assert_allclose(single[name], grouped[name], atol=1e-12, rtol=0)
report = {"passed": True, "checked_at_utc": datetime.now(timezone.utc).isoformat(),
    "run_id": activation["run_id"], "model_sha256": file_sha256(production_model_path()),
    "fe_signature": fe_signature(), "feature_seconds": feature_seconds,
    "saved_card_runners": len(cards), "all_saved_features_exact": True, "saved_probabilities_exact": True,
    "recent_races": int(recent.race_id.nunique()), "recent_runners": len(recent),
    "single_batch_max_error": maximum,
    "scope": "Installed serving-path parity and probability constraints; saved inputs; no prospective accuracy or eligible publication claim"}
(OUTPUT / "live_parity.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
print(json.dumps(report, indent=2), flush=True)
