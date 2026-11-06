import json

import numpy as np
import pytest

import config
from src.data_processor import process_data
from src.feature_store import FeatureStore
from src.featured_dataset import fe_signature
from src.rating_reuse import ValidatedRatingCache, freeze_rating_source
from tests.test_live_parity import _synthetic_raw


@pytest.fixture
def legacy_store(tmp_path):
    raw = _synthetic_raw(n_days=5, field=4)
    raw.loc[raw.finish_position.eq(4), ["finish_position", "lengths_behind"]] = [0, np.nan]
    processed = process_data(raw).drop(columns="finish_pos_label", errors="ignore")
    store = FeatureStore(str(tmp_path / "source"))
    store.build(processed)
    # The memory-bounded builder persists the pure processor output. The
    # monolithic store path also keeps in-place FE annotations; simulate the
    # actual batched source format being reused here.
    processed.to_parquet(store.processed_path, index=False)
    meta = store.meta(); meta["fe_signature"] = fe_signature()
    (tmp_path / "source/meta.json").write_text(json.dumps(meta))
    receipt = tmp_path / "ratings.json"
    freeze_rating_source(store, receipt)
    return store, processed, receipt


def test_added_result_labels_reuse_identical_raw_ratings(legacy_store):
    store, processed, receipt = legacy_store
    updated = processed.assign(finish_pos_label=np.where(processed.finish_position.eq(0), "DNF", ""))
    cache = ValidatedRatingCache(store.root, receipt)
    audit = cache.prepare(updated)
    fresh, cols, state = store._seeded_rating_frame(updated, None)
    reused = cache.for_chunk(updated)
    np.testing.assert_allclose(fresh[cols], reused[cols], atol=1e-12, rtol=0, equal_nan=True)
    assert audit["validated_processed_rows"] == len(processed)
    assert cache.final_state.keys() == state.keys()


def test_changed_original_input_is_rejected(legacy_store):
    store, processed, receipt = legacy_store
    changed = processed.copy(); changed.loc[0, "distance_furlongs"] += 1
    with pytest.raises(ValueError, match="input changed"):
        ValidatedRatingCache(store.root, receipt).prepare(changed)


def test_changed_cache_or_rating_configuration_is_rejected(legacy_store, monkeypatch):
    store, processed, receipt = legacy_store
    monkeypatch.setattr(config, "GLICKO_C", config.GLICKO_C + 1)
    with pytest.raises(ValueError, match="settings"):
        ValidatedRatingCache(store.root, receipt)
