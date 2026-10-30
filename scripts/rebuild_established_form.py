"""Rebuild normal features, reusing only proved-unchanged rating components.

Elo and TrueSkill consume unchanged result/identity inputs and unchanged code.
Glicko is recomputed over the entire history. All other feature stages, and
their carried state, use the ordinary native memory-bounded builder.
"""
from pathlib import Path
import ast
from copy import deepcopy
import json
import os
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
import config

from scripts.build_feature_store_batched import _build
from src.data_processor import process_data
from src.deployment_validation import file_sha256 as sha
from src.feature_store import FeatureStore
from src.featured_dataset import _FE_SOURCE_FILES, fe_signature
from src.ratings import compute_glicko_features
from src.verified_race_metadata import apply_verified_runner_ratings

OUTPUT = ROOT / "data/audit/established_form_029"
KEY = ["race_id", "horse_name"]
GLICKO = {"horse_glicko", "horse_glicko_rd", "has_horse_glicko", "horse_glicko_lcb",
          "horse_glicko_vs_field", "horse_glicko_rank", "horse_glicko_rd_vs_field"}


class ReviewedRatingCache:
    def __init__(self, source):
        self.source = FeatureStore(str(source))

    def prepare(self, processed):
        if any(getattr(config, k, False) for k in ("GLICKO_JOCKEY_TRAINER", "GLICKO_DIMENSIONAL", "GLICKO_MARGIN", "GLICKO2_ENABLED")):
            raise ValueError("Review reuse again if an additional rating pass is enabled")
        old_source = ROOT / "data/audit/established_horse_review_028/ratings_before_fix.py"
        def definitions(path):
            return {getattr(n, "name", f"node{i}"): ast.dump(n, include_attributes=False)
                    for i, n in enumerate(ast.parse(path.read_text(encoding="utf-8")).body)}
        old, new = definitions(old_source), definitions(ROOT / "src/ratings.py")
        changed = {k for k in old.keys() | new.keys() if old.get(k) != new.get(k)}
        if changed != {"_glicko_pass", "_glicko2_pass"}:
            raise ValueError(f"Unreviewed changes to unchanged rating code: {changed}")
        previous = self.source.load_processed()
        expected = apply_verified_runner_ratings(previous.copy())
        # Every original processed column must match the normal corrected
        # processor, including results, margins, entities, types and dates.
        if len(expected) != len(processed):
            raise ValueError("The source population changed")
        a = expected.set_index(KEY).sort_index()
        b = processed.set_index(KEY).sort_index()
        for c in a:
            pd.testing.assert_series_equal(a[c], b[c], check_dtype=False, check_names=False)
        prefix = processed.loc[processed.race_date.le(processed.race_date.min()+pd.Timedelta(days=4))].copy()
        native, columns, _ = self.source._seeded_rating_frame(prefix, None)
        cached = self.source.load_featured(columns=KEY+columns)
        checked = native.merge(cached, on=KEY, validate="one_to_one", suffixes=("_new", "_old"))
        for c in set(columns)-GLICKO:
            np.testing.assert_allclose(checked[c+"_new"], checked[c+"_old"], rtol=0, atol=1e-12, equal_nan=True, err_msg=c)
        print(json.dumps({"stage": "fresh_full_glicko", "processed_rows": len(processed), "unchanged_rating_columns": len(set(columns)-GLICKO)}), flush=True)
        # TrueSkill is an independent pass over unchanged finishing positions.
        # Preserve its exact stored values/state instead of recomputing it.
        with patch.object(config, "TRUESKILL_ENABLED", False):
            fresh, advanced = compute_glicko_features(processed[KEY+["horse_id", "race_date", "off_time", "finish_position"]].copy(), return_state=True)
        if set(advanced) != {"horse"}:
            raise ValueError("Unexpected Glicko state component")
        replaced = cached.drop(columns=list(GLICKO)).merge(fresh[KEY+sorted(GLICKO)], on=KEY, how="left", validate="one_to_one")
        self.ratings = replaced.set_index(KEY, verify_integrity=True)
        self.final_state = deepcopy(self.source.load_state())
        self.final_state["glicko"]["horse"] = advanced["horse"]
        self.columns = columns
        pd.testing.assert_index_equal(self.ratings.index.sort_values(), pd.MultiIndex.from_frame(processed[KEY]).sort_values(), exact=False)
        return {"validated_processed_rows": len(processed), "prefix_rows_checked": len(prefix), "rating_columns": columns,
                "recomputed": sorted(GLICKO), "unchanged_code_definitions_verified": True,
                "new_rating_code_sha256": sha(ROOT / "src/ratings.py")}

    def for_chunk(self, chunk):
        keys = pd.MultiIndex.from_frame(chunk[KEY])
        if not keys.isin(self.ratings.index).all():
            raise ValueError("A rating input is missing")
        return self.ratings.reindex(keys).reset_index()[KEY+self.columns]


def run():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    store = FeatureStore()
    previous = OUTPUT / "previous_generation/feature_store"
    previous.mkdir(parents=True, exist_ok=True)
    # Keep a recoverable generation without copying the multi-GB feature file.
    # Atomic store replacement later detaches the live path from these links.
    with store._access():
        for name in store._FILES:
            source, target = Path(store.root)/name, previous/name
            if not target.exists():
                os.link(source, target)
            elif sha(source) != sha(target):
                raise ValueError("Do not replace the frozen previous generation")
    processed = process_data(save=False)
    signature = fe_signature()
    hashes = {name: sha(ROOT / name) for name in _FE_SOURCE_FILES}
    manifest = {"stage": "rebuilding", "fe_signature": signature, "source_sha256": hashes,
        "processed_rows": len(processed), "previous_meta": FeatureStore(str(previous)).meta(),
        "scope": "Ordinary native feature stages and state; only unchanged Elo/TrueSkill sweeps reused after code/input checks"}
    (OUTPUT / "rebuild_status.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    with store._staged_update() as pending:
        metadata = _build(pending, 60_000, processed=processed, ratings_cache=ReviewedRatingCache(previous))
        if fe_signature() != signature or any(sha(ROOT/name) != digest for name, digest in hashes.items()):
            raise ValueError("Source changed during the normal rebuild")
    manifest.update(stage="rebuilt", store_meta=metadata)
    (OUTPUT / "rebuild_status.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    run()
