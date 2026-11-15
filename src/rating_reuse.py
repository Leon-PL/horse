"""Verified reuse of rating sweeps when only non-rating history inputs change.

The result-label repair changes counts and form, but ratings.py consumes the
same finishing positions and cumulative margins. Reuse is permitted only
with unchanged rating code/config, identical original processed values, a
fingerprinted source snapshot, and a raw-rating versus stored-prefix check.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
from pathlib import Path
import re
import sys

import numpy as np
import pandas as pd
from src.runner_identity import identity_columns, runner_positions
import config
from src.feature_store import FeatureStore
from src.featured_dataset import fe_signature, _portable_config_value

COMPONENT_FILES = ("src/ratings.py", "src/result_margins.py", "src/utils.py", "src/runner_identity.py",
                   "src/foreign_form.py", "src/feature_semantics.py")


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def component_signature():
    h = hashlib.sha256()
    names = set()
    for name in COMPONENT_FILES:
        source = (Path(config.BASE_DIR) / name).read_text(encoding="utf-8")
        h.update(name.encode()); h.update(source.replace("\r\n", "\n").encode())
        names.update(re.findall(r"config\.([A-Z][A-Z0-9_]*)", source))
        names.update(re.findall(r"getattr\(config,\s*['\"]([A-Z][A-Z0-9_]*)['\"]", source))
    values = {n: _portable_config_value(getattr(config, n, None)) for n in sorted(names)}
    versions = {n: importlib.metadata.version(n) for n in ("numpy", "pandas", "numba", "scipy")}
    h.update(json.dumps({"settings": values, "versions": versions, "python": list(sys.version_info[:2])}, sort_keys=True, default=str).encode())
    return h.hexdigest()


def freeze_rating_source(store, receipt_path):
    """Call before changing the feature generation; source store stays intact."""
    if store.meta().get("fe_signature") != fe_signature():
        raise ValueError("Freeze rating reuse while the source generation is current")
    files = [Path(store.featured_path), Path(store.processed_path), Path(store.state_path)]
    receipt = {"source_root": str(Path(store.root).resolve()), "source_fe_signature": fe_signature(),
               "rating_component_signature": component_signature(),
               "source_files_sha256": {p.name: digest(p) for p in files}}
    path = Path(receipt_path)
    if path.exists():
        raise ValueError("Do not overwrite a frozen rating source receipt")
    path.write_text(json.dumps(receipt, indent=2))
    return receipt


class ValidatedRatingCache:
    def __init__(self, source_root, receipt_path):
        self.store = FeatureStore(str(source_root))
        self.receipt = json.loads(Path(receipt_path).read_text())
        if self.receipt["source_root"] != str(Path(source_root).resolve()):
            raise ValueError("Rating receipt belongs to another store")
        if self.receipt["rating_component_signature"] != component_signature():
            raise ValueError("Rating source code, settings or library versions changed")
        for name, expected in self.receipt["source_files_sha256"].items():
            if digest(Path(source_root) / name) != expected:
                raise ValueError(f"Cached rating source changed: {name}")
        self.ratings = None
        self.final_state = None

    def prepare(self, processed):
        # French/UAE races are swept with the domestic ones, so the cached
        # ratings are reusable only over the same foreign history.
        from src.foreign_form import signature as foreign_form_signature
        meta = self.store.meta()
        if meta.get("foreign_form_signature") != foreign_form_signature(pd.Timestamp(meta["last_date"])):
            raise ValueError("Rating input changed: foreign races")
        previous = self.store.load_processed()
        if len(previous) != len(processed) or set(previous) - set(processed):
            raise ValueError("Rating reuse requires the same processed records")
        # All original values, not just guessed rating dependencies, must be
        # identical. A newly added result-label column is allowed.
        for col in previous:
            a, b = previous[col], processed[col]
            if pd.api.types.is_numeric_dtype(a) and pd.api.types.is_numeric_dtype(b):
                equal = np.array_equal(a.to_numpy(float), b.to_numpy(float), equal_nan=True)
            elif pd.api.types.is_datetime64_any_dtype(a):
                equal = np.array_equal(pd.to_datetime(a).to_numpy(), pd.to_datetime(b).to_numpy(), equal_nan=True)
            else:
                equal = a.astype("string").reset_index(drop=True).equals(b.astype("string").reset_index(drop=True))
            if not equal:
                raise ValueError(f"Rating input changed: {col}")
        prefix = processed.loc[processed.race_date.le(processed.race_date.min() + pd.Timedelta(days=10))]
        fresh, columns, _ = self.store._seeded_rating_frame(prefix, None)
        cached = self.store.load_featured(columns=[*identity_columns(previous), *columns])
        positions = runner_positions(fresh, cached)
        for col in columns:
            if not np.allclose(fresh[col], cached.iloc[positions][col], atol=1e-12, rtol=0, equal_nan=True):
                raise ValueError(f"Stored feature is not a reusable raw rating: {col}")
        self.ratings = cached.reset_index(drop=True)
        self.final_state = self.store.load_state()
        self.columns = columns
        return {"source_root": str(Path(self.store.root).resolve()),
                "component_signature": self.receipt["rating_component_signature"],
                "validated_processed_rows": len(processed), "prefix_rows_checked": len(fresh), "rating_columns": columns}

    def for_chunk(self, chunk):
        if self.ratings is None:
            raise ValueError("Validate the complete processed snapshot before using cached ratings")
        positions = runner_positions(chunk, self.ratings)
        return self.ratings.iloc[positions].reset_index(drop=True)
