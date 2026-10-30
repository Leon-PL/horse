"""Resume exact native component fits after interruptions, without changing maths."""
from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
import tempfile

import joblib
import numpy as np
import pandas as pd

from src.model import RacePredictor

logger = logging.getLogger(__name__)


def _hash_value(digest, value):
    if isinstance(value, (np.ndarray, pd.DataFrame, pd.Series)):
        array = np.asarray(value)
        if array.dtype.hasobject:
            raise ValueError("Training checkpoint inputs must be numeric")
        digest.update(json.dumps([str(array.dtype), array.shape]).encode())
        digest.update(memoryview(np.ascontiguousarray(array)).cast("B"))
    elif isinstance(value, dict):
        digest.update(b"dict")
        for name in sorted(value):
            _hash_value(digest, name); _hash_value(digest, value[name])
    elif isinstance(value, (list, tuple)):
        digest.update(json.dumps([type(value).__name__, len(value)]).encode())
        for item in value:
            _hash_value(digest, item)
    else:
        if isinstance(value, np.generic):
            value = value.item()
        digest.update(json.dumps([type(value).__name__, value], sort_keys=True, allow_nan=False).encode())


def fit_key(manifest, role, columns, args, kwargs):
    digest = hashlib.sha256()
    _hash_value(digest, [manifest, role, columns, args, kwargs])
    return digest.hexdigest()


def cached_component(directory, key, function, args, kwargs):
    directory = Path(directory); directory.mkdir(parents=True, exist_ok=True)
    path = directory / (key+".joblib")
    if path.exists():
        value = joblib.load(path)
        if value["fit_key"] != key:
            raise ValueError("Training checkpoint identity differs")
        logger.info("Resuming completed native training component %s", key[:12])
        return value["model"]
    model = function(*args, **kwargs)
    with tempfile.NamedTemporaryFile(dir=directory, prefix=".fit-", suffix=".tmp", delete=False) as stream:
        temporary = Path(stream.name)
    try:
        joblib.dump({"fit_key": key, "model": model}, temporary, compress=3)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    logger.info("Checkpointed native training component %s", key[:12])
    return model


class CheckpointPredictor(RacePredictor):
    def __init__(self, *args, checkpoint_directory, checkpoint_manifest, **kwargs):
        super().__init__(*args, **kwargs)
        self._checkpoint_directory = Path(checkpoint_directory)
        self._checkpoint_manifest = checkpoint_manifest

    def _component(self, role, function, args, kwargs):
        key = fit_key(self._checkpoint_manifest, role, self.feature_cols, args, kwargs)
        logger.info("Native %s component: %d training rows; checkpoint %s", role, len(args[0]), key[:12])
        return cached_component(self._checkpoint_directory, key, function, args, kwargs)

    def _train_win_classifier(self, *args, **kwargs):
        return self._component("win", super()._train_win_classifier, args, kwargs)

    def _train_place_classifier(self, *args, **kwargs):
        return self._component("place", super()._train_place_classifier, args, kwargs)

    def fit_as_of(self, *args, **kwargs):
        result = super().fit_as_of(*args, **kwargs)
        self.eval_split_info["training_checkpoint_manifest"] = self._checkpoint_manifest
        return result
