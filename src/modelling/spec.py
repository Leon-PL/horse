"""Immutable, serialisable model recipes; defaults are resolved once."""
from dataclasses import dataclass, field, replace
import hashlib
import json
from pathlib import Path
import re

import config


def _defaults():
    from src.race_objectives import learner_params
    return tuple(sorted(learner_params().items()))


def _screen_defaults():
    from src.feature_selection import SCREEN_PARAMS
    return tuple(sorted(SCREEN_PARAMS.items()))


def read_features(path):
    text = Path(path).read_text(encoding="utf-8-sig")
    names = json.loads(text) if text.lstrip().startswith("[") else [
        line.strip() for line in text.splitlines() if line.strip() and not line.lstrip().startswith("#")]
    if not isinstance(names, list) or not names or any(not isinstance(n, str) or not n.strip() for n in names):
        raise ValueError("Feature selection must be a nonempty list of names")
    if len(names) != len(set(names)):
        raise ValueError("Feature selection contains duplicate names")
    return tuple(names)


@dataclass(frozen=True)
class ModelSpec:
    id: str
    features: tuple[str, ...]
    objective: str = "exploded"
    soft_label_blend: float = field(default_factory=lambda: config.WIN_SOFT_LABEL_BLEND if config.WIN_SOFT_LABEL_DISTILL else 0.)
    params: tuple = field(default_factory=_defaults)
    screen_params: tuple = field(default_factory=_screen_defaults)
    stage_weights: tuple[float, ...] = field(default_factory=lambda: tuple(config.WIN_EXPLODED_STAGE_WEIGHTS))
    seed: int = field(default_factory=lambda: config.RANDOM_SEED)
    purge_days: int = field(default_factory=lambda: config.PURGE_DAYS)
    early_stopping_rounds: int = field(default_factory=lambda: config.EARLY_STOPPING_ROUNDS)
    half_life_days: float = field(default_factory=lambda: config.RECENCY_HALF_LIFE_DAYS)
    seasonal_boost: float = field(default_factory=lambda: config.RECENCY_SEASONAL_BOOST)
    decay_shape: str = field(default_factory=lambda: config.RECENCY_DECAY_SHAPE)
    burn_in_months: int = field(default_factory=lambda: config.BURN_IN_MONTHS)
    calibration_fraction: float = .2

    def __post_init__(self):
        from src.model import EXCLUDE_COLUMNS
        if not re.fullmatch(r"[a-z][a-z0-9_-]{0,63}", self.id):
            raise ValueError("Model ID must be a simple lower-case name")
        if (not isinstance(self.features, tuple) or not self.features
                or any(not isinstance(n, str) or not n.strip() for n in self.features)
                or len(self.features) != len(set(self.features))):
            raise ValueError("Features must be a nonempty ordered tuple of unique names")
        if set(self.features) & set(EXCLUDE_COLUMNS):
            raise ValueError("Model features include excluded identifiers, prices or outcomes")
        if self.objective not in {"exploded", "cross_entropy"}:
            raise ValueError("Unsupported win objective")
        if not 0 <= self.soft_label_blend <= 1 or not 0 < self.calibration_fraction < .5:
            raise ValueError("Invalid target blend or calibration fraction")
        for params in (self.params, self.screen_params):
            if not isinstance(params, tuple) or len(dict(params)) != len(params):
                raise ValueError("Parameters must be an immutable tuple of unique pairs")
            if any(not isinstance(k, str) or not isinstance(v, (str, float, int, bool, type(None))) for k, v in params):
                raise ValueError("Parameters must contain scalar values")
        object.__setattr__(self, "params", tuple(sorted(self.params)))
        object.__setattr__(self, "screen_params", tuple(sorted(self.screen_params)))
        if not isinstance(self.stage_weights, tuple) or not self.stage_weights or any(w < 0 for w in self.stage_weights) or self.stage_weights[0] <= 0:
            raise ValueError("Stage weights must start with a positive winner weight")
        if self.purge_days < 1 or self.early_stopping_rounds < 0 or self.burn_in_months < 0:
            raise ValueError("Positive purge and nonnegative stopping/burn-in are required")
        if self.half_life_days <= 0 or self.seasonal_boost < 0 or self.decay_shape not in {"exp", "poly", "linear"}:
            raise ValueError("Invalid recency weighting")
        forbidden = {"objective", "random_state", "n_jobs", "metric", "subsample_freq", "verbose"} & (dict(self.params).keys() | dict(self.screen_params).keys())
        if forbidden:
            raise ValueError(f"Parameters duplicate recipe controls: {sorted(forbidden)}")
        json.dumps(self.to_dict(), allow_nan=False)

    def to_dict(self):
        from dataclasses import asdict
        result = asdict(self)
        result["params"] = dict(self.params)
        result["screen_params"] = dict(self.screen_params)
        result["features"] = list(self.features)
        result["stage_weights"] = list(self.stage_weights)
        return result

    @classmethod
    def from_dict(cls, value):
        value = dict(value)
        value["features"] = tuple(value["features"])
        if "params" in value:
            value["params"] = tuple(sorted(value["params"].items()))
        if "screen_params" in value:
            value["screen_params"] = tuple(sorted(value["screen_params"].items()))
        if "stage_weights" in value:
            value["stage_weights"] = tuple(value["stage_weights"])
        return cls(**value)

    @property
    def signature(self):
        return hashlib.sha256(json.dumps(self.to_dict(), sort_keys=True, allow_nan=False).encode()).hexdigest()

    def subset(self, features):
        return replace(self, features=tuple(features))
