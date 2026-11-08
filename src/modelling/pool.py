"""A small conditional-logit pool with complete-field availability and fallback."""
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize


def _inputs(probabilities, race_ids, available=None):
    p = np.asarray(probabilities, dtype=float)
    race, labels = pd.factorize(race_ids, sort=False)
    if p.ndim != 2 or not p.shape[0] or not p.shape[1] or len(race) != len(p) or (race < 0).any():
        raise ValueError("Expected component probabilities for identified nonempty races")
    counts = np.bincount(race)
    if (counts < 2).any():
        raise ValueError("Expected complete fields of at least two runners")
    finite = np.isfinite(p)
    coverage = np.column_stack([np.bincount(race, weights=finite[:, j]) == counts for j in range(p.shape[1])])
    if available is not None:
        flags = np.asarray(available)
        if flags.dtype != np.bool_ or flags.shape != p.shape:
            raise ValueError("Availability must be a boolean array matching predictions")
        for j in range(p.shape[1]):
            n = np.bincount(race, weights=flags[:, j])
            if ((n != 0) & (n != counts)).any():
                raise ValueError("Component availability must describe whole races")
        declared = np.column_stack([np.bincount(race, weights=flags[:, j]) == counts for j in range(p.shape[1])])
        if (declared & ~coverage).any():
            raise ValueError("Available components must supply complete finite fields")
        coverage &= declared
    if not coverage[:, 0].all():
        raise ValueError("The generalist must be available for every race")
    for j in range(p.shape[1]):
        use = coverage[race, j]
        if ((p[use, j] < 0) | (p[use, j] > 1)).any():
            raise ValueError("Invalid component probabilities")
        sums = np.bincount(race, weights=np.where(use, p[:, j], 0), minlength=len(labels))
        if not np.allclose(sums[coverage[:, j]], 1., atol=1e-8, rtol=0):
            raise ValueError("Component probabilities must sum to one over complete fields")
    safe = np.where(coverage[race], p, p[:, :1])
    return safe, race, coverage


def _pool(p, race, coverage, weights):
    effective = coverage * weights
    empty = effective.sum(axis=1) == 0
    effective[empty, 0] = 1.
    effective /= effective.sum(axis=1, keepdims=True)
    # Exact identity, including exact zero probabilities, when only one expert
    # contributes. This also makes all-expert failure a true generalist fallback.
    one = (effective > 0).sum(axis=1) == 1
    scores = np.sum(effective[race] * np.log(np.clip(p, np.finfo(float).tiny, 1)), axis=1)
    top = np.full(len(coverage), -np.inf)
    np.maximum.at(top, race, scores)
    mass = np.exp(scores - top[race])
    out = mass / np.bincount(race, weights=mass)[race]
    rows = one[race]
    out[rows] = p[rows, np.argmax(effective[race[rows]], axis=1)]
    return out


@dataclass
class LogPool:
    components: tuple[str, ...]
    weights: tuple[float, ...] | None = None
    shrinkage: float = .01
    fit_info: dict = None

    def __post_init__(self):
        if not self.components or len(set(self.components)) != len(self.components):
            raise ValueError("Components must have unique ordered names, generalist first")
        if not np.isfinite(self.shrinkage) or self.shrinkage < 0:
            raise ValueError("Pool shrinkage must be finite and nonnegative")
        if self.weights is None:
            self.weights = (1.,) + (0.,) * (len(self.components) - 1)
        w = np.asarray(self.weights)
        if w.shape != (len(self.components),) or not np.isfinite(w).all() or (w < 0).any() or not np.isclose(w.sum(), 1):
            raise ValueError("Pool weights must be nonnegative and sum to one")

    def predict(self, probabilities, race_ids, available=None):
        p, race, coverage = _inputs(probabilities, race_ids, available)
        if p.shape[1] != len(self.components):
            raise ValueError("Component order/width differs from the saved pool")
        return _pool(p, race, coverage, np.asarray(self.weights))

    def fit(self, probabilities, won, race_ids, available=None):
        p, race, coverage = _inputs(probabilities, race_ids, available)
        if p.shape[1] != len(self.components):
            raise ValueError("Component order/width differs from the pool")
        y = np.asarray(won, dtype=float)
        if y.shape != (len(p),) or not np.isin(y, [0, 1]).all() or not np.all(np.bincount(race, weights=y) == 1):
            raise ValueError("Pool fitting needs exactly one winner per complete race")
        prior = np.r_[1., np.zeros(p.shape[1] - 1)]
        def loss(weights):
            pred = _pool(p, race, coverage, weights)
            return -float(np.sum(y * np.log(np.clip(pred, 1e-12, 1))) / len(coverage)) + self.shrinkage * np.square(weights - prior).sum()
        if len(prior) == 1:
            self.weights, self.fit_info = (1.,), {"identity": True, "races": len(coverage)}
            return self
        result = minimize(loss, prior, method="SLSQP", bounds=[(0., 1.)] * len(prior),
            constraints={"type": "eq", "fun": lambda w: w.sum() - 1}, options={"maxiter": 200, "ftol": 1e-10})
        w = np.clip(result.x, 0, 1)
        w /= max(w.sum(), 1e-12)
        accepted = bool(result.success and np.isfinite(result.fun) and loss(w) < loss(prior))
        self.weights = tuple(w if accepted else prior)
        self.fit_info = {"races": len(coverage), "optimised": accepted, "message": str(result.message),
                         "component_races": coverage.sum(axis=0).tolist()}
        return self
