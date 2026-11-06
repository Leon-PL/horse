"""Race-level temperature calibration shared by training and benchmarks.

Inputs contain contiguous runners for each race; ``groups`` contains their
field sizes. Temperature scales log probabilities, retaining within-race
rankings while producing a single probability distribution over each field.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.optimize import minimize
from scipy.special import expit, logit
from dataclasses import dataclass


#: Floor before taking logs. The exploded win model (057) passes
#: ``exp(score - batch max)``, which can fall far below any probability floor
#: while still ranking runners within a weak race; only exact zeros are floored.
_LOG_FLOOR = np.finfo(np.float64).tiny


def _race_inputs(probabilities, groups):
    p = np.asarray(probabilities, dtype=np.float64)
    sizes = np.asarray(groups)
    if p.ndim != 1 or sizes.ndim != 1:
        raise ValueError("Probabilities and race sizes must be one-dimensional.")
    if not np.all(np.isfinite(p)) or np.any((p < 0) | (p > 1)):
        raise ValueError("Probabilities must be finite and between zero and one.")
    if (not np.all(np.isfinite(sizes)) or np.any(sizes <= 0)
            or np.any(sizes != np.floor(sizes)) or sizes.sum() != len(p)):
        raise ValueError("Positive integer race sizes must sum to the number of runners.")
    sizes = sizes.astype(np.intp)
    starts = np.r_[0, np.cumsum(sizes)[:-1]] if len(sizes) else np.array([], dtype=np.intp)
    return p, sizes, starts


def _temperature_probs(log_p, groups, starts, temperature):
    # Centre before dividing so even an extremely small positive temperature
    # cannot produce inf - inf. Negative overflow just gives zero probability.
    centred = log_p - np.repeat(np.maximum.reduceat(log_p, starts), groups)
    with np.errstate(over="ignore", under="ignore"):
        weights = np.exp(centred / temperature)
    return weights / np.repeat(np.add.reduceat(weights, starts), groups)


def race_temperature(probabilities, groups, temperature):
    """Apply one positive temperature; each race sums to one, even for zeros."""
    temperature = float(temperature)
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("Calibration temperature must be finite and positive.")
    p, groups, starts = _race_inputs(probabilities, groups)
    if not len(p):
        return p.copy()
    return _temperature_probs(np.log(np.clip(p, _LOG_FLOOR, 1.0)), groups, starts, temperature)


def fit_temperature(probabilities, groups, winners):
    """Minimise mean winner log loss on held-out, single-winner races.

    No-winner races, dead heats and walkovers do not define the target used by
    this calibrator and are omitted as whole races. No usable races returns
    the identity temperature. Callers must keep these data out of model fit,
    early stopping and final evaluation.
    """
    p, groups, starts = _race_inputs(probabilities, groups)
    y = np.asarray(winners, dtype=np.float64)
    if y.shape != p.shape or not np.all(np.isin(y, [0.0, 1.0])):
        raise ValueError("Winners must contain one binary label per runner.")
    if not len(groups):
        return 1.0
    valid = (np.add.reduceat(y, starts) == 1) & (groups > 1)
    if not np.any(valid):
        return 1.0
    rows = np.repeat(valid, groups)
    p, y, groups = p[rows], y[rows], groups[valid]
    starts = np.r_[0, np.cumsum(groups)[:-1]]
    log_p = np.log(np.clip(p, _LOG_FLOOR, 1.0))

    def loss(log_temperature):
        calibrated = _temperature_probs(log_p, groups, starts, np.exp(log_temperature))
        return -float(np.sum(y * np.log(np.clip(calibrated, 1e-12, 1.0))) / len(groups))

    result = minimize_scalar(loss, bounds=(np.log(0.25), np.log(4.0)), method="bounded")
    if not result.success or not np.isfinite(result.fun) or result.fun >= loss(0.0):
        return 1.0
    return float(np.exp(result.x))


@dataclass
class PlaceProbabilityRefinement:
    """038's qualified smooth place map and learned field-size adjustment.

    Parameters and missing-slot expectations are fitted on earlier forecasts,
    never estimated from the current field's eventual finishing count.
    """
    slope: float = 1.
    intercept: float = 0.
    strength: float = 2.
    deficits: dict | None = None

    def expected_slots(self, codes, places):
        return np.asarray([max(1., k-(self.deficits or {}).get((str(c),int(k)),0.))
                           for c,k in zip(codes,places)])

    def predict(self, raw, win, groups, places, codes):
        raw,groups,starts = _race_inputs(raw,groups)
        win = np.asarray(win,float)
        places = np.asarray(places,float)
        if win.shape != raw.shape or places.shape != groups.shape or len(codes) != len(groups):
            raise ValueError("Refined place arrays do not align")
        target = self.expected_slots(codes,places)
        if (not np.isfinite(win).all() or ((win<0)|(win>1)).any() or ((places<1)|(places>groups)).any()
                or not 0 <= self.strength <= 4 or (np.add.reduceat(win,starts)>target+1e-10).any()):
            raise ValueError("Invalid refined place probabilities/slot expectations")
        base = expit(self.slope*logit(np.clip(raw,1e-9,1-1e-9))+self.intercept)
        logits = logit(np.clip(base,1e-6,1-1e-6))
        means = np.add.reduceat(logits,starts)/groups
        shrink = np.clip(1-self.strength*np.maximum(0,places/groups-.25),.25,1.)
        logits = np.repeat(means,groups)+np.repeat(shrink,groups)*(logits-np.repeat(means,groups))
        lo,hi = np.full(len(groups),-100.),np.full(len(groups),100.)
        for _ in range(60):
            mid = (lo+hi)/2
            p = np.maximum(expit(logits+np.repeat(mid,groups)),win)
            above = np.add.reduceat(p,starts)>target
            hi,lo = np.where(above,mid,hi),np.where(above,lo,mid)
        return np.maximum(expit(logits+np.repeat((lo+hi)/2,groups)),win)

    @classmethod
    def fit(cls,raw,win,placed,finish_positions,groups,places,codes):
        raw,groups,starts = _race_inputs(raw,groups)
        placed,fp,places,codes = np.asarray(placed,float),np.asarray(finish_positions,float),np.asarray(places,int),np.asarray(codes,str)
        if placed.shape != raw.shape or not np.isin(placed,[0,1]).all() or fp.shape != raw.shape:
            raise ValueError("Invalid observed place labels")
        completed = np.add.reduceat((fp>0).astype(int),starts)
        missing = np.maximum(places-completed,0)
        estimates = {}
        for code,k in sorted(set(zip(codes,places))):
            mask = (codes==code)&(places==k)
            estimates[(str(code),int(k))] = float(missing[mask].sum()/(mask.sum()+100.))
        def loss(ab):
            p = expit(ab[0]*logit(np.clip(raw,1e-9,1-1e-9))+ab[1])
            return float(np.mean((p-placed)**2)+1e-4*((ab[0]-1)**2+ab[1]**2))
        result = minimize(loss,[1.,0.],method="L-BFGS-B",bounds=[(.1,3.),(-3.,3.)])
        if not result.success or not np.isfinite(result.fun):
            raise ValueError("Refined place calibration did not converge")
        mapping = cls(*map(float,result.x),deficits=estimates)
        def projected_brier(strength):
            mapping.strength = float(strength)
            return float(np.mean((mapping.predict(raw,win,groups,places,codes)-placed)**2))
        fitted = minimize_scalar(projected_brier,bounds=(0.,4.),method="bounded",options={"xatol":1e-3})
        if not fitted.success or not np.isfinite(fitted.fun):
            raise ValueError("Refined place projection did not converge")
        mapping.strength = float(fitted.x)
        return mapping


def fit_recent_place_refinement(predictor, X, groups, won, placed, finish_positions,
                                race_dates, *, soft_y=None, model_params=None,
                                race_types=None, handicap=None, checkpoint_directory=None):
    """Two fixed-cap earlier heads supply a purged, at-most90-day recent slab.

    The earlier win head supplies honest win-probability lower bounds during
    place fitting. The final win model and its calibrated outputs are unchanged.
    """
    import hashlib
    import json
    from pathlib import Path
    import joblib
    import pandas as pd
    import config
    from src.model import RacePredictor, compute_recency_sample_weights

    groups = np.asarray(groups,int)
    dates = pd.DatetimeIndex(race_dates)
    span = int((dates.max()-dates.min()).days)+1
    # Small unit/development datasets retain a bounded chronological slab;
    # production histories use the registered90-calendar-day window.
    window = min(90,max(14,int(span*.2)))
    first = dates.max().normalize()-pd.Timedelta(days=window-1)
    before = dates <= first-pd.Timedelta(days=int(config.PURGE_DAYS))
    cal = dates >= first
    train_rows,cal_rows = np.repeat(before,groups),np.repeat(cal,groups)
    if train_rows.sum()<100 or cal_rows.sum()<50 or cal.sum()<5:
        return None,{"method":"legacy_fallback_insufficient_recent_history","window_days":window}
    X = np.asarray(X)
    learner = RacePredictor(frameworks=predictor.frameworks,random_seed=predictor.random_seed)
    learner._autotune_njobs = predictor._autotune_njobs
    params = model_params or {}
    row_dates = np.repeat(dates.to_numpy(),groups)
    fingerprints = hashlib.sha256()
    for filename in ("model.py","probability_calibration.py"):
        fingerprints.update((Path(__file__).parent/filename).read_bytes())
    fingerprints.update(np.ascontiguousarray(X[train_rows]).tobytes())
    fingerprints.update(json.dumps({"features":predictor.feature_cols,"frameworks":predictor.frameworks,
        "seed":predictor.random_seed,"parameters":params,"win_defaults":config.CLASSIFIER_PARAMS,
        "place_defaults":config.PLACE_CLASSIFIER_PARAMS},sort_keys=True).encode())
    cache = Path(checkpoint_directory) if checkpoint_directory else None
    if cache:
        cache.mkdir(parents=True,exist_ok=True)
    raw = {}
    for role,target,fn in (("win",np.asarray(won),learner._train_win_classifier),
                           ("place",np.asarray(placed),learner._train_place_classifier)):
        role_params = params.get("classifier" if role=="win" else "place")
        hl,seasonal,shape = predictor._extract_recency_params(role_params)
        weights = compute_recency_sample_weights(row_dates[train_rows],half_life_days=hl,seasonal_boost=seasonal,decay_shape=shape)
        teacher = np.asarray(soft_y)[train_rows] if role=="win" and soft_y is not None else None
        key = fingerprints.copy()
        for value in (target[train_rows],weights,teacher):
            if value is not None:
                key.update(np.ascontiguousarray(value).tobytes())
        key.update(role.encode())
        destination = cache/(key.hexdigest()+".joblib") if cache else None
        if destination and destination.exists():
            model = joblib.load(destination)
        else:
            kwargs = {"params":role_params,"sample_weight":weights}
            if role=="win":
                kwargs["soft_y"] = teacher
                kwargs["groups"] = groups[before]
                kwargs["finish_positions"] = np.asarray(finish_positions)[train_rows]
            if hasattr(predictor,"_component"):
                # A separate earlier head is necessary for place refinement;
                # interrupted qualification must resume it just like the final
                # and legacy calibration estimators.
                model = predictor._component("recent_"+role,fn,(X[train_rows],target[train_rows]),kwargs)
            else:
                model = fn(X[train_rows],target[train_rows],**kwargs)
            if destination:
                joblib.dump(model,destination)
        raw[role] = model.predict_proba(X[cal_rows])[:,1]
    cal_groups = groups[cal]
    starts = np.r_[0,np.cumsum(groups)[:-1]]
    if race_types is None:
        codes = np.full(int(cal.sum()),"Unknown")
    else:
        codes = np.asarray(race_types,str)[starts][cal]
    handicap_column = predictor.feature_cols.index("handicap") if "handicap" in predictor.feature_cols else None
    if handicap is not None:
        hcap = np.asarray(handicap)[starts][cal].astype(bool)
    else:
        hcap = X[starts[cal],handicap_column].astype(bool) if handicap_column is not None else np.zeros(len(cal_groups),bool)
    places_paid = np.minimum(cal_groups,np.where(cal_groups<=4,3,np.where(cal_groups<=7,2,np.where((cal_groups>=16)&hcap,4,3))))
    floor = predictor._calibrate_win_probs(raw["win"],cal_groups)
    fp = np.asarray(finish_positions)[cal_rows]
    labels = ((fp>=1)&(fp<=np.repeat(places_paid,cal_groups))).astype(int)
    mapping = PlaceProbabilityRefinement.fit(raw["place"],floor,labels,fp,cal_groups,places_paid,codes)
    return mapping,{"method":"recent_projected","window_days":window,"base_last_date":str(dates[before].max())[:10],
        "first_date":str(dates[cal].min())[:10],"last_date":str(dates[cal].max())[:10],
        "calibration_races":int(cal.sum()),"calibration_runners":int(cal_rows.sum()),"temporary_heads":2}
