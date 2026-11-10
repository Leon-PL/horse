"""Cheap screens of candidate information against stored out-of-fold forecasts.

The first ensemble benchmark (``data/research/meta_learner/first_ensemble_20261109``)
saved chronological out-of-fold forecasts of the full 063 generalist: seed 42 for
2024Q3-2026Q2 and seeds 43/44 for 2025Q1-2026Q2. A candidate is screened on top
of those forecasts without refitting the generalist.

* **Tier 0, offset logit.** A race conditional logit whose offset is the
  reference's log probability, with a few binned coefficients per candidate
  input. Seconds per candidate; no trees.
* **Tier 1, residual boost.** A small LightGBM race-softmax model that starts
  from the same offset and learns only a correction. One fit per scored
  quarter, early-stopped on the last fifth of its own training races. If the
  correction does not beat the offset on those races, it is dropped.

Corrections are learned on the seed-42 forecasts, which cover more training
quarters, and applied to both the seed-42 forecast and the three-seed average
on the scored quarters. The three-seed average is the bar a component must
clear. Every fit uses earlier quarters only, through the shared fold owner with
the seven-day purge.

Context-only corrections recalibrate the reference by race kind without new
information, so Tier 1 results are read against a context-only control.
"""
from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from src.forecast_evaluation import paired_interval
from src.modelling.data import ordered_frame
from src.wf_eval import expanding_period_folds, oof_predict_expanding

KEYS = ["race_id", "horse_id"]
LABELS = ["race_id", "horse_id", "horse_name", "race_date", "num_runners", "finish_position", "won"]
STUDY = Path("data/research/meta_learner/first_ensemble_20261109")
CACHE_DIR = Path("data/research/meta_learner/screens_20261109")
SCORED = ("2025Q3", "2025Q4", "2026Q1", "2026Q2")
PURGE_DAYS = 7
TINY = np.finfo(float).tiny


# --------------------------------------------------------------------------
# Stored reference forecasts


_CACHE = {}


def _all_cached(study):
    """Every verified cached fold forecast in the study, read once per process."""
    import io
    import pyarrow.parquet as pq
    key = str(Path(study).resolve())
    if key in _CACHE:
        return _CACHE[key]
    folders = [f for f in sorted((Path(study) / "cache").iterdir())
               if (f / "receipt.json").exists() and (f / "predictions.parquet").exists()]
    listing = {f.name: json.loads((f / "receipt.json").read_text())["sha256"] for f in folders}
    # The cached files carry a large metadata footer that is slow to parse, so a
    # consolidated copy is kept beside the study and reused while every receipt matches.
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    copy = CACHE_DIR / f"reference_{Path(study).name}.parquet"
    manifest = copy.with_suffix(".json")
    if copy.exists() and manifest.exists():
        recorded = json.loads(manifest.read_text())
        if recorded["receipts"] == listing and hashlib.sha256(copy.read_bytes()).hexdigest() == recorded["sha256"]:
            _CACHE[key] = pd.read_parquet(copy)
            return _CACHE[key]
    parts = []
    for folder in folders:
        raw = (folder / "predictions.parquet").read_bytes()
        if hashlib.sha256(raw).hexdigest() != listing[folder.name]:
            raise ValueError(f"Cached forecasts in {folder.name} do not match their receipt")
        part = pq.read_table(io.BytesIO(raw), columns=[*KEYS, "race_date", "fold", "probability", "component"])
        parts.append(part.replace_schema_metadata(None).to_pandas())
    combined = pd.concat(parts, ignore_index=True)
    combined.to_parquet(copy, index=False)
    manifest.write_text(json.dumps({"receipts": listing, "sha256": hashlib.sha256(copy.read_bytes()).hexdigest()},
                                   indent=1))
    _CACHE[key] = combined
    return combined


def cached_forecasts(study, component):
    """Verified fold forecasts of one component from the study's cache."""
    out = _all_cached(study)
    out = out.loc[out.component.eq(component), [*KEYS, "race_date", "fold", "probability"]].reset_index(drop=True)
    if out.empty:
        raise ValueError(f"No cached forecasts for {component}")
    if out.duplicated(KEYS).any():
        raise ValueError(f"Duplicate cached forecasts for {component}")
    return out


def reference_forecasts(study=STUDY, *, original=False):
    """Seed-42 full 063 forecasts plus the three-seed average where all exist."""
    single = cached_forecasts(study, "full-063").rename(columns={"probability": "p_single"})
    seeds = [cached_forecasts(study, f"full-063-seed-{s}")[[*KEYS, "probability"]] for s in (43, 44)]
    out = single
    for i, part in enumerate(seeds):
        out = out.merge(part.rename(columns={"probability": f"_seed{i}"}), on=KEYS, how="left", validate="one_to_one")
    out["p_average"] = (out.p_single + out._seed0 + out._seed1) / 3  # NaN where a seed is missing
    out = out.drop(columns=["_seed0", "_seed1"])
    stored = Path(study) / "screen_seed_average.parquet"
    if stored.exists():
        saved = pd.read_parquet(stored, columns=[*KEYS, "probability"])
        saved.attrs = {}
        check = out.merge(saved, on=KEYS, validate="one_to_one")
        if not np.allclose(check.p_average, check.probability, rtol=0, atol=1e-12):
            raise ValueError("Recomputed seed average differs from the benchmark's stored average")
    if original:
        generalist = cached_forecasts(study, "generalist")[[*KEYS, "probability"]]
        out = out.merge(generalist.rename(columns={"probability": "p_original"}), on=KEYS, how="left",
                        validate="one_to_one")
    return out


def screening_frame(study=STUDY, *, original=False):
    """Settled complete fields with their reference forecasts, sorted by race."""
    reference = reference_forecasts(study, original=original)
    first = reference.race_date.min()
    labels = pd.read_parquet(Path(study) / "input.parquet", columns=LABELS,
                             filters=[("race_date", ">=", pd.Timestamp(first))])
    labels.attrs = {}
    frame = labels.merge(reference.drop(columns="race_date"), on=KEYS, how="inner", validate="one_to_one")
    covered = frame.groupby("race_id").race_id.transform("size")
    expected = labels.loc[labels.race_id.isin(frame.race_id)].groupby("race_id").race_id.size()
    if not covered.eq(frame.race_id.map(expected)).all():
        raise ValueError("A reference forecast does not cover its whole field")
    frame, _ = ordered_frame(frame)
    for column in [c for c in frame if c.startswith("p_")]:
        sums = frame.groupby("race_id")[column].transform("sum")
        valid = frame[column].notna().groupby(frame.race_id).transform("all")
        if not np.allclose(sums[valid], 1, atol=1e-6):
            raise ValueError(f"{column} does not sum to one over a field")
    frame["quarter"] = frame.race_date.dt.to_period("Q").astype(str)
    frame.attrs["study"] = str(study)
    return frame


def attach(frame, features):
    """Left-join candidate inputs by canonical runner key; missing stays NaN."""
    if features.duplicated(KEYS).any():
        raise ValueError("Candidate inputs have duplicate runner keys")
    overlap = (set(features) - set(KEYS)) & set(frame)
    if overlap:
        raise ValueError(f"Candidate inputs repeat frame columns: {sorted(overlap)}")
    out = frame.merge(features, on=KEYS, how="left", validate="one_to_one")
    if not out[KEYS].equals(frame[KEYS]):
        raise ValueError("Joining candidate inputs changed the runner order")
    out.attrs = frame.attrs
    return out


# --------------------------------------------------------------------------
# Race arithmetic on contiguous races


def bounds_of(race_ids):
    ids = np.asarray(race_ids)
    change = np.r_[True, ids[1:] != ids[:-1]]
    if len(pd.unique(ids)) != int(change.sum()):
        raise ValueError("Races must be contiguous")
    return np.r_[np.flatnonzero(change), len(ids)].astype(np.int64)


def race_log_softmax(scores, bounds):
    starts, sizes = bounds[:-1], np.diff(bounds)
    top = np.repeat(np.maximum.reduceat(scores, starts), sizes)
    z = scores - top
    return z - np.repeat(np.log(np.add.reduceat(np.exp(z), starts)), sizes)


def race_loss(logp, won, bounds):
    return -np.add.reduceat(won * logp, bounds[:-1])


# --------------------------------------------------------------------------
# Tier 0: offset conditional logit on binned inputs


@dataclass
class Basis:
    """Training-only bins per input: one indicator per upper bin plus missing."""
    columns: list
    edges: list = field(default_factory=list)
    missing: list = field(default_factory=list)

    @classmethod
    def fit(cls, x, columns, bins=5):
        basis = cls(list(columns))
        for j in range(x.shape[1]):
            v = x[:, j][np.isfinite(x[:, j])]
            values = np.unique(v)
            if len(values) <= 1:
                edges = np.array([])
            elif len(values) <= bins:
                edges = (values[1:] + values[:-1]) / 2
            else:
                edges = np.unique(np.quantile(v, np.arange(1, bins) / bins))
                edges = edges[edges > values[0]]
            basis.edges.append(edges)
            basis.missing.append(bool(np.isnan(x[:, j]).any()) and len(v) > 0)
        return basis

    @property
    def width(self):
        return int(sum(len(e) + m for e, m in zip(self.edges, self.missing)))

    def transform(self, x):
        out = np.zeros((len(x), self.width), dtype=np.float64)
        k = 0
        for j, (edges, missing) in enumerate(zip(self.edges, self.missing)):
            v = x[:, j]
            finite = np.isfinite(v)
            level = np.searchsorted(edges, np.where(finite, v, -np.inf), side="left")
            for b in range(1, len(edges) + 1):
                out[:, k] = finite & (level >= b)  # cumulative steps: each coefficient is a jump
                k += 1
            if missing:
                out[:, k] = ~finite
                k += 1
        return out


def fit_offset_logit(design, offset, won, bounds, l2=1e-3):
    races = len(bounds) - 1
    def objective(beta):
        logp = race_log_softmax(offset + design @ beta, bounds)
        loss = race_loss(logp, won, bounds).sum() / races + l2 * beta @ beta
        grad = design.T @ (np.exp(logp) - won) / races + 2 * l2 * beta
        return loss, grad
    result = minimize(objective, np.zeros(design.shape[1]), jac=True, method="L-BFGS-B",
                      options={"maxiter": 1000})
    return result.x, {"converged": bool(result.success), "iterations": int(result.nit)}


def offset_logit(columns, *, base="p_single", bins=5, l2=1e-3):
    """fit_predict(train, test) -> log probability corrections for the test rows."""
    info = []
    def fit_predict(train, test):
        x_train = train[columns].to_numpy(dtype=float, na_value=np.nan)
        basis = Basis.fit(x_train, columns, bins=bins)
        if not basis.width:
            info.append({"fold": test.quarter.iloc[0], "parameters": 0})
            return np.zeros(len(test))
        bounds = bounds_of(train.race_id.to_numpy())
        beta, fit = fit_offset_logit(basis.transform(x_train), np.log(np.maximum(train[base].to_numpy(), TINY)),
                                     train.won.to_numpy(dtype=float), bounds, l2=l2)
        info.append({"fold": test.quarter.iloc[0], "parameters": basis.width, **fit,
                     "largest_step": float(np.abs(beta).max())})
        return basis.transform(test[columns].to_numpy(dtype=float, na_value=np.nan)) @ beta
    fit_predict.info = info
    return fit_predict


# --------------------------------------------------------------------------
# Tier 1: residual LightGBM on the reference offset

BOOST = {"learning_rate": .05, "num_leaves": 15, "min_data_in_leaf": 400, "feature_fraction": .8,
         "lambda_l2": 10., "max_bin": 63, "min_sum_hessian_in_leaf": 1., "verbose": -1, "metric": "None",
         "seed": 7, "deterministic": True, "force_row_wise": True}


def residual_boost(columns, *, base="p_single", rounds=400, stopping=40, validation_fraction=.2,
                   threads=2, params=None, seeds=(7,)):
    """Tier 1 correction. ``stopping=None`` fits a fixed number of rounds on all training rows."""
    import lightgbm as lgb
    info = []
    settings = {**BOOST, "num_threads": threads, **(params or {})}
    if stopping is None:
        return _fixed_boost(columns, base=base, rounds=rounds, settings=settings, info=info, seeds=seeds)

    def fit_predict(train, test):
        started = time.perf_counter()
        bounds = bounds_of(train.race_id.to_numpy())
        race_dates = train.race_date.to_numpy()[bounds[:-1]]
        cut = int((len(bounds) - 1) * (1 - validation_fraction))
        valid_start = race_dates[cut]
        fit_rows = train.race_date.le(pd.Timestamp(valid_start) - pd.Timedelta(days=PURGE_DAYS)).to_numpy()
        valid_rows = train.race_date.ge(valid_start).to_numpy()
        parts = {}
        for name, rows in (("fit", fit_rows), ("valid", valid_rows)):
            part = train.loc[rows]
            parts[name] = (part[columns].to_numpy(dtype=np.float32, na_value=np.nan),
                           np.log(np.maximum(part[base].to_numpy(), TINY)),
                           part.won.to_numpy(dtype=float), bounds_of(part.race_id.to_numpy()))
        x_fit, off_fit, y_fit, b_fit = parts["fit"]
        x_val, off_val, y_val, b_val = parts["valid"]

        def objective(preds, _data):
            p = np.exp(race_log_softmax(off_fit + preds, b_fit))
            return p - y_fit, np.maximum(p * (1 - p), 1e-6)

        def validation(preds, _data):
            return "race_log_loss", float(race_loss(race_log_softmax(off_val + preds, b_val), y_val, b_val).mean()), False

        booster = lgb.train({**settings, "objective": objective}, lgb.Dataset(x_fit, label=y_fit, free_raw_data=False),
                            num_boost_round=rounds, valid_sets=[lgb.Dataset(x_val, label=y_val, free_raw_data=False)],
                            feval=validation, callbacks=[lgb.early_stopping(stopping, verbose=False)])
        best = int(booster.best_iteration or rounds)
        offset_loss = float(race_loss(race_log_softmax(off_val, b_val), y_val, b_val).mean())
        best_loss = float(booster.best_score["valid_0"]["race_log_loss"])
        used = best if best_loss < offset_loss else 0
        x_test = test[columns].to_numpy(dtype=np.float32, na_value=np.nan)
        correction = booster.predict(x_test, num_iteration=used, raw_score=True) if used else np.zeros(len(test))
        gain = booster.feature_importance("gain", iteration=used) if used else np.zeros(len(columns))
        info.append({"fold": test.quarter.iloc[0], "trees": used, "best_iteration": best,
                     "validation_gain": offset_loss - best_loss, "seconds": round(time.perf_counter() - started, 1),
                     "top_inputs": [columns[i] for i in np.argsort(-gain)[:8] if gain[i] > 0]})
        return correction
    fit_predict.info = info
    return fit_predict


def _softmax_objective(offset, bounds):
    def objective(preds, _data):
        p = np.exp(race_log_softmax(offset + preds, bounds))
        return p - _data.get_label(), np.maximum(p * (1 - p), 1e-6)
    return objective


def _fixed_boost(columns, *, base, rounds, settings, info, seeds=(7,)):
    """Fixed-round corrections, averaged over ``seeds`` to damp the fitting noise of one booster."""
    import lightgbm as lgb

    def fit_predict(train, test):
        started = time.perf_counter()
        x = train[columns].to_numpy(dtype=np.float32, na_value=np.nan)
        x_test = test[columns].to_numpy(dtype=np.float32, na_value=np.nan)
        offset = np.log(np.maximum(train[base].to_numpy(), TINY))
        bounds = bounds_of(train.race_id.to_numpy())
        data = lgb.Dataset(x, label=train.won.to_numpy(dtype=float), free_raw_data=False)
        correction, gain = np.zeros(len(test)), np.zeros(len(columns))
        for seed in seeds:
            booster = lgb.train({**settings, "seed": seed, "objective": _softmax_objective(offset, bounds)}, data,
                                num_boost_round=rounds)
            correction += booster.predict(x_test, raw_score=True) / len(seeds)
            gain += booster.feature_importance("gain")
        info.append({"fold": test.quarter.iloc[0], "trees": rounds, "boosters": len(seeds),
                     "training_races": len(bounds) - 1, "seconds": round(time.perf_counter() - started, 1),
                     "top_inputs": [columns[i] for i in np.argsort(-gain)[:8] if gain[i] > 0]})
        return correction
    fit_predict.info = info
    return fit_predict


def permute_within(frame, columns, *, by="quarter", seed=11):
    """Copy of ``frame`` with ``columns`` moved together to random rows of the same ``by`` group.

    The twin keeps each input group's joint distribution and its coverage over
    time but loses any link to the runner, so it measures what the same fit
    gains from capacity alone.
    """
    rng = np.random.default_rng(seed)
    order = np.arange(len(frame))
    for rows in frame.groupby(by, sort=True).indices.values():
        order[rows] = rng.permutation(rows)
    twin = frame.copy()
    twin[columns] = frame[columns].to_numpy()[order]
    return twin


# --------------------------------------------------------------------------
# Walk-forward and paired scoring


def walk_forward(frame, fit_predict, scored=SCORED):
    """Corrections for the scored quarters from fits on earlier quarters only."""
    quarters = sorted(frame.quarter.unique())
    before = quarters.index(scored[0])
    folds = [f for f in expanding_period_folds(frame.race_date, min_train_periods=before, embargo_days=PURGE_DAYS)
             if f.label in scored]
    if [f.label for f in folds] != list(scored):
        raise ValueError("A scored quarter has no usable fold")
    correction = oof_predict_expanding(frame, fit_predict=fit_predict, folds=folds, embargo_days=PURGE_DAYS)
    return correction, folds


def race_table(frame, correction, *, scored=SCORED, bases=("p_single", "p_average"), columns=None):
    """Per-race log loss of each reference before and after the correction."""
    rows = frame.quarter.isin(scored).to_numpy()
    target = frame.loc[rows].reset_index(drop=True)
    delta = correction[rows]
    if not np.isfinite(delta).all():
        raise ValueError("Missing correction on a scored race")
    bounds = bounds_of(target.race_id.to_numpy())
    won = target.won.to_numpy(dtype=float)
    races = target.iloc[bounds[:-1]][["race_id", "race_date", "quarter"]].reset_index(drop=True)
    races["covered"] = True
    if columns:
        present = target[columns].notna().any(axis=1).groupby(target.race_id, sort=False).any()
        races["covered"] = races.race_id.map(present).to_numpy()
    for base in bases:
        reference = np.log(np.maximum(target[base].to_numpy(), TINY))
        races[f"{base}_before"] = race_loss(race_log_softmax(reference, bounds), won, bounds)
        races[f"{base}_after"] = race_loss(race_log_softmax(reference + delta, bounds), won, bounds)
    races.attrs["runners"] = len(target)
    return races


def paired(after, before, races):
    change = np.asarray(after) - np.asarray(before)
    quarters = pd.Series(change).groupby(races.quarter.to_numpy()).mean()
    result = {**paired_interval(change, blocks=races.race_date.to_numpy()),
              "per_quarter": {k: round(float(v), 6) for k, v in quarters.items()},
              "quarters_better": int(quarters.lt(0).sum())}
    covered = races.covered.to_numpy()
    result["covered_races"] = int(covered.sum())
    if covered.any() and not covered.all():
        result["covered_change"] = paired_interval(change[covered], blocks=races.race_date.to_numpy()[covered])
    return result


def summarise(races, bases):
    out = {"races": len(races), "runners": races.attrs.get("runners")}
    for base in bases:
        out[base] = {**paired(races[f"{base}_after"], races[f"{base}_before"], races),
                     "reference_log_loss": float(races[f"{base}_before"].mean()),
                     "log_loss": float(races[f"{base}_after"].mean())}
    return out


def versus(races, control, bases):
    """Paired change of a candidate against a control screen on the same races."""
    if not races.race_id.equals(control.race_id):
        raise ValueError("Screens scored different races")
    return {base: paired(races[f"{base}_after"], control[f"{base}_after"], races) for base in bases}


def run(frame, columns, *, tier, base="p_single", threads=2, l2=1e-3, bins=5, params=None, scored=SCORED,
        bases=("p_single", "p_average"), coverage=None, boost=None):
    started = time.perf_counter()
    if tier == 0:
        fit_predict = offset_logit(columns, base=base, bins=bins, l2=l2)
    elif tier == 1:
        fit_predict = residual_boost(columns, base=base, threads=threads, params=params, **(boost or {}))
    else:
        raise ValueError("tier must be 0 or 1")
    correction, folds = walk_forward(frame, fit_predict, scored=scored)
    races = race_table(frame, correction, scored=scored, bases=bases, columns=coverage or columns)
    result = summarise(races, bases)
    result.update({"tier": tier, "inputs": len(columns), "fits": fit_predict.info,
                   "lightgbm_fits": len(folds) if tier == 1 else 0,
                   "training_base": base, "seconds": round(time.perf_counter() - started, 1),
                   "evaluation_status": "retrospective screen on reused development quarters"})
    return result, races


def reference_inputs(frame, base="p_single"):
    """Runner-level views of the reference forecast, for corrections that interact with it."""
    logp = np.log(np.maximum(frame[base].to_numpy(), TINY))
    centred = logp - pd.Series(logp).groupby(frame.race_id.to_numpy()).transform("mean").to_numpy()
    rank = frame.groupby("race_id")[base].rank(ascending=False, pct=True).to_numpy()
    return pd.DataFrame({"ref_logp_centred": centred, "ref_rank_pct": rank}, index=frame.index)


def summary_line(name, result, bases=("p_average", "p_single")):
    parts = []
    for base in bases:
        r = result[base]
        parts.append(f"{base[2:]} {r['mean']:+.5f} [{r['ci95'][0]:+.5f}, {r['ci95'][1]:+.5f}] {r['quarters_better']}/4")
    return f"{name:<30} T{result['tier']}  " + "   ".join(parts) + f"   ({result['seconds']}s)"
