"""The correction model: a race-softmax LightGBM fitted from the win model's forecast as offset.

The win model's out-of-sample forecast enters as a fixed log offset; the
boosters learn only an adjustment to it, from the inputs of
``src.correction.inputs``. The training target is 0.75 x the closing market
probability + 0.25 x the result (the result alone where a race has no closing
price), the same teacher the win model uses. Prediction needs no price or
result of the race being forecast.

Recipe as measured in ``docs/research/ERROR_ANATOMY_2026-11-10.md`` (sections
10 and 12, variant ``more-groups-deeper``), less the same-day bias group,
which reads earlier races on the same card and cannot be known in the morning.
"""
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.correction import inputs as ci
from src.forecast_evaluation import paired_interval
from src.wf_eval import expanding_period_folds, oof_predict_expanding

KEYS = ci.KEYS
TINY = np.finfo(float).tiny
BLEND = .75
PURGE_DAYS = 7
ROUNDS = 1000
SEEDS = (7, 8, 9)
PARAMS = {"learning_rate": .02, "num_leaves": 31, "min_data_in_leaf": 400, "feature_fraction": .5,
          "lambda_l2": 200., "bagging_fraction": .8, "bagging_freq": 1, "max_bin": 63,
          "min_sum_hessian_in_leaf": 1., "verbose": -1, "metric": "None", "deterministic": True,
          "force_row_wise": True}
GROUPS = ("horse", "market-gap", "residual", "wave2", "tracking", "horse-state", "entity-recent", "franking",
          "hidden-form", "market-moves")


def race_log_softmax(scores, bounds):
    starts, sizes = bounds[:-1], np.diff(bounds)
    top = np.repeat(np.maximum.reduceat(scores, starts), sizes)
    z = scores - top
    return z - np.repeat(np.log(np.add.reduceat(np.exp(z), starts)), sizes)


def race_losses(logp, won, bounds):
    """Winner log loss per race."""
    return -np.add.reduceat(np.where(won > 0, logp, 0.), bounds[:-1])


def soft_target(frame, market="p_market"):
    won = frame.won.to_numpy(float)
    price = frame[market].to_numpy(float)
    return np.where(np.isfinite(price), BLEND * price + (1 - BLEND) * won, won)


def _objective(offset, bounds):
    def objective(preds, data):
        p = np.exp(race_log_softmax(offset + preds, bounds))
        return p - data.get_label(), np.maximum(p * (1 - p), 1e-6)
    return objective


def build_inputs(frame, *, probability="p_base", market="p_market", prices=None, tracking=(), groups=GROUPS,
                 only=None):
    """Inputs for every row of ``frame`` (a history frame, see ``src.correction.inputs``).

    ``frame`` must hold every run needed as history (the franking group follows rivals through all of
    them) with the forecast NaN where a run has none. ``prices`` is row-aligned exchange snapshots for
    the past-moves group. ``tracking`` names feature-store columns already in ``frame`` passed through.
    Returns the values (KEYS plus inputs, rows aligned with ``frame``) and the ordered input names.
    ``only`` (a boolean mask over ``frame``) keeps just those rows of the result: the whole frame is
    still read as history, but nothing wide is built for rows nobody asked for (a day's card).
    """
    unknown = set(groups) - set(GROUPS)
    if unknown:
        raise ValueError(f"Unknown correction groups: {sorted(unknown)}")
    forecast = np.isfinite(frame[probability].to_numpy(float))
    whole = pd.Series(forecast).groupby(frame.race_id.to_numpy()).transform("all").to_numpy()
    if (forecast & ~whole).any():
        raise ValueError("A race has forecasts for only part of its field")
    known = frame.loc[whole].reset_index(drop=True)   # races with a forecast: the rows inputs are built for
    wanted = np.ones(len(known), bool) if only is None else np.asarray(only, bool)[whole]
    residual = ci.run_residuals(known, probability)
    parts, names = [], {}

    def add(group, values):
        columns = [c for c in values if c not in KEYS]
        names[group] = columns
        if not values[KEYS].equals(known[KEYS]):
            raise ValueError(f"Group {group} changed the runner order")
        parts.append(values.loc[wanted, [*KEYS, *columns]].reset_index(drop=True))

    if {"horse", "market-gap", "residual", "wave2"} & set(groups):
        long = ci.long_inputs(known, probability, market, residual)
        columns = [c for c in long if c not in KEYS]
        wave2 = [c for c in columns if any(c.startswith(f"er_{name}_") for name in ci.WAVE2)]
        horse = [c for c in columns if "_horse_" in c]
        rest = [c for c in columns if c not in wave2 and c not in horse]
        split = {"residual": [c for c in rest if c.startswith("er_")], "market-gap": [c for c in rest if c.startswith("mg_")],
                 "horse": horse, "wave2": wave2}
        for group in ("horse", "market-gap", "residual", "wave2"):
            if group in groups:
                add(group, long[[*KEYS, *split[group]]])
    if "tracking" in groups:
        add("tracking", known[[*KEYS, *tracking]])
    if "horse-state" in groups:
        add("horse-state", ci.horse_state(known, probability, market, residual))
    if "entity-recent" in groups:
        add("entity-recent", ci.entity_recent(known, probability, market, residual))
    if "franking" in groups:
        runs = frame[[*KEYS, "race_date", "finish_position"]]
        values = ci.franking_inputs(runs, known, market=market, probability=probability, residual=residual)
        values["race_id"] = values.race_id.astype(known.race_id.dtype)
        add("franking", known[KEYS].merge(values, on=KEYS, how="left", validate="one_to_one"))
    if "hidden-form" in groups:
        add("hidden-form", ci.hidden_form(known, known.form.to_numpy(object),
                                          pd.to_numeric(known.hidden_form_runs, errors="coerce").fillna(0).to_numpy()))
    if "market-moves" in groups:
        if prices is None:
            raise ValueError("The market-moves group needs exchange snapshot prices")
        add("market-moves", ci.market_moves(known, prices.loc[whole].reset_index(drop=True)))
    values = known.loc[wanted, KEYS].reset_index(drop=True)
    columns = []
    for group in groups:                      # declared order fixes the model's input order
        part = parts[list(names).index(group)]
        fresh = [c for c in names[group] if c not in values]
        values = pd.concat([values, part[fresh]], axis=1)
        columns += fresh
    target = frame[KEYS] if only is None else frame.loc[np.asarray(only, bool), KEYS].reset_index(drop=True)
    values = target.merge(values, on=KEYS, how="left", validate="one_to_one")
    return values, columns


@dataclass
class CorrectionModel:
    columns: list
    boosters: list = field(default_factory=list)      # LightGBM model strings, one per seed
    lineage: dict = field(default_factory=dict)

    def fit(self, frame, values, *, probability="p_base", market="p_market", rounds=ROUNDS, params=None,
            seeds=SEEDS, threads=4):
        """Fit on settled races with a forecast. ``values`` is row-aligned with ``frame``."""
        import lightgbm as lgb
        p = frame[probability].to_numpy(float)
        rows = np.isfinite(p) & frame.won.notna().to_numpy()
        train = frame.loc[rows]
        bounds = ci.bounds_of(train.race_id.to_numpy())
        if len(bounds) < 3:
            raise ValueError("No settled races with a forecast to fit the correction on")
        x = values.loc[rows, self.columns].to_numpy(dtype=np.float32, na_value=np.nan)
        offset = np.log(np.maximum(p[rows], TINY))
        data = lgb.Dataset(x, label=soft_target(train, market), free_raw_data=False)
        settings = {**PARAMS, **(params or {}), "num_threads": threads}
        self.boosters = []
        gain = np.zeros(len(self.columns))
        for seed in seeds:
            booster = lgb.train({**settings, "seed": seed, "objective": _objective(offset, bounds)}, data,
                                num_boost_round=rounds)
            gain += booster.feature_importance("gain")
            self.boosters.append(booster.model_to_string())
        self.lineage = {"training_first_date": str(train.race_date.min().date()),
                        "training_last_date": str(train.race_date.max().date()),
                        "training_races": int(len(bounds) - 1), "rounds": rounds, "seeds": list(seeds),
                        "params": {k: v for k, v in settings.items() if k != "num_threads"},
                        "top_inputs": [self.columns[i] for i in np.argsort(-gain)[:12] if gain[i] > 0]}
        return self

    def adjustment(self, values):
        """Raw log-scale adjustment per runner (before the race is renormalised)."""
        import lightgbm as lgb
        if not self.boosters:
            raise ValueError("Correction model is not fitted")
        x = values[self.columns].to_numpy(dtype=np.float32, na_value=np.nan)
        return np.mean([lgb.Booster(model_str=text).predict(x, raw_score=True) for text in self.boosters], axis=0)

    def predict(self, frame, values, *, probability="p_base"):
        """Corrected win probabilities; ``frame`` has contiguous races and a forecast for every row."""
        return corrected(frame[probability].to_numpy(float), self.adjustment(values), frame.race_id.to_numpy())


def corrected(probability, adjustment, race_ids):
    if not np.isfinite(probability).all():
        raise ValueError("Every runner needs a forecast to correct")
    return np.exp(race_log_softmax(np.log(np.maximum(probability, TINY)) + adjustment, ci.bounds_of(race_ids)))


def walk_forward(frame, values, columns, *, first_scored, probability="p_base", market="p_market", rounds=ROUNDS,
                 params=None, seeds=SEEDS, threads=4):
    """Out-of-sample adjustments for quarters from ``first_scored``, each from a fit on earlier quarters.

    ``frame`` holds settled races with a forecast only, in race order; ``values`` is row-aligned.
    Returns the adjustment (NaN before the first scored quarter) and the fold labels.
    """
    # Tracking columns are in both; keep the copy from ``values`` so no input name appears twice.
    data = pd.concat([frame.drop(columns=[c for c in columns if c in frame]).reset_index(drop=True),
                      values[columns].reset_index(drop=True)], axis=1)
    quarters = data.race_date.dt.to_period("Q").astype(str)
    labels = sorted(q for q in quarters.unique() if q >= str(pd.Timestamp(first_scored).to_period("Q")))
    before = int((pd.Index(sorted(quarters.unique())) < labels[0]).sum())
    folds = [f for f in expanding_period_folds(data.race_date, min_train_periods=before, embargo_days=PURGE_DAYS)
             if f.label in labels]
    if [f.label for f in folds] != labels:
        raise ValueError("A scored quarter has no usable fold")

    def fit_predict(train, test):
        model = CorrectionModel(list(columns)).fit(train, train, probability=probability, market=market, rounds=rounds,
                                                   params=params, seeds=seeds, threads=threads)
        return model.adjustment(test)
    adjustment = oof_predict_expanding(data, fit_predict=fit_predict, folds=folds, embargo_days=PURGE_DAYS)
    return adjustment, labels


def compare(frame, adjustment, *, probability="p_base"):
    """Race log loss of the forecast with and without the adjustment, on rows where it is finite."""
    rows = np.isfinite(adjustment)
    part = frame.loc[rows].reset_index(drop=True)
    bounds = ci.bounds_of(part.race_id.to_numpy())
    won = part.won.to_numpy(float)
    base = np.log(np.maximum(part[probability].to_numpy(float), TINY))
    before = race_losses(race_log_softmax(base, bounds), won, bounds)
    after = race_losses(race_log_softmax(base + adjustment[rows], bounds), won, bounds)
    dates = part.race_date.to_numpy()[bounds[:-1]]
    quarter = pd.PeriodIndex(dates, freq="Q").astype(str)
    delta = after - before
    by_quarter = pd.Series(delta).groupby(np.asarray(quarter)).mean()
    return {"races": int(len(delta)), "forecast_log_loss": float(before.mean()),
            "corrected_log_loss": float(after.mean()), **paired_interval(delta, blocks=dates),
            "per_quarter": {k: float(v) for k, v in by_quarter.items()},
            "quarters_better": int((by_quarter < 0).sum()), "quarters": int(len(by_quarter))}
