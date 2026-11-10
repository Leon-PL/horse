"""Forecast-level components from the plan: head-to-head (S10) and jumps completion (S13).

    python -m research.meta_learner.forecast_components register
    python -m research.meta_learner.forecast_components head-to-head [--threads 4]
    python -m research.meta_learner.forecast_components completion [--threads 3]
    python -m research.meta_learner.forecast_components compare

``head-to-head`` fits a Bradley-Terry model on full 063's inputs: a LightGBM
score trained on every pair of runners in a race, each pair's target being 75%
the market's Luce chance that one beats the other and 25% which one finished
ahead (the full 063 target's market share). Win chances are a race softmax of
the score times a scale fitted on the last 20% of the training races by a
separate earlier fit. Out-of-fold forecasts for 2025Q1-2026Q2 at screen
capacity, one seed, through ``first_ensemble.expert_forecasts``.

``completion`` fits the chance that a jumps runner fails to complete (fell,
pulled up, unseated, refused) on full 063's inputs plus the store's completion
and jumping-error history, out of fold for every screening quarter. The
screen tests whether it says anything about winning beyond the generalist
(``screen_candidates`` group ``completion``).

``compare`` pools head-to-head with the three-seed average and
performance/spread, adds the stack's tracking and same-day corrections, and
scores the result against the stack on 2025Q3-2026Q2. Pool weights for each
quarter are fitted on earlier quarters only.

The recipes and gates are registered by ``register`` before any scoring.
"""
import argparse
import hashlib
import inspect

import numpy as np
import pandas as pd
from numba import njit

from research.meta_learner import first_ensemble as fe
from research.meta_learner import screen_candidates as sc
from research.meta_learner import screening as s
from src.betfair import market_prob_bsp_preferred
from src.ingestion.storage import atomic_json, atomic_parquet
from src.model import compute_recency_sample_weights
from src.modelling.data import numeric_matrix, ordered_frame
from src.race_objectives import fit_boosted, race_bounds, race_softmax, raw_scores
from src.rtv_text import fit_scale

ROOT = fe.ARTIFACTS / "first_ensemble_20261109"
OUT = sc.OUT
KEYS = s.KEYS
H2H = "head-to-head-screen"
COMPLETION = OUT / "completion_oof.parquet"
REGISTRATION = OUT / "forecast_components_registration.json"
JUMPS = ("Hurdle", "Chase")
COMPONENT_QUARTERS = (*fe.POOL_HISTORY, *fe.LATER)  # 2025Q1-Q2 fit the pool weights for 2025Q3
SCREEN_QUARTERS = ("2024Q3", "2024Q4", "2025Q1", "2025Q2", "2025Q3", "2025Q4", "2026Q1", "2026Q2")
COMPLETION_HISTORY = ("horse_non_finish_rate", "horse_non_finish_rate_5", "horse_non_finish_rate_10",
                      "horse_last_run_non_finish", "form_dnf_count", "horse_discipline_dnf_rate_shrunk",
                      "bha_jump_error_last", "bha_jump_error_rate_3", "bha_jump_error_rate_5", "bha_jump_error_count_5",
                      "bha_refused_last", "bha_refused_rate_3", "bha_refused_rate_5", "bha_refused_count_5",
                      "rtv_jump_index_mean3", "rtv_lengths_gained_jumping_mean3", "rtv_positive_jump_percentage_mean3",
                      "rtv_jump_index_last", "rtv_jumps_count3", "atr_cmt_nh_error_rate_3", "atr_cmt_nh_error_rate_5")


def design_and_spec():
    design = fe.read_json(ROOT / "registration.json")
    return design, fe.get_selected(ROOT, design)


def code_sha(*objects):
    """Digest of the fitting code only, so editing the comparison does not invalidate cached fits."""
    return hashlib.sha256("".join(inspect.getsource(o) for o in objects).encode()).hexdigest()


# --------------------------------------------------------------------------
# Head-to-head


@njit(cache=True)
def _pairwise_pass(scores, bounds, rank, teacher, has_teacher, blend, weights, grad, hess):
    loss = 0.
    for r in range(len(bounds) - 1):
        a, b = bounds[r], bounds[r + 1]
        n = b - a
        if n < 2:
            continue
        w = weights[a] / (n - 1)  # every runner carries about one unit of pair weight
        for i in range(a, b):
            for j in range(i + 1, b):
                if rank[i] < rank[j]:
                    h = 1.
                elif rank[i] > rank[j]:
                    h = 0.
                else:
                    h = .5  # dead heat, or both failed to finish
                t = blend * teacher[i] / (teacher[i] + teacher[j]) + (1. - blend) * h if has_teacher[r] else h
                d = scores[i] - scores[j]
                if d >= 0:
                    e = np.exp(-d)
                    sig, softplus = 1. / (1. + e), d + np.log1p(e)
                else:
                    e = np.exp(d)
                    sig, softplus = e / (1. + e), np.log1p(e)
                g = w * (sig - t)
                grad[i] += g
                grad[j] -= g
                hh = w * sig * (1. - sig)
                hess[i] += hh
                hess[j] += hh
                loss += w * (softplus - t * d)
    return loss


class PairwiseBT:
    """Bradley-Terry loss over every pair of runners in a race, on a market-blended pairwise target."""

    def __init__(self, race_ids, finish_positions, teacher, blend, weights):
        self.bounds = race_bounds(race_ids)
        position = np.asarray(finish_positions, dtype=float)
        self.rank = np.where(np.isfinite(position) & (position > 0), position, 1e6)
        teacher = np.asarray(teacher, dtype=float)
        usable = np.isfinite(teacher) & (teacher > 0)
        self.has_teacher = np.add.reduceat(usable.astype(np.int64), self.bounds[:-1]) == np.diff(self.bounds)
        self.teacher = np.where(usable, teacher, 1.)
        self.blend = float(blend)
        self.weights = np.asarray(weights, dtype=float)

    def loss_and_gradients(self, scores, lo=0, hi=None):
        hi = len(self.rank) if hi is None else hi
        r0, r1 = np.searchsorted(self.bounds, [lo, hi])
        if self.bounds[r0] != lo or self.bounds[r1] != hi:
            raise ValueError("Rows must start and end on race boundaries")
        scores = np.asarray(scores, dtype=np.float64)
        grad, hess = np.zeros(len(scores)), np.zeros(len(scores))
        loss = _pairwise_pass(scores, self.bounds[r0:r1 + 1] - lo, self.rank[lo:hi], self.teacher[lo:hi],
                              self.has_teacher[r0:r1], self.blend, self.weights[lo:hi], grad, hess)
        return loss, grad, hess

    def objective(self, lo, hi):
        def fobj(_y_true, y_pred):
            _, grad, hess = self.loss_and_gradients(y_pred, lo, hi)
            return grad, np.maximum(hess, 1e-12)
        return fobj


def _fit_pairwise(rows, spec, capacity, threads):
    weights = compute_recency_sample_weights(rows.race_date, half_life_days=spec.half_life_days,
                                             seasonal_boost=spec.seasonal_boost, decay_shape=spec.decay_shape)
    teacher = market_prob_bsp_preferred(rows)
    loss = PairwiseBT(rows.race_id.to_numpy(), rows.finish_position.to_numpy(), teacher, spec.soft_label_blend, weights)
    return fit_boosted(numeric_matrix(rows, spec.features), np.zeros(len(rows)), rows.race_id.to_numpy(),
                       rows.race_date.to_numpy(), objective=loss.objective, metric=None, capacity=capacity,
                       params=dict(spec.params), screen_params=dict(spec.screen_params), seed=spec.seed, n_jobs=threads,
                       early_stopping_rounds=spec.early_stopping_rounds, purge_days=spec.purge_days, refit=True)


def head_to_head_fit_predict(train, test, spec, dataset_start, capacity, threads):
    """Fit on ``train`` (after burn-in), scale on its last 20% of races, predict ``test`` from inputs only."""
    start = pd.Timestamp(dataset_start) + pd.DateOffset(months=spec.burn_in_months)
    ordered, _ = ordered_frame(train.loc[train.race_date.ge(start)])
    starts = race_bounds(ordered.race_id.to_numpy())[:-1]
    cal_start = ordered.race_date.iloc[starts[int(len(starts) * (1 - spec.calibration_fraction))]]
    earlier = ordered.loc[ordered.race_date.le(cal_start - pd.Timedelta(days=spec.purge_days))].reset_index(drop=True)
    held = ordered.loc[ordered.race_date.ge(cal_start)].reset_index(drop=True)
    temporary = _fit_pairwise(earlier, spec, capacity, threads)
    scale = fit_scale(raw_scores(temporary, numeric_matrix(held, spec.features)), held.won.to_numpy(dtype=float),
                      held.race_id.to_numpy())
    del temporary
    final = _fit_pairwise(ordered, spec, capacity, threads)
    inputs = test.drop(columns=["won", "finish_position", "bsp", "odds"], errors="ignore")
    target, positions = ordered_frame(inputs, labelled=False)
    p = np.empty(len(test))
    p[positions] = race_softmax(scale * raw_scores(final, numeric_matrix(target, spec.features)),
                                race_bounds(target.race_id.to_numpy()))
    return p, {"training_last_date": str(ordered.race_date.max().date()), "calibration_first_date": str(cal_start.date()),
               "calibration_fit_boundary": str(earlier.race_date.max().date()), "scale": scale,
               "training_races": int(len(starts))}


def head_to_head_recipe(spec):
    return {"component": "head-to-head", "spec": spec.to_dict(), "capacity": "screen",
            "code": code_sha(_pairwise_pass.py_func, PairwiseBT, _fit_pairwise, head_to_head_fit_predict),
            "pair_target": f"{spec.soft_label_blend} x market Luce pair chance + {1 - spec.soft_label_blend} x finish order",
            "pair_weight": "recency weight / (field size - 1)", "scale": "fitted on the last 20% of training races"}


def head_to_head(threads):
    design, spec = design_and_spec()
    register()
    frame = fe.read_input(ROOT, design, list(spec.features))

    def fit_predict(train, test):
        return head_to_head_fit_predict(train, test, spec, design["receipt"]["dataset_start"], "screen", threads)

    result = fe.expert_forecasts(OUT, design, frame, H2H, head_to_head_recipe(spec), COMPONENT_QUARTERS,
                                 spec.purge_days, fit_predict)
    print(f"head-to-head: {len(result)} forecasts", flush=True)


# --------------------------------------------------------------------------
# Completion


def completion_inputs(spec):
    return [*spec.features, *sc.allowed([c for c in COMPLETION_HISTORY if c not in spec.features])]


def completion(threads):
    from sklearn.metrics import log_loss, roc_auc_score
    from src.wf_eval import oof_predict_expanding

    design, spec = design_and_spec()
    register()
    features = completion_inputs(spec)
    frame = fe.read_input(ROOT, design, list(spec.features))
    frame.attrs = {}
    extra = [c for c in features if c not in spec.features]
    store = sc.read_store(["race_type", *extra]).drop(columns="race_date")
    store["horse_id"] = store.horse_id.astype(frame.horse_id.dtype)
    frame = frame.merge(store, on=KEYS, how="left", validate="one_to_one")
    start = pd.Timestamp(design["receipt"]["dataset_start"]) + pd.DateOffset(months=spec.burn_in_months)
    parts, info = [], []
    for fold in fe.folds_for(frame, SCREEN_QUARTERS, spec.purge_days):
        details = {}

        def fit_predict(train, test):
            train = train.loc[train.race_type.isin(JUMPS) & train.race_date.ge(start)]
            y = train.finish_position.eq(0).to_numpy(dtype=float)
            weights = compute_recency_sample_weights(train.race_date, half_life_days=spec.half_life_days,
                                                     seasonal_boost=spec.seasonal_boost, decay_shape=spec.decay_shape)
            model = fit_boosted(numeric_matrix(train, features), y, train.race_id.to_numpy(), train.race_date.to_numpy(),
                                objective=lambda lo, hi: "cross_entropy", sample_weight=weights, capacity="screen",
                                params=dict(spec.params), screen_params=dict(spec.screen_params), seed=spec.seed,
                                n_jobs=threads, early_stopping_rounds=spec.early_stopping_rounds,
                                purge_days=spec.purge_days)
            details.update(training_last_date=str(train.race_date.max().date()), training_rows=len(train),
                           base_rate=float(y.mean()))
            p = np.full(len(test), np.nan)
            rows = test.race_type.isin(JUMPS).to_numpy()
            p[rows] = model.predict(numeric_matrix(test.loc[rows], features))
            return p

        p = oof_predict_expanding(frame, fit_predict=fit_predict, folds=[fold], embargo_days=spec.purge_days)[fold.test]
        test = frame.loc[fold.test, [*KEYS, "race_date", "finish_position"]].reset_index(drop=True)
        if pd.Timestamp(details["training_last_date"]) > test.race_date.min() - pd.Timedelta(days=spec.purge_days):
            raise ValueError("Completion fit violates the purge")
        part = test[[*KEYS, "race_date"]].assign(fold=fold.label, cp_noncompletion=p)
        scored = np.isfinite(p)
        y = test.finish_position.eq(0).to_numpy()[scored]
        details.update(fold=fold.label, test_rows=int(scored.sum()), test_rate=float(y.mean()),
                       auc=float(roc_auc_score(y, p[scored])), log_loss=float(log_loss(y, p[scored])))
        info.append(details)
        parts.append(part)
        print(details, flush=True)
    result = pd.concat(parts, ignore_index=True)
    atomic_parquet(result, COMPLETION)
    atomic_json(COMPLETION.with_suffix(".json"), {"features": features, "folds": info, "code": code_sha(completion),
                                                  "sha256": fe.sha(COMPLETION)})


def completion_candidate(frame):
    """Screen inputs: the out-of-fold chance of failing to complete, blank for Flat runners."""
    values = pd.read_parquet(COMPLETION).drop(columns=["race_date", "fold"])
    values["race_id"] = values.race_id.astype(str)
    values = values.loc[values.race_id.isin(frame.race_id.unique())].copy()
    values["cp_log_completion"] = np.log1p(-values.cp_noncompletion.clip(upper=1 - 1e-6))
    values["cp_log_completion_vs_field"] = (values.cp_log_completion
                                            - values.groupby("race_id").cp_log_completion.transform("mean"))
    columns = ["cp_noncompletion", "cp_log_completion", "cp_log_completion_vs_field"]
    return values[[*KEYS, *columns]], columns


# --------------------------------------------------------------------------
# Registration and comparison


GATES = {
    "component": ("A pooled component joins the stack if the stack with it in the pool beats the stack: "
                  "95% interval below zero and at least 3 of 4 quarters better (2025Q3-2026Q2)."),
    "correction": ("An input group follows the fixed verdict rule (advance if Tier 0 interval below zero, or Tier 1 "
                   "beats its twin by 0.0003 with 3 of 4 quarters). A survivor is re-checked against three twins and "
                   "summed onto the stack; it joins if that beats the stack with the 95% interval below zero and at "
                   "least 3 of 4 quarters better, and beats the mean of its three twins by at least 0.0003."),
    "scope": "All races. A split by race type is reported but is not a test.",
}


def register():
    _, spec = design_and_spec()
    from research.meta_learner import ability
    fe.immutable_json(REGISTRATION, {
        "registered": "2026-11-09, before any of these components or groups were scored",
        "head_to_head": {k: v for k, v in head_to_head_recipe(spec).items() if k != "code"},
        "completion": {"features": completion_inputs(spec), "target": "finish_position == 0 (did not complete)",
                       "rows": "jumps runners (Hurdle, Chase)", "capacity": "screen", "folds": list(SCREEN_QUARTERS),
                       "screen_inputs": ["cp_noncompletion", "cp_log_completion", "cp_log_completion_vs_field"]},
        "improvement": {"inputs": ["imp_sire_unexposed", "imp_trainer_365", "imp_trainer_unexposed_365",
                                   "imp_horse_last3", "imp_prior_runs", "imp_sire_unexposed_vs_field",
                                   "imp_trainer_365_vs_field"],
                        "unexposed_runs": ability.UNEXPOSED_RUNS, "shrinkage": [ability.SIRE_SHRINK, ability.TRAINER_SHRINK]},
        "bounce": {"inputs": ["bn_last_resid", "bn_last_jump", "bn_last_best", "bn_last_close", "bn_last_distance",
                              "bn_days_since", "bn_runs_60d", "bn_quick_return_x_resid"]},
        "ability_settings": ability.SETTINGS,
        "gates": GATES,
    })


def compare():
    from research.meta_learner.pool_average import cached, pooled_runners, seed_average
    from research.meta_learner.specialists import CORRECTIONS, race_losses

    design, spec = design_and_spec()
    purge, shrinkage = spec.purge_days, design["pool_shrinkage"]
    frame = sc.pool_frame()
    frame = s.attach(frame, pd.read_parquet(CORRECTIONS, columns=[*KEYS, "c_tracking", "c_same-day-bias"]))
    average, performance = seed_average(), cached("performance")
    h2h = pd.read_parquet(OUT / f"{H2H}_oof.parquet")
    h2h.attrs = {}
    h2h = h2h.assign(component="head-to-head")
    labels = fe.read_input(ROOT, design, [])
    weights = {}
    for name, parts in {"p_pool_h2h": [average, performance, h2h], "p_average_h2h": [average, h2h],
                        "p_pool_check": [average, performance]}.items():
        summary, _ = fe.score_pool(labels, parts, s.SCORED, purge, shrinkage)
        weights[name] = [f["weights"] for f in summary["pool_fits"]]
        runners = pooled_runners(labels, parts, purge, shrinkage)
        frame = s.attach(frame, runners[[*KEYS, "probability"]].rename(columns={"probability": name}))
    frame = s.attach(frame, h2h[[*KEYS, "probability"]].rename(columns={"probability": "p_h2h"}))
    scored = frame.loc[frame.quarter.isin(s.SCORED)].reset_index(drop=True)
    if not np.allclose(scored.p_pool_check, scored.p_pool, rtol=0, atol=1e-12):
        raise ValueError("The rebuilt pool differs from the stored pool")
    log = {c: np.log(scored[c].to_numpy()) for c in ("p_single", "p_average", "p_pool", "p_pool_h2h",
                                                      "p_average_h2h", "p_h2h")}
    corrections = scored.c_tracking.to_numpy() + scored["c_same-day-bias"].to_numpy()
    arms = {"single": log["p_single"], "average": log["p_average"], "head_to_head_alone": log["p_h2h"],
            "average_plus_h2h": log["p_average_h2h"], "pool": log["p_pool"], "pool_plus_h2h": log["p_pool_h2h"],
            "stack": log["p_pool"] + corrections, "stack_with_h2h": log["p_pool_h2h"] + corrections}
    bounds = s.bounds_of(scored.race_id.to_numpy())
    races = scored.iloc[bounds[:-1]][["race_id", "race_date", "quarter"]].reset_index(drop=True)
    races["covered"] = True
    loss = {name: race_losses(scored, logp) for name, logp in arms.items()}
    report = {"races": len(races), "pool_weights": weights, "log_loss": {k: float(v.mean()) for k, v in loss.items()},
              "vs_average": {k: s.paired(loss[k], loss["average"], races) for k in arms if k != "average"},
              "vs_stack": {k: s.paired(loss[k], loss["stack"], races) for k in ("stack_with_h2h",)},
              "pool_plus_h2h_vs_pool": s.paired(loss["pool_plus_h2h"], loss["pool"], races),
              "gate": GATES["component"], "evaluation_status": "retrospective screen on reused development quarters"}
    r = report["vs_stack"]["stack_with_h2h"]
    report["passes_gate"] = bool(r["ci95"][1] < 0 and r["quarters_better"] >= 3)
    atomic_json(OUT / "forecast_components.json", report)
    for name, r in {**report["vs_average"], "stack_with_h2h vs stack": report["vs_stack"]["stack_with_h2h"],
                    "pool_plus_h2h vs pool": report["pool_plus_h2h_vs_pool"]}.items():
        print(f"{name:<28} {r['mean']:+.5f} [{r['ci95'][0]:+.5f}, {r['ci95'][1]:+.5f}] {r['quarters_better']}/4")
    print("log loss:", {k: round(v, 5) for k, v in report["log_loss"].items()})
    print("pool weights with head-to-head:", weights["p_pool_h2h"])
    print("passes gate:", report["passes_gate"])


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("stage", choices=["register", "head-to-head", "completion", "compare"])
    parser.add_argument("--threads", type=int, default=3)
    args = parser.parse_args()
    if args.stage == "register":
        register()
    elif args.stage == "head-to-head":
        head_to_head(args.threads)
    elif args.stage == "completion":
        completion(args.threads)
    else:
        compare()


if __name__ == "__main__":
    main()
