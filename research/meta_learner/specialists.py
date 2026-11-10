"""Tracking and same-day specialists: joint, separate and standalone, on the current stack.

    python -m research.meta_learner.specialists corrections [--threads 3]
    python -m research.meta_learner.specialists standalone [--threads 5]
    python -m research.meta_learner.specialists compare

``corrections`` saves out-of-fold Tier 1 corrections (the cheap-screen recipe)
for tracking alone, same-day bias alone and both together, for 2025Q1-2026Q2,
each fitted on earlier seed-42 quarters only.

``standalone`` fits a win model on the tracking domain plus the shared race
context, trained only on rows from July 2023 (RTV from June, ATR from August),
with the full 063 recipe's target and screen capacity. Its forecasts are a
separate pool component.

``compare`` scores, on 2025Q3-2026Q2 and against the stack's joint correction:
the separate corrections summed; the separate corrections with coefficients
fitted on earlier quarters; and the standalone model pooled with the
three-seed average and performance/spread, with and without the same-day
correction. Every combiner is fitted on earlier out-of-fold quarters only.
"""
import argparse

import numpy as np
import pandas as pd

from research.meta_learner import first_ensemble as fe
from research.meta_learner import screen_candidates as sc
from research.meta_learner import screening as s
from research.meta_learner.pool_average import cached, pooled_runners, seed_average
from src.ingestion.storage import atomic_json, atomic_parquet
from src.modelling.evaluation import component_oof
from src.modelling.spec import ModelSpec

ROOT = fe.ARTIFACTS / "first_ensemble_20261109"
OUT = sc.OUT
KEYS = s.KEYS
HISTORY = fe.POOL_HISTORY  # 2025Q1-2025Q2: fits the combiners for the first scored quarter
LATER = s.SCORED
QUARTERS = (*HISTORY, *LATER)
SPECIALIST_START = pd.Timestamp("2023-07-01")
FRAME_START = pd.Timestamp("2023-01-01")  # eight quarters before 2025Q1, as the benchmark's folds require
CORRECTIONS = OUT / "specialist_corrections.parquet"
STANDALONE = OUT / "tracking_specialist_oof.parquet"
SHARED_CONTEXT_EXCLUDED = ("x063_context_month_", "x063_context_field_")


def groups(frame):
    out = {}
    for name in sc.STACK:
        values, columns = sc.CANDIDATES[name]()(frame)
        frame = s.attach(frame, values)
        out[name] = columns
    out["joint"] = [c for name in sc.STACK for c in out[name]]
    return frame, out


def corrections(threads):
    frame, inputs = groups(sc.base_frame())
    keep = frame.quarter.isin(QUARTERS).to_numpy()
    out = frame.loc[keep, [*KEYS, "race_date", "quarter"]].reset_index(drop=True)
    for name, columns in inputs.items():
        fit = s.residual_boost(columns, base="p_single", threads=threads, params=sc.TIER1["params"],
                               **sc.TIER1["boost"])
        correction, _ = s.walk_forward(frame, fit, scored=QUARTERS)
        if not np.isfinite(correction[keep]).all():
            raise ValueError(f"Missing {name} correction")
        out[f"c_{name}"] = correction[keep]
        print(f"{name}: {len(columns)} inputs, {[f['seconds'] for f in fit.info]} s per quarter", flush=True)
    atomic_parquet(out, CORRECTIONS)


def tracking_domain():
    import pyarrow.parquet as pq
    names = pq.read_schema(sc.store_file()).names
    return sc.allowed([n for n in names if n.startswith(("rtv_", "atr_"))
                       and not n.startswith(("rtv_text", "rtv_card", "rtv_pre", "rtv_verdict", "atr_cmt"))
                       and "comment" not in n])


def shared_context():
    return [c for c in sc.spec_features("full-063")
            if c.startswith("x063_context_") and not c.startswith(SHARED_CONTEXT_EXCLUDED)] + ["num_runners", "age"]


def standalone(threads):
    design = fe.read_json(ROOT / "registration.json")
    full = next(spec for spec in design["specs"] if spec["id"] == "full-063")
    domain, context = tracking_domain(), shared_context()
    spec = ModelSpec.from_dict({**full, "id": "tracking-specialist", "features": [*domain, *context]})
    frame = fe.read_input(ROOT, design, [c for c in context if c not in fe.CONTEXT])
    receipt = frame.attrs["modelling_receipt"]
    frame = frame.loc[frame.race_date.ge(FRAME_START)].reset_index(drop=True)
    frame.attrs = {}
    frame = s.attach(frame, sc.read_store(domain, FRAME_START).drop(columns="race_date"))
    frame.attrs["modelling_receipt"] = receipt
    folds = fe.folds_for(frame, QUARTERS, spec.purge_days, min_train_periods=8)
    dataset_start = SPECIALIST_START - pd.DateOffset(months=spec.burn_in_months)
    result, _ = component_oof(frame, [spec], folds=folds, capacity="screen", threads=threads,
                              dataset_start=dataset_start, cache_dir=OUT / "cache")
    result.attrs = {}
    atomic_parquet(result, STANDALONE)
    atomic_json(OUT / "tracking_specialist_registration.json",
                {"spec": spec.to_dict(), "capacity": "screen", "training_from": str(SPECIALIST_START.date()),
                 "frame_from": str(FRAME_START.date()), "domain_inputs": len(domain), "context_inputs": len(context),
                 "folds": list(QUARTERS), "provenance": sc.provenance()})
    print(f"standalone tracking specialist: {len(spec.features)} inputs, {len(result)} forecasts", flush=True)


# --------------------------------------------------------------------------
# Comparison


def race_losses(frame, logp):
    bounds = s.bounds_of(frame.race_id.to_numpy())
    return s.race_loss(s.race_log_softmax(logp, bounds), frame.won.to_numpy(dtype=float), bounds)


def linear_combiner(columns, base="p_average", l2=0.):
    """Coefficients on stored corrections, fitted by a race logit on earlier quarters only.

    Unpenalised: with two coefficients and thousands of races, a ridge of the size used for
    Tier 0 steps (0.001 per unit squared) outweighs gains of 0.0005 and shrinks the weights to 0.2.
    """
    info = []

    def fit_predict(train, test):
        bounds = s.bounds_of(train.race_id.to_numpy())
        beta, fit = s.fit_offset_logit(train[columns].to_numpy(), np.log(train[base].to_numpy()),
                                       train.won.to_numpy(dtype=float), bounds, l2=l2)
        info.append({"fold": test.quarter.iloc[0], **dict(zip(columns, np.round(beta, 4).tolist())), **fit})
        return test[columns].to_numpy() @ beta
    fit_predict.info = info
    return fit_predict


def compare():
    design = fe.read_json(ROOT / "registration.json")
    purge, shrinkage = fe.get_selected(ROOT, design).purge_days, design["pool_shrinkage"]
    frame = sc.pool_frame()
    frame = s.attach(frame, pd.read_parquet(CORRECTIONS, columns=[*KEYS, "c_tracking", "c_same-day-bias", "c_joint"]))

    # The standalone specialist enters the benchmark's log-pool as a third component.
    average, performance = seed_average(), cached("performance")
    specialist = pd.read_parquet(STANDALONE)
    specialist.attrs = {}
    specialist = specialist.assign(component="tracking-specialist")
    labels = fe.read_input(ROOT, design, [])
    weights = {}
    for name, parts in {"p_pool3": [average, performance, specialist],
                        "p_pool_check": [average, performance]}.items():
        summary, _ = fe.score_pool(labels, parts, LATER, purge, shrinkage)
        weights[name] = [f["weights"] for f in summary["pool_fits"]]
        runners = pooled_runners(labels, parts, purge, shrinkage)
        frame = s.attach(frame, runners[[*KEYS, "probability"]].rename(columns={"probability": name}))
    combiner = linear_combiner(["c_tracking", "c_same-day-bias"])
    frame = frame.loc[frame.quarter.isin(QUARTERS)].reset_index(drop=True)
    fitted, _ = s.walk_forward(frame, combiner, scored=LATER)
    later = frame.quarter.isin(LATER).to_numpy()
    scored, fitted = frame.loc[later].reset_index(drop=True), fitted[later]
    if not np.allclose(scored.p_pool_check, scored.p_pool, rtol=0, atol=1e-12):
        raise ValueError("The rebuilt pool differs from the stored pool")
    log = {c: np.log(scored[c].to_numpy()) for c in ("p_single", "p_average", "p_pool", "p_pool3")}
    c = {k: scored[f"c_{k}"].to_numpy() for k in ("tracking", "same-day-bias", "joint")}
    arms = {
        "single": log["p_single"],
        "average": log["p_average"],
        "pool": log["p_pool"],
        "stack_joint": log["p_pool"] + c["joint"],
        "pool_plus_tracking": log["p_pool"] + c["tracking"],
        "pool_plus_same_day": log["p_pool"] + c["same-day-bias"],
        "separate_summed": log["p_pool"] + c["tracking"] + c["same-day-bias"],
        "separate_fitted": log["p_pool"] + fitted,
        "pool_with_specialist": log["p_pool3"],
        "pool_with_specialist_plus_same_day": log["p_pool3"] + c["same-day-bias"],
        "pool_with_specialist_plus_joint": log["p_pool3"] + c["joint"],
    }
    bounds = s.bounds_of(scored.race_id.to_numpy())
    races = scored.iloc[bounds[:-1]][["race_id", "race_date", "quarter"]].reset_index(drop=True)
    races["covered"] = True
    loss = {name: race_losses(scored, logp) for name, logp in arms.items()}
    report = {"races": len(races), "pool_weights": weights, "fitted_coefficients": combiner.info,
              "standalone_alone": {"log_loss": float(race_losses(scored, np.log(np.maximum(
                  scored.merge(specialist[[*KEYS, "probability"]], on=KEYS, how="left").probability.to_numpy(),
                  s.TINY))).mean()), "average_log_loss": float(loss["average"].mean())},
              "vs_average": {}, "vs_stack_joint": {},
              "vs_single": {name: s.paired(loss[name], loss["single"], races)
                            for name in ("average", "stack_joint", "separate_summed", "separate_fitted")},
              "evaluation_status": "retrospective screen on reused development quarters"}
    for name in arms:
        for reference, key in (("average", "vs_average"), ("stack_joint", "vs_stack_joint")):
            if name != reference:
                report[key][name] = s.paired(loss[name], loss[reference], races)
    atomic_json(OUT / "specialists.json", report)
    for name, r in report["vs_average"].items():
        j = report["vs_stack_joint"].get(name)
        joint = f"   vs stack {j['mean']:+.5f} [{j['ci95'][0]:+.5f}, {j['ci95'][1]:+.5f}] {j['quarters_better']}/4" if j else ""
        print(f"{name:<36} vs average {r['mean']:+.5f} [{r['ci95'][0]:+.5f}, {r['ci95'][1]:+.5f}] "
              f"{r['quarters_better']}/4{joint}")
    for name, r in report["vs_single"].items():
        print(f"{name:<36} vs single {r['mean']:+.5f} [{r['ci95'][0]:+.5f}, {r['ci95'][1]:+.5f}] {r['quarters_better']}/4")
    print("pool weights with specialist:", weights["p_pool3"])
    print("fitted coefficients:", combiner.info)
    print("standalone alone:", report["standalone_alone"])


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("stage", choices=["corrections", "standalone", "compare"])
    parser.add_argument("--threads", type=int, default=3)
    args = parser.parse_args()
    if args.stage == "corrections":
        corrections(args.threads)
    elif args.stage == "standalone":
        standalone(args.threads)
    else:
        compare()


if __name__ == "__main__":
    main()
