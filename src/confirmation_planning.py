"""Variance-based feasibility estimates, separate from forecast confirmation.

Normal approximations are planning assumptions, not evidence of a future
candidate's mean or an automatic market-parity decision.
"""
from __future__ import annotations

import math
from statistics import NormalDist

import numpy as np
import pandas as pd

from src.forecast_evaluation import valid_race_mask, validate_probabilities

METRICS = ("race_log_loss", "runner_brier")


def race_differences(frame, candidate_column, market_column):
    """Retain all runners; Brier totals support runner-weighted ratio inference."""
    if frame.empty or not valid_race_mask(frame).all():
        raise ValueError("Planning requires validated complete single-winner fields")
    ordered = frame.sort_values(["race_date", "race_id", "horse_name"], kind="stable").reset_index(drop=True)
    groups = ordered.groupby("race_id", sort=False).size().to_numpy()
    p, y, _, starts = validate_probabilities(ordered[candidate_column], ordered.won, groups)
    q, _, _, _ = validate_probabilities(ordered[market_column], ordered.won, groups)
    dates = pd.to_datetime(ordered.race_date).dt.normalize()
    losses = pd.DataFrame({
        "race_id": ordered.race_id.iloc[starts].to_numpy(),
        "race_date": dates.iloc[starts].to_numpy(), "runners": groups,
        "race_log_loss": np.add.reduceat(y*(np.log(np.clip(q, 1e-12, 1))-np.log(np.clip(p, 1e-12, 1))), starts),
        "brier_total": np.add.reduceat(np.square(p-y)-np.square(q-y), starts),
    })
    return losses


def variance_estimates(races):
    """Per-race variance equivalents for race and whole-date clustered means.

    Brier uses (race loss total - overall runner mean * field size) / mean
    field size. Equal averaging of per-race Brier would change the estimand.
    """
    n = len(races)
    if n < 2 or races.race_id.duplicated().any():
        raise ValueError("At least two unique complete races are required")
    sizes = races.runners.to_numpy(float)
    values = races[["race_log_loss", "brier_total"]].to_numpy(float)
    if (not np.isfinite(values).all() or not np.isfinite(sizes).all()
            or (sizes < 2).any() or (sizes != np.floor(sizes)).any()):
        raise ValueError("Finite race losses and valid field sizes are required")
    means = np.array([values[:, 0].mean(), values[:, 1].sum()/sizes.sum()])
    influence = np.column_stack([values[:, 0]-means[0],
                                (values[:, 1]-means[1]*sizes)/sizes.mean()])
    independent = influence.T@influence/(n-1)
    dates = pd.to_datetime(races.race_date, errors="coerce").dt.normalize()
    if dates.isna().any() or dates.nunique() < 2:
        raise ValueError("At least two valid race dates are required")
    totals = pd.DataFrame(influence, columns=METRICS).groupby(dates.to_numpy(), sort=False).sum().to_numpy()
    g = len(totals)
    clustered = (g/(g-1))*(totals.T@totals)/n
    return {"races": n, "runners": int(sizes.sum()), "dates": g,
            "first_date": str(dates.min().date()), "last_date": str(dates.max().date()),
            "observed_mean_candidate_minus_market": dict(zip(METRICS, means.tolist())),
            "variance_per_race": {"race": dict(zip(METRICS, np.diag(independent).tolist())),
                                  "date": dict(zip(METRICS, np.diag(clustered).tolist()))}}


def sample_size(variance, *, margin, assumed_difference, alpha=.05, joint_power=.8, gates=4):
    """Conservative union-bound power for conjunction of four planned gates.

    Four gates = two scores, each checked using race and date uncertainty.
    Each gate targets 1-(1-joint_power)/gates marginal power. This is a
    normal planning approximation, conditional on the supplied variance.
    """
    vals = (variance, margin, assumed_difference, alpha, joint_power)
    if not all(math.isfinite(float(v)) for v in vals) or variance < 0:
        raise ValueError("Finite assumptions and nonnegative variance required")
    if not 0 < alpha < .5 or not .5 < joint_power < 1 or type(gates) is not int or gates < 1:
        raise ValueError("Invalid error rate, joint power or gate count")
    gap = margin-assumed_difference
    if gap <= 0:
        return None  # No sample size delivers high power under this null-side mean.
    z = NormalDist().inv_cdf(1-alpha)+NormalDist().inv_cdf(1-(1-joint_power)/gates)
    return max(2, math.ceil(z*z*variance/(gap*gap)))


def planning_scenario(summaries, *, margins, assumed_differences, alpha=.05, joint_power=.8):
    requirements = []
    for label, summary in summaries.items():
        for sampling, variances in summary["variance_per_race"].items():
            for metric in METRICS:
                requirements.append({"slice": label, "sampling": sampling, "metric": metric,
                    "variance_per_race": variances[metric],
                    "required_races": sample_size(variances[metric], margin=margins[metric],
                        assumed_difference=assumed_differences[metric], alpha=alpha, joint_power=joint_power)})
    if not requirements:
        raise ValueError("Planning needs at least one variance estimate")
    finite = all(row["required_races"] is not None for row in requirements)
    limiting = max(requirements, key=lambda row: row["required_races"] or 0) if finite else None
    return {"assumed_difference_candidate_minus_market": assumed_differences,
            "required_races": limiting["required_races"] if limiting else None,
            "status": "approximate_planning_requirement" if finite else "assumed_mean_outside_margin",
            "limiting_requirement": limiting, "requirements": requirements}
