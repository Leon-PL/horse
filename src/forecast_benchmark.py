"""Field-safe alignment and pre-test selection of market corrections."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.betfair import bf_horse_key, bf_track_key
from src.forecast_evaluation import probability_metrics, valid_race_mask, validate_probabilities
from src.model import apply_market_anchor, fit_market_anchor
from src.probability_calibration import fit_temperature, race_temperature
from src.utils import normalise_off_time_key
from src.wf_eval import purge_days


def match_market_fields(features, snapshots):
    """Pair identical complete fields; never drop a runner then renormalise.

    The existing store describes final starters. Paired evaluation is therefore
    conditional on no field change after the decision. The separate snapshot
    retains all later withdrawals for as-issued market evaluation.
    """
    source = snapshots.copy()
    source["race_date"] = pd.to_datetime(source.race_date).dt.normalize()
    keys = ["race_date", "track_key", "off_key", "horse_key"]
    required = keys + ["market_id", "selection_id", "market_probability", "active_runners",
                       "decision_ms", "observed_ms", "price_changed_ms", "won", "withdrawn_after_decision"]
    missing = set(required) - set(source)
    if missing:
        raise ValueError(f"Snapshot columns missing: {sorted(missing)}")
    malformed = (source[keys].isna().any(axis=1)
                 | source[keys[1:]].eq("").any(axis=1)
                 | source.duplicated(keys, keep=False)
                 | source.duplicated(["market_id", "selection_id"], keep=False)
                 | source.market_id.isna() | source.selection_id.isna()
                 | source[["decision_ms", "observed_ms", "price_changed_ms"]].isna().any(axis=1)
                 | source.observed_ms.gt(source.decision_ms)
                 | source.price_changed_ms.gt(source.observed_ms))
    bad_ids = set(source.loc[malformed, "market_id"])
    for mid, field in source.groupby("market_id", sort=False):
        try:
            validate_probabilities(field.market_probability, field.won, [len(field)])
        except ValueError:
            bad_ids.add(mid)
        if (not field.active_runners.eq(len(field)).all()
                or field[["race_date", "track_key", "off_key", "decision_ms"]].nunique().ne(1).any()):
            bad_ids.add(mid)
    source = source.loc[~source.market_id.isin(bad_ids)].copy()
    frame = features.copy()
    valid = valid_race_mask(frame)
    frame = frame.loc[valid].copy()
    frame["race_date"] = pd.to_datetime(frame.race_date).dt.normalize()
    frame["track_key"] = frame.track.map(bf_track_key)
    frame["off_key"] = frame.off_time.map(normalise_off_time_key)
    frame["horse_key"] = frame.horse_name.map(bf_horse_key)
    ambiguous_features = frame.duplicated(keys, keep=False)
    ambiguous_feature_ids = set(frame.loc[ambiguous_features, "race_id"])
    frame = frame.loc[~frame.race_id.isin(ambiguous_feature_ids)].copy()
    frame = frame.merge(source[required].rename(columns={"won": "market_won"}),
                        on=keys, how="left", validate="one_to_one")
    grouped = frame.groupby("race_id", sort=False)
    same_field = (grouped.market_id.transform("count").eq(grouped.race_id.transform("size"))
                  & grouped.market_id.transform("nunique").eq(1)
                  & frame.active_runners.eq(frame.num_runners)
                  & frame.withdrawn_after_decision.eq(False))
    same_field = same_field.groupby(frame.race_id).transform("all")
    same_labels = frame.won.eq(frame.market_won).groupby(frame.race_id).transform("all")
    accepted = frame.loc[same_field & same_labels].sort_values(
        ["race_date", "race_id", "horse_name"], kind="stable").reset_index(drop=True)
    return accepted, {
        "feature_races": int(features.race_id.nunique()),
        "invalid_feature_races": int(features.loc[~valid, "race_id"].nunique()),
        "ambiguous_feature_key_races": len(ambiguous_feature_ids),
        "malformed_or_ambiguous_market_races": len(bad_ids),
        "unmatched_or_changed_field_races": int(frame.loc[~same_field, "race_id"].nunique()),
        "outcome_disagreement_races": int(frame.loc[same_field & ~same_labels, "race_id"].nunique()),
        "paired_races": int(accepted.race_id.nunique()), "paired_runners": len(accepted),
        "snapshot_markets": int(snapshots.market_id.nunique()),
        "late_withdrawal_market_races": int(snapshots.loc[snapshots.withdrawn_after_decision, "market_id"].nunique()),
        "scope": "Retrospective complete overlap with no late withdrawals; final-starter features are not as-issued racecards",
    }


def _arrays(frame):
    groups = frame.groupby("race_id", sort=False).size().to_numpy()
    if not np.array_equal(frame.race_id.to_numpy(), np.repeat(frame.race_id.drop_duplicates().to_numpy(), groups)):
        raise ValueError("Sort whole contiguous races before applying market corrections")
    return (frame.model_probability.to_numpy(float), frame.market_probability.to_numpy(float),
            groups, frame.won.to_numpy(float))


def fit_corrections(calibration):
    """Fit earlier OOS rows; select shrinkage on later purged calibration rows.

    Every input model forecast must have been generated without fitting that
    row. Test outcomes are never accepted by this API. The identity fallback
    is selected on pre-test validation; it is not an OOS guarantee.
    """
    frame = calibration.sort_values(["race_date", "race_id", "horse_name"]).reset_index(drop=True)
    days = pd.to_datetime(frame.race_date).dt.normalize()
    unique = sorted(days.unique())
    if len(unique) < 14:
        raise ValueError("At least 14 calibration dates are required")
    boundary = pd.Timestamp(unique[int(len(unique) * .65)])
    fit = frame.loc[days <= boundary - pd.Timedelta(days=purge_days())]
    validation = frame.loc[days >= boundary]
    if fit.race_id.nunique() < 50 or validation.race_id.nunique() < 50:
        raise ValueError("At least 50 complete races required in both correction fitting and validation")
    p, q, g, y = _arrays(fit)
    validate_probabilities(p, y, g)
    validate_probabilities(q, y, g)
    temperature = fit_temperature(q, g, y)
    anchor = fit_market_anchor(p, q, g, y)
    if anchor is None:
        anchor = {"alpha": 0., "beta": 1.}
    # Bounded, deliberately small model family. No test-dependent optimisation.
    anchor = {"alpha": float(np.clip(anchor["alpha"], 0, 1)),
              "beta": float(np.clip(anchor["beta"], .5, 1.5))}
    vp, vq, vg, vy = _arrays(validation)
    baseline = probability_metrics(vq, vy, vg)[0]
    temp_score = probability_metrics(race_temperature(vq, vg, temperature), vy, vg)[0]
    selected_temp = temperature if all(temp_score[k] <= baseline[k]
                                      for k in ("brier", "race_log_loss")) else 1.
    selected = {"alpha": 0., "beta": 1.}
    best = baseline["race_log_loss"]
    trials = []
    for shrink in (.25, .5, 1.):
        params = {"alpha": shrink * anchor["alpha"], "beta": 1 + shrink * (anchor["beta"] - 1)}
        score = probability_metrics(apply_market_anchor(vp, vq, vg, **params), vy, vg)[0]
        trials.append({"shrink": shrink, "params": params, "metrics": score})
        if score["brier"] <= baseline["brier"] and score["race_log_loss"] < best:
            selected, best = params, score["race_log_loss"]
    return {
        "temperature": selected_temp, "anchor": selected,
        "fit_first_date": str(fit.race_date.min())[:10], "fit_last_date": str(fit.race_date.max())[:10],
        "validation_first_date": str(validation.race_date.min())[:10],
        "validation_last_date": str(validation.race_date.max())[:10],
        "fit_races": int(fit.race_id.nunique()), "validation_races": int(validation.race_id.nunique()),
        "validation_market": baseline, "temperature_fitted": temperature,
        "validation_temperature": temp_score, "anchor_fitted_bounded": anchor, "anchor_trials": trials,
        "selection_rule": "Pre-test validation: require no worse Brier and lower race log loss; otherwise exact market",
    }


def corrected_probabilities(frame, recipe):
    p, q, groups, y = _arrays(frame)
    validate_probabilities(p, y, groups)
    validate_probabilities(q, y, groups)
    return {"independent": p, "market": q,
            "market_temperature": race_temperature(q, groups, recipe["temperature"]),
            "market_combination": apply_market_anchor(p, q, groups, **recipe["anchor"])}
