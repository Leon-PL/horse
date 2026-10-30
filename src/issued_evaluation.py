"""Score immutable issued fields, retaining explicitly confirmed withdrawals."""
from __future__ import annotations

from collections import Counter
import json
import numpy as np
import pandas as pd

from src.forecast_evaluation import probability_metrics, valid_race_mask
from src.forward_store import timestamp
from src.utils import normalise_horse_key
from src.deployment_validation import file_sha256
from src.matchbook_signals import clean_runner_name


def tail_diagnostics(probabilities, winners):
    """Describe extreme forecasts without flooring, filtering or selecting them."""
    p,y = np.asarray(probabilities,float),np.asarray(winners,float)
    if p.shape != y.shape or p.ndim != 1 or not np.isfinite(p).all() or not np.isin(y,[0,1]).all():
        raise ValueError("Tail diagnostics require aligned finite forecasts and binary results")
    result = {}
    for name,cutoff in (("one_percent",.01),("one_tenth_percent",.001),("one_millionth",1e-6)):
        mask = p<=cutoff
        result[name] = {"runner_forecasts":int(mask.sum()),"predicted_wins":float(p[mask].sum()),
                        "observed_wins":int(y[mask].sum())}
    won = p[y==1]
    result["minimum_winner_probability"] = float(won.min()) if len(won) else None
    result["maximum_winner_log_loss"] = float(-np.log(np.clip(won,1e-15,1.)).min()) if len(won) else None
    return result


def place_comparison(frame):
    """Paired Brier for the immutable issued top-k event; no model selection."""
    required = {"model_place_probability","legacy_place_probability","forecast_places","finish_position"}
    if not required.issubset(frame):
        return None
    offered = pd.to_numeric(frame.forecast_places,errors="coerce").to_numpy()
    if not np.isfinite(offered).all() or (offered<1).any() or (offered != np.floor(offered)).any():
        raise ValueError("Recorded top-k place definition is invalid")
    label = (frame.finish_position.to_numpy()>=1)&(frame.finish_position.to_numpy()<=offered)
    scores = {}
    for role in ("model","legacy"):
        p = frame[role+"_place_probability"].to_numpy()
        if not np.isfinite(p).all() or ((p<0)|(p>1)).any():
            raise ValueError("Recorded place forecasts are invalid")
        scores[role+"_brier"] = float(np.mean((p-label)**2))
    return {**scores,"delta_model_minus_legacy":scores["model_brier"]-scores["legacy_brier"],
        "scope":"Descriptive paired place scores using the offered top-k definition at issue; no automatic promotion"}


def assess_issued(study, results, withdrawals=None):
    """Keep issued probabilities unchanged; absence alone is not a withdrawal.

    Withdrawal evidence requires event ID, horse key, explicit confirmation,
    source and observation time. New starters and uncertain identities exclude
    a whole field and remain in coverage. This does not replace the existing
    stricter identical-field benchmark.
    """
    labels = results.copy()
    labels["race_id"] = labels.race_id.astype(str)
    labels["horse_key"] = labels.horse_name.map(normalise_horse_key)
    labels["valid_field"] = valid_race_mask(labels)
    withdrawals = pd.DataFrame() if withdrawals is None else withdrawals.copy()
    required = {"event_id", "horse_key", "confirmed_withdrawn", "source", "source_sha256", "observed_at", "runner_id", "market_id"}
    if not withdrawals.empty:
        if not required.issubset(withdrawals):
            raise ValueError("Withdrawal evidence is incomplete")
        if withdrawals[["event_id", "horse_key"]].duplicated().any():
            raise ValueError("Withdrawal identities are ambiguous")
        for row in withdrawals.to_dict("records"):
            if row["confirmed_withdrawn"] is not True or not isinstance(row["source"], str) or not row["source"].strip():
                raise ValueError("Withdrawals require explicit positive source evidence")
            timestamp(row["observed_at"])
            source = (study.path/row["source"]).resolve()
            if not source.is_relative_to(study.path.resolve()) or file_sha256(source) != row["source_sha256"]:
                raise ValueError("Withdrawal source receipt changed or escaped its study")
            proof = json.loads(source.read_text())
            if str(proof["event_id"]) != str(row["event_id"]) or proof["observed_at"] != row["observed_at"]:
                raise ValueError("Withdrawal source identity/time differs from its evidence")
            matches = [r for market in proof["payload"].get("markets",[]) if str(market.get("id")) == str(row["market_id"])
                       for r in market.get("runners",[]) if str(r.get("id")) == str(row["runner_id"])]
            if (len(matches) != 1 or matches[0].get("withdrawn") is not True
                    or normalise_horse_key(clean_runner_name(matches[0].get("name"))) != row["horse_key"]):
                raise ValueError("Withdrawal source does not explicitly confirm this runner")
    coverage, excluded, paired = Counter(), [], []
    changes = Counter()
    for record in study.selected_records():
        folder = study.path/"records"/record["capture_id"]
        issued = pd.read_parquet(folder/"forecasts.parquet")
        if issued.race_id.isna().any() or issued.race_id.nunique() != 1 or issued.horse_key.duplicated().any():
            raise ValueError("Issued field identity is ambiguous")
        if not issued.horse_key.map(normalise_horse_key).eq(issued.horse_key).all():
            raise ValueError("Issued horse keys do not follow the shared identity contract")
        outcome = labels.loc[labels.race_id.eq(str(issued.race_id.iloc[0]))]
        issue_keys, result_keys = set(issued.horse_key), set(outcome.horse_key)
        missing = issue_keys-result_keys
        known = set()
        if not withdrawals.empty:
            evidence = withdrawals.loc[withdrawals.event_id.astype(str).eq(str(record["event_id"]))]
            if any(timestamp(v) < timestamp(record["published_at"]) for v in evidence.observed_at):
                raise ValueError("Later-withdrawal evidence predates the issued field")
            known = set(evidence.horse_key)
            for item in evidence.to_dict("records"):
                matching = issued.loc[issued.horse_key.eq(item["horse_key"])]
                if (matching.empty or str(record["market_id"]) != str(item["market_id"])
                        or "runner_id" not in matching or str(matching.runner_id.iloc[0]) != str(item["runner_id"])):
                    raise ValueError("Withdrawal and issued exchange runner identities differ")
            if known & result_keys:
                raise ValueError("A confirmed withdrawn horse also appears among result starters")
        reason = None
        if outcome.empty:
            reason = "results_pending"
        elif not outcome.valid_field.all():
            reason = "results_incomplete_or_ambiguous"
        elif not outcome.race_date.astype(str).str[:10].eq(timestamp(record["first_start"]).date().isoformat()).all():
            reason = "result_date_mismatch"
        elif result_keys-issue_keys:
            reason = "new_starter_not_in_issued_field"
        elif missing-known:
            reason = "absent_runner_without_confirmed_withdrawal"
        if reason:
            coverage[reason] += 1
            excluded.append({"event_id": record["event_id"], "reason": reason})
            continue
        merged = issued.merge(outcome[["horse_key", "won","finish_position"]], on="horse_key", how="left", validate="one_to_one")
        merged["withdrawn_after_issue"] = merged.horse_key.isin(missing)
        merged.loc[merged.withdrawn_after_issue, "won"] = 0
        merged.loc[merged.withdrawn_after_issue,"finish_position"] = 0
        merged["event_id"] = str(record["event_id"])
        # Save forecast mass lost to withdrawals; never renormalise it away.
        for role in ("model", "market"):
            coverage[role+"_withdrawn_probability_mass"] += float(merged.loc[merged.withdrawn_after_issue, role+"_probability"].sum())
        coverage["scored_fields"] += 1
        coverage["fields_with_later_withdrawals"] += bool(missing)
        coverage["confirmed_withdrawn_runners"] += len(missing)
        if (folder/"racecard.parquet").exists():
            card = pd.read_parquet(folder/"racecard.parquet")
            card["horse_key"] = card.horse_name.map(normalise_horse_key)
            for col in ("going", "jockey", "weight_lbs", "draw", "official_rating"):
                if col in card and col in outcome:
                    comparison = card[["horse_key", col]].merge(outcome[["horse_key", col]], on="horse_key", validate="one_to_one")
                    observed = comparison[col+"_x"].notna() & comparison[col+"_y"].notna()
                    changes[col+"_observed_runner_pairs"] += int(observed.sum())
                    if col in {"weight_lbs", "draw", "official_rating"}:
                        different = ~np.isclose(pd.to_numeric(comparison[col+"_x"], errors="coerce"),
                                               pd.to_numeric(comparison[col+"_y"], errors="coerce"), equal_nan=True)
                    else:
                        different = comparison[col+"_x"].astype(str).ne(comparison[col+"_y"].astype(str))
                    changes[col+"_changed_runner_pairs"] += int((observed & different).sum())
        paired.append(merged)
    report = {"scope": "Issued-field forecast scores; confirmed later withdrawals retained with zero wins and original probability mass",
              "coverage": dict(coverage), "exclusions": excluded, "card_to_result_changes": dict(changes),
              "renormalised_after_withdrawals": False, "formal_confirmation_assessed": False}
    if paired:
        combined = pd.concat(paired, ignore_index=True)
        groups = combined.groupby("event_id", sort=False).size().to_numpy()
        report["metrics"] = {role: probability_metrics(combined[role+"_probability"], combined.won, groups)[0]
                             for role in ("model", "market")}
        report["tail_risk"] = {role:tail_diagnostics(combined[role+"_probability"],combined.won)
                               for role in ("model","market")}
        report["place_comparison"] = place_comparison(combined)
    return report
