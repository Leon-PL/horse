"""Audit saved handicap errors and collect omitted weight evidence; never fit.

Existing declared weights remain unchanged. A jockey claim is an extra source
observation, not evidence that a nominal weight is a corrupted record. Reported
claims alone also do not establish the actual weight after any overweight.
"""
from datetime import datetime, timezone
import argparse
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd

from src.ingestion.sporting_life import SportingLifeScraper, _NEXT_DATA_RE, _parse_weight_to_lbs
from src.deployment_validation import file_sha256
from src.established_form_features import race_discipline, mark_scale
from src.forecast_evaluation import valid_race_mask

OUTPUT = ROOT / "data/audit/handicap_weight_review_030"
REFERENCE = ROOT / "data/audit/established_form_029/native_canonical"
STORE = ROOT / "data/processed/feature_store"
KEY = ["race_id", "horse_name"]


def exact_id(value):
    if isinstance(value, bool) or not re.fullmatch(r"[0-9]+", str(value)) or int(value) < 1:
        raise ValueError(f"Ambiguous source identity: {value!r}")
    return str(int(value))


def extract_weights(payload, race_id, date, horse_ids=None):
    """Read event-specific weights/claims, ignoring current profile statistics."""
    race = payload["props"]["pageProps"]["race"]
    summary = race["race_summary"]
    if exact_id(summary["race_summary_reference"]["id"]) != exact_id(race_id):
        raise ValueError("Returned race differs from requested race")
    if pd.Timestamp(summary["date"]).date().isoformat() != str(date):
        raise ValueError("Returned race date differs")
    rows = []
    for ride in race["rides"]:
        status = re.sub(r"[^A-Z]", "", str(ride.get("ride_status", "")).upper())
        if status in {"NONRUNNER", "WITHDRAWN"}:
            continue
        horse = ride["horse"]
        hid = exact_id(horse["horse_reference"]["id"])
        declared = _parse_weight_to_lbs(ride.get("handicap"))
        original = ride.get("jockey_claim")
        claim = np.nan
        if original is not None:
            try:
                number = float(original)
                if not isinstance(original, bool) and np.isfinite(number) and number >= 0 and number.is_integer():
                    claim = number
            except (ValueError, TypeError):
                pass
        adjusted = declared-claim if np.isfinite(declared) and np.isfinite(claim) and claim < declared else np.nan
        rows.append({"race_id": exact_id(race_id), "horse_id": hid, "source_name": horse.get("name"),
            "source_date": str(date), "source_track": summary.get("course_name"),
            "source_stage": summary.get("race_stage"), "declared_weight_lbs": declared,
            "claim_field_present": "jockey_claim" in ride, "claim_raw": json.dumps(original),
            "reported_claim_lbs": claim, "declared_less_reported_claim_lbs": adjusted,
            "other_weight_fields": json.dumps({k:v for k,v in ride.items()
                if any(t in k for t in ("weight", "penalt", "allowance"))}, sort_keys=True)})
    frame = pd.DataFrame(rows)
    if frame.empty or frame.horse_id.duplicated().any():
        raise ValueError("Empty field or duplicate source horse identities")
    if horse_ids is not None and set(frame.horse_id) != {exact_id(value) for value in horse_ids}:
        raise ValueError("Source and saved complete field have different runners")
    return frame


def weight_mark_decomposition(frame):
    """Keep an official-mark move distinct from a declared-burden move."""
    valid = (frame.handicap.eq(1) & frame.previous_handicap.eq(1)
        & frame.mark_scale.notna() & frame.mark_scale.eq(frame.previous_mark_scale)
        & frame.observed_mark.notna() & frame.previous_observed_mark.notna() & frame.weight_change.notna())
    selected = frame.loc[valid].copy()
    selected["mark_change_observed"] = selected.observed_mark-selected.previous_observed_mark
    selected["burden_move_minus_mark_move"] = selected.weight_change-selected.mark_change_observed
    masks = {"all_comparable_previous_handicaps": pd.Series(True,index=selected.index),
        "absolute_weight_move_at_least_7lb": selected.weight_change.abs().ge(7),
        "weight_drop_at_least_7lb_but_mark_drop_at_most_2lb": selected.weight_change.le(-7)&selected.mark_change_observed.ge(-2),
        "weight_rise_at_least_7lb_but_mark_rise_at_most_2lb": selected.weight_change.ge(7)&selected.mark_change_observed.le(2),
        "absolute_move_at_least_7lb_but_mark_move_at_most_2lb": selected.weight_change.abs().ge(7)&selected.mark_change_observed.abs().le(2)}
    contexts = {name:{"runners":int(mask.sum()),"winners":int(selected.loc[mask,"won"].sum()),
        "winner_log_loss_gap_sum":float(selected.loc[mask,"winner_loss_vs_market"].sum())} for name,mask in masks.items()}
    report = {"scope":"Arithmetic/context audit; no rule treating a weight drop as a probability advantage",
        "comparable_runners":len(selected),
        "weight_mark_change_correlation":(float(selected.weight_change.corr(selected.mark_change_observed))
            if selected.weight_change.nunique()>1 and selected.mark_change_observed.nunique()>1 else None),
        "median_absolute_declared_weight_move":float(selected.weight_change.abs().median()) if len(selected) else None,
        "median_absolute_mark_move":float(selected.mark_change_observed.abs().median()) if len(selected) else None,"contexts":contexts,
        "interpretation":"A declared-weight move also reflects different race allocation/conditions and possible allowances or penalties; it is not the same quantity as an official-mark move."}
    return selected, report


def prepare():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    paired = pd.read_parquet(REFERENCE / "market_paired_predictions.parquet")
    paired.race_id = paired.race_id.astype(str)
    if not valid_race_mask(paired).all():
        raise ValueError("Use saved complete single-winner fields")
    field = paired.groupby("race_id", sort=False)
    eligible = field.horse_prev_races.min().ge(5) & field.handicap.first().eq(1)
    paired = paired.loc[paired.race_id.isin(eligible.index[eligible])].copy()
    if paired.race_id.nunique() != 951:
        raise ValueError("Keep the fixed951 established handicap fields")
    names = ["race_id", "horse_id", "horse_name", "race_date", "off_time", "race_name", "race_type",
        "handicap", "track", "surface", "age", "weight_lbs", "has_observed_weight", "official_rating",
        "has_official_rating", "distance_furlongs", "finish_position", "finish_pos_label"]
    raw = pd.read_parquet(STORE / "processed.parquet", columns=names,
                         filters=[("race_date", "<=", pd.Timestamp("2026-03-31"))])
    raw.race_id = raw.race_id.astype(str)
    raw.horse_id = pd.to_numeric(raw.horse_id, errors="raise").astype("Int64").astype(str)
    raw["discipline"] = race_discipline(raw)
    raw["observed_weight"] = raw.weight_lbs.where(raw.has_observed_weight.eq(1))
    raw["observed_mark"] = raw.official_rating.where(raw.has_official_rating.eq(1))
    raw["mark_scale"] = mark_scale(raw)
    raw = raw.sort_values(["horse_id", "race_date", "off_time", "race_id"], kind="stable")
    previous = ["race_id", "race_date", "discipline", "handicap", "observed_weight", "official_rating", "observed_mark", "mark_scale"]
    raw = pd.concat([raw, raw.groupby("horse_id", sort=False)[previous].shift().add_prefix("previous_")], axis=1)
    raw["expected_weight_change"] = raw.observed_weight-raw.previous_observed_weight
    context = raw.loc[raw.race_date.between("2026-01-01", "2026-03-31")].copy()
    context = context.drop(columns=["race_date", "race_type", "handicap", "finish_position"])
    paired = paired.merge(context, on=KEY, validate="one_to_one")
    if not valid_race_mask(paired).all() or paired.race_id.nunique() != 951:
        raise ValueError("Context join changed a saved complete field")
    features = pd.read_parquet(STORE / "featured.parquet", columns=KEY+["race_date", "weight_change", "weight_change_vs_field"],
        filters=[("race_date", ">=", pd.Timestamp("2026-01-01")), ("race_date", "<=", pd.Timestamp("2026-03-31"))])
    features.race_id = features.race_id.astype(str)
    paired = paired.merge(features.drop(columns="race_date"), on=KEY, validate="one_to_one")
    expected = paired.expected_weight_change
    same_day = paired.previous_race_date.eq(paired.race_date)
    discrepancy = ~np.isclose(expected, paired.weight_change, rtol=0, atol=1e-12, equal_nan=True)
    paired["previous_code_changed"] = paired.previous_discipline.notna() & paired.discipline.ne(paired.previous_discipline)
    paired["previous_race_was_conditions"] = paired.previous_handicap.eq(0)
    paired["winner_loss_vs_market"] = np.where(paired.won.eq(1),
        np.log(paired.market_probability.clip(1e-15)/paired.baseline_probability.clip(1e-15)), 0.)
    paired["runner_brier_gap"] = (paired.baseline_probability-paired.won)**2-(paired.market_probability-paired.won)**2
    comparable, decomposition = weight_mark_decomposition(paired)
    comparable.to_parquet(OUTPUT / "weight_mark_decomposition.parquet", index=False)
    (OUTPUT / "weight_mark_decomposition.json").write_text(json.dumps(decomposition,indent=2),encoding="utf-8")
    masks = {"all": pd.Series(True, index=paired.index), "previous_code_changed": paired.previous_code_changed,
        "previous_race_conditions": paired.previous_race_was_conditions,
        "same_code_previous_handicap": ~paired.previous_code_changed & paired.previous_handicap.eq(1),
        "weight_change_at_least_14lb": paired.weight_change.abs().ge(14)}
    summary = {name: {"runners": int(mask.sum()), "fields": int(paired.loc[mask,"race_id"].nunique()),
        "winners": int(paired.loc[mask,"won"].sum()),
        "winner_log_loss_gap_sum": float(paired.loc[mask,"winner_loss_vs_market"].sum()),
        "runner_brier_gap_sum": float(paired.loc[mask,"runner_brier_gap"].sum())} for name, mask in masks.items()}
    paired.to_parquet(OUTPUT / "cohort_context.parquet", index=False)
    paired.loc[discrepancy & ~same_day].to_csv(OUTPUT / "weight_lag_disagreements.csv", index=False)
    paired.loc[paired.previous_code_changed].sort_values("weight_change", key=lambda s:s.abs(), ascending=False).head(30).to_csv(
        OUTPUT / "cross_code_weight_examples.csv", index=False)
    winners = paired.loc[paired.won.eq(1)].sort_values("winner_loss_vs_market", ascending=False)
    losers = paired.loc[paired.won.eq(0)].sort_values("runner_brier_gap", ascending=False)
    winners.head(30).to_csv(OUTPUT / "missed_handicap_winners.csv", index=False)
    losers.head(30).to_csv(OUTPUT / "overrated_handicap_losers.csv", index=False)
    random_ids = pd.Series(sorted(paired.race_id.unique())).sample(n=32, random_state=42).tolist()
    missed_ids = winners.race_id.drop_duplicates().head(8).tolist()
    loser_ids = losers.race_id.drop_duplicates().head(8).tolist()
    selected = list(dict.fromkeys(random_ids+missed_ids+loser_ids))
    manifest = {"scope": "Saved Q1 descriptive error/source audit; zero learner fits and no forecast or production change",
        "population": "951 fields where every runner has at least5 prior local starts and the race is a handicap",
        "random_seed": 42, "random_fields": random_ids, "missed_winner_fields": missed_ids,
        "overrated_loser_fields": loser_ids, "selected_fields": selected,
        "sampling_note": "32 simple-random fields plus targeted error cases. Report random and targeted samples separately.",
        "weight_lag_disagreements_excluding_same_day": int((discrepancy & ~same_day).sum()),
        "same_day_previous_rows": int(same_day.sum()), "context_summary": summary,
        "evidence_sha256": {str(path.relative_to(ROOT)):file_sha256(path) for path in
            (REFERENCE / "market_paired_predictions.parquet", STORE / "meta.json", Path(__file__))}}
    target = OUTPUT / "manifest.json"
    if target.exists():
        saved = json.loads(target.read_text())
        # The collector can receive a URL/cache fix without changing the
        # already fixed sample or its context. Preserve the original receipt;
        # source_summary records the actual collector version on each run.
        comparable = json.loads(json.dumps(saved))
        script_name = str(Path(__file__).relative_to(ROOT))
        comparable["evidence_sha256"][script_name] = manifest["evidence_sha256"][script_name]
        if comparable != manifest:
            raise ValueError("Do not silently change the source pilot selection")
        return paired, saved
    target.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return paired, manifest


def collect(paired, manifest):
    cache = OUTPUT / "sources"
    cache.mkdir(exist_ok=True)
    scraper = SportingLifeScraper()
    rows, receipts = [], []
    for i, identity in enumerate(manifest["selected_fields"]):
        local = paired.loc[paired.race_id.eq(identity)]
        date = local.race_date.iloc[0].date().isoformat()
        course = re.sub(r"[^a-z0-9]+", "-", local.track.iloc[0].lower()).strip("-")
        slug = re.sub(r"[^a-z0-9]+", "-", local.race_name.iloc[0].lower()).strip("-")
        url = f"https://www.sportinglife.com/racing/results/{date}/{course}/{identity}/{slug}"
        path = cache / f"{identity}.json"
        receipt = {"race_id": identity, "url": url, "random_sample": identity in manifest["random_fields"],
                   "collected_at_utc": datetime.now(timezone.utc).isoformat()}
        try:
            if path.exists():
                receipt["cache_hit"] = True
                receipt["read_at_utc"] = receipt["collected_at_utc"]
                receipt["collected_at_utc"] = datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat()
                payload = json.loads(path.read_text())
            else:
                receipt["cache_hit"] = False
                response = scraper._get(url, max_retries=1)
                match = _NEXT_DATA_RE.search(response.text)
                if match is None:
                    raise ValueError("No public Next.js data")
                payload = json.loads(match.group(1))
                extract_weights(payload, identity, date, local.horse_id)
                path.write_text(json.dumps(payload, ensure_ascii=True, separators=(",", ":")), encoding="utf-8")
            extracted = extract_weights(payload, identity, date, local.horse_id)
            extracted["random_sample"] = receipt["random_sample"]
            joined = extracted.merge(local, on=["race_id", "horse_id"], validate="one_to_one")
            joined["source_declared_weight_difference"] = joined.declared_weight_lbs-joined.observed_weight
            rows.append(joined)
            receipt.update(passed=True, runners=len(joined), positive_claims=int(joined.reported_claim_lbs.gt(0).sum()), sha256=file_sha256(path))
        except Exception as error:
            receipt.update(passed=False, error=f"{type(error).__name__}: {error}")
        receipts.append(receipt)
        print(json.dumps({"field": i+1, "total": len(manifest["selected_fields"]), **receipt}), flush=True)
        (OUTPUT / "source_receipts.json").write_text(json.dumps(receipts, indent=2), encoding="utf-8")
        if not receipt["passed"] and any(code in receipt["error"] for code in ("403", "429")):
            break
    if rows:
        source = pd.concat(rows, ignore_index=True)
        source.to_parquet(OUTPUT / "source_weight_observations.parquet", index=False)
        source.loc[source.reported_claim_lbs.gt(0)].to_csv(OUTPUT / "reported_claim_examples.csv", index=False)
        breakdown = {}
        for label, sample in (("random_fields",source.loc[source.random_sample]), ("all_collected",source)):
            positive = sample.reported_claim_lbs.gt(0)
            breakdown[label] = {"fields": int(sample.race_id.nunique()), "runners": len(sample),
                "runners_with_explicit_positive_claim": int(positive.sum()),
                "fields_with_positive_claim": int(sample.loc[positive,"race_id"].nunique()),
                "claim_distribution": sample.loc[positive,"reported_claim_lbs"].value_counts().sort_index().to_dict(),
                "winner_positive_claims": int(sample.loc[positive,"won"].sum()),
                "unknown_claim_rows": int(sample.reported_claim_lbs.isna().sum()),
                "unknown_declared_weight_rows": int(sample.declared_weight_lbs.isna().sum()),
                "declared_weight_disagreements": int(sample.source_declared_weight_difference.abs().gt(1e-12).sum())}
        (OUTPUT / "source_summary.json").write_text(json.dumps({"collected_at_utc":datetime.now(timezone.utc).isoformat(),
            "collector_sha256":file_sha256(Path(__file__)),
            "samples":breakdown, "failed_fields":[v for v in receipts if not v["passed"]],
            "limitations":["Reported claims are omitted inputs, not proof that any particular prediction was wrong because of the claim.",
                "A claim reduces declared weight; overweight and other allowances must be checked before calling this actual carried weight.",
                "Absent claim fields are preserved as unknown, not silently converted to zero.",
                "No model fitting, gain claim, core-data changes or deployment changes."]},indent=2),encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collect", action="store_true", help="Cache the fixed public sample and extract omitted claims")
    parser.add_argument("--collect-only", action="store_true", help="Reuse the saved context and fixed sample without rebuilding the audit")
    args = parser.parse_args()
    if args.collect_only:
        frame = pd.read_parquet(OUTPUT / "cohort_context.parquet")
        registered = json.loads((OUTPUT / "manifest.json").read_text())
        if file_sha256(REFERENCE / "market_paired_predictions.parquet") != registered["evidence_sha256"][str((REFERENCE / "market_paired_predictions.parquet").relative_to(ROOT))]:
            raise ValueError("The fixed source sample's forecasts changed")
    else:
        frame, registered = prepare()
        print(json.dumps(registered, indent=2), flush=True)
    if args.collect or args.collect_only:
        collect(frame, registered)
