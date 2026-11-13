"""Read-only feature-set/model-role configuration and honest definition metadata."""
import hashlib
import json
from pathlib import Path
import re


ID = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")
FAMILIES = {
    "horse": ("Horse & form", "Sporting Life", "Prior local starts, form and recorded horse attributes."),
    "ratings": ("Ratings", "Sporting Life", "Pre-race ratings and their race-relative derivatives."),
    "connections": ("Connections", "Sporting Life", "Trainer, jockey and partnership histories."),
    "pedigree": ("Pedigree", "Pedigree / Sporting Life", "Breeding attributes and prior family performance."),
    "race": ("Race & course", "Sporting Life / reference data", "Race conditions, field context and course attributes."),
    "market": ("Market history", "Sporting Life / accepted prices", "Market-derived context; individual dependencies vary."),
    "atr": ("At The Races", "At The Races", "Eligible prior-run measurements. Unobserved starts retain window slots."),
    "rtv": ("Racing TV", "Racing TV", "Prior usable observations and captured pre-race context. History uses observation windows."),
    "bha": ("BHA stewards", "BHA", "Prior attributed reports, coverage and incident history."),
    "other": ("Other inputs", "Definition pending", "Numeric inputs whose individual ownership is not yet catalogued."),
}
DEFINITIONS = {
    "horse_prev_races": ("Previous local starts", "Number of eligible previously recorded starts in local history.", "starts"),
    "horse_win_rate": ("Historical win rate", "Prior local wins divided by eligible prior starts.", "proportion"),
    "days_since_last_race": ("Days since last race", "Elapsed days since the horse's previous eligible local race.", "days"),
    "horse_elo": ("Horse Elo", "The horse's rating before this race, from prior result updates.", "rating points"),
    "horse_glicko": ("Horse Glicko", "The horse's Glicko rating before this race.", "rating points"),
    "going_numeric": ("Going code", "Numeric encoding of the prepared going category; a measurement, not a one-hot indicator.", "code"),
    "bha_report_count_5": ("BHA report support", "Covered eligible report observations within the latest five canonical prior starts.", "reports"),
    "atr_observation_count_5": ("ATR history support", "Eligible ATR observations within the latest five canonical prior starts.", "observations"),
    "rtv_history_count3": ("RTV history support", "Available usable prior observations, capped at three.", "observations"),
    "profile_exact_age_days": ("Exact age", "Days from the accepted foaling date to this race. Uses eligible Racing TV evidence or an agreeing Sporting Life date captured on an earlier day.", "days"),
    "profile_foaling_day_of_year": ("Foaling day of year", "Calendar day of the accepted exact foaling date, from eligible Racing TV or Sporting Life evidence.", "day of year"),
    "rtv_profile_foaled_abroad": ("Foaled outside Britain or Ireland", "One for a known country outside Britain and Ireland and zero for Britain or Ireland. Requires a profile captured by the prediction cutoff.", "indicator"),
    "rtv_profile_known_overseas_starts": ("Known overseas starts", "Distinct starts outside Britain and Ireland dated before the eligible profile capture day. Missing when there is no documented overseas start.", "starts"),
    "rtv_profile_days_since_overseas_start": ("Days since known overseas start", "Days since the latest dated overseas start documented by an eligible profile.", "days"),
    "num_runners": ("Field size", "Recorded number of runners used by the feature engine.", "runners"),
    "claim_adjusted_weight_lbs": ("Claim-adjusted weight", "Observed allocated weight minus the reported rider claim. Missing when either input is unknown.", "lb"),
    "weight_vs_field": ("Weight vs field", "Weight burden minus the field's mean burden. The burden is the claim-adjusted weight where the claim is verified and the allocated weight otherwise. Requires an observed weight for this runner and at least one rival.", "lb"),
    "weight_rank": ("Weight rank", "Rank by weight burden (claim-adjusted where the claim is verified, allocated weight otherwise) among runners with an observed weight, when at least two have one. The heaviest ranks first.", "rank"),
    "weight_change": ("Weight change", "Weight burden today minus the burden on the previous run (claim-adjusted where the claim is verified, allocated weight otherwise).", "lb"),
    "weight_change_vs_field": ("Weight change vs field", "Change in weight burden minus the mean of observed changes when at least two runners have a comparison.", "lb"),
    "handicap_weight_vs_last_win": ("Weight vs last handicap win", "Today's claim-adjusted weight minus the weight at the last handicap win on the same rating scale.", "lb"),
    "reported_jockey_claim_lbs": ("Reported jockey claim", "Weight allowance reported for this rider and race. Uses Sporting Life or a rider-matched Racing TV observation.", "lb"),
    "race_conditional_jockeys": ("Conditional jockeys", "The race title advertises eligibility for conditional jockeys.", "indicator"),
    "race_apprentice_jockeys": ("Apprentice jockeys", "The race title advertises eligibility for apprentice jockeys.", "indicator"),
    "race_amateur_riders": ("Amateur riders", "Race title or verified race conditions identify an amateur or qualified-rider restriction.", "indicator"),
    "base_win_rate": ("Equal-share win baseline", "One divided by the recorded field size (with a minimum denominator of one).", "probability"),
    "sex_Female": ("Female at birth", "One for a prepared female-at-birth category, otherwise zero. This avoids using later changes in recorded sex.", "indicator"),
    "sex_Male": ("Male at birth", "One for a prepared male-at-birth category, otherwise zero.", "indicator"),
    "combined_elo_vs_best_in_race": ("Combined Elo vs best rival", "Combined pre-race Elo minus the highest combined Elo of another runner. The sole leader retains its margin over the next-best rival.", "rating points"),
}


def family(name):
    from src.modelling.catalog import feature_metadata
    return feature_metadata(name)["family"]


def definition(name):
    group = family(name)
    title, source, summary = FAMILIES[group]
    specific = DEFINITIONS.get(name)
    import config
    availability = ("Current definition: "+config.BHA_FEATURE_AVAILABILITY_POLICY+
                    ". This is an availability assumption; original publication and amendment times are unverified. Actual capture times remain in source evidence.") if group == "bha" else None
    return {"name": name, "label": specific[0] if specific else name.replace("_", " ").capitalize(),
            "family": group, "family_label": title, "source": source,
            "description": specific[1] if specific else summary,
            "units": specific[2] if specific else None,
            "documentation": "documented" if specific else "family_only",
            "formula": None,
            "definition_note": None if specific else "Individual formula, units and applicability are not yet documented. Family classification is for navigation.",
            "history_rule": summary if group in {"atr", "rtv", "bha"} else None, "availability_rule": availability}


def load_sets(base):
    """Many named selections and many-to-many model roles; never mutate training config."""
    import config
    folder = Path(base)/"feature_sets"
    path = folder/"registry.json"
    if not path.exists():
        return {"feature_sets": [], "model_roles": []}
    raw = json.loads(path.read_text(encoding="utf-8"))
    if raw.get("version") != 1:
        raise ValueError("Unsupported feature-set registry version")
    sets, ids = [], set()
    for item in raw.get("feature_sets", []):
        identifier = item["id"]
        if not ID.fullmatch(identifier) or identifier in ids:
            raise ValueError("Feature-set IDs must be unique simple names")
        ids.add(identifier)
        file = (folder/item["file"]).resolve()
        if file.parent != folder.resolve() or file.suffix != ".txt":
            raise ValueError("Feature selections must be text files inside feature_sets")
        content = file.read_bytes()
        names = [line.strip() for line in content.decode("utf-8-sig").splitlines()
                 if line.strip() and not line.lstrip().startswith("#")]
        if not names or len(names) != len(set(names)):
            raise ValueError(f"Feature set {identifier} is empty or contains duplicate names")
        from src.model import EXCLUDE_COLUMNS
        if set(names) & set(EXCLUDE_COLUMNS):
            raise ValueError(f"Feature set {identifier} includes excluded labels or identifiers")
        configured = isinstance(config.FEATURE_ALLOWLIST, (str, Path)) and Path(config.FEATURE_ALLOWLIST).resolve() == file
        sets.append({"id": identifier, "label": item.get("label", identifier), "description": item.get("description", ""),
                     "file": file.name, "features": names, "count": len(names),
                     "signature": hashlib.sha256(content).hexdigest()[:16], "configured_for_training": configured})
    roles, role_ids = [], set()
    for item in raw.get("model_roles", []):
        if not ID.fullmatch(item["id"]) or item["id"] in role_ids or not item.get("feature_sets") or set(item["feature_sets"])-ids:
            raise ValueError("Model roles must have unique IDs and reference registered feature sets")
        role_ids.add(item["id"])
        roles.append({"id": item["id"], "label": item.get("label", item["id"]),
                      "scope": item.get("scope", "Unspecified"), "feature_sets": item["feature_sets"]})
    return {"feature_sets": sets, "model_roles": roles}
