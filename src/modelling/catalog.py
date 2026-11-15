"""Model-facing family/kind metadata; explicitly not a column formula catalog."""


def feature_metadata(name):
    kind, family, source = "history", "other", "definition_pending"
    if name.startswith(("horse_prior_market_", "prior_market_", "odds_", "implied_", "norm_implied_", "log_odds", "market_")):
        family, source = "market", "prior accepted prices"
    elif name.startswith("rtv_text_") or name == "rtv_verdict_rank":
        family, source, kind = "rtv", "Timeform comment/verdict model", "learned_prediction"
    elif name.startswith(("rtv_card_", "rtv_pre_")):
        family, source, kind = "rtv", "captured RTV card", "snapshot"
    elif name.startswith("rtv_profile_"):
        family, source, kind = "rtv", "captured RTV horse profile", "snapshot"
    elif name.startswith(("rtv_", "atr_", "bha_")):
        family = name.split("_", 1)[0]
        source = {"rtv": "prior RTV observations", "atr": "prior ATR observations", "bha": "prior BHA reports"}[family]
    elif any(token in name for token in ("elo", "glicko", "trueskill")) or name.startswith("horse_ts_"):
        family, source = "ratings", "prior canonical results"
    elif name.startswith(("trainer_", "jockey_", "jt_", "connections_")):
        family, source = "connections", "prior canonical results and runner connections"
    elif name.startswith("profile_"):
        family, source, kind = "pedigree", "captured Racing TV or Sporting Life horse profiles", "snapshot"
    elif name.startswith(("sire_", "dam_", "damsire_", "pedigree_")):
        family, source = "pedigree", "pedigree and prior family results"
    elif name.startswith(("horse_", "days_since_", "prev_", "form_", "career_")):
        family, source = "horse", "prior canonical results"
    elif name in {"reported_jockey_claim_lbs", "race_conditional_jockeys", "race_apprentice_jockeys", "race_amateur_riders"}:
        family, source, kind = "race", "Sporting Life, Racing TV and verified race conditions", "context"
    elif name.startswith(("race_", "field_", "course_", "track_", "going_", "surface_", "sex_", "region_", "season_")):
        family, source, kind = "race", "race context and derived history", "context_or_history"
    elif name in {"num_runners", "age", "claim_adjusted_weight_lbs", "distance_furlongs", "official_rating", "draw", "handicap"}:
        family, source, kind = "race", "recorded runner/race context", "context"
    return {"name": name, "family": family, "kind": kind, "source_family": source,
            "dependency_status": "family_only", "individual_dependencies": None,
            "note": "Navigation metadata; prefixes do not establish point-in-time availability or information independence."}
