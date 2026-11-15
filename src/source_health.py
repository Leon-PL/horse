"""Source attempts, publication delays and consumer readiness are separate."""
from __future__ import annotations

from datetime import date, timedelta

POLICIES = {
    "sl": {"required_for": ("features", "training", "serving"), "publication_lag_days": 0},
    "fx": {"required_for": ("features", "training", "serving"), "publication_lag_days": 0},
    "pedigree": {"required_for": (), "publication_lag_days": 0},
    "weather": {"required_for": (), "publication_lag_days": 0},
    "atr": {"required_for": (), "publication_lag_days": 2},
    "rtv": {"required_for": (), "publication_lag_days": 2},
    "betfair": {"required_for": (), "publication_lag_days": 6},
    "bha": {"required_for": (), "publication_lag_days": 1},
    # French/UAE results: history and rating-only races, never training rows.
    "foreign": {"required_for": (), "publication_lag_days": 0},
}


def consumer_readiness(sources, *, purpose="training", requested_end=None):
    if purpose not in {"features", "training", "serving", "evaluation"}:
        raise ValueError("Unknown source consumer")
    blockers, limitations = [], []
    for provider, policy in POLICIES.items():
        value = sources.get(provider, {})
        state = value.get("state", "unverified")
        # A successful check of the available historical window is healthy;
        # recent dates still inside the publication lag are not failed jobs.
        waiting = state in {"awaiting_publication", "waiting"} or (
            state == "partial" and value.get("deferred_recent_days") and not value.get("failed"))
        healthy = state in {"ok", "cached"}
        if purpose in policy["required_for"] and not healthy:
            blockers.append(f"{provider}: {state}")
        elif not healthy:
            limitations.append({"source": provider, "state": "awaiting_publication" if waiting else state,
                                "expected_through": str(date.fromisoformat(requested_end)-timedelta(days=policy["publication_lag_days"])) if requested_end else None,
                                "reason": value.get("reason") or value.get("error")})
    return {"purpose": purpose, "ready": not blockers, "blockers": blockers, "limitations": limitations}
