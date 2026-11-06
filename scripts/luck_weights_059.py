"""Train on luck-adjusted results (experiment 059).

``build`` gives every development runner a trouble severity from its own race
only: the Racing TV close-up comment and the BHA stewards' events for that
fixture (``src/luck_weights.py``). ``dev`` screens the weighting schemes on the
selection quarters 2024Q1-2025Q2 (screen capacity) against the unweighted
production arm and registers the one with the lowest pooled race log loss.
``confirm`` fits that scheme at native capacity on 2025Q3-2026Q2. Weights act
on training rows only; every arm is scored, unweighted, on the actual results.

Sectional times, the brief's third source, are not used: the ATR cache covers
only 2023-08 to 2026-06 and its derived run figures were found mis-built
(2026-10-29 audit), while the close-ups already describe the same incidents.

    python -m scripts.luck_weights_059 build
    python -m scripts.luck_weights_059 dev
    python -m scripts.luck_weights_059 confirm
    python -m scripts.luck_weights_059 report
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.model_ideas import (OUT, Cache, baseline, compare, predict_production, print_summary,  # noqa: E402
                                 production_inputs, run_arm, save_json, setup_logging, store_columns)
from src import feature_selection as fs  # noqa: E402
from src.luck_weights import SCHEMES, comment_severity, luck_weights, race_luck, stewards_severity  # noqa: E402
from src.source_pipeline import load_matched_text  # noqa: E402
from src.utils import canonical_track_key, normalise_horse_key  # noqa: E402

SEVERITY = OUT / "luck_severity_059.parquet"
REGISTRATION = OUT / "registration_059.json"
STEWARDS = ROOT / "data/processed/stewards_events.parquet"

logger = logging.getLogger("luck_weights_059")


def build(_args) -> None:
    store = store_columns(["race_id", "horse_id", "horse_name", "race_date", "track", "won"])
    store["race_id"] = store.race_id.astype(str)
    store["horse_id"] = pd.to_numeric(store.horse_id, errors="coerce")
    dates = pd.to_datetime(store.race_date)
    chunks = []
    for year in range(dates.min().year, dates.max().year + 1):
        lo = max(dates.min(), pd.Timestamp(f"{year}-01-01")).strftime("%Y-%m-%d")
        hi = min(dates.max(), pd.Timestamp(f"{year}-12-31")).strftime("%Y-%m-%d")
        text = load_matched_text(lo, hi)[["race_id", "horse_id", "close_up"]]
        logger.info("%s: %d close-ups", year, text.close_up.notna().sum())
        chunks.append(text)
    text = pd.concat(chunks, ignore_index=True)
    text["race_id"], text["horse_id"] = text.race_id.astype(str), pd.to_numeric(text.horse_id, errors="coerce")
    text = text.dropna(subset=["horse_id"]).drop_duplicates(["race_id", "horse_id"])
    frame = store.merge(text, on=["race_id", "horse_id"], how="left", validate="many_to_one")

    events = pd.read_parquet(STEWARDS)
    events = events[events.source.eq("bha")]
    events = events.assign(date=pd.to_datetime(events.date).dt.strftime("%Y-%m-%d"),
                           track=events.track_key.map(canonical_track_key),
                           horse=events.horse_key.map(normalise_horse_key))
    tags = events.groupby(["date", "track", "horse"]).tag.agg(list).rename("stewards_tags").reset_index()
    frame["date"] = dates.dt.strftime("%Y-%m-%d").to_numpy()
    frame["track_key"] = frame.track.map(canonical_track_key)
    frame["horse"] = frame.horse_name.map(normalise_horse_key)
    frame = frame.merge(tags, left_on=["date", "track_key", "horse"], right_on=["date", "track", "horse"],
                        how="left", suffixes=("", "_events"), validate="many_to_one")
    if len(frame) != len(store):
        raise ValueError("The joins changed the row count")

    out = pd.DataFrame({
        "has_close_up": frame.close_up.notna().to_numpy(),
        "close_up_severity": comment_severity(frame.close_up),
        "stewards_severity": frame.stewards_tags.map(lambda t: stewards_severity(t) if isinstance(t, list) else 0.0)
                                  .to_numpy(np.float64),
        "has_stewards_event": frame.stewards_tags.notna().to_numpy(),
    })
    out["severity"] = np.maximum(out.close_up_severity, out.stewards_severity)
    OUT.mkdir(parents=True, exist_ok=True)
    out.to_parquet(SEVERITY, index=False)
    cache = Cache()
    luck = race_luck(out.severity, cache.targets.market, cache.targets.won, fs.race_groups(cache.targets.race_ids))
    summary = {
        "runners": int(len(out)), "close_up_coverage": float(out.has_close_up.mean()),
        "stewards_event_runners": int(out.has_stewards_event.sum()),
        "severity_share": {str(k): float(v) for k, v in out.severity.value_counts(normalize=True).items()},
        "from_close_up_only": float(((out.close_up_severity > 0) & (out.stewards_severity == 0)).mean()),
        "from_stewards_only": float(((out.stewards_severity > 0) & (out.close_up_severity == 0)).mean()),
        "races": int(len(luck)), "race_luck_quantiles": {q: float(np.quantile(luck, q)) for q in (0.5, 0.9, 0.99)},
        "races_luck_over_0.1": float((luck > 0.1).mean()), "races_luck_over_0.25": float((luck > 0.25).mean()),
    }
    save_json(OUT / "luck_build_059.json", summary)
    print(json.dumps(summary, indent=2))


def make_predict(cache: Cache, names: list[str], scheme: str):
    t = cache.targets
    severity = pd.read_parquet(SEVERITY).severity.to_numpy(np.float64)
    multiplier = luck_weights(scheme, severity, t.market, t.won, fs.race_groups(t.race_ids))

    def predict(train, test, capacity):
        return predict_production(cache, names, train, test, capacity=capacity, weight_multiplier=multiplier)
    return predict


def dev(_args) -> None:
    cache = Cache()
    names = production_inputs(cache)
    baseline(cache, "dev")
    for scheme in SCHEMES:
        run_arm(cache, "dev", f"luck_{scheme}", make_predict(cache, names, scheme))
    arms = ["production", *(f"luck_{s}" for s in SCHEMES)]
    summary = compare("dev", arms, [(a, "production") for a in arms[1:]])
    print_summary(summary, "059 development screen")
    chosen = min(SCHEMES, key=lambda s: summary["arms"][f"luck_{s}"]["race_ll"])
    if REGISTRATION.exists():
        registered = json.loads(REGISTRATION.read_text(encoding="utf-8"))
        if registered["scheme"] != chosen:
            raise SystemExit(f"Registration already fixed {registered['scheme']}; refusing to change it to {chosen}")
        return
    save_json(REGISTRATION, {"experiment": "059", "rule": "lowest pooled selection-quarter race LL, screen capacity",
                             "schemes": {k: list(v) for k, v in SCHEMES.items()}, "scheme": chosen,
                             "registered_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "dev": summary})
    print(f"\nRegistered scheme {chosen} -> {REGISTRATION}")


def confirm(args) -> None:
    if not REGISTRATION.exists():
        raise SystemExit("Run the development screen first: it fixes the weighting scheme.")
    scheme = json.loads(REGISTRATION.read_text(encoding="utf-8"))["scheme"]
    cache = Cache()
    names = production_inputs(cache)
    baseline(cache, "confirm")
    run_arm(cache, "confirm", "luck", make_predict(cache, names, scheme))
    report(args)


def report(_args) -> dict:
    summary = compare("confirm", ["production", "luck"], [("luck", "production")])
    summary["scheme"] = json.loads(REGISTRATION.read_text(encoding="utf-8"))["scheme"]
    save_json(OUT / "summary_059.json", summary)
    print_summary(summary, f"059 confirmation, luck scheme {summary['scheme']}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", choices=["build", "dev", "confirm", "report"])
    args = parser.parse_args()
    setup_logging()
    {"build": build, "dev": dev, "confirm": confirm, "report": report}[args.stage](args)


if __name__ == "__main__":
    main()
