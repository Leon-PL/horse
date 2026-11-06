"""Audit all selected independent inputs against temporal and price changes."""
from __future__ import annotations

import json
import argparse
import logging
from pathlib import Path
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd

from scripts.audit_atr_invariance import differences
from src.data_processor import process_data
from src.feature_engineer import engineer_features
from src.featured_dataset import fe_signature
from tests.test_live_parity import _synthetic_raw

OUTPUT = ROOT / "data/audit/model_improvement/014_margin_repair"


def fixture(days=45):
    raw = _synthetic_raw(n_days=days, seed=71)
    day = pd.to_datetime(raw.race_date).sub(pd.Timestamp("2025-01-01")).dt.days
    # A horse must not race twice on one day; connections can and do.
    horse = (raw.groupby("race_date", sort=False).cumcount() + day * 7) % 40
    raw["horse_id"], raw["horse_name"] = 100000 + horse, "Horse " + horse.astype(str)
    slot = raw.groupby("race_id", sort=False).cumcount()
    # Unique jockeys per race, but two stablemates for each of three trainers.
    raw["jockey"], raw["trainer"] = "J" + slot.astype(str), "T" + (slot % 3).astype(str)
    for col, modulus in [("sire", 7), ("dam", 17), ("damsire", 9)]:
        raw[col] = col + (horse % modulus).astype(str)
    raw["lengths_behind"] = np.where(raw.finish_position.eq(1), 0., .25 + horse % 5)
    # Exercise legacy persisted non-finishers with NO finish_pos_label column.
    dnf = raw.finish_position.eq(6) & day.mod(3).eq(0)
    raw.loc[dnf, "finish_position"] = 0
    raw.loc[dnf, "lengths_behind"] = np.nan
    return raw


def derive(raw):
    from src.attheraces_scraper import ALL_ATR_COLS
    def empty_merge(f):
        return f.assign(**{c: np.nan for c in ALL_ATR_COLS if c not in f})
    with patch("src.attheraces_scraper.merge_atr_sectionals", side_effect=empty_merge):
        out = engineer_features(process_data(raw.copy()), save=False)
    return out.set_index(["race_id", "horse_id"])


def swap_results(frame, mask):
    frame = frame.copy()
    last = frame.groupby("race_id", sort=False).finish_position.transform("max")
    positive = mask & frame.finish_position.gt(0)
    frame.loc[positive, "finish_position"] = last.loc[positive] + 1 - frame.loc[positive, "finish_position"]
    frame.loc[mask, "won"] = frame.loc[mask, "finish_position"].eq(1).astype(int)
    frame.loc[positive, "lengths_behind"] = np.where(frame.loc[positive, "finish_position"].eq(1), 0., 9.)
    return frame


def audit(output=OUTPUT, include_nonfinishers=False, features=ROOT / "feature_sets/win.txt",
          allow_unexercised=False):
    selected = [c.strip() for c in Path(features).read_text().splitlines() if c.strip() and not c.startswith("#")]
    if include_nonfinishers:
        selected += ["horse_non_finish_rate_shrunk", "horse_last_run_non_finish"]
    raw = fixture()
    original = derive(raw)
    missing = sorted(set(selected) - set(original))
    if missing and not allow_unexercised:
        raise ValueError(f"Fixture does not exercise selected inputs: {missing}")
    selected = [c for c in selected if c in original]
    # Inputs from external caches (ATR, RTV) are empty in the fixture, so they
    # pass vacuously; name them so the report is not read as covering them.
    vacuous = sorted(c for c in selected if original[c].nunique(dropna=True) < 2)
    cutoff = "2025-02-01"
    prefix = derive(raw.loc[raw.race_date.lt(cutoff)])
    changed = raw.copy()
    future = changed.race_date.ge(cutoff)
    changed = swap_results(changed, future)
    day = "2025-01-25"
    today = raw.race_date.eq(day)
    keys = pd.MultiIndex.from_frame(raw.loc[today, ["race_id", "horse_id"]])
    cards = raw.copy()
    cards.loc[today, ["finish_position", "won"]] = 0
    cards.loc[today, "lengths_behind"] = np.nan
    swapped = raw.copy()
    swapped = swap_results(swapped, today)
    prices = raw.copy()
    prices["odds"] = np.linspace(1.02, 300., len(prices))
    checks = {
        "prefix_vs_full": differences(prefix, original, selected),
        "future_results": differences(original.loc[prefix.index], derive(changed), selected),
        "future_append": differences(original, derive(fixture(52)), selected),
        "pending_current_day": differences(original.loc[keys], derive(cards), selected),
        "current_results_swapped": differences(original.loc[keys], derive(swapped), selected),
        "input_permutation": differences(original, derive(raw.sample(frac=1, random_state=51)), selected),
        "prices_changed": differences(original, derive(prices), selected),
    }
    # Controls prevent an audit that accidentally compares stale/cached output.
    sensitivity = differences(original, derive(changed), selected)
    report = {"passed": not any(checks.values()) and bool(sensitivity), "fe_signature": fe_signature(), "selected_inputs": selected,
              "rows": len(raw), "checks": checks, "prior_outcome_sensitivity": sorted(sensitivity),
              "not_in_fixture": missing, "constant_in_fixture": vacuous}
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "manifest.json").exists():
        raise ValueError("Do not overwrite a registered experiment's input audit")
    (output / "input_invariance.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)
    return report


if __name__ == "__main__":
    logging.disable(logging.INFO)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=str(OUTPUT))
    parser.add_argument("--nonfinisher-features", action="store_true")
    parser.add_argument("--features", default=str(ROOT / "feature_sets/win.txt"), help="feature list to audit")
    parser.add_argument("--allow-unexercised", action="store_true",
                        help="report, instead of refusing, inputs the synthetic fixture cannot build")
    args = parser.parse_args()
    audit(args.output_dir, args.nonfinisher_features, args.features, args.allow_unexercised)
