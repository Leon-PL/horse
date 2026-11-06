"""H1: does the sibling lead (ledger 2h) hold on the golden hold-out?

Pre-registered in docs/archive/SESSION_LOG.md before this script existed: among
hold-out runners with <= 1 prior UK/IRE start and sibling history, in races
with a complete BSP market, the partial correlation of
``sibling_best_or_vs_field`` with (won - p_BSP) - within-race demeaned, price
basis on p_BSP partialled out - is negative. Pass: one-sided p < 0.01 from a
race-clustered bootstrap standard error.

The discovery ran against the T-5 price on data before 2026-06-11; the
hold-out has no T-5 price, and BSP is the price a bettor can actually settle
at, so this is the stricter and the economically relevant test.

Run once. Do not re-run with a changed population or statistic.
"""

from __future__ import annotations

import json
import os
import sqlite3
import sys
from math import erfc, sqrt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from scripts.microstructure_screen import partial_corr_clustered
from src.betfair import attach_bsp
from src.sibling_features import compute_sibling_features
from src.wf_eval import holdout_mask

PUBLIC = os.path.join("data", "processed", "betfair_sp_public.parquet")


def main() -> int:
    con = sqlite3.connect("file:data/races.db?mode=ro", uri=True)
    df = pd.read_sql("""
        SELECT r.race_id, r.race_date, r.track, r.off_time, r.horse_name, r.horse_id,
               r.won, r.official_rating, p.dam
        FROM results r LEFT JOIN horse_pedigree p ON p.horse_id = r.horse_id
    """, con)
    df["race_date"] = pd.to_datetime(df["race_date"])
    df = df.sort_values(["race_date", "race_id", "horse_name"], kind="stable").reset_index(drop=True)
    # Sibling form needs the whole history; the population is the hold-out.
    sib = compute_sibling_features(df)
    df = pd.concat([df, sib], axis=1)
    df["prior_starts"] = df.groupby("horse_id").cumcount()

    hold = df[holdout_mask(df["race_date"])].copy()
    hold["race_date"] = hold["race_date"].dt.strftime("%Y-%m-%d")
    hold = attach_bsp(hold, pd.read_parquet(PUBLIC), cols=("bsp",))
    hold["bsp"] = pd.to_numeric(hold["bsp"], errors="coerce")
    g = hold.groupby("race_id")
    complete = ((hold["bsp"] > 1).groupby(hold["race_id"]).transform("all")
                & (g["won"].transform("sum") == 1) & (g["won"].transform("size") >= 2))
    hold = hold[complete].copy()
    inv = 1.0 / hold["bsp"]
    hold["p_bsp"] = inv / inv.groupby(hold["race_id"]).transform("sum")
    hold["resid_close"] = hold["won"] - hold["p_bsp"]

    pop = hold[(hold["prior_starts"] <= 1) & (hold["has_sibling_history"] == 1)
               & hold["sibling_best_or_vs_field"].notna()]
    # Within-race demeaning needs >= 2 eligible runners in a race.
    pop = pop[pop.groupby("race_id")["race_id"].transform("size") >= 2]
    res = partial_corr_clustered(pop, "sibling_best_or_vs_field", "resid_close", "p_bsp",
                                 n_boot=2000)
    one_sided_p = erfc(abs(res["t"]) / sqrt(2)) / 2 if res["t"] < 0 else 1 - erfc(abs(res["t"]) / sqrt(2)) / 2
    res.update({"one_sided_p_negative": one_sided_p, "passes": bool(one_sided_p < 0.01),
                "hold_out_races": int(hold["race_id"].nunique()),
                "population_runners": int(len(pop)), "population_races": int(pop["race_id"].nunique())})
    print(json.dumps(res, indent=2, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
