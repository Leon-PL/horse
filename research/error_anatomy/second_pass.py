"""Second pass of the error anatomy, on the corrected forecast.

    python -m research.error_anatomy.second_pass [--forecast soft_stack_soft-long-rich_forecast.parquet]

Repeats the first pass's questions for the three-seed average and for a
corrected stack on the same scored races: how far each is from BSP, which
runners carry the divergence, whether errors by trainer, jockey and sire still
repeat between two halves, and where each sits on the market's timeline.
"""
import argparse

import numpy as np
import pandas as pd

from research.error_anatomy.frame import OUT, race_table
from research.meta_learner import screening as s
from src.ingestion.storage import atomic_json

HALVES = {"2025Q3": "A", "2025Q4": "A", "2026Q1": "B", "2026Q2": "B"}
SNAPSHOTS = ["ltp_eve1800", "ltp_0800", "ltp_1100", "ltp_3600s", "ltp_300s"]


def persistence(frame, column, entity, minimum=80):
    half = frame.quarter.map(HALVES)
    p = frame[column]
    table = pd.DataFrame({"e": frame[entity], "h": half, "o": frame.won, "p": p, "v": p * (1 - p)})
    g = table.groupby(["e", "h"]).agg(n=("o", "size"), o=("o", "sum"), p=("p", "sum"), v=("v", "sum")).unstack("h")
    g = g.dropna()
    g = g[(g[("n", "A")] >= minimum) & (g[("n", "B")] >= minimum)]
    z = {h: (g[("o", h)] - g[("p", h)]) / np.sqrt(g[("v", h)]) for h in "AB"}
    return {"entities": len(g), "correlation": float(np.corrcoef(z["A"], z["B"])[0, 1])}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--forecast", default="soft_stack_soft-long-rich_forecast.parquet")
    args = parser.parse_args()
    frame = pd.read_parquet(OUT / "scored_prices.parquet")
    corrected = pd.read_parquet(OUT / args.forecast, columns=[*s.KEYS, "logp"])
    frame = s.attach(frame, corrected)
    frame["p_corrected"] = np.exp(frame.logp)
    gaps = pd.read_parquet(OUT / "history_gaps.parquet")
    gaps["race_id"] = gaps.race_id.astype(str)
    frame = frame.merge(gaps, on=s.KEYS, how="left")
    models = ["p_average", "p_corrected"]
    races = race_table(frame, [*models, "p_bsp", *[f"p_{c}" for c in SNAPSHOTS]])
    ok = races.bsp_ok.to_numpy()
    report = {"races_with_bsp": int(ok.sum()),
              "loss": {c: float(races.loc[ok, f"loss_{c}"].mean()) for c in [*models, "p_bsp"]}}
    report["gap_to_bsp"] = {c: report["loss"][c] - report["loss"]["p_bsp"] for c in models}
    ladder = np.isfinite(races[[f"loss_p_{c}" for c in SNAPSHOTS]]).all(axis=1).to_numpy() & ok
    report["timeline"] = {"races": int(ladder.sum()),
                          **{c: float(races.loc[ladder, f"loss_{c}"].mean())
                             for c in [*models, *[f"p_{x}" for x in SNAPSHOTS], "p_bsp"]}}
    eleven = np.isfinite(races.loss_p_ltp_1100).to_numpy() & ok
    report["against_1100"] = {"races": int(eleven.sum()),
                              **{c: float(races.loc[eleven, f"loss_{c}"].mean()) for c in [*models, "p_ltp_1100", "p_bsp"]}}
    priced = frame.loc[frame.bsp_ok].copy()
    kinds = {"first run in database": priced.db_prior.eq(0), "one earlier run": priced.db_prior.eq(1),
             "off more than 120 days": priced.db_days.gt(120), "card shows missing runs": priced.miss.eq(True),
             "five or more earlier runs": priced.db_prior.ge(5), "all": pd.Series(True, index=priced.index)}
    report["divergence_per_1000_runners"] = {}
    for column in models:
        kl = priced.p_bsp * np.log(priced.p_bsp / priced[column])
        report["divergence_per_1000_runners"][column] = {k: float(1000 * kl[m].mean()) for k, m in kinds.items()}
    report["persistence"] = {entity: {c: persistence(frame, c, entity) for c in [*models, "p_bsp"]}
                             for entity in ("trainer", "jockey")}
    atomic_json(OUT / "second_pass.json", report)
    print("loss", {k: round(v, 5) for k, v in report["loss"].items()}, "gap", {k: round(v, 5) for k, v in report["gap_to_bsp"].items()})
    print("timeline", {k: round(v, 4) if isinstance(v, float) else v for k, v in report["timeline"].items()})
    print("11:00", {k: round(v, 4) if isinstance(v, float) else v for k, v in report["against_1100"].items()})
    print(pd.DataFrame(report["divergence_per_1000_runners"]).round(2))
    print(pd.DataFrame({e: {c: round(v["correlation"], 3) for c, v in d.items()} for e, d in report["persistence"].items()}))


if __name__ == "__main__":
    main()
