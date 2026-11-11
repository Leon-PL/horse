"""Third pass: what the corrected forecast still gets wrong on exposed horses.

    python -m research.error_anatomy.third_pass [--forecast FILE --tag NAME]

Three questions, on the scored quarters:

1. How much of each runner type's divergence from BSP is already in the 11:00
   exchange price? That part is public by late morning; the rest is late money.
2. Does the corrected forecast's disagreement with the market still repeat for
   the same horse from one run to the next?
3. Is the disagreement signed by circumstance (last-run result, class or trip
   change, time off)?
"""
import argparse

import numpy as np
import pandas as pd

from research.error_anatomy.frame import OUT
from research.meta_learner import screening as s
from src.ingestion.storage import atomic_json

FORECAST = "soft_stack_soft-long-rich_forecast.parquet"


def centred(values, race_id):
    values = pd.Series(np.asarray(values, float))
    return (values - values.groupby(np.asarray(race_id)).transform("mean")).to_numpy()


def load(forecast=FORECAST):
    frame = pd.read_parquet(OUT / "scored_prices.parquet")
    corrected = pd.read_parquet(OUT / forecast, columns=[*s.KEYS, "logp"])
    frame = s.attach(frame, corrected)
    frame["p_corrected"] = np.exp(frame.logp)
    gaps = pd.read_parquet(OUT / "history_gaps.parquet")
    gaps["race_id"] = gaps.race_id.astype(str)
    frame = frame.merge(gaps, on=s.KEYS, how="left")
    kind = np.select([frame.db_prior.eq(0), frame.db_prior.eq(1), frame.db_prior.between(2, 4),
                      frame.db_prior.between(5, 14)], ["0 runs", "1 run", "2-4 runs", "5-14 runs"], "15+ runs")
    frame["kind"] = kind
    # One runner has an infinite BSP; treat its race as unpriced.
    bad = frame.loc[~(frame.p_bsp > 0) & frame.bsp_ok, "race_id"].unique()
    frame.loc[frame.race_id.isin(bad), "bsp_ok"] = False
    return frame


def timeline(frame):
    """Divergence per 1,000 runners, on races with a full-field 11:00 price and BSP."""
    f = frame.loc[frame.bsp_ok & frame.p_ltp_1100.notna()].copy()
    def kl(a, b):
        return 1000 * f[a] * np.log(f[a] / f[b])
    table = pd.DataFrame({"kind": f.kind, "runners": 1.,
                          "bsp_vs_average": kl("p_bsp", "p_average"), "bsp_vs_corrected": kl("p_bsp", "p_corrected"),
                          "bsp_vs_1100": kl("p_bsp", "p_ltp_1100"), "1100_vs_corrected": kl("p_ltp_1100", "p_corrected")})
    out = table.groupby("kind").agg({"runners": "sum", **{c: "mean" for c in table.columns[2:]}})
    out.loc["all"] = [len(table), *table.iloc[:, 2:].mean()]
    return out


def repeat(frame):
    """Correlation of a horse's race-centred market gap with its previous run's, before and after correction."""
    f = frame.loc[frame.bsp_ok].sort_values(["horse_id", "race_date"]).copy()
    rows = {}
    for column in ("p_average", "p_corrected"):
        gap = centred(np.log(f.p_bsp / f[column]), f.race_id)
        f["gap"] = gap
        f["previous"] = f.groupby("horse_id").gap.shift(1)
        ok = f.previous.notna()
        rows[column] = {"pairs": int(ok.sum()), "correlation": float(np.corrcoef(f.gap[ok], f.previous[ok])[0, 1]),
                        "sd": float(np.std(gap))}
        for kind, part in f.loc[ok].groupby("kind"):
            rows[column][kind] = float(np.corrcoef(part.gap, part.previous)[0, 1])
    return rows


def circumstances(frame):
    """Mean race-centred market gap, and wins against forecast, by circumstance among horses with 5+ runs."""
    f = frame.loc[frame.bsp_ok].copy()
    f["gap"] = centred(np.log(f.p_bsp / f.p_corrected), f.race_id)
    f["gap_1100"] = centred(np.log(f.p_ltp_1100 / f.p_corrected), f.race_id)
    f["rank"] = f.groupby("race_id").p_corrected.rank(ascending=False)
    f["days"] = pd.cut(f.db_days, [-1, 7, 14, 28, 60, 120, 365, 1e5]).astype(str)
    f["share"] = pd.cut(f.p_corrected, [0, .03, .08, .15, .3, 1]).astype(str)
    exposed = f.loc[f.db_prior.ge(5)]
    out = {}
    for name in ("days", "share", "race_type", "handicap", "headgear", "age"):
        g = exposed.groupby(name).agg(runners=("gap", "size"), gap=("gap", "mean"), gap_1100=("gap_1100", "mean"),
                                      abs_gap=("gap", lambda v: float(np.abs(v).mean())),
                                      won=("won", "sum"), model=("p_corrected", "sum"), market=("p_bsp", "sum"))
        g = g.loc[g.runners >= 800]
        g["won_vs_model"] = g.won / g.model - 1
        g["won_vs_market"] = g.won / g.market - 1
        out[name] = g.drop(columns=["won", "model", "market"])
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--forecast", default=FORECAST)
    parser.add_argument("--tag", default="")
    args = parser.parse_args()
    frame = load(args.forecast)
    report = {}
    table = timeline(frame)
    print(table.round(2).to_string())
    report["timeline_divergence_per_1000"] = table.to_dict("index")
    report["repeat"] = repeat(frame)
    print(pd.DataFrame(report["repeat"]).round(3).to_string())
    for name, table in circumstances(frame).items():
        print(f"\n{name}\n{table.round(3).to_string()}")
        report[f"by_{name}"] = {str(k): v for k, v in table.to_dict("index").items()}
    atomic_json(OUT / f"third_pass{args.tag}.json", report)


if __name__ == "__main__":
    main()
