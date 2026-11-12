"""Fourth pass: where the best development forecast still loses to each exchange snapshot.

    python -m research.error_anatomy.fourth_pass

The best forecast is the native-capacity full 063 with the combined joint
correction (``more-groups-deeper``, section 10). It is 0.0076 behind the 08:00
exchange price. This pass asks, race by race on races with every snapshot:

1. Which kinds of race carry the loss to the 08:00 price and to BSP?
2. How much would an equal-weight mix with each snapshot gain? A mix that
   gains much means the two hold different information.
3. When the 08:00 price and the forecast disagree about a runner, which side
   does the result favour, by runner type?
"""
import numpy as np
import pandas as pd

from research.error_anatomy import joint_variants as jv
from research.error_anatomy.frame import OUT
from research.meta_learner import screening as s
from src.ingestion.storage import atomic_json

BEST = "more-groups-deeper"
# The evening-before price exists for few races, so it is left out to keep the sample.
SNAPSHOTS = ("p_ltp_0800", "p_ltp_1100", "p_ltp_3600s", "p_bsp")


def load():
    scored = pd.read_parquet(OUT / "scored_prices.parquet")
    scored["race_id"] = scored.race_id.astype(str)
    frame = jv.native_frame(scored)
    saved = pd.read_parquet(jv.SAVED, columns=[*s.KEYS, BEST])
    saved["race_id"] = saved.race_id.astype(str)
    frame = frame.merge(saved, on=s.KEYS, how="left", validate="one_to_one")
    bounds = s.bounds_of(frame.race_id.to_numpy())
    frame["p_best"] = np.exp(s.race_log_softmax(np.log(frame.p_native.to_numpy()) + frame[BEST].to_numpy(), bounds))
    gaps = pd.read_parquet(OUT / "history_gaps.parquet")
    gaps["race_id"] = gaps.race_id.astype(str)
    frame = frame.merge(gaps, on=s.KEYS, how="left")
    ok = frame[list(SNAPSHOTS)].gt(0).all(axis=1) & np.isfinite(frame[list(SNAPSHOTS)]).all(axis=1) & frame.bsp_ok
    full = ok.groupby(frame.race_id).transform("all")
    frame = frame.loc[full].reset_index(drop=True)
    for column in SNAPSHOTS:  # renormalise each snapshot within the race
        frame[column] = frame[column] / frame.groupby("race_id")[column].transform("sum")
    return frame


def race_losses(frame):
    winners = frame.loc[frame.won.eq(1)].drop_duplicates("race_id").set_index("race_id")
    out = pd.DataFrame({c: -np.log(winners[c]) for c in ("p_native", "p_best", *SNAPSHOTS)})
    for column in SNAPSHOTS:
        mix = np.sqrt(frame.p_best * frame[column])
        mix = mix / mix.groupby(frame.race_id).transform("sum")
        out[f"mix_{column}"] = -np.log(mix[frame.won.eq(1).to_numpy()].to_numpy()[:len(out)])
    context = ["race_date", "race_type", "handicap", "race_class", "num_runners", "surface", "region", "quarter",
               "race_is_maiden", "race_is_novice", "race_is_bumper", "distance_furlongs", "prize_money"]
    out = out.join(winners[context])
    exposure = frame.groupby("race_id").db_prior.median().rename("median_prior_runs")
    return out.join(exposure)


def by(races, key):
    g = races.groupby(key, observed=True)
    table = pd.DataFrame({"races": g.size(), "best": g.p_best.mean(),
                          "vs_0800": g.p_best.mean() - g.p_ltp_0800.mean(),
                          "vs_1100": g.p_best.mean() - g.p_ltp_1100.mean(),
                          "vs_bsp": g.p_best.mean() - g.p_bsp.mean(),
                          "mix_0800_gain": g.mix_p_ltp_0800.mean() - g.p_ltp_0800.mean(),
                          "mix_bsp_gain": g.mix_p_bsp.mean() - g.p_bsp.mean()})
    table["share_of_0800_loss"] = table.vs_0800 * table.races / (races.p_best - races.p_ltp_0800).sum()
    return table.loc[table.races >= 300]


def disagreement(frame):
    """Among runners where the 08:00 price and the forecast differ most, who is right?"""
    f = frame.copy()
    f["gap"] = np.log(f.p_ltp_0800 / f.p_best)
    f["side"] = pd.cut(f.gap, [-9, -.5, -.2, .2, .5, 9], labels=["market much lower", "market lower", "agree",
                                                                 "market higher", "market much higher"])
    f["kind"] = np.select([f.db_prior.eq(0), f.db_prior.between(1, 4)], ["0 runs", "1-4 runs"], "5+ runs")
    g = f.groupby(["kind", "side"], observed=True)
    table = g.agg(runners=("won", "size"), won=("won", "sum"), model=("p_best", "sum"), market=("p_ltp_0800", "sum"),
                  bsp=("p_bsp", "sum"))
    table["won_vs_model"] = table.won / table.model - 1
    table["won_vs_0800"] = table.won / table.market - 1
    table["bsp_vs_0800"] = table.bsp / table.market - 1
    return table


def main():
    frame = load()
    races = race_losses(frame)
    report = {"races": int(len(races)), "runners": int(len(frame)),
              "log_loss": {c: float(races[c].mean()) for c in races.columns if c.startswith(("p_", "mix_"))}}
    print(f"{len(races)} races, {len(frame)} runners")
    print(pd.Series(report["log_loss"]).round(5).to_string())
    races["field"] = pd.cut(races.num_runners, [0, 6, 9, 12, 16, 99]).astype(str)
    races["exposure"] = pd.cut(races.median_prior_runs, [-1, 1, 4, 9, 19, 999]).astype(str)
    races["kind"] = races.race_type.astype(str) + np.where(races.handicap.astype(bool), " handicap", " non-handicap")
    races["prize"] = pd.qcut(races.prize_money, 4, duplicates="drop").astype(str)
    for key in ("kind", "field", "exposure", "race_class", "surface", "region", "prize", "quarter"):
        table = by(races, key)
        print(f"\n{key}\n{table.round(4).to_string()}")
        report[f"by_{key}"] = {str(k): v for k, v in table.to_dict("index").items()}
    table = disagreement(frame)
    print(f"\n{table.round(3).to_string()}")
    report["disagreement_0800"] = {f"{a} | {b}": v for (a, b), v in table.to_dict("index").items()}
    atomic_json(OUT / "fourth_pass.json", report)


if __name__ == "__main__":
    main()
