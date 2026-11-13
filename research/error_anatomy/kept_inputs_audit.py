"""Profile of every input the 12 November selection kept: is each one working as intended?

    python -m research.error_anatomy.kept_inputs_audit

Per input, on the current feature store: how often it is filled by year, how
much of it is one repeated value, whether a zero stands in for "no history",
how much it varies inside a race (a value shared by the whole field cannot
separate runners), its outliers, and how the win rate moves across its
within-race quintiles. Flags are prompts to read the feature code, not
verdicts.
"""
import numpy as np
import pandas as pd

from research.error_anatomy.frame import OUT

STORE = "data/processed/feature_store/featured.parquet"
LIST = "feature_sets/win_selected_20261112.txt"


def names():
    return [line.strip() for line in open(LIST, encoding="utf-8") if line.strip() and not line.startswith("#")]


def profile(frame, column):
    x = pd.to_numeric(frame[column], errors="coerce")
    year = frame.race_date.dt.year
    filled = x.notna().groupby(year).mean()
    known = x.dropna()
    top = known.value_counts(normalize=True)
    new = frame.horse_prev_races.eq(0) if "horse_prev_races" in frame else pd.Series(False, index=frame.index)
    race = frame.race_id
    centred = x - x.groupby(race).transform("mean")
    within = float(centred.std() / x.std()) if x.std() > 0 else 0.
    q = known.quantile([.001, .01, .5, .99, .999])
    spread = q[.99] - q[.01]
    # Win rate by within-race rank fifth, on fields of 6+ where every runner has a value.
    whole = x.notna().groupby(race).transform("all") & frame.num_runners.ge(6)
    rank = x[whole].groupby(race[whole]).rank(pct=True, method="average")
    fifth = pd.cut(rank, [0, .2, .4, .6, .8, 1.0], labels=False)
    lift = (frame.won[whole].groupby(fifth).mean() * frame.num_runners[whole].groupby(fifth).mean())
    lift = lift.reindex(range(5))
    steps = np.sign(np.diff(lift.to_numpy()))
    return {"filled": float(x.notna().mean()), "filled_2021": float(filled.get(2021, np.nan)),
            "filled_2023": float(filled.get(2023, np.nan)), "filled_2025": float(filled.get(2025, np.nan)),
            "top_value": float(top.index[0]) if len(top) else np.nan, "top_share": float(top.iloc[0]) if len(top) else np.nan,
            "distinct": int(known.nunique()), "zero_share": float((known == 0).mean()),
            "zero_share_newcomers": float((x[new] == 0).mean()) if new.any() else np.nan,
            "missing_newcomers": float(x[new].isna().mean()) if new.any() else np.nan,
            "within_race_sd_ratio": within,
            "p0.1": float(q[.001]), "p1": float(q[.01]), "median": float(q[.5]), "p99": float(q[.99]), "p99.9": float(q[.999]),
            "tail_ratio": float((q[.999] - q[.001]) / spread) if spread > 0 else np.nan,
            "lift_low": float(lift.iloc[0]), "lift_mid": float(lift.iloc[2]), "lift_high": float(lift.iloc[4]),
            "monotone": bool(len(steps) and (np.all(steps >= 0) or np.all(steps <= 0)))}


def flags(row):
    out = []
    if row.filled < .5:
        out.append("mostly missing")
    if abs(row.filled_2025 - row.filled_2021) > .3:
        out.append("coverage shifts over time")
    if row.top_share > .5 and row.distinct > 2:
        out.append(f"one value is {row.top_share:.0%}")
    if row.zero_share_newcomers > .9 and row.zero_share < .5:
        out.append("zero stands for no history")
    if row.within_race_sd_ratio < .15:
        out.append("almost constant within a race")
    if row.tail_ratio > 6:
        out.append("extreme tails")
    if abs(row.lift_high - row.lift_low) < .1:
        out.append("flat against winning")
    if not row.monotone and abs(row.lift_high - row.lift_low) >= .1:
        out.append("not monotone")
    return "; ".join(out)


def main():
    columns = names()
    frame = pd.read_parquet(STORE, columns=list(dict.fromkeys(["race_id", "race_date", "won", "num_runners",
                                                               "horse_prev_races", *columns])))
    frame = frame.loc[frame.race_date >= "2021-01-01"].reset_index(drop=True)
    table = pd.DataFrame({c: profile(frame, c) for c in columns}).T
    table = table.apply(pd.to_numeric, errors="ignore")
    table["flags"] = [flags(row) for row in table.itertuples()]
    table.to_csv(OUT / "kept_inputs_audit.csv")
    pd.set_option("display.width", 250)
    show = ["filled", "filled_2021", "filled_2025", "top_share", "zero_share", "zero_share_newcomers",
            "within_race_sd_ratio", "tail_ratio", "lift_low", "lift_mid", "lift_high", "flags"]
    print(table[show].round(2).to_string())


if __name__ == "__main__":
    main()
