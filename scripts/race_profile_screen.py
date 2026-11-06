"""Research screen: can the *shape* of a race be predicted before the off?

Each RTV race is labelled with what it actually turned out to be, then a
race-level model projects those labels from pre-race information, adding
inputs in nested layers. If the layers beyond nominal trip/going/course add
nothing, a "projected race profile" feature collapses into the existing
going x distance x course features and is not worth building.

Labels (realised, race level):
  log_duration   log winning time (the stamina demand in seconds)
  tempo          log winning time minus its course/distance/going/surface par
  finish_shape   median finishing-speed % of the first three home
                 (>100: slow early, sprint finish; <100: attritional)
  winner_early   winner's position percentile after ~25% of the race
                 (0: raced up front, 1: held up last)

Development period only (wf_eval.development_mask); quarterly expanding folds.

Usage::

    python scripts/race_profile_screen.py
"""
from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import lightgbm as lgb

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402
from src.wf_eval import development_mask, expanding_period_folds  # noqa: E402

TABLES = Path(config.DATA_DIR) / "rtv_cache" / "tables"
OUT = Path(config.DATA_DIR) / "research" / "race_profile"

_GOING_BASE = {"hard": 0.0, "firm": 1.0, "good to firm": 2.0, "good": 3.0, "good to yielding": 3.5,
    "good to soft": 4.0, "yielding": 4.0, "yielding to soft": 4.5, "soft": 5.0, "soft to heavy": 5.5,
    "heavy": 6.0, "fast": 2.0, "standard to fast": 2.5, "standard/fast": 2.5, "standard": 3.0,
    "standard to slow": 3.5, "standard/slow": 3.5, "slow": 4.0}
PACE_HINT = {"Uncontested": 0.0, "Very Weak": 1.0, "Weak": 2.0, "Weakly Contested": 2.0, "Even": 3.0,
    "Contested": 3.0, "Strong": 4.0, "Strongly Contested": 4.0, "Very Strong": 5.0, "Extreme": 6.0}
LABELS = ("log_duration", "tempo", "finish_shape", "winner_early")


def going_number(text) -> float:
    """Ordinal going; '(X in places)' moves a quarter step toward X."""
    if text is None or pd.isna(text):
        return np.nan
    t = str(text).strip().casefold().replace(" / ", "/")
    main, _, rest = t.partition("(")
    value = _GOING_BASE.get(main.strip())
    if value is None:
        return np.nan
    places = rest.replace("in places", "").replace(")", "").strip()
    other = _GOING_BASE.get(places)
    return value if other is None else value + 0.25 * np.sign(other - value)


def _read(name, columns):
    parts = []
    for f in sorted(glob.glob(str(TABLES / name / "*.parquet"))):
        have = pq.read_schema(f).names
        parts.append(pd.read_parquet(f, columns=[c for c in columns if c in have]))
    return pd.concat(parts, ignore_index=True)


def load_races() -> pd.DataFrame:
    races = _read("races", ["race_id", "race_date", "track_name", "going", "race_type", "race_class",
        "surface", "is_handicap", "distance_yards", "win_time_s", "card_ip_hints_overall_pace",
        "res_runner_count", "first_prize", "is_abandoned", "is_arab_race", "card_eligibility_age__start"])
    races = races.loc[~races.is_abandoned.fillna(False).astype(bool)
                      & ~races.is_arab_race.fillna(False).astype(bool)
                      & races.race_type.notna()].copy()
    races["going_num"] = races.going.map(going_number)
    races["aw"] = races.surface.isin(["AllWeather", "Polytrack", "Sand", "Tapeta"]).astype(int)
    races["class_num"] = pd.to_numeric(races.race_class, errors="coerce")
    races["pace_hint"] = races.card_ip_hints_overall_pace.map(PACE_HINT)
    races["month"] = pd.to_datetime(races.race_date).dt.month
    races["log_prize"] = np.log1p(pd.to_numeric(races.first_prize, errors="coerce"))
    races["min_age"] = pd.to_numeric(races.card_eligibility_age__start, errors="coerce")
    speed = races.distance_yards / races.win_time_s
    races["win_time_s"] = races.win_time_s.where(speed.between(10, 20))  # yards/sec sanity
    races["log_duration"] = np.log(races.win_time_s)
    return races


def load_runners(races: pd.DataFrame) -> pd.DataFrame:
    """One row per starter with its early position percentile (RTV identities)."""
    runners = _read("runners", ["race_id", "runner_id", "horse_id", "final_position", "is_finisher",
                                "is_nonrunner", "sect_fsp", "beaten_lengths"])
    runners = runners.loc[runners.race_id.isin(races.race_id)
                          & ~runners.is_nonrunner.fillna(False).astype(bool)].copy()
    runners = runners.merge(races[["race_id", "race_date"]], on="race_id")
    runners["pos"] = pd.to_numeric(runners.final_position, errors="coerce")
    runners["fsp"] = pd.to_numeric(runners.sect_fsp, errors="coerce").where(lambda s: s.between(70, 140))

    sect = _read("sectionals", ["race_id", "runner_id", "sector_number", "sector_position"])
    sect = sect.loc[sect.race_id.isin(races.race_id)]
    n_sect = sect.groupby("race_id").sector_number.transform("max") + 1
    # First sector at or beyond a quarter of the race (the opening part-furlong is noisy).
    early = sect.loc[(sect.sector_number + 1) >= np.ceil(0.25 * n_sect)]
    early = early.sort_values("sector_number").groupby(["race_id", "runner_id"], as_index=False).first()
    field = early.groupby("race_id").runner_id.transform("count")
    early["early_pct"] = (early.sector_position - 1) / np.maximum(field - 1, 1)
    return runners.merge(early[["race_id", "runner_id", "early_pct"]], on=["race_id", "runner_id"], how="left")


def race_labels(runners: pd.DataFrame) -> pd.DataFrame:
    finishers = runners.loc[runners.is_finisher.fillna(False).astype(bool)]
    shape = finishers.loc[finishers.pos.le(3)].groupby("race_id").fsp.median().rename("finish_shape")
    winner_early = finishers.loc[finishers.pos.eq(1)].groupby("race_id").early_pct.first().rename("winner_early")
    return pd.concat([shape, winner_early], axis=1)


def field_pace(runners: pd.DataFrame) -> pd.DataFrame:
    """Pre-race field composition from each runner's runs on *earlier dates* only."""
    r = runners[["race_id", "horse_id", "race_date", "early_pct"]].copy()
    r["race_date"] = pd.to_datetime(r.race_date)
    r = r.dropna(subset=["horse_id"]).sort_values(["horse_id", "race_date"])
    # Same-day runs (rare doubles) are pooled first so no row sees its own date.
    daily = r.groupby(["horse_id", "race_date"]).early_pct.agg(["sum", "count"]).reset_index()
    by = daily.groupby("horse_id")
    prior_sum = by["sum"].transform(lambda s: s.rolling(5, min_periods=1).sum().shift(1))
    prior_n = by["count"].transform(lambda s: s.rolling(5, min_periods=1).sum().shift(1))
    daily["mean_early"] = prior_sum / prior_n.where(prior_n > 0)
    r = r.merge(daily[["horse_id", "race_date", "mean_early"]], on=["horse_id", "race_date"], how="left")
    g = r.groupby("race_id").mean_early
    return pd.DataFrame({"field_leaders": g.apply(lambda s: (s < 0.2).sum()),
                         "field_early_min": g.min(), "field_early_mean": g.mean(),
                         "field_known_share": g.apply(lambda s: s.notna().mean())})


LAYERS = {
    "L0 trip": ["distance_yards", "race_type", "aw"],
    "L1 +going,course": ["going_num", "track_name"],
    "L2 +class,field,season": ["class_num", "is_handicap", "res_runner_count", "log_prize", "month", "min_age"],
    "L3 +field run styles": ["field_leaders", "field_early_min", "field_early_mean", "field_known_share"],
    "L4 +Timeform pace hint": ["pace_hint"],
}


def add_tempo(frame: pd.DataFrame, folds) -> pd.Series:
    """Tempo = log time minus an out-of-fold par from trip/going/course only."""
    tempo = pd.Series(np.nan, index=frame.index)
    cols = LAYERS["L0 trip"] + LAYERS["L1 +going,course"]
    for fold in folds:
        tr = fold.train & frame.log_duration.notna().to_numpy()
        model = _fit(frame.loc[tr, cols], frame.log_duration[tr])
        te = fold.test
        tempo[te] = frame.log_duration[te] - model.predict(_cat(frame.loc[te, cols]))
    return tempo


def _cat(x: pd.DataFrame) -> pd.DataFrame:
    x = x.copy()
    for c in ("race_type", "track_name"):
        if c in x:
            x[c] = x[c].astype("category")
    return x


def _fit(x, y):
    model = lgb.LGBMRegressor(n_estimators=400, learning_rate=0.05, num_leaves=31, min_child_samples=40,
                              subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1)
    return model.fit(_cat(x), y)


def screen(frame: pd.DataFrame) -> pd.DataFrame:
    folds = expanding_period_folds(frame.race_date, freq="Q", min_train_periods=2, embargo_days=0)
    frame["tempo"] = add_tempo(frame, folds)
    rows = []
    for label in LABELS:
        cols: list[str] = []
        for layer, extra in LAYERS.items():
            cols = cols + extra
            pred = pd.Series(np.nan, index=frame.index)
            for fold in folds:
                tr = fold.train & frame[label].notna().to_numpy()
                if tr.sum() < 500:
                    continue
                model = _fit(frame.loc[tr, cols], frame.loc[tr, label])
                pred[fold.test] = model.predict(_cat(frame.loc[fold.test, cols]))
            ok = pred.notna() & frame[label].notna()
            y, p = frame.loc[ok, label], pred[ok]
            r2 = 1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum()
            rows.append({"label": label, "layer": layer, "n": int(ok.sum()), "oos_r2": r2,
                         "corr": float(np.corrcoef(y, p)[0, 1])})
            frame[f"proj_{label}_{layer.split()[0]}"] = pred
    return pd.DataFrame(rows)


def main() -> None:
    argparse.ArgumentParser(description=__doc__.splitlines()[0]).parse_args()
    races = load_races()
    runners = load_runners(races)
    races = races.join(race_labels(runners), on="race_id").join(field_pace(runners), on="race_id")
    keep = development_mask(races.race_date) & races.race_date.ge("2023-06-01").to_numpy()
    races = races.loc[keep].sort_values("race_date").reset_index(drop=True)
    print(f"{len(races):,} development races {races.race_date.min()}..{races.race_date.max()}")
    print(races[list(LABELS[:1]) + ["finish_shape", "winner_early", "field_leaders", "pace_hint"]]
          .notna().mean().round(3).to_string())
    result = screen(races)
    OUT.mkdir(parents=True, exist_ok=True)
    races.to_parquet(OUT / "race_profiles.parquet", index=False)
    result.to_csv(OUT / "screen.csv", index=False)
    pivot = result.pivot(index="layer", columns="label", values="oos_r2")[list(LABELS)]
    print("\nOut-of-sample R^2 by nested input layer:\n" + pivot.round(4).to_string())


if __name__ == "__main__":
    main()
