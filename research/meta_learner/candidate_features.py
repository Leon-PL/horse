"""Candidate inputs computed for screening, not yet in the feature store.

Both use only information available before the forecast: relatives' form uses
races on earlier dates, and same-day bias uses races on the same card that
started at least ``lag_minutes`` before this one.
"""
import numpy as np
import pandas as pd

KEYS = ["race_id", "horse_id"]


def _name(values):
    return values.fillna("").astype(str).str.lower().str.replace(r"[^a-z0-9]", "", regex=True)


def _finish_pct(frame):
    finished = frame.finish_position.gt(0)
    n = frame.num_runners.clip(lower=2)
    return np.where(finished, (frame.finish_position - 1) / (n - 1), 1.0), finished


def _before(table, keys, sums):
    """Totals over strictly earlier dates, indexed by (*keys, race_date)."""
    by_day = table.groupby([*keys, "race_date"], sort=True)[sums].sum()
    return by_day.groupby(level=list(range(len(keys)))).cumsum() - by_day


def _earlier_count(groups, dates, query_groups, query_dates):
    """Per query, how many (group, date) entries have the same group and an earlier date."""
    scale = np.int64(1_000_000)

    def days(d):
        return pd.to_datetime(d).to_numpy().astype("datetime64[D]").astype(np.int64)
    stored = np.sort(groups.astype(np.int64) * scale + days(dates))
    query = query_groups.astype(np.int64) * scale
    return np.searchsorted(stored, query + days(query_dates), "left") - np.searchsorted(stored, query, "left")


def relatives_form(history):
    """Form of other offspring of the same dam (keyed by dam and dam-sire) before today."""
    h = history[["race_id", "horse_id", "race_date", "dam", "damsire", "won", "finish_position",
                 "num_runners", "official_rating"]].copy()
    dam = _name(h.dam)
    key = (dam + "|" + _name(h.damsire)).where(dam.ne(""))
    # One family per horse: its most frequent recorded dam/dam-sire pair.
    pairs = pd.DataFrame({"horse_id": h.horse_id, "family": key}).dropna()
    counts = pairs.value_counts().rename("n").reset_index()
    modal = counts.sort_values(["horse_id", "n", "family"], ascending=[True, False, True]).drop_duplicates("horse_id")
    h["family"] = h.horse_id.map(modal.set_index("horse_id").family)
    h = h.dropna(subset=["family"]).copy()
    fpct, finished = _finish_pct(h)
    h["runs"] = 1.
    h["wins"] = h.won.astype(float)
    h["places"] = (finished & h.finish_position.le(3)).astype(float)
    h["fpct"] = fpct
    rated = h.official_rating.gt(0)
    h["or_sum"] = h.official_rating.where(rated, 0.).astype(float)
    h["or_n"] = rated.astype(float)
    sums = ["runs", "wins", "places", "fpct", "or_sum", "or_n"]
    family = _before(h, ["family"], sums)
    own = _before(h, ["family", "horse_id"], sums)
    rows = h[["race_id", "horse_id", "race_date", "family"]].reset_index(drop=True)
    sib = (family.reindex(pd.MultiIndex.from_frame(rows[["family", "race_date"]])).to_numpy()
           - own.reindex(pd.MultiIndex.from_frame(rows[["family", "horse_id", "race_date"]])).to_numpy())
    sib = pd.DataFrame(sib, columns=sums)

    code, _ = pd.factorize(rows.family)
    code_of = dict(zip(rows.family, code))
    first = h.groupby(["family", "horse_id"]).race_date.min().reset_index()
    first_win = h.loc[h.won.eq(1)].groupby(["family", "horse_id"]).race_date.min().reset_index()
    horses = _earlier_count(first.family.map(code_of).to_numpy(), first.race_date, code, rows.race_date)
    winners = _earlier_count(first_win.family.map(code_of).to_numpy(), first_win.race_date, code, rows.race_date)
    own = rows[["family", "horse_id"]].merge(first.rename(columns={"race_date": "own_first"}), how="left",
                                             on=["family", "horse_id"], validate="many_to_one")
    own = own.merge(first_win.rename(columns={"race_date": "own_win"}), how="left",
                    on=["family", "horse_id"], validate="many_to_one")
    horses = horses - (own.own_first < rows.race_date).to_numpy()
    winners = winners - (own.own_win < rows.race_date).to_numpy()  # NaT compares False

    runs = sib.runs.to_numpy()
    out = rows[KEYS].copy()
    out["rel_runs"] = runs
    out["rel_horses"] = horses
    out["rel_winning_horses"] = winners
    out["rel_win_rate"] = (sib.wins + 1.) / (runs + 10.)
    out["rel_place_rate"] = (sib.places + 3.) / (runs + 10.)
    out["rel_finish_pct"] = np.where(runs > 0, sib.fpct / np.maximum(runs, 1), np.nan)
    out["rel_mean_or"] = np.where(sib.or_n > 0, sib.or_sum / np.maximum(sib.or_n, 1), np.nan)
    return out


def _off_minutes(values):
    parts = values.astype(str).str.extract(r"^(\d{1,2}):(\d{2})")
    return pd.to_numeric(parts[0], errors="coerce") * 60 + pd.to_numeric(parts[1], errors="coerce")


def same_day_bias(history, lag_minutes=30):
    """Draw and running-style bias shown by earlier races on the same card today."""
    h = history[["race_id", "horse_id", "race_date", "track", "off_time", "race_type", "num_runners",
                 "finish_position", "draw", "pace_style_index"]].copy()
    h["fpct"], _ = _finish_pct(h)
    by_race = h.groupby("race_id")
    h["draw_rank"] = by_race.draw.rank(pct=True).where(h.draw.gt(0))
    h["pace_rel"] = (h.pace_style_index - by_race.pace_style_index.transform("mean")).fillna(0.)

    races = by_race.agg(race_date=("race_date", "first"), track=("track", "first"), off_time=("off_time", "first"),
                        race_type=("race_type", "first"), runners=("horse_id", "size"),
                        drawn=("draw_rank", "count"), styled=("pace_style_index", "count"))
    # Draw: mean finishing percentile of the high half minus the low half (positive: low draws did better).
    half = h.assign(low=h.draw_rank.le(.5)).groupby(["race_id", "low"]).fpct.mean().unstack()
    races["draw_adv"] = (half.get(False) - half.get(True)).where(
        races.race_type.eq("Flat") & races.drawn.eq(races.runners) & races.runners.ge(6))
    # Style: within-race correlation of style with finishing quality (positive: forward runners did better).
    quality = -h.fpct
    centred_q = quality - quality.groupby(h.race_id).transform("mean")
    cov = (h.pace_rel * centred_q).groupby(h.race_id).sum()
    spread = np.sqrt((h.pace_rel ** 2).groupby(h.race_id).sum() * (centred_q ** 2).groupby(h.race_id).sum())
    races["pace_adv"] = (cov / spread.replace(0, np.nan)).where(races.styled.ge(4))
    races["off"] = _off_minutes(races.off_time)
    races = races.dropna(subset=["off"]).sort_values(["track", "race_date", "off"])
    out = []
    for _, card in races.groupby(["track", "race_date"], sort=False):
        off = card.off.to_numpy()
        draw, pace = card.draw_adv.to_numpy(), card.pace_adv.to_numpy()
        n_draw = np.r_[0, np.cumsum(np.isfinite(draw))]
        s_draw = np.r_[0, np.cumsum(np.nan_to_num(draw))]
        n_pace = np.r_[0, np.cumsum(np.isfinite(pace))]
        s_pace = np.r_[0, np.cumsum(np.nan_to_num(pace))]
        done = np.searchsorted(off, off - lag_minutes, side="right")  # races off at least lag earlier
        out.append(pd.DataFrame({"race_id": card.index, "day_races_before": done,
                                 "day_draw_adv": s_draw[done] / (n_draw[done] + 2.),
                                 "day_pace_adv": s_pace[done] / (n_pace[done] + 2.)}))
    day = pd.concat(out, ignore_index=True)
    result = h[KEYS + ["draw_rank", "pace_rel"]].merge(day, on="race_id", how="left")
    result["day_draw_x_low"] = result.day_draw_adv * (.5 - result.draw_rank)
    result["day_pace_x_style"] = result.day_pace_adv * result.pace_rel
    return result.drop(columns=["draw_rank", "pace_rel"])
