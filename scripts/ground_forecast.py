"""EXPLORATION_PLAN C: do early prices under-react to forecast ground changes?

Hypothesis: when the forecast says rain will fall between an early decision
time and the off, early prices under-react for horses with a strong ground
preference; the market catches up once the official going changes. A timing
edge of the same shape as ledger 14's.

Pieces, each restricted to what was known at the decision time:

* **forecast rain** at the course between the decision and the off, summed
  from Open-Meteo's ``precipitation_previous_day1`` (each hour's value as
  forecast 24 h earlier, so every forecast used was issued before an
  evening-before or morning decision; ``fetch_forecast_previous_runs.py``);
* **ground preference** per horse from its *previous* runs only: how it
  finished relative to its SP rank on soft ground (Good to Soft or softer)
  minus the same on firm ground (Good to Firm or firmer), each shrunk toward
  zero. Not the store's ``horse_going_*`` columns: those are keyed on the
  race's final official going, which is look-ahead at the decision time;
* **decision prices** at 18:00 UK the evening before and 08:00 UK race day
  (``betfair_timing.parquet``), with later non-runner reductions applied.

Turf only. Development period only; the forecasts start in early 2024.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sqlite3
import sys
from math import erfc, sqrt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from src.betfair import attach_bsp
from src.bet_settlement import apply_reduction_factor
from src.wf_eval import development_mask

TIMING = os.path.join("data", "processed", "betfair_timing.parquet")
FORECAST_DIR = os.path.join("data", "weather_forecast")
PUBLIC = os.path.join("data", "processed", "betfair_sp_public.parquet")
SHRINK_K = 2.0

_GOING_SCALE = [
    ("soft to heavy", 5.5), ("heavy", 6.0), ("yielding to soft", 4.5), ("good to soft", 4.0),
    ("soft", 5.0), ("good to yielding", 3.5), ("yielding", 4.0), ("good to firm", 2.0),
    ("firm", 1.0), ("hard", 1.0), ("good", 3.0),
]


def going_score(text) -> float:
    """Numeric going from the primary description (before any 'in places')."""
    if not isinstance(text, str) or not text.strip():
        return np.nan
    primary = re.split(r"\(", text.lower())[0].strip()
    for key, val in _GOING_SCALE:
        if primary.startswith(key):
            return val
    return np.nan


def ground_preference(res: pd.DataFrame) -> pd.Series:
    """Prior-only soft-minus-firm performance, per runner row of *res*.

    *res* needs horse_id, race_date, off_time, race_id, going, surface,
    finish_position, odds. Performance = finish percentile minus SP-rank
    percentile (non-finishers last); each runner's preference uses only its
    runs strictly before the current race (hard rule 1).
    """
    d = res.copy()
    d["_order"] = pd.to_datetime(d["race_date"].astype(str) + " " + d["off_time"].astype(str), errors="coerce")
    d = d.sort_values(["horse_id", "_order", "race_id"])
    n = d.groupby("race_id")["race_id"].transform("size").astype(float)
    fp = pd.to_numeric(d["finish_position"], errors="coerce").fillna(0)
    fp = fp.where(fp > 0, n)  # non-finishers rank last
    perf = 1.0 - (fp - 1.0) / (n - 1.0).clip(lower=1.0)
    sp_rank = pd.to_numeric(d["odds"], errors="coerce").groupby(d["race_id"]).rank(method="average")
    expected = 1.0 - (sp_rank - 1.0) / (n - 1.0).clip(lower=1.0)
    resid = (perf - expected).where(n > 1)
    g = d["going"].map(going_score)
    turf = d["surface"].fillna("TURF").str.upper().eq("TURF")
    soft = (turf & (g >= 4.0) & resid.notna()).astype(float)
    firm = (turf & (g <= 2.0) & resid.notna()).astype(float)
    r = resid.fillna(0.0)
    out = {}
    for name, flag in (("soft", soft), ("firm", firm)):
        s = (r * flag).groupby(d["horse_id"]).cumsum() - r * flag   # strictly prior
        c = flag.groupby(d["horse_id"]).cumsum() - flag
        out[f"{name}_n"] = c
        out[f"{name}_mean"] = s / (c + SHRINK_K)
    pref = out["soft_mean"] - out["firm_mean"]
    frame = pd.DataFrame({"pref": pref, "soft_n": out["soft_n"], "firm_n": out["firm_n"]}, index=d.index)
    return frame.reindex(res.index)


# ---------------------------------------------------------------------------
# Forecasts
# ---------------------------------------------------------------------------

def load_forecasts() -> pd.DataFrame:
    """Hourly day-ahead forecasts per course (UTC), long format."""
    from src.utils import normalise_track_key
    frames = []
    for name in sorted(os.listdir(FORECAST_DIR)):
        if not name.endswith(".json"):
            continue
        with open(os.path.join(FORECAST_DIR, name), encoding="utf-8") as fh:
            payload = json.load(fh)
        h = payload["hourly"]
        f = pd.DataFrame({k: v for k, v in h.items()})
        f["time"] = pd.to_datetime(f["time"], utc=True)
        f["track_key"] = normalise_track_key(payload["_track"])
        frames.append(f)
    return pd.concat(frames, ignore_index=True)


def window_forecast(races: pd.DataFrame, fc: pd.DataFrame, start_col: str, end_col: str) -> pd.DataFrame:
    """Sum/mean the day-ahead forecast over (start, end] for each race.

    Each hourly value is the forecast for the hour ending at its timestamp,
    as issued 24 h before that hour, so for a window ending at the off and
    starting at most 24 h earlier every value was issued before the start.
    """
    out = []
    by_track = {k: g.set_index("time").sort_index() for k, g in fc.groupby("track_key")}
    for row in races[["race_id", "track_key", start_col, end_col]].itertuples(index=False):
        g = by_track.get(row.track_key)
        if g is None or pd.isna(getattr(row, start_col)):
            continue
        w = g.loc[(g.index > getattr(row, start_col)) & (g.index <= getattr(row, end_col))]
        if w.empty:
            continue
        out.append({
            "race_id": row.race_id,
            "rain_fc": float(w["precipitation_previous_day1"].sum(min_count=1)),
            "temp_fc": float(w["temperature_2m_previous_day1"].mean()),
            "wind_fc": float(w["wind_speed_10m_previous_day1"].mean()),
            "rh_fc": float(w["relative_humidity_2m_previous_day1"].mean()),
            "hours": int(len(w)),
            "n_missing": int(w["precipitation_previous_day1"].isna().sum()),
        })
    return pd.DataFrame(out)


# ---------------------------------------------------------------------------
# Frame and tests
# ---------------------------------------------------------------------------

FC_START = "2024-03-01"
CUT_LOCAL = {"0800": (8, 0, 0), "eve1800": (18, 0, -1)}


def _cut_utc(race_date: pd.Series, hour: int, minute: int, day_offset: int) -> pd.Series:
    day = pd.to_datetime(race_date) + pd.Timedelta(days=day_offset)
    local = (day + pd.Timedelta(hours=hour, minutes=minute)).dt.tz_localize(
        "Europe/London", nonexistent="shift_forward", ambiguous=False)
    return local.dt.tz_convert("UTC")


def frame(cut: str) -> pd.DataFrame:
    """Turf development races from FC_START: decision price at *cut*, BSP,
    forecast rain from the cut to the off, prior-only ground preference."""
    from scripts.microstructure_screen import load
    from src.utils import normalise_track_key
    con = sqlite3.connect("file:data/races.db?mode=ro", uri=True)
    res = pd.read_sql("select race_id, race_date, off_time, track, horse_name, horse_id, going, surface, "
                      "finish_position, odds from results", con)
    pref = ground_preference(res)
    res = res.join(pref)
    res["race_id"] = res["race_id"].astype(str)
    df = load()
    df["race_id"] = df["race_id"].astype(str)
    df = df[df["race_date"].astype(str) >= FC_START]
    df = df.merge(res[["race_id", "horse_name", "surface", "pref", "soft_n", "firm_n"]],
                  on=["race_id", "horse_name"], how="left")
    df = df[df["surface"].fillna("").str.upper().eq("TURF")].copy()
    cols = (f"ltp_{cut}", f"rf_after_{cut}", f"rf_stale_{cut}", f"book_removed_after_{cut}")
    df = attach_bsp(df.drop(columns=["ltp_300s"], errors="ignore"), pd.read_parquet(TIMING), cols=cols)
    for c in cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    priced = (df[f"ltp_{cut}"] > 1).groupby(df["race_id"]).transform("all")
    df = df[priced & df["pref"].notna()].copy()
    race = df["race_id"]
    p = df[f"ltp_{cut}"]
    over = (1.0 / p).groupby(race).transform("sum") + df[f"book_removed_after_{cut}"].fillna(0).groupby(race).transform("first")
    df["p_dec"] = (1.0 / p) / over
    df["P_eff"] = apply_reduction_factor(p.to_numpy(float), (df[f"rf_after_{cut}"].fillna(1.0) * df[f"rf_stale_{cut}"].fillna(1.0)).to_numpy(float))
    df["y"] = np.log(df["P_eff"] / df["bsp"])
    # Forecast window: decision -> off (UTC).
    hour, minute, offset = CUT_LOCAL[cut]
    df["dec_utc"] = _cut_utc(df["race_date"].astype(str), hour, minute, offset)
    df["off_utc"] = pd.to_datetime(df["race_date"].astype(str) + " " + df["off_time"].astype(str)).dt.tz_localize("UTC")
    df["track_key"] = df["track"].map(normalise_track_key)
    races = df.drop_duplicates("race_id")[["race_id", "track_key", "dec_utc", "off_utc"]]
    fc = window_forecast(races, load_forecasts(), "dec_utc", "off_utc")
    df = df.merge(fc, on="race_id", how="inner")
    df = df[df["n_missing"] == 0].copy()
    df["x"] = df["pref"] * df["rain_fc"]
    df["q"] = pd.to_datetime(df["race_date"]).dt.to_period("Q").astype(str)
    return df.reset_index(drop=True)


def partial_corr(df: pd.DataFrame, x: str, y: str, price: str, extra: tuple[str, ...],
                 n_boot: int, seed: int = 0) -> dict:
    """Within-race partial correlation of x and y given the price basis and
    *extra* controls, with a race-clustered bootstrap SE (as
    microstructure_screen.partial_corr_clustered, plus extra controls)."""
    from scripts.triage_features import _residualise, _within_race_demean, market_controls
    sub = df[[x, y, price, "race_id", *extra]].dropna()
    codes = pd.factorize(sub["race_id"])[0]
    xv = _within_race_demean(sub[x].to_numpy(float), codes)
    yv = _within_race_demean(sub[y].to_numpy(float), codes)
    z = market_controls(sub[price].to_numpy(float), codes)
    if extra:
        ez = np.column_stack([_within_race_demean(sub[e].to_numpy(float), codes) for e in extra])
        z = np.column_stack([z, ez])
    xr, yr = _residualise(xv, z), _residualise(yv, z)
    r = float(np.corrcoef(xr, yr)[0, 1])
    rng = np.random.default_rng(seed)
    n_c = codes.max() + 1
    order = np.argsort(codes, kind="stable")
    starts = np.r_[0, np.flatnonzero(np.diff(codes[order])) + 1]
    sizes = np.diff(np.r_[starts, len(order)])
    sums = {k: np.add.reduceat(v, starts) for k, v in (
        ("xx", xr[order] ** 2), ("yy", yr[order] ** 2), ("xy", xr[order] * yr[order]),
        ("x", xr[order]), ("y", yr[order]))}
    boots = np.empty(n_boot)
    for b in range(n_boot):
        w = rng.multinomial(n_c, np.full(n_c, 1.0 / n_c)).astype(float)
        n = (w * sizes).sum()
        mx, my = (w * sums["x"]).sum() / n, (w * sums["y"]).sum() / n
        cov = (w * sums["xy"]).sum() / n - mx * my
        vx = (w * sums["xx"]).sum() / n - mx * mx
        vy = (w * sums["yy"]).sum() / n - my * my
        boots[b] = cov / np.sqrt(vx * vy)
    se = float(boots.std(ddof=1))
    t = r / se if se > 0 else float("nan")
    return {"partial_corr": round(r, 5), "se": round(se, 5), "t": round(t, 2),
            "p": erfc(abs(t) / sqrt(2)) if np.isfinite(t) else float("nan"),
            "n": int(len(sub)), "races": int(n_c)}


# ---------------------------------------------------------------------------
# Ledger 23
# ---------------------------------------------------------------------------

RAIN_MIN_MM = 2.0
PREF_MIN = 0.15
BAND = (1.5, 50.0)


def economic(df: pd.DataFrame, cut: str, n_boot: int) -> dict:
    from src.bet_settlement import betfair_tick_down, settle_win_exchange_bets
    from src.wf_eval import bootstrap_median_ci, bootstrap_roi_ci, clv_summary, fold_summary
    d = df[df["rain_fc"] >= RAIN_MIN_MM].copy()
    p = d[f"ltp_{cut}"].to_numpy(float)
    rf = (d[f"rf_after_{cut}"].fillna(1.0) * d[f"rf_stale_{cut}"].fillna(1.0)).to_numpy(float)
    d["taken"] = apply_reduction_factor(betfair_tick_down(p, 1), rf)
    d["band"] = d[f"ltp_{cut}"].between(*BAND)
    top = d["pref"].where(d["band"]).groupby(d["race_id"]).transform("max")
    d["pick"] = d["band"] & (d["pref"] == top) & (d["pref"] >= PREF_MIN)
    won = d["won"].to_numpy().astype(bool)
    d["pnl_taken"] = settle_win_exchange_bets(d["taken"].to_numpy(), won)
    d["pnl_bsp"] = settle_win_exchange_bets(d["bsp"].to_numpy(float), won)
    out = {}
    for name, mask in (("baseline: band in rain races", d["band"]), ("rule: top preference", d["pick"])):
        s = d[mask]
        if s.empty:
            out[name] = {"n": 0}
            continue
        cl = s["race_id"].to_numpy()
        clv = clv_summary(s["taken"], s["bsp"])
        out[name] = {
            "n": int(len(s)), "win_rate": round(float(s["won"].mean()), 4),
            "median_clv_ci": bootstrap_median_ci((s["taken"] / s["bsp"]).to_numpy(), clusters=cl, n_boot=n_boot),
            "geo_clv": clv.get("geometric_mean_clv"), "beat_close": clv.get("pct_beat_close"),
            "roi_taken": bootstrap_roi_ci(s["pnl_taken"], clusters=cl, n_boot=n_boot),
            "roi_bsp": bootstrap_roi_ci(s["pnl_bsp"], clusters=cl, n_boot=n_boot),
            "quarters": fold_summary(s.groupby("q")["pnl_taken"].mean().mul(100).tolist()),
        }
    r = out["rule: top preference"]
    out["pass"] = bool(r.get("n") and r["median_clv_ci"]["ci_low"] > 1.0 and (r["geo_clv"] or 0) > 1.0
                       and r["roi_taken"]["mean"] > 0)
    return out


def _sire_soft(df: pd.DataFrame) -> pd.Series:
    import pyarrow.parquet as pq
    store = pq.read_table("data/processed/feature_store/featured.parquet",
                          columns=["race_id", "horse_name", "sire_soft_edge"]).to_pandas()
    store["race_id"] = store["race_id"].astype(str)
    m = df[["race_id", "horse_name"]].merge(store.drop_duplicates(["race_id", "horse_name"]),
                                            on=["race_id", "horse_name"], how="left")
    return pd.Series(m["sire_soft_edge"].to_numpy(), index=df.index)


def run(n_boot: int) -> dict:
    report: dict = {}
    d = frame("0800")
    print(f"08:00 frame: {d['race_id'].nunique():,} races, {len(d):,} runners "
          f"({d['race_date'].min()}..{d['race_date'].max()})", flush=True)
    prim = partial_corr(d, "x", "y", "p_dec", ("pref",), n_boot)
    prim["pass"] = bool(prim["partial_corr"] >= 0.01 and prim["p"] < 0.05)
    report["primary 08:00"] = prim
    report["economic 08:00"] = economic(d, "0800", n_boot)
    both = d[(d["soft_n"] > 0) & (d["firm_n"] > 0)]
    report["08:00, soft and firm history"] = partial_corr(both, "x", "y", "p_dec", ("pref",), n_boot)
    d["sire_soft"] = _sire_soft(d)
    d["x_sire"] = d["sire_soft"] * d["rain_fc"]
    report["08:00, sire soft edge"] = partial_corr(d, "x_sire", "y", "p_dec", ("sire_soft",), n_boot)
    e = frame("eve1800")
    print(f"evening frame: {e['race_id'].nunique():,} races, {len(e):,} runners", flush=True)
    report["evening 18:00"] = partial_corr(e, "x", "y", "p_dec", ("pref",), n_boot)
    report["economic evening"] = economic(e, "eve1800", n_boot)
    return report


def render(report: dict) -> str:
    L = ["", "=== ledger 23: forecast ground changes (development, turf, 2024-03..) ==="]
    for k, v in report.items():
        if k.startswith("economic"):
            L.append(f"  {k}: PASS={v.get('pass')}")
            for name, c in v.items():
                if name == "pass" or not isinstance(c, dict) or not c.get("n"):
                    continue
                m, r = c["median_clv_ci"], c["roi_taken"]
                L.append(f"    {name:30s} n={c['n']:6,} win {c['win_rate']:.3f} medCLV {m['estimate']:.4f} "
                         f"[{m['ci_low']:.4f},{m['ci_high']:.4f}] geo {c['geo_clv']:.4f} beat {c['beat_close']:.3f} "
                         f"ROI@taken {r['mean']:+.2f}% [{r['ci_low']:+.2f},{r['ci_high']:+.2f}] "
                         f"ROI@BSP {c['roi_bsp']['mean']:+.2f}% q+ {c['quarters']['folds_positive']}/{c['quarters']['n_folds']}")
        else:
            L.append(f"  {k:32s} partial {v['partial_corr']:+.4f} (se {v['se']:.4f}, t {v['t']:+.2f}, "
                     f"p {v['p']:.3g}) n={v['n']:,} races={v['races']:,}"
                     + (f"  PASS={v['pass']}" if "pass" in v else ""))
    return "\n".join(L) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()
    report = run(args.n_boot)
    print(render(report))
    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2, default=str)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
