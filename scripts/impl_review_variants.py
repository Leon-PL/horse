"""Experiment 047 round 1: repaired versions of model inputs from the feature store.

Each repaired column gets a ``__fix`` (or ``__loo``) suffix. Originals are
first reproduced with the pipeline's own semantics and compared with the
store, so the only difference between arms is the intended repair.
Development period only. Run before ``scripts/ablate_impl_review.py``.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd

from src.feature_engineer import _add_result_history_flags, _event_sort_key, _rolling
from src.feature_store import FeatureStore
from src.wf_eval import development_mask

OUT = ROOT / "data/audit/feature_impl_review_047"
OUT.mkdir(parents=True, exist_ok=True)

HIGHER = ["horse_dist_cat_wr_shrunk", "jockey_elo", "jockey_win_rate_shrunk", "jockey_place_rate_shrunk",
          "jockey_track_win_rate", "jt_prev_runs", "trainer_freq", "trainer_elo_vs_field",
          "trainer_place_rate_shrunk", "trainer_track_win_rate", "trainer_class_wr", "trainer_dist_cat_wr",
          "trainer_going_wr", "field_avg_elo_10", "field_avg_elo_ewma", "horse_elo_momentum",
          "horse_elo_momentum_vs_field", "horse_margin_elo_momentum", "combined_elo_vs_field"]
LOWER = ["horse_avg_position", "days_since_last_win", "jockey_te_norm_pos", "jockey_te_ewma",
         "trainer_te_norm_pos", "trainer_te_ewma", "jt_te_norm_pos", "jockey_te_norm_pos_365d",
         "trainer_te_norm_pos_365d"]
NAN_VS_FIELD = {"speed_fig_best_5": False, "speed_fig_adj_best_5": False, "field_avg_elo_10": True}

BASE = ["race_id", "race_date", "off_time", "horse_id", "horse_name", "won", "finish_position",
        "finish_pos_label", "num_runners", "horse_prev_races", "is_maiden", "horse_elo", "draw",
        "prize_money", "horse_avg_prize_contested", "years_from_peak_abs", "age_decline_curve",
        "race_type", "speed_fig_best_5", "speed_fig_adj_best_5"]
STORED = ["horse_avg_pos_3", "horse_best_pos_5", "form_slope", "elo_x_recent_form", "days_since_last_race",
          "days_log", "days_since_last_race_z_in_race", "days_log_vs_field", "freshness_relative_score",
          "days_since_last_win", "prize_money_change_vs_avg", "draw_x_field_size",
          *[f"{c}_vs_best_in_race" for c in HIGHER + LOWER],
          *[f"{c}_vs_field" for c in NAN_VS_FIELD]]


def race_relative(x: pd.Series, races: pd.Series, *, higher: bool, loo: bool, fill: bool):
    """Return (vs_field, z, vs_best) with the pipeline's or the repaired semantics."""
    g = x.groupby(races)
    mean, std = g.transform("mean"), g.transform("std").fillna(0.0)
    signed = x - mean if higher else mean - x
    z = pd.Series(np.where(std > 1e-12, signed / std, 0.0), index=x.index)
    if not fill:
        z = z.where(x.notna())
    if loo:
        s = x if higher else -x
        top1 = s.groupby(races).transform("max")
        n_top = s.eq(top1).groupby(races).transform("sum")
        second = s.where(~s.eq(top1)).groupby(races).transform("max")
        other_best = np.where(s.eq(top1) & n_top.eq(1), second, top1)
        vs_best = s - other_best  # top runner: positive margin over the next best
    else:
        best = g.transform("max" if higher else "min")
        vs_best = x - best if higher else best - x
    if fill:
        return signed.fillna(0.0), z, pd.Series(vs_best, index=x.index).fillna(0.0)
    return signed, z, pd.Series(vs_best, index=x.index)


def main() -> None:
    store = FeatureStore()
    frame = pd.read_parquet(store.featured_path, columns=list(dict.fromkeys(BASE + HIGHER + LOWER + STORED)))
    frame = frame.loc[development_mask(frame.race_date)].copy()
    frame["_event_dt"] = _event_sort_key(frame)
    frame = frame.sort_values(["_event_dt", "race_id"], kind="stable").reset_index(drop=True)
    frame = _add_result_history_flags(frame)
    races = frame.race_id
    out = pd.DataFrame({"race_id": races, "horse_name": frame.horse_name, "race_date": frame.race_date})
    checks = {}

    def check(name, mine):
        stored = pd.to_numeric(frame[name], errors="coerce").to_numpy(float)
        mine = np.asarray(mine, dtype=float)
        both = np.isnan(stored) & np.isnan(mine)
        close = np.isclose(stored, mine, rtol=1e-6, atol=1e-6) | both
        checks[name] = float(close.mean())

    # ── reproduce originals ───────────────────────────────────────
    pos_ops = [("p3", "mean", 3), ("p10", "mean", 10), ("b5", "min", 5)]
    orig = _rolling(frame, "horse_id", "_finish_pos_for_history", pos_ops)
    check("horse_avg_pos_3", orig["p3"]); check("horse_best_pos_5", orig["b5"])
    check("form_slope", np.nan_to_num(orig["p10"]) - np.nan_to_num(orig["p3"]))
    for col in HIGHER + LOWER:
        _, _, vb = race_relative(pd.to_numeric(frame[col], errors="coerce"), races, higher=col in HIGHER, loo=False, fill=True)
        check(f"{col}_vs_best_in_race", vb)
    for col, higher in NAN_VS_FIELD.items():
        vf, _, _ = race_relative(frame[col], races, higher=higher, loo=False, fill=True)
        check(f"{col}_vs_field", vf)
    zs = {}
    for col in ["days_since_last_race", "days_since_last_win", "years_from_peak_abs", "age_decline_curve"]:
        zs[col] = race_relative(frame[col], races, higher=False, loo=False, fill=True)[1]
    check("days_since_last_race_z_in_race", zs["days_since_last_race"])
    check("freshness_relative_score", pd.DataFrame(zs).mean(axis=1))
    check("days_log_vs_field", race_relative(frame.days_log, races, higher=False, loo=False, fill=True)[0])
    print("reproduction (share of rows matching the store):")
    for name, share in checks.items():
        print(f"  {share:8.5f}  {name}")

    # ── A: NaN-preserving relative transforms (cold-start bases) ──
    for col, higher in NAN_VS_FIELD.items():
        out[f"{col}_vs_field__fix"] = race_relative(frame[col], races, higher=higher, loo=False, fill=False)[0]
    for col in ["field_avg_elo_10", "field_avg_elo_ewma"]:
        out[f"{col}_vs_best_in_race__fix"] = race_relative(frame[col], races, higher=True, loo=False, fill=False)[2]

    # ── B: leave-one-out "vs best other" for every vs_best input ──
    for col in HIGHER + LOWER:
        x = pd.to_numeric(frame[col], errors="coerce")
        out[f"{col}_vs_best_in_race__loo"] = race_relative(x, races, higher=col in HIGHER, loo=True, fill=False)[2]

    # ── C: cold starts stay missing instead of 60 / 365 days ──────
    debut = frame.horse_prev_races.eq(0)
    never_won = frame.is_maiden.eq(1)
    dslr = frame.days_since_last_race.where(~debut)
    dslw = frame.days_since_last_win.where(~(debut | never_won))
    out["days_since_last_race__fix"] = dslr
    out["days_log__fix"] = np.log1p(dslr)
    out["days_log_vs_field__fix"] = race_relative(np.log1p(dslr), races, higher=False, loo=False, fill=False)[0]
    z_fix = {
        "a": race_relative(dslr, races, higher=False, loo=False, fill=False)[1].where(dslr.notna()),
        "b": race_relative(dslw, races, higher=False, loo=False, fill=False)[1].where(dslw.notna()),
        "c": zs["years_from_peak_abs"], "d": zs["age_decline_curve"]}
    out["days_since_last_race_z_in_race__fix"] = z_fix["a"]
    out["freshness_relative_score__fix"] = pd.DataFrame(z_fix).mean(axis=1)
    out["days_since_last_win__fix"] = dslw
    out["days_since_last_win_vs_best_in_race__fix"] = race_relative(dslw, races, higher=False, loo=False, fill=False)[2]
    out["days_since_last_win_vs_best_in_race__fixloo"] = race_relative(dslw, races, higher=False, loo=True, fill=False)[2]

    # ── D: field-size-normalised form; non-finishers count as last ─
    n = frame.num_runners.astype(float)
    fin = frame._finished_result.eq(1)
    dnf = frame._non_finisher_result.eq(1)
    frame["_npos"] = np.where(fin, (frame.finish_position - 1) / np.maximum(n - 1, 1), np.where(dnf, 1.0, np.nan))
    roll = _rolling(frame, "horse_id", "_npos", [("n3", "mean", 3), ("n10", "mean", 10), ("nb5", "min", 5)])
    filled, seen = frame._npos.fillna(0), frame._npos.notna().astype(float)
    cum_sum = filled.groupby(frame.horse_id).cumsum() - filled
    cum_n = seen.groupby(frame.horse_id).cumsum() - seen
    career = (cum_sum / cum_n).where(cum_n > 0)
    out["horse_avg_pos_3__fix"] = roll["n3"]
    out["horse_best_pos_5__fix"] = roll["nb5"]
    out["form_slope__fix"] = roll["n10"] - roll["n3"]
    out["elo_x_recent_form__fix"] = (frame.horse_elo - 1500) / 400 * (1 - roll["n3"])
    out["horse_avg_position_vs_best_in_race__fix"] = race_relative(career, races, higher=False, loo=False, fill=False)[2]
    out["horse_avg_position_vs_best_in_race__fixloo"] = race_relative(career, races, higher=False, loo=True, fill=False)[2]

    # ── E: prize ratio on a log scale; draw by stall, not runners ─
    avg = frame.horse_avg_prize_contested
    out["prize_money_change_vs_avg__fix"] = np.log(frame.prize_money.clip(lower=1) / avg.where(avg > 0))
    draw = frame.draw.where(frame.draw > 0)
    stall_pct = draw / draw.groupby(races).transform("max")
    jumps = frame.race_type.astype(str).isin(["Hurdle", "Chase"])
    out["draw_x_field_size__fix"] = (stall_pct * (n / 20.0).clip(0, 2)).where(~jumps)

    out.to_parquet(OUT / "projection.parquet", index=False)
    pd.Series(checks).to_json(OUT / "reproduction.json", indent=2)
    print(f"wrote {len(out):,} rows x {out.shape[1] - 3} repaired columns")


if __name__ == "__main__":
    main()
