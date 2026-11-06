"""Experiment 047 round 2: observed-only OR context, shrunk connection dimensions, rank shares.

Run before ``scripts/ablate_impl_review_b.py``.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd

from src.feature_engineer import _add_result_history_flags, _cumulative, _entity_day_start_view, _event_sort_key
from src.feature_semantics import observed_rating
from src.feature_store import FeatureStore
from src.wf_eval import development_mask

OUT = ROOT / "data/audit/feature_impl_review_047b"
OUT.mkdir(parents=True, exist_ok=True)

# (entity, dimension column, minimum runs for the stored raw rate, stored rate name)
DIMS = [("jockey", "going", 2, "jockey_going_wr"), ("jockey", "dist_category", 2, "jockey_dist_cat_wr"),
        ("jockey", "track", 0, "jockey_track_win_rate"), ("trainer", "going", 2, "trainer_going_wr"),
        ("trainer", "dist_category", 2, "trainer_dist_cat_wr"), ("trainer", "race_class", 2, "trainer_class_wr"),
        ("trainer", "track", 0, "trainer_track_win_rate")]
RELATIVE = {"trainer_going_wr": ["vs_field", "vs_best_in_race"], "trainer_class_wr": ["vs_field", "z_in_race", "vs_best_in_race"],
            "trainer_dist_cat_wr": ["vs_best_in_race"], "trainer_track_win_rate": ["vs_field", "z_in_race", "vs_best_in_race"],
            "jockey_track_win_rate": ["vs_best_in_race", "z_in_race"]}
CONNECTIONS = ["jockey_elo_z_in_race", "jockey_wr_14d_z_in_race", "trainer_wr_14d_z_in_race",
               "trainer_win_rate_shrunk_z_in_race", "jockey_track_win_rate_z_in_race", "trainer_track_win_rate_z_in_race"]
COLUMNS = ["race_id", "race_date", "off_time", "horse_name", "won", "finish_position", "finish_pos_label", "num_runners",
           "jockey", "trainer", "going", "dist_category", "track", "race_class", "official_rating", "has_official_rating",
           "jockey_win_rate_shrunk", "trainer_win_rate_shrunk", "horse_glicko", "horse_glicko_rank", "horse_elo",
           "horse_elo_rank", "or_rank_in_race", "or_vs_second_top", "field_std_or", "connections_relative_score",
           *[d[3] for d in DIMS], *[f"{k}_{s}" for k, v in RELATIVE.items() for s in v], *CONNECTIONS]


def relative(x, races, kind, fill):
    g = x.groupby(races)
    mean, std = g.transform("mean"), g.transform("std").fillna(0.0)
    if kind == "vs_field":
        out = x - mean
    elif kind == "z_in_race":
        out = pd.Series(np.where(std > 1e-12, (x - mean) / std, 0.0), index=x.index)
        return out if fill else out.where(x.notna())
    else:
        out = x - g.transform("max")
    return out.fillna(0.0) if fill else out


def main() -> None:
    store = FeatureStore()
    frame = pd.read_parquet(store.featured_path, columns=list(dict.fromkeys(COLUMNS)))
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
        checks[name] = float((np.isclose(stored, mine, rtol=1e-6, atol=1e-9) | (np.isnan(stored) & np.isnan(mine))).mean())

    # ── shrunk connection dimensions ──────────────────────────────
    shrunk = {}
    for entity, dim, minimum, name in DIMS:
        key = frame[entity].astype(str) + "||" + frame[dim].astype(str)
        c = _cumulative(key.to_numpy(), races.to_numpy(),
                        {f"_{name}_runs": frame._result_start.to_numpy(), f"_{name}_wins": frame.won.to_numpy()})
        work = pd.DataFrame({"race_date": frame.race_date, "_k": key, "runs": c[f"_{name}_runs"], "wins": c[f"_{name}_wins"]})
        work = _entity_day_start_view(work, ["_k"], ["runs", "wins"])
        raw = np.where(work.runs > minimum, work.wins / work.runs.where(work.runs > 0), 0)
        check(name, raw)
        prior = frame[f"{entity}_win_rate_shrunk"]
        shrunk[name] = (work.wins + 10.0 * prior) / (work.runs + 10.0)
    out["jockey_going_wr__fix"] = shrunk["jockey_going_wr"]
    out["jockey_dist_cat_wr__fix"] = shrunk["jockey_dist_cat_wr"]
    out["trainer_going_wr__fix"] = shrunk["trainer_going_wr"]
    for base, kinds in RELATIVE.items():
        for kind in kinds:
            check(f"{base}_{kind}", relative(frame[base], races, kind, fill=True))
            out[f"{base}_{kind}__fix"] = relative(shrunk[base], races, kind, fill=False)
    zs = {c: frame[c] for c in CONNECTIONS}
    zs["jockey_track_win_rate_z_in_race"] = out["jockey_track_win_rate_z_in_race__fix"]
    zs["trainer_track_win_rate_z_in_race"] = out["trainer_track_win_rate_z_in_race__fix"]
    check("connections_relative_score", pd.DataFrame({c: frame[c] for c in CONNECTIONS}).mean(axis=1))
    out["connections_relative_score__fix"] = pd.DataFrame(zs).mean(axis=1)

    # ── OR context from observed marks only ───────────────────────
    obs = observed_rating(frame)
    rank_first = obs.groupby(races).rank(ascending=False, method="first")
    second = obs.where(rank_first.eq(2)).groupby(races).transform("max")
    out["or_rank_in_race__fix"] = obs.groupby(races).rank(ascending=False, method="min")
    out["or_vs_second_top__fix"] = obs - second
    out["field_std_or__fix"] = obs.groupby(races).transform("std")

    # ── rating ranks as a share of the field ──────────────────────
    for col, rating in [("horse_glicko_rank", "horse_glicko"), ("horse_elo_rank", "horse_elo")]:
        rank = frame[rating].groupby(races).rank(ascending=False, method="min")
        check(col, rank)
        count = frame[rating].notna().groupby(races).transform("sum")
        out[f"{col}__fix"] = (rank - 1) / (count - 1).where(count > 1)

    print("reproduction (share of rows matching the store):")
    for name, share in checks.items():
        print(f"  {share:8.5f}  {name}")
    out.to_parquet(OUT / "projection.parquet", index=False)
    pd.Series(checks).to_json(OUT / "reproduction.json", indent=2)
    print(f"wrote {len(out):,} rows x {out.shape[1] - 3} repaired columns")


if __name__ == "__main__":
    main()
