"""Print a race as a punter would read it, beside what the model was given.

    python -m research.error_anatomy.card_audit RACE_ID [RACE_ID ...]
    python -m research.error_anatomy.card_audit --sample 3 --seed 1 [--kind Hurdle]

For each runner: the card line, the model's and the market's chance, the
horse's earlier runs from the results database, and the raw values behind the
main inputs. Nothing is fitted; this is for reading inputs against the record.
"""
import argparse
import sqlite3

import numpy as np
import pandas as pd

from research.error_anatomy.frame import OUT
from research.meta_learner import screen_candidates as sc

RAW = ["official_rating", "has_official_rating", "horse_elo", "horse_margin_elo", "horse_glicko", "horse_glicko_rd",
       "horse_ts_mu", "horse_ts_sigma", "horse_prev_races", "prior_form_runs", "hidden_form_runs", "horse_avg_pos_5",
       "days_since_last_race", "speed_fig_best_5", "speed_fig_rel_avg_5", "rtv_time_fig_last", "rtv_time_fig_best3",
       "rtv_card_tf_stars", "rtv_card_tf_tipped", "rtv_text_logit", "rtv_verdict_rank", "x063_tf_vs_field",
       "trainer_pr_30d", "trainer_elo", "jockey_elo", "jt_win_rate", "sire_win_rate_shrunk", "horse_track_win_rate",
       "horse_cd_win_rate", "class_change", "distance_change", "or_change", "weight_change",
       "horse_prior_market_logrel_3_vs_field", "horse_non_finish_rate_shrunk", "base_win_rate", "form", "horse_runs"]


def history(connection, horse_id, before, limit=8):
    return pd.read_sql_query(
        "select race_date, track, race_type, distance_furlongs as fur, going, race_class as cls, num_runners as n, "
        "finish_position as pos, lengths_behind as lb, official_rating as o_r, weight_lbs as wt, odds as sp, "
        "jockey, trainer, race_name from results where horse_id = ? and race_date < ? order by race_date desc limit ?",
        connection, params=(str(horse_id), str(before)[:10], limit))


def show(race_id, frame, store, connection):
    race = frame.loc[frame.race_id.eq(str(race_id))].sort_values("p_average", ascending=False)
    if race.empty:
        print(f"{race_id}: not in the scored frame")
        return
    first = race.iloc[0]
    print("=" * 150)
    print(f"{race_id}  {str(first.race_date)[:10]} {first.off_time} {first.track}  {first.race_name}")
    print(f"class {first.race_class}  {first.race_type}  {first.distance_furlongs}f  {first.going}  {first.surface}  "
          f"handicap {first.handicap}  runners {first.num_runners}")
    raw = store.loc[store.race_id.eq(str(race_id))].set_index("horse_id")
    for _, r in race.iterrows():
        print("-" * 150)
        print(f"{r.horse_name}  age {r.age} {r.sex}  wt {r.weight_lbs}  OR {r.official_rating}  draw {r.draw}  "
              f"hg {r.headgear}  J {r.jockey}  T {r.trainer}  card days {r.days_since_last_run}")
        print(f"   model {r.p_average:.3f}  BSP {r.p_bsp:.3f}  SP odds {r.odds}  finished {r.finish_pos_label} "
              f"beaten {r.beaten_lengths}")
        if r.horse_id in raw.index:
            values = raw.loc[r.horse_id]
            line = "  ".join(f"{c}={values[c]:.3g}" if isinstance(values[c], (float, np.floating, int, np.integer))
                             else f"{c}={values[c]}" for c in raw.columns if c not in ("race_id", "race_date"))
            print("   inputs: " + line)
        past = history(connection, r.horse_id, r.race_date)
        if past.empty:
            print("   no earlier runs in the database")
        else:
            print("   " + past.to_string(index=False).replace("\n", "\n   "))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("races", nargs="*")
    parser.add_argument("--sample", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--kind")
    parser.add_argument("--max-runners", type=int, default=9)
    args = parser.parse_args()
    pd.set_option("display.width", 250, "display.max_columns", 40, "display.max_colwidth", 38)
    frame = pd.read_parquet(OUT / "scored_prices.parquet")
    races = list(args.races)
    if args.sample:
        pool = frame.loc[frame.num_runners.le(args.max_runners)]
        if args.kind:
            pool = pool.loc[pool.race_type.eq(args.kind)]
        races += list(pd.Series(pool.race_id.unique()).sample(args.sample, random_state=args.seed))
    import pyarrow.parquet as pq
    available = set(pq.read_schema(sc.store_file()).names)
    columns = [c for c in RAW if c in available and c not in ("form", "horse_runs")]
    missing = [c for c in RAW if c not in available]
    if missing:
        print("not in the store:", missing)
    store = sc.read_store(columns, start=frame.race_date.min())
    store["race_id"] = store.race_id.astype(str)
    connection = sqlite3.connect("file:data/races.db?mode=ro", uri=True)
    for race_id in races:
        show(race_id, frame, store.loc[store.race_id.eq(str(race_id))], connection)


if __name__ == "__main__":
    main()
