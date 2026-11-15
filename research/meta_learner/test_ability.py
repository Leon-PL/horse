import numpy as np
import pandas as pd

from research.meta_learner.ability import (ability_in_pounds, bounce_inputs, ground_preference,
                                          improvement_inputs, relatives_ratings)

RESULT_COLUMNS = ["finish_position", "beaten_lengths", "official_rating", "weight_lbs"]


def synthetic_history(seed=0, days=40, horses=60, field=8):
    """Flat races with true abilities; families of three share a dam; trainers of ten horses."""
    rng = np.random.default_rng(seed)
    ability = rng.normal(70, 12, horses)
    rows = []
    race = 0
    for day in range(days):
        for slot in range(2):
            runners = rng.choice(horses, field, replace=False)
            weight = 130 + np.round((ability[runners] - 70) / 3)
            performance = ability[runners] - weight + rng.normal(0, 6, field)
            order = np.argsort(-performance)
            behind = (performance[order[0]] - performance) / 2.
            for i, horse in enumerate(runners):
                rows.append({
                    "race_id": race, "horse_id": int(horse), "race_date": pd.Timestamp("2024-03-01") + pd.Timedelta(days=day),
                    "off_time": "14:00" if slot == 0 else "15:30", "race_type": "Flat", "distance_furlongs": 8.,
                    "finish_position": int(np.flatnonzero(order == i)[0] + 1), "beaten_lengths": float(behind[i]),
                    "weight_lbs": float(weight[i]), "reported_jockey_claim_lbs": 0.,
                    "official_rating": float(np.round(ability[horse])), "age": 4,
                    "trainer": f"T{horse // 10}", "dam": f"Dam{horse // 3}", "damsire": "Sire", "sire": f"S{horse % 5}", "won": 0,
                    "going": ["Good", "Soft", "Heavy", "Fast"][day % 4], "track": "Ascot"})
            race += 1
    frame = pd.DataFrame(rows)
    frame["won"] = frame.finish_position.eq(1).astype(int)
    return frame, ability


def unchanged_before(function, history, cutoff_race, *args):
    """Scrambling results from ``cutoff_race`` on must not move features of races before it, or of it."""
    base = function(history, *args).set_index(["race_id", "horse_id"]).sort_index()
    changed = history.copy()
    later = changed.race_id >= cutoff_race
    rng = np.random.default_rng(5)
    changed.loc[later, "finish_position"] = rng.permutation(changed.loc[later, "finish_position"].to_numpy())
    changed.loc[later, "beaten_lengths"] = rng.uniform(0, 20, later.sum())
    revised = function(changed, *args).set_index(["race_id", "horse_id"]).sort_index()
    upto = base.index.get_level_values("race_id") <= cutoff_race
    pd.testing.assert_frame_equal(base.loc[upto], revised.loc[upto])
    return base, revised


def test_ability_uses_only_earlier_races():
    history, _ = synthetic_history()
    base, revised = unchanged_before(ability_in_pounds, history, 41)
    later = base.index.get_level_values("race_id") > 41
    assert not np.allclose(base.loc[later, "ab_mu"].fillna(0), revised.loc[later, "ab_mu"].fillna(0))


def test_ability_tracks_true_ability():
    history, ability = synthetic_history(days=80)
    features = ability_in_pounds(history)
    last = features.merge(history[["race_id", "horse_id"]]).sort_values("race_id").groupby("horse_id").ab_mu.last()
    assert np.corrcoef(last.to_numpy(), ability[last.index.to_numpy()])[0, 1] > .9
    first = features.loc[features.ab_runs.eq(0)]
    assert first.ab_mu.isna().all()  # a debut has no prior run in the code


def test_relatives_use_earlier_dates_only():
    history, _ = synthetic_history()
    # The second race of a day is later than the first but on the same date, so cut at a day boundary.
    unchanged_before(relatives_ratings, history, 40)
    features = relatives_ratings(history).set_index(["race_id", "horse_id"])
    assert features.loc[features.index.get_level_values("race_id") < 2, "relr_siblings"].eq(0).all()


def test_relatives_ignore_same_day_runs_of_siblings():
    history, _ = synthetic_history()
    base = relatives_ratings(history).set_index(["race_id", "horse_id"]).sort_index()
    changed = history.copy()
    second = changed.race_id.eq(41)  # the later race on day 20; race 40 is the same day
    changed.loc[second, "finish_position"] = changed.loc[second, "finish_position"].to_numpy()[::-1]
    revised = relatives_ratings(changed).set_index(["race_id", "horse_id"]).sort_index()
    pd.testing.assert_frame_equal(base.loc[[40]], revised.loc[[40]])


def test_ground_preference_waits_for_publication():
    history, _ = synthetic_history(days=10)
    horse = int(history.horse_id.iloc[0])
    withdrawals = pd.DataFrame({"horse_id": [str(horse)], "withdrawn_going": [5.],  # Heavy
                                "available_at": [pd.Timestamp("2024-03-04 12:00", tz="UTC")]})
    features = ground_preference(history, withdrawals).merge(history[["race_id", "horse_id", "race_date", "going"]])
    mine = features.loc[features.horse_id.eq(horse)]
    before = mine.race_date < pd.Timestamp("2024-03-04")
    assert mine.loc[before, "gw_count"].eq(0).all() and mine.loc[before, "gw_softest"].isna().all()
    after = mine.loc[mine.race_date > pd.Timestamp("2024-03-04")]
    assert after.gw_count.eq(1).all()
    heavy = after.loc[after.going.eq("Heavy")]
    assert heavy.gw_same_going.eq(1).all() and heavy.gw_distance.eq(0).all()


def test_improvement_and_bounce_use_only_earlier_races():
    history, _ = synthetic_history()
    # Cut at a day boundary: trainer and sire summaries use strictly earlier dates.
    for function in (improvement_inputs, bounce_inputs):
        base, revised = unchanged_before(function, history, 40)
        later = base.index.get_level_values("race_id") > 41
        assert not base.loc[later].equals(revised.loc[later])


def test_bounce_reads_the_previous_run():
    history, _ = synthetic_history()
    features = bounce_inputs(history).merge(history[["race_id", "horse_id", "race_date", "finish_position"]])
    horse = features.loc[features.horse_id.eq(features.horse_id.iloc[0])].sort_values("race_id")
    assert np.isnan(horse.bn_last_close.iloc[0]) and horse.bn_runs_60d.iloc[0] == 0
    expected = (horse.finish_position.iloc[0] == 1) or (history.set_index(["race_id", "horse_id"]).loc[
        (horse.race_id.iloc[0], horse.horse_id.iloc[0]), "beaten_lengths"] < 1)
    assert horse.bn_last_close.iloc[1] == float(expected)
    assert horse.bn_days_since.iloc[1] == (horse.race_date.iloc[1] - horse.race_date.iloc[0]).days
