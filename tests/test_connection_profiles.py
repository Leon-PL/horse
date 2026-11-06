import numpy as np
import pandas as pd

from src.connection_profiles import (context_columns, design, feature_names, fit_profiles, new_yard,
                                     quarterly_features)


def test_new_yard_flags_a_change_of_trainer_only():
    frame = pd.DataFrame({
        "horse_id": [1, 1, 1, 2, 2],
        "race_date": pd.to_datetime(["2024-01-01", "2024-02-01", "2024-03-01", "2024-01-05", "2024-02-05"]),
        "trainer": ["A", "A", "B", "C", "C"],
    }).iloc[[4, 2, 0, 3, 1]]
    assert new_yard(frame).sort_index().tolist() == [False, False, True, False, False]


def _frame(n, rng, start="2023-01-01", days=900):
    dates = pd.to_datetime(start) + pd.to_timedelta(np.sort(rng.integers(0, days, n)), unit="D")
    return pd.DataFrame({
        "race_date": dates, "race_type": rng.choice(["Flat", "Hurdle", "Chase"], n),
        "surface": rng.choice(["Turf", "Polytrack"], n), "race_name": "Some Handicap",
        "going": rng.choice(["Good", "Soft", "Heavy", "Standard"], n),
        "race_class": rng.choice(["1.0", "4.0", "6.0", "Nan"], n), "handicap": rng.integers(0, 2, n),
        "age": rng.integers(2, 8, n), "distance_furlongs": rng.uniform(5, 24, n),
        "num_runners": rng.integers(4, 16, n), "horse_prev_races": rng.choice([0, 1, 5], n),
        "days_since_last_race": rng.uniform(5, 300, n), "new_yard": rng.random(n) < 0.1,
        "trainer": rng.choice([f"t{i}" for i in range(40)], n), "jockey": rng.choice([f"j{i}" for i in range(30)], n),
        "sire": rng.choice([f"s{i}" for i in range(25)], n),
    })


def test_contexts_cover_every_group():
    rng = np.random.default_rng(0)
    contexts = context_columns(_frame(500, rng))
    active, names = design(contexts, ["discipline", "state", "trip"])
    assert active.shape == (500, len(names))
    np.testing.assert_allclose(active.sum(axis=1), 3.0)  # one value per group


def test_profiles_recover_a_planted_style():
    rng = np.random.default_rng(1)
    frame = _frame(60_000, rng)
    active, names = design(context_columns(frame), ["state", "handicap"])
    trainers = frame.trainer.to_numpy()
    # half the yards do far better with debutants than the market expects
    good_with_debutants = np.isin(trainers, [f"t{i}" for i in range(20)])
    debut = frame.horse_prev_races.eq(0).to_numpy()
    signal = np.where(good_with_debutants & debut, 0.15, np.where(debut, -0.05, 0.0))
    residual = signal + rng.normal(0, 0.25, len(frame))
    profiles = fit_profiles(trainers, active, names, residual, np.ones(len(frame)))
    scored = profiles.score(trainers, active)
    assert np.corrcoef(scored["fit"], signal)[0, 1] > 0.8
    debut_projection = profiles.projection(scored["theta"], "state=debut")
    assert debut_projection[good_with_debutants].mean() > debut_projection[~good_with_debutants].mean() + 0.1


def test_unknown_entities_score_zero():
    rng = np.random.default_rng(2)
    frame = _frame(2_000, rng)
    active, names = design(context_columns(frame), ["state"])
    profiles = fit_profiles(frame.trainer, active, names, rng.normal(size=2_000), np.ones(2_000))
    scored = profiles.score(["nobody", "t1"], active[:2])
    assert scored["fit"][0] == 0.0 and scored["runs"][0] == 0.0 and scored["runs"][1] > 0


def test_quarterly_features_never_see_their_own_quarter():
    rng = np.random.default_rng(3)
    frame = _frame(8_000, rng, start="2023-01-01", days=540)
    q = frame.race_date.dt.to_period("Q")
    last = q.eq(q.max())
    frame.loc[last, "trainer"] = "newcomer"          # appears only in the final quarter
    residual = np.where(frame.trainer.eq("newcomer"), 1.0, rng.normal(0, 0.2, len(frame)))
    features = quarterly_features(frame.reset_index(drop=True), residual)
    assert list(features.columns) == feature_names()
    final = features[last.to_numpy()]
    assert (final.trainer_prof_fit == 0).all() and (final.trainer_prof_runs == 0).all()
    first = q.eq(q.min()).to_numpy()
    assert features[first].isna().all().all()        # nothing earlier to learn from
