"""Walk-forward feature selection primitives (experiment 048)."""

import numpy as np
import pandas as pd
import pytest

from src import feature_selection as fs


def test_hygiene_drops_empty_constant_and_duplicate_columns():
    rng = np.random.default_rng(0)
    n = 1000
    signal = rng.normal(size=n).astype(np.float32)
    indicator = np.where(rng.random(n) < 0.3, 1.0, np.nan).astype(np.float32)
    negative_zero = np.zeros(n, dtype=np.float32)
    negative_zero[:500] = -0.0
    negative_zero[500:] = 1.0
    positive_zero = np.where(np.arange(n) < 500, 0.0, 1.0).astype(np.float32)
    matrix = np.column_stack([
        signal,                                # kept
        np.full(n, np.nan, np.float32),        # empty
        np.full(n, 3.0, np.float32),           # constant
        signal.copy(),                         # duplicate of signal
        indicator,                             # constant where present: an indicator
        negative_zero,                         # equal to positive_zero
        positive_zero,
    ])
    names = ["a", "empty", "const", "a_copy", "flag", "neg", "pos"]
    kept, reasons = fs.hygiene_filter(matrix, names, prefer=["pos", "a_copy"])
    assert kept == ["a_copy", "flag", "pos"]
    assert reasons["a"] == "duplicate of a_copy"
    assert reasons["neg"] == "duplicate of pos"
    assert reasons["const"] == "constant"
    assert "empty" in reasons


def test_race_groups_requires_contiguous_races():
    assert fs.race_groups(["r1", "r1", "r2", "r3", "r3", "r3"]).tolist() == [2, 1, 3]
    with pytest.raises(ValueError):
        fs.race_groups(["r1", "r2", "r1"])


def test_race_normalise_sums_to_one_per_race():
    p = fs.race_normalise([0.2, 0.2, 0.1, 0.3], np.array([2, 2]))
    assert np.allclose(p, [0.5, 0.5, 0.25, 0.75])
    with pytest.raises(ValueError):
        fs.race_normalise([0.0, 0.0], np.array([2]))


def test_clustered_se_matches_iid_formula_for_singleton_clusters():
    rng = np.random.default_rng(1)
    delta = rng.normal(size=500)
    se = fs.clustered_se(delta, np.arange(500))
    assert se == pytest.approx(delta.std(ddof=1) / np.sqrt(500), rel=1e-9)


def test_clustered_se_widens_when_races_share_a_day():
    rng = np.random.default_rng(2)
    day_effect = np.repeat(rng.normal(size=50), 10)
    delta = day_effect + 0.1 * rng.normal(size=500)
    assert fs.clustered_se(delta, np.repeat(np.arange(50), 10)) > 2 * fs.clustered_se(delta, np.arange(500))


def test_one_se_choice_takes_the_smallest_step_within_noise():
    rng = np.random.default_rng(3)
    clusters = np.arange(4000) // 10
    base = rng.exponential(2.0, size=4000)
    noise = 0.05 * rng.normal(size=4000)
    noise -= noise.mean()
    within = noise + 0.5 * fs.clustered_se(noise, clusters)   # worse than best by half an SE
    losses = np.vstack([base, base + within, base + 0.05 + noise])
    choice = fs.one_se_choice([400, 200, 100], losses, clusters)
    assert choice["best_step"] == 0
    assert choice["chosen_step"] == 1
    assert not choice["steps"][2]["within_one_se"]


def test_elimination_schedule_is_coarse_then_fine_and_stops_at_the_floor():
    sizes = fs.elimination_schedule(1000, n_min=30, fine_below=250)
    assert sizes[0] == 1000 and sizes[-1] == 30
    assert all(a > b for a, b in zip(sizes, sizes[1:]))
    assert sizes[1] == 800
    above = [b / a for a, b in zip(sizes, sizes[1:]) if a > 250]
    below = [b / a for a, b in zip(sizes, sizes[1:]) if a <= 250 and b > 30]
    assert all(r == pytest.approx(0.8, abs=0.01) for r in above)
    assert all(r == pytest.approx(0.9, abs=0.02) for r in below)


def test_importance_is_averaged_as_per_fold_shares():
    combined = fs.aggregate_importance([np.array([10.0, 0.0]), np.array([0.0, 1.0])])
    assert combined.tolist() == [0.5, 0.5]
    assert fs.keep_strongest(["b", "a", "c"], np.array([1.0, 1.0, 2.0]), 2) == ["c", "a"]


def test_market_tripwire_skips_incomplete_fields():
    market = np.array([0.6, 0.4, 0.5, np.nan])
    loss = fs._market_race_loss(market, np.array([1, 0, 0, 1]), np.array([2, 2]))
    assert loss[0] == pytest.approx(-np.log(0.6))
    assert np.isnan(loss[1])


def _synthetic(n_races=400, runners=8, n_noise=12, seed=4):
    """Races whose winner depends on three inputs; the rest are noise."""
    rng = np.random.default_rng(seed)
    n = n_races * runners
    informative = rng.normal(size=(n, 3))
    noise = rng.normal(size=(n, n_noise))
    strength = informative @ np.array([1.2, -0.9, 0.7])
    race = np.repeat(np.arange(n_races), runners)
    gumbel = -np.log(-np.log(rng.random(n)))
    score = pd.Series(strength + gumbel)
    won = (score.groupby(race).transform("max") == score).to_numpy().astype(int)
    names = ["s1", "s2", "s3"] + [f"noise_{i}" for i in range(n_noise)]
    matrix = np.column_stack([informative, noise]).astype(np.float32)
    dates = pd.DatetimeIndex(pd.Timestamp("2024-01-01") + pd.to_timedelta(race // 4, unit="D"))
    targets = fs.Targets(won=won, soft_y=None, race_ids=np.array([f"r{r}" for r in race]),
                         race_dates=dates, market=None)
    rows = np.arange(n)
    folds = [fs.FoldData("f1", rows[race < 250], rows[(race >= 260) & (race < 330)]),
             fs.FoldData("f2", rows[race < 320], rows[race >= 330])]
    index = {name: i for i, name in enumerate(names)}
    take = lambda r, cols: np.ascontiguousarray(matrix[np.ix_(r, [index[c] for c in cols])])  # noqa: E731
    return names, take, folds, targets


@pytest.fixture
def small_learner(monkeypatch):
    monkeypatch.setattr(fs, "SCREEN_PARAMS", {"n_estimators": 60, "learning_rate": 0.1,
                                              "min_child_samples": 20})


def test_elimination_keeps_the_informative_inputs(small_learner):
    names, take, folds, targets = _synthetic()
    steps = fs.run_elimination(take, names, folds, targets, n_min=3, n_jobs=1)
    assert [len(s["features"]) for s in steps] == fs.elimination_schedule(len(names), n_min=3)
    assert sorted(steps[-1]["features"]) == ["s1", "s2", "s3"]
    first = steps[0]
    assert sorted(fs.keep_strongest(first["features"], first["importance"], 3)) == ["s1", "s2", "s3"]
    # Informative inputs beat a uniform guess over eight runners out of sample.
    assert steps[-1]["pooled"]["race_log_loss"] < np.log(8) - 0.1
    assert first["races"].race_id.tolist() == steps[-1]["races"].race_id.tolist()


def test_elimination_resumes_without_refitting(small_learner, monkeypatch):
    names, take, folds, targets = _synthetic()
    stored = {}
    fs.run_elimination(take, names, folds, targets, n_min=8, n_jobs=1,
                       on_step=lambda i, r: stored.__setitem__(i, r))

    def refuse(*_a, **_k):
        raise AssertionError("a stored step was refitted")

    monkeypatch.setattr(fs, "evaluate_subset", refuse)
    again = fs.run_elimination(take, names, folds, targets, n_min=8, n_jobs=1, resume=stored.get)
    assert [s["features"] for s in again] == [stored[i]["features"] for i in sorted(stored)]


def test_native_capacity_early_stops_then_refits_on_every_row(monkeypatch):
    import config
    monkeypatch.setattr(config, "CLASSIFIER_PARAMS", {**config.CLASSIFIER_PARAMS,
                                                      "n_estimators": 80, "learning_rate": 0.1})
    monkeypatch.setattr(config, "EARLY_STOPPING_ROUNDS", 5)
    names, take, folds, targets = _synthetic()
    rows = folds[1].train_rows
    x = take(rows, names)
    model = fs.fit_win_model(x, targets.won[rows], None, np.ones(len(rows)), targets.race_ids[rows],
                             targets.race_dates[rows], capacity="native", n_jobs=1)
    assert model.n_estimators <= 80
    assert model.predict_proba(x).shape == (len(rows), 2)


def test_near_copy_filter_keeps_one_of_each_cluster_and_prefers_named_inputs():
    rng = np.random.default_rng(3)
    a, b = rng.normal(size=(2, 4000))
    monotone = np.exp(a)                       # same ranks as a
    noisy = a + rng.normal(scale=.05, size=4000)
    sparse = np.where(rng.random(4000) < .6, np.nan, a)   # mostly missing: not a copy
    matrix = np.column_stack([a, monotone, b, noisy, -a, sparse])
    names = ["a", "a_exp", "b", "a_noisy", "a_negated", "a_sparse"]
    kept, reasons = fs.near_copy_filter(matrix, names, threshold=.95)
    assert kept == ["a", "b", "a_sparse"]
    assert set(reasons) == {"a_exp", "a_noisy", "a_negated"} and all("near-copy of a " in r for r in reasons.values())
    kept, reasons = fs.near_copy_filter(matrix, names, threshold=.95, prefer=["a_noisy"])
    assert kept == ["b", "a_noisy", "a_sparse"] and "near-copy of a_noisy" in reasons["a"]
    # Chosen from the given rows only.
    early = np.arange(4000) < 2000
    late_only = np.where(early, 0., a)
    kept, _ = fs.near_copy_filter(np.column_stack([a, late_only]), ["a", "late_copy"], rows=early)
    assert kept == ["a", "late_copy"]


def test_permutation_importance_finds_what_the_forecast_needs():
    rng = np.random.default_rng(5)
    groups = np.full(300, 6)
    x = rng.normal(size=(1800, 3))
    strength = 1.5 * x[:, 0]
    race = np.repeat(np.arange(300), 6)
    score = pd.Series(strength + rng.gumbel(size=1800))
    won = (score.groupby(race).transform("max") == score).to_numpy().astype(float)

    def predict(m):
        # A forecaster that uses column 0 properly, leans on column 1 for nothing useful, ignores column 2.
        raw = np.exp(1.5 * m[:, 0] + 0.0 * m[:, 1])
        return fs.race_normalise(raw, groups)

    importance = fs.permutation_importance(predict, x, won, groups, seed=1)
    assert importance[0] > 0.2 and importance[1] == 0 and importance[2] == 0
    np.testing.assert_array_equal(importance, fs.permutation_importance(predict, x, won, groups, seed=1))
