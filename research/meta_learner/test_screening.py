import numpy as np
import pandas as pd
import pytest

from research.meta_learner import screening as s


def synthetic(effect=0.0, races_per_quarter=250, runners=8, seed=0):
    """Eight quarters of fields whose true scores are offset + effect * x."""
    rng = np.random.default_rng(seed)
    rows = []
    dates = pd.date_range("2024-07-01", "2026-06-30", freq="D")
    quarters = pd.Series(dates).dt.to_period("Q")
    race = 0
    for quarter in quarters.unique():
        days = dates[(quarters == quarter).to_numpy()]
        for day in np.sort(rng.choice(days, races_per_quarter)):
            base = rng.normal(0, 1, runners)
            x = rng.integers(0, 2, runners).astype(float)
            true = base + effect * x
            p_true = np.exp(true - true.max())
            p_true /= p_true.sum()
            winner = rng.choice(runners, p=p_true)
            p_base = np.exp(base - base.max())
            p_base /= p_base.sum()
            for i in range(runners):
                rows.append({"race_id": race, "horse_id": race * 100 + i, "horse_name": f"h{race}-{i}",
                             "race_date": pd.Timestamp(day), "num_runners": runners,
                             "finish_position": 1 if i == winner else 2, "won": int(i == winner),
                             "p_single": p_base[i], "p_average": p_base[i], "x": x[i],
                             "noise": rng.normal()})
            race += 1
    frame = pd.DataFrame(rows).sort_values(["race_date", "race_id", "horse_id"], kind="stable").reset_index(drop=True)
    frame["quarter"] = frame.race_date.dt.to_period("Q").astype(str)
    return frame


def test_basis_handles_binary_constant_and_missing_columns():
    x = np.array([[0., 5., np.nan], [1., 5., 1.], [0., 5., 2.], [1., 5., 3.]])
    basis = s.Basis.fit(x, ["binary", "constant", "gappy"], bins=5)
    design = basis.transform(x)
    assert basis.edges[1].size == 0
    assert design.shape == (4, basis.width)
    assert design[:, 0].tolist() == [0, 1, 0, 1]
    assert design[0, -1] == 1 and design[1:, -1].sum() == 0


def test_offset_logit_recovers_a_planted_effect():
    frame = synthetic(effect=0.8)
    bounds = s.bounds_of(frame.race_id.to_numpy())
    beta, info = s.fit_offset_logit(frame[["x"]].to_numpy(), np.log(frame.p_single.to_numpy()),
                                    frame.won.to_numpy(float), bounds, l2=0)
    assert info["converged"]
    assert beta[0] == pytest.approx(0.8, abs=0.12)


def test_walk_forward_fits_only_on_earlier_purged_quarters():
    frame = synthetic()
    seen = []
    def fit_predict(train, test):
        seen.append((train.race_date.max(), test.race_date.min(), test.quarter.iloc[0]))
        return np.zeros(len(test))
    s.walk_forward(frame, fit_predict)
    assert [q for *_, q in seen] == list(s.SCORED)
    assert all(last <= first - pd.Timedelta(days=s.PURGE_DAYS) for last, first, _ in seen)


def test_zero_correction_scores_exactly_zero():
    frame = synthetic()
    bases = ("p_single", "p_average")
    result = s.summarise(s.race_table(frame, np.zeros(len(frame))), bases)
    assert result["p_average"]["mean"] == 0 and result["p_single"]["mean"] == 0


def test_candidate_against_control_is_paired_on_identical_races():
    frame = synthetic(effect=0.6)
    bases = ("p_single", "p_average")
    _, control = s.run(frame, ["noise"], tier=0)
    _, candidate = s.run(frame, ["noise", "x"], tier=0)
    change = s.versus(candidate, control, bases)
    assert change["p_average"]["ci95"][1] < 0


def test_planted_effect_is_found_and_noise_is_not():
    frame = synthetic(effect=0.6)
    found, _ = s.run(frame, ["x"], tier=0)
    noise, _ = s.run(frame, ["noise"], tier=0)
    assert found["p_average"]["ci95"][1] < 0
    assert abs(noise["p_average"]["mean"]) < 0.002


def test_permuted_twin_keeps_each_quarter_and_breaks_the_runner_link():
    frame = synthetic(effect=0.6)
    frame["y"] = frame.x * 2
    twin = s.permute_within(frame, ["x", "y"], seed=3)
    for column in ("x", "y"):
        assert (twin.groupby("quarter")[column].apply(sorted) == frame.groupby("quarter")[column].apply(sorted)).all()
    assert (twin.y == twin.x * 2).all()  # the group moves together
    assert (twin.x != frame.x).mean() > 0.3
    assert twin.drop(columns=["x", "y"]).equals(frame.drop(columns=["x", "y"]))


def test_fixed_round_boost_beats_its_twin_on_a_planted_effect():
    frame = synthetic(effect=0.6)
    options = {"params": {"min_data_in_leaf": 50, "num_leaves": 3}, "boost": {"rounds": 60, "stopping": None,
                                                                               "seeds": (1, 2)}}
    found, races = s.run(frame, ["x"], tier=1, **options)
    _, twin = s.run(s.permute_within(frame, ["x"], seed=5), ["x"], tier=1, **options)
    assert found["fits"][0]["boosters"] == 2
    assert s.versus(races, twin, ("p_average",))["p_average"]["ci95"][1] < 0


def test_residual_boost_finds_effect_and_drops_noise():
    frame = synthetic(effect=0.6)
    found, _ = s.run(frame, ["x"], tier=1, params={"min_data_in_leaf": 50})
    noise, _ = s.run(frame, ["noise"], tier=1, params={"min_data_in_leaf": 50})
    assert found["p_average"]["mean"] < -0.01
    assert noise["p_average"]["mean"] < 0.003
    assert all(f["trees"] >= 0 for f in noise["fits"])
