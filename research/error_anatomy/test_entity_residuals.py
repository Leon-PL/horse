import numpy as np
import pandas as pd
import pytest

from research.error_anatomy.entity_residuals import (HORSE, HORSE_SHRINK, SHRINK, columns_of, entity_inputs,
                                                     run_residuals)

SHRINK_ORDER = SHRINK[1]


def reverse_results(frame, rows):
    out = frame.copy()
    out.loc[rows, "finish_position"] = out.loc[rows].groupby("race_id").finish_position.transform(
        lambda x: x.to_numpy()[::-1])
    out["won"] = out.finish_position.eq(1).astype(int)
    return out


def test_decayed_market_gap_and_horse_inputs_see_only_earlier_dates():
    frame = history(days=8)
    frame["horse_id"] = frame.groupby(["race_id"]).cumcount() + 10 * (frame.race_type == "Hurdle")
    rng = np.random.default_rng(3)
    frame["p_market"] = rng.dirichlet(np.ones(4), size=len(frame) // 4).ravel()
    options = dict(market="p_market", half_life=2., entities={**HORSE, "trainer": ["trainer"]}, shrink=HORSE_SHRINK)
    before = entity_inputs(frame, **options)
    assert {"mg_horse_gap", "mg_trainer_gap_vs_field", "er_horse_order"} <= set(before)
    last = frame.race_date.eq(frame.race_date.max()).to_numpy()
    changed = reverse_results(frame, last)
    changed.loc[last, "p_market"] = changed.loc[last, "p_market"].to_numpy()[::-1]
    pd.testing.assert_frame_equal(before, entity_inputs(changed, **options))
    # A recent run outweighs an old one under decay: totals differ from the undecayed version.
    plain = entity_inputs(frame, **{**options, "half_life": None})
    assert not np.allclose(plain.er_horse_order, before.er_horse_order)
    assert before.loc[last, "er_horse_runs"].max() < plain.loc[last, "er_horse_runs"].max()


def history(days=6, runners=4, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for day in range(days):
        for race in range(2):
            order = rng.permutation(runners) + 1
            for k in range(runners):
                rows.append({"race_id": f"{day}-{race}", "horse_id": 100 * day + 10 * race + k,
                             "race_date": pd.Timestamp("2025-01-01") + pd.Timedelta(days=day),
                             "trainer": f"T{k % 2}", "jockey": f"J{k}", "sire": f"S{k % 3}",
                             "race_type": "Hurdle" if race else "Flat", "days_since_last_run": 10 * k,
                             "finish_position": order[k], "won": int(order[k] == 1), "p_single": 1 / runners})
    return pd.DataFrame(rows)


def test_run_residuals_are_zero_sum_and_rank_the_winner_highest():
    frame = history()
    residual = run_residuals(frame)
    by_race = residual.groupby(frame.race_id)
    assert by_race.win.sum().abs().max() == pytest.approx(0, abs=1e-12)
    assert by_race.order.sum().abs().max() == pytest.approx(0, abs=1e-12)
    # With a uniform forecast the winner beat every rival against an expectation of half.
    assert residual.order[frame.won.eq(1)].to_numpy() == pytest.approx(.5)


def test_non_finishers_rank_behind_finishers_and_level_with_each_other():
    frame = history(days=1).query("race_id == '0-0'").reset_index(drop=True)
    frame["finish_position"] = [1, 2, 0, 0]
    order = run_residuals(frame).order.to_numpy()
    assert order[2] == pytest.approx(order[3])
    assert order[1] > order[2]


def test_inputs_never_see_the_same_day_or_later_results():
    frame = history()
    before = entity_inputs(frame)
    changed = frame.copy()
    last = changed.race_date.eq(changed.race_date.max()).to_numpy()
    changed.loc[last, "finish_position"] = changed.loc[last].groupby("race_id").finish_position.transform(
        lambda x: x.to_numpy()[::-1])
    changed["won"] = changed.finish_position.eq(1).astype(int)
    after = entity_inputs(changed)
    pd.testing.assert_frame_equal(before, after)
    earlier = frame.copy()
    first = earlier.race_date.eq(earlier.race_date.min()).to_numpy()
    earlier.loc[first, "finish_position"] = earlier.loc[first].groupby("race_id").finish_position.transform(
        lambda x: x.to_numpy()[::-1])
    earlier["won"] = earlier.finish_position.eq(1).astype(int)
    moved = entity_inputs(earlier)
    assert not moved[columns_of(moved)].equals(before[columns_of(before)])
    assert moved.loc[first, columns_of(moved)].equals(before.loc[first, columns_of(before)])


def test_first_date_has_no_history_and_shrinkage_matches_the_definition():
    frame = history()
    values = entity_inputs(frame)
    first = frame.race_date.eq(frame.race_date.min()).to_numpy()
    assert values.loc[first, "er_trainer_runs"].eq(0).all()
    assert values.loc[first, "er_trainer_order"].eq(0).all()
    second = frame.race_date.eq(frame.race_date.min() + pd.Timedelta(days=1)).to_numpy()
    residual = run_residuals(frame)
    expected = residual.order[first & frame.trainer.eq("T0")].sum() / (first & frame.trainer.eq("T0")).sum()
    got = values.loc[second & frame.trainer.eq("T0"), "er_trainer_order"].iloc[0]
    runs = (first & frame.trainer.eq("T0")).sum()
    assert got == pytest.approx(expected * runs / (runs + SHRINK_ORDER), rel=1e-5)
