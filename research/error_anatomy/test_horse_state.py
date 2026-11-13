import numpy as np
import pandas as pd

from research.error_anatomy import horse_state as hs


def _frame(seed=0, races=60, field=5, horses=12):
    rng = np.random.default_rng(seed)
    rows = []
    for race in range(races):
        date = pd.Timestamp("2024-01-01") + pd.Timedelta(days=race // 2)  # two races a day
        # A horse can be drawn in both of a day's races, which exercises the same-date rule.
        runners = rng.choice(horses, size=field, replace=False)
        p = rng.dirichlet(np.ones(field))
        market = rng.dirichlet(np.ones(field))
        order = rng.permutation(field) + 1
        for i, horse in enumerate(runners):
            rows.append({"race_id": f"r{race:03d}", "horse_id": f"h{horse}", "race_date": date,
                         "p_single": p[i], "p_market": market[i], "finish_position": order[i],
                         "won": float(order[i] == 1), "trainer": f"t{horse % 4}", "jockey": f"j{horse % 3}",
                         "sire": f"s{horse % 5}", "race_type": "Flat" if race % 2 else "Hurdle",
                         "surface": "Turf", "distance_furlongs": 8. + race % 3, "days_since_last_run": 10.})
    return pd.DataFrame(rows)


def _equal(a, b, rows):
    columns = [c for c in a if c not in hs.KEYS]
    np.testing.assert_allclose(a.loc[rows, columns].to_numpy(float), b.loc[rows, columns].to_numpy(float),
                               equal_nan=True, rtol=1e-6)


def test_first_lag_is_the_previous_runs_centred_gap():
    frame = _frame()
    out = hs.horse_state(frame)
    signal = hs.signals(frame)
    horse = frame.loc[frame.horse_id.eq("h3")]
    dates = horse.race_date.to_numpy()
    for previous, current in zip(horse.index[:-1], horse.index[1:]):
        if frame.race_date[previous] == frame.race_date[current]:
            assert np.isnan(out.hs_gap_l1[current])
        elif (dates == frame.race_date[current]).sum() == 1 and (dates == frame.race_date[previous]).sum() == 1:
            assert np.isclose(out.hs_gap_l1[current], signal.gap[previous], atol=1e-6)


def test_state_ignores_the_race_itself_and_everything_after():
    frame = _frame(1)
    cut = pd.Timestamp("2024-01-20")
    changed = frame.copy()
    later = changed.race_date >= cut
    rng = np.random.default_rng(9)
    changed.loc[later, "p_market"] = rng.uniform(.05, .5, later.sum())
    changed.loc[later, "finish_position"] = changed.loc[later].groupby("race_id").finish_position.transform(
        lambda v: v.to_numpy()[::-1])
    changed.loc[later, "won"] = (changed.loc[later, "finish_position"] == 1).astype(float)
    on_cut = frame.race_date.eq(cut)
    assert on_cut.any()
    _equal(hs.horse_state(frame), hs.horse_state(changed), on_cut)
    _equal(hs.entity_recent(frame), hs.entity_recent(changed), on_cut)


def test_moving_average_weights_recent_runs_more():
    group = np.zeros(4, dtype=np.int64)
    out = hs._moving(group, np.array([1., 0., np.nan, 5.]), .5)
    assert np.isnan(out[0]) and out[1] == 1.
    assert np.isclose(out[2], (.5 * 1 + 0) / 1.5)
    assert np.isclose(out[3], (.25 * 1) / .75)


def test_market_history_uses_earlier_dates_only():
    from research.error_anatomy import market_history as mh
    frame = _frame(2)
    frame["num_runners"] = 5
    rng = np.random.default_rng(3)
    prices = pd.DataFrame({"bf_bsp": rng.uniform(2, 30, len(frame)), "ltp_0800": rng.uniform(2, 30, len(frame)),
                           "ltp_1100": rng.uniform(2, 30, len(frame)), "ltp_3600s": rng.uniform(2, 30, len(frame))})
    cut = pd.Timestamp("2024-01-20")
    later = (frame.race_date >= cut).to_numpy()
    changed, changed_prices = frame.copy(), prices.copy()
    changed.loc[later, "p_market"] = rng.uniform(.05, .5, later.sum())
    changed_prices.loc[later] = rng.uniform(2, 30, (later.sum(), 4))
    on_cut = frame.race_date.eq(cut)
    _equal(mh.market_level(frame), mh.market_level(changed), on_cut)
    lag = lambda f, p: mh._lagged(f, mh.moves(f, p), 3, "mh", relative=("move_1100", "move_3600s"))
    _equal(lag(frame, prices), lag(frame, changed_prices), on_cut)
    first = mh.market_level(frame)
    horse = frame.loc[frame.horse_id.eq("h5")]
    share = np.log(frame.p_market * 5)
    for previous, current in zip(horse.index[:-1], horse.index[1:]):
        if frame.race_date[previous] < frame.race_date[current] and (horse.race_date == frame.race_date[previous]).sum() == 1:
            assert np.isclose(first.mh_share_l1[current], share[previous], atol=1e-6)


def test_moves_are_centred_and_signed():
    from research.error_anatomy import market_history as mh
    frame = pd.DataFrame({"race_id": ["a", "a", "a"]})
    prices = pd.DataFrame({"bf_bsp": [2., 10., 10.], "ltp_0800": [4., 10., 10.], "ltp_1100": [4., 10., 10.],
                           "ltp_3600s": [2., 10., 10.]})
    out = mh.moves(frame, prices)
    assert out.move_1100[0] > 0 > out.move_1100[1]          # the first horse shortened
    assert np.isclose(out.move_1100.sum(), 0.) and np.allclose(out.move_3600s, 0.)


def test_hidden_form_reads_only_the_oldest_figures():
    from research.error_anatomy import hidden_form as hf
    out = hf.figures(["1/P2-34", "21", None, "0F"], [3, 0, 2, 2])
    assert out[0].tolist() == [1., 2., 1., 1., 1.5, 2., 3.]      # "1", "P", "2": the later 3 and 4 are database runs
    assert np.isnan(out[1]).all() and np.isnan(out[2]).all()
    assert out[3, 3] == 10. and out[3, 2] == 1. and out[3, 5] == 12.


def test_store_hidden_form_matches_the_screened_definition():
    from research.error_anatomy import hidden_form as hf
    from src.feature_engineer import hidden_form_figures
    forms = np.array(["1/P2-34", "21", None, "0F", "U11", "7-", "PPP000"], dtype=object)
    hidden = np.array([3, 0, 2, 2, 2, 1, 6])
    np.testing.assert_allclose(hidden_form_figures(forms, hidden), hf.figures(forms, hidden)[:, :6], equal_nan=True)


def test_owner_key_ignores_titles_and_punctuation():
    from research.error_anatomy.owner import owner_key
    out = owner_key(["Mr J. P. McManus", "J P McManus", "Mrs Sue Magnier & Mr M. Tabor", "  ", None])
    assert out[0] == out[1] == "j p mcmanus"
    assert out[2] == "sue magnier m tabor"
    assert out[3:].isna().all()


def test_early_score_takes_the_first_style_phrase():
    from research.error_anatomy.pace_comments import early_score
    assert early_score("led, ridden and headed over 1f out, kept on one pace") == 1.
    assert early_score("held up in rear, headway to lead 1f out") == .25
    assert early_score("chased leader, led 4 out, headed flat") == .75
    assert early_score("always behind") == 0.
    assert np.isnan(early_score("ran on well")) and np.isnan(early_score(None))


def test_going_index_blends_in_places():
    from research.error_anatomy.matched_form import going_index
    assert going_index("Good") == 3. and going_index("Heavy") == 7.
    assert going_index("Good to Firm (Good in places)") == .75 * 2 + .25 * 3
    assert going_index("Standard / Slow") == 3.5
    assert np.isnan(going_index("Unknown")) and np.isnan(going_index(None))


def test_matched_form_uses_earlier_dates_and_weights_similar_runs():
    from research.error_anatomy import matched_form as mf
    horse = np.zeros(4, dtype=np.int64)
    day = np.array([1, 2, 3, 3], dtype=np.int64)
    dist = np.array([6., 12., 6., 6.])
    going = np.array([3., 3., 3., 3.])
    ones = np.ones(4, dtype=np.int64)
    klass = np.full(4, 4.)
    beat = np.array([1., 0., .5, .5])
    means, weights = mf._matched(horse, day, dist, going, ones, ones, klass, ones, beat, beat - .5, .85)
    assert np.isnan(means[0]).all() and weights[0].sum() == 0
    assert np.allclose(means[2], means[3])                    # the same-day run is not history
    assert np.isclose(means[2, 0], (0. * 1 + 1. * .85) / 1.85)  # unmatched: recency only
    assert means[2, 1] > .95                                  # distance-matched: the 6f run dominates
