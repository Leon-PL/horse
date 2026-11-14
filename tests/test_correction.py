"""The forecast correction: inputs read earlier dates only, and the history refuses in-sample forecasts."""
import numpy as np
import pandas as pd
import pytest

from src.correction import history as fh
from src.correction import inputs as ci
from src.correction import model as cm


def make_frame(days=60, races_per_day=4, runners=6, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    start = pd.Timestamp("2024-01-01")
    for d in range(days):
        for r in range(races_per_day):
            horses = rng.choice(120, runners, replace=False)
            strength = rng.normal(size=runners)
            p = np.exp(strength) / np.exp(strength).sum()
            market = np.exp(strength + rng.normal(scale=.3, size=runners))
            market /= market.sum()
            order = np.argsort(-(strength + rng.gumbel(size=runners)))
            position = np.empty(runners, int)
            position[order] = np.arange(1, runners + 1)
            for i in range(runners):
                rows.append({"race_id": f"{d:03d}{r}", "horse_id": int(horses[i]), "race_date": start + pd.Timedelta(days=d),
                             "won": float(position[i] == 1), "finish_position": position[i], "p_base": p[i],
                             "p_market": market[i], "trainer": f"t{horses[i] % 9}", "jockey": f"j{rng.integers(12)}",
                             "sire": f"s{horses[i] % 7}", "damsire": f"d{horses[i] % 5}", "track": f"k{r % 3}",
                             "going": "Good", "race_type": "Flat" if r % 2 else "Hurdle",
                             "days_since_last_run": float(rng.integers(0, 90)), "handicap": r % 2, "surface": "Turf",
                             "distance_furlongs": 8. + r, "form": "12P3", "hidden_form_runs": float(i % 3)})
    frame = pd.DataFrame(rows)
    prices = pd.DataFrame({"bf_bsp": 1 / frame.p_market, "ltp_0800": 1 / frame.p_base,
                           "ltp_1100": 1 / frame.p_base, "ltp_3600s": 1 / frame.p_market})
    return frame, prices


GROUPS = tuple(g for g in cm.GROUPS if g != "tracking")


def build(frame, prices):
    return cm.build_inputs(frame, prices=prices, groups=GROUPS)


def test_inputs_do_not_see_the_day_they_describe():
    frame, prices = make_frame()
    values, columns = build(frame, prices)
    day = frame.race_date.eq(frame.race_date.unique()[40]).to_numpy()
    changed, changed_prices = frame.copy(), prices.copy()
    # Rewrite that day's results and closing prices: its own inputs must not move.
    changed.loc[day, "won"] = 1 - changed.loc[day, "won"]
    changed.loc[day, "finish_position"] = 7 - changed.loc[day, "finish_position"]
    changed.loc[day, "p_market"] = changed.loc[day, "p_market"].to_numpy()[::-1]
    changed_prices.loc[day, "bf_bsp"] = changed_prices.loc[day, "bf_bsp"].to_numpy()[::-1]
    again, _ = build(changed, changed_prices)
    earlier = (frame.race_date <= frame.race_date.unique()[40]).to_numpy()
    pd.testing.assert_frame_equal(values.loc[earlier, columns], again.loc[earlier, columns])
    later = ~earlier
    assert not values.loc[later, columns].equals(again.loc[later, columns])  # and later days do use it


def test_unsettled_card_gets_inputs_and_teaches_nothing():
    frame, prices = make_frame()
    last = frame.race_date.eq(frame.race_date.max()).to_numpy()
    values, columns = build(frame, prices)
    card, card_prices = frame.copy(), prices.copy()
    card.loc[last, ["won", "finish_position", "p_market"]] = np.nan
    card_prices.loc[last] = np.nan
    again, _ = build(card, card_prices)
    pd.testing.assert_frame_equal(values[columns], again[columns])
    assert again.loc[last, columns].notna().any().any()


def test_history_runs_without_a_forecast_are_kept_but_not_scored():
    frame, prices = make_frame()
    early = (frame.race_date < frame.race_date.unique()[10]).to_numpy()
    frame.loc[early, "p_base"] = np.nan
    values, columns = build(frame, prices)
    assert values.loc[early, columns].isna().all().all()
    assert values.loc[~early, columns].notna().any().any()


def test_partial_field_forecast_is_refused():
    frame, prices = make_frame(days=5)
    frame.loc[0, "p_base"] = np.nan
    with pytest.raises(ValueError, match="only part of its field"):
        build(frame, prices)


def test_model_corrects_a_biased_forecast():
    frame, prices = make_frame(days=120, seed=3)
    rng = np.random.default_rng(1)
    signal = rng.normal(size=len(frame))
    # Results follow the forecast tilted by the signal; the correction should find the tilt.
    bounds = ci.bounds_of(frame.race_id.to_numpy())
    truth = np.exp(cm.race_log_softmax(np.log(frame.p_base.to_numpy()) + .8 * signal, bounds))
    won = np.zeros(len(frame))
    for a, b in zip(bounds[:-1], bounds[1:]):
        won[a + rng.choice(b - a, p=truth[a:b])] = 1.
    frame["won"], frame["p_market"] = won, truth
    values = frame[ci.KEYS].assign(signal=signal)
    train = (frame.race_date < frame.race_date.unique()[90]).to_numpy()
    params = {"min_data_in_leaf": 20, "lambda_l2": 1., "learning_rate": .1}
    model = cm.CorrectionModel(["signal"]).fit(frame.loc[train], values.loc[train], rounds=60, params=params,
                                               seeds=(7,), threads=1)
    test, test_values = frame.loc[~train].reset_index(drop=True), values.loc[~train].reset_index(drop=True)
    p = model.predict(test, test_values)
    assert np.allclose(pd.Series(p).groupby(test.race_id.to_numpy()).sum(), 1.)
    result = cm.compare(test, model.adjustment(test_values))
    assert result["corrected_log_loss"] < result["forecast_log_loss"] - .05


def records(dates, cutoff, signature="r1"):
    rows = []
    for k, date in enumerate(dates):
        for horse in (1, 2):
            rows.append({"race_id": f"x{k}", "horse_id": horse, "race_date": pd.Timestamp(date), "probability": .5,
                         "component": "generalist", "fold": "2024Q1", "recipe_signature": signature,
                         "training_last_date": cutoff})
    return pd.DataFrame(rows)


def test_history_accepts_only_out_of_sample_forecasts(tmp_path):
    bench = tmp_path / "bench"
    bench.mkdir()
    records(["2024-02-01", "2024-02-02"], "2024-01-20").to_parquet(bench / "oof.parquet")
    root = tmp_path / "history"
    fh.seed(bench, root=root)
    assert fh.status(root)["races"] == 2
    with pytest.raises(ValueError, match="already exists"):
        fh.seed(bench, root=root)
    fresh = records(["2024-03-01"], "2024-02-10").assign(race_id="y0")[["race_id", "horse_id", "race_date", "probability"]]
    with pytest.raises(ValueError, match="not out of sample"):
        fh.append(fresh, source="run:a", recipe_signature="r1", training_last_date="2024-03-01", root=root)
    with pytest.raises(ValueError, match="different recipe"):
        fh.append(fresh, source="run:a", recipe_signature="r2", training_last_date="2024-02-10", root=root)
    assert fh.append(fresh, source="run:a", recipe_signature="r1", training_last_date="2024-02-10", root=root) == 2
    assert fh.append(fresh, source="run:a", recipe_signature="r1", training_last_date="2024-02-10", root=root) == 0
    half = fresh.iloc[:1].assign(race_id="z0")
    with pytest.raises(ValueError, match="whole field"):
        fh.append(half, source="run:a", recipe_signature="r1", training_last_date="2024-02-10", root=root)
    assert fh.status(root)["last_date"] == "2024-03-01"


def test_walk_forward_accepts_inputs_also_present_in_the_frame():
    frame, prices = make_frame(days=400, races_per_day=3)
    frame["extra"] = np.arange(len(frame), dtype=float)          # a tracking column lives in both
    values = frame[ci.KEYS].assign(extra=frame.extra)
    adjustment, labels = cm.walk_forward(frame, values, ["extra"], first_scored="2024-10-01", rounds=5,
                                         params={"min_data_in_leaf": 20}, seeds=(7,), threads=1)
    scored = frame.race_date.ge("2024-10-01").to_numpy()
    assert np.isfinite(adjustment[scored]).all() and np.isnan(adjustment[~scored]).all()
    assert cm.compare(frame, adjustment)["races"] == frame.loc[scored].race_id.nunique()


def test_card_path_matches_the_same_day_scored_as_history():
    from src.correction import serve
    frame, prices = make_frame(days=80, seed=5)
    last = frame.race_date.eq(frame.race_date.max()).to_numpy()
    values, columns = build(frame, prices)
    model = cm.CorrectionModel(columns).fit(frame.loc[~last], values.loc[~last], rounds=20,
                                            params={"min_data_in_leaf": 20}, seeds=(7,), threads=1)
    expected = model.predict(frame.loc[last].reset_index(drop=True), values.loc[last].reset_index(drop=True))
    # The same day as a card: placeholder results (as the live path gives), no prices.
    featured = frame.loc[last].drop(columns=["p_base", "p_market"]).assign(won=0, finish_position=0, horse_name="x")
    card = serve.card_rows(featured, frame.loc[last, "p_base"].to_numpy(),
                           [c for c in featured.columns if c not in ("won", "finish_position")])
    settled = frame.loc[~last].reset_index(drop=True).assign(horse_name="x")
    original = cm.build_inputs
    cm.build_inputs = lambda f, prices, tracking, only=None: original(f, prices=prices, groups=GROUPS, only=only)
    try:
        out = serve.correct_card(settled, prices.loc[~last].reset_index(drop=True).assign(
            race_id=settled.race_id, horse_id=settled.horse_id), card, model, ())
    finally:
        cm.build_inputs = original
    assert np.allclose(out.probability.to_numpy(), expected)
    with pytest.raises(ValueError, match="dated after"):
        serve.correct_card(frame.reset_index(drop=True), prices, card, model, ())
