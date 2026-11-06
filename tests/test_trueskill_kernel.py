"""The compiled TrueSkill sweep must agree with the trueskill library to the bit."""
import numpy as np
import pandas as pd
import pytest
import trueskill

from src import ratings


def library_pass(keys, fp, dates, race_ids, groups, *, state=None):
    """The sweep written against ``env.rate``: the reference for the kernel."""
    env = trueskill.TrueSkill(
        mu=ratings.TRUESKILL_MU, sigma=ratings.TRUESKILL_SIGMA, beta=ratings.TRUESKILL_BETA,
        tau=ratings.TRUESKILL_TAU, draw_probability=0.001,
    )
    state = state or {}
    current = dict(state.get("ratings", {}))
    last_dt = dict(state.get("last_dt", {}))
    n = len(keys)
    mu_col, sigma_col = np.full(n, np.nan), np.full(n, np.nan)
    has_col = np.zeros(n, dtype=np.int8)
    refused = 0
    for race_id in race_ids:
        idx = groups[race_id]
        race_dt = dates[idx[0]]
        for i in idx:
            k = keys[i]
            if k is None:
                continue
            rating = current.get(k)
            if rating is not None:
                months = max(float((race_dt - last_dt[k]) / np.timedelta64(1, "D")), 0.0) / 30.44
                mu_col[i] = rating[0]
                sigma_col[i] = min(float(np.sqrt(rating[1] ** 2 + ratings.TRUESKILL_C ** 2 * months)),
                                   ratings.TRUESKILL_SIGMA)
                has_col[i] = 1
            else:
                mu_col[i] = ratings.TRUESKILL_MU
                sigma_col[i] = ratings.TRUESKILL_SIGMA
        if fp is None:
            continue
        part, pos = ratings._race_outcome(fp[idx], ratings.DNF_AS_LOSS)
        upd = [j for j in range(len(idx)) if part[j] and keys[idx[j]] is not None]
        if len(upd) < 2:
            continue
        rows = idx[upd]
        field = [(env.create_rating(mu_col[i], sigma_col[i]),) for i in rows]
        try:
            new = env.rate(field, ranks=[int(pos[j]) for j in upd])
        except (FloatingPointError, ValueError):
            refused += 1
            continue
        for i, (rating,) in zip(rows, new):
            current[keys[i]] = (float(rating.mu), float(rating.sigma))
            last_dt[keys[i]] = race_dt
    return mu_col, sigma_col, has_col, {"ratings": current, "last_dt": last_dt}, refused


def random_card(rng, *, races, horses, start):
    """Fields with dead heats, non-finishers, unknown keys and repeated horses."""
    rows = []
    day = pd.Timestamp(start)
    for race in range(races):
        day += pd.Timedelta(days=int(rng.integers(0, 4)))
        size = int(rng.integers(1, 16))
        field = rng.integers(0, horses, size)
        for place, horse in enumerate(field, 1):
            roll = rng.random()
            position = float(place)
            if roll < .08:
                position = 0.0
            elif roll < .12:
                position = np.nan
            elif roll < .22:
                position = float(max(1, place - 1))  # dead heat
            elif roll < .24:
                position = place + .5
            rows.append((race, None if rng.random() < .03 else f"horse:{horse}", position, day))
    frame = pd.DataFrame(rows, columns=["race_id", "key", "finish", "race_date"])
    keys = [k if pd.notna(k) else None for k in frame["key"]]
    return (keys, frame["finish"].to_numpy(dtype=float), pd.to_datetime(frame["race_date"]).values,
            frame["race_id"].drop_duplicates().values, frame.groupby("race_id", sort=False).indices)


def assert_same(fast, reference):
    for a, b in zip(fast[:3], reference[:3]):
        assert a.dtype == b.dtype
        assert np.array_equal(a, b, equal_nan=a.dtype.kind == "f")
    assert fast[3]["ratings"] == reference[3]["ratings"]
    assert fast[3]["last_dt"] == reference[3]["last_dt"]
    assert all(type(v[0]) is float and type(v[1]) is float for v in fast[3]["ratings"].values())
    assert {type(v) for v in fast[3]["last_dt"].values()} <= {np.datetime64}


@pytest.mark.parametrize("seed", range(6))
def test_sweep_matches_the_library_bit_for_bit(seed):
    rng = np.random.default_rng(seed)
    card = random_card(rng, races=250, horses=60, start="2020-01-01")
    assert_same(ratings._trueskill_pass(*card, return_state=True), library_pass(*card))


def test_seeded_chunks_chain_like_the_library():
    rng = np.random.default_rng(21)
    first = random_card(rng, races=200, horses=50, start="2021-01-01")
    second = random_card(rng, races=200, horses=70, start="2021-09-01")
    fast = ratings._trueskill_pass(*first, return_state=True)
    reference = library_pass(*first)
    assert_same(fast, reference)
    carried = {key: dict(value) for key, value in fast[3].items()}
    assert_same(ratings._trueskill_pass(*second, state=fast[3], return_state=True),
                library_pass(*second, state=reference[3]))
    assert fast[3] == carried


def test_races_the_library_refuses_are_skipped_the_same_way():
    # A huge upset drives the win correction past what doubles can hold.
    keys = [f"horse:{i}" for i in range(6)]
    state = {
        "ratings": {key: (25.0 + 400.0 * i, 0.2) for i, key in enumerate(keys)},
        "last_dt": {key: np.datetime64("2020-01-01", "ns") for key in keys},
    }
    frame = pd.DataFrame({"race_id": [1] * 6 + [2] * 6,
                          "key": keys * 2,
                          "finish": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0]})
    card = (frame["key"].tolist(), frame["finish"].to_numpy(),
            pd.to_datetime(pd.Series(["2020-02-01"] * 12)).values,
            frame["race_id"].drop_duplicates().values, frame.groupby("race_id", sort=False).indices)
    reference = library_pass(*card, state=state)
    assert reference[4] > 0
    assert_same(ratings._trueskill_pass(*card, state=state, return_state=True), reference)


def test_pending_cards_only_read_the_ratings():
    rng = np.random.default_rng(4)
    card = random_card(rng, races=60, horses=30, start="2022-01-01")
    state = ratings._trueskill_pass(*card, return_state=True)[3]
    pending = (card[0], None, card[2], card[3], card[4])
    fast = ratings._trueskill_pass(*pending, state=state, return_state=True)
    assert_same(fast, library_pass(*pending, state=state))
    assert fast[3]["ratings"] == state["ratings"]
