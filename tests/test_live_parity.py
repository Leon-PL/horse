"""Training/serving feature parity — the golden test for live prediction.

Takes a synthetic multi-month dataset, treats the final day as a "live"
racecard (outcomes stripped exactly like the Predict page does), and runs
BOTH paths:

  training path:  process_data(all) -> engineer_features(all)
  live path:      process_data(history), process_data(cards stripped of
                  outcomes) -> feature_engineer_with_history_core

Every model feature for the final-day rows must be identical in both
paths. A mismatch means either training/serving skew (the live model
sees different inputs than it was trained on) or outcome leakage (the
feature changes depending on whether the outcome is known — impossible
for a genuinely pre-race feature).
"""

import numpy as np
import pandas as pd
import pytest
import config

from src.data_processor import process_data
from src.feature_engineer import engineer_features
from src.live_prediction import feature_engineer_with_history_core
from src.model import get_feature_columns
from src.store_columns import unpruned


@pytest.fixture(autouse=True)
def all_feature_parity(monkeypatch):
    """Validate every feature, independently of a local/private allowlist."""
    monkeypatch.setattr(config, "FEATURE_ALLOWLIST", None)


def _synthetic_raw(n_days: int = 100, races_per_day: int = 2, field: int = 6, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    horses = list(range(40))
    jockeys = [f"J{i}" for i in range(12)]
    trainers = [f"T{i}" for i in range(8)]
    tracks = ["Ascot", "York", "Chester"]
    rows = []
    rid = 0
    start = pd.Timestamp("2025-01-01")
    for d in range(n_days):
        dt = start + pd.Timedelta(days=d)
        for r in range(races_per_day):
            rid += 1
            runners = rng.choice(horses, size=field, replace=False)
            finish = rng.permutation(field) + 1
            for slot, (h, fp) in enumerate(zip(runners, finish)):
                fp = int(fp)
                rows.append({
                    "race_id": f"R{rid:05d}",
                    "race_date": dt.strftime("%Y-%m-%d"),
                    "off_time": f"{13 + r}:30",
                    "track": tracks[rid % len(tracks)],
                    "region": "UK",
                    "race_name": f"Synthetic Stakes {rid}",
                    "race_class": f"Class {1 + rid % 6}",
                    "race_type": ["Flat", "Hurdle", "Chase"][rid % 3],
                    "distance_furlongs": float(6 + rid % 5),
                    "going": ["Good", "Soft", "Heavy"][rid % 3],
                    "prize_money": 5000.0 + 100 * (rid % 10),
                    "num_runners": field,
                    "horse_name": f"Horse {h}",
                    # NUMERIC horse_id on purpose: real data keys history on an
                    # integer horse_id, and an earlier incremental-FE bug only
                    # bit numeric keys (string seed rows vs numeric frame rows
                    # grouped apart, orphaning carried history). A string id here
                    # silently hid it — keep this numeric so the parity tests
                    # actually exercise the real-data key path.
                    "horse_id": 100000 + h,
                    "jockey": jockeys[h % len(jockeys)],
                    "trainer": trainers[h % len(trainers)],
                    "age": 4 + h % 5,
                    "sex": ["G", "F", "M"][h % 3],
                    "headgear": "" if h % 4 else "Blinkers",
                    "weight_lbs": 130.0 + (h % 10),
                    "reported_jockey_claim_lbs": 0.,
                    "draw": slot + 1,
                    "form": "321",
                    "days_since_last_run": 20,
                    "odds": float(2 + fp + int(rng.integers(0, 6))),
                    "official_rating": 70 + (h % 20),
                    "finish_position": fp,
                    "won": int(fp == 1),
                    "lengths_behind": 0.0 if fp == 1 else fp * 1.5,
                    "surface": "Turf",
                    "handicap": rid % 2,
                    # Pre-populated weather so FE never calls the API
                    "weather_temp_max": 15.0 + (d % 10),
                    "weather_temp_min": 6.0 + (d % 5),
                    "weather_precip_mm": float(d % 4),
                    "weather_wind_kmh": 12.0 + (d % 15),
                    "weather_precip_prev3": float(d % 7),
                })
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def parity_frames():
    raw = _synthetic_raw()
    last_date = raw["race_date"].max()
    last_day_races = raw.loc[raw["race_date"].eq(last_date), "race_id"].unique()
    raw.loc[raw["race_id"].eq(last_day_races[0]), "race_name"] = "Handicap (Conditional & Amateur Jockeys)"
    raw.loc[raw["race_id"].eq(last_day_races[1]), "race_name"] = "Apprentice Handicap"
    mixed = raw.index[raw["race_id"].eq(last_day_races[0])]
    apprentice = raw.index[raw["race_id"].eq(last_day_races[1])]
    raw.loc[mixed, "reported_jockey_claim_lbs"] = np.nan
    raw.loc[mixed[0], "reported_jockey_claim_lbs"] = -2
    raw.loc[apprentice[0], "reported_jockey_claim_lbs"] = 7
    horse = raw.loc[apprentice[0], "horse_id"]
    prior = raw.index[raw.horse_id.eq(horse) & raw.race_date.lt(last_date)][-1]
    raw.loc[prior, "reported_jockey_claim_lbs"] = 3

    # Unpruned, so parity and the rider-input checks cover columns the store drops.
    with unpruned():
        return _parity_frames(raw, last_date)


def _parity_frames(raw, last_date):
    # ── Training path: everything processed and featured together ──
    feat_train = engineer_features(process_data(raw.copy()), save=False)

    # ── Live path: history vs cards, outcomes stripped like the app ──
    hist_raw = raw[raw["race_date"] < last_date].copy()
    cards = raw[raw["race_date"] == last_date].copy()
    cards["won"] = 0
    cards["finish_position"] = 0
    cards["lengths_behind"] = np.nan

    proc_hist = process_data(hist_raw)
    proc_cards = process_data(cards)
    feat_live = feature_engineer_with_history_core(proc_hist, proc_cards)

    train_day = feat_train[feat_train["race_date"] == last_date]
    return feat_train, train_day, feat_live, last_date


def test_no_dataframe_fragmentation():
    """The FE pipeline must not fragment the DataFrame (column-by-column
    inserts past ~100 blocks trigger pandas PerformanceWarning and O(n)
    inserts). _defrag between stages keeps the block count low."""
    import warnings

    from pandas.errors import PerformanceWarning

    raw = _synthetic_raw(n_days=30)
    proc = process_data(raw)
    with warnings.catch_warnings():
        warnings.simplefilter("error", PerformanceWarning)
        engineer_features(proc, save=False)


def test_live_rows_complete(parity_frames):
    _, train_day, feat_live, _ = parity_frames
    assert len(feat_live) == len(train_day) > 0
    assert set(feat_live["race_id"]) == set(train_day["race_id"])


def test_live_rider_inputs(parity_frames):
    _, _, featured, _ = parity_frames
    apprentice = featured[featured.race_apprentice_jockeys.eq(1)]
    mixed = featured[featured.race_conditional_jockeys.eq(1)]
    assert len(apprentice) == len(mixed) == 6
    assert apprentice.race_amateur_riders.eq(0).all()
    assert mixed.race_amateur_riders.eq(1).all()
    assert sorted(apprentice.reported_jockey_claim_lbs) == [0, 0, 0, 0, 0, 7]
    adjusted = apprentice.weight_lbs - apprentice.reported_jockey_claim_lbs
    np.testing.assert_allclose(apprentice.claim_adjusted_weight_lbs, adjusted)
    np.testing.assert_allclose(apprentice.weight_vs_field, adjusted - adjusted.mean())
    np.testing.assert_allclose(apprentice.weight_rank, adjusted.rank(ascending=False, method="min"))
    np.testing.assert_allclose(apprentice.weight_per_furlong, adjusted / apprentice.distance_furlongs)
    np.testing.assert_allclose(apprentice.uphill_x_weight, adjusted * apprentice.track_uphill_finish)
    # In the fixture each horse has the same allocation on every appearance.
    change = -apprentice.reported_jockey_claim_lbs + 3 * apprentice.reported_jockey_claim_lbs.eq(7)
    np.testing.assert_allclose(apprentice.weight_change, change)
    np.testing.assert_allclose(apprentice.weight_change_vs_field, change - change.mean())
    np.testing.assert_allclose(apprentice.is_big_weight_drop, change.lt(-3))
    np.testing.assert_allclose(apprentice.weight_dropped, change.lt(0))
    np.testing.assert_allclose(apprentice.weight_raised, change.gt(0))
    # An unverified claim leaves the claim itself and the adjusted weight missing; comparisons
    # fall back to the allocated weight, which is observed for every runner.
    assert mixed[["reported_jockey_claim_lbs", "claim_adjusted_weight_lbs"]].isna().all(axis=None)
    np.testing.assert_allclose(mixed.weight_vs_field, mixed.weight_lbs - mixed.weight_lbs.mean())
    np.testing.assert_allclose(mixed.weight_per_furlong, mixed.weight_lbs / mixed.distance_furlongs)
    compared = ["weight_rank", "uphill_x_weight", "weight_change", "weight_change_vs_field",
                "is_big_weight_drop", "weight_dropped", "weight_raised"]
    assert mixed[compared].notna().all(axis=None)
    selectable = get_feature_columns(featured)
    assert "claim_adjusted_weight_lbs" in selectable and "weight_lbs" not in selectable


def test_no_model_features_missing_in_live(parity_frames):
    feat_train, _, feat_live, _ = parity_frames
    model_cols = get_feature_columns(feat_train)
    missing = [c for c in model_cols if c not in feat_live.columns]
    assert not missing, (
        f"{len(missing)} model features absent from live path (would be "
        f"silently zero-filled at predict time): {missing[:20]}"
    )


# Features that legitimately differ because they are derived from the
# outcome placeholders themselves rather than horse history. Keep this
# list EMPTY unless a difference is argued and documented.
KNOWN_DIVERGENT: set[str] = set()


def test_feature_values_match(parity_frames):
    feat_train, train_day, feat_live, _ = parity_frames
    model_cols = [
        c for c in get_feature_columns(feat_train)
        if c in feat_live.columns and c not in KNOWN_DIVERGENT
    ]
    key = ["race_id", "horse_name"]
    t = train_day.set_index(key)[model_cols].sort_index()
    l = feat_live.set_index(key)[model_cols].sort_index()
    assert t.index.equals(l.index)

    bad = []
    for col in model_cols:
        a = pd.to_numeric(t[col], errors="coerce").to_numpy(dtype=float)
        b = pd.to_numeric(l[col], errors="coerce").to_numpy(dtype=float)
        both_nan = np.isnan(a) & np.isnan(b)
        # Tolerances absorb float order-of-operations noise (cumsums run
        # over different array lengths in the two paths, ~1e-6 worst) —
        # genuine skew shows up at 1e-2 scales and beyond.
        close = np.isclose(a, b, rtol=1e-4, atol=1e-5, equal_nan=False) | both_nan
        if not close.all():
            n_bad = int((~close).sum())
            i = int(np.argmax(~close))
            bad.append(f"{col}: {n_bad}/{len(a)} rows differ "
                       f"(e.g. train={a[i]!r} live={b[i]!r} at {t.index[i]})")
    assert not bad, (
        "Training/serving skew — these features differ between the "
        "training path and the live path:\n" + "\n".join(bad[:40])
    )
