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

from src.data_processor import process_data
from src.feature_engineer import engineer_features
from src.live_prediction import feature_engineer_with_history_core
from src.model import get_feature_columns


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
                    "horse_id": f"hid{h}",
                    "jockey": jockeys[h % len(jockeys)],
                    "trainer": trainers[h % len(trainers)],
                    "age": 4 + h % 5,
                    "sex": ["G", "F", "M"][h % 3],
                    "headgear": "" if h % 4 else "Blinkers",
                    "weight_lbs": 130.0 + (h % 10),
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

    # ── Training path: everything processed and featured together ──
    feat_train = engineer_features(process_data(raw.copy(), save=False), save=False)

    # ── Live path: history vs cards, outcomes stripped like the app ──
    hist_raw = raw[raw["race_date"] < last_date].copy()
    cards = raw[raw["race_date"] == last_date].copy()
    cards["won"] = 0
    cards["finish_position"] = 0
    cards["lengths_behind"] = np.nan

    proc_hist = process_data(hist_raw, save=False)
    proc_cards = process_data(cards, save=False)
    feat_live = feature_engineer_with_history_core(proc_hist, proc_cards)

    train_day = feat_train[feat_train["race_date"] == last_date]
    return feat_train, train_day, feat_live, last_date


def test_live_rows_complete(parity_frames):
    _, train_day, feat_live, _ = parity_frames
    assert len(feat_live) == len(train_day) > 0
    assert set(feat_live["race_id"]) == set(train_day["race_id"])


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
