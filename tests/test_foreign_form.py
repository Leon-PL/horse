import numpy as np
import pandas as pd
import pytest

from src import feature_engineer as fe
from src.atr_features import published_atr_rolling
from src.feature_context import counts_state_scope
from src.feature_store import FeatureStore
from src.ingestion.foreign_form import COLUMNS, accepted_foreign_form, race_rows
from src.ratings import compute_elo_features, compute_glicko_features
from src.source_contracts import TableContract
from src.store_build import build_store
from src.store_refresh import append_current_store


def _foreign(day, *, horse=1, country="fr", track="longchamp", provider="sl", race_id="f1",
             position=2, field=5):
    return {
        "horse_id": horse, "horse_name": f"Horse {horse}", "race_date": day, "off_time": "15:00",
        "available_date": (pd.Timestamp(day) + pd.Timedelta(days=1)).date().isoformat(),
        "country": country, "track": track, "race_type": "Flat", "surface": "Turf",
        "distance_furlongs": 8.0, "finish_position": position, "finish_pos_label": "",
        "num_runners": field, "lengths_behind": 1.0, "beaten_lengths": 1.0,
        "provider": provider, "native_race_id": race_id, "source_url": "fixture",
    }


def _field(day, race_id, horses):
    """A complete foreign race; *horses* in finishing order."""
    return [_foreign(day, horse=h, race_id=race_id, position=i + 1, field=len(horses))
            for i, h in enumerate(horses)]


def _publish(publish_dataset, canonical, rows, atr=None):
    contract = TableContract("horse start", COLUMNS, ("horse_id", "race_date"), role="history")
    sources = {"sl": {"results": (canonical, TableContract("canonical runner", (), ("race_id", "horse_id"))),
                      "foreign_form": (pd.DataFrame(rows), contract)}}
    if atr is not None:
        sources["atr"] = {"sectionals": (atr, TableContract("runner", ()))}
    return publish_dataset(canonical, **sources)


def _domestic():
    return pd.DataFrame({
        "race_id": ["r1", "r2", "r3", "r4", "r5", "r6"], "horse_id": 1,
        "horse_name": "Test Horse", "track": "Kempton", "off_time": "14:00",
        "race_date": pd.to_datetime(["2025-01-01", "2025-01-02", "2025-01-03",
                                     "2025-01-04", "2025-01-05", "2025-01-09"]),
        "finish_position": [1, 2, 3, 4, 5, 1], "num_runners": 5,
        "won": [1, 0, 0, 0, 0, 1], "race_type": "Flat", "race_class": 3,
        "distance_furlongs": 8.0, "lengths_behind": [0., 1., 2., 3., 4., 0.],
        "beaten_lengths": [0., 1., 2., 3., 4., 0.],
    })


def test_every_runner_is_kept_and_non_finishers_follow_the_domestic_convention():
    frame = pd.DataFrame({
        "race_id": "948432", "race_date": "2026-11-10", "off_time": "11:00", "track": "ParisLongchamp",
        "region": "INT", "horse_name": ["A", "B", "C"], "horse_id": [11, 12, 13],
        "finish_position": [1, 2, 99], "finish_pos_label": ["", "", "PU"], "won": [1, 0, 0],
        "num_runners": 3, "race_type": "Flat", "surface": "TURF", "distance_furlongs": 10.0,
        "lengths_behind": [0.0, 2.0, np.nan],
    })
    info = {"race_id": "948432", "country": "France", "url": "fixture"}
    rows = race_rows(frame, info, "2026-11-10")
    assert rows.horse_id.tolist() == [11, 12, 13]
    assert rows.finish_position.tolist() == [1, 2, 0]
    assert rows.finish_pos_label.tolist() == ["", "", "PU"]
    assert rows.track.unique().tolist() == ["longchamp"] and rows.surface.unique().tolist() == ["Turf"]
    assert rows.available_date.unique().tolist() == ["2026-11-11"]
    with pytest.raises(ValueError):
        race_rows(frame.assign(region="UK"), info, "2026-11-10")


def test_a_horse_in_two_races_on_one_day_withholds_both_races():
    keep = _field("2025-01-06", "f1", [1, 2])
    clash = _field("2025-01-06", "f2", [3, 4]) + _field("2025-01-06", "f3", [4, 5])
    legacy = [_foreign("2025-01-06", horse=9, provider="rtv", race_id="r9")]
    result = accepted_foreign_form(pd.DataFrame(keep + clash + legacy))
    assert result.native_race_id.unique().tolist() == ["f1"]


def test_foreign_runs_change_prior_form_and_speed_slots_without_adding_rows(publish_dataset):
    domestic = _domestic()
    _publish(publish_dataset, domestic, [_foreign("2025-01-05"), _foreign("2025-01-10", race_id="f2")])

    def calculate(frame, state):
        frame = fe._add_result_history_flags(frame.copy())
        frame["_event_dt"] = fe._event_sort_key(frame)
        with counts_state_scope(state):
            frame = fe.add_horse_features(frame)
            return fe.add_speed_figure_features(frame.drop(columns="_event_dt"))

    full = calculate(domestic, {})
    state = {}
    append = pd.concat([calculate(domestic.iloc[:3], state),
                        calculate(domestic.iloc[3:5], state),
                        calculate(domestic.iloc[5:], state)])
    assert full.race_id.tolist() == domestic.race_id.tolist()
    assert full.horse_prev_races.tolist() == [0, 1, 2, 3, 4, 6]
    assert full.iloc[-1].days_since_last_race == 4
    assert full.iloc[-1].speed_fig_avg_5 == pytest.approx(2.5 / 8.0 ** 0.6)
    pd.testing.assert_frame_equal(full, append)


def test_atr_uses_the_same_window_and_preserves_missing_sectionals(publish_dataset):
    domestic = _domestic()
    atr = domestic.iloc[:5][["race_date", "track", "off_time", "horse_name"]].copy()
    atr["atr_raceid"] = ["atr1", "atr2", "atr3", "atr4", "atr5"]
    atr["atr_horse_id"] = "native-1"
    atr["atr_finishing_speed_pct"] = [100., 101., 102., 103., 104.]
    _publish(publish_dataset, domestic, [_foreign("2025-01-06")], atr=atr)
    result = published_atr_rolling(domestic)
    assert np.isnan(result["atr_finishing_speed_pct_avg_3"][0])
    assert result["atr_finishing_speed_pct_avg_3"][-1] == pytest.approx(103.5)
    assert result["atr_observation_count_5"][-1] == 4


def _two_horse_domestic():
    """Horses 1 and 2 meet twice in Britain; 1 also races in France in between."""
    days = pd.to_datetime(["2025-01-01", "2025-01-01", "2025-01-10", "2025-01-10"])
    return pd.DataFrame({
        "race_id": ["r1", "r1", "r2", "r2"], "horse_id": [1, 2, 1, 2],
        "horse_name": ["Horse 1", "Horse 2", "Horse 1", "Horse 2"], "off_time": "14:00",
        "race_date": days, "finish_position": [2, 1, 1, 2], "num_runners": 2,
        "race_type": "Flat", "distance_furlongs": 8.0,
        "lengths_behind": [1.0, 0.0, 0.0, 1.0], "beaten_lengths": [1.0, 0.0, 0.0, 1.0],
    })


@pytest.mark.parametrize("rate,state_key", [(compute_elo_features, "elo_state"),
                                            (compute_glicko_features, "glicko_state")])
def test_foreign_races_update_ratings_only_and_chunks_match_one_sweep(publish_dataset, rate, state_key):
    domestic = _two_horse_domestic()
    # Horse 1 beats two horses that never race in Britain, three days before r2.
    france = _field("2025-01-05", "f1", [1, 900, 901])
    publish_dataset(domestic)
    without = rate(domestic.copy())
    _publish(publish_dataset, domestic, france)
    full = rate(domestic.copy())
    rating = "horse_elo" if rate is compute_elo_features else "horse_glicko"

    assert full.race_id.tolist() == domestic.race_id.tolist()
    assert set(full.columns) == set(without.columns)
    assert full[rating].iloc[:2].tolist() == without[rating].iloc[:2].tolist()
    assert full[rating].iloc[2] > without[rating].iloc[2]          # horse 1 after its French win
    assert full[rating].iloc[3] == pytest.approx(without[rating].iloc[3])  # horse 2 never left

    first, state = rate(domestic.iloc[:2].copy(), return_state=True)
    second = rate(domestic.iloc[2:].copy(), **{state_key: state})
    pd.testing.assert_frame_equal(pd.concat([first, second], ignore_index=True)[full.columns], full,
                                  check_dtype=False)


def test_old_rating_state_without_a_foreign_mark_is_refused(publish_dataset):
    domestic = _two_horse_domestic()
    publish_dataset(domestic)
    _, state = compute_elo_features(domestic.iloc[:2].copy(), return_state=True)
    state.pop("foreign_races")
    _publish(publish_dataset, domestic, _field("2025-01-05", "f1", [1, 900]))
    with pytest.raises(ValueError, match="predates foreign races"):
        compute_elo_features(domestic.iloc[2:].copy(), elo_state=state)


def test_store_rows_stay_domestic_ratings_see_foreign_races_and_backfill_invalidates_store(
        publish_dataset, tmp_path):
    from src.data_processor import process_data
    from tests.test_live_parity import _synthetic_raw

    raw = _synthetic_raw(n_days=8, races_per_day=1, field=5).assign(region="UK")
    horse_id = int(raw.iloc[0].horse_id)
    first = pd.Timestamp(raw.race_date.min())
    overseas = _field((first + pd.Timedelta(days=2)).date().isoformat(), "f1", [horse_id, 9001, 9002])
    _publish(publish_dataset, raw, overseas)
    processed = process_data(raw)
    store = FeatureStore(str(tmp_path / "with_foreign"))
    build_store(store, 12, processed=processed, use_helpers=False)
    columns = ["race_id", "horse_id", "won", "horse_prev_races", "horse_elo", "horse_glicko"]
    with_foreign = store.load_featured(columns=columns)
    one_sweep = fe.engineer_features(processed.copy(), save=False)

    publish_dataset(raw)
    baseline = fe.engineer_features(processed.copy(), save=False)
    keys = ["race_id", "horse_id"]
    before = baseline.set_index(keys).sort_index()
    after = with_foreign.set_index(keys).sort_index()
    assert before.index.equals(after.index)
    assert not after.index.get_level_values("race_id").astype(str).str.startswith("foreign").any()
    pd.testing.assert_series_equal(before.won, after.won, check_dtype=False)
    assert (after.horse_prev_races - before.horse_prev_races).groupby(level="horse_id").max().max() == 1
    assert not np.allclose(before.horse_elo, after.horse_elo)
    # A chunked store build sweeps the foreign races exactly as one full pass.
    single = one_sweep.set_index(keys).sort_index()
    pd.testing.assert_frame_equal(single[["horse_elo", "horse_glicko"]], after[["horse_elo", "horse_glicko"]],
                                  check_dtype=False)

    _publish(publish_dataset, raw, overseas + _field(first.date().isoformat(), "f0", [9001, 9002]))
    assert not append_current_store(processed, store=store)
    assert store._rebuild_reason == "foreign_form_history_changed"


def test_an_old_empty_result_page_is_unavailable_not_a_failure(tmp_path, monkeypatch):
    import config
    from datetime import date, timedelta
    from src.ingestion import foreign_form

    class EmptyPages:
        def __init__(self):
            self.result_listing_evidence = {}

        def get_results_urls(self, day, countries):
            self.result_listing_evidence[day] = True
            return [{"race_id": day, "race_name": "Prix Test", "url": "fixture", "country": "France"}]

        def scrape_race_result(self, url, day, raise_if_not_run=True):
            return []

    monkeypatch.setattr(config, "DATA_DIR", str(tmp_path))
    monkeypatch.setattr(foreign_form, "SportingLifeScraper", EmptyPages)
    old = (date.today() - timedelta(days=10)).isoformat()
    recent = (date.today() - timedelta(days=1)).isoformat()
    result = foreign_form.refresh_foreign_form(old, old)
    assert result["state"] == "ok" and result["unavailable"] == [f"sl {old} ({old})"]
    result = foreign_form.refresh_foreign_form(recent, recent)
    assert result["state"] == "partial" and result["failures"]
