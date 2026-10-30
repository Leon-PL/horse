import numpy as np
import pandas as pd

from src import database
from src.data_processor import clean_data
from src.feature_engineer import _add_result_history_flags, add_horse_features
from src.result_status import result_labels, pending_card


def field(positions, labels=None, declared=3):
    f = pd.DataFrame({"race_id": 1, "finish_position": positions, "num_runners": declared,
                      "horse_name": [f"H{i}" for i in range(len(positions))],
                      "won": np.equal(positions, 1).astype(int)})
    if labels is not None:
        f["finish_pos_label"] = labels
    return f


def test_legacy_nonfinishers_are_reconstructed_from_a_settled_whole_field():
    f = clean_data(field([1, 0, 2]).assign(race_date="2025-01-01"))
    assert f.finish_pos_label.tolist() == ["", "DNF", ""]
    out = _add_result_history_flags(f)
    assert out._result_start.tolist() == [1, 1, 1]
    assert out._non_finisher_result.tolist() == [0, 1, 0]


def test_pending_cards_and_incomplete_results_do_not_invent_dnfs():
    assert result_labels(field([0, 0, 0])).eq("").all()
    assert result_labels(field([1, 0], declared=3)).eq("").all()


def test_reported_reasons_survive_and_explicit_nonstarters_do_not_count():
    f = field([1, 0, 0], ["", "PU", "NR"])
    assert result_labels(f).tolist() == ["", "PU", "NR"]
    f["won"] = [1, 0, 0]
    assert _add_result_history_flags(f)._result_start.tolist() == [1, 1, 0]


def test_prior_noncompletion_is_a_start_and_a_loss_not_an_unknown_card():
    f = pd.concat([field([1, 0, 2]), field([2, 1, 3]).assign(race_id=2)], ignore_index=True)
    f["race_date"] = pd.to_datetime(["2025-01-01"]*3 + ["2025-01-02"]*3)
    f["won"] = f.finish_position.eq(1).astype(int)
    f["lengths_behind"] = np.where(f.finish_position.eq(0), np.nan, f.finish_position - 1.)
    out = add_horse_features(_add_result_history_flags(clean_data(f)))
    next_start = out.loc[out.race_id.eq(2) & out.horse_name.eq("H1")].iloc[0]
    assert next_start.horse_prev_races == 1
    assert next_start.horse_last_run_non_finish == 1
    assert next_start.horse_non_finish_rate == 1


def test_database_roundtrip_preserves_the_scraped_reason(tmp_path, monkeypatch):
    monkeypatch.setattr(database, "DB_PATH", str(tmp_path / "results.db"))
    database.init_db()
    f = field([1, 0, 2], ["", "UR", ""]).assign(race_date="2025-01-01", won=[1, 0, 0])
    database.insert_results(f)
    loaded = database.load_from_database()
    assert loaded.loc[loaded.horse_name.eq("H1"), "finish_pos_label"].iloc[0] == "UR"


def test_live_card_does_not_keep_a_stale_nonfinisher_label_or_margin():
    f = field([1, 0, 2], ["", "PU", ""]).assign(beaten_lengths=[0., np.nan, 4.])
    card = pending_card(f)
    assert card.finish_pos_label.eq("").all()
    assert card.beaten_lengths.isna().all()
    assert card.finish_position.eq(0).all()
    assert _add_result_history_flags(card)._result_start.eq(0).all()


def test_raw_processor_bypass_does_not_make_legacy_dnfs_pending():
    f = field([1, 0, 2])
    assert _add_result_history_flags(f)._result_start.tolist() == [1, 1, 1]


def test_numeric_result_labels_and_clocks_cannot_be_selected(monkeypatch):
    import config
    from src.model import get_feature_columns
    monkeypatch.setattr(config, "FEATURE_ALLOWLIST", None)
    f = field([1, 0, 2]).assign(finish_pos_label=1, winning_time_secs=80., off_time_actual=42., safe_feature=2.)
    assert not {"finish_pos_label", "winning_time_secs", "off_time_actual"}.intersection(get_feature_columns(f))
    assert "safe_feature" in get_feature_columns(f)
