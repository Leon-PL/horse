from types import SimpleNamespace
import pytest

from scripts.adopt_capacity_upgrade import capacity_defaults, borrow_place


def test_only_classifier_cap_changes_and_original_bytes_survive():
    source = b'# Settings\r\nCLASSIFIER_PARAMS = {"n_estimators": 2000, "learning_rate": .008}\r\nPLACE_CLASSIFIER_PARAMS = {"n_estimators": 3000}\r\n'
    changed = capacity_defaults(source)
    assert changed == source.replace(b'"n_estimators": 2000', b'"n_estimators": 4000')
    assert capacity_defaults(changed) == changed


@pytest.mark.parametrize("source", [b'CLASSIFIER_PARAMS = {}', b'CLASSIFIER_PARAMS = {"n_estimators": 500}',
    b'CLASSIFIER_PARAMS = {"n_estimators": 2000, "n_estimators": 2000}'])
def test_unexpected_defaults_are_rejected(source):
    with pytest.raises(ValueError):
        capacity_defaults(source)


def predictor():
    return SimpleNamespace(feature_cols=["a", "b"], frameworks={"place": "lgbm"},
        place_model=object(), place_platt_a=1.2, place_platt_b=-.1, place_iso_cal=object(),
        eval_split_info={"training_runners": 100})


def test_place_reuse_keeps_all_calibration_state_and_fit_provenance():
    new, old = predictor(), predictor()
    borrow_place(new, old, {"run_id": "parent"})
    assert new.place_model is old.place_model
    assert new.place_iso_cal is old.place_iso_cal
    assert (new.place_platt_a, new.place_platt_b) == (old.place_platt_a, old.place_platt_b)
    assert new.eval_split_info["training_runners"] == 100
    assert new.eval_split_info["reused_place_component"] == {"run_id": "parent"}


def test_place_reuse_rejects_different_feature_order():
    new, old = predictor(), predictor(); old.feature_cols.reverse()
    with pytest.raises(ValueError, match="input order"):
        borrow_place(new, old, {})
