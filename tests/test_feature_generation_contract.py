"""A promoted model must match the feature generation it will be served.

``train_run`` records the ``fe_signature`` of the build it trained on, and the
live path refuses to featurise when the store disagrees with the current FE
code. Without the check under test here, the middle link was missing: a model
trained on one feature generation could be promoted and served against
another, producing confident nonsense with no error anywhere.
"""

import json

import pytest

import config
from src import run_store as registry


@pytest.fixture
def registry_dirs(tmp_path, monkeypatch):
    runs = tmp_path / "runs"
    runs.mkdir()
    models = tmp_path / "models"
    models.mkdir()
    monkeypatch.setattr(registry, "RUNS_DIR", str(runs))
    monkeypatch.setattr(registry, "PRODUCTION_POINTER", str(models / "production.json"))
    monkeypatch.setattr(config, "MODELS_DIR", str(models))
    return runs


def _make_run(runs, run_id: str, fe_signature: str | None) -> str:
    run_dir = runs / run_id
    run_dir.mkdir()
    (run_dir / "model.joblib").write_bytes(b"model")
    meta: dict = {"run_id": run_id, "training_config": {}}
    if fe_signature is not None:
        meta["training_config"]["signatures"] = {"fe_signature": fe_signature}
    (run_dir / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    return run_id


@pytest.fixture
def store_signature(monkeypatch):
    """Control the store's reported fe_signature."""
    def _set(value):
        monkeypatch.setattr(registry, "store_fe_signature", lambda: value)
    return _set


def test_matching_generation_promotes_and_reports_match(registry_dirs, store_signature):
    store_signature("abc123")
    rid = _make_run(registry_dirs, "20260101_120000", "abc123")

    assert registry.promote(rid) is True
    status = registry.feature_generation_status()
    assert status["state"] == registry.GENERATION_MATCH
    assert status["run_id"] == rid


def test_mismatched_generation_is_refused(registry_dirs, store_signature):
    store_signature("new_store_build")
    rid = _make_run(registry_dirs, "20260101_120000", "old_training_build")

    assert registry.promote(rid) is False
    assert registry.get_production_run_id() is None, "a refused promotion must not take effect"

    status = registry.feature_generation_status(rid)
    assert status["state"] == registry.GENERATION_MISMATCH
    # The message must name both generations so the fix is obvious.
    assert "old_training_build" in status["message"]
    assert "new_store_build" in status["message"]


def test_mismatch_can_be_overridden_deliberately(registry_dirs, store_signature):
    store_signature("new_store_build")
    rid = _make_run(registry_dirs, "20260101_120000", "old_training_build")

    assert registry.promote(rid, allow_feature_mismatch=True) is True
    assert registry.get_production_run_id() == rid
    # Overriding promotes but must not launder the status into a match.
    assert registry.feature_generation_status()["state"] == registry.GENERATION_MISMATCH


def test_legacy_run_without_signature_needs_a_deliberate_override(registry_dirs, store_signature):
    """An unverifiable run is refused by default: on 2026-10-13 every unsigned
    run in the registry predated the trainer_freq leak fix."""
    store_signature("abc123")
    rid = _make_run(registry_dirs, "20260101_120000", None)

    assert registry.promote(rid) is False
    assert registry.get_production_run_id() is None

    assert registry.promote(rid, allow_feature_mismatch=True) is True
    status = registry.feature_generation_status()
    assert status["state"] == registry.GENERATION_UNKNOWN, "an override must not claim safety"
    assert status["run_fe_signature"] is None


def test_unbuilt_store_is_unverified(registry_dirs, store_signature):
    store_signature(None)
    rid = _make_run(registry_dirs, "20260101_120000", "abc123")

    assert registry.promote(rid) is False
    assert registry.promote(rid, allow_feature_mismatch=True) is True
    assert registry.feature_generation_status()["state"] == registry.GENERATION_UNKNOWN


def test_rebuilding_the_store_invalidates_an_already_promoted_model(
    registry_dirs, store_signature, monkeypatch
):
    """The check is evaluated at serving time, not frozen at promotion."""
    store_signature("gen_one")
    rid = _make_run(registry_dirs, "20260101_120000", "gen_one")
    assert registry.promote(rid) is True
    assert registry.feature_generation_status()["state"] == registry.GENERATION_MATCH

    store_signature("gen_two")  # a store rebuild happens later
    assert registry.feature_generation_status()["state"] == registry.GENERATION_MISMATCH


def test_no_promoted_model_is_its_own_state(registry_dirs, store_signature):
    store_signature("abc123")
    status = registry.feature_generation_status()
    assert status["state"] == registry.GENERATION_NO_MODEL
    assert status["run_id"] is None


def test_signatures_are_read_from_the_recorded_training_config(registry_dirs):
    rid = _make_run(registry_dirs, "20260101_120000", "abc123")
    assert registry.run_signatures(rid) == {"fe_signature": "abc123"}
    assert registry.run_signatures("20991231_000000") == {}


def test_store_signature_reads_the_live_store_meta(tmp_path, monkeypatch):
    """The real reader must come from the store's own meta.json."""
    monkeypatch.setattr(config, "PROCESSED_DATA_DIR", str(tmp_path))
    store_root = tmp_path / "feature_store"
    store_root.mkdir()

    assert registry.store_fe_signature() is None, "an unbuilt store has no signature"

    (store_root / "meta.json").write_text(
        json.dumps({"fe_signature": "from_disk"}), encoding="utf-8",
    )
    assert registry.store_fe_signature() == "from_disk"
