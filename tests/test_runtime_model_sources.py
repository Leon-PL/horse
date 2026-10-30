import hashlib

from src import run_store


def setup_guard(tmp_path, monkeypatch):
    source = tmp_path / "src/recent_probability.py"
    source.parent.mkdir()
    source.write_text("coefficient = 1\n")
    record = {"training_config": {"signatures": {"fe_signature": "same"},
        "runtime_model_sources": {"src/recent_probability.py": hashlib.sha256(source.read_bytes()).hexdigest()}}}
    monkeypatch.setattr(run_store, "load_run_meta", lambda run_id: record)
    monkeypatch.setattr(run_store, "store_fe_signature", lambda: "same")
    monkeypatch.setattr(run_store.config, "BASE_DIR", str(tmp_path))
    return source, record


def test_unchanged_probability_helpers_allow_the_recorded_generation(tmp_path, monkeypatch):
    setup_guard(tmp_path, monkeypatch)
    assert run_store.feature_generation_status("recorded")["state"] == run_store.GENERATION_MATCH


def test_changed_probability_math_blocks_promotion_even_when_feature_signature_matches(tmp_path, monkeypatch):
    source, _ = setup_guard(tmp_path, monkeypatch)
    source.write_text("coefficient = 2\n")
    result = run_store.feature_generation_status("recorded")
    assert result["state"] == run_store.GENERATION_MISMATCH
    assert result["runtime_source_mismatches"] == ["src/recent_probability.py"]
    monkeypatch.setattr(run_store, "run_has_model", lambda run_id: True)
    assert not run_store.promote("recorded")


def test_missing_or_outside_project_source_cannot_pass_the_runtime_guard(tmp_path, monkeypatch):
    source, record = setup_guard(tmp_path, monkeypatch)
    source.unlink()
    assert run_store.feature_generation_status("recorded")["state"] == run_store.GENERATION_MISMATCH
    record["training_config"]["runtime_model_sources"] = {"../outside.py": "unverified"}
    assert run_store.feature_generation_status("recorded")["state"] == run_store.GENERATION_MISMATCH


def test_legacy_metadata_retains_existing_generation_checks(tmp_path, monkeypatch):
    _, record = setup_guard(tmp_path, monkeypatch)
    del record["training_config"]["runtime_model_sources"]
    assert run_store.feature_generation_status("recorded")["state"] == run_store.GENERATION_MATCH
