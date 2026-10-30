"""Exercise real SQLite and five-file publication across injected failures."""
import json
import sqlite3

import pandas as pd
import pytest

from scripts import install_amended_results as install
from src.feature_store import FeatureStore
from src.result_amendment_repair import apply_amendments


def publication(tmp_path, monkeypatch):
    output = tmp_path / "installation"
    output.mkdir()
    (output / "qa_before").mkdir()
    current = FeatureStore(str(tmp_path / "current"))
    ready = FeatureStore(str(output / "ready_store"))
    for store, generation in ((current, "old"), (ready, "corrected")):
        for name in store._FILES:
            (tmp_path / store.root / name).write_bytes(
                json.dumps({"fe_signature": "same", "generation": generation}).encode()
                if name == "meta.json" else f"{generation}:{name}".encode())
    before = pd.DataFrame({"race_id": ["one"] * 2, "horse_id": ["101", "102"],
        "horse_name": ["First", "Second"], "race_date": ["2025-01-01"] * 2,
        "track": ["Kempton"] * 2, "off_time": ["14:00"] * 2, "num_runners": [2] * 2,
        "finish_position": [1, 2], "finish_pos_label": [None] * 2,
        "won": [1, 0], "lengths_behind": [0., .1], "scraped_at": ["2025-01-02"] * 2})
    proposed = before.assign(source_position=[2, 1], exchange_won=[0, 1], atr_raceid="source")
    database = tmp_path / "results.db"
    raw = tmp_path / "results.csv"
    pointer = tmp_path / "production.json"
    pointer.write_text('"old"')
    with sqlite3.connect(database) as connection:
        before.to_sql("results", connection, index=False)
        install.repair.export_csv(connection, raw)
    with sqlite3.connect(":memory:") as preview:
        before.to_sql("results", preview, index=False)
        apply_amendments(preview, before, proposed, repair_id=install.repair.REPAIR_ID, evidence={})
        install.repair.export_csv(preview, output / "source_preview.csv")
    prepared = {"raw_preview_sha256": install.sha(output / "source_preview.csv"), "source_rows": 2,
        "store_files_sha256": {name: install.sha(ready.root + "/" + name) for name in ready._FILES}}
    for name, value in {"prepared.json": prepared,
            "plan.json": {"source_sha256": {}, "evidence_sha256": {}},
            "recorder_paused.json": {"paused": True, "model_id": "old"},
            "qa_before/report.json": {"passed": True, "run_id": "new"}}.items():
        (output / name).write_text(json.dumps(value))
    fit = {"run_id": "new", "model_sha256": "fixture", "fe_signature": "same"}
    monkeypatch.setattr(install, "OUTPUT", output)
    monkeypatch.setattr(install, "ROOT", tmp_path)
    monkeypatch.setattr(install, "DB_PATH", database)
    monkeypatch.setattr(install, "RAW", raw)
    monkeypatch.setattr(install, "PRODUCTION_POINTER", pointer)
    monkeypatch.setattr(install, "FeatureStore", lambda root=None: FeatureStore(root or current.root))
    monkeypatch.setattr(install, "verified_inputs", lambda: (fit, {"parent_run": "old"}, before, proposed, {}))
    monkeypatch.setattr(install, "fe_signature", lambda: "same")

    def promote(run_id):
        pointer.write_text(json.dumps(run_id))
        return True

    monkeypatch.setattr(install, "promote", promote)
    old_files = {name: (tmp_path / current.root / name).read_bytes() for name in current._FILES}
    return output, current, ready, database, raw, pointer, before, old_files


@pytest.mark.parametrize("failure", ["changed_source", "new_tail", "promotion", "before_commit", "after_commit", None])
def test_database_history_raw_and_model_move_or_rollback_together(tmp_path, monkeypatch, failure):
    output, current, ready, database, raw, pointer, before, old_files = publication(tmp_path, monkeypatch)
    old_raw = raw.read_bytes()
    if failure == "changed_source":
        with sqlite3.connect(database) as connection:
            connection.execute("ALTER TABLE results ADD COLUMN newly_added TEXT")
    if failure == "new_tail":
        with sqlite3.connect(database) as connection:
            before.assign(race_id="later", race_date="2025-01-02").to_sql("results", connection, if_exists="append", index=False)
    with sqlite3.connect(database) as connection:
        expected_before = pd.read_sql_query("SELECT * FROM results", connection)
    if failure == "promotion":
        monkeypatch.setattr(install, "promote", lambda run_id: False)
    if failure in {"before_commit", "after_commit"}:
        ordinary_write = install.write

        def interrupted_write(path, value):
            if path.name == ("commit_intent.json" if failure == "before_commit" else "published.json"):
                raise OSError("injected receipt failure")
            ordinary_write(path, value)

        monkeypatch.setattr(install, "write", interrupted_write)
    if failure:
        with pytest.raises((ValueError, OSError, RuntimeError)):
            install.publish()
    else:
        install.publish()
    committed = failure in {None, "after_commit"}
    with sqlite3.connect(database) as connection:
        rows = pd.read_sql_query("SELECT * FROM results", connection)
        ledger = connection.execute("SELECT name FROM sqlite_master WHERE name='result_amendment_batches'").fetchone()
    assert bool(ledger) == committed
    if committed:
        assert rows.won.tolist() == [0, 1]
        assert raw.read_bytes() == (output / "source_preview.csv").read_bytes()
        assert json.loads(pointer.read_text()) == "new"
        for name in current._FILES:
            assert (tmp_path / current.root / name).read_bytes() == (tmp_path / ready.root / name).read_bytes()
        assert (output / "commit_intent.json").exists()
    else:
        pd.testing.assert_frame_equal(rows, expected_before)
        assert raw.read_bytes() == old_raw
        assert json.loads(pointer.read_text()) == "old"
        for name, original in old_files.items():
            assert (tmp_path / current.root / name).read_bytes() == original
        assert (output / "publication_failure.json").exists()
