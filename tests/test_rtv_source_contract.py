"""Verified-source exports must match the ordinary database CSV contract."""
import sqlite3
import pandas as pd

from scripts.install_rtv_generation_044 import export


def test_operational_timestamp_cannot_change_export_but_a_result_correction_can(tmp_path):
    con = sqlite3.connect(":memory:")
    con.execute("CREATE TABLE results (race_id TEXT,race_date TEXT,horse_name TEXT,finish_position INTEGER,scraped_at TEXT)")
    con.execute("INSERT INTO results VALUES ('one','2026-10-29','Runner',1,'2026-10-30 09:00:00')")
    path = tmp_path / "results.csv"
    assert export(con, path) == 1
    assert "scraped_at" not in pd.read_csv(path)
    before = path.read_bytes()
    con.execute("UPDATE results SET scraped_at='2026-10-30 17:00:00'")
    export(con, path)
    assert path.read_bytes() == before
    con.execute("UPDATE results SET finish_position=2")
    export(con, path)
    assert path.read_bytes() != before
    con.close()
