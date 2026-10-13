"""The field-timing audit must flag a scrape-time field (the sex leak's class)."""

import sqlite3

from scripts.audit_field_timing import audit


def _db(tmp_path, scrape_time_trainer: bool):
    path = tmp_path / "races.db"
    con = sqlite3.connect(path)
    con.execute("create table results (horse_id text, age int, official_rating int, form text, "
                "days_since_last_run int, weight_lbs real, trainer text, headgear text, sex text)")
    rows = []
    for h in range(30):
        for k in range(12):
            trainer = "T0" if scrape_time_trainer else f"T{k // 6}"
            rows.append((f"h{h}", 3 + k // 4, 60 + k, f"{k}", 10 + k, 120 + k,
                         trainer, ["", "b", "v"][k % 3], "g"))
    con.executemany("insert into results values (?,?,?,?,?,?,?,?,?)", rows)
    con.commit(); con.close()
    return str(path)


def test_flags_a_field_constant_per_horse(tmp_path):
    report = audit(_db(tmp_path, scrape_time_trainer=True))
    assert not report.set_index("field").loc["trainer", "ok"]


def test_passes_per_race_fields(tmp_path):
    report = audit(_db(tmp_path, scrape_time_trainer=False))
    assert report["ok"].all()
