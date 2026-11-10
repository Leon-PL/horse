"""Data-integrity gate for the scraped results DB.

Run after each scrape/backfill to stop erroneous data creeping back in.
These assert invariants that must *always* hold for the Thoroughbred results
store; they are deliberately conservative so legitimate edge cases do not
false-fail:

* ``draw > num_runners`` is NOT asserted — it is legitimate (non-runners
  scratch after stalls are numbered, lowering ``num_runners``).
* "one winner per race" is NOT asserted — dead-heats produce ties.

The whole module skips cleanly when the DB is absent (e.g. fresh checkout / CI
without data).
"""

import sqlite3
from pathlib import Path

import pytest

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "races.db"
ALLOWED_REGIONS = {"UK", "INT"}


@pytest.fixture(scope="module")
def con():
    if not DB_PATH.exists():
        pytest.skip(f"results DB not present at {DB_PATH}")
    c = sqlite3.connect(str(DB_PATH))
    yield c
    c.close()


def _count(con, where: str, params=()) -> int:
    return con.execute(f"SELECT COUNT(*) FROM results WHERE {where}", params).fetchone()[0]


class TestNoBreedContamination:
    def test_no_arabian_horses(self, con):
        assert _count(con, "horse_name LIKE '%(A)'") == 0

    def test_no_feed_channel_tracks(self, con):
        assert _count(con, "track LIKE '%Extras%'") == 0

    def test_no_arab_race_names(self, con):
        # "Arab Race" / "(Arab ..." in the name — but NOT "Arabella" (a
        # thoroughbred sponsor/horse name).
        assert _count(con, "race_name LIKE '%Arab Race%'") == 0
        assert _count(con, "race_name LIKE '%(Arab%'") == 0


class TestResultInvariants:
    def test_odds_are_positive_decimal(self, con):
        # Decimal odds are always > 1.0 (stake returned + profit).
        assert _count(con, "odds <= 1.0") == 0

    def test_won_is_binary(self, con):
        assert _count(con, "won NOT IN (0, 1)") == 0

    def test_winner_finished_first(self, con):
        # won == 1  <=>  finish_position == 1 (dead-heats both have pos 1).
        assert _count(con, "won = 1 AND finish_position != 1") == 0

    def test_finish_position_non_negative(self, con):
        assert _count(con, "finish_position < 0") == 0

    def test_field_size_positive(self, con):
        assert _count(con, "num_runners <= 0") == 0

    def test_draw_non_negative(self, con):
        assert _count(con, "draw < 0") == 0

    def test_region_in_allowed_set(self, con):
        regions = {r[0] for r in con.execute("SELECT DISTINCT region FROM results")}
        assert regions <= ALLOWED_REGIONS, f"unexpected regions: {regions - ALLOWED_REGIONS}"


def _duplicate_runners(con):
    # Namesakes with different canonical IDs can run in the same race. Names
    # remain the fallback key for legacy rows without a usable horse ID.
    return con.execute(
        "SELECT COUNT(*) FROM (SELECT race_id, "
        "CASE WHEN horse_id IS NULL OR TRIM(CAST(horse_id AS TEXT)) IN ('', '0') "
        "THEN 'name:' || horse_name ELSE 'id:' || CAST(horse_id AS TEXT) END AS runner_key "
        "FROM results GROUP BY race_id, runner_key HAVING COUNT(*) > 1)"
    ).fetchone()[0]


@pytest.mark.parametrize("runners,expected", [
    ([(1, "Namesake"), (2, "Namesake")], 0),
    ([(1, "Original"), (1, "Alias")], 1),
    ([(None, "Unknown ID"), (None, "Unknown ID")], 1),
])
def test_duplicate_gate_uses_runner_identity(runners, expected):
    with sqlite3.connect(":memory:") as connection:
        connection.execute("CREATE TABLE results (race_id TEXT, horse_id INTEGER, horse_name TEXT)")
        connection.executemany("INSERT INTO results VALUES ('race', ?, ?)", runners)
        assert _duplicate_runners(connection) == expected


class TestNoDuplicateRunners:
    def test_unique_horse_per_race(self, con):
        assert _duplicate_runners(con) == 0
