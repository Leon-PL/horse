"""The label audit must be capable of finding a wrong label.

An audit that cannot fail is worse than no audit, because it licenses trust.
The specific trap here is real and was nearly shipped: ``attach_bsp`` assigns
``out[c] = merged[c]``, so asking it for ``won`` overwrites the caller's own
label with Betfair's. An audit written that way compares the exchange against
itself and reports flawless agreement forever.
"""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def audit_mod():
    spec = importlib.util.spec_from_file_location(
        "audit_labels", ROOT / "scripts" / "audit_labels.py",
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["audit_labels"] = module
    spec.loader.exec_module(module)
    return module


def _results(n_races=40, runners=6):
    rows = []
    for r in range(n_races):
        for i in range(runners):
            rows.append({
                "race_id": f"r{r}", "race_date": pd.Timestamp("2025-03-01") + pd.Timedelta(days=r),
                "track": "Ascot", "off_time": "14:30",
                "horse_name": f"Horse {r}_{i}",
                "finish_position": i + 1, "won": 1 if i == 0 else 0,
                "num_runners": runners,
            })
    return pd.DataFrame(rows)


class TestInternalChecks:
    def test_clean_data_reports_no_problems(self, audit_mod):
        out = audit_mod.internal_checks(_results())
        assert out["races_without_a_winner"] == 0
        assert out["races_with_multiple_winners"] == 0
        assert out["won_vs_finish_position_disagreements"] == 0
        assert out["races_with_fewer_rows_than_declared"] == 0

    def test_a_race_with_no_winner_is_caught(self, audit_mod):
        df = _results()
        df.loc[df.race_id == "r3", "won"] = 0
        out = audit_mod.internal_checks(df)
        assert out["races_without_a_winner"] == 1

    def test_a_dead_heat_is_reported_separately_not_as_an_error(self, audit_mod):
        """Two winners is legitimate; it must not be conflated with a bug."""
        df = _results()
        df.loc[(df.race_id == "r5") & (df.finish_position == 2), "won"] = 1
        out = audit_mod.internal_checks(df)
        assert out["races_with_multiple_winners"] == 1
        assert out["races_without_a_winner"] == 0

    def test_won_disagreeing_with_finish_position_is_caught(self, audit_mod):
        df = _results()
        df.loc[(df.race_id == "r7") & (df.finish_position == 1), "won"] = 0
        df.loc[(df.race_id == "r7") & (df.finish_position == 3), "won"] = 1
        out = audit_mod.internal_checks(df)
        assert out["won_vs_finish_position_disagreements"] == 2

    def test_a_short_field_is_caught(self, audit_mod):
        df = _results()
        df = df[~((df.race_id == "r9") & (df.finish_position >= 5))]
        out = audit_mod.internal_checks(df)
        assert out["races_with_fewer_rows_than_declared"] == 1
        assert out["median_rows_missing_when_short"] == 2


class TestBetfairCrossCheck:
    """The overwrite trap, and the audit's ability to detect a real mismatch."""

    @staticmethod
    def _bsp_from(df, flip_race=None):
        """Betfair-shaped frame agreeing with *df*, except on *flip_race*."""
        from src.betfair import bf_horse_key, bf_track_key
        from src.utils import normalise_off_time_key

        bsp = pd.DataFrame({
            "race_date": pd.to_datetime(df.race_date).dt.strftime("%Y-%m-%d"),
            "track": df.track, "horse_name": df.horse_name,
            "off_key": df.off_time.map(normalise_off_time_key),
            "horse_key": df.horse_name.map(bf_horse_key),
            "won": df.won.to_numpy().copy(),
            "n_runners": df.num_runners,
            "bsp": 5.0,
        })
        if flip_race is not None:
            # A placing reversal: the runner-up is promoted to winner.
            sel = df.race_id == flip_race
            first = sel & (df.finish_position == 1)
            second = sel & (df.finish_position == 2)
            bsp.loc[first.to_numpy(), "won"] = 0
            bsp.loc[second.to_numpy(), "won"] = 1
        return bsp

    def test_our_label_survives_the_join(self, audit_mod, monkeypatch):
        """The trap: if attach_bsp overwrites `won`, everything always agrees.

        Betfair here disagrees on one race by construction, so an audit that
        compared the exchange with itself would report zero disagreements and
        this test would fail.
        """
        df = _results()
        bsp = self._bsp_from(df, flip_race="r4")
        monkeypatch.setattr(audit_mod, "load_bsp", lambda: bsp)
        out = audit_mod.betfair_checks(df)
        assert out["available"] is True
        assert out["label_disagreements"] == 2, out
        assert out["we_say_won_betfair_says_lost"] == 1
        assert out["betfair_says_won_we_say_lost"] == 1

    def test_full_agreement_reports_zero(self, audit_mod, monkeypatch):
        df = _results()
        monkeypatch.setattr(audit_mod, "load_bsp", lambda: self._bsp_from(df))
        out = audit_mod.betfair_checks(df)
        assert out["label_disagreements"] == 0
        assert out["runners_matched"] == len(df)

    def test_missing_betfair_data_degrades_cleanly(self, audit_mod, monkeypatch):
        monkeypatch.setattr(audit_mod, "load_bsp", lambda: None)
        out = audit_mod.betfair_checks(_results())
        assert out["available"] is False and "reason" in out

    def test_unmatched_runners_are_excluded_not_counted_as_agreeing(
        self, audit_mod, monkeypatch,
    ):
        df = _results()
        bsp = self._bsp_from(df)
        bsp = bsp[bsp.race_date > "2025-03-10"]          # drop early races
        monkeypatch.setattr(audit_mod, "load_bsp", lambda: bsp)
        out = audit_mod.betfair_checks(df)
        assert out["runners_matched"] < len(df)
        assert out["match_rate_pct"] < 100.0


def test_render_is_console_safe(audit_mod, monkeypatch):
    df = _results()
    monkeypatch.setattr(audit_mod, "load_bsp", lambda: None)
    report = {
        "since": "2020-09-15", "until": "2026-10-11",
        "internal": audit_mod.internal_checks(df),
        "vs_betfair_win": {"available": False, "reason": "none"},
        "vs_betfair_place": {"available": False, "reason": "none"},
    }
    audit_mod.render(report).encode("cp1252")
