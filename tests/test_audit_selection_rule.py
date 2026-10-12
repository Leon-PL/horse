"""The selection-rule audit must not invent a pattern that is not there.

This script's whole job is to locate where a betting filter loses money, and
its output will be used to decide whether to change the rule. So the tests
build worlds with a *known* answer and check the audit reports that answer:
a filter that chases longshots must show up as chasing longshots, and a filter
that is harmless must not be accused.
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
        "audit_selection_rule", ROOT / "scripts" / "audit_selection_rule.py",
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["audit_selection_rule"] = module
    spec.loader.exec_module(module)
    return module


def make_bets(n_races=900, runners=8, seed=0, longshot_bias=0.0, pnl_by_odds=None):
    """Per-bet rows shaped like the walk-forward dump.

    ``longshot_bias`` makes the selection rule prefer longer prices, which is
    the hypothesis under test. It is deliberately a *tilt*, not a cliff — a
    bias strong enough to empty the short bands would make detection trivial
    and would not resemble a real value filter. ``pnl_by_odds`` lets a test
    decide the true return as a function of price.
    """
    rng = np.random.default_rng(seed)
    n = n_races * runners
    race_ids = np.repeat(np.arange(n_races), runners)
    odds = rng.uniform(1.6, 45.0, size=n)
    mkt = (1.0 / odds) * 0.92
    won = rng.random(size=n) < (1.0 / odds)

    if pnl_by_odds is None:
        pnl = np.where(won, (odds - 1.0) * 0.98, -1.0)
    else:
        pnl = pnl_by_odds(odds, won)

    # Selection: a value signal, optionally tilted toward longshots.
    score = rng.normal(size=n) + longshot_bias * (odds / 45.0)
    selected = score > np.quantile(score, 0.75)

    return pd.DataFrame({
        "race_id": [f"r{i}" for i in race_ids],
        "race_date": pd.to_datetime("2024-01-01") + pd.to_timedelta(race_ids // 8, unit="D"),
        "track": "T", "horse_name": "H",
        "num_runners": float(runners), "race_type": "Flat",
        "won": won.astype(np.int8),
        "entry_odds": odds,
        "close_odds": odds * rng.uniform(0.95, 1.05, size=n),
        "mkt_imp": mkt,
        "model_prob": mkt * rng.uniform(0.8, 1.3, size=n),
        "combo_prob": np.where(selected, mkt * 1.3, mkt * 0.9),
        "pnl_bsp": pnl,
        "in_band": odds < 50,
        "sel_combo": selected,
        "sel_model": selected,
        "fold": [f"2024Q{1 + (i % 4)}" for i in race_ids],
    })


def _write(tmp_path, bets) -> str:
    path = tmp_path / "bets.parquet"
    bets.to_parquet(path, index=False)
    return str(path)


class TestLongshotDetection:
    def test_a_longshot_chasing_filter_is_visible_in_the_price_bands(
        self, audit_mod, tmp_path,
    ):
        """The hypothesis the script exists to test must actually be detectable."""
        report = audit_mod.audit(_write(tmp_path, make_bets(longshot_bias=2.0, seed=1)))
        bands = report["by_price_band"]
        rates = [r["select_rate"] for r in bands]
        assert rates[-1] > rates[0], (
            f"selection rate should rise with price: {rates}"
        )

    def test_an_unbiased_filter_does_not_look_like_one(self, audit_mod, tmp_path):
        report = audit_mod.audit(_write(tmp_path, make_bets(longshot_bias=0.0, seed=2)))
        rates = [r["select_rate"] for r in report["by_price_band"]]
        # No systematic trend: the first and last buckets should be close.
        assert abs(rates[-1] - rates[0]) < 0.15, rates

    def test_stake_share_sums_to_one(self, audit_mod, tmp_path):
        report = audit_mod.audit(_write(tmp_path, make_bets(seed=3)))
        total = sum(r["share_of_bets"] for r in report["by_price_band"])
        assert total == pytest.approx(1.0, abs=0.01)


class TestRoiAttribution:
    def test_a_filter_that_loses_only_at_long_odds_is_localised_there(
        self, audit_mod, tmp_path,
    ):
        """Make long prices genuinely unprofitable and check the gap appears there."""
        def pnl(odds, won):
            # Short prices fair, long prices badly negative.
            edge = np.where(odds < 10, 0.0, -0.5)
            return np.where(won, (odds - 1.0) * 0.98, -1.0) + edge

        report = audit_mod.audit(
            _write(tmp_path, make_bets(longshot_bias=2.0, seed=4, pnl_by_odds=pnl)),
        )
        bands = {r["band"]: r for r in report["by_price_band"]}
        assert all(b["selected"] > 0 for b in bands.values()), (
            "attribution is meaningless if a band has no selected bets"
        )
        short = bands["1.5-3"]["roi_selected"]
        longest = bands["25-50"]["roi_selected"]
        assert longest < short, f"long band {longest} should trail short band {short}"

    def test_reports_the_filter_s_value_against_no_filter(self, audit_mod, tmp_path):
        report = audit_mod.audit(_write(tmp_path, make_bets(seed=5)))
        overall = report["overall"]
        assert overall["bet_all"]["n"] > overall["selected"]["n"]
        rendered = audit_mod.render(report)
        assert "worth" in rendered and "vs no filter" in rendered
        # The median must be what the table shows; the mean is biased high.
        assert "MEDIAN" in rendered

    def test_intervals_are_race_clustered(self, audit_mod, tmp_path):
        report = audit_mod.audit(_write(tmp_path, make_bets(seed=6)))
        sel = report["overall"]["selected"]
        assert sel["races"] <= sel["n"]
        assert sel["ci_low"] < sel["roi"] < sel["ci_high"]


class TestCandidateRules:
    def test_baseline_and_current_rule_are_both_reported(self, audit_mod, tmp_path):
        report = audit_mod.audit(_write(tmp_path, make_bets(seed=7)))
        names = [r["rule"] for r in report["candidate_rules"]]
        assert "bet all in band (baseline)" in names
        assert "combo_prob > mkt_imp (current)" in names
        assert "INVERSE of current" in names

    def test_bet_all_baseline_is_never_flagged_as_beating_itself(
        self, audit_mod, tmp_path,
    ):
        report = audit_mod.audit(_write(tmp_path, make_bets(seed=8)))
        baseline = next(r for r in report["candidate_rules"]
                        if r["rule"] == "bet all in band (baseline)")
        assert baseline["vs_matched_baseline"] == pytest.approx(0.0, abs=1e-6)
        assert baseline["beats_matched_baseline"] is False

    def test_a_price_restricted_rule_is_judged_against_its_own_universe(
        self, audit_mod, tmp_path,
    ):
        """Comparing a price-capped rule to the whole band inverts the verdict.

        Make short prices profitable and long prices terrible, with a filter
        that adds nothing. Against the whole band an "odds < 12" rule looks
        brilliant purely for avoiding long prices; against its own universe it
        must show no gain.
        """
        def pnl(odds, won):
            edge = np.where(odds < 12, 0.20, -0.80)
            return np.where(won, (odds - 1.0) * 0.98, -1.0) + edge

        report = audit_mod.audit(
            _write(tmp_path, make_bets(n_races=6000, seed=13, pnl_by_odds=pnl)),
        )
        rule = next(r for r in report["candidate_rules"] if r["rule"] == "current, odds < 12")
        assert rule["vs_whole_band"] > 10.0, (
            "against the whole band the price cap should look like a big win"
        )
        # The point of the matched baseline: it strips out the gain that came
        # from the price restriction rather than from the filter.
        inflation = rule["vs_whole_band"] - rule["vs_matched_baseline"]
        assert inflation > 10.0, (
            "the whole-band comparison should be inflated relative to the "
            f"matched one; got vs_whole_band={rule['vs_whole_band']}, "
            f"vs_matched={rule['vs_matched_baseline']}"
        )

    def test_an_odds_capped_rule_wins_when_long_odds_are_the_problem(
        self, audit_mod, tmp_path,
    ):
        def pnl(odds, won):
            edge = np.where(odds < 12, 0.10, -0.60)
            return np.where(won, (odds - 1.0) * 0.98, -1.0) + edge

        report = audit_mod.audit(
            _write(tmp_path, make_bets(longshot_bias=2.0, seed=9, pnl_by_odds=pnl)),
        )
        rules = {r["rule"]: r for r in report["candidate_rules"]}
        assert rules["current, odds < 12"]["roi"] > rules["combo_prob > mkt_imp (current)"]["roi"]


class TestRobustness:
    def test_only_in_band_rows_are_analysed(self, audit_mod, tmp_path):
        bets = make_bets(seed=10)
        bets.loc[bets.index[:500], "in_band"] = False
        report = audit_mod.audit(_write(tmp_path, bets))
        assert report["rows_in_band"] == len(bets) - 500

    def test_empty_selection_does_not_crash_the_report(self, audit_mod, tmp_path):
        bets = make_bets(seed=11)
        bets["sel_combo"] = False
        report = audit_mod.audit(_write(tmp_path, bets))
        assert report["selected"] == 0
        assert report["overall"]["selected"]["roi"] is None

    def test_missing_file_is_a_clear_message_not_a_traceback(self, audit_mod, monkeypatch, capsys):
        monkeypatch.setattr(sys, "argv", ["audit", "--bets", "does/not/exist.parquet"])
        assert audit_mod.main() == 2
        assert "Generate them with" in capsys.readouterr().out

    def test_render_is_console_safe(self, audit_mod, tmp_path):
        report = audit_mod.audit(_write(tmp_path, make_bets(seed=12)))
        audit_mod.render(report).encode("cp1252")
