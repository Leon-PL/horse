"""The orthogonality triage decides which features are worth an expensive test.

A false positive here costs a walk-forward run. A false *negative* discards a
real signal permanently and silently, which is worse. So the tests build cases
with a known answer: a feature that only proxies the price must be rejected
even though it correlates with the residual, and a feature carrying genuine
orthogonal information must survive.
"""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def triage_mod():
    spec = importlib.util.spec_from_file_location(
        "triage_features", ROOT / "scripts" / "triage_features.py",
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["triage_features"] = module
    spec.loader.exec_module(module)
    return module


class TestPartialCorrelation:
    """The core guard: separate real signal from a price proxy."""

    def test_a_pure_price_proxy_is_stripped_out(self, triage_mod):
        """The reason this script does not use a raw correlation.

        The residual has real structure against price (favourite-longshot
        bias), so a feature that merely restates the price will correlate with
        the residual while carrying no information of its own.
        """
        rng = np.random.default_rng(0)
        market = rng.normal(size=20_000)
        # Residual is partly explained by price - as it is in reality.
        residual = -0.35 * market + rng.normal(scale=1.0, size=20_000)
        # Feature is nothing but a noisy copy of the price.
        feature = market + rng.normal(scale=0.3, size=20_000)

        raw, partial, n = triage_mod._partial_corr(feature, residual, market)
        assert abs(raw) > 0.2, f"raw correlation should look interesting, got {raw}"
        assert abs(partial) < 0.02, (
            f"a pure price proxy must have ~no partial correlation, got {partial}"
        )
        assert n == 20_000

    def test_genuine_orthogonal_signal_survives(self, triage_mod):
        rng = np.random.default_rng(1)
        market = rng.normal(size=20_000)
        signal = rng.normal(size=20_000)              # independent of price
        residual = -0.35 * market + 0.25 * signal + rng.normal(scale=1.0, size=20_000)
        feature = signal + rng.normal(scale=0.4, size=20_000)

        _, partial, _ = triage_mod._partial_corr(feature, residual, market)
        assert partial > 0.1, f"real signal should survive partialling, got {partial}"

    def test_degenerate_inputs_do_not_raise(self, triage_mod):
        constant = np.ones(500)
        noise = np.random.default_rng(2).normal(size=500)
        raw, partial, _ = triage_mod._partial_corr(constant, noise, noise)
        assert np.isnan(raw) and np.isnan(partial)

    def test_too_few_rows_is_nan_not_a_lucky_correlation(self, triage_mod):
        small = np.arange(20, dtype=float)
        raw, partial, n = triage_mod._partial_corr(small, small, small)
        assert np.isnan(raw) and np.isnan(partial) and n == 20

    def test_nan_rows_are_excluded_pairwise(self, triage_mod):
        rng = np.random.default_rng(3)
        x = rng.normal(size=5_000)
        y = x * 0.5 + rng.normal(scale=0.5, size=5_000)
        z = rng.normal(size=5_000)
        x[:2_000] = np.nan
        _, _, n = triage_mod._partial_corr(x, y, z)
        assert n == 3_000


class TestWithinRaceCentring:
    def test_race_means_are_removed(self, triage_mod):
        races = np.array([0, 0, 0, 1, 1, 1])
        values = np.array([1.0, 2.0, 3.0, 11.0, 12.0, 13.0])
        out = triage_mod._within_race_demean(values, races)
        np.testing.assert_allclose(out, [-1, 0, 1, -1, 0, 1])

    def test_a_feature_constant_within_each_race_carries_no_information(self, triage_mod):
        """Only relative standing in the field can pay."""
        races = np.repeat(np.arange(200), 8)
        rng = np.random.default_rng(4)
        race_level = np.repeat(rng.normal(size=200), 8)   # same for every runner
        out = triage_mod._within_race_demean(race_level, races)
        assert np.allclose(out, 0.0, atol=1e-9)


class TestFamilyCollapse:
    @pytest.mark.parametrize("column,expected", [
        ("atr_finishing_speed_pct_avg_3", "atr_finishing_speed_pct"),
        ("atr_finishing_speed_pct_avg_5_vs_best_in_race", "atr_finishing_speed_pct"),
        ("atr_max_speed_best_3_z_in_race", "atr_max_speed"),
        ("atr_closing_speed_trend", "atr_closing_speed"),
        ("atr_cmt_trouble_rate_5_rank_pct_in_race", "atr_cmt_trouble"),
        ("atr_speed_score", "atr_speed_score"),
        ("atr_early_pace_avg_3_vs_field", "atr_early_pace"),
    ])
    def test_variants_collapse_to_one_signal(self, triage_mod, column, expected):
        assert triage_mod._family(column) == expected

    def test_distinct_signals_stay_distinct(self, triage_mod):
        families = {
            triage_mod._family(c) for c in
            ("atr_max_speed_avg_3", "atr_closing_speed_avg_3",
             "atr_peak_stride_length_avg_3", "atr_peak_stride_freq_avg_3")
        }
        assert len(families) == 4


class TestMultipleTesting:
    def test_q_values_are_never_below_their_p_values(self, triage_mod):
        p = np.array([0.001, 0.01, 0.04, 0.2, 0.9])
        q = triage_mod._bh_qvalues(p)
        assert np.all(q >= p - 1e-12)
        assert np.all(q <= 1.0)

    def test_q_values_are_monotone_in_p(self, triage_mod):
        p = np.sort(np.random.default_rng(5).uniform(size=50))
        q = triage_mod._bh_qvalues(p)
        assert np.all(np.diff(q) >= -1e-12)

    def test_nans_are_preserved(self, triage_mod):
        q = triage_mod._bh_qvalues(np.array([0.01, np.nan, 0.5]))
        assert np.isnan(q[1]) and np.isfinite(q[0]) and np.isfinite(q[2])


class TestEndToEnd:
    """Drive triage() against a synthetic store via load_candidates."""

    @staticmethod
    def _frame(n_races=1500, runners=8, seed=0):
        rng = np.random.default_rng(seed)
        n = n_races * runners
        race_codes = np.repeat(np.arange(n_races), runners)
        strength = rng.normal(size=n)

        market_view = np.exp(strength + rng.normal(scale=0.8, size=n))
        mkt = market_view / pd.Series(market_view).groupby(race_codes).transform("sum")

        # A signal the market does NOT see, which moves the outcome.
        hidden = rng.normal(size=n)
        true_view = np.exp(strength * 1.1 + hidden * 0.6)
        true_prob = (true_view / pd.Series(true_view).groupby(race_codes).transform("sum")).to_numpy()
        won = np.zeros(n, dtype=np.int8)
        for r in range(n_races):
            start = r * runners
            p = true_prob[start:start + runners]
            won[start + rng.choice(runners, p=p / p.sum())] = 1

        return pd.DataFrame({
            "race_id": [f"r{i}" for i in race_codes],
            "race_date": pd.to_datetime("2024-01-01") + pd.to_timedelta(race_codes // 6, unit="D"),
            "won": won,
            "mkt_imp": mkt.to_numpy(),
            "has_xx_history": 1,
            # Carries the hidden signal: should be shortlisted.
            "xx_real_signal_avg_3": hidden + rng.normal(scale=0.5, size=n),
            # Only restates the price: must be rejected.
            "xx_price_proxy_avg_3": np.log(mkt.to_numpy()) + rng.normal(scale=0.2, size=n),
            # Pure noise: must be rejected.
            "xx_noise_avg_3": rng.normal(size=n),
        })

    @pytest.fixture
    def patched(self, triage_mod, monkeypatch):
        frame = self._frame()
        candidates = ["xx_real_signal_avg_3", "xx_price_proxy_avg_3", "xx_noise_avg_3"]
        monkeypatch.setattr(triage_mod, "load_candidates",
                            lambda prefix: (frame, candidates))
        return triage_mod.triage("xx_")

    def test_real_signal_is_shortlisted(self, patched):
        verdicts = {r["feature"]: r["verdict"] for r in patched["all_columns"]}
        assert verdicts["xx_real_signal_avg_3"] == "shortlist", patched["all_columns"]

    def test_price_proxy_and_noise_are_rejected(self, patched):
        """Either rejection reason is fine; being shortlisted is not.

        "priced" means the effect is too small to matter, "noise" means it is
        not distinguishable from zero after the multiple-testing correction.
        Both send the candidate to the ledger rather than to a walk-forward.
        """
        verdicts = {r["feature"]: r["verdict"] for r in patched["all_columns"]}
        assert verdicts["xx_price_proxy_avg_3"] in {"priced", "noise"}
        assert verdicts["xx_noise_avg_3"] in {"priced", "noise"}
        assert patched["shortlist"] == [
            r for r in patched["shortlist"] if r["family"] == "xx_real_signal"
        ], "only the genuinely orthogonal signal should be shortlisted"

    def test_noise_at_small_n_is_not_shortlisted_on_effect_size_alone(self, patched):
        """The sample-size trap: 1/sqrt(12000) = 0.009, so 0.01 is reachable.

        Pure noise here scores a partial correlation above the effect-size
        threshold. Only the q-value gate keeps it out.
        """
        noise = next(r for r in patched["all_columns"]
                     if r["feature"] == "xx_noise_avg_3")
        assert abs(noise["partial_corr"]) >= 0.01, (
            "this test is only meaningful while noise clears the effect-size gate"
        )
        assert noise["verdict"] != "shortlist"
        assert noise["q"] > 0.05

    def test_price_proxy_looks_interesting_until_the_price_is_controlled_for(self, patched):
        """The finding that justifies the partial correlation."""
        proxy = next(r for r in patched["all_columns"]
                     if r["feature"] == "xx_price_proxy_avg_3")
        assert abs(proxy["raw_corr"]) > abs(proxy["partial_corr"]) * 3, (
            f"raw {proxy['raw_corr']} should far exceed partial {proxy['partial_corr']}"
        )

    def test_report_is_renderable_and_console_safe(self, triage_mod, patched):
        triage_mod.render(patched).encode("cp1252")

    def test_sparse_columns_are_flagged_not_scored(self, triage_mod, monkeypatch):
        frame = self._frame()
        frame["xx_sparse_avg_3"] = np.nan
        frame.loc[frame.index[:50], "xx_sparse_avg_3"] = 1.0
        monkeypatch.setattr(triage_mod, "load_candidates",
                            lambda prefix: (frame, ["xx_sparse_avg_3", "xx_noise_avg_3"]))
        report = triage_mod.triage("xx_", min_coverage=0.5)
        sparse = next(r for r in report["all_columns"] if r["feature"] == "xx_sparse_avg_3")
        assert sparse["verdict"] == "too sparse"
        assert sparse["partial_corr"] is None
