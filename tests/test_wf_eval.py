"""Tests for the shared walk-forward evaluation primitives.

Each test pins down one of the errors that motivated the module: folds without
an embargo, second stages fitted on in-sample predictions, and bootstrap
intervals that ignore clustering. These are measurement bugs, so a regression
here does not break the app — it quietly changes what the project believes.
"""

import numpy as np
import pandas as pd
import pytest

import config
from src.wf_eval import (
    bootstrap_roi_ci,
    check_combination_nesting,
    expanding_period_folds,
    fold_summary,
    oof_predict_expanding,
    race_logloss,
)


class TestExpandingFolds:
    @staticmethod
    def _dates(n_quarters=12, per_quarter=400):
        starts = pd.period_range("2023Q1", periods=n_quarters, freq="Q")
        out = []
        for start in starts:
            span = (start.end_time - start.start_time).days
            out.append(pd.Series(
                start.start_time + pd.to_timedelta(
                    np.linspace(0, span, per_quarter).astype(int), unit="D",
                )
            ))
        return pd.concat(out, ignore_index=True)

    def test_train_is_strictly_before_test(self):
        dates = self._dates()
        for fold in expanding_period_folds(dates, min_train_periods=4, min_train_rows=10):
            assert dates[fold.train].max() < dates[fold.test].min()

    def test_embargo_leaves_a_real_gap(self):
        dates = self._dates()
        folds = expanding_period_folds(
            dates, min_train_periods=4, min_train_rows=10, embargo_days=14,
        )
        assert folds
        for fold in folds:
            gap = (dates[fold.test].min() - dates[fold.train].max()).days
            assert gap >= 14, f"{fold.label} gap was {gap} days"
            assert fold.n_purged > 0, "the embargo should have dropped rows"

    def test_embargo_defaults_to_the_production_purge(self, monkeypatch):
        """Research protocol must match the training path, or numbers diverge."""
        monkeypatch.setattr(config, "PURGE_DAYS", 30)
        dates = self._dates()
        folds = expanding_period_folds(dates, min_train_periods=4, min_train_rows=10)
        assert folds
        for fold in folds:
            assert (dates[fold.test].min() - dates[fold.train].max()).days >= 30

    def test_window_expands_and_tests_each_period_once(self):
        dates = self._dates(n_quarters=10)
        folds = expanding_period_folds(dates, min_train_periods=4, min_train_rows=10)
        assert [f.label for f in folds] == [
            str(p) for p in pd.period_range("2024Q1", periods=6, freq="Q")
        ]
        sizes = [f.n_train for f in folds]
        assert sizes == sorted(sizes), "training window must only grow"

    def test_folds_never_overlap(self):
        dates = self._dates()
        for fold in expanding_period_folds(dates, min_train_periods=4, min_train_rows=10):
            assert not (fold.train & fold.test).any()

    def test_insufficient_history_yields_no_folds(self):
        dates = self._dates(n_quarters=3, per_quarter=5)
        assert expanding_period_folds(dates, min_train_periods=8) == []


class TestOutOfFoldPredictions:
    """The fix for the in-sample stacking bug."""

    @staticmethod
    def _frame(n=1200):
        rng = np.random.default_rng(0)
        return pd.DataFrame({
            "race_date": pd.date_range("2023-01-01", periods=n, freq="D"),
            "x": rng.normal(size=n),
            "y": rng.integers(0, 2, size=n),
        })

    def test_predictions_never_come_from_a_model_that_saw_the_row(self):
        frame = self._frame()
        seen: list[set] = []

        def fit_predict(train_df, predict_df):
            seen.append(set(train_df.index))
            overlap = seen[-1] & set(predict_df.index)
            assert not overlap, f"{len(overlap)} prediction rows were in training"
            return np.full(len(predict_df), 0.5)

        oof_predict_expanding(frame, fit_predict=fit_predict, n_splits=4, min_train_rows=10)
        assert len(seen) == 3, "every block after the first should be predicted"

    def test_training_rows_respect_the_embargo(self):
        frame = self._frame()
        gaps = []

        def fit_predict(train_df, predict_df):
            gaps.append((predict_df["race_date"].min() - train_df["race_date"].max()).days)
            return np.full(len(predict_df), 0.5)

        oof_predict_expanding(
            frame, fit_predict=fit_predict, n_splits=4,
            embargo_days=21, min_train_rows=10,
        )
        assert gaps and all(g >= 21 for g in gaps), gaps

    def test_earliest_block_is_nan_not_silently_in_sample(self):
        frame = self._frame()
        oof = oof_predict_expanding(
            frame, fit_predict=lambda tr, pr: np.full(len(pr), 0.5),
            n_splits=4, min_train_rows=10,
        )
        assert np.isnan(oof).any(), "the first block has no history and must be NaN"
        assert np.isfinite(oof).sum() > 0.5 * len(frame)

    def test_predictions_land_on_the_right_rows(self):
        """Blocks are built in date order; results must map back to input order."""
        frame = self._frame(n=400).sample(frac=1.0, random_state=1).reset_index(drop=True)
        oof = oof_predict_expanding(
            frame,
            # Encode each row's identity so misalignment is detectable.
            fit_predict=lambda tr, pr: pr["x"].to_numpy(),
            n_splits=4, min_train_rows=10,
        )
        finite = np.isfinite(oof)
        np.testing.assert_allclose(oof[finite], frame["x"].to_numpy()[finite])

    def test_wrong_prediction_length_is_an_error_not_a_misalignment(self):
        frame = self._frame(n=400)
        with pytest.raises(ValueError, match="predictions"):
            oof_predict_expanding(
                frame, fit_predict=lambda tr, pr: np.zeros(len(pr) - 1),
                n_splits=4, min_train_rows=10,
            )


class TestClusteredBootstrap:
    def test_matches_the_iid_bootstrap_when_every_bet_is_its_own_race(self):
        rng = np.random.default_rng(0)
        pnl = rng.normal(size=500)
        clustered = bootstrap_roi_ci(pnl, clusters=np.arange(500), seed=7)
        iid = bootstrap_roi_ci(pnl, seed=7)
        assert clustered["ci_low"] == pytest.approx(iid["ci_low"], abs=0.5)
        assert clustered["ci_high"] == pytest.approx(iid["ci_high"], abs=0.5)

    def test_correlated_bets_widen_the_interval(self):
        """The whole point: ignoring clustering reports false significance."""
        rng = np.random.default_rng(1)
        # 100 races; every bet in a race shares that race's outcome shock.
        race_effect = rng.normal(scale=1.0, size=100)
        clusters = np.repeat(np.arange(100), 10)
        pnl = np.repeat(race_effect, 10) + rng.normal(scale=0.05, size=1000)

        iid = bootstrap_roi_ci(pnl, seed=3)
        clustered = bootstrap_roi_ci(pnl, clusters=clusters, seed=3)
        iid_width = iid["ci_high"] - iid["ci_low"]
        clustered_width = clustered["ci_high"] - clustered["ci_low"]
        assert clustered_width > 2.5 * iid_width, (
            f"clustered width {clustered_width:.2f} vs iid {iid_width:.2f}: "
            "the iid bootstrap understates uncertainty on correlated bets"
        )

    def test_point_estimate_is_the_plain_mean_either_way(self):
        pnl = np.array([1.0, -1.0, 2.0, -0.5])
        clusters = np.array(["a", "a", "b", "b"])
        assert bootstrap_roi_ci(pnl, clusters=clusters)["mean"] == pytest.approx(
            pnl.mean() * 100,
        )

    def test_reports_cluster_count_and_flags_clustering(self):
        pnl = np.ones(10)
        out = bootstrap_roi_ci(pnl, clusters=np.repeat([1, 2], 5))
        assert out["n"] == 10 and out["n_clusters"] == 2 and out["clustered"] is True
        assert bootstrap_roi_ci(pnl)["clustered"] is False

    def test_empty_input_is_not_an_exception(self):
        out = bootstrap_roi_ci([], clusters=[])
        assert out["mean"] is None and out["n"] == 0

    def test_is_reproducible_for_a_given_seed(self):
        pnl = np.random.default_rng(5).normal(size=300)
        clusters = np.repeat(np.arange(30), 10)
        a = bootstrap_roi_ci(pnl, clusters=clusters, seed=11)
        b = bootstrap_roi_ci(pnl, clusters=clusters, seed=11)
        assert a == b

    def test_chunking_does_not_change_the_distribution(self, monkeypatch):
        """The statistic is a ratio of sums, so chunked draws must be identical.

        Chunking exists only to bound memory on a long backtest; if it altered
        the draws, every reported interval would depend on the dataset size.
        """
        import src.wf_eval as wf

        pnl = np.random.default_rng(2).normal(size=400)
        clusters = np.repeat(np.arange(40), 10)

        one_chunk = bootstrap_roi_ci(pnl, clusters=clusters, n_boot=500, seed=4)
        # 40 clusters x 500 draws = 20k cells; a 100-cell budget forces ~200
        # chunks over the same stream of random draws.
        monkeypatch.setattr(wf, "BOOT_CHUNK_CELLS", 100)
        many_chunks = bootstrap_roi_ci(pnl, clusters=clusters, n_boot=500, seed=4)
        assert one_chunk == many_chunks


class TestFoldSummary:
    def test_reports_dispersion_not_just_the_pooled_number(self):
        out = fold_summary([5.0, -3.0, 1.0, 12.0, -8.0])
        assert out["n_folds"] == 5 and out["folds_positive"] == 3
        assert out["worst"] == -8.0 and out["best"] == 12.0
        assert out["std"] > 0

    def test_ignores_non_finite_folds(self):
        assert fold_summary([1.0, float("nan"), 3.0])["n_folds"] == 2

    def test_empty_is_safe(self):
        assert fold_summary([])["n_folds"] == 0


class TestCombinationNesting:
    def test_accepts_a_combination_at_least_as_good_as_the_market(self):
        assert check_combination_nesting(0.95, 1.00)["ok"] is True

    def test_fires_when_the_combination_is_worse_than_what_it_nests(self):
        verdict = check_combination_nesting(1.20, 1.00)
        assert verdict["ok"] is False
        assert verdict["excess_logloss"] == pytest.approx(0.20)
        assert "suspect the fit" in verdict["reason"]
        assert "does not prove a pipeline error" in verdict["reason"]

    def test_tolerates_noise_sized_differences(self):
        assert check_combination_nesting(1.005, 1.000, tolerance=0.01)["ok"] is True

    def test_missing_scores_do_not_assert_a_verdict(self):
        assert check_combination_nesting(float("nan"), 1.0)["ok"] is True


class TestRaceLogLoss:
    def test_scores_the_probability_given_to_the_actual_winner(self):
        probs = [0.5, 0.3, 0.2]
        loss = race_logloss(probs, [1, 0, 0], ["r1", "r1", "r1"])
        assert loss == pytest.approx(-np.log(0.5))

    def test_a_sharper_correct_model_scores_better(self):
        races = ["r1"] * 3 + ["r2"] * 3
        won = [1, 0, 0, 1, 0, 0]
        sharp = [0.8, 0.1, 0.1, 0.7, 0.2, 0.1]
        flat = [0.34, 0.33, 0.33, 0.34, 0.33, 0.33]
        assert race_logloss(sharp, won, races) < race_logloss(flat, won, races)

    def test_skips_races_without_exactly_one_winner(self):
        races = ["r1", "r1", "r2", "r2"]
        # r2 is a void/dead-heat style record with two winners.
        loss = race_logloss([0.6, 0.4, 0.5, 0.5], [1, 0, 1, 1], races)
        assert loss == pytest.approx(-np.log(0.6))

    def test_no_usable_race_is_nan_not_zero(self):
        assert np.isnan(race_logloss([0.5, 0.5], [0, 0], ["r1", "r1"]))


def test_development_and_holdout_masks_partition_with_a_purge_gap(monkeypatch):
    """Research rows end PURGE_DAYS before the golden hold-out; none overlap."""
    import config
    from src.wf_eval import development_mask, holdout_mask

    monkeypatch.setattr(config, "GOLDEN_HOLDOUT_START", "2026-06-11")
    monkeypatch.setattr(config, "PURGE_DAYS", 7)
    dates = pd.Series(pd.date_range("2026-05-30", "2026-06-20", freq="D"))
    dev, hold = development_mask(dates), holdout_mask(dates)
    assert not (dev & hold).any()
    assert dates[dev].max() == pd.Timestamp("2026-06-03")
    assert dates[hold].min() == pd.Timestamp("2026-06-11")
    gap = dates[~dev & ~hold]
    assert len(gap) == 7  # the purge: 2026-06-04 .. 2026-06-10


def test_bootstrap_median_ci_is_wider_when_values_cluster_by_race():
    """Bets in one race share an outcome; the CI must respect that."""
    from src.wf_eval import bootstrap_median_ci

    rng = np.random.default_rng(1)
    race_effect = rng.normal(0, 0.05, size=300)
    values_clustered = np.repeat(1.0 + race_effect, 10) + rng.normal(0, 0.005, 3000)
    races = np.repeat(np.arange(300), 10)
    clustered = bootstrap_median_ci(values_clustered, clusters=races, n_boot=400)
    naive = bootstrap_median_ci(values_clustered, clusters=np.arange(3000), n_boot=400)
    assert clustered["n_clusters"] == 300 and naive["n_clusters"] == 3000
    width = lambda r: r["ci_high"] - r["ci_low"]
    assert width(clustered) > 1.5 * width(naive)
    assert clustered["ci_low"] < clustered["estimate"] < clustered["ci_high"]


def test_bootstrap_median_ci_geometric_mean_and_point_estimate():
    from src.wf_eval import bootstrap_median_ci

    values = np.array([0.5, 1.0, 2.0, 1.0])
    out = bootstrap_median_ci(values, clusters=np.arange(4), n_boot=200, log_mean=True)
    assert out["estimate"] == pytest.approx(1.0)
    med = bootstrap_median_ci(values, clusters=np.arange(4), n_boot=200)
    assert med["estimate"] == pytest.approx(1.0)


