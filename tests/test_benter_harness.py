"""The Benter combination harness must not be biased against its hypothesis.

The previous harness fitted the market blend on *in-sample* model predictions
and, on that basis, the market-combination lead was recorded as dead. These
tests build a synthetic world where a genuine orthogonal edge exists by
construction, then check that

* the old in-sample fitting really does distort the fitted weights in the
  direction that would bury such an edge, and
* the rewritten harness recovers a sane combination, with the nesting tripwire
  passing, purged folds and race-clustered intervals.

The first group is the evidence that the recorded negative result came from the
measurement rather than from the data.
"""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.model import fit_market_anchor, market_probs_from_odds
from src.wf_eval import oof_predict_expanding, race_logloss

ROOT = Path(__file__).resolve().parent.parent

FEATURES = [f"feat_{i}" for i in range(4)]

#: Small, fast model for tests. The bias being measured is a property of how
#: the second stage is fitted, not of the first stage's capacity.
TEST_WIN_PARAMS = dict(
    n_estimators=60, num_leaves=15, max_depth=5, learning_rate=0.08,
    min_child_samples=20, subsample=0.8, colsample_bytree=0.8,
    objective="binary", subsample_freq=1, n_jobs=-1, verbose=-1, random_state=42,
)


@pytest.fixture(scope="module")
def harness():
    spec = importlib.util.spec_from_file_location(
        "wf_benter_combination", ROOT / "scripts" / "wf_benter_combination.py",
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["wf_benter_combination"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def fast_model(harness, monkeypatch):
    monkeypatch.setattr(harness, "WIN_PARAMS", TEST_WIN_PARAMS)


@pytest.fixture
def synthetic_feature_set(monkeypatch):
    """Let get_feature_columns see the synthetic feat_* columns.

    The project allowlist names real features, none of which exist here.
    """
    import config
    monkeypatch.setattr(config, "FEATURE_ALLOWLIST", list(FEATURES))
    return FEATURES


def synthetic_races(n_races=9000, runners=6, seed=0, races_per_day=8,
                    market_noise=0.9, feature_noise=1.4):
    """A world where form features carry information the market misses.

    Each runner has a latent strength. The market sees it with noise and prices
    accordingly; the feature columns see a *different* noisy view. The winner is
    drawn from the true strengths, so a model fitted on the features holds
    information genuinely orthogonal to the price — a correctly fitted
    combination should therefore beat the market.

    Races are packed *densely* in time (``races_per_day``) so that quarterly
    folds and their purged inner splits each have enough rows to train on.
    """
    rng = np.random.default_rng(seed)
    n = n_races * runners
    strength = rng.normal(size=n)
    race_index = np.repeat(np.arange(n_races), runners)

    frame = pd.DataFrame({
        "race_id": [f"r{i}" for i in race_index],
        "race_date": pd.to_datetime("2020-01-01") + pd.to_timedelta(
            race_index // races_per_day, unit="D",
        ),
    })

    # Market price: the latent strength seen through its own noise.
    market_view = np.exp(strength + rng.normal(scale=market_noise, size=n))
    market_prob = market_view / pd.Series(market_view).groupby(race_index).transform("sum")
    frame["mkt_imp"] = market_prob.to_numpy()
    # Overround, so 1/odds is not already a clean probability.
    frame["_entry"] = (1.0 / frame["mkt_imp"]) * 0.92
    frame["_close"] = frame["_entry"] * rng.uniform(0.9, 1.1, size=n)

    for i in range(len(FEATURES)):
        frame[FEATURES[i]] = strength + rng.normal(scale=feature_noise, size=n)

    # Winner drawn from the true strengths.
    true_view = np.exp(strength * 1.2)
    true_prob = (true_view / pd.Series(true_view).groupby(race_index).transform("sum")).to_numpy()
    won = np.zeros(n, dtype=int)
    for r in range(n_races):
        start = r * runners
        p = true_prob[start:start + runners]
        won[start + rng.choice(runners, p=p / p.sum())] = 1
    frame["won"] = won
    return frame


def _fit_win_model(harness, train, predict, params=TEST_WIN_PARAMS):
    import lightgbm as lgb
    model = lgb.LGBMClassifier(**params)
    model.fit(train[FEATURES], train["won"])
    return model.predict_proba(predict[FEATURES])[:, 1]


def _fit_anchor_on(frame: pd.DataFrame, probs, harness) -> dict:
    """Fit (alpha, beta) given per-runner model probabilities."""
    work = frame.assign(_p=harness._normalise_per_race(probs, frame["race_id"]))
    ordered, groups = harness._race_blocks(work)
    market, valid = market_probs_from_odds(ordered["_entry"].to_numpy(), groups)
    return fit_market_anchor(
        ordered["_p"].to_numpy(), market, groups,
        ordered["won"].to_numpy(), race_valid=valid,
    )


class TestInSampleStackingBias:
    """The bug: fitting the blend on in-sample predictions distorts it."""

    def test_in_sample_fitting_overweights_the_model(self, harness):
        frame = synthetic_races(n_races=4000, seed=1)

        # OLD behaviour: the model predicts the very rows it was trained on.
        anchor_in_sample = _fit_anchor_on(
            frame, _fit_win_model(harness, frame, frame), harness,
        )
        # NEW behaviour: out-of-fold predictions only.
        oof = oof_predict_expanding(
            frame,
            fit_predict=lambda tr, pr: _fit_win_model(harness, tr, pr),
            n_splits=4, min_train_rows=500,
        )
        usable = np.isfinite(oof)
        anchor_oof = _fit_anchor_on(frame[usable], oof[usable], harness)

        assert anchor_in_sample is not None and anchor_oof is not None
        # An over-sharp model probability buys more weight than it deserves.
        assert anchor_in_sample["alpha"] > anchor_oof["alpha"], (
            f"in-sample alpha {anchor_in_sample['alpha']:.3f} should exceed "
            f"out-of-fold alpha {anchor_oof['alpha']:.3f}"
        )

    def test_the_distorted_weights_score_worse_out_of_sample(self, harness):
        """The bias is not cosmetic: it costs real out-of-sample accuracy."""
        train = synthetic_races(n_races=4000, seed=2)
        test = synthetic_races(n_races=1500, seed=99)

        anchor_bad = _fit_anchor_on(
            train, _fit_win_model(harness, train, train), harness,
        )
        oof = oof_predict_expanding(
            train,
            fit_predict=lambda tr, pr: _fit_win_model(harness, tr, pr),
            n_splits=4, min_train_rows=500,
        )
        usable = np.isfinite(oof)
        anchor_good = _fit_anchor_on(train[usable], oof[usable], harness)

        test_probs = harness._normalise_per_race(
            _fit_win_model(harness, train, test), test["race_id"],
        )
        loss_bad = race_logloss(
            harness._anchor(test_probs, test["_entry"], test, anchor_bad),
            test["won"], test["race_id"],
        )
        loss_good = race_logloss(
            harness._anchor(test_probs, test["_entry"], test, anchor_good),
            test["won"], test["race_id"],
        )
        assert loss_good < loss_bad, (
            f"out-of-fold-fitted blend ({loss_good:.4f}) must beat the "
            f"in-sample-fitted blend ({loss_bad:.4f})"
        )


class TestRewrittenHarness:
    def test_recovers_a_combination_that_beats_the_market(
        self, harness, synthetic_feature_set, fast_model,
    ):
        """With a real orthogonal edge present, the harness must find it."""
        report = harness.run(
            synthetic_races(n_races=9000, seed=3),
            max_folds=3, n_boot=300, min_train_rows=3000,
        )
        assert "error" not in report, report
        assert report["nesting_check"]["ok"], report["nesting_check"]["reason"]
        assert report["alpha"]["mean"] > 0, (
            "a genuinely informative model should earn positive weight, got "
            f"{report['alpha']['mean']}"
        )

    def test_baselines_cover_every_fold_not_just_combination_folds(
        self, harness, synthetic_feature_set, fast_model,
    ):
        """The old version compared ROIs computed over different quarters."""
        report = harness.run(
            synthetic_races(n_races=9000, seed=4),
            max_folds=3, n_boot=200, min_train_rows=3000,
        )
        assert "error" not in report, report
        assert report["bet_all"]["n"] > report["combo"]["n"]
        assert report["bet_all"]["n"] > report["model_only"]["n"]

    def test_confidence_intervals_are_race_clustered(
        self, harness, synthetic_feature_set, fast_model,
    ):
        report = harness.run(
            synthetic_races(n_races=9000, seed=5),
            max_folds=2, n_boot=200, min_train_rows=3000,
        )
        assert "error" not in report, report
        for key in ("combo", "model_only", "bet_all"):
            assert report[key]["clustered"] is True, f"{key} CI was not clustered"
            assert report[key]["n_clusters"] <= report[key]["n"]

    def test_folds_carry_the_production_purge(
        self, harness, synthetic_feature_set, fast_model,
    ):
        report = harness.run(
            synthetic_races(n_races=9000, seed=6),
            max_folds=2, n_boot=100, min_train_rows=3000,
        )
        assert "error" not in report, report
        assert all(f["purged_rows"] > 0 for f in report["per_fold"]), (
            "every fold should drop training rows inside the embargo"
        )

    def test_nesting_tripwire_fails_the_run_when_the_fit_is_broken(self, harness):
        """A combination worse than the market it nests must not pass silently."""
        verdict = harness.check_combination_nesting(1.30, 1.00)
        assert verdict["ok"] is False
        rendered = harness.render({
            "nesting_check": verdict,
            "combo": {"mean": -3.0, "ci_low": -5.0, "ci_high": -1.0,
                      "n": 100, "n_clusters": 90},
            "model_only": {"mean": -2.0, "ci_low": -4.0, "ci_high": 0.0, "n": 120},
            "bet_all": {"mean": -2.0, "ci_low": -3.0, "ci_high": -1.0, "n": 900},
            "mean_clv": 1.0,
            "folds": {"folds_positive": 0, "n_folds": 4, "median": -3.0,
                      "worst": -5.0, "best": -1.0},
            "alpha": {"mean": 0.1, "positive_folds": 2, "n_folds": 4},
        })
        assert "Do NOT read this run" in rendered
        assert "FAIL" in rendered
