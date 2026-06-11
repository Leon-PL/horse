"""Tests for the Glicko-1 horse rating system."""

import numpy as np
import pandas as pd
import pytest

from src.ratings import (
    GLICKO_RATING_INIT,
    GLICKO_RD_INIT,
    GLICKO_RD_MIN,
    _glicko_race_update,
    compute_glicko_features,
)


def _race_df(rows):
    return pd.DataFrame(rows)


class TestGlickoRaceUpdate:
    def test_winner_gains_loser_loses(self):
        r = np.array([1500.0, 1500.0, 1500.0])
        rd = np.array([200.0, 200.0, 200.0])
        fp = np.array([1.0, 2.0, 3.0])
        new_r, new_rd = _glicko_race_update(r, rd, fp)
        assert new_r[0] > 1500.0
        assert new_r[2] < 1500.0
        # mid finisher roughly unchanged
        assert abs(new_r[1] - 1500.0) < abs(new_r[0] - 1500.0)

    def test_rd_shrinks_with_evidence(self):
        r = np.array([1500.0, 1500.0])
        rd = np.array([300.0, 300.0])
        fp = np.array([1.0, 2.0])
        _, new_rd = _glicko_race_update(r, rd, fp)
        assert (new_rd < 300.0).all()
        assert (new_rd >= GLICKO_RD_MIN).all()

    def test_uncertain_rating_moves_more(self):
        # Same result, different RD: the uncertain horse's rating moves more.
        r = np.array([1500.0, 1500.0])
        fp = np.array([1.0, 2.0])
        gain_uncertain = _glicko_race_update(r.copy(), np.array([300.0, 200.0]), fp)[0][0] - 1500.0
        gain_certain = _glicko_race_update(r.copy(), np.array([60.0, 200.0]), fp)[0][0] - 1500.0
        assert gain_uncertain > gain_certain > 0

    def test_upset_moves_more_than_expected_win(self):
        # Low-rated horse beating a high-rated one gains more than the
        # reverse case where the favourite wins.
        r = np.array([1300.0, 1700.0])
        rd = np.array([150.0, 150.0])
        underdog_gain = _glicko_race_update(r.copy(), rd.copy(), np.array([1.0, 2.0]))[0][0] - 1300.0
        favourite_gain = _glicko_race_update(r.copy(), rd.copy(), np.array([2.0, 1.0]))[0][1] - 1700.0
        assert underdog_gain > favourite_gain > 0


class TestComputeGlickoFeatures:
    def _build_history(self):
        rows = []
        # Horse A wins three races against B and C
        for k, date in enumerate(["2025-01-01", "2025-01-15", "2025-02-01"]):
            for horse, fp in [("A", 1), ("B", 2), ("C", 3)]:
                rows.append({
                    "race_id": f"r{k}", "race_date": date,
                    "horse_name": horse, "finish_position": fp,
                })
        # Final race: all three meet again
        for horse, fp in [("A", 1), ("B", 2), ("C", 3)]:
            rows.append({
                "race_id": "r_final", "race_date": "2025-03-01",
                "horse_name": horse, "finish_position": fp,
            })
        return _race_df(rows)

    def test_prerace_values_no_lookahead(self):
        out = compute_glicko_features(self._build_history())
        first = out[out["race_id"] == "r0"]
        # Nobody has raced before race 0 — defaults recorded
        assert (first["horse_glicko"] == GLICKO_RATING_INIT).all()
        assert (first["horse_glicko_rd"] == GLICKO_RD_INIT).all()
        assert (first["has_horse_glicko"] == 0).all()

    def test_repeated_winner_rated_highest(self):
        out = compute_glicko_features(self._build_history())
        final = out[out["race_id"] == "r_final"].set_index("horse_name")
        assert final.loc["A", "horse_glicko"] > final.loc["B", "horse_glicko"]
        assert final.loc["B", "horse_glicko"] > final.loc["C", "horse_glicko"]
        assert final.loc["A", "horse_glicko_rank"] == 1
        # RD has come down from the initial maximum for all of them
        assert (final["horse_glicko_rd"] < GLICKO_RD_INIT).all()

    def test_layoff_inflates_rd(self):
        rows = []
        for horse, fp in [("A", 1), ("B", 2)]:
            rows.append({"race_id": "r0", "race_date": "2024-01-01",
                         "horse_name": horse, "finish_position": fp})
        # A returns after 18 months; B raced monthly meanwhile vs horse C
        for k in range(17):
            for horse, fp in [("B", 1), ("C", 2)]:
                rows.append({"race_id": f"rb{k}",
                             "race_date": (pd.Timestamp("2024-02-01") + pd.Timedelta(days=30 * k)).strftime("%Y-%m-%d"),
                             "horse_name": horse, "finish_position": fp})
        for horse, fp in [("A", 1), ("B", 2)]:
            rows.append({"race_id": "r_return", "race_date": "2025-07-01",
                         "horse_name": horse, "finish_position": fp})
        out = compute_glicko_features(_race_df(rows))
        ret = out[out["race_id"] == "r_return"].set_index("horse_name")
        # The horse off for 18 months is far more uncertain than the
        # one that kept racing.
        assert ret.loc["A", "horse_glicko_rd"] > ret.loc["B", "horse_glicko_rd"] + 50


class TestJockeyTrainerGlicko:
    def _build(self):
        rows = []
        for k in range(3):
            for horse, jockey, trainer, fp in [
                ("A", "J1", "T1", 1),
                ("B", "J2", "T2", 2),
                ("C", "J3", "T2", 3),
            ]:
                rows.append({
                    "race_id": f"r{k}", "race_date": f"2025-01-{1 + 7 * k:02d}",
                    "horse_name": horse, "jockey": jockey, "trainer": trainer,
                    "finish_position": fp,
                })
        return _race_df(rows)

    def test_flag_off_no_columns(self, monkeypatch):
        import config
        monkeypatch.setattr(config, "GLICKO_JOCKEY_TRAINER", False, raising=False)
        out = compute_glicko_features(self._build())
        assert "jockey_glicko" not in out.columns
        assert "trainer_glicko" not in out.columns

    def test_jockey_trainer_rated(self, monkeypatch):
        import config
        monkeypatch.setattr(config, "GLICKO_JOCKEY_TRAINER", True, raising=False)
        out = compute_glicko_features(self._build())
        last = out[out["race_id"] == "r2"].set_index("horse_name")
        # Winning jockey rated above the losers entering race 3
        assert last.loc["A", "jockey_glicko"] > last.loc["B", "jockey_glicko"]
        assert last.loc["A", "has_jockey_glicko"] == 1
        # Winning trainer above the trainer of the two beaten runners
        assert last.loc["A", "trainer_glicko"] > last.loc["B", "trainer_glicko"]
        # T2's two runners share one rating
        assert last.loc["B", "trainer_glicko"] == last.loc["C", "trainer_glicko"]

    def test_trainer_never_plays_itself(self, monkeypatch):
        import config
        monkeypatch.setattr(config, "GLICKO_JOCKEY_TRAINER", True, raising=False)
        rows = []
        # Two-horse races where ONE trainer saddles both runners: there is
        # no opponent, so the trainer's rating must never update.
        for k in range(2):
            for horse, fp in [("A", 1), ("B", 2)]:
                rows.append({
                    "race_id": f"r{k}", "race_date": f"2025-01-{1 + 7 * k:02d}",
                    "horse_name": horse, "jockey": f"J{fp}", "trainer": "T1",
                    "finish_position": fp,
                })
        out = compute_glicko_features(_race_df(rows))
        last = out[out["race_id"] == "r1"]
        assert (last["trainer_glicko"] == GLICKO_RATING_INIT).all()
        assert (last["has_trainer_glicko"] == 0).all()


class TestDimensionalGlicko:
    def test_surface_specialist_edge(self, monkeypatch):
        import config
        monkeypatch.setattr(config, "GLICKO_DIMENSIONAL", True, raising=False)
        rows = []
        dates = iter(pd.date_range("2025-01-01", periods=7, freq="7D"))
        # A beats B three times on turf, loses to B three times on the AW
        for surface, a_fp in [("turf", 1)] * 3 + [("aw", 2)] * 3:
            d = next(dates).strftime("%Y-%m-%d")
            for horse, fp in [("A", a_fp), ("B", 3 - a_fp)]:
                rows.append({
                    "race_id": f"r{d}", "race_date": d, "surface": surface,
                    "horse_name": horse, "finish_position": fp,
                })
        d = next(dates).strftime("%Y-%m-%d")
        for horse in ["A", "B"]:
            rows.append({
                "race_id": "r_final", "race_date": d, "surface": "turf",
                "horse_name": horse, "finish_position": 0,
            })
        out = compute_glicko_features(_race_df(rows))
        final = out[out["race_id"] == "r_final"].set_index("horse_name")
        # ≥3 turf runs: turf-specific rating reflects A's turf dominance
        assert final.loc["A", "horse_glicko_surface_edge"] > 0
        assert final.loc["B", "horse_glicko_surface_edge"] < 0

        # Cold start (< 3 runs in dimension) falls back to global: edge 0
        second = out[out["race_id"] == out["race_id"].iloc[2]]
        assert (second["horse_glicko_surface_edge"] == 0).all()
        assert (second["horse_glicko_surface"] == second["horse_glicko"]).all()
