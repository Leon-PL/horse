"""Persisted, append-only feature store.

Builds the featured dataset once and appends new race dates instead of
re-engineering the whole history every time. This is phase 1 of the
incremental-FE work:

  * **Persistence + ergonomics** — `build()` once, then `append()` new
    dates. The featured rows and the processed history are kept on disk;
    historical featured rows are never recomputed (the no-leakage rule
    guarantees a settled row's features never change as later races
    arrive — see CLAUDE.md hard rule #1).
  * **Correctness** — `append()` re-features new rows through the
    parity-tested `feature_engineer_with_history_core` path, so the
    appended rows are identical to a full batch rebuild. This is enforced
    by `tests/test_feature_store.py`.
  * **First incremental component** — the Elo and Glicko rating sweeps now
    carry serialisable state (`compute_elo_features(..., return_state=True)`,
    `compute_glicko_features(..., return_state=True)`). The store persists
    and advances that state on every append, so the rating sweeps continue
    from where they left off rather than restarting from scratch. Parity of
    the seeded sweep vs a full batch sweep is proven in
    `tests/test_incremental_ratings.py`.

What is NOT yet incremental: the remaining stateful pandas stages (career
counts, target encodings, rolling form, …) are still recomputed over the
full history inside the history-core pass. Converting those stage-by-stage
(each gated by a parity test) is what unlocks the full speed-up; the rating
state above is the template for that work.
"""

from __future__ import annotations

import json
import logging
import os
import pickle
from datetime import datetime

import pandas as pd

import config
from src.feature_engineer import engineer_features
from src.live_prediction import feature_engineer_with_history_core
from src.ratings import compute_elo_features, compute_glicko_features

logger = logging.getLogger(__name__)

STORE_SCHEMA_VERSION = 2  # 2: carries counts_state.pkl for new-rows-only appends


def _event_date(df: pd.DataFrame) -> pd.Series:
    return pd.to_datetime(df["race_date"], errors="coerce")


def _stringify_objects(df: pd.DataFrame) -> pd.DataFrame:
    """Make object columns parquet-safe.

    Concatenating a reloaded frame with a freshly-computed one can leave an
    object column holding a mix of ``str`` and ``int`` (e.g. ``form_str``),
    which pyarrow refuses to serialise. Coerce object columns to a uniform
    string representation, preserving NaN. Numeric (model) columns are
    untouched, so this affects on-disk storage only.
    """
    df = df.copy()
    for c in df.select_dtypes(include="object").columns:
        s = df[c]
        df[c] = s.where(s.isna(), s.astype(str))
    return df


class FeatureStore:
    """An append-only featured dataset backed by parquet + a state file."""

    def __init__(self, root: str | None = None):
        self.root = root or os.path.join(config.PROCESSED_DATA_DIR, "feature_store")
        os.makedirs(self.root, exist_ok=True)
        self.featured_path = os.path.join(self.root, "featured.parquet")
        self.processed_path = os.path.join(self.root, "processed.parquet")
        self.state_path = os.path.join(self.root, "ratings_state.pkl")
        self.counts_state_path = os.path.join(self.root, "counts_state.pkl")
        self.meta_path = os.path.join(self.root, "meta.json")

    # ── status ────────────────────────────────────────────────────────
    def exists(self) -> bool:
        return os.path.exists(self.featured_path) and os.path.exists(self.meta_path)

    def meta(self) -> dict:
        if not os.path.exists(self.meta_path):
            return {}
        with open(self.meta_path, encoding="utf-8") as fh:
            return json.load(fh)

    def last_date(self) -> pd.Timestamp | None:
        m = self.meta()
        return pd.Timestamp(m["last_date"]) if m.get("last_date") else None

    # ── load ──────────────────────────────────────────────────────────
    def load_featured(self) -> pd.DataFrame | None:
        if not os.path.exists(self.featured_path):
            return None
        return pd.read_parquet(self.featured_path, engine="pyarrow")

    def load_processed(self) -> pd.DataFrame | None:
        if not os.path.exists(self.processed_path):
            return None
        return pd.read_parquet(self.processed_path, engine="pyarrow")

    def load_state(self) -> dict:
        if not os.path.exists(self.state_path):
            return {"elo": None, "glicko": None}
        with open(self.state_path, "rb") as fh:
            return pickle.load(fh)

    def load_counts_state(self) -> dict | None:
        """Carried pandas-stage state (counts/rolling/time-window/lag/extrema/
        expanding). ``None`` if the store predates the incremental flip — the
        caller must then fall back to the full-recompute append."""
        if not os.path.exists(self.counts_state_path):
            return None
        with open(self.counts_state_path, "rb") as fh:
            return pickle.load(fh)

    # ── persistence helpers ───────────────────────────────────────────
    @staticmethod
    def _with_dist_category(df: pd.DataFrame) -> pd.DataFrame:
        """Add ``dist_category`` (a pure bin of distance_furlongs).

        The rating sweeps read it, but in the full pipeline it is added
        mid-way by ``add_course_distance_features``. Mirror that one-liner
        so a standalone rating sweep produces identical dimensional Elo.
        """
        if "distance_furlongs" in df.columns and "dist_category" not in df.columns:
            df = df.copy()
            df["dist_category"] = (
                pd.cut(
                    df["distance_furlongs"], bins=[0, 6.5, 8.5, 11.5, 50],
                    labels=[0, 1, 2, 3], include_lowest=True,
                ).astype(float).fillna(1)
            )
        return df

    def _seeded_rating_frame(
        self, new_proc: pd.DataFrame, seed: dict | None
    ) -> tuple[pd.DataFrame, list[str], dict]:
        """Compute rating columns for *new_proc* from seeded state.

        Returns ``(rating_df, rating_cols, new_state)`` where ``rating_df``
        holds the merge keys plus every rating column, in the same order as
        a full sweep would produce for these rows.
        """
        seed = seed or {}
        work = self._with_dist_category(new_proc)
        base_cols = set(work.columns)
        rated, elo_state = compute_elo_features(
            work, elo_state=seed.get("elo"), return_state=True
        )
        glicko_state = None
        if getattr(config, "GLICKO_ENABLED", True):
            rated, glicko_state = compute_glicko_features(
                rated, glicko_state=seed.get("glicko"), return_state=True
            )
        rating_cols = [
            c for c in rated.columns if c not in base_cols and c != "_event_dt"
        ]
        keyed = rated[["race_id", "horse_name"] + rating_cols].copy()
        return keyed, rating_cols, {"elo": elo_state, "glicko": glicko_state}

    def _compute_ratings_state(
        self, processed: pd.DataFrame, seed: dict | None = None
    ) -> dict:
        """Advance (or build) the rating end-state over *processed*.

        When *seed* is given, the sweeps continue from the persisted state
        and process only the rows in *processed* (the new dates). Without a
        seed they build from scratch over the full history.
        """
        seed = seed or {}
        # dist_category is added mid-pipeline but the dimensional Elo/Glicko
        # read it; add it here so the persisted state accumulates the
        # dimensional counts exactly as the full pipeline does.
        processed = self._with_dist_category(processed)
        _, elo_state = compute_elo_features(
            processed.copy(), elo_state=seed.get("elo"), return_state=True
        )
        glicko_state = None
        if getattr(config, "GLICKO_ENABLED", True):
            _, glicko_state = compute_glicko_features(
                processed.copy(), glicko_state=seed.get("glicko"), return_state=True
            )
        return {"elo": elo_state, "glicko": glicko_state}

    def _write_meta(self, processed: pd.DataFrame, featured: pd.DataFrame, *, op: str):
        m = self.meta()
        m.update({
            "schema_version": STORE_SCHEMA_VERSION,
            "last_date": str(_event_date(processed).max().date()),
            "featured_rows": int(len(featured)),
            "processed_rows": int(len(self.load_processed())),
            f"{op}_at": datetime.now().isoformat(timespec="seconds"),
        })
        with open(self.meta_path, "w", encoding="utf-8") as fh:
            json.dump(m, fh, indent=2, sort_keys=True, default=str)

    def _save_state(self, state: dict):
        with open(self.state_path, "wb") as fh:
            pickle.dump(state, fh, protocol=pickle.HIGHEST_PROTOCOL)

    def _save_counts_state(self, state: dict):
        with open(self.counts_state_path, "wb") as fh:
            pickle.dump(state, fh, protocol=pickle.HIGHEST_PROTOCOL)

    # ── build / append ────────────────────────────────────────────────
    def build(self, processed: pd.DataFrame) -> pd.DataFrame:
        """Build the store from scratch over the full processed history.

        Runs the FE pass with an (empty) ``counts_state`` collector so every
        stateful pandas stage writes its end-state — identical features to a
        plain build (empty seed == no seed), but the carried state is persisted
        so later appends can run on new rows only. Rating end-state is persisted
        separately (the existing Elo/Glicko mechanism)."""
        processed = processed.copy()
        counts_state: dict = {}
        featured = engineer_features(processed, save=False, counts_state=counts_state)

        _stringify_objects(featured).to_parquet(self.featured_path, index=False, engine="pyarrow")
        _stringify_objects(processed).to_parquet(self.processed_path, index=False, engine="pyarrow")
        self._save_state(self._compute_ratings_state(processed))
        self._save_counts_state(counts_state)
        self._write_meta(processed, featured, op="built")
        logger.info("FeatureStore built: %d featured rows", len(featured))
        return featured

    def _fast_featurize(self, new: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
        """Incremental featurise on NEW ROWS ONLY.

        Seed the rating sweeps from persisted state and attach those columns to
        the new rows, then run the full pipeline with ``skip_ratings=True`` under
        the carried ``counts_state`` so every stateful pandas stage (counts,
        rolling, time-window, lag, extrema, expanding) resumes from its persisted
        end-state instead of recomputing history. No history rows enter the FE
        pass — this is the actual incremental speed-up. Parity with a full
        rebuild is enforced by tests/test_feature_store.py (fast path)."""
        seed = self.load_state()
        keyed_ratings, _rating_cols, advanced = self._seeded_rating_frame(new, seed)
        new_rated = new.merge(keyed_ratings, on=["race_id", "horse_name"], how="left")

        counts_state = self.load_counts_state() or {}
        featured_new = engineer_features(
            new_rated, save=False, skip_ratings=True, counts_state=counts_state
        )
        self._save_counts_state(counts_state)  # advanced in place by the stages
        return featured_new, advanced

    def append(self, processed_new: pd.DataFrame, *, fast: bool = False) -> pd.DataFrame:
        """Feature-engineer and append new race dates.

        Only rows strictly after the store's ``last_date`` are taken (the
        store holds settled history; re-appending the same dates is a
        no-op). Returns the newly appended featured rows.

        With ``fast=True`` (and a store carrying ``counts_state``) the whole FE
        pass runs on the NEW rows only — ratings seeded from persisted state and
        every pandas stage resumed from carried state — so history is never
        recomputed. Without carried state (a pre-flip store) it falls back to the
        full-recompute path. Both paths are parity-tested against a full rebuild.
        """
        if not self.exists():
            raise RuntimeError("FeatureStore is empty; call build() first.")

        processed_hist = self.load_processed()
        last = self.last_date()
        new = processed_new.copy()
        if last is not None:
            new = new[_event_date(new) > last]
        if new.empty:
            logger.info("FeatureStore.append: no rows after %s — nothing to do", last)
            return new

        if fast and self.load_counts_state() is not None:
            featured_new, advanced = self._fast_featurize(new)
        else:
            # Correct re-featuring against full history (parity-tested path):
            # also the fallback for pre-flip stores that lack carried state.
            featured_new = feature_engineer_with_history_core(processed_hist, new)
            advanced = self._compute_ratings_state(new, seed=self.load_state())

        # Persist: append featured + processed, advance + save rating state.
        featured_all = pd.concat(
            [self.load_featured(), featured_new], ignore_index=True, sort=False
        )
        processed_all = pd.concat(
            [processed_hist, new], ignore_index=True, sort=False
        )
        _stringify_objects(featured_all).to_parquet(self.featured_path, index=False, engine="pyarrow")
        _stringify_objects(processed_all).to_parquet(self.processed_path, index=False, engine="pyarrow")

        self._save_state(advanced)
        self._write_meta(processed_all, featured_all, op="appended")
        logger.info(
            "FeatureStore.append: +%d featured rows (now %d)",
            len(featured_new), len(featured_all),
        )
        return featured_new
