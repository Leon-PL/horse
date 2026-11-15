"""Persisted, append-only feature store.

Five files committed together: featured rows, processed history, rating state,
carried pandas-stage state and metadata.

Who writes it:

  * ``src.store_build.build_store`` builds the real store in date-ordered
    chunks, never holding more than one chunk in memory. Training runs it when
    the store must be rebuilt; ``python -m scripts.build_feature_store_batched``
    runs it on request.
  * ``src.store_refresh.append_current_store`` appends new race dates through
    :meth:`FeatureStore._fast_featurize`, seeded from the carried state, and
    returns False when the store must be rebuilt instead.
  * :meth:`FeatureStore.build` featurises a whole history in memory. It is the
    reference the tests prove the other two against, and suits small stores
    only: on the full history it needs more than 16 GB.
  * :meth:`FeatureStore.featurize_live` reads the carried state for upcoming
    cards and never writes.

Design record from the incremental-FE work:

  * **Persistence + ergonomics** — build once, then append new
    dates. The featured rows and the processed history are kept on disk;
    historical featured rows are never recomputed (the no-leakage rule
    guarantees a settled row's features never change as later races
    arrive — see CLAUDE.md hard rule #1).
  * **Correctness** — appending features only new rows through the same
    engine, seeded from persisted counts and ratings. Schema, dtype and value
    parity with a full build is checked by `tests/test_feature_store.py`.
  * **First incremental component** — the Elo and Glicko rating sweeps now
    carry serialisable state (`compute_elo_features(..., return_state=True)`,
    `compute_glicko_features(..., return_state=True)`). The store persists
    and advances that state on every append, so the rating sweeps continue
    from where they left off rather than restarting from scratch. Parity of
    the seeded sweep vs a full batch sweep is proven in
    `tests/test_incremental_ratings.py`.

Counts, target encodings, rolling history and ratings carry state across
chunks. Live calculation owns private working state and completes only the
categorical columns declared by the store; it never publishes state advances.
"""

from __future__ import annotations

import json
import logging
import os
import pickle
import secrets
import shutil
import threading
from contextlib import contextmanager
from datetime import datetime
from functools import wraps

import pandas as pd

import config
from src.runner_identity import identity_columns, join_runner_columns
from src.feature_engineer import engineer_features
from src.ratings import compute_elo_features, compute_glicko_features
from src.feature_schema import CATEGORICAL_FIELDS, categorical_columns, complete_categorical_columns
from src.feature_semantics import distance_category

logger = logging.getLogger(__name__)

STORE_SCHEMA_VERSION = 2  # 2: carries counts_state.pkl for new-rows-only appends


def _make_stage_dir(root: str) -> str:
    """Create a unique ``.commit-*`` staging folder that inherits *root*'s permissions.

    Not ``tempfile.mkdtemp``: since Python 3.13 its 0o700 mode gives the new
    folder an owner-only ACL on Windows, and ``os.replace`` carries that ACL
    onto the published store files. A store built by one account (the Codex
    sandbox, 2026-11-11 and 2026-11-12) then could not be read by any other.
    """
    for _ in range(100):
        stage = os.path.join(root, f".commit-{secrets.token_hex(4)}")
        try:
            os.mkdir(stage)  # default mode: no explicit ACL, inherits from root
        except FileExistsError:
            continue
        return stage
    raise FileExistsError(f"Could not create a unique staging folder in {root}")


def _locked(method):
    """Keep a store operation on one coherent generation, across processes."""
    @wraps(method)
    def wrapped(self, *args, **kwargs):
        with self._access():
            return method(self, *args, **kwargs)
    return wrapped


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
    df = df.copy(deep=False)
    for c in df.select_dtypes(include="object").columns:
        s = df[c]
        df[c] = s.where(s.isna(), s.astype(str))
    return df


class FeatureStore:
    """An append-only featured dataset backed by parquet + a state file."""

    _FILES = ("featured.parquet", "processed.parquet", "ratings_state.pkl",
              "counts_state.pkl", "meta.json")

    def __init__(self, root: str | None = None):
        self.root = root or os.path.join(config.PROCESSED_DATA_DIR, "feature_store")
        os.makedirs(self.root, exist_ok=True)
        self.featured_path = os.path.join(self.root, "featured.parquet")
        self.processed_path = os.path.join(self.root, "processed.parquet")
        self.state_path = os.path.join(self.root, "ratings_state.pkl")
        self.counts_state_path = os.path.join(self.root, "counts_state.pkl")
        self.meta_path = os.path.join(self.root, "meta.json")
        self._journal_path = os.path.join(self.root, ".pending_commit.json")
        self._mutex = threading.RLock()
        self._lock_depth = 0

    @contextmanager
    def _access(self):
        """Nonblocking OS lock, released automatically even after process death.

        Nested calls on this instance reuse the lock. Other readers/writers
        must retry, rather than read a mixture during a multi-file commit.
        Direct file access bypasses this protection.
        """
        with self._mutex:
            if self._lock_depth:
                yield
                return
            with open(os.path.join(self.root, ".store.lock"), "a+b") as fh:
                if os.name == "nt":
                    import msvcrt
                    if os.fstat(fh.fileno()).st_size == 0:
                        fh.write(b"\0")
                        fh.flush()
                    fh.seek(0)
                    acquire = lambda: msvcrt.locking(fh.fileno(), msvcrt.LK_NBLCK, 1)
                    release = lambda: msvcrt.locking(fh.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    import fcntl
                    acquire = lambda: fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    release = lambda: fcntl.flock(fh, fcntl.LOCK_UN)
                try:
                    acquire()
                except OSError as exc:
                    raise RuntimeError(
                        "FeatureStore is in use by another operation; retry shortly."
                    ) from exc
                self._lock_depth = 1
                try:
                    self._recover_commit()
                    yield
                finally:
                    self._lock_depth = 0
                    release()

    def _recover_commit(self):
        """Roll a journaled, fully staged update forward before any read.

        Replacing a staged file atomically removes its source. A missing
        source therefore means that replacement already completed. The
        journal stays until ALL five replacements finish, including metadata.
        """
        if not os.path.exists(self._journal_path):
            return
        with open(self._journal_path, encoding="utf-8") as fh:
            name = json.load(fh)["stage"]
        if not isinstance(name, str) or os.path.basename(name) != name or not name.startswith(".commit-"):
            raise RuntimeError("Invalid feature-store commit journal; rebuild required.")
        stage = os.path.join(self.root, name)
        if not os.path.isdir(stage):
            raise RuntimeError("Feature-store commit files are missing; rebuild required.")
        for filename in self._FILES:
            source = os.path.join(stage, filename)
            target = os.path.join(self.root, filename)
            if os.path.exists(source):
                os.replace(source, target)
            elif not os.path.exists(target):
                raise RuntimeError(f"Incomplete feature-store commit: {filename} is missing.")
        os.remove(self._journal_path)
        shutil.rmtree(stage, ignore_errors=True)

    @contextmanager
    def _staged_update(self):
        """Yield a staging store; publish only after every output is complete.

        Used by build, the training refresh and the batched builder. A failure
        before journaling leaves the old store untouched. After journaling,
        the next operation finishes the commit before exposing any data.
        Requires room for both old and new files; no old-data copies are made.
        """
        with self._access():
            stage = _make_stage_dir(self.root)
            try:
                yield FeatureStore(stage)
                # Flush all completed outputs before publishing the journal.
                for filename in self._FILES:
                    # Windows _commit requires a writable file descriptor.
                    with open(os.path.join(stage, filename), "r+b") as fh:
                        os.fsync(fh.fileno())
                journal_tmp = self._journal_path + ".tmp"
                with open(journal_tmp, "w", encoding="utf-8") as fh:
                    json.dump({"stage": os.path.basename(stage)}, fh)
                    fh.flush()
                    os.fsync(fh.fileno())
                os.replace(journal_tmp, self._journal_path)
                self._recover_commit()
            finally:
                # Never remove files needed to recover a partially applied commit.
                if not os.path.exists(self._journal_path):
                    shutil.rmtree(stage, ignore_errors=True)

    # ── status ────────────────────────────────────────────────────────
    @_locked
    def exists(self) -> bool:
        return os.path.exists(self.featured_path) and os.path.exists(self.meta_path)

    @_locked
    def meta(self) -> dict:
        if not os.path.exists(self.meta_path):
            return {}
        with open(self.meta_path, encoding="utf-8") as fh:
            return json.load(fh)

    def last_date(self) -> pd.Timestamp | None:
        m = self.meta()
        return pd.Timestamp(m["last_date"]) if m.get("last_date") else None

    # ── load ──────────────────────────────────────────────────────────
    @_locked
    def load_featured(self, columns: list[str] | None = None) -> pd.DataFrame | None:
        if not os.path.exists(self.featured_path):
            return None
        return pd.read_parquet(self.featured_path, columns=columns, engine="pyarrow")

    @_locked
    def featured_schema_names(self) -> list[str]:
        """Column names of the stored featured parquet (no data read)."""
        import pyarrow.parquet as pq
        return pq.read_schema(self.featured_path).names if os.path.exists(self.featured_path) else []

    @_locked
    def categorical_schema(self) -> list[str]:
        """The explicit dummy domain; older reference stores derive it from inputs."""
        declared = self.meta().get("categorical_columns")
        if declared is not None:
            return declared
        import pyarrow.parquet as pq
        names = pq.read_schema(self.processed_path).names
        columns = [c for c in (*CATEGORICAL_FIELDS, "race_date") if c in names]
        return categorical_columns(pd.read_parquet(self.processed_path, columns=columns))

    @_locked
    def load_processed(self) -> pd.DataFrame | None:
        if not os.path.exists(self.processed_path):
            return None
        return pd.read_parquet(self.processed_path, engine="pyarrow")

    @_locked
    def load_state(self) -> dict:
        if not os.path.exists(self.state_path):
            return {"elo": None, "glicko": None}
        with open(self.state_path, "rb") as fh:
            return pickle.load(fh)

    @_locked
    def load_counts_state(self) -> dict | None:
        """Carried pandas-stage state (counts/rolling/time-window/lag/extrema/
        expanding). ``None`` if the store predates the incremental flip, which
        then has to be rebuilt."""
        if not os.path.exists(self.counts_state_path):
            return None
        with open(self.counts_state_path, "rb") as fh:
            return pickle.load(fh)

    # ── persistence helpers ───────────────────────────────────────────
    @staticmethod
    def _with_dist_category(df: pd.DataFrame) -> pd.DataFrame:
        """Add ``dist_category`` (a pure bin of distance_furlongs).

        The rating sweeps and the distance-change stage share this definition,
        so standalone seeded ratings produce the same dimensional Elo.
        """
        if "distance_furlongs" in df.columns and "dist_category" not in df.columns:
            df = df.copy()
            df["dist_category"] = distance_category(df["distance_furlongs"])
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
        keyed = rated[identity_columns(rated) + rating_cols].copy()
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

    def _write_meta(self, processed: pd.DataFrame, featured: pd.DataFrame, *, op: str,
                    previous: dict | None = None):
        m = dict(previous) if previous is not None else self.meta()
        m.update({
            "schema_version": STORE_SCHEMA_VERSION,
            "last_date": str(_event_date(processed).max().date()),
            "featured_rows": int(len(featured)),
            "processed_rows": int(len(processed)),
            "categorical_columns": categorical_columns(processed),
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
    @_locked
    def build(self, processed: pd.DataFrame) -> pd.DataFrame:
        """Featurise *processed* in memory and commit it as the whole store.

        The in-memory reference build, for tests and small stores. Build the
        real store with ``python -m scripts.build_feature_store_batched``:
        holding the full history here needs more than 16 GB.

        Runs the FE pass with an (empty) ``counts_state`` collector so every
        stateful pandas stage writes its end-state — identical features to a
        plain build (empty seed == no seed), but the carried state is persisted
        so later appends can run on new rows only. Rating end-state is persisted
        separately (the existing Elo/Glicko mechanism)."""
        processed = processed.copy()
        counts_state: dict = {}
        featured = engineer_features(processed, save=False, counts_state=counts_state)

        state = self._compute_ratings_state(processed)
        with self._staged_update() as pending:
            _stringify_objects(featured).to_parquet(pending.featured_path, index=False, engine="pyarrow")
            _stringify_objects(processed).to_parquet(pending.processed_path, index=False, engine="pyarrow")
            pending._save_state(state)
            pending._save_counts_state(counts_state)
            pending._write_meta(processed, featured, op="built", previous=self.meta())
        logger.info("FeatureStore built: %d featured rows", len(featured))
        return featured

    def _fast_featurize(self, new: pd.DataFrame) -> tuple[pd.DataFrame, dict, dict]:
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
        new_rated = join_runner_columns(new, keyed_ratings, _rating_cols)

        counts_state = self.load_counts_state() or {}
        featured_new = engineer_features(
            new_rated, save=False, skip_ratings=True, counts_state=counts_state
        )
        # Stages advance counts in memory only. The caller commits them with
        # the matching rows and ratings, never before parquet writes succeed.
        return featured_new, advanced, counts_state

    # Entity → its expanding-frequency column (data_processor.encode_categoricals:
    # ``cumcount()+1`` over the full frame; ``track`` is collapsed to a per-race min).
    _FREQ_COLS = {"horse_name": "horse_name_freq", "jockey": "jockey_freq",
                  "trainer": "trainer_freq", "track": "track_freq"}

    def _continue_freq_counts(self, card: pd.DataFrame) -> pd.DataFrame:
        """Continue the cumulative ``*_freq`` counts from full history.

        ``encode_categoricals`` builds ``*_freq`` as a per-frame ``cumcount``; on a
        card processed alone each entity restarts at 1, leaving e.g.
        ``trainer_freq`` (and its race-relative derivatives) on the wrong scale vs
        training. Offset the card's per-entity freq by that entity's historical
        appearance count (from the store's processed history) so the count resumes
        where training left off. Read-only."""
        present = {e: f for e, f in self._FREQ_COLS.items()
                   if e in card.columns and f in card.columns}
        if not present:
            return card
        card = card.copy()
        hist = pd.read_parquet(self.processed_path, columns=list(present))
        for ent, fcol in present.items():
            prior = hist[ent].value_counts()  # historical appearances per entity
            card[fcol] = pd.to_numeric(card[fcol], errors="coerce").fillna(0) \
                + card[ent].map(prior).fillna(0)
            card[fcol] = card[fcol].astype(int)
        if "track" in present and "race_id" in card.columns:  # race-level: per-race min
            card["track_freq"] = card.groupby("race_id")["track_freq"].transform("min")
        return card

    @_locked
    def featurize_live(self, card: pd.DataFrame) -> pd.DataFrame:
        """Feature-engineer an UNSETTLED future *card* against the store's
        persisted rating + counts state — **read-only**.

        Same bounded fast path as the training refresh
        (``src.store_refresh.append_current_store``): ratings seeded from
        persisted state, every stateful stage resumed from carried
        ``counts_state``, ``skip_ratings=True``, so live features match
        training exactly. Unlike the refresh it operates on freshly
        deserialised state and never persists,
        so result-less rows can't corrupt the store. This is the correct live-FE
        path: it avoids both the all-default ratings of a from-scratch sweep over
        stale/empty history AND the OOM of feature-engineering full history in one
        shot."""
        if not self.exists():
            raise RuntimeError(
                "FeatureStore is empty; build it with "
                "`python -m scripts.build_feature_store_batched` before live featurisation.")
        counts_state = self.load_counts_state()
        if counts_state is None:
            raise RuntimeError(
                "FeatureStore lacks carried state (pre-flip store); rebuild it with "
                "`python -m scripts.build_feature_store_batched` before live featurisation.")
        from src.result_status import pending_card
        card = self._continue_freq_counts(pending_card(card))
        seed = self.load_state()
        keyed_ratings, _rating_cols, _advanced = self._seeded_rating_frame(card, seed)
        card_rated = join_runner_columns(card, keyed_ratings, _rating_cols)
        # pickle.load creates a private object graph for this request. Discard
        # its in-memory advances; neither a second load nor a deep copy is needed.
        featured = engineer_features(
            card_rated, save=False, skip_ratings=True, counts_state=counts_state)

        return complete_categorical_columns(
            featured, self.featured_schema_names(), self.categorical_schema())
