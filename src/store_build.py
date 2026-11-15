"""Memory-bounded history build, shared by normal training and explicit rebuilds.

A monolithic in-memory build (``FeatureStore.build``) over the whole DB (~707k
rows → a ~6 GB featured frame, plus several intermediate copies during FE)
peaks past 16 GB of RAM and thrashes the machine. ``build_store`` produces a
byte-equivalent store while never holding more than one chunk in memory:

  1. Load the full processed history once (cheap; processed is narrow).
  2. Walk it in date-ordered chunks. Each chunk is featured on its OWN rows
     only, with ratings + every pandas stage seeded from the *previous*
     chunk's carried state — exactly the incremental append path, which is
     parity-proven against a monolithic build (tests/test_feature_store.py).
     So N chunks == one full build.
  3. Stream each featured chunk straight into the final parquet file, so every
     row is encoded once. Only a chunk whose column types differ starts another
     file; those are reconciled with pyarrow at the end (schemas unified, one
     bounded block at a time), so the whole frame never reaches pandas.
  4. Persist ratings_state + counts_state and stamp meta with the same
     fe_signature / data_signature the training refresh
     (src.store_refresh.append_current_store) expects, so the very next train
     APPENDS instead of rebuilding.
  5. Commit all five outputs together under the feature-store lock. A failed
     build leaves the previous store intact; an interrupted commit is resumed
     before the next store operation.

Callers: ``src.store_refresh.get_training_dataset`` when the store must be
rebuilt, and the command ``python -m scripts.build_feature_store_batched``.
Until 2026-11-07 this code lived in that command, so src/ imported scripts/.
"""

from __future__ import annotations

import gc
import glob
import json
import logging
import multiprocessing
import os
import pickle
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from contextlib import closing, contextmanager
from contextvars import copy_context
from datetime import datetime

import numpy as np
import pandas as pd

import config
from src.data_processor import process_data
from src.feature_engineer import engineer_features
from src.feature_store import STORE_SCHEMA_VERSION, FeatureStore, _stringify_objects
from src.feature_schema import categorical_columns
from src.featured_dataset import data_signature, fe_signature
from src.feature_inputs import atr_cache_signature, rtv_cache_signature
from src.parquet_io import RollingParquetWriter, combine_parquet_parts
from src.source_dataset import current_manifest, dataset_build, install_manifest

LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"
logger = logging.getLogger(__name__)


def _rss_gb() -> float:
    try:
        import psutil  # optional

        return psutil.Process().memory_info().rss / (1024 ** 3)
    except Exception:
        return float("nan")


# A long history is built by three processes in a line, each carrying its own
# state: one rates ahead, this one runs the first half of FE, and one finishes
# FE and streams the store. A shorter history stays in this process.
HELPERS_MIN_ROWS = 100_000
RATINGS_AHEAD_CHUNKS = 2

_worker_state: dict | None = None
_worker_seconds = 0.0
_worker_tail: _StoreTail | None = None


def _settings() -> dict:
    """Everything a fresh import of config decides, including where files live."""
    return {k: v for k, v in vars(config).items() if k.isupper()}


def _settings_match(settings: dict) -> bool:
    return settings == _settings()


def _start_helper(manifest, log_level: int) -> None:
    """Initialise a spawned helper: the parent's dataset versions and log level.

    A spawned process starts with no logging set up. The builder used to
    configure logging on import, which also covered the helpers.
    """
    logging.basicConfig(level=log_level, format=LOG_FORMAT)
    install_manifest(manifest)


def _rate_ahead(chunk: pd.DataFrame) -> pd.DataFrame:
    """Worker side: rate the next chunk from the state this process carries."""
    global _worker_state, _worker_seconds
    started = time.perf_counter()
    # Rating touches no store files, so the worker needs no store of its own.
    keyed, _cols, _worker_state = FeatureStore.__new__(FeatureStore)._seeded_rating_frame(chunk, _worker_state)
    _worker_seconds += time.perf_counter() - started
    return keyed


def _worker_result() -> tuple[dict | None, float]:
    return _worker_state, _worker_seconds


class _StoreTail:
    """The end of the build: streams finished chunks into the store's file."""

    def __init__(self, directory: str, counts_state: dict):
        self.counts_state = counts_state
        self._stream = RollingParquetWriter(directory)
        self._encoder = ThreadPoolExecutor(max_workers=1, thread_name_prefix="store-part")
        self._encoding = None

    def write(self, featured: pd.DataFrame) -> None:
        ready = _stringify_objects(featured)
        if self._encoding is not None:
            self._encoding.result()
        # Encoding releases the GIL, so the next chunk starts straight away.
        self._encoding = self._encoder.submit(self._stream.write, ready)

    def finish_chunk(self, partial: pd.DataFrame) -> None:
        self.write(engineer_features(partial, save=False, skip_ratings=True,
                                     counts_state=self.counts_state, half="second"))

    def close(self) -> tuple[list[str], int, dict]:
        try:
            if self._encoding is not None:
                self._encoding.result()
        finally:
            self._encoder.shutdown()
            self._stream.close()
        return self._stream.paths, self._stream.rows, self.counts_state


def _open_tail(directory: str) -> None:
    global _worker_tail
    _worker_tail = _StoreTail(directory, {})


def _finish_chunk(partial: pd.DataFrame) -> None:
    _worker_tail.finish_chunk(partial)


def _close_tail(state_path: str) -> tuple[list[str], int]:
    """Finish the store's file. The state goes through a file because it is too
    large to pass back quickly: it took about four minutes through the pipe."""
    paths, rows, state = _worker_tail.close()
    with open(state_path, "wb") as fh:
        pickle.dump(state, fh, protocol=pickle.HIGHEST_PROTOCOL)
    return paths, rows


def _load_second_half_state(path: str) -> dict:
    """Read the state with collection off, which would otherwise rescan every
    container already loaded, again and again, as millions more arrive."""
    was_collecting = gc.isenabled()
    gc.disable()
    try:
        with open(path, "rb") as fh:
            return pickle.load(fh)
    finally:
        if was_collecting:
            gc.enable()
        os.remove(path)


def _add_second_half_state(state: dict, second: dict) -> None:
    """Merge the second half's carried state into the first half's.

    Each half advanced its own copy, so a state key the two share is only sound
    when it holds different entities. ``_tg_wins`` does: trainer||going in the
    first half, horse-track-going in the second.
    """
    for key, entities in second.items():
        if key in state and state[key].keys() & entities.keys():
            raise RuntimeError(f"FE stages either side of the handover carry the same state: {key}")
        state[key] = {**state.get(key, {}), **entities}


@contextmanager
def _helper_processes(wanted: bool):
    """A rating and a finishing process, or None twice to build in this one.

    A spawned process imports config afresh, so a run that changed settings at
    runtime (a research script, say) stays here instead of quietly diverging.
    """
    spawn = multiprocessing.get_context("spawn")
    pools = [ProcessPoolExecutor(max_workers=1, mp_context=spawn, initializer=_start_helper,
                                 initargs=(current_manifest(), logging.getLogger().getEffectiveLevel()))
             for _ in range(2)] if wanted else []
    usable = False
    try:
        checks = [pool.submit(_settings_match, _settings()) for pool in pools]
        usable = bool(pools) and all([check.result() for check in checks])
        if pools and not usable:
            logger.warning("Runtime settings differ from config.py, so the build stays in this process")
    except (BrokenProcessPool, OSError) as exc:
        logger.warning("No helper processes (%s), so the build stays in this process", exc)
    try:
        yield pools if usable else (None, None)
    finally:
        for pool in pools:
            pool.shutdown(cancel_futures=True)


def _save_inputs(store: FeatureStore, processed: pd.DataFrame, cutoff) -> dict:
    """Write the processed rows and fingerprint the inputs the features read.

    None of this needs a feature, so it runs beside the first chunks. Taking the
    cache signatures before FE reads the caches also means that a cache changed
    mid-build shows up as a mismatch, where a signature taken at the end hid it.
    """
    _stringify_objects(processed).to_parquet(store.processed_path, index=False, engine="pyarrow")
    from src.foreign_form import signature as foreign_form_signature
    return {
        "data_signature": data_signature(processed, cutoff=cutoff),
        # The sectional cache is read inside FE, outside both signatures.
        "atr_signature": {"cutoff": str(cutoff.date()), "sig": atr_cache_signature(cutoff)},
        "rtv_signature": {"cutoff": str(cutoff.date()), "sig": rtv_cache_signature(cutoff)},
        "foreign_form_signature": foreign_form_signature(cutoff),
    }


def _finish_featured(paths: list[str], rows: int, destination: str) -> int:
    """Put the streamed rows at ``destination`` and return how many there are."""
    if len(paths) == 1:
        os.replace(paths[0], destination)
        return rows
    logger.info("Reconciling %d schemas → %s", len(paths), destination)
    return combine_parquet_parts(paths, destination)


def _date_chunks(dates: np.ndarray, chunk_rows: int, counts: dict) -> list[tuple]:
    """Group sorted unique dates into spans of ~chunk_rows rows each.

    Splits only on date boundaries so a race is never cut across chunks.
    Returns a list of (lo_date, hi_date) inclusive ranges.
    """
    spans, lo, acc = [], None, 0
    for d in dates:
        if lo is None:
            lo = d
        acc += counts[d]
        if acc >= chunk_rows:
            spans.append((lo, d))
            lo, acc = None, 0
    if lo is not None:
        spans.append((lo, dates[-1]))
    return spans


@dataset_build
def build_store(store: FeatureStore, chunk_rows: int, *, ratings_cache=None, processed=None, use_helpers=True) -> dict:
    """Build into the supplied staging store without touching the live files."""
    t0 = time.perf_counter()
    # Fix the signature BEFORE any work. Reading it at the end described
    # whatever the source files said after an hour of building, not the code
    # that computed the values: the store in use on 2026-10-12 was stamped
    # mid-edit that way and matched neither generation (ledger 12).
    fe_sig_at_start = fe_signature()
    parts_dir = os.path.join(store.root, "_build_parts")
    os.makedirs(parts_dir, exist_ok=True)
    for old in glob.glob(os.path.join(parts_dir, "*.parquet")):
        os.remove(old)

    logger.info("Loading full processed history…")
    processed = process_data() if processed is None else processed.copy()
    processed["race_date"] = pd.to_datetime(processed["race_date"], errors="coerce")
    processed = processed.sort_values("race_date", kind="stable").reset_index(drop=True)
    reused_ratings = ratings_cache.prepare(processed) if ratings_cache is not None else None
    if reused_ratings:
        logger.info("Reusing %d rating columns after validating all %d processed rows",
                    len(reused_ratings["rating_columns"]), len(processed))
    rd = processed["race_date"]
    vc = rd.value_counts()
    dates = np.sort(rd.dropna().unique())
    counts = {d: int(vc[d]) for d in dates}
    spans = _date_chunks(dates, chunk_rows, counts)
    logger.info("Full history: %d rows, %d dates → %d chunks (RSS %.1f GB)",
                len(processed), len(dates), len(spans), _rss_gb())

    ratings_state: dict | None = None
    counts_state: dict = {}
    total_rows = 0
    rating_jobs: list = []
    rating_wait = 0.0
    finishing = None
    last_date = rd.max()
    with _helper_processes(use_helpers and len(processed) >= HELPERS_MIN_ROWS) as (rater, finisher), \
            closing(_StoreTail(parts_dir, counts_state)) as tail, \
            ThreadPoolExecutor(max_workers=1, thread_name_prefix="store-inputs") as input_writer:
        saving_inputs = input_writer.submit(copy_context().run, _save_inputs, store, processed, last_date)
        if finisher is not None:
            finisher.submit(_open_tail, parts_dir).result()
        for i, (lo, hi) in enumerate(spans):
            chunk = processed[(rd >= lo) & (rd <= hi)].copy()
            # Seed ratings from the prior chunk's end-state (None == from scratch),
            # attach the rating columns, then feature the chunk's rows only with
            # every pandas stage resumed from carried counts_state.
            if ratings_cache is not None:
                keyed = ratings_cache.for_chunk(chunk)
                ratings_state = ratings_cache.final_state
            elif rater is None:
                keyed, _cols, ratings_state = store._seeded_rating_frame(chunk, ratings_state)
            else:
                for first, last in spans[len(rating_jobs):i + RATINGS_AHEAD_CHUNKS]:
                    rating_jobs.append(rater.submit(_rate_ahead, processed[(rd >= first) & (rd <= last)]))
                waited_from = time.perf_counter()
                keyed = rating_jobs[i].result()
                rating_wait += time.perf_counter() - waited_from
                rating_jobs[i] = None
            from src.runner_identity import join_runner_columns
            rated = join_runner_columns(chunk, keyed, [col for col in keyed if col not in {"race_id", "horse_id", "horse_name"}])
            if finisher is None:
                tail.write(engineer_features(
                    rated, save=False, skip_ratings=True, counts_state=counts_state,
                ))
            else:
                partial = engineer_features(
                    rated, save=False, skip_ratings=True, counts_state=counts_state, half="first",
                )
                if finishing is not None:
                    finishing.result()
                finishing = finisher.submit(_finish_chunk, partial)
                del partial
            total_rows += len(rated)
            logger.info(
                "chunk %d/%d [%s..%s]: %d rows | cum %d | RSS %.1f GB | %.0fs",
                i + 1, len(spans), str(lo)[:10], str(hi)[:10], len(rated),
                total_rows, _rss_gb(), time.perf_counter() - t0,
            )
            del chunk, keyed, rated
            gc.collect()

        if finisher is None:
            featured_paths, streamed_rows, _ = tail.close()
        else:
            if finishing is not None:
                finishing.result()
            second_half_path = os.path.join(parts_dir, "second_half_state.pkl")
            featured_paths, streamed_rows = finisher.submit(_close_tail, second_half_path).result()
            _add_second_half_state(counts_state, _load_second_half_state(second_half_path))
        if ratings_cache is None and rater is not None:
            ratings_state, rating_seconds = rater.submit(_worker_result).result()
            logger.info("Ratings took %.0fs in the worker; the build waited %.0fs for them",
                        rating_seconds, rating_wait)
        input_signatures = saving_inputs.result()

    # Preserve missing speed histories; no future-dependent global fill.

    # Write only to staging. Consolidation must finish before any of these
    # states, rows or metadata can replace the previous store generation.
    if fe_signature() != fe_sig_at_start:
        # Raising here aborts the staged update, so the previous store stays.
        raise RuntimeError(
            "FE code changed while the store was building; its values describe "
            "neither generation. Rebuild without editing FE sources meanwhile."
        )
    store._save_state(ratings_state or {"elo": None, "glicko": None})
    store._save_counts_state(counts_state)
    meta = {
        "schema_version": STORE_SCHEMA_VERSION,
        "last_date": str(last_date.date()),
        "featured_rows": int(total_rows),
        "processed_rows": int(len(processed)),
        "categorical_columns": categorical_columns(processed),
        "built_at": datetime.now().isoformat(timespec="seconds"),
        "fe_signature": fe_sig_at_start,
        **input_signatures,
    }
    meta["dataset_manifest"] = current_manifest()
    from src.prize_currency_features import reference_signature
    meta["prize_reference_signature"] = reference_signature()
    if reused_ratings:
        meta["rating_reuse"] = reused_ratings

    # ── one file is the store already; otherwise reconcile the schemas ──
    # Later chunks can gain columns or widen a type. Conform each file to the
    # unified schema (add missing columns as typed nulls; never .cast a narrower
    # table) without imputing unknown histories.
    if _finish_featured(featured_paths, streamed_rows, store.featured_path) != total_rows:
        raise RuntimeError("Batched feature consolidation changed the row count")
    logger.info("featured.parquet written (RSS %.1f GB)", _rss_gb())

    with open(store.meta_path, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2, sort_keys=True, default=str)

    # Best-effort temp cleanup: a leftover part is harmless (the next build
    # clears _build_parts first), and on Windows the parquet handle can linger.
    for p in featured_paths:
        try:
            os.remove(p)
        except OSError:
            pass
    try:
        os.rmdir(parts_dir)
    except OSError:
        pass

    logger.info(
        "Prepared: %d featured rows over %s..%s in %.0fs (RSS %.1f GB)",
        total_rows, str(dates[0])[:10], str(dates[-1])[:10],
        time.perf_counter() - t0, _rss_gb(),
    )
    return meta

