"""Immutable source publication and a build-wide dataset manifest.

Collectors retain their native storage. Only fully written, validated versions
are referenced by the atomic pointer. Readers never follow that pointer again
after selecting a manifest. Source generations are retained for run replay.
"""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps
import hashlib
import json
from pathlib import Path
import shutil
from uuid import uuid4

import pandas as pd

import config
from src.ingestion.storage import atomic_json, atomic_parquet, read_json, writer_lock

CONTRACT_VERSION = 1
_current = ContextVar("dataset_manifest", default=None)


def registry_root():
    return Path(config.DATA_DIR) / "sources"


def _hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str, separators=(",", ":")).encode()).hexdigest()


def publish_source(provider, tables, *, contracts, provenance=None, reports=None, artifacts=None):
    """Publish one accepted unit; a failure leaves the previous pointer intact.

    Validation happens before any publication. Content-addressing makes replay
    idempotent. Observations and rejected-link receipts commit in one directory.
    """
    if not provider.replace("_", "").isalnum():
        raise ValueError("Invalid source name")
    base = registry_root() / provider
    with writer_lock(base / ".publish.lock") as acquired:
        if not acquired:
            raise RuntimeError(f"{provider} publication is busy")
        stage = base / (".stage-" + uuid4().hex)
        stage.mkdir(parents=True)
        try:
            entries = {}
            for name, frame in tables.items():
                if not name.replace("_", "").isalnum():
                    raise ValueError("Invalid table name")
                contract = contracts[name]
                contract.validate(frame)
                path = stage / (name + ".parquet")
                atomic_parquet(frame, path)
                import pyarrow.parquet as pq
                schema = pq.read_schema(path)
                entries[name] = {"file": path.name, "sha256": _hash(path), "rows": len(frame),
                                 "schema": {field.name: str(field.type) for field in schema},
                                 "contract": contract.__dict__}
            for name, path in (artifacts or {}).items():
                path = Path(path)
                target = stage / (name + path.suffix)
                before = path.stat()
                shutil.copyfile(path, target)
                after = path.stat()
                if (before.st_mtime_ns, before.st_size) != (after.st_mtime_ns, after.st_size):
                    raise RuntimeError(f"Source changed during capture: {path}")
                entries[name] = {"file": target.name, "sha256": _hash(target), "rows": None,
                                 "schema": None, "contract": {"grain": "reference", "role": "reference"}}
            content = {"contract_version": CONTRACT_VERSION, "provider": provider, "tables": entries,
                       "provenance": provenance or {}}
            version = _digest(content)
            receipt = {**content, "version": version, "reports": reports or {},
                       "published_at": pd.Timestamp.now(tz="UTC").isoformat()}
            atomic_json(stage / "receipt.json", receipt)
            target = base / version
            if target.exists():
                # Read-back verifies an already published version before reuse.
                receipt = read_json(target / "receipt.json")
                verify_source(receipt)
                shutil.rmtree(stage)
            else:
                stage.rename(target)
            atomic_json(base / "latest.json", receipt)
            return receipt
        finally:
            if stage.exists():
                shutil.rmtree(stage)


def verify_source(receipt):
    expected = _digest({key: receipt[key] for key in ("contract_version", "provider", "tables", "provenance")})
    if expected != receipt["version"]:
        raise ValueError("Source receipt does not match its version")
    for entry in receipt["tables"].values():
        path = registry_root() / receipt["provider"] / receipt["version"] / entry["file"]
        if _hash(path) != entry["sha256"]:
            raise ValueError(f"Published source is corrupt: {path}")


def current_manifest():
    return _current.get()


def install_manifest(manifest):
    """Process-pool initializer: all spawned workers read the parent's versions."""
    _current.set(manifest)


@contextmanager
def use_manifest(manifest):
    token = _current.set(manifest)
    try:
        yield manifest
    finally:
        _current.reset(token)


def source_path(provider, table, fallback=None):
    manifest = current_manifest()
    if manifest is None:
        return Path(fallback) if fallback is not None else None
    receipt = manifest["sources"].get(provider)
    if provider == "rtv" and table in {"pre", "links_pre", "observations_pre"}:
        receipt = manifest["sources"].get("rtv_snapshots", receipt)
    entry = receipt and receipt["tables"].get(table)
    if entry is None:
        # A selected missing input stays missing even if a collector creates it.
        return registry_root() / "absent" / provider / (table + ".parquet")
    return registry_root() / receipt["provider"] / receipt["version"] / entry["file"]


def read_table(provider, table, *, linked=False):
    path = source_path(provider, table)
    if path is None or not path.exists():
        return pd.DataFrame()
    frame = pd.read_parquet(path)
    if linked:
        links_path = source_path(provider, "links_" + table)
        if links_path is not None and links_path.exists():
            links = pd.read_parquet(links_path).sort_values("source_row")
            if links.source_row.tolist() != list(range(len(frame))):
                raise ValueError("Source crosswalk does not describe the selected table")
            for column in ("sl_race_id", "sl_horse_id", "match_status"):
                frame[column] = links[column].to_numpy()
    return frame


def model_table(provider, table, path):
    if current_manifest() is None:
        return pd.read_parquet(path)
    frame = read_table(provider, table, linked=True)
    if "match_status" in frame:
        frame = frame.loc[frame.match_status.eq("matched")].copy()
    return frame


def save_manifest(sources, *, purpose="features"):
    payload = {"contract_version": CONTRACT_VERSION, "purpose": purpose, "sources": sources}
    identifier = _digest({"contract_version": CONTRACT_VERSION, "purpose": purpose,
                          "sources": {name: value["version"] for name, value in sources.items()}})
    manifest = {**payload, "id": identifier, "created_at": pd.Timestamp.now(tz="UTC").isoformat()}
    path = registry_root() / "datasets" / (identifier + ".json")
    if not path.exists():
        atomic_json(path, manifest)
    if purpose in {"features", "training"}:
        # Provider pointers may advance independently. Consumers select only
        # this complete dataset commit, never a half-published refresh.
        atomic_json(path.parent/"latest.json", {"id": identifier, "purpose": purpose,
                    "selected_at": pd.Timestamp.now(tz="UTC").isoformat()})
    return read_json(path)


def capture_dataset(*, include_prices=False, canonical_cards=None):
    from src.ingestion.publication import capture_sources
    with writer_lock(registry_root()/".adoption.lock") as acquired:
        if not acquired:
            raise RuntimeError("Source publication is already running; use the last complete dataset or retry shortly")
        with use_manifest(None):
            return save_manifest(capture_sources(include_prices=include_prices, canonical_cards=canonical_cards),
                                 purpose="training" if include_prices else "features")


def select_dataset(*, include_prices=False):
    """Select the last complete publication; building never reruns collectors.

    A first installation adopts its native stores once. Existing publications
    made before the dataset pointer was introduced are migrated without replay.
    """
    root = registry_root()/"datasets"
    pointer = read_json(root/"latest.json")
    if pointer:
        manifest = load_manifest(pointer["id"])
        if {"sl", "fx"}.issubset(manifest["sources"]) and (not include_prices or "betfair" in manifest["sources"]):
            return manifest
    candidates = []
    if not pointer:
        for path in root.glob("*.json"):
            if len(path.stem) != 64:
                continue
            candidate = read_json(path)
            names = set(candidate.get("sources", {}))
            if candidate.get("purpose") in {"features", "training"} and {"sl", "fx"}.issubset(names) and (not include_prices or "betfair" in names):
                candidates.append((path.stat().st_mtime_ns, path.stem))
        if candidates:
            manifest = load_manifest(max(candidates)[1])
            with writer_lock(registry_root()/".adoption.lock") as acquired:
                if acquired and not (root/"latest.json").exists():
                    atomic_json(root/"latest.json", {"id": manifest["id"], "purpose": manifest["purpose"],
                                "selected_at": pd.Timestamp.now(tz="UTC").isoformat()})
            return manifest
    return capture_dataset(include_prices=include_prices)


def load_manifest(identifier):
    if len(identifier) != 64 or any(c not in "0123456789abcdef" for c in identifier):
        raise ValueError("Expected a dataset manifest ID")
    manifest = read_json(registry_root() / "datasets" / (identifier + ".json"))
    if not manifest or manifest.get("id") != identifier:
        raise ValueError("Dataset manifest is missing or invalid")
    expected = _digest({"contract_version": manifest["contract_version"], "purpose": manifest["purpose"],
                        "sources": {name: value["version"] for name, value in manifest["sources"].items()}})
    if expected != identifier:
        raise ValueError("Dataset manifest content does not match its ID")
    for receipt in manifest["sources"].values():
        verify_source(receipt)
    return manifest


def dataset_build(function):
    """Public build entry points select inputs once; explicit test frames stay explicit."""
    @wraps(function)
    def wrapped(*args, **kwargs):
        if current_manifest() is not None:
            return _frame_result(function(*args, **kwargs), current_manifest())
        processed = kwargs.get("processed", kwargs.get("df"))
        if processed is None and args and isinstance(args[0], pd.DataFrame):
            processed = args[0]
        manifest = frame_manifest(processed) if processed is not None else select_dataset()
        with use_manifest(manifest):
            return _frame_result(function(*args, **kwargs), manifest)
    return wrapped


def _frame_result(result, manifest):
    if isinstance(result, pd.DataFrame) and manifest and manifest.get("id"):
        result.attrs["dataset_manifest_id"] = manifest["id"]
    return result


def frame_manifest(frame):
    identifier = frame.attrs.get("dataset_manifest_id")
    if identifier:
        return load_manifest(identifier)
    return frame.attrs.get("dataset_manifest")


def live_dataset(function):
    @wraps(function)
    def wrapped(*args, **kwargs):
        from src.feature_store import FeatureStore
        store = kwargs.get("store") or FeatureStore()
        history_manifest = store.meta().get("dataset_manifest")
        if current_manifest() is not None or not history_manifest:
            return function(*args, **kwargs)
        # Serving reads the store's historical versions. Only expiring card
        # captures are published here; a prediction never runs a backfill or
        # republishes hundreds of thousands of historical observations.
        from src.ingestion.publication import publish_live_snapshots
        cards = args[0] if args else kwargs.get("cards")
        sources = dict(history_manifest["sources"])
        receipt = publish_live_snapshots(cards, sources)
        if receipt is not None:
            sources["rtv_snapshots"] = receipt
        with use_manifest(save_manifest(sources, purpose="serving")):
            return function(*args, **kwargs)
    return wrapped


def processing_inputs(function):
    @wraps(function)
    def wrapped(*args, **kwargs):
        frame = kwargs.get("df", args[0] if args else None)
        if current_manifest() is not None:
            return function(*args, **kwargs)
        manifest = frame_manifest(frame) if frame is not None else select_dataset()
        with use_manifest(manifest):
            return function(*args, **kwargs)
    return wrapped
