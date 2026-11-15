"""Bounded artifact inspection for Features. Browsing never computes features."""
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
import threading
import re
import hashlib
import json

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq

from src.workbench.data import date_range, read_json, records, typed_value
from src.workbench.feature_catalog import FAMILIES, definition, load_sets


class StoreUnavailable(ValueError):
    pass


class Features:
    def __init__(self, catalog):
        self.catalog = catalog
        self.root = catalog.root/"processed/feature_store"
        self._reader_lock = threading.RLock()

    @contextmanager
    def store(self):
        with self._reader_lock, self._store() as store:
            yield store

    @contextmanager
    def _store(self):
        # Do not create directories, a lock file or recover a commit on a GET.
        if not all((self.root/name).is_file() for name in ("featured.parquet", "meta.json", ".store.lock")):
            raise StoreUnavailable("No inspectable feature store. Queue a feature update to build it.")
        if (self.root/".store.lock").stat().st_size == 0:
            raise StoreUnavailable("The feature store has no initialized read lock. Run a feature update.")
        from src.feature_store import FeatureStore
        class ReadOnlyStore(FeatureStore):
            def _recover_commit(self):
                if Path(self._journal_path).exists():
                    raise StoreUnavailable("A feature-store commit needs recovery. Run an update before inspecting it.")
        store = ReadOnlyStore(str(self.root))
        try:
            with store._access():
                yield store
        except RuntimeError as exc:
            raise StoreUnavailable(str(exc)) from exc

    def registry(self):
        return load_sets(self.catalog.base)

    def selection(self, identifier=None):
        registry = self.registry()
        if identifier == "all":
            return None
        sets = registry["feature_sets"]
        identifier = identifier or next((s["id"] for s in sets if s["configured_for_training"]), sets[0]["id"] if sets else None)
        selected = next((s for s in sets if s["id"] == identifier), None)
        if selected is None:
            raise ValueError("Choose a registered feature set")
        return selected

    @staticmethod
    def eligible(schema):
        from src.model import EXCLUDE_COLUMNS
        return [f.name for f in schema if f.name not in EXCLUDE_COLUMNS and
                (pa.types.is_integer(f.type) or pa.types.is_floating(f.type) or pa.types.is_boolean(f.type))]

    def code_signature(self):
        from src.featured_dataset import fe_signature
        return self.catalog.cached(("feature_code",), 10, fe_signature)

    @staticmethod
    def build_id(store, meta):
        stat = Path(store.featured_path).stat()
        identity = [meta.get("fe_signature"), meta.get("data_signature"),
                    (meta.get("dataset_manifest") or {}).get("id"), stat.st_mtime_ns, stat.st_size]
        return hashlib.sha256(json.dumps(identity).encode()).hexdigest()[:16]

    def overview(self, dataset=None):
        registry = self.registry()
        selected_dataset = self.catalog.manifest(dataset).get("id")
        output = {**registry, "selected_dataset": selected_dataset, "families": [
            {"id": key, "label": value[0], "description": value[2]} for key, value in FAMILIES.items()],
            "checked_at": datetime.now(timezone.utc).isoformat(), "code_signature": self.code_signature(),
            "store": {"state": "missing", "rows": 0, "columns": 0, "reasons": []},
            "architecture": "A shared feature store supplies named selections to model roles. Selecting a set here does not change training configuration."}
        try:
            with self.store() as store:
                meta = store.meta()
                schema = pq.read_schema(store.featured_path)
                names = self.eligible(schema)
                stored_dataset = (meta.get("dataset_manifest") or {}).get("id")
                reasons = []
                if meta.get("fe_signature") != output["code_signature"]:
                    reasons.append("Feature definitions changed since this store was built.")
                if not stored_dataset:
                    reasons.append("This legacy store has no recorded source dataset.")
                elif selected_dataset and selected_dataset != stored_dataset:
                    reasons.append("The selected source dataset differs from this store's recorded inputs.")
                missing_state = [n for n in ("processed.parquet", "ratings_state.pkl", "counts_state.pkl") if not (self.root/n).is_file()]
                if missing_state:
                    reasons.append("Missing persisted history/state: "+", ".join(missing_state))
                output["store"] = {"state": "stale" if reasons else "current", "reasons": reasons,
                    "rows": meta.get("featured_rows", pq.ParquetFile(store.featured_path).metadata.num_rows),
                    "columns": len(names), "last_date": meta.get("last_date"),
                    "built_at": meta.get("appended_at") or meta.get("built_at"),
                    "signature": meta.get("fe_signature"), "data_signature": meta.get("data_signature"), "dataset": stored_dataset,
                    "build_id": self.build_id(store, meta),
                    "categorical_columns": len(meta.get("categorical_columns", []))}
                for item in output["feature_sets"]:
                    item["missing_columns"] = [n for n in item["features"] if n not in names]
        except (StoreUnavailable, OSError, ValueError) as exc:
            output["store"]["reasons"] = [str(exc)]
        # Deployment is independent of configured model roles; no model files are loaded.
        pointer = read_json(self.catalog.base/"models/production.json")
        rid = pointer.get("run_id")
        metadata = read_json(self.catalog.root/"runs"/rid/"meta.json") if isinstance(rid, str) and re.fullmatch(r"[A-Za-z0-9_-]{1,100}", rid) else {}
        sig = ((metadata.get("training_config") or {}).get("signatures") or {}).get("fe_signature")
        output["saved_model"] = {"run_id": rid, "feature_signature": sig,
            "state": "unrecorded" if not sig else "definition_match" if sig == output["store"].get("signature") else "definition_mismatch",
            "note": "This checks definition signatures only. Source revisions, fit dates and saved-model input membership need separate qualification."}
        return output

    def catalog_rows(self, feature_set=None, query="", family="", offset=0, limit=60):
        selected = self.selection(feature_set)
        registry = self.registry()
        schema = pa.schema([])
        try:
            with self.store() as store:
                schema = pq.read_schema(store.featured_path)
        except (StoreUnavailable, OSError):
            pass
        names = set(self.eligible(schema))
        names.update(n for item in registry["feature_sets"] for n in item["features"])
        if selected:
            names &= set(selected["features"])
        rows = []
        for name in sorted(names):
            row = definition(name)
            if family and row["family"] != family:
                continue
            if query.casefold() not in (name+" "+row["label"]+" "+row["description"]).casefold():
                continue
            memberships = [s["id"] for s in registry["feature_sets"] if name in s["features"]]
            row.update({"stored": name in schema.names, "dtype": str(schema.field(name).type) if name in schema.names else None,
                        "feature_sets": memberships, "model_roles": [m["id"] for m in registry["model_roles"] if set(m["feature_sets"]) & set(memberships)]})
            rows.append(row)
        offset, limit = max(0, int(offset)), min(max(1, int(limit)), 100)
        return {"rows": rows[offset:offset+limit], "total": len(rows), "offset": offset, "limit": limit,
                "selected_set": selected["id"] if selected else "all"}

    @staticmethod
    def scan(path, columns, expression, limit):
        scanner = ds.dataset(path, format="parquet").scanner(columns=columns, filter=expression, batch_size=512)
        frame = scanner.head(limit+1).to_pandas()
        return frame.iloc[:limit].copy(), len(frame) > limit

    def race(self, race_id, feature_set=None, family="", query="", offset=0):
        selected = self.selection(feature_set)
        with self.store() as store:
            schema = pq.read_schema(store.featured_path)
            names = selected["features"] if selected else self.eligible(schema)
            names = [n for n in names if (not family or definition(n)["family"] == family) and query.casefold() in n.casefold()]
            offset = max(0, int(offset))
            page = names[offset:offset+24]
            present = [n for n in page if n in schema.names]
            keys = [n for n in ("race_id", "horse_id", "horse_name", "race_date", "track", "off_time") if n in schema.names]
            if not {"race_id", "horse_id"}.issubset(keys):
                raise ValueError("Stored features lack canonical runner IDs")
            expression = ds.field("race_id") == typed_value(schema, "race_id", race_id)
            frame, truncated = self.scan(store.featured_path, list(dict.fromkeys(keys+present)), expression, 300)
            if not frame.empty:
                frame = frame.sort_values("horse_name" if "horse_name" in frame else "horse_id", kind="stable")
            meta = store.meta()
            invalid = [{"horse_id": str(row.horse_id), "column": name}
                       for name in present for row in frame.loc[np.isinf(pd.to_numeric(frame[name], errors="coerce")), ["horse_id"]].itertuples()]
            return {"rows": records(frame), "columns": [definition(n) | {"stored": n in schema.names} for n in page],
                    "invalid_values": invalid,
                    "total_columns": len(names), "offset": offset, "limit": 24, "truncated": truncated,
                    "race_id": str(race_id), "feature_set": selected["id"] if selected else "all",
                    "feature_set_signature": selected["signature"] if selected else None,
                    "store": {"signature": meta.get("fe_signature"), "dataset": (meta.get("dataset_manifest") or {}).get("id"),
                              "last_date": meta.get("last_date"), "build_id": self.build_id(store, meta)},
                    "message": "Stored values from this generation. No historical recalculation or use of today's accumulated state."}

    def races(self, start, end):
        first, last = date_range(start, end, 93)
        with self.store() as store:
            schema = pq.read_schema(store.featured_path)
            columns = [n for n in ("race_id", "race_date", "off_time", "track", "race_name") if n in schema.names]
            exp = ((ds.field("race_date") >= typed_value(schema, "race_date", first.isoformat())) &
                   (ds.field("race_date") < typed_value(schema, "race_date", (last+timedelta(days=1)).isoformat())))
            frame, clipped = self.scan(store.featured_path, columns, exp, 10000)
            order = [n for n in ("race_date", "off_time") if n in frame]
            frame = frame.drop_duplicates("race_id").sort_values(order, ascending=False, kind="stable").head(50)
            return {"rows": records(frame), "truncated": clipped, "scope": "Up to 50 races from the first 10,000 stored runners in the range."}

    def value(self, race_id, horse_id, column):
        spec = definition(column)
        with self.store() as store:
            schema = pq.read_schema(store.featured_path)
            if column not in self.eligible(schema):
                raise ValueError("Choose a stored eligible numeric feature")
            keys = [n for n in ("race_id", "horse_id", "horse_name", "race_date", "track", "off_time") if n in schema.names]
            support = {"atr": "atr_observation_count_5", "rtv": "rtv_history_count3", "bha": "bha_report_count_5"}.get(spec["family"])
            columns = list(dict.fromkeys(keys+[column]+([support] if support in schema.names else [])))
            expression = ((ds.field("race_id") == typed_value(schema, "race_id", race_id)) &
                          (ds.field("horse_id") == typed_value(schema, "horse_id", horse_id)))
            frame, _ = self.scan(store.featured_path, columns, expression, 2)
            if len(frame) != 1:
                raise ValueError("Expected exactly one stored runner for these canonical IDs")
            row = records(frame)[0]
            invalid = bool(np.isinf(float(frame[column].iloc[0]))) if not pd.isna(frame[column].iloc[0]) else False
            meta = store.meta()
            history, history_message = [], "Prepared history is unavailable."
            if Path(store.processed_path).is_file() and row.get("race_date"):
                hs = pq.read_schema(store.processed_path)
                if {"horse_id", "race_id", "race_date"}.issubset(hs.names):
                    cols = [n for n in ("race_id", "horse_id", "race_date", "off_time", "track", "horse_name", "finish_position") if n in hs.names]
                    exp = ((ds.field("horse_id") == typed_value(hs, "horse_id", horse_id)) &
                           (ds.field("race_date") <= typed_value(hs, "race_date", str(row["race_date"])[:10])))
                    past, clipped = self.scan(store.processed_path, cols, exp, 1000)
                    past = past.loc[past.race_id.astype(str).ne(str(race_id))]
                    target = pd.to_datetime(str(row["race_date"])[:10]+" "+str(row.get("off_time") or "00:00"), utc=True, errors="coerce")
                    moments = pd.to_datetime(past.race_date.astype(str).str[:10]+" "+past.get("off_time", pd.Series("00:00", index=past.index)).fillna("00:00").astype(str), utc=True, errors="coerce")
                    past = past.loc[moments.lt(target)].sort_values(["race_date", "off_time"] if "off_time" in past else ["race_date"], ascending=False).head(10)
                    history = records(past)
                    history_message = "Prior starts for context, not an exact contributor trace. Per-observation eligibility is not recorded in this store." + (" History query was capped at 1,000 starts." if clipped else "")
            missing = row[column] is None
            note = "This stored value is missing. Applicability and exact cause are not recorded; inspect support and source evidence."
            if support and row.get(support) == 0 and missing:
                note = "No eligible history support is recorded for this source family. This is not a measured zero."
            elif not missing:
                note = "A stored numeric value; zero is preserved as zero."
            if invalid:
                note = "This stored value is infinite and invalid for feature inspection; it is not a missing observation."
            import config
            return {"feature": spec, "runner": row, "missing": missing and not invalid, "invalid": invalid, "explanation": note,
                    "support": {"column": support, "value": row.get(support), "note": "Family-level support; individual metric support can differ."} if support else None,
                    "history": history, "history_message": history_message,
                    "generation": meta.get("fe_signature"), "build_id": self.build_id(store, meta), "dataset": (meta.get("dataset_manifest") or {}).get("id"),
                    "availability": "Current BHA policy: "+config.BHA_FEATURE_AVAILABILITY_POLICY+". Original publication/amendment times are unverified; actual capture times remain source evidence." if spec["family"] == "bha" else spec["history_rule"]}

    def quality(self, start, end, feature_set=None, family=""):
        first, last = date_range(start, end, 31)
        selected = self.selection(feature_set)
        with self.store() as store:
            schema = pq.read_schema(store.featured_path)
            requested = selected["features"] if selected else self.eligible(schema)
            requested = [n for n in requested if not family or definition(n)["family"] == family]
            if not requested:
                raise ValueError("No feature columns in this selection. Choose another set/family or update the store.")
            if len(requested) > 256:
                raise ValueError("Choose a family or feature set with at most 256 columns for a bounded quality check")
            missing = [n for n in requested if n not in schema.names]
            names = [n for n in requested if n in schema.names]
            if "race_date" not in schema.names:
                raise ValueError("Stored features have no race dates")
            exp = ((ds.field("race_date") >= typed_value(schema, "race_date", first.isoformat())) &
                   (ds.field("race_date") < typed_value(schema, "race_date", (last+timedelta(days=1)).isoformat())))
            keys = [n for n in ("race_id", "horse_id") if n in schema.names]
            frame, truncated = self.scan(store.featured_path, list(dict.fromkeys(keys+names)), exp, 5000)
            rows = []
            for name in names:
                values = pd.to_numeric(frame[name], errors="coerce").to_numpy(dtype=float, na_value=np.nan)
                finite = values[np.isfinite(values)]
                infinite = int(np.isinf(values).sum())
                rows.append({"name": name, "family": definition(name)["family"], "missing": int(np.isnan(values).sum()),
                    "missing_pct": round(float(np.isnan(values).mean()*100), 2) if len(values) else None,
                    "infinite": infinite, "distinct": int(len(np.unique(finite))), "zeros": int((finite == 0).sum()),
                    "minimum": float(finite.min()) if len(finite) else None, "maximum": float(finite.max()) if len(finite) else None,
                    "mean": float(finite.mean()) if len(finite) else None,
                    "state": "invalid" if infinite else "no_values" if not len(finite) else "constant" if len(np.unique(finite)) == 1 else "observed"})
            return {"start": first.isoformat(), "end": last.isoformat(), "rows_checked": len(frame),
                    "columns_checked": len(names), "missing_columns": missing, "truncated": truncated,
                    "duplicate_keys": int(frame.duplicated(keys).sum()) if len(keys) == 2 else None,
                    "rows": rows, "generation": store.meta().get("fe_signature"),
                    "build_id": self.build_id(store, store.meta()),
                    "feature_set": selected["id"] if selected else "all", "feature_set_signature": selected["signature"] if selected else None,
                    "scope": "At most 5,000 stored runners, 256 columns and 31 days. Missing and constant values require applicability/source review; this is not a leakage certificate."}


def update_store(catalog, dataset, rebuild=False, progress=lambda **kw: None):
    """Explicit worker action. Shared values, independent of selected model inputs."""
    from src.data_processor import process_data
    from src.feature_store import FeatureStore
    from src.source_dataset import load_manifest, use_manifest
    from src.store_refresh import append_current_store
    from src.store_build import build_store
    manifest = load_manifest(dataset, root=catalog.sources)
    with use_manifest(manifest):
        progress(phase="preparing", message="Preparing the selected accepted dataset")
        processed = process_data()
        store = FeatureStore(str(catalog.root/"processed/feature_store"))
        progress(phase="checking", rows=len(processed), message="Checking whether the existing store can append")
        if not rebuild and append_current_store(processed, store=store):
            operation = "append_or_current"
        else:
            progress(phase="building", rows=len(processed), message="Building shared feature history; live stage messages are in the log")
            with store._staged_update() as pending:
                build_store(pending, 60000, processed=processed)
            operation = "rebuild"
        return {"operation": operation, "meta": store.meta(), "model_fitted": False}
