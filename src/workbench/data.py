"""Bounded, read-only queries for the ingestion workbench."""
from __future__ import annotations

from contextlib import contextmanager
from datetime import date, datetime, timedelta, timezone
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import re
import sqlite3
import threading
import time
import zlib

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src.utils import normalise_off_time_key, normalise_track_key
from src.source_health import POLICIES, consumer_readiness

SOURCES = {"sl": "Sporting Life", "rtv": "Racing TV", "atr": "At The Races",
           "betfair": "Betfair", "bha": "BHA stewards", "pedigree": "Pedigree",
           "fx": "Currency references", "weather": "Weather",
           "foreign": "France/UAE results"}
HEX = re.compile(r"^[a-f0-9]{64}$")


def read_json(path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8-sig"))
    except FileNotFoundError:
        return {}


def records(frame):
    return json.loads(frame.to_json(orient="records", date_format="iso"))


def fingerprint(rows):
    return hashlib.sha256(json.dumps(rows, sort_keys=True, default=str, allow_nan=False).encode()).hexdigest()


def date_range(start, end, maximum=366):
    first, last = date.fromisoformat(start), date.fromisoformat(end)
    if first > last or (last-first).days >= maximum:
        raise ValueError(f"Choose an ordered range of at most {maximum} days")
    return first, last


@contextmanager
def connection(path):
    if not Path(path).is_file():
        raise FileNotFoundError("Source database is not available")
    con = sqlite3.connect(Path(path).resolve().as_uri()+"?mode=ro", uri=True, timeout=5)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA query_only=ON")
    deadline = time.monotonic()+20
    con.set_progress_handler(lambda: time.monotonic() > deadline, 10000)
    try:
        con.execute("BEGIN")
        yield con
    finally:
        con.close()


@lru_cache(maxsize=96)
def parquet_rows(path, modified, size, filters, columns=None):
    return pq.read_table(path, filters=list(filters) or None, columns=list(columns) if columns is not None else None).to_pandas()


def read_parquet(path, filters=(), columns=None):
    path = Path(path)
    stat = path.stat()
    return parquet_rows(str(path), stat.st_mtime_ns, stat.st_size, tuple(filters),
                        tuple(columns) if columns is not None else None).copy()


def typed_value(schema, column, value):
    kind = schema.field(column).type
    if pa.types.is_integer(kind):
        return int(value)
    if pa.types.is_floating(kind):
        return float(value)
    if pa.types.is_date(kind):
        return date.fromisoformat(str(value)[:10])
    if pa.types.is_timestamp(kind):
        return pd.Timestamp(value).to_pydatetime()
    return str(value)


def select_rows(path, ordinals):
    """Read only row groups containing selected source-row ordinals."""
    wanted = sorted(set(int(i) for i in ordinals))
    file = pq.ParquetFile(path)
    parts, offset = [], 0
    for group in range(file.num_row_groups):
        count = file.metadata.row_group(group).num_rows
        local = [i-offset for i in wanted if offset <= i < offset+count]
        if local:
            frame = file.read_row_group(group).take(pa.array(local)).to_pandas()
            frame["_source_row"] = [i+offset for i in local]
            parts.append(frame)
        offset += count
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


class Catalog:
    def __init__(self, data_dir, base_dir=None):
        self.root = Path(data_dir).resolve()
        self.base = Path(base_dir or self.root.parent).resolve()
        self.db = self.root/"races.db"
        self.sources = self.root/"sources"
        self._cache = {}
        self._lock = threading.Lock()

    def cached(self, key, seconds, function):
        with self._lock:
            entry = self._cache.get(key)
            if entry and time.monotonic()-entry[0] < seconds:
                return entry[1]
        result = function()
        with self._lock:
            self._cache[key] = (time.monotonic(), result)
        return result

    def manifest(self, identifier=None):
        identifier = identifier or read_json(self.sources/"datasets/latest.json").get("id")
        if not identifier:
            return {"id": None, "sources": {}}
        if not HEX.fullmatch(identifier):
            raise ValueError("Invalid dataset ID")
        manifest = read_json(self.sources/"datasets"/(identifier+".json"))
        if manifest.get("id") != identifier:
            raise ValueError("Dataset is missing or its identity is invalid")
        from src.source_dataset import _digest
        expected = _digest({"contract_version": manifest["contract_version"], "purpose": manifest["purpose"],
                            "sources": {name: value["version"] for name, value in manifest["sources"].items()}})
        if expected != identifier:
            raise ValueError("Dataset manifest content does not match its ID")
        return manifest

    def manifests(self):
        files = sorted((self.sources/"datasets").glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
        return [{"id": p.stem, "created_at": read_json(p).get("created_at"),
                 "purpose": read_json(p).get("purpose")} for p in files if HEX.fullmatch(p.stem)][:100]

    def table_path(self, receipt, table):
        if receipt.get("provider") not in SOURCES or not HEX.fullmatch(receipt.get("version", "")):
            raise ValueError("Invalid source receipt")
        entry = receipt.get("tables", {}).get(table)
        if not entry:
            return None
        path = self.sources/receipt["provider"]/receipt["version"]/entry["file"]
        if path.resolve().parent != (self.sources/receipt["provider"]/receipt["version"]).resolve():
            raise ValueError("Invalid source file")
        return path

    def latest_day(self):
        if not self.db.exists():
            return None
        with connection(self.db) as con:
            return con.execute("SELECT max(race_date) FROM results").fetchone()[0]

    def workers(self):
        paths = {"atr": "atr_cache/rebuild_v2/status.json", "bha": "bha/status.json",
                 "rtv": "rtv_cache/pipeline/history_status.json", "betfair": "betfair/history/status.json"}
        output = []
        for provider, relative in paths.items():
            value = read_json(self.root/relative)
            if not value:
                continue
            stamp = value.get("checked_at")
            age = (datetime.now(timezone.utc)-pd.Timestamp(stamp).to_pydatetime()).total_seconds() if stamp else None
            process_alive = None
            if value.get("pid"):
                import psutil
                try:
                    process = psutil.Process(int(value["pid"]))
                    process_alive = "python" in process.name().lower() and (not stamp or process.create_time() <= pd.Timestamp(stamp).timestamp()+10)
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    process_alive = False
            active = value.get("state") in {"running", "waiting", "importing_legacy"}
            stale = active and ((process_alive is False) or age is None or age > 300)
            launch = read_json(self.root/"atr_cache/rebuild_v2/launch_options.json") if provider == "atr" else {}
            output.append({"source": provider, "label": SOURCES[provider], "status": value,
                           "age_seconds": age, "process_alive": process_alive,
                           "active": active and not stale, "stale": stale,
                           "request_delay": launch.get("request_delay")})
        return output

    def overview(self, dataset=None):
        manifest = self.manifest(dataset)
        attempt = read_json(self.root/"source_refresh/status.json")
        workers = self.workers()
        sources = []
        for provider, label in SOURCES.items():
            receipt = manifest.get("sources", {}).get(provider, {})
            newest = read_json(self.sources/provider/"latest.json")
            adoption = read_json(self.sources/provider/"publication_status.json")
            native = next((w for w in workers if w["source"] == provider), None)
            checked = attempt.get("sources", {}).get(provider, {})
            if native and native["status"].get("checked_at", "") > checked.get("checked_at", attempt.get("checked_at", "")):
                checked = native["status"]
            from src.ingestion.publication import source_health
            health = source_health(checked, receipt or None)
            checked_at = checked.get("checked_at") or attempt.get("checked_at")
            age = (datetime.now(timezone.utc)-pd.Timestamp(checked_at).to_pydatetime()).total_seconds() if checked_at else None
            sources.append({"id": provider, "label": label, **health,
                            "checked_at": checked_at, "check_stale": age is None or age > 86400,
                            "required": "features" in POLICIES.get(provider, {}).get("required_for", ()),
                            "publication_lag_days": POLICIES.get(provider, {}).get("publication_lag_days"),
                            "worker": native, "adoption": adoption,
                            "tables": receipt.get("tables", {}), "provenance": receipt.get("provenance", {}),
                            "newer_source_version": bool(newest and newest.get("version") != receipt.get("version")),
                            "native_available": self.native_available(provider)})
        readiness = consumer_readiness({s["id"]: {"state": s["attempt_state"], "reason": s["reason"]} for s in sources}, purpose="features")
        readiness["blockers"] = [SOURCES.get(reason.split(":")[0], reason.split(":")[0])+":"+reason.split(":", 1)[1] for reason in readiness["blockers"]]
        for source in sources:
            if source["required"] and not source["accepted_version"]:
                readiness["blockers"].append(f"{source['label']}: no accepted version in this dataset")
            if source["required"] and source["check_stale"]:
                readiness["blockers"].append(f"{source['label']}: coverage check is missing or over 24 hours old")
        readiness["ready"] = not readiness["blockers"]
        def stats():
            if not self.db.exists():
                return {"runners": 0, "races": 0, "first_date": None, "latest_date": None}
            with connection(self.db) as con:
                return dict(con.execute("SELECT count(*) AS runners,count(DISTINCT race_id) AS races,min(race_date) AS first_date,max(race_date) AS latest_date FROM results").fetchone())
        return {"dataset": {k: v for k, v in manifest.items() if k != "sources"},
                "sources": sources, "workers": workers, "database": self.cached("stats", 120, stats),
                "readiness": readiness, "generated_at": datetime.now(timezone.utc).isoformat()}

    def native_available(self, provider):
        paths = {"sl": self.db, "pedigree": self.db, "weather": self.db,
                 "atr": self.root/"atr_cache/rebuild_v2/rebuild.sqlite", "bha": self.root/"bha/reports.sqlite",
                 "rtv": self.native_rtv("runs"), "betfair": self.root/"processed/betfair_bsp.parquet",
                 "fx": self.base/"reference/ecb_gbp_per_eur.csv",
                 "foreign": self.root/"foreign_form/races_v2.parquet"}
        return paths[provider].exists()

    def races(self, start, end, query="", track="", limit=50, offset=0):
        date_range(start, end, 93)
        limit, offset = min(max(int(limit), 1), 100), max(int(offset), 0)
        if offset > 100000:
            raise ValueError("Page offset is too large")
        if not self.db.exists():
            return {"rows": [], "total": 0}
        where, params = "race_date BETWEEN ? AND ?", [start, end]
        if track:
            where += " AND track=?"
            params.append(track)
        if query:
            if len(query) > 120:
                raise ValueError("Search is too long")
            term = "%"+query.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")+"%"
            where += " AND race_id IN (SELECT race_id FROM results WHERE race_date BETWEEN ? AND ? AND (horse_name LIKE ? ESCAPE '\\' OR race_name LIKE ? ESCAPE '\\' OR track LIKE ? ESCAPE '\\' OR race_id=?))"
            params += [start, end, term, term, term, query]
        sql = ("SELECT race_id,race_date,track,off_time,race_name,max(race_type) AS race_type,"
               "max(distance_furlongs) AS distance_furlongs,max(going) AS going,max(surface) AS surface,"
               "count(*) AS stored_runners,max(num_runners) AS expected_runners,sum(won=1) AS winners,"
               "count(*) OVER() AS total FROM results WHERE "+where+
               " GROUP BY race_id,race_date,track,off_time,race_name ORDER BY race_date DESC,off_time,track LIMIT ? OFFSET ?")
        with connection(self.db) as con:
            rows = [dict(r) for r in con.execute(sql, [*params, limit, offset])]
        total = rows[0]["total"] if rows else 0
        for row in rows:
            row.pop("total", None)
        return {"rows": rows, "total": total, "limit": limit, "offset": offset}

    def canonical(self, race_id):
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,80}", str(race_id)):
            raise ValueError("Invalid race ID")
        with connection(self.db) as con:
            frame = pd.read_sql_query("SELECT * FROM results WHERE race_id=? ORDER BY finish_position,horse_name", con, params=(race_id,))
        if frame.empty:
            raise ValueError("Race was not found")
        return frame

    def validate_race(self, frame):
        from src.database import _validate_race_results
        findings = []
        try:
            _validate_race_results(frame, str(frame.race_date.iloc[0]), str(frame.race_id.iloc[0]))
        except ValueError as exc:
            findings.append({"severity": "error", "code": "invalid_field", "message": str(exc)})
        for column in ("track", "off_time", "distance_furlongs", "surface", "going"):
            if column in frame and frame[column].nunique(dropna=False) > 1:
                findings.append({"severity": "warning", "code": "inconsistent_"+column,
                                 "message": f"Runners disagree on {column.replace('_', ' ')}"})
        for column in ("distance_furlongs", "weight_lbs", "odds", "prize_money", "age", "draw", "lengths_behind"):
            if column not in frame:
                continue
            values = pd.to_numeric(frame[column], errors="coerce")
            invalid = values.le(0) if column in {"distance_furlongs", "weight_lbs"} else values.lt(1) if column == "odds" else values.lt(0)
            if invalid.any():
                findings.append({"severity": "error", "code": "invalid_"+column,
                                 "message": f"{column.replace('_', ' ')} contains an impossible non-positive or negative value"})
        return findings

    def race(self, race_id, dataset=None):
        frame = self.canonical(race_id)
        first = records(frame.head(1))[0]
        header = {c: first.get(c) for c in ("race_id", "race_date", "track", "off_time", "race_name", "race_type", "race_class", "distance_furlongs", "going", "surface", "num_runners", "prize_money")}
        off = normalise_off_time_key(first.get("off_time", ""))
        if re.fullmatch(r"\d{2}:\d{2}", off):
            local = pd.Timestamp(str(first["race_date"])+" "+off, tz="UTC").tz_convert("Europe/London")
            header["off_time_local"] = local.strftime("%H:%M %Z")
        else:
            header["off_time_local"] = "Off-time unknown"
        return {"header": header, "rows": records(frame), "fingerprint": fingerprint(records(frame)),
                "findings": self.validate_race(frame), "dataset_id": self.manifest(dataset).get("id")}

    def native_rtv(self, table):
        generation = read_json(self.root/"rtv_cache/pipeline/generation.json")
        relative = generation.get("files", {}).get(table)
        root = self.root/"rtv_cache"
        path = root/relative if relative else root/("feature_"+table+".parquet")
        if not path.resolve().is_relative_to(root.resolve()):
            raise ValueError("Invalid RTV generation path")
        return path

    def accepted_table(self, receipt, name, race):
        path = self.table_path(receipt, name)
        if path is None or not path.exists() or path.suffix != ".parquet":
            return pd.DataFrame()
        schema = pq.read_schema(path)
        links = self.table_path(receipt, "links_"+name)
        if links and links.exists():
            link_schema = pq.read_schema(links)
            selected = read_parquet(links, [("sl_race_id", "=", typed_value(link_schema, "sl_race_id", race.race_id.iloc[0]))])
            selected = selected.head(300)
            frame = select_rows(path, selected.source_row.tolist())
            if not frame.empty:
                frame = frame.merge(selected, left_on="_source_row", right_on="source_row", how="left", validate="one_to_one", suffixes=("", "_link"))
                timing = self.table_path(receipt, "observations_"+name)
                if timing and timing.exists():
                    meta = select_rows(timing, selected.source_row.tolist())
                    frame = frame.merge(meta, on="_source_row", how="left", suffixes=("", "_timing"))
            return frame
        filters = []
        if "race_id" in schema.names:
            filters = [("race_id", "=", typed_value(schema, "race_id", race.race_id.iloc[0]))]
        elif "horse_id" in schema.names:
            filters = [("horse_id", "in", tuple(typed_value(schema, "horse_id", i) for i in race.horse_id.dropna()))]
        elif "race_date" in schema.names:
            filters = [("race_date", "=", typed_value(schema, "race_date", race.race_date.iloc[0]))]
        elif "track" in schema.names:
            frame = read_parquet(path)
            return frame.loc[frame.track.map(normalise_track_key).eq(normalise_track_key(race.track.iloc[0]))]
        elif "observation_date" in schema.names:
            frame = read_parquet(path, [("observation_date", "<", typed_value(schema, "observation_date", race.race_date.iloc[0]))])
            return frame.tail(3)
        else:
            return pd.DataFrame()
        return read_parquet(path, filters).head(300)

    def native_tables(self, provider, race):
        race_id, day = str(race.race_id.iloc[0]), str(race.race_date.iloc[0])
        if provider == "sl":
            return {"results": race}
        if provider == "pedigree":
            ids = race.horse_id.dropna().astype(str).tolist()
            with connection(self.db) as con:
                if not con.execute("SELECT 1 FROM sqlite_master WHERE name='horse_pedigree'").fetchone():
                    return {}
                return {"horses": pd.read_sql_query("SELECT * FROM horse_pedigree WHERE horse_id IN ("+",".join("?" for _ in ids)+")", con, params=ids)} if ids else {}
        if provider == "weather":
            cols = [c for c in race if c.startswith("weather_")]
            return {"races": race[["race_id", "race_date", *cols]].drop_duplicates("race_id")}
        if provider == "bha":
            return self.bha_native(day)
        if provider == "fx":
            units = read_json(self.base/"reference/track_prize_currency.json").get("currency_by_track", {})
            mapping = [{"track": key, "currency": value} for key, value in units.items()
                       if normalise_track_key(key) == normalise_track_key(race.track.iloc[0])]
            path = self.base/"reference/ecb_gbp_per_eur.csv"
            rates = pd.read_csv(path) if path.exists() else pd.DataFrame()
            if not rates.empty:
                rates = rates.loc[rates.observation_date.astype(str).lt(day)].tail(3)
            return {"currency_units": pd.DataFrame(mapping), "rate_values": rates}
        if provider == "atr" and (self.root/"atr_cache/rebuild_v2/rebuild.sqlite").exists():
            with connection(self.root/"atr_cache/rebuild_v2/rebuild.sqlite") as con:
                rows = [json.loads(r[0]) for r in con.execute("SELECT payload FROM runners WHERE race_id=?", (race_id,))]
            return {"sectionals": pd.DataFrame(rows)}
        names = {"rtv": ("runs", "cards", "pre", "text", "scores"), "betfair": ("bsp", "bsp_early", "timing", "place_bsp")}.get(provider, ())
        output = {}
        for name in names:
            path = self.native_rtv(name) if provider == "rtv" else self.root/"processed"/("betfair_"+name+".parquet")
            if not path.exists():
                continue
            schema = pq.read_schema(path)
            if "race_id" in schema.names:
                filters = [("race_id", "=", typed_value(schema, "race_id", race_id))]
            elif "race_date" in schema.names:
                filters = [("race_date", "=", typed_value(schema, "race_date", day))]
            else:
                continue
            frame = read_parquet(path, filters)
            if not frame.empty and "race_id" not in schema.names:
                from src.source_contracts import resolve_runners
                with connection(self.db) as con:
                    canonical = pd.read_sql_query("SELECT * FROM results WHERE race_date=?", con, params=(day,))
                kwargs = {}
                if provider == "betfair":
                    from src.betfair import bf_horse_key, bf_track_key
                    kwargs = {"horse_key": bf_horse_key, "track_key": bf_track_key}
                links = resolve_runners(frame, canonical, provider=provider, **kwargs)
                mask = links.sl_race_id.astype("string").eq(race_id).fillna(False)
                frame = pd.concat([frame.reset_index(drop=True), links.add_prefix("link_")], axis=1).loc[mask.to_numpy()]
            output[name] = frame.head(300)
        return output

    def bha_native(self, day):
        path = self.root/"bha/reports.sqlite"
        if not path.exists():
            return {}
        with connection(path) as con:
            reports = [dict(json.loads(r["row_json"]), observed_at=r["observed_at"], available_at=r["observed_at"],
                            source_url=r["source_url"], source_sha256=r["source_sha256"], source_kind=r["source_kind"], record_hash=r["record_hash"])
                       for r in con.execute("SELECT * FROM reports WHERE race_date=? LIMIT 100", (day,))]
            documents, notes, revisions = [], [], []
            for report in reports:
                key = report["report_key"]
                for row in con.execute("SELECT * FROM documents WHERE report_key=?", (key,)):
                    doc = dict(row)
                    parsed = json.loads(doc.pop("parsed_json") or "{}")
                    documents.append({**doc, "text": parsed.get("text"), "sections_json": parsed.get("sections"), "source_url": doc["url"], "source_sha256": doc["sha256"]})
                    notes.extend({"report_key": key, "source_url": doc["url"], "source_sha256": doc["sha256"], **note} for note in parsed.get("races", []))
                revisions.extend({"report_key": key, **dict(r)} for r in con.execute("SELECT record_hash,observed_at,source_url,source_sha256,source_kind FROM revisions WHERE report_key=? ORDER BY observed_at DESC LIMIT 20", (key,)))
        return {"reports": pd.DataFrame(reports), "documents": pd.DataFrame(documents), "race_notes": pd.DataFrame(notes), "revisions": pd.DataFrame(revisions)}

    def qualify_bha(self, tables, race):
        tables = {k: v.copy() for k, v in tables.items()}
        reports = tables.get("reports", pd.DataFrame())
        if reports.empty or "course" not in reports:
            return {}, "No fixture report for this race's course and date"
        reports = reports.loc[reports.course.map(normalise_track_key).eq(normalise_track_key(race.track.iloc[0]))]
        keys = set(reports.report_key)
        if not keys:
            return {}, "No BHA fixture report for this course; BHA covers British fixtures"
        output = {"reports": reports}
        for name, frame in tables.items():
            if name != "reports" and not frame.empty and "report_key" in frame:
                output[name] = frame.loc[frame.report_key.isin(keys)]
        notes = output.get("race_notes", pd.DataFrame())
        if not notes.empty:
            day = str(race.race_date.iloc[0])
            with connection(self.db) as con:
                field = pd.read_sql_query("SELECT DISTINCT race_id,track,off_time FROM results WHERE race_date=?", con, params=(day,))
            same = field.loc[field.track.map(normalise_track_key).eq(normalise_track_key(race.track.iloc[0]))]
            off = normalise_off_time_key(race.off_time.iloc[0])
            unique = same.loc[same.off_time.map(normalise_off_time_key).eq(off)].race_id.nunique() == 1
            local = pd.Timestamp(day+" "+off, tz="UTC").tz_convert("Europe/London").strftime("%H:%M")
            notes = notes.loc[notes.off_time_local.map(normalise_off_time_key).eq(local)]
            if not unique or notes.report_key.duplicated().any():
                output["unassigned_race_notes"] = notes
                output["race_notes"] = notes.iloc[:0]
                return output, "Ambiguous race association; fixture context only, no runner attribution"
            output["race_notes"] = notes
        return output, "Fixture context; race notes use a unique date/course/local-time match. No runner attribution."

    def race_source(self, race_id, provider, layer="accepted", dataset=None):
        if provider not in SOURCES or layer not in {"accepted", "native"}:
            raise ValueError("Invalid source or data layer")
        race = self.canonical(race_id)
        manifest = self.manifest(dataset)
        receipt = manifest.get("sources", {}).get(provider)
        message = None
        if layer == "native":
            tables = self.native_tables(provider, race)
        elif receipt:
            tables = {name: self.accepted_table(receipt, name, race) for name, entry in receipt["tables"].items()
                      if not name.startswith(("links_", "observations_")) and entry.get("schema") is not None}
        else:
            tables, message = {}, "This source is absent from the selected dataset; native observations are shown separately"
        if provider == "bha":
            tables, message = self.qualify_bha(tables, race)
        return {"source": provider, "label": SOURCES[provider], "layer": layer,
                "dataset_id": manifest.get("id"), "version": receipt.get("version") if receipt and layer == "accepted" else None,
                "message": message, "tables": {name: {"rows": records(frame), "count": len(frame)} for name, frame in tables.items()},
                "identity": "Source-native fixture/race identities" if provider == "bha" else "Accepted crosswalk receipts" if layer == "accepted" else "Native observations; not necessarily published"}

    def prepared(self, race_id, dataset=None):
        manifest = self.manifest(dataset)
        receipt = manifest.get("sources", {}).get("sl")
        if not receipt:
            raise ValueError("Publish an accepted dataset before previewing prepared inputs")
        native = self.canonical(race_id)
        frame = self.accepted_table(receipt, "results", native)
        if frame.empty:
            raise ValueError("Race is absent from the selected accepted dataset")
        from src.data_processor import process_data
        from src.source_dataset import use_manifest
        with use_manifest(manifest):
            prepared = process_data(frame.copy())
        original, after = records(frame), records(prepared)
        by_id = {str(row.get("horse_id")): row for row in original}
        changes = []
        for row in after:
            before = by_id.get(str(row.get("horse_id")), {})
            for column, value in row.items():
                if column in before and before[column] != value:
                    changes.append({"horse_name": row.get("horse_name"), "field": column, "original": before[column], "prepared": value})
        return {"dataset_id": manifest["id"], "original_count": len(original), "prepared_count": len(after),
                "rows": after, "changes": changes, "message": "Uses the existing processing pipeline on this accepted race. No features are generated or files written. Categorical codes are preview-local."}

    def evidence(self, provider, sha):
        if provider not in SOURCES or not HEX.fullmatch(sha):
            raise ValueError("Invalid evidence reference")
        paths = {"sl": ["sl_cache/raw.sqlite"], "atr": ["atr_cache/raw.sqlite"],
                 "bha": ["bha/raw.sqlite"], "rtv": ["rtv_cache/rtv_raw.sqlite"]}.get(provider, [])
        for relative in paths:
            path = self.root/relative
            if not path.exists():
                continue
            with connection(path) as con:
                tables = {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
                if "payloads" not in tables:
                    continue
                row = con.execute("SELECT body FROM payloads WHERE sha256=?", (sha,)).fetchone()
                if row:
                    body = zlib.decompress(row[0])
                    if hashlib.sha256(body).hexdigest() != sha:
                        raise ValueError("Retained evidence failed its content hash")
                    if body.startswith(b"%PDF"):
                        return body, "application/pdf"
                    if len(body) > 20000000:
                        raise ValueError("Retained response exceeds the 20 MB browser download limit")
                    return body, "text/plain; charset=utf-8"
        raise FileNotFoundError("Raw evidence is not in this provider's retained payload store")

    def bha_fixtures(self, start, end, layer="native", dataset=None):
        date_range(start, end, 32)
        if layer not in {"native", "accepted"}:
            raise ValueError("Invalid BHA data layer")
        tables = {}
        if layer == "native":
            first, last = date.fromisoformat(start), date.fromisoformat(end)
            for i in range((last-first).days+1):
                for name, frame in self.bha_native((first+timedelta(days=i)).isoformat()).items():
                    tables.setdefault(name, []).append(frame)
            tables = {name: pd.concat(parts, ignore_index=True) for name, parts in tables.items()}
        else:
            receipt = self.manifest(dataset).get("sources", {}).get("bha", {})
            for name, entry in receipt.get("tables", {}).items():
                if name.startswith(("observations_", "links_")) or "race_date" not in (entry.get("schema") or {}):
                    continue
                path = self.table_path(receipt, name)
                schema = pq.read_schema(path)
                tables[name] = read_parquet(path, [("race_date", ">=", typed_value(schema, "race_date", start)),
                                                  ("race_date", "<=", typed_value(schema, "race_date", end))]).head(500)
        return {"source": "bha", "layer": layer, "message": "Fixture reports and source-native race notes. These records do not establish canonical runner attribution.",
                "tables": {name: {"rows": records(frame.head(500)), "count": len(frame), "truncated": len(frame)>500} for name, frame in tables.items()}}

    def coverage(self, start, end, dataset=None):
        first, last = date_range(start, end)
        manifest = self.manifest(dataset)
        days = [(first+timedelta(days=i)).isoformat() for i in range((last-first).days+1)]
        canonical = {}
        if self.db.exists():
            with connection(self.db) as con:
                canonical = {r[0]: {"races": r[1], "runners": r[2]} for r in con.execute(
                    "SELECT race_date,count(DISTINCT race_id),count(*) FROM results WHERE race_date BETWEEN ? AND ? GROUP BY race_date", (start, end))}
        primary = {"sl": "results", "atr": "sectionals", "rtv": "runs", "betfair": "bsp", "bha": "reports", "weather": "races"}
        output = []
        today = date.today()
        for provider, label in SOURCES.items():
            counts, grain, error = {}, "reference", None
            receipt = manifest.get("sources", {}).get(provider, {})
            table = primary.get(provider)
            if table and table in receipt.get("tables", {}):
                entry = receipt["tables"][table]
                grain = entry.get("contract", {}).get("grain", "observation")
                path = self.table_path(receipt, table)
                try:
                    schema = pq.read_schema(path)
                    if "race_date" in schema.names:
                        frame = read_parquet(path, [("race_date", ">=", typed_value(schema, "race_date", start)),
                                                    ("race_date", "<=", typed_value(schema, "race_date", end))], ["race_date"])
                        counts = pd.to_datetime(frame.race_date).dt.strftime("%Y-%m-%d").value_counts().to_dict()
                except (OSError, ValueError) as exc:
                    error = str(exc)
            lag = POLICIES.get(provider, {}).get("publication_lag_days", 0)
            cells = []
            for day in days:
                state = ("unavailable" if error else "reference" if table is None else "observed" if counts.get(day, 0) else
                         "awaiting_publication" if date.fromisoformat(day) > today-timedelta(days=lag) else
                         "no_canonical_races" if not canonical.get(day) else "unverified_gap")
                cells.append({"day": day, "count": int(counts.get(day, 0)), "state": state,
                              "canonical": canonical.get(day, {"races": 0, "runners": 0})})
            output.append({"source": provider, "label": label, "grain": grain, "cells": cells,
                           "error": error, "version": receipt.get("version")})
        return {"days": days, "sources": output, "dataset_id": manifest.get("id"),
                "message": "Counts are accepted observations at each source's own grain, not coverage percentages. A populated day may still be partial or contain rejected links. Reference sources have no race-day denominator."}

    def issues(self, start, end):
        date_range(start, end, 32)
        output, scanned = [], 0
        if self.db.exists():
            with connection(self.db) as con:
                selected = [r[0] for r in con.execute("SELECT DISTINCT race_id FROM results WHERE race_date BETWEEN ? AND ? ORDER BY race_date DESC LIMIT 1000", (start, end))]
                for race_id in selected:
                    frame = pd.read_sql_query("SELECT * FROM results WHERE race_id=?", con, params=(race_id,))
                    scanned += 1
                    for issue in self.validate_race(frame):
                        output.append({**issue, "source": "sl", "race_id": race_id,
                                       "race_date": str(frame.race_date.iloc[0]), "track": str(frame.track.iloc[0])})
                    if len(output) >= 100:
                        break
        queue = self.root/"atr_cache/rebuild_v2/rebuild.sqlite"
        if queue.exists():
            with connection(queue) as con:
                for row in con.execute("SELECT race_id,race_date,track,state,error,native_id FROM jobs WHERE race_date BETWEEN ? AND ? AND (state IN ('review','retry') OR native_id IS NULL) ORDER BY race_date DESC LIMIT 100", (start, end)):
                    item = dict(row)
                    output.append({"source": "atr", "race_id": item["race_id"], "race_date": item["race_date"], "track": item["track"],
                                   "severity": "warning", "code": "missing_native_id" if not item["native_id"] else item["state"],
                                   "message": item["error"] or "No verified ATR native ID; collection is waiting for discovery"})
        for issue in output:
            issue["id"] = fingerprint({k: issue[k] for k in ("source", "race_id", "code")})[:24]
        return {"rows": output, "scanned_races": scanned, "limit": 1000,
                "message": "Bounded checks of up to 1,000 canonical races plus 100 ATR queue issues in the chosen range. Findings remain reviewable even when acknowledged."}

    def compare(self, race_id, provider, before, after):
        first = self.race_source(race_id, provider, "accepted", before)
        last = self.race_source(race_id, provider, "accepted", after)
        changes = []
        for table in sorted(set(first["tables"]) | set(last["tables"])):
            old = first["tables"].get(table, {}).get("rows", [])
            new = last["tables"].get(table, {}).get("rows", [])
            def key(row):
                identity = [c for c in ("sl_horse_id", "horse_id", "horse_name", "report_key", "race_number", "observed_at") if row.get(c) is not None]
                return json.dumps({c: row[c] for c in identity}, sort_keys=True, default=str) if identity else "single"
            previous, current = {key(row): row for row in old}, {key(row): row for row in new}
            for identity in sorted(set(previous) | set(current)):
                left, right = previous.get(identity), current.get(identity)
                if left != right:
                    fields = [c for c in set(left or {}) | set(right or {}) if (left or {}).get(c) != (right or {}).get(c)]
                    changes.append({"table": table, "identity": identity, "kind": "added" if left is None else "removed" if right is None else "changed",
                                    "fields": sorted(fields), "before": left, "after": right})
        return {"before": before, "after": after, "source": provider, "changes": changes,
                "message": "Compares the selected race's accepted source rows, including saved link and timing receipts."}
