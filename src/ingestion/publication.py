"""Publish native source stores through the common accepted-data contract.

No network requests or collector edits. ATR promotion reads a SQLite snapshot
while the worker continues, and never replaces the worker or legacy cache.
"""
from __future__ import annotations

import json
import logging
import hashlib
from pathlib import Path
import sqlite3

import pandas as pd

import config
from src.source_contracts import TableContract, coverage_report, observation_metadata, resolve_runners
from src.source_dataset import publish_source, registry_root
from src.ingestion.storage import read_json

SIDECAR = TableContract("source row", ("source_row",), ("source_row",), role="quality")
logger = logging.getLogger(__name__)


def _adopt(provider, operation, *, required=False, canonical=None):
    """A provider's bad candidate cannot discard its last accepted version."""
    from src.ingestion.storage import atomic_json
    from src.source_dataset import verify_source
    previous = read_json(registry_root()/provider/"latest.json")
    try:
        receipt = operation()
        attempt = {"state": "ok", "version": receipt["version"]}
    except Exception as exc:
        logger.warning("Source publication %s failed: %s", provider, exc)
        attempt = {"state": "error", "error": str(exc), "retained_version": previous.get("version")}
        receipt = previous or None
        if receipt:
            verify_source(receipt)
            if canonical is not None:
                # Retain valid observations, but requalify their crosswalks if
                # canonical fields changed (including newly restored namesakes).
                tables, contracts = {}, {}
                for name, entry in receipt["tables"].items():
                    if name.startswith(("links_", "observations_")) or entry.get("schema") is None:
                        continue
                    tables[name] = pd.read_parquet(registry_root()/provider/receipt["version"]/entry["file"])
                    contracts[name] = TableContract(**entry["contract"])
                eligible = len(canonical)
                if provider == "atr":
                    from src.ingestion import attheraces as atr
                    eligible = int((~canonical.track.map(atr._normalise_track_key).isin(atr._NO_SECTIONALS_TRACKS)).sum())
                receipt = _publish(provider, tables, contracts, canonical, eligible=eligible,
                                   provenance={"retained_from": previous["version"], "candidate_error": str(exc)})
    attempt["checked_at"] = pd.Timestamp.now(tz="UTC").isoformat()
    atomic_json(registry_root()/provider/"publication_status.json", attempt)
    if required and not receipt:
        raise RuntimeError(f"Required source {provider} has no accepted version: {attempt.get('error')}")
    return receipt


def _publish(provider, tables, contracts, canonical, *, provenance=None, eligible=None):
    # Rechecks of identical observations reuse the accepted generation without
    # rebuilding large crosswalks and timing tables. Canonical identity changes
    # invalidate this cache even when native observations have not changed.
    def fingerprint(frame):
        digest = hashlib.sha256(str([(c, str(t)) for c, t in frame.dtypes.items()]).encode())
        digest.update(pd.util.hash_pandas_object(frame, index=False).values.tobytes())
        return digest.hexdigest()
    provenance = {**(provenance or {}), "input_fingerprints": {name: fingerprint(frame) for name, frame in tables.items()},
                  "identity_reference": fingerprint(canonical[[c for c in ("race_id", "horse_id", "horse_name", "race_date", "track", "off_time") if c in canonical]]),
                  "contracts": {name: json.loads(json.dumps(contract.__dict__)) for name, contract in contracts.items()}}
    provenance["acceptance_policy"] = hashlib.sha256(
        Path(__file__).read_bytes() + (Path(__file__).parents[1]/"source_contracts.py").read_bytes()).hexdigest()
    previous = read_json(registry_root()/provider/"latest.json")
    if previous and previous.get("provenance") == provenance:
        from src.source_dataset import verify_source
        verify_source(previous)
        return previous
    tables = dict(tables)
    contracts = dict(contracts)
    reports = {}
    for name, frame in list(tables.items()):
        contract = contracts[name]
        links = None
        if contract.grain in {"runner", "runner snapshot"}:
            kwargs = {}
            if provider == "betfair":
                from src.betfair import bf_horse_key, bf_track_key
                kwargs = {"horse_key": bf_horse_key, "track_key": bf_track_key}
            links = resolve_runners(frame, canonical, provider=provider, **kwargs)
            tables["links_" + name] = links
            contracts["links_" + name] = SIDECAR
        metadata = observation_metadata(frame, contract, links)
        tables["observations_" + name] = metadata
        contracts["observations_" + name] = SIDECAR
        reports[name] = coverage_report(frame, links, eligible=eligible if contract.grain in {"runner", "runner snapshot", "canonical runner"} else None,
                                       total=len(canonical) if contract.grain in {"runner", "runner snapshot", "canonical runner"} else None, metadata=metadata)
    return publish_source(provider, tables, contracts=contracts, provenance=provenance, reports=reports)


def canonical_snapshot():
    """Results and pedigree from one database read transaction."""
    from src import database
    with database._get_conn() as con:
        con.execute("BEGIN")
        results = pd.read_sql_query("SELECT * FROM results ORDER BY race_date", con).drop(columns="scraped_at", errors="ignore")
        tables = {row[0] for row in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        pedigree = (pd.read_sql_query("SELECT horse_id,sire,dam,damsire,foaled FROM horse_pedigree", con)
                    if "horse_pedigree" in tables else pd.DataFrame(columns=["horse_id", "sire", "dam", "damsire", "foaled"]))
        revision = con.execute("SELECT database_id,revision FROM result_source_state WHERE id=1").fetchone()
    return database.results_snapshot_frame(results), pedigree, {"database_id": revision[0], "revision": revision[1]}


def _atr_rows(canonical, rebuild_root=None):
    from src.ingestion import attheraces as atr
    from src.ingestion.atr_tabs import COLLECTION_VERSION, PERFORMANCE_TABS, TERMINAL_TAB_STATES
    path = Path(atr.ATR_CACHE_FILE)
    legacy = pd.read_parquet(path) if path.exists() else pd.DataFrame()
    root = Path(rebuild_root or Path(config.DATA_DIR) / "atr_cache/rebuild_v2")
    queue = root / "rebuild.sqlite"
    rows, rejected, units = [], [], []
    if queue.exists():
        con = sqlite3.connect(queue.resolve().as_uri() + "?mode=ro", uri=True, timeout=30)
        con.row_factory = sqlite3.Row
        try:
            con.execute("BEGIN")
            jobs = [dict(row) for row in con.execute("SELECT * FROM jobs WHERE state IN ('data','data_partial') ORDER BY race_date,race_id")]
            receipts = {}
            for row in con.execute("SELECT race_id,kind,state,receipt FROM tab_coverage"):
                receipts.setdefault(row[0], {})[row[1]] = (row[2], json.loads(row[3] or "{}"))
            payloads = {}
            for race_id, payload in con.execute("SELECT race_id,payload FROM runners"):
                payloads.setdefault(race_id, []).append(json.loads(payload))
        finally:
            con.close()
        for job in jobs:
            race_id = job["race_id"]
            try:
                tabs = receipts.get(race_id, {})
                if set(tabs) != set(PERFORMANCE_TABS) or any(state not in TERMINAL_TAB_STATES for state, _ in tabs.values()):
                    raise ValueError("incomplete_tab_contract")
                current = payloads.get(race_id, [])
                if not current:
                    raise ValueError("missing_runner_payloads")
                if any(str(row.get("atr_collection_version")) != str(COLLECTION_VERSION) for row in current):
                    raise ValueError("wrong_collection_version")
                if any(str(row.get("sl_race_id")) != str(race_id) for row in current):
                    raise ValueError("wrong_canonical_race")
                native_keys = [(row.get("atr_raceid"), row.get("atr_horse_id")) for row in current]
                if any(not race or not horse for race, horse in native_keys) or len(set(native_keys)) != len(native_keys):
                    raise ValueError("duplicate_or_missing_native_identity")
                for row in current:
                    row["collected_at"] = job["updated_at"]
                rows.extend(current)
                units.append({"race_id": str(race_id), "race_date": job["race_date"], "rows": len(current), "state": job["state"]})
            except (ValueError, KeyError, AttributeError) as exc:
                rejected.append({"race_id": str(race_id), "reason": str(exc)})
    fresh = atr._encode_arrays(pd.DataFrame(rows)) if rows else pd.DataFrame()
    if not fresh.empty:
        links = resolve_runners(fresh, canonical, provider="atr")
        bad_identity = ~links.match_status.eq("matched").to_numpy()
        bad_identity |= fresh.duplicated(["atr_raceid", "atr_horse_id"], keep=False).to_numpy()
        awaiting = ~atr._publication_ready(fresh).to_numpy()
        bad_races = set(fresh.loc[bad_identity | awaiting, "sl_race_id"].astype(str))
        identity_races = set(fresh.loc[bad_identity, "sl_race_id"].astype(str))
        rejected.extend({"race_id": race, "reason": "canonical_identity_changed" if race in identity_races else "awaiting_publication"}
                        for race in sorted(bad_races))
        fresh = fresh.loc[~fresh.sl_race_id.astype(str).isin(bad_races)].reset_index(drop=True)
        units = [unit for unit in units if unit["race_id"] not in bad_races]
    # Keep earlier promotions if an upgrade has reopened a worker job.
    previous = read_json(registry_root()/"atr/latest.json")
    if previous and "sectionals" in previous["tables"]:
        previous_path = registry_root()/"atr"/previous["version"]/previous["tables"]["sectionals"]["file"]
        accepted = pd.read_parquet(previous_path)
        old_links = resolve_runners(legacy, canonical, provider="atr")
        accepted_links = resolve_runners(accepted, canonical, provider="atr")
        old_keys = pd.MultiIndex.from_frame(old_links[["sl_race_id", "sl_horse_id"]])
        accepted_keys = pd.MultiIndex.from_frame(accepted_links[["sl_race_id", "sl_horse_id"]])
        legacy = pd.concat([accepted, legacy.loc[~old_keys.isin(accepted_keys)]], ignore_index=True, sort=False)
    # Replacement is by a proven runner identity, not by a whole day or race.
    # Unavailable new runners therefore cannot delete useful old observations.
    if not fresh.empty and not legacy.empty:
        old_links = resolve_runners(legacy, canonical, provider="atr")
        new_links = resolve_runners(fresh, canonical, provider="atr")
        old_keys = pd.MultiIndex.from_frame(old_links[["sl_race_id", "sl_horse_id"]])
        new_keys = pd.MultiIndex.from_frame(new_links[["sl_race_id", "sl_horse_id"]])
        legacy = legacy.loc[~old_keys.isin(new_keys)]
    combined = pd.concat([legacy, fresh], ignore_index=True, sort=False)
    if combined.empty:
        combined = pd.DataFrame(columns=["race_date", "track", "off_time", "horse_name"])
    return combined, units, rejected


def publish_atr(canonical, *, rebuild_root=None):
    from src.ingestion import attheraces as atr
    frame, units, rejected = _atr_rows(canonical, rebuild_root)
    eligible = int((~canonical.track.map(atr._normalise_track_key).isin(atr._NO_SECTIONALS_TRACKS)).sum())
    # A changed canonical snapshot must also refresh persisted crosswalks.
    provenance = {"publication_policy": "validated_runners_preserve_history", "promoted_units": units}
    receipt = _publish("atr", {"sectionals": frame},
                       {"sectionals": TableContract("runner", ("race_date", "track", "off_time", "horse_name"), delay_hours=48)},
                       canonical, provenance=provenance, eligible=eligible)
    # Rejections are durable receipts and never become a no-data decision.
    from src.ingestion.storage import atomic_json
    atomic_json(registry_root() / "atr" / "promotion_status.json", {
        "version": receipt["version"], "checked_at": pd.Timestamp.now(tz="UTC").isoformat(),
        "promoted_races": len(units), "rejected_units": rejected})
    return receipt


def capture_sources(*, include_prices=False, canonical_cards=None):
    """Adopt accepted native stores into one immutable consumer boundary."""
    canonical, pedigree, revision = canonical_snapshot()
    identity_reference = canonical
    if canonical_cards is not None:
        required = ["race_id", "horse_id", "horse_name", "race_date", "track", "off_time"]
        cards = canonical_cards[required]
        identity_reference = pd.concat([canonical[required], cards], ignore_index=True).drop_duplicates(["race_id", "horse_id"], keep="last")
    sources = {}
    from src.database import RESULTS_SNAPSHOT_COLUMNS
    sources["sl"] = _publish("sl", {"results": canonical},
        {"results": TableContract("canonical runner", tuple(RESULTS_SNAPSHOT_COLUMNS), ("race_id", "horse_id"))},
        canonical, provenance=revision, eligible=len(canonical))
    sources["pedigree"] = _publish("pedigree", {"horses": pedigree},
        {"horses": TableContract("horse", ("horse_id", "sire", "dam", "damsire"), ("horse_id",), role="reference")}, canonical)
    weather_columns = [c for c in canonical if c.startswith("weather_")]
    weather = canonical[["race_id", "race_date", *weather_columns]].drop_duplicates("race_id")
    sources["weather"] = _publish("weather", {"races": weather},
        {"races": TableContract("race", ("race_id", "race_date"), ("race_id",), role="research")}, canonical)
    sources["atr"] = _adopt("atr", lambda: publish_atr(canonical), canonical=canonical)
    from src.rtv_features import runs_path, pre_path, cards_path
    from src.rtv_text import scores_path, text_path
    from src.ingestion.rtv_storage import pin_generation
    with pin_generation():
        paths = {"runs": runs_path(), "pre": pre_path(), "cards": cards_path(), "scores": scores_path(), "text": text_path()}
    def publish_rtv():
        tables, contracts = {}, {}
        for name, path in paths.items():
            if path.exists():
                tables[name] = pd.read_parquet(path)
                snapshot = name == "pre"
                required = ("race_date", "track_key", "off_key", "horse_key", "observed_at") if snapshot else ("race_id", "horse_id")
                contracts[name] = TableContract("runner snapshot" if snapshot else "runner", required,
                    role="snapshot" if snapshot else "history", delay_hours=48 if name == "runs" else 0)
        return _publish("rtv", tables, contracts, identity_reference, eligible=len(identity_reference))
    sources["rtv"] = _adopt("rtv", publish_rtv, canonical=identity_reference)
    base = Path(config.BASE_DIR) / "reference"
    def publish_fx():
        rates = pd.read_csv(base/"ecb_gbp_per_eur.csv", parse_dates=["observation_date"], float_precision="round_trip")
        if not rates.observation_date.is_monotonic_increasing or not pd.to_numeric(rates.gbp_per_eur, errors="raise").gt(0).all():
            raise ValueError("FX rates must be ordered and positive")
        mapping = read_json(base/"track_prize_currency.json")["currency_by_track"]
        units = pd.DataFrame({"track": list(mapping), "currency": list(mapping.values())})
        if not units.currency.isin(["GBP", "EUR"]).all():
            raise ValueError("Unrecognised prize currency")
        contracts = {
            "rate_values": TableContract("dated rate", ("observation_date", "gbp_per_eur"), ("observation_date",),
                role="reference", delay_hours=24, types=(("gbp_per_eur", "number"), ("observation_date", "timestamp")), event_column="observation_date"),
            "currency_units": TableContract("course currency", ("track", "currency"), ("track",), role="reference", types=(("track", "text"), ("currency", "text")))}
        tables = {"rate_values": rates, "currency_units": units}
        for name, frame in list(tables.items()):
            tables["observations_"+name] = observation_metadata(frame, contracts[name])
            contracts["observations_"+name] = SIDECAR
        return publish_source("fx", tables, contracts=contracts, artifacts={
            "units": base/"track_prize_currency.json", "rates": base/"ecb_gbp_per_eur.csv"})
    sources["fx"] = _adopt("fx", publish_fx, required=True)
    if include_prices:
        def publish_prices():
            tables, contracts = {}, {}
            for name in ("bsp", "bsp_early", "timing", "place_bsp"):
                path = Path(config.PROCESSED_DATA_DIR) / ("betfair_" + name + ".parquet")
                if path.exists():
                    tables[name] = pd.read_parquet(path)
                    contracts[name] = TableContract("runner", ("race_date", "track", "off_key", "horse_key"), role="evaluation", delay_hours=144)
            return _publish("betfair", tables, contracts, canonical, eligible=len(canonical))
        sources["betfair"] = _adopt("betfair", publish_prices, canonical=canonical)
    return {name: receipt for name, receipt in sources.items() if receipt is not None}


def source_health(attempt, receipt=None):
    """Collection state and accepted data readiness are independent dimensions."""
    coverage = json.loads(json.dumps(receipt.get("reports", {}))) if receipt else {}
    for report in coverage.values():
        latest = report.get("latest_usable_event")
        report["age_days"] = (pd.Timestamp.now(tz="UTC")-pd.Timestamp(latest)).total_seconds()/86400 if latest else None
    return {"attempt_state": attempt.get("state", "unknown"),
            "reason": attempt.get("reason") or attempt.get("error"),
            "available_through": attempt.get("available_through"),
            "publication_lag_days": attempt.get("publication_lag_days"),
            "accepted_version": receipt.get("version") if receipt else None,
            "published_at": receipt.get("published_at") if receipt else None,
            "coverage": coverage,
            "readiness": "accepted" if receipt else "unverified"}


def latest_source_health(provider, attempt):
    return source_health(attempt, read_json(registry_root() / provider / "latest.json") or None)


def publish_live_snapshots(cards, sources):
    """Publish the small pre-race stream against historical and upcoming IDs."""
    path = Path(getattr(config, "RTV_FEATURE_PRE", Path(config.DATA_DIR)/"rtv_cache/feature_pre.parquet"))
    if not path.exists():
        return None
    frame = pd.read_parquet(path)
    columns = ["race_id", "horse_id", "horse_name", "race_date", "track", "off_time"]
    sl = sources["sl"]
    results_path = registry_root()/"sl"/sl["version"]/sl["tables"]["results"]["file"]
    days = frame.race_date.astype(str).str[:10].unique().tolist()
    reference = pd.read_parquet(results_path, columns=columns, filters=[("race_date", "in", days)])
    reference = pd.concat([reference, cards[columns]], ignore_index=True).drop_duplicates(["race_id", "horse_id"], keep="last")
    contract = TableContract("runner snapshot", ("race_date", "track_key", "off_key", "horse_key", "observed_at"), role="snapshot")
    return _adopt("rtv_snapshots", lambda: _publish("rtv_snapshots", {"pre": frame}, {"pre": contract}, reference, eligible=len(reference)), canonical=reference)
