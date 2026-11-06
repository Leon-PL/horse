"""Consolidate existing downloads into verified, deduplicated raw payloads.

The normal Historical API collector can replay these payloads by market ID.
Archive paths never determine race dates, countries or outcomes.
"""
import argparse
from contextlib import closing
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import sqlite3
import tarfile

import config
from src.ingestion.storage import atomic_json, writer_lock

MARKET = re.compile(r"1\.\d+\.bz2$")


class _CheckedMember(tarfile.TarInfo):
    @classmethod
    def frombuf(cls, buf, encoding, errors):
        try:
            return super().frombuf(buf, encoding, errors)
        except tarfile.InvalidHeaderError as error:
            # Permit the known interleaved zero padding, but never silently skip
            # damaged nonzero headers and then remove their original archive.
            raise tarfile.ReadError("Corrupt archive header") from error


def store_path(root):
    return Path(root) / "archive_payloads.sqlite"


def payloads(root, market_id):
    """Yield retained revisions; each must still pass the requested API filters."""
    path = store_path(root)
    if not path.exists():
        return
    with closing(sqlite3.connect(path.resolve().as_uri()+"?mode=ro", uri=True, timeout=30)) as con:
        rows = con.execute(
            "SELECT p.sha256,p.body FROM payloads p JOIN members m ON m.sha256=p.sha256 "
            "WHERE m.market_id=? AND p.verified=1 GROUP BY p.sha256 ORDER BY MAX(m.modified) DESC,p.sha256",
            (market_id,),
        ).fetchall()
    yield from rows


def _digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def market_ids(root):
    path = store_path(root)
    if not path.exists():
        return set()
    with closing(sqlite3.connect(path.resolve().as_uri()+"?mode=ro", uri=True, timeout=30)) as con:
        return {row[0] for row in con.execute("SELECT DISTINCT market_id FROM members WHERE market_id IS NOT NULL")}


def _import_members(con, source, known, known_payloads, progress):
    count = 0
    with tarfile.open(source, "r|", bufsize=1024*1024, ignore_zeros=True, tarinfo=_CheckedMember) as archive:
        for ordinal, member in enumerate(archive):
            if member.isdir():
                continue
            if not member.isfile():
                raise ValueError(f"Unsupported archive member retained for review: {source.name}")
            count += 1
            if ordinal in known:
                if known[ordinal] != member.name:
                    raise ValueError("Archive member ordering changed")
                continue
            with archive.extractfile(member) as stream:
                body = stream.read()
            if len(body) != member.size:
                raise ValueError("An archive member is incomplete")
            sha = hashlib.sha256(body).hexdigest()
            if sha not in known_payloads:
                con.execute("INSERT INTO payloads VALUES (?,?,?,1)", (sha, body, len(body)))
                known_payloads.add(sha)
            name = PurePosixPath(member.name).name
            market_id = name.removesuffix(".bz2") if MARKET.fullmatch(name) else None
            con.execute("INSERT INTO members VALUES (?,?,?,?,?,?)",
                        (source.name, ordinal, member.name, sha, market_id, member.mtime))
            if count % 250 == 0:
                con.commit()
                if progress:
                    progress({"archive": source.name, "members": count, "state": "importing"})
    return count


def _consolidate(directory, *, root=None, remove_verified=False, progress=None):
    """Retain every regular member, including event files and older revisions.

    Originals are removed individually only after payload read-back verification,
    a complete member manifest, and a second matching hash of the source archive.
    Interrupted imports resume from their saved member ordinals.
    """
    directory = Path(directory).resolve()
    root = Path(root or directory / "history").resolve()
    root.mkdir(parents=True, exist_ok=True)
    report = {"state": "running", "archives": [], "removed_bytes": 0}
    with closing(sqlite3.connect(store_path(root), timeout=60)) as con:
        con.execute("PRAGMA journal_mode=WAL")
        # Originals cover interrupted batches. A FULL commit and completed checkpoint
        # make the entire retained archive durable before its original is removed.
        con.execute("PRAGMA synchronous=NORMAL")
        con.execute("PRAGMA cache_size=-262144")
        con.execute("PRAGMA wal_autocheckpoint=32768")
        con.executescript(
            "CREATE TABLE IF NOT EXISTS payloads (sha256 TEXT PRIMARY KEY,body BLOB NOT NULL,"
            "bytes INTEGER NOT NULL,verified INTEGER NOT NULL);"
            "CREATE TABLE IF NOT EXISTS archives (name TEXT PRIMARY KEY,sha256 TEXT NOT NULL,"
            "bytes INTEGER NOT NULL,state TEXT NOT NULL,members INTEGER NOT NULL DEFAULT 0);"
            "CREATE TABLE IF NOT EXISTS members (archive TEXT NOT NULL,ordinal INTEGER NOT NULL,"
            "name TEXT NOT NULL,sha256 TEXT NOT NULL,market_id TEXT,modified REAL,"
            "PRIMARY KEY(archive,ordinal));"
            "CREATE INDEX IF NOT EXISTS member_market ON members(market_id);"
            "CREATE TABLE IF NOT EXISTS archive_issues (name TEXT PRIMARY KEY,problem TEXT,retained_path TEXT);"
        )
        known_payloads = {row[0] for row in con.execute("SELECT sha256 FROM payloads")}
        for source in sorted(directory.glob("*.tar")):
            if source.is_symlink() or source.resolve().parent != directory:
                raise ValueError("Only regular archives directly inside the selected directory can be consolidated")
            digest = _digest(source)
            size = source.stat().st_size
            previous = con.execute("SELECT sha256 FROM archives WHERE name=?", (source.name,)).fetchone()
            if previous and previous[0] != digest:
                raise ValueError(f"Archive changed since its previous import: {source.name}")
            with con:
                con.execute("INSERT OR IGNORE INTO archives(name,sha256,bytes,state) VALUES (?,?,?,'importing')",
                            (source.name, digest, size))
            known = dict(con.execute("SELECT ordinal,name FROM members WHERE archive=?", (source.name,)))
            if progress:
                progress({"archive": source.name, "members": len(known), "state": "importing"})
            count = 0
            try:
                count = _import_members(con, source, known, known_payloads, progress)
            except tarfile.ReadError:
                con.commit()
                if _digest(source) != digest:
                    raise ValueError("Incomplete archive changed during import")
                retained = con.execute("SELECT COUNT(*) FROM members WHERE archive=?", (source.name,)).fetchone()[0]
                retained_path = str(source)
                if remove_verified:
                    quarantine = root / "quarantine"
                    quarantine.mkdir(parents=True, exist_ok=True)
                    target = quarantine / source.name
                    target.resolve().relative_to(root)
                    if source.resolve().parent != directory or source.is_symlink():
                        raise ValueError("Archive recovery target changed")
                    if target.exists():
                        if _digest(target) != digest:
                            raise ValueError("A different archive occupies the recovery location")
                        source.unlink()
                    else:
                        source.replace(target)
                    if _digest(target) != digest:
                        raise ValueError("Recovery archive hash mismatch")
                    retained_path = target.relative_to(root).as_posix()
                con.execute("PRAGMA synchronous=FULL")
                with con:
                    con.execute("UPDATE archives SET state='truncated',members=? WHERE name=?",
                                (retained, source.name))
                    con.execute("INSERT OR REPLACE INTO archive_issues VALUES (?,?,?)",
                                (source.name, "Incomplete tar member; complete retained markets remain reusable", retained_path))
                con.execute("PRAGMA synchronous=NORMAL")
                issue = {"archive": source.name, "state": "quarantined", "retained_path": retained_path}
                report.setdefault("issues", []).append(issue)
                atomic_json(root / "archive_consolidation.json", report)
                if progress:
                    progress(issue)
                continue
            con.commit()
            retained, bad = con.execute(
                "SELECT COUNT(*),SUM(CASE WHEN p.verified=1 THEN 0 ELSE 1 END) "
                "FROM members m LEFT JOIN payloads p ON p.sha256=m.sha256 WHERE m.archive=?",
                (source.name,),
            ).fetchone()
            if retained != count or bad:
                raise ValueError("Archive preservation did not account for every member")
            for sha, body in con.execute("SELECT sha256,body FROM payloads WHERE sha256 IN "
                                         "(SELECT sha256 FROM members WHERE archive=?)", (source.name,)):
                if hashlib.sha256(body).hexdigest() != sha:
                    raise ValueError("Retained archive payload failed its final hash check")
            if _digest(source) != digest or source.stat().st_size != size:
                raise ValueError("Source archive changed while it was being retained")
            con.execute("PRAGMA synchronous=FULL")
            with con:
                con.execute("UPDATE archives SET state='verified',members=? WHERE name=?", (count, source.name))
            checkpoint = con.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()
            if checkpoint[0]:
                raise ValueError("Archive checkpoint is busy; the original is retained")
            entry = {"archive": source.name, "sha256": digest, "bytes": size, "members": count, "removed": False}
            if remove_verified:
                # Recheck the absolute parent immediately before this nonrecursive removal.
                if source.resolve().parent != directory or source.is_symlink():
                    raise ValueError("Archive removal target changed")
                source.unlink()
                entry["removed"] = True
                report["removed_bytes"] += size
            report["archives"].append(entry)
            atomic_json(root / "archive_consolidation.json", report)
            if progress:
                progress(entry)
            con.execute("PRAGMA synchronous=NORMAL")
        unique, stored_bytes = con.execute("SELECT COUNT(*),COALESCE(SUM(bytes),0) FROM payloads").fetchone()
        report["archives"] = [
            {"archive": name, "sha256": sha, "bytes": size, "members": count,
             "removed": not (directory / name).exists()}
            for name, sha, size, count in con.execute(
                "SELECT name,sha256,bytes,members FROM archives WHERE state='verified' ORDER BY name")
        ]
        report["removed_bytes"] = sum(item["bytes"] for item in report["archives"] if item["removed"])
        report["issues"] = [{"archive": name, "problem": problem, "retained_path": retained_path}
                            for name, problem, retained_path in con.execute("SELECT * FROM archive_issues ORDER BY name")]
        report.update(state="ok_with_issues" if report["issues"] else "ok",
                      unique_payloads=unique, retained_payload_bytes=stored_bytes)
    atomic_json(root / "archive_consolidation.json", report)
    return report


def consolidate(directory, *, root=None, remove_verified=False, progress=None):
    root = Path(root or Path(directory) / "history")
    with writer_lock(root / ".archive.lock") as acquired:
        if not acquired:
            return {"state": "busy", "reason": "Archive consolidation is already running"}
        return _consolidate(directory, root=root, remove_verified=remove_verified, progress=progress)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", default=str(Path(config.DATA_DIR) / "betfair"))
    parser.add_argument("--remove-verified", action="store_true")
    args = parser.parse_args()
    root = Path(args.directory) / "history"
    def progress(value):
        atomic_json(root / "archive_import_status.json", {
            "state": "running", "pid": os.getpid(), "checked_at": datetime.now(timezone.utc).isoformat(), **value})
    progress({"state": "starting"})
    try:
        report = consolidate(args.directory, remove_verified=args.remove_verified, progress=progress)
    except (OSError, ValueError, sqlite3.Error, tarfile.TarError):
        report = {"state": "error", "message": "Archive consolidation stopped; unverified originals are retained."}
        atomic_json(root / "archive_import_status.json", report)
        raise
    atomic_json(root / "archive_import_status.json", report)
    print(json.dumps(report))


if __name__ == "__main__":
    main()
