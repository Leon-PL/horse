"""Small storage primitives shared by collectors; no feature transformations."""
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import time
from uuid import uuid4
import zlib


def read_json(path):
    path = Path(path)
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _replace_after_readers(temporary, path):
    """Windows readers can briefly prevent an otherwise atomic replacement."""
    for attempt in range(8):
        try:
            os.replace(temporary, path)
            return
        except PermissionError:
            if os.name != "nt" or attempt == 7:
                raise
            time.sleep(min(.05 * 2**attempt, .5))


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + uuid4().hex + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, default=str, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        _replace_after_readers(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_parquet(frame, path, *, schema=None):
    """Prepare a whole cache before replacing its accepted file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name+"."+uuid4().hex+".tmp")
    try:
        frame.to_parquet(temporary, index=False, schema=schema)
        import pyarrow.parquet as pq
        if pq.ParquetFile(temporary).metadata.num_rows != len(frame):
            raise ValueError("Source cache publication changed the row count")
        with temporary.open("r+b") as stream:
            os.fsync(stream.fileno())
        _replace_after_readers(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


@contextmanager
def writer_lock(path):
    """Nonblocking process lock; an OS lock is released even after a crash."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as stream:
        if os.name == "nt":
            import msvcrt
            if not os.fstat(stream.fileno()).st_size:
                stream.write(b"0")
                stream.flush()
            stream.seek(0)
            def lock():
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            def unlock():
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl
            def lock():
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            def unlock():
                fcntl.flock(stream, fcntl.LOCK_UN)
        try:
            lock()
        except OSError:
            yield False
            return
        try:
            yield True
        finally:
            unlock()


class RawPayloadStore:
    """Content-addressed source responses, preserving each distinct revision.

    One short-lived connection per operation supports parallel HTTP workers.
    Re-observing identical bytes updates last_seen without duplicating the body.
    """

    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connection() as con:
            con.execute("CREATE TABLE IF NOT EXISTS payloads (sha256 TEXT PRIMARY KEY, body BLOB NOT NULL)")
            con.execute("CREATE TABLE IF NOT EXISTS observations (url TEXT, status INTEGER, sha256 TEXT, "
                        "first_seen TEXT NOT NULL, last_seen TEXT NOT NULL, PRIMARY KEY(url,status,sha256))")

    @contextmanager
    def connection(self):
        con = sqlite3.connect(self.path, timeout=30)
        try:
            con.execute("PRAGMA journal_mode=WAL")
            with con:
                yield con
        finally:
            con.close()

    def put(self, url, status, body):
        digest = hashlib.sha256(body).hexdigest()
        now = datetime.now(timezone.utc).isoformat()
        with self.connection() as con:
            con.execute("INSERT OR IGNORE INTO payloads VALUES (?,?)", (digest, zlib.compress(body, 6)))
            con.execute("INSERT INTO observations VALUES (?,?,?,?,?) ON CONFLICT(url,status,sha256) "
                        "DO UPDATE SET last_seen=excluded.last_seen", (url, status, digest, now, now))
        return digest

    def latest(self, url):
        with self.connection() as con:
            row = con.execute("SELECT p.body FROM observations o JOIN payloads p USING(sha256) "
                              "WHERE o.url=? AND o.status=200 ORDER BY o.last_seen DESC LIMIT 1", (url,)).fetchone()
        return None if row is None else zlib.decompress(row[0])
