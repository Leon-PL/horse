"""Atomic source-table key migrations; no changes to accepted result values."""
from datetime import datetime, timezone
import re

from src.runner_identity import MISSING_IDS


def rebuild_without_key(conn, table, old_columns):
    """Remove one legacy table constraint, preserving rowids, values and SQL."""
    indexes = conn.execute(f"PRAGMA index_list({table})").fetchall()
    legacy = [index for index in indexes if index[2] and not index[4] and
              [row[2] for row in conn.execute(f'PRAGMA index_info("{index[1]}")')] == old_columns]
    if not legacy:
        return False
    sql = conn.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name=?", (table,)).fetchone()[0]
    key = r",\s*(?:UNIQUE|PRIMARY\s+KEY)\s*\(\s*" + r"\s*,\s*".join(map(re.escape, old_columns)) + r"\s*\)"
    sql, removed = re.subn(key, "", sql, flags=re.IGNORECASE)
    if removed != 1:
        raise ValueError(f"Cannot safely migrate legacy identity constraint in {table}")
    temporary = table+"_identity_migration"
    sql, renamed = re.subn(r"^(CREATE\s+TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?)(?:\""+table+r"\"|"+table+r")(?=\s*\()",
                           lambda match: match.group(1)+temporary, sql, count=1, flags=re.IGNORECASE)
    if renamed != 1:
        raise ValueError(f"Cannot identify source table {table}")
    auxiliary = conn.execute("SELECT type,name,sql FROM sqlite_master WHERE tbl_name=? "
                             "AND type IN ('index','trigger') AND sql IS NOT NULL", (table,)).fetchall()
    old_names = {index[1] for index in legacy}
    count = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
    columns = [row[1] for row in conn.execute(f"PRAGMA table_info({table})")]
    projection = ",".join('"'+name.replace('"','""')+'"' for name in columns)
    conn.execute(sql)
    conn.execute(f"INSERT INTO {temporary}(rowid,{projection}) SELECT rowid,{projection} FROM {table} ORDER BY rowid")
    if conn.execute(f"SELECT COUNT(*) FROM {temporary}").fetchone()[0] != count:
        raise ValueError("Identity migration changed the row population")
    conn.execute(f"DROP TABLE {table}")
    conn.execute(f"ALTER TABLE {temporary} RENAME TO {table}")
    for _, name, statement in auxiliary:
        if name not in old_names:
            conn.execute(statement)
    conn.execute("CREATE TABLE IF NOT EXISTS source_schema_migrations (name TEXT PRIMARY KEY, "
                 "applied_at TEXT, preserved_rows INTEGER)")
    conn.execute("INSERT OR IGNORE INTO source_schema_migrations VALUES (?,?,?)",
                 (table+"_source_horse_identity_v1", datetime.now(timezone.utc).isoformat(), count))
    return True


def ensure_source_keys(conn, table, tail=()):
    # The removed name constraint also indexed race_id. Partial identity
    # indexes cannot serve a lookup spanning both known and missing IDs.
    conn.execute(f"CREATE INDEX IF NOT EXISTS {table}_race_lookup ON {table}(race_id)")
    invalid = "("+",".join("'"+value+"'" for value in MISSING_IDS)+")"
    valid = f"horse_id IS NOT NULL AND lower(trim(CAST(horse_id AS TEXT))) NOT IN {invalid}"
    end = ","+",".join(tail) if tail else ""
    conn.execute(f"CREATE UNIQUE INDEX IF NOT EXISTS {table}_horse_identity ON {table}(race_id,horse_id{end}) WHERE {valid}")
    conn.execute(f"CREATE UNIQUE INDEX IF NOT EXISTS {table}_legacy_identity ON {table}(race_id,horse_name{end}) WHERE NOT ({valid})")


def result_rowid(conn, record):
    """Resolve a repair target by ID; name-only targets must be unambiguous."""
    import pandas as pd
    from src.runner_identity import runner_keys, horse_ids
    frame = pd.read_sql_query("SELECT rowid AS _rowid,horse_id,horse_name FROM results WHERE race_id=?", conn,
                              params=(str(record["race_id"]),))
    requested = pd.DataFrame([record])
    if horse_ids(requested).notna().all():
        match = runner_keys(frame).eq(runner_keys(requested).iloc[0])
    else:
        from src.utils import normalise_horse_key
        match = frame.horse_name.map(normalise_horse_key).eq(normalise_horse_key(record["horse_name"]))
    if match.sum() != 1:
        raise ValueError("Repair target is missing or has an ambiguous runner identity")
    return int(frame.loc[match, "_rowid"].iloc[0])


def adopt_stored_ids(frame, reference):
    """Legacy name-only writes may retain a known ID, never choose a namesake."""
    import pandas as pd
    from src.runner_identity import horse_ids, runner_positions
    from src.utils import normalise_horse_key
    ids = horse_ids(frame)
    if ids.notna().all() or reference.empty:
        return frame
    positions = runner_positions(frame, reference, require_all=False)
    ambiguous = ids.isna().to_numpy() & (positions < 0) & frame.horse_name.map(normalise_horse_key).isin(
        reference.horse_name.map(normalise_horse_key)).to_numpy()
    if ambiguous.any():
        raise ValueError("Ambiguous legacy runner name requires a source horse ID")
    known = horse_ids(reference).reset_index(drop=True).reindex(positions).reset_index(drop=True)
    known.index = frame.index
    out = frame.copy()
    out["horse_id"] = ids.fillna(known).astype(object).where(lambda series: pd.notna(series), None)
    return out


def ensure_repair_keys(conn, table, original_column, replacement_column):
    """Preserve original receipt JSON while replacing its display-name key."""
    import json
    import pandas as pd
    from src.runner_identity import runner_keys
    columns = {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}
    if "runner_key" not in columns:
        rebuild_without_key(conn, table, ["repair_id", "race_id", "horse_name"])
        conn.execute(f"ALTER TABLE {table} ADD COLUMN runner_key TEXT")
        rows = conn.execute(f"SELECT rowid,horse_name,{original_column},{replacement_column} FROM {table}").fetchall()
        updates = []
        for rowid, name, original, replacement in rows:
            value = json.loads(original or replacement) if original or replacement else {"horse_name": name}
            value.setdefault("horse_name", name)
            updates.append((runner_keys(pd.DataFrame([value])).iloc[0], rowid))
        conn.executemany(f"UPDATE {table} SET runner_key=? WHERE rowid=?", updates)
    conn.execute(f"CREATE UNIQUE INDEX IF NOT EXISTS {table}_runner_identity ON {table}(repair_id,race_id,runner_key)")
