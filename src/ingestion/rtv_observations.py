"""Materialise numeric RTV observations; keep network/collection out of FE."""
from datetime import date, datetime, timezone
import logging
import os
from pathlib import Path
import sqlite3
import hashlib
from uuid import uuid4

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

import config
from src import rtv_features as features
from src.ingestion import racing_tv as rtv
from src import rtv_text
from src.ingestion.pipeline import match_runners, canonical_rows
from src.utils import canonical_track_key, normalise_horse_key
from src.wf_eval import purge_days
from src.ingestion.rtv_storage import accepted_days, partition_paths
from src.ingestion.storage import atomic_json, atomic_parquet, writer_lock

logger = logging.getLogger(__name__)


def _read_columns(path, columns):
    if not path.exists():
        return pd.DataFrame(columns=columns)
    schema = pq.read_schema(path).names
    return pd.read_parquet(path, columns=[c for c in columns if c in schema])


def observation_tables(start, end, *, source_root=None, names=None, manifest=None):
    root = Path(source_root or Path(config.DATA_DIR)/"rtv_cache")
    overrides = accepted_days(root, manifest=manifest)
    overrides = {d: v for d, v in overrides.items() if start <= d <= end}
    columns = {
        "races": ["race_id", "race_date", "track_name", "off_time_utc", "is_abandoned", "is_arab_race", "distance_furlongs",
            "distance_yards", "win_time_s", "going", "race_type", "surface", "card_analyst_verdict_text"],
        "runners": ["race_id", "runner_id", "horse_name", "is_finisher", "is_nonrunner", "in_results",
            "card_timeform_rating", "card_timeform_star_rating", "card_timeform_tipped_place", "card_focus_horse",
            "card_analyst_comment_text"],
        "metrics": ["race_id", "runner_id", "metric", "value"],
        "sectionals": ["race_id", "runner_id", "sector_number", "sector_time", "sector_length_yards", "sector_position",
            "cumulative_sector_time"],
        "quality_flags": ["race_id", "runner_id", "check", "severity"]}
    columns = {name: cols for name, cols in columns.items() if names is None or name in names or name == "races"}
    parts = {name: [] for name in columns}
    seen = set()
    for race_path in partition_paths(root/"tables/races", start, end):
        races = _read_columns(race_path, columns["races"])
        races = races.loc[races.race_date.between(start, end) & ~races.race_date.isin(overrides) & ~races.race_id.isin(seen)]
        seen.update(races.race_id)
        parts["races"].append(races)
        for name in columns.keys()-{"races"}:
            frame = _read_columns(root/"tables"/name/race_path.name, columns[name])
            if not frame.empty:
                parts[name].append(frame.loc[frame.race_id.isin(races.race_id)])
    for day, entry in overrides.items():
        for name, cols in columns.items():
            frame = _read_columns(root/"pipeline"/entry["directory"]/(name+".parquet"), cols)
            if not frame.empty:
                parts[name].append(frame)
    tables = {name: pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame(columns=columns[name])
              for name, chunks in parts.items()}
    if "sectionals" in tables:
        tables["sectionals"] = tables["sectionals"].rename(columns={"sector_position": "race_position"})
    return tables


def time_par(*, source_root=None):
    """Reuse the fixed par until its actual training facts or fitting code change."""
    import joblib
    races = observation_tables(features.PAR_START, features.PAR_END, source_root=source_root, names=("races",))["races"]
    digest = hashlib.sha256(Path(features.__file__).read_bytes())
    ordered = races.sort_values("race_id").reset_index(drop=True) if not races.empty else races
    digest.update(pd.util.hash_pandas_object(ordered.astype("string"), index=False).values.tobytes())
    directory = Path(source_root or Path(config.DATA_DIR)/"rtv_cache")/"derived_models"
    path = directory/("time_par_"+digest.hexdigest()+".joblib")
    if path.exists():
        return joblib.load(path)
    model = features.fit_time_par(races)
    directory.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name+"."+uuid4().hex+".tmp")
    try:
        joblib.dump(model, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return model


def _merge_dated(output, path, start, end, key, *, previous_path=None):
    previous_path = Path(previous_path or path)
    if previous_path.exists():
        previous = pd.read_parquet(previous_path)
        previous = previous.loc[~pd.to_datetime(previous.race_date).between(start, end)]
        output = pd.concat([p for p in (previous, output) if not p.empty] or [output], ignore_index=True)
    output = output.sort_values(["race_date", "race_id", "horse_id"], kind="stable").reset_index(drop=True)
    if output.duplicated(["race_id", "horse_id"]).any():
        raise ValueError(f"Duplicate materialised RTV {key}")
    atomic_parquet(output, path)
    return output


def refresh_run_observations(start, end, *, database=None, source_root=None, destination=None, cards_destination=None,
                              manifest=None, publish=True):
    options = dict(database=database, source_root=source_root, destination=destination,
                   cards_destination=cards_destination, manifest=manifest, publish=publish)
    if publish and destination is None:
        root = Path(source_root or Path(config.DATA_DIR)/"rtv_cache")
        with writer_lock(root/"pipeline/.writer.lock") as acquired:
            if not acquired:
                raise RuntimeError("Racing TV materialisation is already running")
            return _refresh_run_observations(start, end, **options)
    return _refresh_run_observations(start, end, **options)


def _refresh_run_observations(start, end, *, database=None, source_root=None, destination=None, cards_destination=None,
                              manifest=None, publish=True):
    tables = observation_tables(start, end, source_root=source_root, manifest=manifest)
    canonical = canonical_rows(start, end, db_path=database)
    links = match_runners(tables["races"], tables["runners"], canonical)
    output = features.build_run_observations(tables["races"], tables["runners"], tables["metrics"],
        tables["sectionals"], tables["quality_flags"], links, par=time_par(source_root=source_root))
    cards = features.build_card_observations(tables["runners"], links)
    root = Path(source_root or Path(config.DATA_DIR)/"rtv_cache")
    relative = Path("pipeline/materialised")/uuid4().hex
    directory = root/relative
    path = Path(destination) if destination is not None else directory/"runs.parquet"
    card_path = (Path(cards_destination) if cards_destination is not None else
                 Path(destination).with_name(Path(destination).stem+"_cards.parquet") if destination is not None else directory/"cards.parquet")
    output = _merge_dated(output, path, start, end, "run", previous_path=features.runs_path() if destination is None else path)
    cards = _merge_dated(cards, card_path, start, end, "card", previous_path=features.cards_path() if destination is None and cards_destination is None else card_path)
    receipt = {"runs": len(output), "horses": int(output.horse_id.nunique()), "cards": len(cards), "path": str(path)}
    if destination is None:
        # Card text, then comment scores for the window; a new quarter's model is fitted once.
        text = rtv_text.build_text_observations(tables["races"], tables["runners"], links)
        text_path = directory/"text.parquet"
        _merge_dated(text, text_path, start, end, "text", previous_path=rtv_text.text_path())
        ensure_text_models(max(pd.Timestamp(end), pd.Timestamp(date.today())), database=database, text_source=text_path)
        receipt.update(refresh_text_scores(start, end, database=database, text_source=text_path,
                                          destination=directory/"scores.parquet", previous_path=rtv_text.scores_path()))
        files = {name: (relative/(name+".parquet")).as_posix() for name in ("runs", "cards", "text", "scores")}
        receipt["files"] = files
        if publish:
            from src.ingestion.rtv_storage import generation, history_manifest
            value = generation(root)
            value.update(files=files, history=manifest if manifest is not None else history_manifest(root))
            atomic_json(root/"pipeline/generation.json", value)
    return receipt


# ------------------------------------------------------- comment scores --

def _starters(start, end, database=None):
    db_path = Path(database or Path(config.DATA_DIR)/"races.db").resolve()
    with sqlite3.connect(db_path.as_uri()+"?mode=ro", uri=True) as con:
        field = pd.read_sql_query("SELECT race_id, horse_id, horse_name, race_date, finish_position FROM results "
                                  "WHERE race_date BETWEEN ? AND ?", con, params=(start, end))
    field["race_id"], field["horse_id"] = field.race_id.astype(str), pd.to_numeric(field.horse_id, errors="coerce")
    return field.loc[~field.duplicated(["race_id", "horse_id"], keep=False)]


def _text_rows(start, end, database=None, *, text_source=None):
    """Every SL starter in [start, end] with its card text, prepared and sorted by date and race."""
    field = _starters(start, end, database)
    path = Path(text_source or rtv_text.text_path())
    columns = ["race_id", "horse_id", "tf_comment", "tf_verdict"]
    text = pd.read_parquet(path, filters=[("race_date", ">=", start), ("race_date", "<=", end)]) if path.exists() else pd.DataFrame(columns=columns+["race_date"])
    text = text.loc[text.race_date.between(start, end), columns]
    text = text.assign(race_id=text.race_id.astype(str), horse_id=pd.to_numeric(text.horse_id))
    frame = field.merge(text, on=["race_id", "horse_id"], how="left", validate="one_to_one")
    frame["won"] = pd.to_numeric(frame.finish_position, errors="coerce").eq(1).astype(np.int8)
    frame = rtv_text.prepare(frame)
    return frame.sort_values(["race_date", "race_id", "horse_id"], kind="stable").reset_index(drop=True)


def ensure_text_models(through, *, database=None, text_source=None):
    """Fit, in order, each missing quarter model up to the quarter of ``through``."""
    directory = rtv_text.models_dir()
    labels = [str(q) for q in pd.period_range(rtv_text.FIRST_QUARTER, pd.Timestamp(through).to_period("Q"), freq="Q")]
    missing = [q for q in labels if not (directory/f"{q}.npz").exists()]
    if not missing:
        return []
    gap = purge_days()
    end = (rtv_text.quarter_start(missing[-1])-pd.Timedelta(days=gap)).strftime("%Y-%m-%d")
    frame = _text_rows(rtv_text.START, end, database, text_source=text_source)
    frame = frame.loc[frame.groupby("race_id").won.transform("sum").gt(0)].reset_index(drop=True)
    position = labels.index(missing[0])
    warm = rtv_text.load_model(labels[position-1]) if position else None
    fitted = []
    for model in rtv_text.fit_models(frame, missing, purge_days=gap, warm=warm):
        model.save(directory)
        fitted.append(model.quarter)
        logger.info("Comment model %s: %d rows through %s, %d n-grams", model.quarter, model.rows,
                    model.trained_through, len(model.names))
    return fitted


def text_model(quarter, *, database=None):
    """The quarter's comment model, fitted first if it does not exist yet."""
    model = rtv_text.load_model(quarter)
    if model is None:
        ensure_text_models(rtv_text.quarter_start(quarter), database=database)
        model = rtv_text.load_model(quarter)
    return model


def refresh_text_scores(start, end, *, database=None, text_source=None, destination=None, previous_path=None):
    """Score every starter in [start, end] with its quarter's model and merge the result."""
    start = max(start, str(rtv_text.quarter_start(rtv_text.FIRST_QUARTER).date()))
    columns = ["race_id", "horse_id", "race_date", *rtv_text.SNAPSHOT_FEATURES]
    frame = _text_rows(start, end, database, text_source=text_source) if start <= end else pd.DataFrame()
    parts = []
    if not frame.empty:
        for quarter, rows in frame.groupby(rtv_text.quarters(frame.race_date).to_numpy(), sort=True):
            model = text_model(quarter, database=database)
            parts.append(rows.assign(rtv_text_logit=model.logits(rows),
                                     rtv_verdict_rank=rows.verdict_rank.astype("float64"))[columns])
    output = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=columns)
    output = _merge_dated(output, Path(destination or rtv_text.scores_path()), start, end, "text score", previous_path=previous_path)
    return {"text_scores": len(output)}


def refresh_text_history(start=rtv_text.START, end=None, *, database=None, source_root=None):
    """Backfill the card text, fit every quarter model and score all history."""
    end = end or (date.today()-pd.Timedelta(days=1)).isoformat()
    chunks = []
    for year in range(int(start[:4]), int(end[:4])+1):
        lo, hi = max(start, f"{year}-01-01"), min(end, f"{year}-12-31")
        tables = observation_tables(lo, hi, source_root=source_root, names=("runners",))
        links = match_runners(tables["races"], tables["runners"], canonical_rows(lo, hi, db_path=database))
        chunks.append(rtv_text.build_text_observations(tables["races"], tables["runners"], links))
        logger.info("Card text %s: %d linked starters", year, len(chunks[-1]))
    text = _merge_dated(pd.concat(chunks, ignore_index=True), rtv_text.text_path(), start, end, "text")
    fitted = ensure_text_models(date.today(), database=database)
    return {"text": len(text), "models": fitted, **refresh_text_scores(start, end, database=database)}


def _snapshot_text(frame):
    """Comment score for each runner of a pre-race card capture (NaN without card text)."""
    output = pd.DataFrame(np.nan, index=frame.index, columns=list(rtv_text.SNAPSHOT_FEATURES))
    if "card_analyst_comment" not in frame and "card_analyst_verdict" not in frame:
        return output
    get = lambda c: frame[c] if c in frame else pd.Series(index=frame.index, dtype=object)  # noqa: E731
    state = get("card_status__state").astype("string").str.lower()
    # Ordered among the runners still in the race, as stored history is.
    out_of_race = (get("card_withdrawn").astype("boolean").fillna(False).astype(bool)
                   | get("card_reserve").astype("boolean").fillna(False).astype(bool)
                   | state.isin(["scratched", "withdrawn", "non_runner", "nonrunner", "reserve"]).fillna(False).astype(bool))
    rows = pd.DataFrame({"race": frame.race_key.astype(str)+"|"+frame.fetched_at.astype(str),
        "race_date": frame.race_key.astype(str).str[:10], "horse_name": get("card_horse_name"),
        "tf_comment": get("card_analyst_comment").map(rtv.strip_markup),
        "tf_verdict": get("card_analyst_verdict").map(rtv.strip_markup)}, index=frame.index).loc[~out_of_race]
    if rows.empty:
        return output
    rows = rtv_text.prepare(rows, race="race")
    for quarter, part in rows.groupby(rtv_text.quarters(rows.race_date).to_numpy()):
        try:
            model = text_model(quarter)
        except Exception as error:  # a missing model must not lose the numeric card
            logger.warning("No comment model for %s: %s", quarter, error)
            continue
        output.loc[part.index, "rtv_text_logit"] = model.logits(part)
        output.loc[part.index, "rtv_verdict_rank"] = part.verdict_rank.astype("float64").to_numpy()
    return output


def refresh_pre_observations(day=None, *, raw_path=None, destination=None):
    store = rtv.RawStore(str(raw_path or Path(config.DATA_DIR)/"rtv_cache/rtv_raw.sqlite"))
    try:
        tables = rtv.build_snapshot_tables(start=day, end=day, store=store)
    finally:
        store.close()
    observations = []

    def card(frame):
        values = features.card_values(frame)
        # Parsed like the card: '111p' is a rating of 111, not missing.
        values["rtv_pre_timeform_rating"] = values.rtv_card_tf_rating
        values["rtv_pre_timeform_stars"] = values.rtv_card_tf_stars
        return pd.concat([values, _snapshot_text(frame)], axis=1)

    def pace(frame):
        return pd.DataFrame({"rtv_pre_pace_position": pd.to_numeric(frame.get("pm_pace_map__position"), errors="coerce"),
            "rtv_pre_draw_advantage": pd.to_numeric(frame.get("pm_draw__lengths_advantage"), errors="coerce")}, index=frame.index)

    specs = (("pre_runners", "card_horse_name", card), ("pace_maps", "pm_runner_name", pace))
    for table, horse_name, values in specs:
        frame = tables[table]
        if frame.empty:
            continue
        pieces = frame.race_key.str.split("/", expand=True)
        out = pd.DataFrame({"race_date": pieces[0], "track_key": pieces[1].str.replace("-", " ").map(canonical_track_key),
            "horse_key": frame[horse_name].map(normalise_horse_key),
            "observed_at": pd.to_datetime(frame.fetched_at, utc=True)})
        local = pieces[0]+" "+pieces[2].str[:2]+":"+pieces[2].str[2:]
        off = pd.to_datetime(local).dt.tz_localize("Europe/London").dt.tz_convert("UTC")
        out["off_key"] = off.dt.strftime("%H:%M")
        out = pd.concat([out, values(frame).reindex(columns=list(features.SNAPSHOT_FEATURES))], axis=1)
        # Captures made after the intended five-minute forecast decision cannot
        # become pre-race observations for that race.
        out = out.loc[out.observed_at.le(off-pd.Timedelta(minutes=5))]
        observations.append(out)
    keys = ["race_date", "track_key", "off_key", "horse_key"]
    snapshot = list(features.SNAPSHOT_FEATURES)
    output = pd.concat(observations, ignore_index=True) if observations else pd.DataFrame(columns=keys+["observed_at", *snapshot])
    if not output.empty:
        # Coalesce concurrent payloads only at their actual timestamps; ffill
        # cannot bring a later pace observation into an earlier card capture.
        output = output.groupby(keys+["observed_at"], as_index=False)[snapshot].first()
        output = output.sort_values("observed_at", kind="stable")
        output[snapshot] = output.groupby(keys)[snapshot].ffill()
    path = Path(destination or features.pre_path())
    if day and path.exists():
        previous = pd.read_parquet(path)
        output = pd.concat([previous.loc[previous.race_date.ne(day)], output], ignore_index=True)
    atomic_parquet(output, path)
    return {"observations": len(output), "races": int(output[keys[:3]].drop_duplicates().shape[0]),
            "published_at": datetime.now(timezone.utc).isoformat()}
