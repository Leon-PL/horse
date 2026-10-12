"""Trainer yard locations (GB) from the BHA licensed-trainer directory.

The BHA "find a trainer" API (same api09.horseracing.software backend and
Bearer-token scheme as the stewards reports — token extraction is reused from
``src.bha_stewards_scraper``) lists every currently licensed GB trainer with a
``county``. County centroids give a yard location good to a few tens of km,
which is enough for a *travel-distance* signal (Fife -> Bath is ~600 km; the
noise is the county radius).

Artifacts / API:
    fetch_and_save()          -- pull the directory, save data/trainer_locations.csv
    load_trainer_directory()  -- cached load of that CSV (empty df if absent)
    resolve_trainer_latlon()  -- map scraped trainer names -> (lat, lon)
    haversine_km()            -- vectorised great-circle distance

Coverage notes:
    * GB only. Irish trainers (IHRB directory is a JS shell without a public
      JSON endpoint) resolve to NaN -> ``trainer_travel_known = 0``.
    * The directory is *current* licence holders; long-retired trainers in old
      rows also resolve to NaN. The feature matters most for live cards, where
      coverage is near-total.

CLI:
    python -m src.trainer_locations            # fetch + save + match report
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd
import requests

logger = logging.getLogger(__name__)

BHA_TRAINERS_API = "https://api09.horseracing.software/bha/v1/trainers"

_DATA_DIR = Path(__file__).resolve().parent.parent / "data"
TRAINER_LOCATIONS_CSV = _DATA_DIR / "trainer_locations.csv"

# ---------------------------------------------------------------------------
# County centroids (approximate, degrees). BHA uses postal/ceremonial county
# names, occasionally old-style ones (Avon, Cleveland, Humberside). A few tens
# of km of centroid error is fine for a travel signal. Unknown counties are
# logged at fetch time — extend the table as they appear.
# ---------------------------------------------------------------------------
COUNTY_CENTROIDS: dict[str, tuple[float, float]] = {
    # England
    "bedfordshire": (52.08, -0.45),
    "berkshire": (51.45, -1.05),          # Lambourn valley
    "bristol": (51.45, -2.58),
    "buckinghamshire": (51.80, -0.85),
    "cambridgeshire": (52.30, 0.05),
    "cheshire": (53.20, -2.55),
    "cornwall": (50.40, -4.75),
    "cumbria": (54.55, -2.95),
    "derbyshire": (53.10, -1.60),
    "devon": (50.75, -3.75),
    "dorset": (50.80, -2.35),
    "durham": (54.70, -1.75),
    "county durham": (54.70, -1.75),
    "east riding of yorkshire": (53.90, -0.55),
    "east yorkshire": (53.90, -0.55),
    "east sussex": (50.95, 0.25),
    "essex": (51.80, 0.55),
    "gloucestershire": (51.85, -2.20),
    "greater london": (51.51, -0.13),
    "london": (51.51, -0.13),
    "greater manchester": (53.50, -2.30),
    "hampshire": (51.05, -1.30),
    "herefordshire": (52.10, -2.75),
    "hertfordshire": (51.85, -0.25),
    "isle of wight": (50.68, -1.30),
    "kent": (51.20, 0.70),
    "lancashire": (53.85, -2.65),
    "leicestershire": (52.65, -1.10),
    "lincolnshire": (53.10, -0.20),
    "merseyside": (53.40, -2.95),
    "norfolk": (52.65, 1.00),
    "northamptonshire": (52.25, -0.90),
    "northumberland": (55.25, -2.00),
    "north yorkshire": (54.15, -1.45),    # Middleham / Malton
    "nottinghamshire": (53.10, -1.00),
    "oxfordshire": (51.75, -1.30),
    "rutland": (52.65, -0.65),
    "shropshire": (52.65, -2.70),
    "somerset": (51.10, -2.90),
    "south yorkshire": (53.45, -1.35),
    "staffordshire": (52.85, -2.00),
    # Most Suffolk licences are Newmarket yards — use Newmarket, not the
    # geographic county centroid 50 km east.
    "suffolk": (52.24, 0.40),
    "surrey": (51.25, -0.45),             # Epsom
    "tyne and wear": (54.95, -1.55),
    "warwickshire": (52.30, -1.55),
    "west midlands": (52.48, -1.90),
    "west sussex": (50.95, -0.45),
    "west yorkshire": (53.75, -1.60),
    "wiltshire": (51.35, -1.95),
    "worcestershire": (52.20, -2.20),
    # Old-style / postal county names
    "avon": (51.45, -2.50),
    "cleveland": (54.55, -1.20),
    "humberside": (53.75, -0.40),
    "north humberside": (53.80, -0.45),
    "south humberside": (53.55, -0.55),
    "hereford and worcester": (52.15, -2.50),
    "middlesex": (51.55, -0.35),
    "sussex": (50.95, -0.10),
    "yorkshire": (54.00, -1.20),
    # Wales
    "carmarthenshire": (51.85, -4.30),
    "ceredigion": (52.25, -4.00),
    "clwyd": (53.15, -3.30),
    "denbighshire": (53.10, -3.30),
    "flintshire": (53.20, -3.15),
    "wrexham": (53.05, -3.00),
    "glamorgan": (51.55, -3.45),
    "vale of glamorgan": (51.45, -3.45),
    "south glamorgan": (51.45, -3.40),
    "mid glamorgan": (51.60, -3.45),
    "west glamorgan": (51.65, -3.85),
    "gwent": (51.75, -2.95),
    "monmouthshire": (51.75, -2.90),
    "gwynedd": (52.90, -4.00),
    "powys": (52.30, -3.40),
    "pembrokeshire": (51.85, -4.90),
    "conwy": (53.15, -3.80),
    "dyfed": (52.00, -4.40),
    # Scotland
    "aberdeenshire": (57.20, -2.60),
    "angus": (56.70, -2.90),
    "ayrshire": (55.45, -4.55),
    "north ayrshire": (55.65, -4.70),
    "south ayrshire": (55.35, -4.60),
    "east ayrshire": (55.50, -4.35),
    "scottish borders": (55.55, -2.70),
    "borders": (55.55, -2.70),
    "roxburghshire": (55.50, -2.55),
    "berwickshire": (55.75, -2.40),
    "selkirkshire": (55.50, -2.90),
    "peeblesshire": (55.60, -3.20),
    "dumfries and galloway": (55.05, -3.80),
    "dumfriesshire": (55.15, -3.55),
    "kirkcudbrightshire": (54.95, -4.05),
    "wigtownshire": (54.90, -4.60),
    "fife": (56.25, -3.15),
    "lanarkshire": (55.65, -3.85),
    "north lanarkshire": (55.85, -3.95),
    "south lanarkshire": (55.55, -3.80),
    "lothian": (55.90, -3.10),
    "midlothian": (55.85, -3.10),
    "east lothian": (55.95, -2.75),
    "west lothian": (55.90, -3.55),
    "perthshire": (56.50, -3.60),
    "perth and kinross": (56.50, -3.60),
    "kinross-shire": (56.20, -3.40),
    "stirlingshire": (56.10, -4.00),
    "stirling": (56.10, -4.00),
    "highland": (57.50, -4.50),
    "moray": (57.50, -3.25),
    "aberdeen city": (57.15, -2.10),
    "clackmannanshire": (56.10, -3.75),
}

# Honorifics that appear before initials in trainer styles.
_TITLES = {"mr", "mrs", "ms", "miss", "dr", "sir", "lady", "lord",
           "major", "capt", "captain", "rev", "prof", "the hon", "hon"}


# ---------------------------------------------------------------------------
# Name normalisation / matching
# ---------------------------------------------------------------------------

def normalise_trainer_name(name: str) -> str:
    """Canonical lower-case form matching data_processor's trainer cleaning:
    dots/hyphens in initials -> spaces, honorifics stripped, spaces collapsed."""
    s = str(name or "").lower()
    s = re.sub(r"\.(?=[a-z])", " ", s)
    s = re.sub(r"\.\s*", " ", s)
    s = re.sub(r"[^a-z&' ]+", " ", s)
    s = re.sub(r"\s{2,}", " ", s).strip()
    words = s.split()
    while words and words[0] in _TITLES:
        words = words[1:]
    return " ".join(words)


def _surname_initial_key(norm_name: str) -> tuple[str, str] | None:
    """('alexander', 'n') from 'n w alexander' or 'nick walter alexander'.
    Joint names ('a king & b smith') use the first partner."""
    first = norm_name.split("&")[0].strip()
    parts = [p for p in first.split() if p]
    if len(parts) < 2:
        return None
    return parts[-1], parts[0][0]


# ---------------------------------------------------------------------------
# BHA directory fetch
# ---------------------------------------------------------------------------

def fetch_bha_trainer_directory(
    per_page: int | None = 100, timeout: int = 30
) -> pd.DataFrame:
    """Fetch every licensed trainer (name/licence/county) from the BHA API."""
    from src.bha_stewards_scraper import (
        _DEFAULT_HEADERS,
        _extract_bearer_token,
        _get_with_backoff,
    )

    session = requests.Session()
    session.headers.update(_DEFAULT_HEADERS)
    session.headers["Authorization"] = f"Bearer {_extract_bearer_token(session)}"

    rows: list[dict] = []
    page = 1
    while True:
        resp = _get_with_backoff(
            session,
            BHA_TRAINERS_API,
            {"page": page, "per_page": per_page},
            timeout=timeout,
        )
        if resp.status_code == 422 and per_page:
            # API caps per_page; fall back to its default page size.
            per_page = None
            resp = _get_with_backoff(
                session, BHA_TRAINERS_API, {"page": page}, timeout=timeout,
            )
        resp.raise_for_status()
        payload = resp.json()
        data = payload.get("data", [])
        for t in data:
            rows.append({
                "bha_id": t.get("trainerId"),
                "bha_style": t.get("trainerStyle", ""),
                "bha_fullname": t.get("fullName", ""),
                "licence_type": t.get("licenceType", ""),
                "county": t.get("county", "") or "",
            })
        last_page = payload.get("last_page", page)
        if page >= last_page or not data:
            break
        page += 1

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    county_key = df["county"].str.strip().str.lower()
    latlon = county_key.map(COUNTY_CENTROIDS)
    df["lat"] = latlon.map(lambda v: v[0] if isinstance(v, tuple) else np.nan)
    df["lon"] = latlon.map(lambda v: v[1] if isinstance(v, tuple) else np.nan)

    unknown = sorted(set(county_key[df["lat"].isna() & (county_key != "")]))
    if unknown:
        logger.warning(
            "No centroid for %d BHA counties (extend COUNTY_CENTROIDS): %s",
            len(unknown), unknown,
        )
    return df


def fetch_and_save(path: Path | str = TRAINER_LOCATIONS_CSV) -> pd.DataFrame:
    df = fetch_bha_trainer_directory()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    logger.info("Saved %d BHA trainers (%d located) to %s",
                len(df), int(df["lat"].notna().sum()) if len(df) else 0, path)
    return df


_DIRECTORY_CACHE: dict[str, pd.DataFrame] = {}


def load_trainer_directory(path: Path | str = TRAINER_LOCATIONS_CSV) -> pd.DataFrame:
    """Cached CSV load; empty DataFrame when the artifact hasn't been built."""
    key = str(path)
    if key in _DIRECTORY_CACHE:
        return _DIRECTORY_CACHE[key]
    p = Path(path)
    if not p.exists():
        df = pd.DataFrame(
            columns=["bha_id", "bha_style", "bha_fullname",
                     "licence_type", "county", "lat", "lon"])
    else:
        df = pd.read_csv(p)
    _DIRECTORY_CACHE[key] = df
    return df


# ---------------------------------------------------------------------------
# Resolution: scraped trainer name -> (lat, lon)
# ---------------------------------------------------------------------------

def _build_indexes(directory: pd.DataFrame):
    """(exact-name index, unique surname+initial index, unique surname index)."""
    exact: dict[str, tuple[float, float]] = {}
    by_si: dict[tuple[str, str], list[tuple[float, float]]] = {}
    by_surname: dict[str, list[tuple[float, float]]] = {}

    located = directory[directory["lat"].notna()]
    for _, row in located.iterrows():
        loc = (float(row["lat"]), float(row["lon"]))
        for col in ("bha_style", "bha_fullname"):
            norm = normalise_trainer_name(row.get(col, ""))
            if norm:
                exact.setdefault(norm, loc)
                si = _surname_initial_key(norm)
                if si:
                    by_si.setdefault(si, []).append(loc)
                    by_surname.setdefault(si[0], []).append(loc)

    def _uniques(d):
        return {k: v[0] for k, v in d.items() if len(set(v)) == 1}

    return exact, _uniques(by_si), _uniques(by_surname)


def resolve_trainer_latlon(
    names, path: Path | str = TRAINER_LOCATIONS_CSV
) -> dict[str, tuple[float, float]]:
    """Map each scraped trainer name to (lat, lon) where resolvable.

    Match order: exact normalised name -> unique surname+first-initial ->
    unique surname. Unresolved names are simply absent from the result.
    """
    directory = load_trainer_directory(path)
    if directory.empty:
        return {}
    exact, by_si, by_surname = _build_indexes(directory)

    out: dict[str, tuple[float, float]] = {}
    for name in names:
        norm = normalise_trainer_name(name)
        if not norm:
            continue
        loc = exact.get(norm)
        if loc is None:
            si = _surname_initial_key(norm)
            if si:
                loc = by_si.get(si) or by_surname.get(si[0])
        if loc is not None:
            out[name] = loc
    return out


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------

def haversine_km(lat1, lon1, lat2, lon2):
    """Vectorised great-circle distance in km (inputs in degrees)."""
    lat1, lon1, lat2, lon2 = (np.radians(np.asarray(a, dtype=float))
                              for a in (lat1, lon1, lat2, lon2))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 6371.0 * 2 * np.arcsin(np.sqrt(a))


# ---------------------------------------------------------------------------
# CLI: fetch + coverage report against races.db
# ---------------------------------------------------------------------------

def _main() -> None:
    import sqlite3

    logging.basicConfig(level=logging.INFO)
    df = fetch_and_save()
    print(f"BHA directory: {len(df)} trainers, "
          f"{int(df['lat'].notna().sum())} with a county centroid")

    db = _DATA_DIR / "races.db"
    if db.exists():
        con = sqlite3.connect(db)
        names = [r[0] for r in con.execute(
            "SELECT DISTINCT trainer FROM results WHERE trainer != ''")]
        con.close()
        resolved = resolve_trainer_latlon(names)
        print(f"races.db: {len(names)} distinct trainer names, "
              f"{len(resolved)} resolved ({100 * len(resolved) / max(len(names), 1):.1f}%)")
        # Row-weighted coverage matters more than name-weighted:
        con = sqlite3.connect(db)
        counts = dict(con.execute(
            "SELECT trainer, COUNT(*) FROM results GROUP BY trainer"))
        con.close()
        total = sum(counts.values())
        covered = sum(counts.get(n, 0) for n in resolved)
        print(f"row-weighted coverage: {covered:,}/{total:,} "
              f"({100 * covered / max(total, 1):.1f}%)")


if __name__ == "__main__":
    _main()
