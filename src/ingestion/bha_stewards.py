"""BHA stewards reports scraper.

Fetches stewards reports from the public BHA website/API used by:
https://www.britishhorseracing.com/racing/stewards-reports/

The page frontend calls:
    https://api09.horseracing.software/bha/v1/stewards-room/reports
with a Bearer token set in BHA's public Angular app bundle.

This module retrieves that token dynamically from app.js, then fetches
report pages with optional date/query filters.
"""

from __future__ import annotations

import argparse
import html
import json
import logging
import re
from pathlib import Path
from typing import Any

import pandas as pd
import requests

logger = logging.getLogger(__name__)

BHA_BASE = "https://www.britishhorseracing.com"
BHA_APP_JS = (
    "https://www.britishhorseracing.com/wp-content/themes/bha/"
    "library/js/angular/app.js?ver=1.19"
)
BHA_STEWARDS_API = "https://api09.horseracing.software/bha/v1/stewards-room/reports"

_DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/126.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, text/plain, */*",
    "Origin": BHA_BASE,
    "Referer": f"{BHA_BASE}/racing/stewards-reports/",
}


def _get_with_backoff(
    session: requests.Session,
    url: str,
    params: dict[str, Any],
    *,
    max_retries: int = 6,
    base_delay: float = 2.0,
    timeout: int = 30,
) -> requests.Response:
    """GET that retries on 429/5xx with exponential backoff + jitter.

    The BHA API rate-limits aggressively in bursts (HTTP 429, no Retry-After).
    Without backoff a long backfill silently loses every throttled date, which
    front-loads coverage and leaves multi-year gaps. We honour Retry-After when
    present, else back off 2,4,8,... seconds with jitter, then return the last
    response so the caller can decide.
    """
    import random
    import time

    resp = session.get(url, params=params, timeout=timeout)
    for attempt in range(max_retries):
        if resp.status_code != 429 and resp.status_code < 500:
            return resp
        retry_after = resp.headers.get("Retry-After")
        if retry_after and retry_after.isdigit():
            wait = float(retry_after)
        else:
            wait = base_delay * (2 ** attempt) + random.uniform(0, 1.0)
        logger.debug("BHA %s on %s; backing off %.1fs (attempt %d/%d)",
                     resp.status_code, params, wait, attempt + 1, max_retries)
        time.sleep(wait)
        resp = session.get(url, params=params, timeout=timeout)
    return resp


def _extract_bearer_token(session: requests.Session) -> str:
    """Read BHA's public Angular bundle and extract the live Bearer token.

    app.js assigns ``$httpProvider.defaults.headers.common['Authorization']``
    more than once (a dev/localhost block followed by the live block). Angular
    keeps the *last* assignment, so we must take the final match, not the first
    -- the earlier token is stale and the API answers 401 for it.
    """
    resp = session.get(BHA_APP_JS, timeout=25)
    resp.raise_for_status()

    # In app.js: $httpProvider.defaults.headers.common['Authorization'] = 'Bearer ...';
    matches = re.findall(
        r"Authorization'\]\s*=\s*'Bearer\s+([^']+)'",
        resp.text,
    )
    if matches:
        return matches[-1].strip()

    # Fallback broader pattern (also take the last occurrence).
    matches = re.findall(r"Bearer\s+([0-9]+\|[A-Za-z0-9]+)", resp.text)
    if matches:
        return matches[-1].strip()

    raise RuntimeError("Could not extract BHA Bearer token from app.js")


def _strip_html(text: str | None) -> str:
    if not text:
        return ""
    no_tags = re.sub(r"<[^>]+>", " ", text)
    unescaped = html.unescape(no_tags)
    return re.sub(r"\s+", " ", unescaped).strip()


def _normalise_report_row(row: dict[str, Any]) -> dict[str, Any]:
    enquiry_html = row.get("enquiry") or row.get("enquiryComments") or ""

    # Legacy BHA format occasionally stores "why they ran badly" after
    # a special delimiter in the same field.
    enquiry_main = enquiry_html
    why_badly = ""
    if "</p>~<p>" in enquiry_html:
        split = enquiry_html.split("</p>~<p>", 1)
        enquiry_main = split[0]
        why_badly = split[1]

    out = {
        "fixture_id": row.get("fixtureId"),
        "year": row.get("year"),
        "date": row.get("date"),
        "course": row.get("course"),
        "pdf_report_url": row.get("pdfReportUrl"),
        "abandoned_reason": row.get("abandonedReason"),
        "fixture_comments": row.get("fixtureComments") or "",
        "enquiry_html": enquiry_main,
        "enquiry_text": _strip_html(enquiry_main),
        "why_they_ran_badly_html": why_badly,
        "why_they_ran_badly_text": _strip_html(why_badly),
        "data_source": row.get("dataSource"),
    }
    return out


def _read_existing_output(output_path: str | Path) -> pd.DataFrame:
    output = Path(output_path)
    if not output.exists():
        return pd.DataFrame()
    try:
        if output.suffix.lower() == ".parquet":
            return pd.read_parquet(output)
        return pd.read_csv(output)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _with_row_key(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        out = df.copy()
        out["__row_key"] = pd.Series(dtype="object")
        return out

    key_cols = [
        "fixture_id",
        "date",
        "course",
        "pdf_report_url",
        "enquiry_text",
        "why_they_ran_badly_text",
    ]
    out = df.copy()
    parts: list[pd.Series] = []
    for col in key_cols:
        if col in out.columns:
            parts.append(out[col].astype(str).fillna(""))
        else:
            parts.append(pd.Series([""] * len(out), index=out.index, dtype="object"))
    out["__row_key"] = parts[0]
    for p in parts[1:]:
        out["__row_key"] = out["__row_key"] + "|" + p
    return out


def fetch_bha_stewards_reports(
    *,
    date: str | None = None,
    query: str | None = None,
    from_date: str | None = None,
    to_date: str | None = None,
    include_archive: bool = False,
    max_pages: int | None = 1,
    bearer_token: str | None = None,
    session: requests.Session | None = None,
) -> pd.DataFrame:
    """Fetch BHA stewards reports into a dataframe.

    Args:
        date: Optional single day filter, format YYYY-MM-DD.
        query: Optional text search query (equivalent to page q=).
        from_date: Optional archive lower date bound.
        to_date: Optional archive upper date bound.
        include_archive: Whether to pass archive=1.
        max_pages: Number of paginated pages to fetch. None = fetch all pages.
        bearer_token: Optional explicit Bearer token override.
        session: Optional existing requests session.
    """
    owns_session = session is None
    s = session or requests.Session()
    s.headers.update(_DEFAULT_HEADERS)

    try:
        token = bearer_token or _extract_bearer_token(s)
        s.headers.update({"Authorization": f"Bearer {token}"})

        params: dict[str, Any] = {"archive": 1 if include_archive else 0}
        if date:
            params["ondate"] = date
        if query:
            params["q"] = query
        if from_date:
            params["fromdate"] = from_date
        if to_date:
            params["todate"] = to_date

        page = 1
        records: list[dict[str, Any]] = []

        while True:
            page_params = dict(params)
            page_params["pagenum"] = page
            r = _get_with_backoff(s, BHA_STEWARDS_API, page_params)
            if r.status_code in (401, 403):
                raise RuntimeError(
                    "BHA API denied access (401/403). This endpoint is protected and "
                    "may require a fresh browser-issued Bearer token/IP reputation. "
                    "Try passing a current token via --bearer-token."
                )
            r.raise_for_status()

            payload = r.json()
            data = payload.get("data", [])
            if not isinstance(data, list):
                raise RuntimeError(
                    f"Unexpected BHA response structure: keys={list(payload.keys())}"
                )

            for row in data:
                records.append(_normalise_report_row(row))

            last_page = int(payload.get("last_page") or page)
            if max_pages is not None and page >= max_pages:
                break
            if page >= last_page:
                break
            page += 1

        df = pd.DataFrame.from_records(records)
        if not df.empty and "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
        return df

    finally:
        if owns_session:
            s.close()


def save_bha_stewards_reports(
    output_path: str | Path,
    **fetch_kwargs: Any,
) -> pd.DataFrame:
    """Fetch reports and save to CSV or parquet based on extension."""
    incremental = bool(fetch_kwargs.pop("incremental", False))
    existing_df = pd.DataFrame()
    existing_keys: set[str] = set()

    if incremental:
        existing_df = _read_existing_output(output_path)
        if not existing_df.empty:
            existing_with_key = _with_row_key(existing_df)
            existing_keys = set(existing_with_key["__row_key"].astype(str).tolist())
        logger.info("Incremental mode: found %d existing BHA rows in %s", len(existing_keys), output_path)

    df = fetch_bha_stewards_reports(**fetch_kwargs)
    df = _with_row_key(df)

    if incremental and not df.empty and existing_keys:
        before = len(df)
        df = df[~df["__row_key"].astype(str).isin(existing_keys)].reset_index(drop=True)
        skipped = before - len(df)
        if skipped:
            logger.info("Incremental mode: skipped %d already-fetched BHA rows", skipped)

    if incremental and not existing_df.empty:
        existing_with_key = _with_row_key(existing_df)
        df = pd.concat([existing_with_key, df], ignore_index=True)

    if "__row_key" in df.columns:
        df = df.drop_duplicates(subset=["__row_key"], keep="first").drop(columns=["__row_key"])

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    if output.suffix.lower() == ".parquet":
        df.to_parquet(output, index=False)
    else:
        df.to_csv(output, index=False)

    logger.info("Saved %d BHA steward reports to %s", len(df), output)
    return df


def backfill_bha_stewards_by_date(
    output_path: str | Path,
    dates: "list[str]",
    *,
    bearer_token: str | None = None,
    sleep_secs: float = 0.4,
    flush_every: int = 50,
) -> pd.DataFrame:
    """Backfill the full BHA report history by querying one ``ondate`` per day.

    The public BHA feed only serves the most recent ~2 days; history is only
    reachable via the per-day ``ondate`` filter (date-range params are ignored
    by the API). This walks an explicit list of dates -- pass the unique race
    dates you actually model -- and accumulates into ``output_path``.

    Resumable: dates already present in the output (by the ``date`` column) are
    skipped, so it can be re-run to top up or recover from an interruption.
    """
    import time

    output = Path(output_path)
    existing = _read_existing_output(output)
    done_dates: set[str] = set()
    if not existing.empty and "date" in existing.columns:
        done_dates = {
            str(pd.to_datetime(d, errors="coerce").date())
            for d in existing["date"].dropna()
            if not pd.isna(pd.to_datetime(d, errors="coerce"))
        }

    want = [str(pd.to_datetime(d, errors="coerce").date()) for d in dates]
    want = sorted({d for d in want if d != "NaT"})
    todo = [d for d in want if d not in done_dates]
    logger.info(
        "BHA date backfill: %d target dates, %d already done, %d to fetch",
        len(want), len(done_dates), len(todo),
    )

    session = requests.Session()
    session.headers.update(_DEFAULT_HEADERS)
    token = bearer_token or _extract_bearer_token(session)
    session.headers.update({"Authorization": f"Bearer {token}"})

    collected: list[pd.DataFrame] = []
    new_rows = 0
    for i, d in enumerate(todo, start=1):
        try:
            day_df = fetch_bha_stewards_reports(
                date=d, include_archive=True, max_pages=None,
                bearer_token=token, session=session,
            )
        except Exception as exc:  # pragma: no cover - network drift
            logger.warning("BHA backfill failed for %s (%s)", d, exc)
            day_df = pd.DataFrame()
        if not day_df.empty:
            collected.append(day_df)
            new_rows += len(day_df)
        if i % 25 == 0:
            logger.info("BHA backfill %d/%d dates (%d new rows)", i, len(todo), new_rows)
        # Periodically flush so a long run is crash-safe.
        if collected and i % flush_every == 0:
            _flush_backfill(output, existing, collected)
            existing = _read_existing_output(output)
            collected = []
        if sleep_secs:
            time.sleep(sleep_secs)

    session.close()
    final = _flush_backfill(output, existing, collected)
    logger.info("BHA date backfill complete: %d total rows in %s", len(final), output)
    return final


def _flush_backfill(
    output: Path, existing: pd.DataFrame, new_frames: list[pd.DataFrame]
) -> pd.DataFrame:
    frames = [f for f in ([existing] + new_frames) if not f.empty]
    if not frames:
        return existing
    combined = pd.concat(frames, ignore_index=True)
    combined = _with_row_key(combined)
    combined = combined.drop_duplicates(subset=["__row_key"], keep="first").drop(
        columns=["__row_key"]
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.suffix.lower() == ".parquet":
        combined.to_parquet(output, index=False)
    else:
        combined.to_csv(output, index=False)
    return combined


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Fetch BHA stewards reports")
    p.add_argument("--date", help="Single report date YYYY-MM-DD")
    p.add_argument("--query", help="Archive text query")
    p.add_argument("--from-date", dest="from_date", help="Archive from date YYYY-MM-DD")
    p.add_argument("--to-date", dest="to_date", help="Archive to date YYYY-MM-DD")
    p.add_argument("--archive", action="store_true", help="Include archive mode")
    p.add_argument("--max-pages", type=int, default=1, help="Pages to fetch (default 1)")
    p.add_argument(
        "--bearer-token",
        dest="bearer_token",
        help="Optional explicit Bearer token for the BHA API",
    )
    p.add_argument(
        "--out",
        default="data/raw/bha_stewards_reports.csv",
        help="Output path (.csv or .parquet)",
    )
    p.add_argument(
        "--incremental",
        action="store_true",
        help="Skip rows already present in output and append only new rows",
    )
    p.add_argument(
        "--backfill-from-races",
        dest="backfill_from_races",
        help=(
            "Path to a results CSV/parquet with a 'race_date' column; backfill "
            "the full report history by querying ondate for each unique race "
            "date (resumable into --out)."
        ),
    )
    p.add_argument(
        "--sleep-secs", dest="sleep_secs", type=float, default=0.4,
        help="Per-request delay during date backfill (default 0.4)",
    )
    p.add_argument("--verbose", action="store_true", help="Enable debug logs")
    return p


def main() -> None:
    args = _build_arg_parser().parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if args.backfill_from_races:
        src_path = Path(args.backfill_from_races)
        races = (
            pd.read_parquet(src_path)
            if src_path.suffix.lower() == ".parquet"
            else pd.read_csv(src_path, usecols=["race_date"], low_memory=False)
        )
        dates = races["race_date"].dropna().astype(str).unique().tolist()
        df = backfill_bha_stewards_by_date(
            args.out, dates,
            bearer_token=args.bearer_token, sleep_secs=args.sleep_secs,
        )
        print(json.dumps({"rows": int(len(df)), "out": str(args.out)}))
        return

    df = save_bha_stewards_reports(
        output_path=args.out,
        date=args.date,
        query=args.query,
        from_date=args.from_date,
        to_date=args.to_date,
        include_archive=args.archive,
        max_pages=args.max_pages,
        bearer_token=args.bearer_token,
        incremental=args.incremental,
    )
    print(json.dumps({"rows": int(len(df)), "out": str(args.out)}))


if __name__ == "__main__":
    main()
