"""IHRB Daily Enquiries scraper.

Scrapes daily enquiry detail pages from https://www.ihrb.ie/daily-enquiries
and the site sitemap, then outputs structured rows.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path
from typing import Iterable
from urllib.parse import unquote, urljoin, urlparse

import pandas as pd
import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

IHRB_BASE = "https://www.ihrb.ie"
IHRB_LISTING_URL = f"{IHRB_BASE}/daily-enquiries"
IHRB_SITEMAP_URL = f"{IHRB_BASE}/sitemap"
IHRB_SITEMAP_XML_URL = f"{IHRB_BASE}/sitemap.xml"

_DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/126.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}


def _normalise_url(url: str) -> str:
    parsed = urlparse(url)
    return f"{parsed.scheme}://{parsed.netloc}{parsed.path}".rstrip("/")


def _extract_report_links_from_html(html_text: str) -> list[str]:
    soup = BeautifulSoup(html_text, "lxml")
    links: list[str] = []

    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        if not href:
            continue
        abs_url = urljoin(IHRB_BASE, href)
        norm = _normalise_url(abs_url)

        # Detail pages are /daily-enquiries/<slug>
        if "/daily-enquiries/" not in norm:
            continue
        if norm in {IHRB_LISTING_URL, IHRB_LISTING_URL + "/"}:
            continue

        links.append(norm)

    if not links:
        # Some IHRB pages are app shells without rendered anchors; fall back
        # to raw text extraction for absolute/relative daily-enquiries URLs.
        pattern = re.compile(
            r"(?:https?://www\\.ihrb\\.ie)?/daily-enquiries/[^\"'\\s<)]+",
            flags=re.IGNORECASE,
        )
        for match in pattern.findall(html_text):
            abs_url = urljoin(IHRB_BASE, match)
            norm = _normalise_url(abs_url)
            if norm in {IHRB_LISTING_URL, IHRB_LISTING_URL + "/"}:
                continue
            links.append(norm)

    # Preserve order while deduping
    return list(dict.fromkeys(links))


def _extract_report_links_from_sitemap_xml(xml_text: str) -> list[str]:
    soup = BeautifulSoup(xml_text, "xml")
    links: list[str] = []

    for loc in soup.find_all("loc"):
        url = loc.get_text(" ", strip=True)
        if not url:
            continue
        norm = _normalise_url(url)
        if "/daily-enquiries/" not in norm:
            continue
        if norm in {IHRB_LISTING_URL, IHRB_LISTING_URL + "/"}:
            continue
        links.append(norm)

    return list(dict.fromkeys(links))


def _get_listing_links(session: requests.Session) -> list[str]:
    r = session.get(IHRB_LISTING_URL, timeout=30)
    r.raise_for_status()
    return _extract_report_links_from_html(r.text)


def _get_sitemap_links(session: requests.Session) -> list[str]:
    # Prefer sitemap.xml because it contains full archive URLs.
    r_xml = session.get(IHRB_SITEMAP_XML_URL, timeout=30)
    r_xml.raise_for_status()
    xml_links = _extract_report_links_from_sitemap_xml(r_xml.text)
    if xml_links:
        return xml_links

    # Fall back to HTML sitemap if XML structure changes.
    r_html = session.get(IHRB_SITEMAP_URL, timeout=30)
    r_html.raise_for_status()
    return _extract_report_links_from_html(r_html.text)


def _title_to_date(title: str) -> pd.Timestamp | pd.NaT:
    # Example: "Naas 24th June 2026"
    m = re.search(r"(\d{1,2})(?:st|nd|rd|th)?\s+([A-Za-z]+)\s+(\d{4})", title)
    if not m:
        return pd.NaT
    day = m.group(1)
    month = m.group(2)
    year = m.group(3)
    return pd.to_datetime(f"{day} {month} {year}", errors="coerce", dayfirst=True)


def _url_slug_to_title(url: str) -> str:
    parsed = urlparse(url)
    slug = parsed.path.rsplit("/", 1)[-1]
    slug = unquote(slug)
    slug = slug.replace("-", " ")
    return slug.strip()


def _candidate_url_date(url: str) -> pd.Timestamp | pd.NaT:
    return _title_to_date(_url_slug_to_title(url))


def _parse_report_page(session: requests.Session, report_url: str) -> dict:
    r = session.get(report_url, timeout=30)
    r.raise_for_status()

    soup = BeautifulSoup(r.text, "lxml")
    main = soup.find("main")
    article = main.find("article") if main else None

    title_node = article.find(["h1", "h2"]) if article else soup.find("h1")
    title = title_node.get_text(" ", strip=True) if title_node else ""

    # Keep full article text so downstream regex/LLM tagging can be applied.
    article_text = article.get_text("\n", strip=True) if article else ""
    article_html = str(article) if article else ""

    report_date = _title_to_date(title)

    return {
        "source": "ihrb_daily_enquiries",
        "url": _normalise_url(report_url),
        "title": title,
        "report_date": report_date,
        "content_text": article_text,
        "content_html": article_html,
    }


def _iter_unique(urls: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for u in urls:
        n = _normalise_url(u)
        if n not in seen:
            seen.add(n)
            out.append(n)
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


def fetch_ihrb_daily_enquiries(
    *,
    include_listing: bool = True,
    include_sitemap: bool = True,
    max_reports: int | None = None,
    from_date: str | None = None,
    to_date: str | None = None,
    session: requests.Session | None = None,
) -> pd.DataFrame:
    """Fetch IHRB Daily Enquiries reports into a dataframe."""
    owns_session = session is None
    s = session or requests.Session()
    s.headers.update(_DEFAULT_HEADERS)

    try:
        candidate_urls: list[str] = []
        if include_listing:
            candidate_urls.extend(_get_listing_links(s))
        if include_sitemap:
            candidate_urls.extend(_get_sitemap_links(s))

        report_urls = _iter_unique(candidate_urls)

        from_ts = pd.to_datetime(from_date, errors="coerce") if from_date else None
        to_ts = pd.to_datetime(to_date, errors="coerce") if to_date else None

        if from_ts is not None or to_ts is not None:
            filtered_urls: list[str] = []
            kept_unknown = 0
            for u in report_urls:
                d = _candidate_url_date(u)
                if pd.isna(d):
                    kept_unknown += 1
                    filtered_urls.append(u)
                    continue
                if from_ts is not None and d < from_ts:
                    continue
                if to_ts is not None and d > to_ts:
                    continue
                filtered_urls.append(u)
            logger.info(
                "Date prefilter kept %d/%d URLs (including %d undated slugs for post-filter)",
                len(filtered_urls),
                len(report_urls),
                kept_unknown,
            )
            report_urls = filtered_urls

        if max_reports is not None:
            report_urls = report_urls[:max_reports]

        logger.info("Found %d IHRB daily enquiry report URLs", len(report_urls))

        records: list[dict] = []
        for i, url in enumerate(report_urls, start=1):
            try:
                records.append(_parse_report_page(s, url))
            except Exception as exc:  # pragma: no cover - network/page drift
                logger.warning("Failed to parse %s (%s)", url, exc)
            if i % 25 == 0:
                logger.info("Parsed %d/%d report pages", i, len(report_urls))

        df = pd.DataFrame.from_records(records)
        if not df.empty and "report_date" in df.columns:
            df["report_date"] = pd.to_datetime(df["report_date"], errors="coerce")
            if from_ts is not None:
                df = df[df["report_date"] >= from_ts]
            if to_ts is not None:
                df = df[df["report_date"] <= to_ts]
            df = df.sort_values(["report_date", "title"], ascending=[False, True]).reset_index(drop=True)
        return df

    finally:
        if owns_session:
            s.close()


def save_ihrb_daily_enquiries(output_path: str | Path, **fetch_kwargs) -> pd.DataFrame:
    incremental = bool(fetch_kwargs.pop("incremental", False))
    existing_df = pd.DataFrame()
    existing_urls: set[str] = set()

    if incremental:
        existing_df = _read_existing_output(output_path)
        if not existing_df.empty and "url" in existing_df.columns:
            existing_urls = {
                _normalise_url(u)
                for u in existing_df["url"].dropna().astype(str).tolist()
                if str(u).strip()
            }

        logger.info("Incremental mode: found %d existing URLs in %s", len(existing_urls), output_path)

    df = fetch_ihrb_daily_enquiries(**fetch_kwargs)

    if incremental and not df.empty and existing_urls:
        before = len(df)
        df = df[~df["url"].astype(str).map(_normalise_url).isin(existing_urls)].reset_index(drop=True)
        skipped = before - len(df)
        if skipped:
            logger.info("Incremental mode: skipped %d already-fetched report URLs", skipped)

    if incremental and not existing_df.empty:
        combined = pd.concat([existing_df, df], ignore_index=True)
        if "url" in combined.columns:
            combined = combined.drop_duplicates(subset=["url"], keep="first")
        df = combined

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    if output.suffix.lower() == ".parquet":
        df.to_parquet(output, index=False)
    else:
        df.to_csv(output, index=False)

    logger.info("Saved %d IHRB daily enquiry rows to %s", len(df), output)
    return df


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Fetch IHRB Daily Enquiries")
    p.add_argument("--max-reports", type=int, default=None, help="Maximum report pages to fetch")
    p.add_argument("--no-sitemap", action="store_true", help="Do not use /sitemap link discovery")
    p.add_argument("--no-listing", action="store_true", help="Do not use /daily-enquiries list discovery")
    p.add_argument("--from-date", dest="from_date", help="Lower date bound YYYY-MM-DD")
    p.add_argument("--to-date", dest="to_date", help="Upper date bound YYYY-MM-DD")
    p.add_argument(
        "--out",
        default="data/raw/ihrb_daily_enquiries.csv",
        help="Output path (.csv or .parquet)",
    )
    p.add_argument(
        "--incremental",
        action="store_true",
        help="Skip URLs already present in output and append only new rows",
    )
    p.add_argument("--verbose", action="store_true", help="Enable debug logs")
    return p


def main() -> None:
    args = _build_arg_parser().parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    df = save_ihrb_daily_enquiries(
        output_path=args.out,
        include_listing=not args.no_listing,
        include_sitemap=not args.no_sitemap,
        max_reports=args.max_reports,
        from_date=args.from_date,
        to_date=args.to_date,
        incremental=args.incremental,
    )
    print(json.dumps({"rows": int(len(df)), "out": str(args.out)}))


if __name__ == "__main__":
    main()
