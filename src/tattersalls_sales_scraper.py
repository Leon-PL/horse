"""Tattersalls bloodstock sales-results scraper.

Public sale prices (hammer price in guineas) for every lot at Tattersalls
auctions 1999-present. Used to attach a horse's PUBLIC SALE PRICE as a feature
— the orthogonal, market-underweighted proxy for the private stable expectation
that beats our model on debutants/lightly-raced horses (see memory
feature-store-saturation / market-info-gap).

Data path (discovered by inspecting the search-lots JS app's XHRs; the lot data
is NOT in the WordPress content API):

    secure.tattersalls.com/4DCGI/Entry/Query   -> establishes a session cookie
    secure.tattersalls.com/4DCGI/Entry/list?...{SALE}%20{YY}...  -> lot table HTML

A session cookie from Entry/Query is REQUIRED; without it Entry/list returns the
empty search form. robots.txt allows all (Disallow: empty). One request per
(sale, year); ~29 sale codes x 28 years, many empty, so a full backfill is a few
hundred GETs.

Lots at yearling/foal sales are UNNAMED — identified by sire x dam (+ foaling
year, implied by sale type). That (sire, dam, foaling_year) tuple is exactly the
join key to our pedigree table (horse_pedigree.sire/dam/foaled), so no horse-name
normalisation is needed.
"""
from __future__ import annotations

import argparse
import logging
import re
import time
from html import unescape

import pandas as pd
import requests

logger = logging.getLogger(__name__)

BASE = "https://secure.tattersalls.com"
QUERY_URL = f"{BASE}/4DCGI/Entry/Query"
LIST_URL = f"{BASE}/4DCGI/Entry/list"

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
    ),
    "Accept": "*/*",
    "Referer": QUERY_URL,
}

# Sale codes from Entry/Query QuerySetup. type hints the age-at-sale so foaling
# year can be derived: "yearling"->sale_year-1, "foal"->sale_year, others unknown
# (named horses / HIT — join on (sire,dam) and disambiguate by date instead).
SALE_CODES: dict[str, tuple[str, str]] = {
    "FEB": ("February", "mixed"), "MCH": ("March", "mixed"),
    "CBU": ("Craven Breeze Up", "2yo"), "GBU": ("Guineas Breeze Up", "2yo"),
    "GHT": ("Guineas Horses in Training", "hit"), "MNH": ("May National Hunt", "mixed"),
    "JUL": ("July", "mixed"), "SUM": ("Summer Sale", "mixed"), "JUN": ("June Sale", "mixed"),
    "AUG": ("August", "mixed"), "SEP": ("September", "mixed"),
    "SOM": ("Somerville Yearlings", "yearling"),
    "OC1": ("October Yearlings, Book 1", "yearling"),
    "OC2": ("October Yearlings, Book 2", "yearling"),
    "OC3": ("October Yearlings, Book 3", "yearling"),
    "OC4": ("October Yearlings, Book 4", "yearling"),
    "AUY": ("Autumn Yearlings", "yearling"), "HOU": ("Houghton Yearling Sale", "yearling"),
    "OCT": ("October Yearlings", "yearling"), "AUT": ("Autumn Horses in Training", "hit"),
    "YRL": ("Autumn Yearlings", "yearling"), "NOV": ("November Sale", "mixed"),
    "DEY": ("December Yearlings", "yearling"), "DEF": ("December Foals", "foal"),
    "DEM": ("December Mares", "mixed"),
}

# one lot row: <td lot><a class=ll>N</a>  <td tdh> <a class=hn>SIRE</a>/ <a class=hn>DAM</a> COL.SEX <span sn>(CTRY)</span> ...
#   <td>VENDOR  <td>PURCHASER  <td class=price> PRICE
_ROW_RE = re.compile(
    r"<td class='lot[^>]*>.*?class=\"ll\">\s*([0-9A-Za-z]+)\s*</a>"   # lot
    r".*?<td class='tdh'>(.*?)</td>"                                   # sire/dam block
    r"\s*<td[^>]*>(.*?)</td>"                                          # vendor
    r"\s*<td[^>]*>(.*?)</td>"                                          # purchaser
    r"\s*<td class='price[^>]*>(.*?)</td>",                            # price
    re.S,
)
_HN_RE = re.compile(r'class="hn">(.*?)</a>', re.S)
_COLSEX_RE = re.compile(r"</a>\s*([A-Za-z.]+?\.)\s*(?:<span|</span|$)", re.S)
_TAG_RE = re.compile(r"<[^>]+>")
_SEX_MAP = {"C": "C", "F": "F", "G": "G", "R": "R"}  # last letter of e.g. B.C./Ch.F./B.G.


def _text(s: str) -> str:
    return unescape(_TAG_RE.sub(" ", s)).replace("\xa0", " ").strip()


def _strip_country(name: str) -> str:
    return re.sub(r"\s*\([A-Z]{2,3}\)\s*$", "", name).strip()


def _parse_price(cell: str) -> int | None:
    t = _text(cell).replace(",", "")
    m = re.search(r"\d{3,}", t)
    return int(m.group(0)) if m else None  # None = not sold / no price (NEVER impute)


def parse_lots(html: str, sale: str, year: int) -> list[dict]:
    rows: list[dict] = []
    for lot, tdh, vendor, purchaser, price in _ROW_RE.findall(html):
        names = _HN_RE.findall(tdh)
        if len(names) < 2:
            continue
        sire = _strip_country(_text(names[0]))
        dam = _strip_country(_text(names[1]))
        colsex = _COLSEX_RE.search(tdh)
        colsex_txt = _text(colsex.group(1)) if colsex else ""
        sex = _SEX_MAP.get(colsex_txt.rstrip(".")[-1:].upper(), "") if colsex_txt else ""
        rows.append({
            "house": "tattersalls", "sale": sale, "sale_year": year, "lot": lot,
            "sire": sire, "dam": dam, "colour_sex": colsex_txt, "sex": sex,
            "vendor": _text(vendor), "purchaser": _text(purchaser),
            "price_guineas": _parse_price(price),
        })
    return rows


def make_session() -> requests.Session:
    s = requests.Session()
    s.headers.update(_HEADERS)
    s.get(QUERY_URL, params={"rndval": int(time.time())}, timeout=30)  # sets cookie
    return s


def fetch_sale(session: requests.Session, sale: str, year: int) -> list[dict]:
    """Fetch one (sale, year); returns [] if that sale did not run that year."""
    yy = f"{year % 100:02d}"
    # raw query string must NOT be re-encoded (4D uses literal %09 tabs / %20)
    url = f"{LIST_URL}?30%096%09=%20{sale}%20{yy}%09by163"
    r = session.get(url, timeout=45)
    r.raise_for_status()
    m = re.search(r"(\d+) lots found", r.text)
    if not m or m.group(1) == "0":
        return []
    return parse_lots(r.text, sale, year)


def backfill(years: range, sales: list[str] | None = None, delay: float = 1.5) -> pd.DataFrame:
    sales = sales or list(SALE_CODES)
    session = make_session()
    out: list[dict] = []
    for year in years:
        for sale in sales:
            try:
                lots = fetch_sale(session, sale, year)
            except Exception as e:  # noqa: BLE001 — keep backfill resilient
                logger.warning("fetch %s %s failed: %s", sale, year, e)
                lots = []
            if lots:
                logger.info("%s %s: %d lots", sale, year, len(lots))
                out.extend(lots)
            time.sleep(delay)
    return pd.DataFrame(out)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--from-year", type=int, default=2009)
    ap.add_argument("--to-year", type=int, default=2026)
    ap.add_argument("--sales", nargs="*", default=None, help="sale codes (default: all)")
    ap.add_argument("--out", default="data/processed/tattersalls_sales.parquet")
    ap.add_argument("--delay", type=float, default=1.5)
    args = ap.parse_args()
    df = backfill(range(args.from_year, args.to_year + 1), args.sales, args.delay)
    if df.empty:
        logger.warning("no lots scraped")
        return
    df.to_parquet(args.out, index=False)
    sold = df["price_guineas"].notna().mean()
    logger.info("wrote %d lots -> %s  (%.0f%% with a price)", len(df), args.out, sold * 100)


if __name__ == "__main__":
    main()
