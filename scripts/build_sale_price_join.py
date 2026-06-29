"""Join scraped Tattersalls sale lots to our horses -> data/processed/horse_sale_price.parquet.

Lots sell unnamed (sire x dam); foaling year is implied by sale type
(yearling=sale_year-1, foal=sale_year, 2yo breeze-up=sale_year-2). Join to
horse_pedigree on (sire, dam, foaling_year) — no horse-name normalisation needed.
Only deterministic-age sale types are joined; mixed/HIT/mares omitted (ambiguous age).

Verdict (scripts/test_sale_price_signal.py, 2026-06-29): the resulting feature is
fully priced by the market (residual corr 0.0000) — see memory sale-price-fully-priced.
Kept for reproducibility; the price FEATURE is not used in the model.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd
from src.database import _get_conn
from src.tattersalls_sales_scraper import SALE_CODES

SALES_PARQUET = "data/processed/tattersalls_sales.parquet"
OUT = "data/processed/horse_sale_price.parquet"
AGE_OFFSET = {"yearling": 1, "foal": 0, "2yo": 2}  # foaling_year = sale_year - offset


def _norm(s: pd.Series) -> pd.Series:
    return s.astype(str).str.lower().str.strip()


def main():
    sales = pd.read_parquet(SALES_PARQUET)
    with _get_conn() as c:
        ped = pd.read_sql("SELECT horse_id, sire, dam, foaled FROM horse_pedigree", c)

    sales["age_off"] = sales["sale"].map(lambda s: AGE_OFFSET.get(SALE_CODES.get(s, ("", ""))[1]))
    det = sales[sales["age_off"].notna() & sales["price_guineas"].notna()].copy()
    det["foaled_year"] = det["sale_year"] - det["age_off"].astype(int)

    for d in (det, ped):
        d["k_sire"] = _norm(d["sire"]); d["k_dam"] = _norm(d["dam"])
    ped["foaled_year"] = pd.to_datetime(ped["foaled"], errors="coerce").dt.year

    key = ["k_sire", "k_dam", "foaled_year"]
    # multiple lots (re-offered) -> take the headline (max) price for that animal
    agg = det.groupby(key)["price_guineas"].max().reset_index()
    j = ped.merge(agg, on=key, how="left").rename(columns={"price_guineas": "sale_price_guineas"})

    cov = j["sale_price_guineas"].notna().mean()
    print(f"pedigree horses {len(ped):,}  matched {j['sale_price_guineas'].notna().sum():,} ({cov*100:.1f}%)")
    j[["horse_id", "sale_price_guineas"]].to_parquet(OUT, index=False)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
