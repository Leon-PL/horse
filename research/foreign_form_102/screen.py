"""Hidden form from French/UAE runs: does the stored forecast miss runners with recent foreign runs? (102, early screen)

Uses the foreign results scraped so far (data/foreign_form/races_v2.parquet, from June 2025), so only races from
2025-09-01 on, when a 90-day look-back is covered. A foreign run counts only once it is available
(``available_date`` before the race).
"""
import numpy as np, pandas as pd
from pathlib import Path
import research.error_anatomy.frame  # noqa: F401
from research.error_anatomy.frame import scored_frame

START = "2025-09-01"

def z(g, col):
    return (g.won.sum() - g[col].sum()) / np.sqrt((g[col] * (1 - g[col])).sum())

def main():
    f = scored_frame(scored_only=False)
    f = f[(f.race_date >= START) & (f.race_date < "2026-06-11")].copy()
    f["p_market"] = f.p_bsp.fillna(f.p_sp)
    f = f[f.p_market.notna()]
    fo = pd.read_parquet("data/foreign_form/races_v2.parquet")
    fo["available_date"] = pd.to_datetime(fo.available_date); fo["race_date"] = pd.to_datetime(fo.race_date)
    fo["horse_id"] = pd.to_numeric(fo.horse_id).astype(f.horse_id.dtype)
    pos = pd.to_numeric(fo.finish_position, errors="coerce"); n = pd.to_numeric(fo.num_runners, errors="coerce")
    fo["pct"] = (1 - (pos - 1) / (n - 1).clip(lower=1)).where(pos >= 1, 0.).clip(0, 1)
    fo = fo.sort_values("available_date")
    left = f[["race_id", "horse_id", "race_date"]].sort_values("race_date")
    # Latest foreign run available before the race.
    m = pd.merge_asof(left, fo[["horse_id", "available_date", "race_date", "pct", "country"]].rename(
        columns={"race_date": "f_date"}), left_on="race_date", right_on="available_date", by="horse_id",
        allow_exact_matches=False, direction="backward")
    f = f.merge(m[["race_id", "horse_id", "f_date", "pct", "country"]], on=["race_id", "horse_id"], how="left")
    f["f_days"] = (f.race_date - f.f_date).dt.days
    # Was the foreign run the most recent run? Previous domestic date from the store's own rows.
    from research.meta_learner import screen_candidates as sc
    dom = sc.read_store([]).drop_duplicates(["horse_id", "race_date"]).sort_values(["horse_id", "race_date"])
    dom["prev_dom"] = dom.groupby("horse_id").race_date.shift()
    dom["horse_id"] = dom.horse_id.astype(f.horse_id.dtype)
    f = f.merge(dom[["horse_id", "race_date", "prev_dom"]], on=["horse_id", "race_date"], how="left")
    f["last_was_abroad"] = f.f_date.gt(f.prev_dom) | (f.f_date.notna() & f.prev_dom.isna())
    rows = []
    def add(name, g):
        rows.append({"group": name, "n": len(g), "won": int(g.won.sum()), "model": g.p_single.sum(),
                     "market": g.p_market.sum(), "z_model": z(g, "p_single"), "z_market": z(g, "p_market")})
    add("all", f)
    add("foreign run within 365d", f[f.f_days.le(365)])
    add("last run abroad", f[f.last_was_abroad])
    lw = f[f.last_was_abroad]
    add("last run abroad, top third there", lw[lw.pct >= 2/3])
    add("last run abroad, bottom third there", lw[lw.pct < 1/3])
    add("last run abroad, debut here", lw[pd.to_numeric(lw.prior_form_runs, errors="coerce").eq(0)])
    out = pd.DataFrame(rows); print(out.round(2).to_string())
    Path("data/research/foreign_form_102").mkdir(parents=True, exist_ok=True)
    out.to_csv("data/research/foreign_form_102/screen.csv", index=False)

if __name__ == "__main__":
    main()
