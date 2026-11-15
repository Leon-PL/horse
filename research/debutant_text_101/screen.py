"""Debutants: does Timeform's pre-race comment carry more than the 87-input model reads from it? (101, screen)"""
import re
import numpy as np, pandas as pd
from pathlib import Path
import research.error_anatomy.frame  # noqa: F401  (current store when the study store is gone)
from research.meta_learner import screen_candidates as sc

def main():
    from research.error_anatomy.frame import scored_frame
    f = scored_frame(scored_only=False)
    f["p_market"] = f.p_bsp.fillna(f.p_sp)
    f = f[f.p_market.notna()]
    f["debut"] = pd.to_numeric(f.prior_form_runs, errors="coerce").eq(0)
    d = sorted(Path("data/sources/rtv").glob("*/text.parquet"), key=lambda p: p.stat().st_mtime)[-1]
    t = pd.read_parquet(d, columns=["race_id", "horse_id", "tf_comment"])
    t["race_id"] = t.race_id.astype(str); t["race_id"] = t.race_id.astype(f.race_id.dtype); t["horse_id"] = t.horse_id.astype(f.horse_id.dtype)
    f = f.merge(t, on=["race_id", "horse_id"], how="left")
    deb = f[f.debut.fillna(False)].copy()
    print("debutant rows", len(deb), "with comment", round(deb.tf_comment.notna().mean(), 3))
    c = deb.tf_comment.fillna("").str.lower()
    flags = {
        "market": r"market|money|support|whisper",
        "cost": r"€|guineas|gns|cost|bought|purchased|£\d",
        "relation": r"brother|sister|sibling|half-|dam|related|family|out of",
        "breeze": r"breeze",
        "expensive": r"expensive|pricey|valuable",
        "yard_newcomers": r"newcomers|debutants|first-time-out|debut winners|stable's",
        "positive": r"should|interesting|promising|potential|well bred|worth|respect",
        "negative": r"modest|ordinary|needs time|later|bred for|will need|stamina",
        "no_comment": r"^$",
    }
    rows = []
    for name, pat in flags.items():
        m = c.str.contains(pat, regex=True)
        for lab, g in (("yes", deb[m]), ("no", deb[~m])):
            rows.append({"flag": name, "has": lab, "n": len(g), "won": g.won.sum(), "model": g.p_single.sum(),
                         "market": g.p_market.sum(),
                         "z_model": (g.won.sum()-g.p_single.sum())/np.sqrt((g.p_single*(1-g.p_single)).sum()),
                         "z_market": (g.won.sum()-g.p_market.sum())/np.sqrt((g.p_market*(1-g.p_market)).sum())})
    out = pd.DataFrame(rows)
    print(out.round(2).to_string())
    Path("data/research/debutant_text_101").mkdir(parents=True, exist_ok=True)
    out.to_csv("data/research/debutant_text_101/screen.csv", index=False)

if __name__ == "__main__":
    main()
