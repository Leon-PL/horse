"""Find WHICH feature still leaks same-race stablemate results on the rebuilt store.

For every numeric feature, restrict to multi-runner-trainer groups (same race,
same trainer, >=2 runners) and measure the within-group correlation between the
feature value and `won`. A race-safe pre-race feature must NOT know who wins the
current race, so within a (race, trainer) block it should be either constant or
uncorrelated with the outcome. A leaking feature (current-race result bleeding in)
will show a strong positive within-group corr with `won`.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd, pyarrow.parquet as pq
import config
from src.feature_store import FeatureStore

s = FeatureStore()
have = set(pq.read_schema(s.featured_path).names)
allow = [l.strip() for l in open(config.FEATURE_ALLOWLIST, encoding="utf-8")
         if l.strip() and not l.lstrip().startswith("#")]
cols = [c for c in allow if c in have]
need = ["race_id", "trainer", "won"]
df = pd.read_parquet(s.featured_path, columns=sorted(set(cols) | set(need)))
df["won"] = pd.to_numeric(df["won"], errors="coerce").fillna(0).astype(int)

# multi-runner-trainer blocks only
g = df.groupby(["race_id", "trainer"])
df["_n"] = g["won"].transform("size")
multi = df[df["_n"] >= 2].copy()
print(f"multi-runner-trainer rows: {len(multi):,} in {multi.groupby(['race_id','trainer']).ngroups:,} blocks")

# de-mean each feature & won WITHIN (race, trainer) block, then corr over residuals.
gm = multi.groupby(["race_id", "trainer"])
won_res = multi["won"] - gm["won"].transform("mean")
feat_cols = [c for c in cols if c in multi.columns and multi[c].dtype.kind in "fiu"]
rows = []
won_var = float((won_res ** 2).sum())
for c in feat_cols:
    x = pd.to_numeric(multi[c], errors="coerce")
    xr = x - gm[c].transform("mean")
    m = xr.notna() & won_res.notna()
    denom = np.sqrt(float((xr[m] ** 2).sum()) * float((won_res[m] ** 2).sum()))
    if denom < 1e-9:
        continue
    corr = float((xr[m] * won_res[m]).sum()) / denom
    # fraction of blocks where the feature is non-constant (varies among stablemates)
    nonconst = float((gm[c].transform("std").fillna(0) > 1e-9).mean())
    rows.append((c, corr, nonconst))

res = pd.DataFrame(rows, columns=["feature", "within_block_corr_won", "frac_blocks_nonconstant"])
res = res.reindex(res["within_block_corr_won"].abs().sort_values(ascending=False).index)
pd.set_option("display.max_rows", 60); pd.set_option("display.width", 120)
print("\n=== features most correlated with same-race `won` WITHIN (race,trainer) blocks ===")
print("(high |corr| + nonconstant => the feature changes between stablemates in step with who wins => LEAK)\n")
print(res.head(30).to_string(index=False))
