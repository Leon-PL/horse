"""Debutant-only comment model on top of the stored forecast (101).

Ridge regression of (won - forecast) on TF-IDF n-grams of the Timeform comment, debutants only, fitted on races before a
cut and applied after it as a multiplicative tilt exp(k * score) on debutants, renormalised within race. k is fitted on
the latest fifth of the training dates, held out of the ridge fit. Scored by race log loss on all runners in races with a debutant, against the untilted forecast.
"""
import numpy as np, pandas as pd
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from scipy.optimize import minimize_scalar
import research.error_anatomy.frame  # noqa: F401
from research.error_anatomy.frame import scored_frame

CUTS = ["2025-01-01", "2025-07-01", "2026-01-01"]

def race_ll(p, won, rid):
    d = pd.DataFrame({"p": p, "w": won, "r": rid})
    d["p"] = d.p / d.groupby("r").p.transform("sum")
    return -np.log(d.p[d.w > 0]).groupby(d.r[d.w > 0]).sum()

def main():
    f = scored_frame(scored_only=False)
    f["debut"] = pd.to_numeric(f.prior_form_runs, errors="coerce").eq(0)
    d = sorted(Path("data/sources/rtv").glob("*/text.parquet"), key=lambda p: p.stat().st_mtime)[-1]
    t = pd.read_parquet(d, columns=["race_id", "horse_id", "tf_comment"])
    t["race_id"] = t.race_id.astype(f.race_id.dtype); t["horse_id"] = t.horse_id.astype(f.horse_id.dtype)
    f = f.merge(t, on=["race_id", "horse_id"], how="left")
    f["tf_comment"] = f.tf_comment.fillna("")
    f = f[f.groupby("race_id").debut.transform("any")].reset_index(drop=True)
    print("races with a debutant", f.race_id.nunique(), "runners", len(f))
    rows = []
    for i, cut in enumerate(CUTS):
        end = CUTS[i+1] if i+1 < len(CUTS) else "2026-06-11"   # hold-out untouched
        tr = f[(f.race_date < cut)]; te = f[(f.race_date >= cut) & (f.race_date < end)]
        dates = np.sort(tr.race_date.unique()); split = dates[int(.8*len(dates))]
        cal = tr[tr.race_date >= split]; tr = tr[tr.race_date < split]
        dtr = tr[tr.debut]
        vec = TfidfVectorizer(ngram_range=(1, 2), min_df=20, sublinear_tf=True)
        x = vec.fit_transform(dtr.tf_comment)
        model = Ridge(alpha=30.).fit(x, dtr.won - dtr.p_single)
        def tilt(frame, k):
            s = np.zeros(len(frame)); m = frame.debut.to_numpy()
            s[m] = model.predict(vec.transform(frame.tf_comment[m]))
            return frame.p_single.to_numpy() * np.exp(k * s)
        # k on the latest fifth of the training dates, which the ridge did not see
        k = minimize_scalar(lambda k: race_ll(tilt(cal, k), cal.won, cal.race_id).mean(), bounds=(0, 50), method="bounded").x
        base = race_ll(te.p_single.to_numpy(), te.won, te.race_id)
        new = race_ll(tilt(te, k), te.won, te.race_id)
        diff = (new - base)
        boot = [diff.sample(frac=1, replace=True, random_state=b).mean() for b in range(500)]
        rows.append({"from": cut, "to": end, "races": len(diff), "k": round(k, 2), "diff": diff.mean(),
                     "lo": np.percentile(boot, 2.5), "hi": np.percentile(boot, 97.5)})
        terms = pd.Series(model.coef_, index=vec.get_feature_names_out()).sort_values()
        print(cut, "top +", terms.tail(8).index.tolist(), "top -", terms.head(8).index.tolist())
    out = pd.DataFrame(rows); print(out.to_string())
    out.to_csv("data/research/debutant_text_101/text_fit.csv", index=False)

if __name__ == "__main__":
    main()
