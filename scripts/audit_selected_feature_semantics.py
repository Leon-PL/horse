"""Audit the selected model inputs and their source semantics, without fitting.

Reads a projected, immutable candidate store. No market-residual gate: an
independent forecast may benefit from information already reflected in prices.
Outcome columns are used for settled-history validity only, never for ranking
features by results in reserved periods.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
import pyarrow.parquet as pq


def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as stream:
        for part in iter(lambda: stream.read(1024*1024),b""):
            h.update(part)
    return h.hexdigest()


def definitions(features):
    assigned={}
    functions={}
    for relative in ("src/data_processor.py","src/feature_engineer.py","src/ratings.py","src/ingestion/sporting_life.py"):
        tree=ast.parse((ROOT/relative).read_text(encoding="utf-8"))
        for function in tree.body:
            if not isinstance(function,(ast.FunctionDef,ast.AsyncFunctionDef)):
                continue
            functions[function.name]=f"{relative}:{function.lineno} ({function.name}, dynamic output)"
            for node in ast.walk(function):
                if not isinstance(node,ast.Assign):
                    continue
                for target in node.targets:
                    if (isinstance(target,ast.Subscript) and isinstance(target.slice,ast.Constant)
                            and isinstance(target.slice.value,str)):
                        assigned.setdefault(target.slice.value,[]).append(f"{relative}:{node.lineno} ({function.name})")
    out={}
    dynamic={"num_runners":"raw results/card metadata in src/ingestion/sporting_life.py:_ride_to_dict",
        "race_type_Chase":functions["encode_categorical"],"sex_Female":functions["encode_categorical"],
        "horse_avg_pos_3":functions["add_horse_features"],"horse_best_pos_5":functions["add_horse_features"],
        "jockey_avg_pos_20":functions["add_jockey_features"],"trainer_freq":functions["encode_categorical"],
        "speed_z_class_best_5":functions["add_speed_figure_features"],
        "speed_fig_best_5":functions["add_speed_figure_features"],
        "speed_fig_adj_best_5":functions["add_speed_figure_features"],
        "connections_relative_score":functions["add_race_relative_signal_features"],
        "freshness_relative_score":functions["add_race_relative_signal_features"]}
    suffixes=("_abs_rank_pct_in_race","_vs_field","_vs_best_in_race","_z_in_race","_rank_pct_in_race")
    for feature in features:
        base=feature
        while any(base.endswith(s) for s in suffixes):
            base=next(base[:-len(s)] for s in suffixes if base.endswith(s))
        locations=assigned.get(feature,assigned.get(base,[]))
        if not locations and base in dynamic:
            locations=[dynamic[base]]
        if not locations:
            if base.startswith(("sire_","damsire_","dam_")):
                locations=["src/feature_engineer.py:add_pedigree_features (dynamic columns)"]
            elif any(t in base for t in ("elo","glicko","horse_ts")):
                locations=["src/ratings.py (dynamic rating columns)"]
            elif "_te_" in base:
                locations=["src/feature_engineer.py:add_target_encoded_features (dynamic columns)"]
        out[feature]={"base_signal":base,"source_locations":"; ".join(locations),
            "relative_transform":feature!=base,"definition_located":bool(locations)}
    return out


def profile(frame,features,output):
    sampled=frame.iloc[np.linspace(0,len(frame)-1,min(len(frame),50_000),dtype=int)]
    lineage=definitions(features)
    rows=[]
    for name in features:
        s=frame[name]; a=s.to_numpy(); finite=np.isfinite(a)
        lo=float(a[finite].min()) if finite.any() else None
        hi=float(a[finite].max()) if finite.any() else None
        quant=np.nanquantile(sampled[name],[.01,.5,.99]) if sampled[name].notna().any() else [np.nan]*3
        rows.append({"feature":name,"rows":len(s),"known_rows":int(s.notna().sum()),
            "missing_fraction":float(s.isna().mean()),"distinct_values":int(s.nunique()),
            "infinite_rows":int(np.isinf(a).sum()),"min":lo,"max":hi,
            "constant_when_known":lo is not None and lo==hi,
            "sample_p01":quant[0],"sample_median":quant[1],"sample_p99":quant[2],**lineage[name]})
    report=pd.DataFrame(rows)
    monthly=[]; typed=[]; context={}
    years=frame.race_date.dt.year
    for start in range(0,len(features),16):
        cols=features[start:start+16]
        miss=frame[cols].isna()
        monthly.append(miss.groupby(years).mean().T)
        typed.append(miss.groupby(frame.race_type).mean().T)
        low=frame[cols].groupby(frame.race_id,sort=False).min()
        high=frame[cols].groupby(frame.race_id,sort=False).max()
        for col in cols:
            known=low[col].notna() & high[col].notna()
            context[col]=float(low.loc[known,col].eq(high.loc[known,col]).mean()) if known.any() else np.nan
    report["within_field_constant_fraction"]=report.feature.map(context)
    report.to_csv(output/"selected_input_profiles.csv",index=False)
    pd.concat(monthly).to_csv(output/"missing_by_year.csv",index_label="feature")
    pd.concat(typed).to_csv(output/"missing_by_race_type.csv",index_label="feature")
    correlations=sampled[features].corr()
    aliases=[]
    for i,left in enumerate(features):
        for j in range(i+1,len(features)):
            right=features[j]; corr=correlations.iloc[i,j]
            if not np.isfinite(corr) or abs(corr)<.9999999:
                continue
            a,b=frame[left].to_numpy(),frame[right].to_numpy()
            same_mask=np.array_equal(np.isnan(a),np.isnan(b))
            finite=np.isfinite(a)&np.isfinite(b)
            error=float(np.max(np.abs(a[finite]-b[finite]))) if finite.any() else None
            inverse_error=float(np.max(np.abs(a[finite]+b[finite]))) if finite.any() else None
            aliases.append({"left":left,"right":right,"sample_correlation":float(corr),
                "same_missing_mask":same_mask,"max_absolute_difference":error,
                "max_inverse_difference":inverse_error,
                "exact_at_model_precision":same_mask and error==0})
    (output/"aliases.json").write_text(json.dumps(aliases,indent=2))
    return report,aliases


def semantics(store,frame,output):
    names=pq.read_schema(store/"processed.parquet").names
    wanted=["race_id","race_date","race_name","horse_name","horse_id","surface","sire","damsire",
        "race_class","race_type","track","region","weight_lbs","prize_money","handicap",
        "official_rating","has_official_rating","distance_furlongs","num_runners"]
    raw=pd.read_parquet(store/"processed.parquet",columns=[c for c in wanted if c in names])
    assert raw[["race_id","horse_name"]].equals(frame[["race_id","horse_name"]])
    surface=raw.surface.fillna("").str.lower().str.replace(r"[^a-z]","",regex=True)
    synthetic=surface.isin(["aw","allweather","polytrack","tapeta","fibresand","fibersand"])
    old_mask=raw.surface.fillna("Turf").str.lower().str.contains("aw|all weather|all-weather")
    lost=synthetic&~old_mask
    surf=raw.groupby("surface",dropna=False).agg(rows=("race_id","size"),races=("race_id","nunique"))
    surf.to_csv(output/"source_surfaces.csv")
    invalid_weight=raw.weight_lbs.notna() & ~raw.weight_lbs.between(50,220)
    raw.loc[invalid_weight].to_csv(output/"implausible_weights.csv",index=False)
    class_num=pd.to_numeric(raw.race_class.astype(str).str.extract(r"(\d+)",expand=False),errors="coerce")
    ordered=raw[["horse_id","race_date","race_id"]].copy()
    ordered["class_known"]=class_num.between(1,7)
    ordered=ordered.sort_values(["horse_id","race_date","race_id"],kind="stable")
    ordered["previous_class_known"]=ordered.groupby("horse_id").class_known.shift()
    crossing=ordered.previous_class_known.notna() & ordered.class_known.ne(ordered.previous_class_known)
    imputed=raw.has_official_rating.eq(0)&raw.official_rating.notna()
    for c in ("handicap","prize_money","distance_furlongs","surface","race_class"):
        if c in raw:
            count=raw.groupby("race_id")[c].nunique(dropna=False)
            if count.gt(1).any():
                raw.loc[raw.race_id.isin(count[count.gt(1)].index)].head(100).to_csv(output/f"inconsistent_field_{c}.csv",index=False)
    actual=raw.groupby("race_id").size()
    expected=raw.groupby("race_id").num_runners.first()
    result={"all_weather": {"synthetic_rows":int(synthetic.sum()),"synthetic_races":int(raw.loc[synthetic,"race_id"].nunique()),
        "rows_recognised_by_current_pedigree_mask":int((old_mask&synthetic).sum()),
        "missed_rows":int(lost.sum()),"natural_sand_rows_kept_separate":int(surface.eq("sand").sum())},
        "implausible_weight_rows":int(invalid_weight.sum()),
        "missing_class_rows_encoded_as_zero_in_class_change":int(class_num.isna().sum()),
        "known_unknown_class_transitions":int(crossing.sum()),
        "unrated_rows_with_median_imputed_rating":int(imputed.sum()),
        "constant_region_metadata":raw.region.nunique(dropna=False)==1,
        "region_values":raw.region.value_counts(dropna=False).to_dict(),
        "mismatched_declared_fields":int(actual.ne(expected).sum()),
        "small_field_base_place_semantics": "Feature uses everyone-places for n<=4; native place labels use top3. Descriptive inconsistency, not established predictive harm.",
        "weight_feature_semantics": "Venue-distance and weight-per-furlong inputs are proxies; weight-per-furlong decreases with trip despite its comment claiming weight matters more at longer distances."}
    if "sire_aw_edge" in frame:
        # Expected false identity if the AW numerator has never been updated.
        source=pd.read_parquet(store/"featured.parquet",columns=["sire_aw_win_rate_shrunk"])
        result["all_weather"]["conditional_rate_known_unique"]=int(source.sire_aw_win_rate_shrunk.nunique())
        result["all_weather"]["maximum_error_from_false_identity"] = float((frame.sire_aw_edge-(.1-frame.sire_win_rate_shrunk)).abs().max())
    (output/"semantic_findings.json").write_text(json.dumps(result,indent=2,default=lambda x:x.item()))
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store",default="data/audit/model_improvement/015_nonfinisher_history/candidate_store")
    parser.add_argument("--registration",default="data/audit/model_improvement/015_nonfinisher_history/registration.json")
    parser.add_argument("--output",default="data/audit/feature_review_017")
    args=parser.parse_args(); store=Path(args.store);output=Path(args.output);output.mkdir(exist_ok=True)
    features=json.loads(Path(args.registration).read_text())["candidate_features"]
    columns=list(dict.fromkeys([*features,"race_id","horse_name","race_date","race_type"]))
    frame=pd.read_parquet(store/"featured.parquet",columns=columns)
    frame[features]=frame[features].astype(np.float32)
    report,aliases=profile(frame,features,output)
    findings=semantics(store,frame,output)
    summary={"rows":len(frame),"races":frame.race_id.nunique(),"selected_inputs":len(features),
        "constant_selected":report.loc[report.constant_when_known,"feature"].tolist(),
        "entirely_missing_selected":report.loc[report.known_rows.eq(0),"feature"].tolist(),
        "infinite_selected":report.loc[report.infinite_rows.gt(0),"feature"].tolist(),
        "exact_alias_pairs":[a for a in aliases if a["exact_at_model_precision"]],
        "findings":findings,"source_store_meta_sha256":sha(store/"meta.json"),
        "audit_source_sha256":sha(Path(__file__)),
        "scope":"Full input validity audit; no feature-selection gate based on reserved outcomes; missing values/context-only features are not automatic deletion recommendations"}
    (output/"summary.json").write_text(json.dumps(summary,indent=2,default=lambda x:x.item()))
    print(json.dumps(summary,indent=2,default=lambda x:x.item()))


if __name__=="__main__":
    main()
