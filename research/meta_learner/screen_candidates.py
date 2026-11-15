"""Cheap screens of the meta-learner plan's candidates on stored forecasts.

    python -m research.meta_learner.screen_candidates validate
    python -m research.meta_learner.screen_candidates candidates [--only NAME ...]

Every screen leaves the stored full 063 forecasts fixed and learns only a small
correction from the candidate's inputs, on earlier quarters. Two instruments:

- Tier 0, a race conditional logit with the reference as offset and a few
  step coefficients per input.
- Tier 1, three heavily regularised LightGBM corrections averaged, compared
  with the same fit on a twin whose candidate inputs were moved to random rows
  of the same quarter. The twin measures what the fit gains or loses from
  capacity alone.

``validate`` replays known results: pooled performance/spread, explicit text
and handicap leeway forecasts, and the 64 inputs full 063 added to the original
129. Results go to ``data/research/meta_learner/screens_20261109``.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from research.meta_learner import screening as s
from research.meta_learner.candidate_features import relatives_form, same_day_bias
from research.meta_learner.ability import (ability_in_pounds, ability_in_pounds_jumps, bounce_inputs,
                                          ground_from_withdrawals, improvement_inputs, jumps_margin_inputs,
                                          relatives_ratings)
from src.feature_semantics import CLAIM_ADJUSTED_WEIGHT_INPUTS
from src.ingestion.storage import atomic_json, atomic_parquet

OUT = Path("data/research/meta_learner/screens_20261109")
KEYS = s.KEYS


def registration():
    return json.loads((s.STUDY / "registration.json").read_text())


def spec_features(name):
    return next(spec["features"] for spec in registration()["specs"] if spec["id"] == name)


def store_file():
    return Path(registration()["receipt"]["store_root"]) / "featured.parquet"


def read_store(columns, start=None):
    filters = [("race_date", ">=", pd.Timestamp(start))] if start is not None else None
    frame = pd.read_parquet(store_file(), columns=list(dict.fromkeys([*KEYS, "race_date", *columns])), filters=filters)
    frame.attrs = {}
    frame["race_id"] = frame.race_id.astype(str)  # the stored forecasts key races by string
    return frame


def read_input(columns, start):
    frame = pd.read_parquet(s.STUDY / "input.parquet", columns=list(dict.fromkeys([*KEYS, *columns])),
                            filters=[("race_date", ">=", pd.Timestamp(start))])
    frame.attrs = {}
    return frame


def base_frame(original=False):
    return s.screening_frame(original=original)


# --------------------------------------------------------------------------
# Candidate inputs


def allowed(columns):
    """Drop the columns every model must exclude (identifiers, prices, outcomes, unsafe legacy composites)."""
    from src.model import EXCLUDE_COLUMNS
    blocked = set(EXCLUDE_COLUMNS)
    return [c for c in columns if c not in blocked]


def tracking_columns():
    full = set(spec_features("full-063"))
    import pyarrow.parquet as pq
    names = pq.read_schema(store_file()).names
    return allowed([n for n in names if n.startswith(("rtv_", "atr_")) and n not in full
                    and not n.startswith(("rtv_text", "rtv_card", "rtv_pre", "rtv_verdict", "atr_cmt"))
                    and "comment" not in n])


def store_candidate(columns):
    def load(frame):
        kept = allowed(columns)
        return read_store(kept, frame.race_date.min()).drop(columns="race_date"), kept
    return load


def computed_candidate(function, columns_needed):
    def load(frame):
        history = read_store(columns_needed)  # all dates: features need earlier history
        values = function(history)
        values = values.loc[values.race_id.isin(frame.race_id.unique())]
        return values, [c for c in values if c not in KEYS]
    return load


def pedigree_columns():
    full = set(spec_features("full-063"))
    import pyarrow.parquet as pq
    names = pq.read_schema(store_file()).names
    return allowed([n for n in names if n.startswith(("sire_", "dam_", "damsire_")) and n not in full])


ABILITY_INPUTS = ["off_time", "race_type", "distance_furlongs", "finish_position", "beaten_lengths",
                  *CLAIM_ADJUSTED_WEIGHT_INPUTS, "official_rating", "age", "trainer", "dam", "damsire"]


def going_store_columns():
    """BHA going and withdrawal columns in the store; full 063 uses no BHA column."""
    import pyarrow.parquet as pq
    full = set(spec_features("full-063"))
    names = pq.read_schema(store_file()).names
    return allowed([n for n in names if n.startswith("bha_") and n not in full
                    and any(k in n for k in ("going", "softer", "faster", "withdraw"))])


def with_store(load_computed, columns):
    def load(frame):
        values, names = load_computed(frame)
        stored = read_store(columns, start=frame.race_date.min())
        stored = stored.loc[stored.race_id.isin(frame.race_id.unique())].drop(columns="race_date")
        values["horse_id"] = values.horse_id.astype(stored.horse_id.dtype)
        values = values.merge(stored, on=KEYS, how="outer", validate="one_to_one")
        return values, [*names, *columns]
    return load


def completion_load(frame):
    from research.meta_learner.forecast_components import completion_candidate
    return completion_candidate(frame)


CANDIDATES = {
    "first-string": lambda: store_candidate([
        "trainer_first_string_by_or", "trainer_first_string_by_elo", "trainer_first_string_by_odds",
        "trainer_first_string_odds_or_agree", "trainer_first_string_odds_elo_agree",
        "trainer_first_string_stable_jockey", "trainer_first_string_by_jockey"]),
    # The odds-ranked columns use `odds`, a market-derived input (src/model.py), so this variant drops them.
    "first-string-no-odds": lambda: store_candidate([
        "trainer_first_string_by_or", "trainer_first_string_by_elo",
        "trainer_first_string_stable_jockey", "trainer_first_string_by_jockey"]),
    "pedigree-store": lambda: store_candidate(pedigree_columns()),
    "relatives-form": lambda: computed_candidate(relatives_form, [
        "dam", "damsire", "won", "finish_position", "num_runners", "official_rating"]),
    "same-day-bias": lambda: computed_candidate(same_day_bias, [
        "track", "off_time", "race_type", "num_runners", "finish_position", "draw", "pace_style_index"]),
    "tracking": lambda: store_candidate(tracking_columns()),
    # New information (plan step 1): ability in pounds, relatives in rating points, going reasons.
    "ability-lb": lambda: computed_candidate(ability_in_pounds, ABILITY_INPUTS),
    # Exploratory: the jumps-only version, chosen after the all-races ability screen.
    "ability-lb-jumps": lambda: computed_candidate(ability_in_pounds_jumps, ABILITY_INPUTS),
    "relatives-ratings": lambda: computed_candidate(relatives_ratings, ABILITY_INPUTS),
    # The jumps margin specialist candidate: ability and bounce inputs, jumps races only.
    "jumps-margin": lambda: computed_candidate(jumps_margin_inputs, ABILITY_INPUTS),
    "going-reasons": lambda: with_store(computed_candidate(ground_from_withdrawals, ["track", "off_time", "going"]),
                                        going_store_columns()),
    # Plan components S13-S15 (registered in forecast_components.REGISTRATION before scoring).
    "completion": lambda: completion_load,
    "improvement": lambda: computed_candidate(improvement_inputs, [*ABILITY_INPUTS, "sire"]),
    "bounce": lambda: computed_candidate(bounce_inputs, ABILITY_INPUTS),
    # Only the new comparisons of today's going with withdrawn goings, without the store's BHA columns.
    "going-withdrawn": lambda: computed_candidate(ground_from_withdrawals, ["track", "off_time", "going"]),
}


# --------------------------------------------------------------------------
# Runs

# Chosen on the validation screens below, then fixed for every candidate.
TIER1 = {"params": {"learning_rate": .02, "num_leaves": 3, "min_data_in_leaf": 5000, "lambda_l2": 200.,
                    "bagging_fraction": .8, "bagging_freq": 1},
         "boost": {"rounds": 100, "stopping": None, "seeds": (7, 8, 9)}}
TWIN_SEED = 11
WIDE = 40  # Tier 0 uses three bins and heavier shrinkage above this many inputs


def tier0_options(columns):
    return {"bins": 3, "l2": .1} if len(columns) > WIDE else {"bins": 5, "l2": 1e-3}


def provenance():
    files = [Path(s.__file__), Path(__file__), Path(__file__).with_name("candidate_features.py")]
    return {"code": {f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in files},
            "study_input_sha256": registration()["input_sha256"],
            "store": str(store_file()), "store_fe_signature": registration()["receipt"]["fe_signature"],
            "scored_quarters": list(s.SCORED), "purge_days": s.PURGE_DAYS, "tier1": TIER1, "twin_seed": TWIN_SEED}


def save(name, result, races):
    OUT.mkdir(parents=True, exist_ok=True)
    atomic_json(OUT / f"{name}.json", {**result, "provenance": provenance()})
    atomic_parquet(races, OUT / f"{name}_races.parquet")


def screen(frame, name, columns, *, tier, bases, base="p_single", threads=3, **options):
    if tier == 1:
        options = {**TIER1, **options}
    result, races = s.run(frame, columns, tier=tier, base=base, bases=bases, coverage=columns, threads=threads,
                          **options)
    save(name, result, races)
    print(s.summary_line(name, result, bases), flush=True)
    return result, races


def compact(result, bases):
    keep = ("mean", "ci95", "per_quarter", "quarters_better", "covered_races", "covered_change")
    return {b: {k: result[b][k] for k in keep if k in result[b]} for b in bases}


def assess(frame, name, columns, *, bases, threads):
    """Tier 0, Tier 1 and the Tier 1 twin for one input group."""
    entry = {"inputs": len(columns), "columns": columns,
             "rows_with_any_input": float(frame[columns].notna().any(axis=1).mean())}
    result, _ = screen(frame, f"{name}-t0", columns, tier=0, bases=bases, **tier0_options(columns))
    entry["tier0"] = compact(result, bases)
    result, races = screen(frame, f"{name}-t1", columns, tier=1, bases=bases, threads=threads)
    entry["tier1"] = compact(result, bases)
    entry["tier1_top_inputs"] = {f["fold"]: f["top_inputs"] for f in result["fits"]}
    twin = s.permute_within(frame, columns, seed=TWIN_SEED)
    result, twin_races = screen(twin, f"{name}-t1-twin", columns, tier=1, bases=bases, threads=threads)
    entry["tier1_twin"] = compact(result, bases)
    entry["tier1_vs_twin"] = s.versus(races, twin_races, bases)
    for base in bases:
        r = entry["tier1_vs_twin"][base]
        print(f"   {name} T1 vs twin [{base[2:]}]: {r['mean']:+.5f} [{r['ci95'][0]:+.5f}, {r['ci95'][1]:+.5f}] "
              f"{r['quarters_better']}/4", flush=True)
    entry["verdict"] = verdict(entry, bases[-1])
    atomic_json(OUT / f"{name}_summary.json", entry)
    return entry


def verdict(entry, base):
    """Pre-declared rule: advance on a clear Tier 0 gain, or a Tier 1 gain over the twin in 3+ quarters."""
    t0, t1 = entry["tier0"][base], entry["tier1_vs_twin"][base]
    if t0["ci95"][1] < 0 or (t1["mean"] <= -0.0003 and t1["quarters_better"] >= 3):
        return "advance"
    if t0["ci95"][0] > -0.0003 and t1["ci95"][0] > -0.0003:
        return "reject: any gain is under 0.0003"
    return "unresolved"


KNOWN = {"performance": "pooled with single full 063: -0.001142 [-0.001646, -0.000598]",
         "text": "pooled with the three-seed average: +0.000085 [-0.000060, +0.000220]",
         "leeway": "pooled with the three-seed average: +0.000032 [-0.000107, +0.000166]",
         "x063": "full 063 refit minus original: -0.004909 [-0.007048, -0.002654]; includes the 0.75 market-share "
                 "target, so a residual screen of the inputs alone should recover only part"}


def component_frame(frame, name):
    if name == "performance":
        part = s.cached_forecasts(s.STUDY, "performance")
    else:
        part = pd.read_parquet(s.STUDY / f"{dict(text='text_oof', leeway='leeway-screen_oof')[name]}.parquet")
        part.attrs = {}
        part = part.loc[part.available]
    out = s.attach(frame, part[[*KEYS, "probability"]].rename(columns={"probability": f"p_{name}"}))
    out[f"{name}_log_ratio"] = np.log(np.maximum(out[f"p_{name}"], s.TINY)) - np.log(out.p_single)
    return out.drop(columns=f"p_{name}")


def validate(threads):
    report = {"known": KNOWN}
    bases = ("p_single", "p_average")
    frame = base_frame()
    for name in ("performance", "text", "leeway"):
        report[name] = assess(component_frame(frame, name), f"validate-{name}", [f"{name}_log_ratio"],
                              bases=bases, threads=threads)
    frame = base_frame(original=True)
    extras = [c for c in spec_features("full-063") if c not in set(spec_features("generalist"))]
    frame = s.attach(frame, read_input(extras, frame.race_date.min()))
    report["x063"] = assess(frame, "validate-x063", extras, bases=("p_original",), threads=threads)
    atomic_json(OUT / "validation.json", report)


def candidates(only, threads):
    bases = ("p_single", "p_average")
    frame = base_frame()
    summary = {}
    for name in only or CANDIDATES:
        values, columns = CANDIDATES[name]()(frame)
        summary[name] = assess(s.attach(frame, values), name, columns, bases=bases, threads=threads)
        print(f"   {name}: {summary[name]['verdict']}", flush=True)
    return summary


FOLLOWUP_TWINS = (11, 12, 13)
POOL_BASES = ("p_single", "p_average", "p_pool")
STACK = ("tracking", "same-day-bias")  # the survivors that held against three twins


def pool_frame():
    """Screening frame plus the average-plus-performance pool written by ``pool_average.py``."""
    frame = base_frame()
    pool = pd.read_parquet(OUT / "pool_average_performance.parquet", columns=[*KEYS, "probability"])
    frame = s.attach(frame, pool.rename(columns={"probability": "p_pool"}))
    if frame.loc[frame.quarter.isin(s.SCORED), "p_pool"].isna().any():
        raise ValueError("The pool forecast misses scored runners")
    return frame


def mean_of(tables, bases):
    """Per-race losses averaged over several screens of the same races."""
    out = tables[0].copy()
    for table in tables[1:]:
        if not table.race_id.equals(out.race_id):
            raise ValueError("Screens scored different races")
    for base in bases:
        out[f"{base}_after"] = np.mean([t[f"{base}_after"].to_numpy() for t in tables], axis=0)
    return out


def followup(threads, only=None):
    """Survivors against three twins each, then together, also on the average-plus-performance pool.

    With ``only``, just the named groups (written to ``followup_<names>.json``), with no combined group
    when there is one name.

    The pool base comes from ``pool_average.py``: the three-seed average log-pooled with
    performance/spread, weights fitted on earlier quarters only. A correction scored on it is the
    first full stack: generalist average, a pooled component and residual specialists.
    """
    bases = POOL_BASES
    frame = pool_frame()
    survivors = list(only) if only else [
        n for n in CANDIDATES if json.loads((OUT / f"{n}_summary.json").read_text())["verdict"] == "advance"]
    target = OUT / ("followup.json" if not only else f"followup_{'_'.join(only)}.json")
    groups = {}
    for name in survivors:
        values, columns = CANDIDATES[name]()(frame)
        frame = s.attach(frame, values)
        groups[name] = columns
    if len(survivors) > 1:
        groups["combined"] = [c for name in survivors for c in groups[name]]
    report = {"survivors": survivors, "twin_seeds": list(FOLLOWUP_TWINS), "pool_base": "pool_average_performance"}
    for name, columns in groups.items():
        result, races = screen(frame, f"followup-{name}", columns, tier=1, bases=bases, threads=threads)
        twins = []
        for seed in FOLLOWUP_TWINS:
            twin = s.permute_within(frame, columns, seed=seed)
            twins.append(screen(twin, f"followup-{name}-twin{seed}", columns, tier=1, bases=bases,
                                threads=threads)[1])
        entry = {"inputs": len(columns), "vs_reference": compact(result, bases),
                 "vs_each_twin": [s.versus(races, t, bases) for t in twins],
                 "vs_mean_twin": s.versus(races, mean_of(twins, bases), bases)}
        for base in bases:
            r = entry["vs_mean_twin"][base]
            each = [round(t[base]["mean"], 5) for t in entry["vs_each_twin"]]
            print(f"   {name} vs mean of twins [{base[2:]}]: {r['mean']:+.5f} [{r['ci95'][0]:+.5f}, "
                  f"{r['ci95'][1]:+.5f}] {r['quarters_better']}/4; each twin {each}", flush=True)
        report[name] = entry
        atomic_json(target, report)


def stack(threads):
    """The stack recipe: three-seed average, pooled with performance/spread, plus one residual from STACK."""
    frame = pool_frame()
    columns = []
    for name in STACK:
        values, group = CANDIDATES[name]()(frame)
        frame = s.attach(frame, values)
        columns += group
    result, races = screen(frame, "stack", columns, tier=1, bases=POOL_BASES, threads=threads)
    arms = {"pool": "p_pool_before", "pool_plus_residual": "p_pool_after", "average_plus_residual": "p_average_after"}
    report = {"groups": list(STACK), "inputs": len(columns),
              "vs_average": {arm: s.paired(races[col], races.p_average_before, races) for arm, col in arms.items()},
              "vs_single": {arm: s.paired(races[col], races.p_single_before, races) for arm, col in arms.items()},
              "evaluation_status": result["evaluation_status"], "provenance": provenance()}
    for arm in arms:
        r = report["vs_average"][arm]
        print(f"   {arm:<22} vs average: {r['mean']:+.5f} [{r['ci95'][0]:+.5f}, {r['ci95'][1]:+.5f}] "
              f"{r['quarters_better']}/4", flush=True)
    atomic_json(OUT / "stack.json", report)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("stage", choices=["validate", "candidates", "followup", "stack"])
    parser.add_argument("--only", nargs="*", choices=sorted(CANDIDATES))
    parser.add_argument("--threads", type=int, default=3)
    args = parser.parse_args()
    if args.stage == "validate":
        validate(args.threads)
    elif args.stage == "candidates":
        candidates(args.only, args.threads)
    elif args.stage == "followup":
        followup(args.threads, args.only)
    else:
        stack(args.threads)


if __name__ == "__main__":
    main()
