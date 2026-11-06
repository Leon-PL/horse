"""Does the rest of the Racing TV card make the win model stronger? (experiment 053)

Builds per-runner inputs from Racing TV card data the model does not use, and
tests them on the 048/049 harness alongside experiment 052's comment score.

Card inputs (``extras.parquet``):

* badges, as 0/1 flags: trainer ("in form", "out of form", "break return",
  "headgear first", "uplift", "handicap debut"), jockey ("uplift"), icon
  ("course positive", sire strike rate at the course/distance/going,
  "warning") and text (C, D, CD, BF);
* ``pundit_tips``: Racing TV pundits tipping the runner;
* ``draw_fit``: the race's draw comment ("Favours Low") times the runner's
  relative stall, +1 = best drawn by the comment, -1 = worst;
* owner history from earlier days only: runs, wins and a smoothed strike rate.

Text for a second comment model (``text_v2.parquet``): 052's comment and
verdict sentences, plus the pace-note sentences naming the horse and the
close-up comments from its previous three runs (written after those races,
known before this one).

Left out because the card copy changes after the off (checked on 2026-10-30):
the race pace forecast (2 of 34 races) and smart stats (1 of 34).

    python -m scripts.rtv_card_extras build
    python -m scripts.rtv_card_extras score
    python -m scripts.rtv_card_extras evaluate
    python -m scripts.rtv_card_extras report
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.select_features import CONFIRM_QUARTERS, SELECTED, Cache, read_names  # noqa: E402
from scripts.specialist_models import OUT as SPECIALIST_DIR, _interval, _race_frame, blended  # noqa: E402
from scripts.text_model import (COMMENT_FIELDS, OUT as TEXT_DIR, TEXT_FEATURES, ExtraCache, fit_arm,  # noqa: E402
                                fit_scores)
from src.source_pipeline import canonical_rows, load_history_table, match_runners  # noqa: E402

OUT = ROOT / "data/research/rtv_extras_053"
BADGES = {
    "card_trainer_icon_badges": ("tb", ["in_form", "out_of_form", "break_return", "headgear_first", "uplift",
                                        "handicap_debut"]),
    "card_jockey_icon_badges": ("jb", ["uplift"]),
    "card_icon_badges": ("ib", ["course_positive", "sire_course_strike_rate", "sire_distance_strike_rate",
                                "sire_going_strike_rate", "warning"]),
    "card_text_badges": ("tx", ["C", "D", "CD", "BF"]),
}
RUNNER_COLUMNS = ["race_id", "runner_id", *BADGES, "owner", "comment", "draw"]
RACE_COLUMNS = ["race_id", "card_draw_comment", "card_ip_hints_specific_pace_text", "card_ip_hints_individual_pace"]
V2_FIELDS = {**COMMENT_FIELDS, "pace_text": "p", "closeups": "u"}
OWNER_PRIOR = 20      # pseudo-runs at the overall strike rate

logger = logging.getLogger("rtv_card_extras")


# ------------------------------------------------------------------ build --

def _linked(start: str, end: str) -> pd.DataFrame:
    races = load_history_table("races", start, end)
    runners = load_history_table("runners", start, end)
    links = match_runners(races, runners, canonical_rows(start, end))
    tips = load_history_table("race_tips", start, end)
    pundits = (tips[tips.tip_tipster__id.notna()].groupby(["race_id", "runner_id"]).size().rename("pundit_tips")
               .reset_index().rename(columns={"race_id": "rtv_race_id", "runner_id": "rtv_runner_id"}))
    rtv_runners = runners[RUNNER_COLUMNS].rename(columns={"race_id": "rtv_race_id", "runner_id": "rtv_runner_id"})
    rtv_races = races[RACE_COLUMNS].rename(columns={"race_id": "rtv_race_id"})
    out = (links.merge(rtv_runners, on=["rtv_race_id", "rtv_runner_id"], validate="one_to_one")
           .merge(rtv_races, on="rtv_race_id", validate="many_to_one")
           .merge(pundits, on=["rtv_race_id", "rtv_runner_id"], how="left", validate="one_to_one"))
    out["pundit_tips"] = out.pundit_tips.fillna(0)
    return (out.drop(columns=["rtv_race_id", "rtv_runner_id", "horse_name", "race_date"])
            .rename(columns={"sl_race_id": "race_id", "sl_horse_id": "horse_id"}))


def _sentences_naming(texts: pd.Series, names: pd.Series) -> list[str]:
    out = []
    for text, name in zip(texts, names):
        if not isinstance(text, str) or not name:
            out.append("")
            continue
        pattern = re.compile(r"\b" + re.escape(name) + r"\b", re.I)
        out.append(" ".join(s for s in re.split(r"(?<=[.!?:])\s+", text) if pattern.search(s)))
    return out


def _draw_fit(frame: pd.DataFrame) -> pd.Series:
    comment = frame.card_draw_comment.astype("string").str.casefold().str.replace("strongly ", "", regex=False)
    strength = frame.card_draw_comment.astype("string").str.startswith("Strongly").map({True: 2.0, False: 1.0})
    direction = pd.Series(np.nan, index=frame.index)
    direction[comment.isin(["favours low", "against high"])] = 1.0
    direction[comment.isin(["favours high", "against low"])] = -1.0
    direction = direction * strength      # "Favours Mid"/"Against Mid" carry no low-high direction
    draw = pd.to_numeric(frame.draw, errors="coerce").where(lambda d: d > 0)
    field = draw.groupby(frame.race_id).transform("max")
    relative = (draw - 1) / (field - 1).where(field > 1)
    return direction * (1 - 2 * relative)


def _owner_history(frame: pd.DataFrame) -> pd.DataFrame:
    """Owner runs and wins on earlier days only (same-day runs are excluded)."""
    owner = frame.owner.astype("string").str.strip().str.casefold()
    day = pd.DataFrame({"owner": owner, "date": pd.to_datetime(frame.race_date), "won": frame.won}).dropna(subset=["owner"])
    daily = day.groupby(["owner", "date"]).won.agg(runs="size", wins="sum").sort_index()
    daily["runs_before"] = daily.groupby(level="owner").runs.cumsum() - daily.runs
    daily["wins_before"] = daily.groupby(level="owner").wins.cumsum() - daily.wins
    keyed = day.join(daily[["runs_before", "wins_before"]], on=["owner", "date"])
    base = frame.won.mean()
    result = pd.DataFrame(index=frame.index, columns=["owner_runs", "owner_win_rate"], dtype=float)
    result.loc[keyed.index, "owner_runs"] = keyed.runs_before
    result.loc[keyed.index, "owner_win_rate"] = (keyed.wins_before + OWNER_PRIOR * base) / (keyed.runs_before + OWNER_PRIOR)
    return result


def build(_args) -> None:
    text = pd.read_parquet(TEXT_DIR / "text.parquet")
    start, end = str(text.race_date.min())[:10], str(text.race_date.max())[:10]
    chunks = []
    for year in range(int(start[:4]), int(end[:4]) + 1):
        lo, hi = max(start, f"{year}-01-01"), min(end, f"{year}-12-31")
        chunk = _linked(lo, hi)
        logger.info("%s: %d linked runners", year, len(chunk))
        chunks.append(chunk)
    card = pd.concat(chunks, ignore_index=True)
    card["race_id"], card["horse_id"] = card.race_id.astype(str), pd.to_numeric(card.horse_id)
    frame = text.merge(card, on=["race_id", "horse_id"], how="left", validate="one_to_one")
    linked = frame.owner.notna() | frame.comment.notna()
    extras = frame[["race_id", "horse_id"]].copy()
    for column, (prefix, values) in BADGES.items():
        parsed = frame[column].map(lambda v: set(json.loads(v)) if isinstance(v, str) and v.startswith("[") else set())
        for value in values:
            extras[f"{prefix}_{value.lower()}"] = parsed.map(lambda s, v=value: float(v in s)).where(linked)
    extras["pundit_tips"] = frame.pundit_tips.where(linked)
    extras["draw_fit"] = _draw_fit(frame)
    extras[["owner_runs", "owner_win_rate"]] = _owner_history(frame).to_numpy()
    OUT.mkdir(parents=True, exist_ok=True)
    extras.to_parquet(OUT / "extras.parquet", index=False)

    plain = frame.horse_name.astype(str).str.replace(r"\s*\([A-Z]{2,3}\)\s*$", "", regex=True).str.strip()
    pace = (frame.card_ip_hints_specific_pace_text.fillna("") + " " + frame.card_ip_hints_individual_pace.fillna(""))
    frame["pace_text"] = _sentences_naming(pace, plain)
    order = frame.assign(_date=pd.to_datetime(frame.race_date)).sort_values(["horse_id", "_date"])
    previous = [order.groupby("horse_id").comment.shift(k) for k in (1, 2, 3)]
    closeups = pd.concat(previous, axis=1).apply(lambda r: " | ".join(x for x in r if isinstance(x, str)), axis=1)
    frame["closeups"] = closeups.reindex(frame.index)
    keep = ["race_id", "horse_id", "race_date", "won", "tf_comment", "verdict_text", "verdict_rank", "pace_text", "closeups"]
    frame[keep].to_parquet(OUT / "text_v2.parquet", index=False)
    summary = {c: round(float(extras[c].mean()), 4) for c in extras.columns if c not in ("race_id", "horse_id")}
    print("linked", round(float(linked.mean()), 4), "| pace text", round(float(frame.pace_text.ne("").mean()), 3),
          "| close-ups", round(float(frame.closeups.ne("").mean()), 3))
    print(json.dumps(summary, indent=1))


# ------------------------------------------------------- score / evaluate --

def score(_args) -> None:
    fit_scores(pd.read_parquet(OUT / "text_v2.parquet"), V2_FIELDS, OUT)


def _card_names() -> list[str]:
    return [c for c in pd.read_parquet(OUT / "extras.parquet").columns if c not in ("race_id", "horse_id")]


def evaluate(_args) -> None:
    extras = pd.read_parquet(OUT / "extras.parquet")
    scores = pd.read_parquet(OUT / "scores.parquet", columns=["race_id", "horse_id", *TEXT_FEATURES])
    scores["race_id"] = scores.race_id.astype(str)
    extras["race_id"] = extras.race_id.astype(str)
    cache = ExtraCache(Cache(), extras.merge(scores, on=["race_id", "horse_id"], how="left", validate="one_to_one"))
    base = read_names(SELECTED)
    fit_arm(cache, base + _card_names(), OUT, "card")
    fit_arm(cache, base + _card_names() + TEXT_FEATURES, OUT, "all")


def report(_args) -> dict:
    def load(directory, arm):
        return pd.concat([pd.read_parquet(directory / f"{arm}_{q}.parquet").assign(fold=q) for q in CONFIRM_QUARTERS],
                         ignore_index=True)
    arms = {"model (126 inputs)": load(SPECIALIST_DIR, "general")}
    for name, directory, arm in (("+ comments (052)", TEXT_DIR, "with_text"), ("+ card data", OUT, "card"),
                                 ("+ card data + comments, pace, close-ups", OUT, "all")):
        if all((directory / f"{arm}_{q}.parquet").exists() for q in CONFIRM_QUARTERS):
            arms[name] = load(directory, arm)
    base = arms["model (126 inputs)"]
    races = base.drop_duplicates("race_id").set_index("race_id")[["race_date", "fold"]]
    market, reference = _race_frame(base, "market"), _race_frame(base, "model")
    rows = []
    for name, runners in arms.items():
        if not np.array_equal(runners.row.to_numpy(), base.row.to_numpy()):
            raise ValueError(f"{name} was not scored on the same rows")
        loss = _race_frame(runners, "model")
        delta = loss - reference
        per_quarter = delta.groupby(races.fold).mean()
        blend, _ = blended(runners)
        blend_gain = blend - market.loc[blend.index]
        rows.append({"arm": name, "race_ll": float(loss.mean()), "vs_model": float(delta.mean()),
                     "ci": _interval(delta, races.race_date) if name != "model (126 inputs)" else [0.0, 0.0],
                     "quarters_better": f"{int((per_quarter < 0).sum())}/{len(per_quarter)}",
                     "per_quarter": {q: round(float(v), 5) for q, v in per_quarter.items()},
                     "gap_to_market": float((loss - market).mean()),
                     "blend_vs_market": float(blend_gain.mean()),
                     "blend_ci": _interval(blend_gain, races.race_date.loc[blend.index])})
    summary = {"races": int(len(races)), "market_ll": float(market.mean()), "arms": rows}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, default=float), encoding="utf-8")
    print(f"Race log loss on {', '.join(CONFIRM_QUARTERS)}: {len(races):,} races, market {market.mean():.4f}")
    for r in rows:
        print(f"  {r['arm']:42} {r['race_ll']:.4f}  vs model {r['vs_model']:+.4f} [{r['ci'][0]:+.4f}, {r['ci'][1]:+.4f}]"
              f"  better {r['quarters_better']}  gap to market {r['gap_to_market']:+.4f}"
              f"  blend vs market {r['blend_vs_market']:+.4f} [{r['blend_ci'][0]:+.4f}, {r['blend_ci'][1]:+.4f}]")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("step", choices=["build", "score", "evaluate", "report"])
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    {"build": build, "score": score, "evaluate": evaluate, "report": report}[args.step](args)


if __name__ == "__main__":
    main()
