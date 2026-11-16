"""Read historical evaluation artifacts and summarise their recorded bets."""
import re

import numpy as np
import pandas as pd

from src.bet_analysis import max_drawdown
from src.bet_settlement import apply_reduction_factor, betfair_tick_down, settle_win_exchange_bets, value_bet_selection
from src.clv import summarise_clv
from src.feature_selection import race_groups
from src.forecast_evaluation import probability_metrics
from src.workbench.data import connection, records
from src.workbench.modelling import RUN_ID, artifact_json

STRATEGIES = ("top_pick", "value", "each_way")
ODDS_BANDS = {"1–3": (1, 3), "3–5": (3, 5), "5–8": (5, 8), "8–12": (8, 12), "12–20": (12, 20), "20+": (20, np.inf)}
BET_COLUMNS = {"race_id", "race_date", "race_type", "race_name", "track", "horse_id", "horse_name",
               "strategy", "odds", "won", "placed", "pnl", "stake", "bsp"}
TIMING_CUTS = {"3600s": "T-60", "1100": "11:00"}
TIMING_THRESHOLDS = ("0", "0.02", "0.05", "0.10")
PER_PAGE = 30


def settlement_summary(frame):
    staked = float(frame.stake.sum())
    pnl = float(frame.pnl.sum())
    daily = frame.groupby("race_date").pnl.sum()
    return {"bets": len(frame), "staked": staked, "pnl": pnl,
            "roi": 100*pnl/staked if staked else None,
            "strike_rate": 100*float(frame.won.mean()) if "won" in frame and frame.won.notna().any() else None,
            "drawdown": max_drawdown(daily.to_numpy())}


def timing_odds(frame, cut, ticks):
    odds = apply_reduction_factor(frame[f"ltp_{cut}"], frame[f"rf_after_{cut}"])
    return betfair_tick_down(odds, ticks) if ticks else odds


def timing_bets(frame, cut, threshold, ticks):
    """Reproduce study 105 using its saved forecasts, snapshots and settlement assumptions."""
    odds = timing_odds(frame, cut, ticks)
    valid = frame[f"mkt_{cut}"].notna() & np.isfinite(frame.bsp) & frame.bsp.gt(1) & (odds > 1) & (odds <= 51)
    cohort = frame.loc[valid]
    odds = odds[valid]
    picks = value_bet_selection(cohort[f"p_{cut}"], odds, threshold, implied_prob=cohort[f"mkt_{cut}"])["mask"]
    picked = cohort.loc[picks, ["race_id", "horse_id", "race_date", "bsp", "won"]].copy()
    return picked.assign(odds=odds[picks], stake=1., pnl=settle_win_exchange_bets(odds[picks], picked.won, commission=.02))


def bet_history_records(frame, db):
    rows = records(frame)
    race_ids = sorted({str(row["race_id"]) for row in rows if row.get("race_id") is not None})
    if not race_ids or not db.is_file():
        return rows
    placeholders = ",".join("?" for _ in race_ids)
    races, horses = {}, {}
    with connection(db) as con:
        for race_id, race_date, horse_id, horse_name, track, race_name in con.execute(
            f"""SELECT race_id, race_date, horse_id, horse_name, track, race_name
                FROM results WHERE race_id IN ({placeholders})""", race_ids
        ):
            key = (str(race_id), str(race_date)[:10])
            races[key] = (track, race_name)
            horses[*key, str(horse_id)] = horse_name
    for row in rows:
        key = (str(row.get("race_id")), row["race_date"])
        track, name = races.get(key, (None, None))
        row["track"] = row.get("track") or track
        row["race_name"] = row.get("race_name") or name
        row["horse_name"] = row.get("horse_name") or horses.get((*key, str(row.get("horse_id"))))
    return rows


def in_odds_band(frame, band):
    if not band:
        return frame
    if band not in ODDS_BANDS:
        raise ValueError("Choose an available odds band")
    low, high = ODDS_BANDS[band]
    return frame.loc[np.isfinite(frame.odds) & frame.odds.gt(low) & frame.odds.le(high)]


def bet_clv(frame):
    if "bsp" not in frame:
        return None
    closing = frame.bsp.where(np.isfinite(frame.bsp) & frame.bsp.gt(1))
    return summarise_clv(frame.odds/closing)


def bet_view(frame, params, db):
    frame = frame.assign(race_date=pd.to_datetime(frame.race_date, errors="coerce").dt.date.astype("string"))
    for column in ("odds", "pnl", "stake", "won", "placed", "bsp"):
        if column in frame:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if "won" in frame:
        frame["won"] = frame.won.where(frame.won.isin([0, 1]))
    frame = frame.loc[frame.race_date.notna() & np.isfinite(frame.pnl) & np.isfinite(frame.stake) & frame.stake.gt(0)].copy()
    frame["quarter"] = pd.to_datetime(frame.race_date).dt.to_period("Q").astype(str)
    quarters = sorted(frame.quarter.unique())
    types = sorted(frame.race_type.dropna().unique()) if "race_type" in frame else []
    quarter = params.get("quarter")
    if quarter:
        frame = frame.loc[frame.quarter.eq(quarter)]
    if params.get("race_type"):
        if params["race_type"] not in types:
            raise ValueError("Choose an available race type")
        frame = frame.loc[frame.race_type.eq(params["race_type"])]
    odds_breakdown = []
    for band in ODDS_BANDS:
        group = in_odds_band(frame, band)
        odds_breakdown.append({"band": band, **settlement_summary(group), "clv": bet_clv(group)})
    frame = in_odds_band(frame, params.get("band"))
    summary = settlement_summary(frame)
    periods = [{"quarter": label, **settlement_summary(group)} for label, group in frame.groupby("quarter")]
    curve = frame.groupby("race_date").pnl.sum().cumsum()
    page = max(1, min(int(params.get("page", 1)), (len(frame)+PER_PAGE-1)//PER_PAGE or 1))
    shown = frame.sort_values("race_date", ascending=False, kind="stable").iloc[(page-1)*PER_PAGE:page*PER_PAGE]
    return {"summary": summary, "quarterly": periods, "curve": [{"date": day, "pnl": pnl} for day, pnl in curve.items()],
            "clv": bet_clv(frame), "odds_breakdown": odds_breakdown if any(b["bets"] for b in odds_breakdown) else [],
            "rows": bet_history_records(shown.drop(columns="quarter"), db), "quarters": quarters, "bands": list(ODDS_BANDS),
            "race_types": types, "page": page, "page_size": PER_PAGE}


def run_forecast(meta):
    config = meta.get("training_config") or {}
    if (config.get("split") or {}).get("mode") == "fit_as_of":
        return {}
    analysis = meta.get("test_analysis") or {}
    metrics = (config.get("evals") or {}).get("calibration") or meta.get("metrics") or {}
    heads = {}
    for key in ("win_classifier", "place_classifier"):
        head = metrics.get(key) or {}
        if head:
            heads[key] = {name: head.get(name) for name in ("race_log_loss", "log_loss", "brier_score", "ece")}
    win = metrics.get("win_classifier") or {}
    bins = [{"predicted": b.get("mean_pred"), "observed": b.get("obs_rate"), "runners": b.get("count")}
            for b in win.get("reliability_bins") or [] if isinstance(b, dict) and b.get("count")]
    if not bins:
        bins = [{"predicted": b["avg_model_pct"]/100, "observed": b["actual_win_rate"]/100, "runners": b["runners"]}
                for b in analysis.get("calibration") or []
                if isinstance(b, dict) and b.get("runners") and b.get("avg_model_pct") is not None
                and b.get("actual_win_rate") is not None]
    return {"heads": heads, "bins": bins, "races": analysis.get("test_races"),
            "scope": "Saved hold-out results for the full evaluation period."}


def timing_forecast(frame, cut):
    scored = frame[["race_id", "horse_id", "won", f"p_{cut}"]].rename(columns={f"p_{cut}": "probability"})
    race = scored.race_id
    valid = (scored.horse_id.notna() & scored.won.isin([0, 1]) & scored.probability.between(0, 1)
             & ~scored.duplicated(["race_id", "horse_id"], keep=False))
    complete = (valid.groupby(race).transform("all") & scored.won.groupby(race).transform("sum").eq(1)
                & scored.won.groupby(race).transform("size").gt(1)
                & np.isclose(scored.probability.groupby(race).transform("sum"), 1, rtol=0, atol=1e-8))
    scored = scored.loc[complete.fillna(False)].sort_values("race_id")
    if scored.empty:
        return {}
    metrics, _, _ = probability_metrics(scored.probability, scored.won, race_groups(scored.race_id))
    bands = pd.cut(scored.probability, np.linspace(0, 1, 11), include_lowest=True)
    bins = scored.groupby(bands, observed=True).agg(
        predicted=("probability", "mean"), observed=("won", "mean"), runners=("won", "size"))
    return {"heads": {"win_classifier": {"race_log_loss": metrics["race_log_loss"], "brier_score": metrics["brier"],
                                        "log_loss": metrics["binary_log_loss"], "ece": metrics["ece"]}},
            "bins": records(bins.reset_index(drop=True)), "races": scored.race_id.nunique(),
            "scope": "Retrospective walk-forward scores for complete archived race fields."}


def timing_race(field, cut, threshold, ticks, band, db):
    """Keep the full archived field and mark the runners selected by the report rule."""
    if field.empty:
        raise FileNotFoundError("This race has no saved forecasts in the price-timing study.")
    if field.horse_id.isna().any() or field.horse_id.duplicated().any() or field.race_date.nunique(dropna=False) != 1:
        raise ValueError("This archived race has ambiguous runner identities or dates.")
    bets = in_odds_band(timing_bets(field, cut, threshold, ticks), band)
    shown = field[["race_id", "horse_id", "won", "bsp", "p_native", f"ltp_{cut}", f"p_{cut}", f"mkt_{cut}"]].rename(
        columns={"p_native": "base_probability", f"ltp_{cut}": "market_price",
                 f"p_{cut}": "corrected_probability", f"mkt_{cut}": "market_probability"})
    shown = shown.assign(race_date=pd.to_datetime(field.race_date).dt.strftime("%Y-%m-%d"),
                         simulated_price=np.where(shown.market_price.gt(1), timing_odds(field, cut, ticks), np.nan))
    shown = shown.merge(bets[["horse_id", "pnl"]], on="horse_id", how="left", validate="one_to_one")
    rows = bet_history_records(shown.sort_values("corrected_probability", ascending=False, kind="stable"), db)
    return {"id": rows[0]["race_id"], "date": rows[0]["race_date"], "track": rows[0].get("track"),
            "name": rows[0].get("race_name"), "rows": rows, "bets": len(bets), "pnl": float(bets.pnl.sum())}


class Evaluation:
    def __init__(self, catalog):
        self.catalog = catalog
        self.runs = catalog.root/"runs"
        self.timing_file = catalog.root/"research/t60_clv_105/frame.parquet"

    def menu(self):
        choices = []
        for folder in sorted(self.runs.glob("*"), reverse=True):
            path = folder/"meta.json"
            if not RUN_ID.fullmatch(folder.name) or path.resolve() != path:
                continue
            meta = artifact_json(path)
            if meta.get("run_id") != folder.name:
                continue
            analysis = meta.get("test_analysis") or {}
            evals = (meta.get("training_config") or {}).get("evals") or {}
            forecast = run_forecast(meta)
            ledger = folder/"bets.csv"
            if (analysis.get("stats") or evals.get("roi") or forecast.get("heads") or forecast.get("bins")
                    or ledger.is_file() and ledger.resolve() == ledger):
                choices.append({"id": folder.name, "name": meta.get("name") or folder.name})
        return {"runs": choices, "timing": self.timing_file.is_file() and self.timing_file.resolve() == self.timing_file}

    def run(self, identifier, params):
        if not RUN_ID.fullmatch(identifier):
            raise FileNotFoundError("Evaluation run was not found")
        folder = self.runs/identifier
        path = folder/"meta.json"
        if path.resolve() != path:
            raise FileNotFoundError("Evaluation run was not found")
        meta = artifact_json(path)
        if meta.get("run_id") != identifier:
            raise FileNotFoundError("Evaluation run was not found")
        analysis = meta.get("test_analysis") or {}
        evals = (meta.get("training_config") or {}).get("evals") or {}
        saved = evals.get("roi") or analysis.get("stats") or {}
        rows = None
        ledger = folder/"bets.csv"
        if ledger.is_file() and ledger.resolve() == ledger:
            stat = ledger.stat()
            rows = self.catalog.cached((ledger, stat.st_mtime_ns, stat.st_size), 60, lambda: pd.read_csv(
                ledger, usecols=lambda column: column in BET_COLUMNS, dtype={"race_id": "string", "horse_id": "string"}))
        required = {"race_date", "strategy", "odds", "stake", "pnl"}
        if rows is not None and not required.issubset(rows.columns):
            rows = None
        recorded = set(rows.strategy) if rows is not None else set()
        strategies = [key for key in STRATEGIES if key in saved or key in recorded]
        strategy = params.get("strategy") or (strategies[0] if strategies else None)
        if strategy and strategy not in strategies:
            raise ValueError("Choose a strategy in the saved report")
        settings = analysis.get("value_config") or {}
        output = {"title": meta.get("name") or identifier, "dates": analysis.get("test_date_range"),
                  "forecast": run_forecast(meta), "strategies": strategies, "strategy": strategy,
                  "clv": (evals.get("clv") or {}).get("by_strategy", {}).get(strategy),
                  "bets": None, "summary": saved.get(strategy), "stake_mode": settings.get("staking_mode") if strategy == "value" else None,
                  "value_threshold": settings.get("value_threshold") if strategy == "value" else None}
        if strategy and rows is not None:
            output["bets"] = bet_view(rows.loc[rows.strategy.eq(strategy)], params, self.catalog.db)
        return output

    def timing(self, params):
        if not self.timing_file.is_file() or self.timing_file.resolve() != self.timing_file:
            raise FileNotFoundError("The archived price-timing study was not found")
        cut, edge, ticks = params.get("cut", "1100"), params.get("edge", "0.02"), params.get("ticks", "0")
        if cut not in TIMING_CUTS or edge not in TIMING_THRESHOLDS or ticks not in ("0", "1"):
            raise ValueError("Choose a saved price timing, threshold and tick assumption")
        settings = {"cut": cut, "edge": edge, "ticks": ticks, "cuts": TIMING_CUTS, "edges": TIMING_THRESHOLDS}
        columns = ["race_id", "horse_id", "race_date", "won", "bsp",
                   f"ltp_{cut}", f"rf_after_{cut}", f"p_{cut}", f"mkt_{cut}"]
        race_id = params.get("race_id")
        if race_id:
            if not re.fullmatch(r"[A-Za-z0-9_-]{1,80}", race_id):
                raise ValueError("Invalid race ID")
            field = pd.read_parquet(self.timing_file, columns=[*columns, "p_native"], filters=[("race_id", "=", race_id)])
            return {**settings, "race": timing_race(field, cut, float(edge), int(ticks), params.get("band"), self.catalog.db)}
        stat = self.timing_file.stat()
        frame = self.catalog.cached((self.timing_file, cut, stat.st_mtime_ns, stat.st_size), 60,
                                    lambda: pd.read_parquet(self.timing_file, columns=columns))
        selected = timing_bets(frame, cut, float(edge), int(ticks))
        dates = pd.to_datetime(frame.race_date)
        return {**settings, "title": "Corrected win forecast · Study 105", "dates": [str(dates.min().date()), str(dates.max().date())],
                "bets": bet_view(selected, params, self.catalog.db),
                "forecast": self.catalog.cached(("timing_forecast", cut, stat.st_mtime_ns, stat.st_size), 60,
                                                lambda: timing_forecast(frame, cut))}
