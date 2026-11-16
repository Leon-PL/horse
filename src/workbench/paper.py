"""Workbench view of the Matchbook paper record (read-only; ``src.matchbook`` writes it)."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.matchbook import paper, record

BET_COLUMNS = ("race_date", "off_time", "track", "horse_name", "p", "share", "edge", "back1", "back1_size",
               "fill_price", "fill_stake", "race_tight", "bsp", "clv", "won", "pnl_unit", "pnl_fill", "status", "mb_close_back")


def _clean(value):
    if isinstance(value, (np.floating, float)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (pd.Timestamp,)):
        return str(value)
    return value


def _rows(frame):
    cols = [c for c in BET_COLUMNS if c in frame]
    return [{c: _clean(v) for c, v in zip(cols, row)} for row in frame[cols].itertuples(index=False)]


RUNNER_COLUMNS = ("horse_name", "p_model", "p", "share", "edge", "price_0800", "back1", "back1_size", "lay1",
                  "lay1_size", "price_1100", "latest_back", "latest_lay", "latest_at", "withdrawn_now", "bet",
                  "fill_price", "fill_stake", "bsp", "won", "pnl_fill", "clv")


def _num(series):
    return pd.to_numeric(series, errors="coerce")


class Paper:
    def __init__(self, root=None):
        self.root = Path(root) if root else paper.PAPER

    def day(self, date=None):
        """Every race of one paper day with its whole field: model forecasts and Matchbook prices.

        Before 11:00 only the prepared win-model forecast exists; after it, the corrected forecast, the 11:00
        book and the bets. The latest stored Matchbook quote and, once settled, the bet results are joined on.
        """
        dates = sorted((p.name for p in self.root.glob("*")
                        if (p / "forecast.parquet").exists() or (p / "prepared.parquet").exists()), reverse=True)
        date = date or (dates[0] if dates else None)
        if not date or date not in dates:
            return {"date": date, "dates": dates, "stage": None, "races": []}
        folder = self.root / date
        run = json.loads((folder / "run.json").read_text()) if (folder / "run.json").exists() else {}
        if (folder / "forecast.parquet").exists():
            frame, stage = pd.read_parquet(folder / "forecast.parquet"), "forecast"
        else:
            frame, stage = pd.read_parquet(folder / "prepared.parquet",
                                           columns=["race_id", "horse_id", "race_date", "p_base", "horse_name",
                                                    "track", "off_time"]), "prepared"
        race = frame.race_id.to_numpy()
        # The win model's own forecast, normalised within the race so it reads beside the corrected one.
        frame["p_model"] = frame.p_base / frame.p_base.groupby(race).transform("sum")
        if "runner_id" in frame:
            closes = record.closes(date)
            if not closes.empty:
                last = closes.drop_duplicates("runner_id", keep="last").set_index("runner_id")
                ids = frame.runner_id
                frame["latest_back"] = _num(ids.map(last.back1))
                frame["latest_lay"] = _num(ids.map(last.lay1))
                frame["latest_at"] = ids.map(last.taken_at).map(
                    lambda t: t.tz_convert(record.LONDON).strftime("%H:%M") if isinstance(t, pd.Timestamp) else None)
                frame["withdrawn_now"] = ids.map(last.withdrawn)
        if (folder / "field.parquet").exists():      # SP and result for every runner
            field = pd.read_parquet(folder / "field.parquet")
            frame = frame.merge(field[["race_id", "horse_id", "bsp", "won"]], on=["race_id", "horse_id"], how="left")
        if (folder / "settled.parquet").exists():
            settled = pd.read_parquet(folder / "settled.parquet")
            keep = [c for c in ("race_id", "horse_id", "bsp", "won", "pnl_fill", "clv") if c in settled
                    and (c in ("race_id", "horse_id") or c not in frame)]
            frame = frame.merge(settled[keep], on=["race_id", "horse_id"], how="left")
        races, cols = [], [c for c in RUNNER_COLUMNS if c in frame]
        for race_id, field in frame.groupby("race_id", sort=False):
            field = field.sort_values("p" if "p" in field else "p_model", ascending=False)
            head = field.iloc[0]
            book = (1 / field.back1.where(field.back1 > 1)).sum() if "back1" in field else np.nan
            races.append({
                "race_id": _clean(race_id), "off_time": head.off_time, "track": head.track,
                "runners": int(len(field)),
                "eligible": _clean(bool(field.eligible_race.all())) if "eligible_race" in field else None,
                "tight": _clean(head.race_tight) if "race_tight" in field else None,
                "book_pct": _clean(100 * book) if np.isfinite(book) and book > 0 else None,
                "bets": int(field.bet.sum()) if "bet" in field else 0,
                "field": [{c: _clean(v) for c, v in zip(cols, row)} for row in field[cols].itertuples(index=False)],
            })
        races.sort(key=lambda r: (str(r["off_time"]), str(r["track"])))
        return {"date": date, "dates": dates, "stage": stage, "races": races,
                "snapshot_1100": run.get("snapshot_1100"), "snapshot_0800": run.get("snapshot_0800"),
                "win_run": run.get("win_run")}

    def overview(self):
        days = []
        for folder in sorted(self.root.glob("*"), reverse=True):
            run = folder / "run.json"
            if run.exists():
                info = json.loads(run.read_text())
                errors = sorted(folder.glob("error*.json"))
                if errors:
                    info["error"] = json.loads(errors[-1].read_text()).get("error", "").strip().splitlines()[-1:]
                days.append({k: info.get(k) for k in ("date", "status", "card_races", "matchbook_races", "races_linked",
                                                      "eligible_races", "bets", "settled_bets", "snapshot_1100",
                                                      "error")})
        today = str(record.uk_now().date())
        bets, settled = [], []
        for folder in sorted(self.root.glob("*")):
            if (folder / "settled.parquet").exists():
                settled.append(pd.read_parquet(folder / "settled.parquet"))
            elif (folder / "bets.parquet").exists():
                bets.append(pd.read_parquet(folder / "bets.parquet"))
        open_bets = pd.concat(bets, ignore_index=True) if bets else pd.DataFrame()
        done = pd.concat(settled, ignore_index=True) if settled else pd.DataFrame()
        curve = []
        if not done.empty:
            ok = done.loc[done.status.eq("settled")].sort_values(["race_date", "off_time"])
            daily = ok.groupby("race_date").agg(bets=("pnl_unit", "size"), pnl=("pnl_unit", "sum"),
                                                log_clv=("clv", lambda s: float(np.log(s).mean())))
            daily["cumulative"] = daily.pnl.cumsum()
            curve = [{"date": d, **{k: _clean(v) for k, v in r.items()}} for d, r in daily.iterrows()]
        snapshots = sorted(record.SNAPSHOTS.glob("*/*.parquet"))
        try:
            cfg = paper.config()
        except SystemExit:
            cfg = None
        return {"summary": paper.summary(), "rule": paper.RULE, "config": cfg, "days": days, "today": today,
                "open_bets": _rows(open_bets), "settled_bets": _rows(done.sort_values(["race_date", "off_time"],
                                                                                       ascending=False)) if not done.empty else [],
                "curve": curve,
                "last_snapshot": snapshots[-1].parent.name + " " + snapshots[-1].stem if snapshots else None}
