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


class Paper:
    def __init__(self, root=None):
        self.root = Path(root) if root else paper.PAPER

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
