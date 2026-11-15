"""The 11:00 rule without Betfair trade-activity inputs, for a Matchbook paper record (research 108).

    python -m research.matchbook_variant_108.run [--threads 7]

Matchbook's public price feed gives back and lay prices with stakes, but no last traded price, no time since the
last trade and no count of price changes. The live record therefore cannot feed ``pm_age`` or ``pm_changes``. This
re-runs the research 107 rehearsal (development races, walk-forward 2025Q3-2026Q2, the registered rule) with those
two inputs left out, beside the registered rehearsal (geo CLV 1.0372, ROI +5.5%). Development only; the hold-out is
spent for this rule and is not read.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from research.holdout_1100_107 import run as h
from src.correction import model as cm

OUT = Path("data/research/matchbook_variant_108")
DROPPED = ("pm_age", "pm_changes")


def main(threads):
    frame, values, columns, receipt = h.build(development=True)
    columns = [c for c in columns if c not in DROPPED]
    values = values.drop(columns=list(DROPPED))
    adjustment, labels = cm.walk_forward(frame, values, columns, first_scored="2025-07-01", threads=threads)
    kept = np.isfinite(adjustment)
    frame = frame.loc[kept].reset_index(drop=True)
    p = cm.corrected(frame.p_base.to_numpy(), adjustment[kept], frame.race_id.to_numpy())
    report = {"quarters": labels, "inputs": len(columns), "dropped": list(DROPPED), **h.evaluate(frame, p),
              "log_loss_corrected": float(h.race_log_loss(frame, p).mean()),
              "log_loss_base": float(h.race_log_loss(frame, frame.p_base).mean())}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "report.json").write_text(json.dumps(report, indent=1, default=float))
    rule = report["one_tick_worse threshold 0.02"]
    print(json.dumps({"primary": report["primary"], "roi": rule["roi"], "bets": rule["bets"],
                      "log_loss_corrected": report["log_loss_corrected"]}, indent=1, default=float))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--threads", type=int, default=6)
    main(parser.parse_args().threads)
