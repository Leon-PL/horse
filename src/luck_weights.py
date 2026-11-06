"""Training weights that discount results decided by bad luck (experiment 059).

Trouble in running added almost nothing as a model input (ledgers 2b, 7). This
uses it on the label side instead: a race where the likely winner fell or was
badly hampered teaches less about ability than a clean one, so it gets less
weight in training. Evaluation is always on the actual results, unweighted.

A runner's trouble comes only from the race being trained on, never a later
one:

* the Racing TV close-up comment for that run (``source_pipeline.load_matched_text``);
* the BHA stewards' events for that fixture (``data/processed/stewards_events.parquet``;
  the IHRB half is mis-keyed, 055, and is not used).

Each runner gets a severity: 1 for a result-ending or result-deciding event
(fell, unseated, brought down, badly hampered, lame, bled), 0.5 for ordinary
interference (hampered, short of room, no clear run, lost a shoe), 0
otherwise. Habitual traits (slowly away, jumping errors, hanging, pulling) are
not counted: they are part of the horse, not luck. A winner's trouble is
ignored, since it won anyway.

Two ways to turn severities into weights are offered; which one is used, and
how strongly, is fixed on development quarters before the confirmation run:

* race level: ``w_race = max(0, 1 - beta * L)``, where ``L`` is the market
  probability carried by troubled losers, weighted by severity;
* runner level: a troubled loser's own row gets ``1 - severity``.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

SEVERE_COMMENT = re.compile(
    r"\b(?:fell|unseated(?: (?:rider|jockey))?|brought down|slipped up|carried out|saddle slipped|"
    r"(?:badly|severely) hampered|hampered badly|nearly (?:fell|unseated)|almost (?:fell|unseated)|"
    r"clipped heels|lost (?:many|several|a lot of|considerable) (?:lengths|ground)|lost all chance|"
    r"lost (?:his|her|its) action|went lame|broke down|bled|stumbled badly|blundered badly)\b",
    re.IGNORECASE)
MODERATE_COMMENT = re.compile(
    r"\b(?:hampered|short of (?:racing )?room|no room|not (?:a )?clear run|denied (?:a )?clear run|"
    r"no clear run|squeezed out|checked|snatched up|blocked|nowhere to go|impeded|interfered with|"
    r"carried (?:left|right|wide)|forced wide|stumbled|blundered|lost (?:a|both|front|hind) shoes?)\b",
    re.IGNORECASE)

STEWARDS_SEVERITY = {
    "fell": 1.0, "brought_down": 1.0, "unseated": 1.0, "lame": 1.0, "bled": 1.0, "distressed": 1.0,
    "trouble_in_running": 0.5, "lost_shoe": 0.5, "breathing": 0.5,
}

SCHEMES = {
    "race_beta2": ("race", 2.0),
    "race_beta4": ("race", 4.0),
    "runner": ("runner", 1.0),
}


def comment_severity(comments) -> np.ndarray:
    """Severity of each close-up comment: 1, 0.5 or 0 (missing text is 0)."""
    text = pd.Series(comments, dtype="object").fillna("").astype(str)
    severe = text.str.contains(SEVERE_COMMENT)
    moderate = text.str.contains(MODERATE_COMMENT)
    return np.where(severe, 1.0, np.where(moderate, 0.5, 0.0))


def stewards_severity(tags) -> float:
    return max((STEWARDS_SEVERITY.get(t, 0.0) for t in tags), default=0.0)


def race_luck(severity, market, won, groups) -> np.ndarray:
    """Per-race market probability carried by troubled losers, severity-weighted.

    Where a race's market is incomplete every runner counts as 1/field.
    """
    severity = np.asarray(severity, dtype=np.float64)
    market = np.asarray(market, dtype=np.float64)
    won = np.asarray(won)
    groups = np.asarray(groups, dtype=np.int64)
    starts = np.r_[0, np.cumsum(groups)[:-1]]
    priced = np.minimum.reduceat(np.isfinite(market) & (market > 0), starts).astype(bool)
    prob = np.where(np.repeat(priced, groups), np.nan_to_num(market), 1.0 / np.repeat(groups, groups))
    return np.add.reduceat(prob * severity * (won != 1), starts)


def luck_weights(scheme: str, severity, market, won, groups) -> np.ndarray:
    """Row multipliers in [0, 1] for one of :data:`SCHEMES`."""
    kind, strength = SCHEMES[scheme]
    severity = np.asarray(severity, dtype=np.float64)
    if kind == "runner":
        return np.where(np.asarray(won) == 1, 1.0, 1.0 - strength * severity).clip(0.0, 1.0)
    luck = race_luck(severity, market, won, groups)
    return np.repeat(np.clip(1.0 - strength * luck, 0.0, 1.0), np.asarray(groups, dtype=np.int64))
