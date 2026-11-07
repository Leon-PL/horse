"""At The Races sectional history features.

Moved out of ``src/feature_engineer.py`` on 2026-11-07 (research 065).
"""

import logging

import numpy as np
import pandas as pd

from src.features.common import _with_columns, _without

logger = logging.getLogger(__name__)


def add_atr_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derive historical features from At The Races (TPD) sectionals.

    ATR replaced RacingTV as the sectional source (RTV retired 2026-10):
    broader coverage and a far richer per-run record (full velocity curve,
    energy distribution, stride, sectional-adjusted placing and the in-running
    comment).  See :mod:`src.atr_features` for the per-run derivations.

    **Leakage prevention**: the selected dataset's ATR rows are *post-race
    outcomes*.  :func:`src.atr_features.published_atr_rolling` turns each run
    into per-run scalars and tags, joined by saved race and horse IDs, and
    rolls them over the horse's PRIOR starts whose observations were
    available at the decision.  Every raw/per-run column is then dropped, so
    nothing from the current race reaches the model.

    Feature families: (1) in-running comment rates, (2) cleaned sectional
    upgrade, (3) early-pace / closing run-style,
    (4) closing speed "in hand", (5) stride biomechanics + energy efficiency.
    """
    from src.atr_features import ALL_ATR_COLS, COMMENT_TAGS, RUN_METRICS, published_atr_rolling

    logger.info("Engineering ATR sectional features...")
    new = published_atr_rolling(df)
    race_ids = df.race_id
    for metric, _ in RUN_METRICS:
        column = f"{metric}_avg_3"
        if column in new:
            values = pd.Series(new[column], index=df.index)
            new[f"{column}_vs_field"] = (values - values.groupby(race_ids).transform("mean")).to_numpy()
    # These unused legacy composites depended on batch-wide statistics.
    # Field-relative scaling is causal and invariant to chunk boundaries.
    zscores = []
    for column in ("atr_max_speed_avg_3", "atr_finishing_speed_pct_avg_3"):
        values = pd.Series(new.get(column, np.nan), index=df.index)
        deviation = values.groupby(race_ids).transform("std")
        zscores.append((values-values.groupby(race_ids).transform("mean")) / deviation.where(deviation.gt(0)))
    new["atr_speed_score"] = ((zscores[0]+zscores[1])/2).to_numpy()
    from src.track_config import get_track_config
    circumference = pd.to_numeric(df.track.map(lambda track: get_track_config(track).get("circumference_f", np.nan)), errors="coerce")
    # Circumference in miles is a fixed physical scale. Normalising a
    # race's course against its own field would make every value missing.
    new["atr_stride_x_gallop"] = np.asarray(new.get("atr_peak_stride_length_avg_3", np.nan)) * (circumference/8.).to_numpy()
    raw = set(ALL_ATR_COLS) | {name for name, _ in RUN_METRICS} | set(COMMENT_TAGS) | {"atr_closing_in_hand"}
    return _with_columns(_without(df, [c for c in raw if c in df]), pd.DataFrame(new, index=df.index))
