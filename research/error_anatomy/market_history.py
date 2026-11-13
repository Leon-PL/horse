"""A horse's price history without the model: past closing prices and past late moves.

``horse_state`` found that a horse's last few closing prices against the
model's forecast carry most of the own-horse gain. Those inputs need a stored
out-of-fold forecast for every past race. Two questions follow, and each is a
group declared before scoring:

- ``market-level``: is the forecast needed at all? The same lags of the
  closing price alone, as log(field size x market probability): 1.0 means an
  even share. Plain feature-store material if it works.
- ``market-moves``: did the money come late last time? Lags of the horse's
  move from the 08:00, 11:00 and one-hour exchange prices to BSP, centred
  within the race (which also removes the common shift when a runner is
  withdrawn). Past races only; no price of the race being forecast is read.
"""
import numpy as np
import pandas as pd

from research.error_anatomy import horse_state as hs
from research.meta_learner import screening as s
from src.source_dataset import load_manifest, read_table, use_manifest
from src.source_prices import attach_bsp

KEYS = s.KEYS
from src.correction.inputs import SNAPSHOTS  # noqa: F401  (one owner: src)
from src.correction.inputs import lagged as _lagged
from src.correction.inputs import price_moves as moves


def market_level(frame, market="p_market"):
    p = frame[market].to_numpy(float)
    share = np.log(np.where(p > 0, p, np.nan) * frame.num_runners.to_numpy(float))
    return _lagged(frame, pd.DataFrame({"share": share}), 5, "mh", relative=("share",))


def snapshots(frame):
    """Exchange prices at each snapshot and at the off for the frame's runners, by the dataset's saved links."""
    # Only the benchmark's dataset ID is needed here. ``t2.design()`` also checks that modelling code is
    # unchanged, which protects fits; nothing in this module fits with that code.
    from research.meta_learner import first_ensemble as fe
    receipt = fe.read_json(s.STUDY / "registration.json")["receipt"]["dataset_manifest"]
    with use_manifest(load_manifest(receipt["id"], root=receipt.get("root"))):
        table = read_table("betfair", "timing", linked=True)
    table = table.rename(columns={"bsp": "bf_bsp"})
    return attach_bsp(frame[[*KEYS]].copy(), table, cols=("bf_bsp", *SNAPSHOTS))


def market_moves(frame):
    return _lagged(frame, moves(frame, snapshots(frame)), 3, "mh", relative=("move_1100", "move_3600s"))


def load(name, frame):
    values = {"market-level": market_level, "market-moves": market_moves}[name](frame)
    return values, [c for c in values if c not in KEYS]
