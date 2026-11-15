"""Match Matchbook runners to card runners by track, UK off time and horse name.

Races join on ``canonical_track_key`` and ``normalise_off_time_key`` (``src.utils``). Within a race, horses join
on ``normalise_horse_key``, then on letters alone (apostrophes, full stops and country suffixes differ between
feeds). A race whose names do not all pair one to one is left unmatched rather than guessed.
"""
import re

import numpy as np
import pandas as pd

from src.utils import canonical_track_key, normalise_horse_key, normalise_off_time_key

_SUFFIX = re.compile(r"\s*\((?:[A-Z]{2,3})\)\s*$")


def loose_key(name):
    return re.sub(r"[^a-z]", "", _SUFFIX.sub("", str(name)).lower())


def race_key(track, off_time):
    return canonical_track_key(track) + "|" + normalise_off_time_key(off_time)


def link(card, book):
    """Card rows (``race_id``, ``horse_id``, ``horse_name``, ``track``, ``off_time``) with Matchbook ids.

    Returns the card with ``event_id`` and ``runner_id`` (NaN where unmatched) and a per-race report.
    """
    card = card.copy()
    card["_race"] = [race_key(t, o) for t, o in zip(card.track, card.off_time)]
    book = book.drop_duplicates("runner_id").copy()
    book["_race"] = [race_key(t, o) for t, o in zip(book.track, book.off_time)]
    events = book.groupby("_race").event_id.agg(lambda s: s.unique().tolist())
    card["event_id"], card["runner_id"] = np.nan, np.nan
    card[["event_id", "runner_id"]] = card[["event_id", "runner_id"]].astype(object)
    report = []
    for race, rows in card.groupby("_race", sort=False):
        ids = events.get(race, [])
        if len(ids) != 1:
            report.append({"race": race, "status": "no market" if not ids else "several markets"})
            continue
        runners = book.loc[book.event_id.eq(ids[0])]
        pairs = {}
        for key in (normalise_horse_key, loose_key):
            mine = rows.horse_name.map(key)
            theirs = runners.horse_name.map(key)
            if mine.is_unique and theirs.is_unique:
                lookup = dict(zip(theirs, runners.runner_id))
                for index, k in mine.items():
                    if pairs.get(index) is None:
                        pairs[index] = lookup.get(k)
        matched = {i: r for i, r in pairs.items() if r is not None}
        if len(set(matched.values())) != len(matched):
            report.append({"race": race, "status": "ambiguous names"})
            continue
        for index, runner in matched.items():
            card.at[index, "event_id"], card.at[index, "runner_id"] = ids[0], runner
        report.append({"race": race, "status": "matched", "card_runners": len(rows), "linked": len(matched),
                       "book_runners": int(len(runners)),
                       "unlinked_card": sorted(rows.loc[~rows.index.isin(matched)].horse_name.tolist())})
    return card.drop(columns="_race"), pd.DataFrame(report)
