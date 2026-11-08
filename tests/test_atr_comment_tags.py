"""ATR in-running comment tags: running style comes from the early race only (2026-11-08)."""
import numpy as np
import pyarrow as pa
import pytest

import src.atr_features as af

CASES = {
    # A hold-up ride that won is a hold-up ride, however the comment ends.
    "held up in rear, headway 2f out, led inside final furlong, readily":
        {"atr_cmt_front": 0, "atr_cmt_held_up": 1, "atr_cmt_comfortable": 1},
    # The start and the preliminaries come before the first position.
    "wore hood to post, slowly away, prominent, weakened over 1f out":
        {"atr_cmt_front": 1, "atr_cmt_held_up": 0, "atr_cmt_weakened": 1},
    "raced keenly tracking leaders, not clear run 1f out, ran on":
        {"atr_cmt_front": 1, "atr_cmt_trouble": 1, "atr_cmt_stayed_on": 1},
    "close-up, pushed along 2f out, no extra": {"atr_cmt_front": 1, "atr_cmt_held_up": 0},
    # Behind the leaders is with them, not held up.
    "in touch behind leaders, ridden 2f out, kept on": {"atr_cmt_front": 0, "atr_cmt_held_up": 0},
    # A bunched field with the horse at the back is a hold-up ride.
    "close up towards rear, weakened final furlong": {"atr_cmt_front": 0, "atr_cmt_held_up": 1},
    # Dropping to the rear late is not a running style.
    "mid-division, pushed along over 2f out and soon behind": {"atr_cmt_front": 0, "atr_cmt_held_up": 0},
    # A position first stated after the first three clauses is not read.
    "wore hood, dwelt, awkward start, led after 2f": {"atr_cmt_front": 0, "atr_cmt_held_up": 0},
    # "Never nearer" is a closer still making ground, not an eased horse.
    "towards rear, ridden 2f out, ran on well, never nearer": {"atr_cmt_eased": 0, "atr_cmt_stayed_on": 1},
    "tailed off, eased": {"atr_cmt_eased": 1},
    # "Ridden out wide" is a route, not a comfortable finish.
    "pulled hard towards rear, ridden out wide entering final 2f, soon weakened": {"atr_cmt_comfortable": 0},
    "led, ridden out": {"atr_cmt_front": 1, "atr_cmt_comfortable": 1},
}


@pytest.mark.parametrize("comment, expected", CASES.items())
def test_tags_read_the_race_as_a_race_reader_would(comment, expected):
    tags = af.tag_comment(comment)
    assert {tag: tags[tag] for tag in expected} == expected


def test_bulk_tags_equal_the_reference_tagger_including_non_ascii_rows():
    comments = [*CASES, *(comment.replace("e", "é", 1) for comment in CASES), "", "K", "LED, Weakened"] * 3
    tags = af._comment_tags(comments)
    for row, comment in enumerate(comments):
        expected = {tag: float(flag) for tag, flag in af.tag_comment(comment).items()}
        assert {tag: tags[tag][row] for tag in af.COMMENT_TAGS} == expected, comment
    assert all(values.dtype == np.float64 for values in tags.values())


def test_early_clauses_in_arrow_equal_the_python_reading():
    comments = [af._rewrite(comment) for comment in CASES] + ["", "a, b, c, d", "led"]
    assert af._early_clauses(pa.array(comments, pa.string())).to_pylist() == [af._early_clause(c) for c in comments]
