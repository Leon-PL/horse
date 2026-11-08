"""Bulk per-run ATR derivations against the row-by-row reading of the same cells."""
import json

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytest

from src import atr_features as af
from src.parquet_io import combine_parquet_parts


def row_by_row(frame):
    """Sectional metrics as one horse at a time would be read."""
    n = len(frame)
    early, top, closing, surge, gap = (np.full(n, np.nan) for _ in range(5))
    for i, (speeds, labels, energy, optimum) in enumerate(zip(
            frame["atr_sectional_speeds"], frame["atr_sectional_speed_labels"],
            frame["atr_energy_dist"], frame["atr_optimum_energy"])):
        sp = af._sectional_speeds(speeds, labels)
        if sp:
            early[i], top[i], closing[i] = sp[0], max(sp), sp[-1]
            if len(sp) >= 2:
                surge[i] = max(0.0, max(b - a for a, b in zip(sp[:-1], sp[1:])))
        e, o = af._as_list(energy), af._as_list(optimum)
        if e and o:
            m = min(len(e), len(o))
            gap[i] = -float(np.mean([abs(e[j] - o[j]) for j in range(m)]))
    return {"atr_early_pace": early, "atr_max_speed": top, "atr_closing_speed": closing,
            "atr_turn_of_foot": surge, "atr_energy_eff": gap}


def card(rng, n, *, damaged=()):
    def series(low, high, size):
        return json.dumps(np.round(rng.uniform(low, high, size), 2).tolist())

    labels = ['["Start to 5f", "5f to 2f", "2f to Finish"]', '["Start to 2f", "max mph", "2f to Finish"]', "", None]
    sizes = rng.integers(1, 12, n)
    frame = pd.DataFrame({
        "atr_sectional_speeds": [series(30, 40, 3) for _ in range(n)],
        "atr_sectional_speed_labels": [labels[int(i)] for i in rng.integers(0, 4, n)],
        "atr_energy_dist": [series(80, 120, int(s)) for s in sizes],
        "atr_optimum_energy": [series(80, 120, int(s) + int(rng.integers(0, 2))) for s in sizes],
        "atr_finishing_speed_pct": rng.uniform(80, 120, n), "atr_efficiency_score": rng.uniform(1, 30, n),
        "atr_peak_stride_length": rng.uniform(15, 30, n), "atr_peak_stride_freq": rng.uniform(1.5, 3, n),
        "atr_official_position": rng.integers(1, 15, n).astype(float),
        "atr_adj_placing": rng.integers(1, 15, n).astype(float),
        "won": rng.integers(0, 2, n).astype(float), "lengths_behind": rng.uniform(0, 30, n),
        "atr_in_running_comment": ["led, weakened" if i % 3 else "hampered, stayed on" for i in range(n)],
    })
    frame.loc[rng.random(n) < .1, "won"] = np.nan
    frame.loc[rng.random(n) < .1, "lengths_behind"] = np.nan
    frame.loc[rng.random(n) < .1, "atr_adj_placing"] = np.nan
    for column, cells in damaged:
        values = frame[column].astype(object).to_numpy(copy=True)
        for row in rng.integers(0, n, n // 4):
            values[row] = cells[int(rng.integers(0, len(cells)))]
        frame[column] = pd.Series(list(values), dtype=object)
    return frame


TEXT_DAMAGE = ["", "  ", None, float("nan"), "not json", "[1, 2", "[]", "[null, 30.5, 31.5]",
               "[NaN, 30.0]", "[Infinity, 30.0, 31.0]", "[true, 32.0]", '["31.2", 33.0]']


@pytest.mark.parametrize("column", ["atr_sectional_speeds", "atr_energy_dist", "atr_optimum_energy"])
@pytest.mark.parametrize("cells", [TEXT_DAMAGE, [[30.0, None, 31.0], (30.0, "x"), np.array([30.0, np.nan])]],
                         ids=["text", "containers"])
def test_damaged_cells_read_as_they_do_row_by_row(column, cells):
    frame = card(np.random.default_rng(3), 600, damaged=[(column, cells)])
    out = af.compute_atr_run_features(frame)
    for name, expected in row_by_row(frame).items():
        plausible = af._PLAUSIBLE.get(name)
        if plausible:
            expected = np.where((expected < plausible[0]) | (expected > plausible[1]), np.nan, expected)
        assert np.array_equal(out[name].to_numpy(), expected, equal_nan=True), name


def test_unreadable_section_labels_drop_only_their_own_row():
    frame = card(np.random.default_rng(5), 300, damaged=[
        ("atr_sectional_speed_labels", ["not json", "[]", '["a", "b", "c"]', '["Start to 2f"]',
                                        ["Start to 2f", "2f to Finish"]])])
    out = af.compute_atr_run_features(frame)
    expected = row_by_row(frame)
    assert np.array_equal(out["atr_closing_speed"].to_numpy(), expected["atr_closing_speed"], equal_nan=True)


def test_upgrade_and_in_hand_rules():
    frame = pd.DataFrame({
        "atr_sectional_speeds": [None] * 5, "atr_sectional_speed_labels": [None] * 5,
        "atr_energy_dist": [None] * 5, "atr_optimum_energy": [None] * 5,
        "atr_finishing_speed_pct": [101.0, 102.0, 103.0, np.nan, 105.0],
        "atr_efficiency_score": [5.0] * 5, "atr_peak_stride_length": [20.0] * 5,
        "atr_peak_stride_freq": [2.0] * 5,
        "atr_official_position": [9.0, 9.0, 9.0, 1.0, np.nan], "atr_adj_placing": [1.0, 5.0, 5.0, 1.0, 1.0],
        "won": [0.0, 0.0, np.nan, 1.0, 1.0], "lengths_behind": [3.0, 3.0, 20.0, 0.0, 0.0],
        "atr_in_running_comment": ["stayed on", "eased, stayed on", "", "readily", ""],
    })
    out = af.compute_atr_run_features(frame)
    assert out["atr_upgrade_clean"].iloc[:4].tolist() == [6.0, 0.0, 0.0, 0.0]
    assert np.isnan(out["atr_upgrade_clean"].iloc[4])
    assert out["atr_closing_in_hand"].iloc[0] == 101.0
    assert out["atr_closing_in_hand"].iloc[[1, 2, 3]].isna().all()
    assert out["atr_closing_in_hand"].iloc[4] == 105.0


def test_comment_tags_agree_with_python_regexes():
    comments = ["", "LED, Weakened quickly", "hit 3 out", "café held up", "ſtayed on", "bleed, ledge",
                "made all", "Kept On WELL", "K", "denied a clear run", "jumped left", "ledé"] * 20
    tags = af._comment_tags(comments)
    for tag in af.COMMENT_TAGS:
        expected = [float(af.tag_comment(text)[tag]) for text in comments]
        assert tags[tag].tolist() == expected, tag
        assert tags[tag].dtype == np.float64


@pytest.mark.parametrize("n", [*range(1, 40), 64, 127, 128])
def test_mean_of_a_short_array_follows_numpy_exactly(n):
    rng = np.random.default_rng(n)
    for _ in range(50):
        values = np.abs(rng.normal(0, 3, 128)) * 10.0 ** rng.integers(-3, 4)
        assert af._numpy_sum(values, n) / n == float(np.mean(values[:n].tolist()))


def test_consolidation_keeps_rows_types_and_batch_size(tmp_path):
    frames = [pd.DataFrame({"a": [1, 2, 3], "b": [1.5, np.nan, 2.5], "text": ["x", None, "z"]}),
              pd.DataFrame({"a": [4.5, 5.5], "c": [7.0, 8.0]})]
    paths = []
    for i, frame in enumerate(frames):
        paths.append(tmp_path / f"part_{i}.parquet")
        frame.to_parquet(paths[-1], index=False)
    target = tmp_path / "whole.parquet"
    assert combine_parquet_parts(paths, target, batch_rows=2) == 5
    table = pq.read_table(target)
    assert table.num_rows == 5
    assert table.schema.field("a").type == "double"
    assert table["a"].to_pylist() == [1.0, 2.0, 3.0, 4.5, 5.5]
    assert table["c"].to_pylist() == [None, None, None, 7.0, 8.0]
    groups = pq.ParquetFile(target).metadata
    assert [groups.row_group(i).num_rows for i in range(groups.num_row_groups)] == [2, 1, 2]


def test_consolidation_reads_big_files_in_bounded_blocks(tmp_path):
    from src.parquet_io import _row_group_blocks
    frame = pd.DataFrame({"a": np.arange(100.0)})
    path = tmp_path / "groups.parquet"
    frame.to_parquet(path, index=False, row_group_size=10)
    metadata = pq.ParquetFile(path).metadata
    blocks = list(_row_group_blocks(metadata, 25))
    assert blocks[0] == [0, 1] and sum(map(len, blocks)) == 10
    assert list(_row_group_blocks(metadata, 5))[0] == [0]  # a group larger than the limit still reads
