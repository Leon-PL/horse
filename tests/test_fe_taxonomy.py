"""Keeps the FE stage taxonomy in lock-step with the real pipeline.

If a stage is added to / removed from ``engineer_features`` without being
classified in ``src/fe_taxonomy.py``, this fails — so no feature can slip into
the dataset un-audited, and none can be class ``G`` (append-breaking / leaky).
"""

import inspect
import re

import pytest

from src.feature_engineer import engineer_features
from src.fe_taxonomy import (
    FORBIDDEN_CLASSES,
    STAGE_TAXONOMY,
    VALID_CLASSES,
    stages_by_class,
    stateless_stages,
)


def _pipeline_stage_names() -> set[str]:
    """Stage callables referenced inside engineer_features's stage list.

    Stages follow the ``add_*`` / ``compute_*`` / ``apply_*`` naming convention
    and appear as bare identifiers (the \\b avoids matching the ``_add_*``
    private helpers). Comments are stripped so a mention in a comment doesn't
    count.
    """
    src = inspect.getsource(engineer_features)
    code = "\n".join(
        line for line in src.splitlines() if not line.lstrip().startswith("#")
    )
    return set(re.findall(r"\b(?:add|compute|apply)_\w+", code))


def test_every_pipeline_stage_is_classified():
    referenced = _pipeline_stage_names()
    classified = set(STAGE_TAXONOMY)
    missing = referenced - classified
    extra = classified - referenced
    assert not missing, f"pipeline stages missing from STAGE_TAXONOMY: {sorted(missing)}"
    assert not extra, f"STAGE_TAXONOMY has stages not in the pipeline: {sorted(extra)}"


def test_all_classes_valid():
    bad = {n: c for n, c in STAGE_TAXONOMY.items() if c not in VALID_CLASSES}
    assert not bad, f"unknown classes: {bad}"


def test_no_forbidden_global_stage():
    forbidden = {n: c for n, c in STAGE_TAXONOMY.items() if c in FORBIDDEN_CLASSES}
    assert not forbidden, (
        "stage(s) classified as global cross-time (G) — these break "
        f"append-only and likely leak: {forbidden}"
    )


def test_helpers_expose_partition():
    # The stateless (R/W) partition is the Phase-1 "run on new rows only" set.
    assert stateless_stages() <= set(STAGE_TAXONOMY)
    assert sum(len(v) for v in stages_by_class().values()) == len(STAGE_TAXONOMY)
    # Sanity: the bulk should be cumulative.
    assert len(stages_by_class()["C"]) >= 10
