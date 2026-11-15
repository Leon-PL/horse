"""Every test runs against an explicit source dataset.

Feature, processing and price code read external sources only through the
selected dataset. Tests get a small one: the committed FX reference files and
nothing else, so no test reads a collector's cache or the real data folder.
A test that needs results, ATR, Racing TV, pedigree or prices publishes its
own sources with the ``publish_dataset`` fixture. Mark a test ``no_dataset``
to start with nothing selected, as a command-line entry point does.
"""
from contextlib import contextmanager
from pathlib import Path

import pytest

import config as settings
from src import source_dataset


def pytest_configure(config):
    config.addinivalue_line("markers", "no_dataset: start the test with no source dataset selected")


@contextmanager
def _registry(directory):
    saved = settings.DATA_DIR
    settings.DATA_DIR = str(directory)
    try:
        yield
    finally:
        settings.DATA_DIR = saved


def _publish_fx():
    base = Path(settings.BASE_DIR) / "reference"
    return source_dataset.publish_source("fx", {}, contracts={}, artifacts={
        "units": base / "track_prize_currency.json", "rates": base / "ecb_gbp_per_eur.csv"})


@pytest.fixture(scope="session", autouse=True)
def _private_fe_cache(tmp_path_factory):
    """Saved feature histories go to a temporary folder, never the real ``data/processed/fe_cache``."""
    from src import fe_cache
    saved = fe_cache.ROOT
    fe_cache.ROOT = tmp_path_factory.mktemp("fe_cache")
    yield
    fe_cache.ROOT = saved


@pytest.fixture(scope="session")
def reference_dataset(tmp_path_factory):
    """FX published from the committed reference files into a session registry."""
    data = tmp_path_factory.mktemp("reference_dataset")
    with _registry(data):
        receipt = _publish_fx()
    return {"id": None, "purpose": "test", "root": str(data / "sources"), "sources": {"fx": receipt}}


@pytest.fixture(autouse=True)
def selected_dataset(request, reference_dataset):
    manifest = None if request.node.get_closest_marker("no_dataset") else reference_dataset
    with source_dataset.use_manifest(manifest):
        yield manifest


@pytest.fixture(scope="session", autouse=True)
def session_dataset(reference_dataset):
    """Module-scoped fixtures run before function fixtures and need isolation too."""
    with source_dataset.use_manifest(reference_dataset):
        yield


@pytest.fixture
def publish_dataset(tmp_path_factory):
    """Publish fixture sources and select them for the rest of the test.

    ``publish_dataset(canonical, atr={"sectionals": frame})`` publishes
    *canonical* as the SL results and links each provider's runner tables to
    it, as source publication does. A table may be given as ``(frame,
    TableContract)``; the default contract is a runner table available at
    once. The reference FX is included. Returns the selected manifest, which
    names its own registry, so spawned workers and later reads find it.
    """
    from src.ingestion import publication
    from src.source_contracts import TableContract

    def publish(canonical=None, **providers):
        data = tmp_path_factory.mktemp("dataset")
        with _registry(data):
            sources = {"fx": _publish_fx()}
            if canonical is not None:
                key = ("race_id", "horse_id")
                sources["sl"] = source_dataset.publish_source("sl", {"results": canonical},
                    contracts={"results": TableContract("canonical runner", key, key)})
            for provider, tables in providers.items():
                frames, contracts = {}, {}
                for name, value in tables.items():
                    frame, contract = value if isinstance(value, tuple) else (value, TableContract("runner", ()))
                    frames[name], contracts[name] = frame, contract
                sources[provider] = publication._publish(provider, frames, contracts,
                    canonical if canonical is not None else _empty_canonical())
            manifest = {**source_dataset.save_manifest(sources), "root": str(data / "sources")}
        source_dataset.install_manifest(manifest)
        return manifest
    return publish


@pytest.fixture
def publish_prices(publish_dataset):
    """Publish one race's runners and their Betfair BSP rows, then select them.

    The price rows carry Betfair's own name keys and are linked to the runners
    by source publication, as real prices are.
    """
    import pandas as pd
    from src.betfair import bf_horse_key
    from src.utils import normalise_off_time_key, normalise_track_key

    def publish(names, bsp, *, race_date="2025-01-01", track="Ascot", off_time="14:30"):
        canonical = pd.DataFrame({"race_id": "r1", "horse_id": range(1, len(names) + 1), "horse_name": names,
                                  "race_date": race_date, "track": track, "off_time": off_time})
        prices = pd.DataFrame({"race_date": race_date, "track_key": normalise_track_key(track),
                               "off_key": normalise_off_time_key(off_time),
                               "horse_key": [bf_horse_key(name) for name in names], "bsp": bsp})
        return publish_dataset(canonical, betfair={"bsp": prices})
    return publish


def _empty_canonical():
    import pandas as pd
    return pd.DataFrame(columns=["race_id", "horse_id", "horse_name", "race_date", "track", "off_time"])
