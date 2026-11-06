"""Existing downloads are removed only after every raw member is preserved."""
import bz2
import hashlib
import io
import sqlite3
import tarfile

import pytest

from src.ingestion import betfair_archives as archives, betfair_history as history
from test_betfair_history import FakeClient, FIRST, LAST, payload, remote
import test_betfair_history as history_tests

environment = history_tests.environment


def archive(path, members):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(path, "w") as target:
        for name, body in members:
            member = tarfile.TarInfo(name)
            member.size = len(body)
            target.addfile(member, io.BytesIO(body))


def test_overlapping_archives_keep_events_and_all_market_revisions_once(tmp_path):
    market = payload()
    event = bz2.compress(b"event information not consumed by the parser")
    members = [("BASIC/2026/Jan/10/123/1.100.bz2", market), ("BASIC/2026/Jan/10/123/123.bz2", event)]
    archive(tmp_path / "a.tar", members)
    archive(tmp_path / "b.tar", members)
    report = archives.consolidate(tmp_path, remove_verified=True)
    assert report["state"] == "ok" and report["unique_payloads"] == 2
    assert not list(tmp_path.glob("*.tar"))
    with sqlite3.connect(archives.store_path(tmp_path / "history")) as con:
        assert con.execute("SELECT COUNT(*) FROM members").fetchone()[0] == 4
        assert {row[0] for row in con.execute("SELECT body FROM payloads")} == {market, event}
    assert list(archives.payloads(tmp_path / "history", "1.100")) == [(hashlib.sha256(market).hexdigest(), market)]


def test_repeated_consolidation_is_idempotent_and_can_finish_removal(tmp_path):
    archive(tmp_path / "a.tar", [("BASIC/123/1.100.bz2", payload())])
    archives.consolidate(tmp_path)
    result = archives.consolidate(tmp_path, remove_verified=True)
    assert result["unique_payloads"] == 1 and result["archives"][0]["removed"]


def test_corrupted_retained_payload_prevents_original_removal(tmp_path):
    source = tmp_path / "a.tar"
    archive(source, [("BASIC/123/1.100.bz2", payload())])
    archives.consolidate(tmp_path)
    with sqlite3.connect(archives.store_path(tmp_path / "history")) as con:
        con.execute("UPDATE payloads SET body=?", (b"corrupted",))
    with pytest.raises(ValueError, match="final hash"):
        archives.consolidate(tmp_path, remove_verified=True)
    assert source.exists()


def test_member_paths_are_metadata_and_never_extracted(tmp_path):
    archive(tmp_path / "a.tar", [("../../outside/1.100.bz2", payload())])
    assert archives.consolidate(tmp_path, remove_verified=True)["state"] == "ok"
    assert not (tmp_path.parent / "outside").exists()


def test_non_file_archive_member_prevents_original_removal(tmp_path):
    source = tmp_path / "a.tar"
    with tarfile.open(source, "w") as target:
        member = tarfile.TarInfo("link")
        member.type = tarfile.SYMTYPE
        member.linkname = "outside"
        target.addfile(member)
    with pytest.raises(ValueError, match="Unsupported"):
        archives.consolidate(tmp_path, remove_verified=True)
    assert source.exists()


def test_history_replays_consolidated_payload_without_network_or_another_raw_copy(environment):
    source = environment / "manual"
    root = environment / "betfair/history"
    archive(source / "a.tar", [("BASIC/2026/Jan/10/123/1.100.bz2", payload())])
    archives.consolidate(source, root=root, remove_verified=True)
    client = FakeClient(files={remote(100): payload()})
    result = history.sync_history(FIRST, LAST, client=client)
    assert result["state"] == "ok" and result["downloaded"] == 0 and result["reused_local"] == 1
    assert not client.downloads and not history.local_path(root, remote(100)).exists()
    assert (environment / "processed/betfair_bsp.parquet").exists()


def test_archive_replay_only_defers_network_markets_until_consolidation_finishes(environment):
    source = environment / "manual"
    root = environment / "betfair/history"
    archive(source / "a.tar", [("BASIC/123/1.100.bz2", payload())])
    archives.consolidate(source, root=root, remove_verified=True)
    client = FakeClient()
    result = history.sync_history(FIRST, LAST, client=client, local_only=True)
    assert not client.downloads and result["reused_local"] == 1
    assert result["files"] == {"published": 1, "pending": 1}
    assert history.sync_history(FIRST, LAST, client=client)["state"] == "ok"
    assert client.downloads == [remote(101)]


def test_bad_local_revision_does_not_trigger_a_download_during_consolidation(environment):
    source = environment / "manual"
    root = environment / "betfair/history"
    archive(source / "a.tar", [("BASIC/123/1.100.bz2", b"invalid retained revision")])
    archives.consolidate(source, root=root, remove_verified=True)
    client = FakeClient(files={remote(100): payload()})
    result = history.sync_history(FIRST, LAST, client=client, local_only=True)
    assert not client.downloads and result["processed"] == 0 and result["deferred_local"] == 1
    assert history.sync_history(FIRST, LAST, client=client)["state"] == "ok"


def test_truncated_archive_is_preserved_and_does_not_block_other_archives(tmp_path):
    source = tmp_path / "a.tar"
    archive(source, [("BASIC/123/1.100.bz2", payload()), ("BASIC/123/123.bz2", b"partial source"*1000)])
    with tarfile.open(source, "r:") as original:
        cut = original.getmembers()[1].offset_data+32
    damaged = source.read_bytes()[:cut]
    source.write_bytes(damaged)
    archive(tmp_path / "b.tar", [("BASIC/123/1.101.bz2", payload(101))])
    result = archives.consolidate(tmp_path, remove_verified=True)
    assert result["state"] == "ok_with_issues" and len(result["issues"]) == 1
    assert (tmp_path / "history/quarantine/a.tar").read_bytes() == damaged
    assert not list(tmp_path.glob("*.tar"))
    assert list(archives.payloads(tmp_path / "history", "1.100"))
    assert list(archives.payloads(tmp_path / "history", "1.101"))


def test_nonzero_corrupt_headers_are_preserved_instead_of_silently_skipped(tmp_path):
    source = tmp_path / "a.tar"
    archive(source, [("BASIC/123/1.100.bz2", payload())])
    damaged = source.read_bytes()+b"corrupt header".ljust(512, b"X")
    source.write_bytes(damaged)
    result = archives.consolidate(tmp_path, remove_verified=True)
    assert result["state"] == "ok_with_issues"
    assert (tmp_path / "history/quarantine/a.tar").read_bytes() == damaged


def test_interleaved_zero_padding_does_not_hide_valid_following_members(tmp_path):
    source = tmp_path / "a.tar"
    second = tmp_path / "other.part"
    archive(source, [("BASIC/123/1.100.bz2", payload())])
    archive(second, [("BASIC/123/1.101.bz2", payload(101))])
    source.write_bytes(source.read_bytes()+second.read_bytes())
    result = archives.consolidate(tmp_path, remove_verified=True)
    assert result["state"] == "ok" and result["archives"][0]["members"] == 2
