"""Restartable, bounded public-feed forecast recording and daily assessment."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from datetime import timedelta
import json
import os
from pathlib import Path
import time
from uuid import uuid4

import pandas as pd

from src.forward_store import timestamp, utcnow, publish_json
from src.forward_capture import discover_events, capture_event


def replace_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name+"."+uuid4().hex+".tmp")
    try:
        temporary.write_text(json.dumps(value, indent=2, default=str, allow_nan=False), encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


@contextmanager
def recorder_lock(study):
    """One recorder per study, automatically released on process termination."""
    with open(study.path/".recorder.lock", "a+b") as stream:
        if os.name == "nt":
            import msvcrt
            if not os.fstat(stream.fileno()).st_size:
                stream.write(b"\0")
                stream.flush()
            stream.seek(0)
            lock = lambda: msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            unlock = lambda: msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl
            lock = lambda: fcntl.flock(stream, fcntl.LOCK_EX|fcntl.LOCK_NB)
            unlock = lambda: fcntl.flock(stream, fcntl.LOCK_UN)
        try:
            lock()
        except OSError as error:
            raise RuntimeError("Another recorder already owns this study") from error
        try:
            yield
        finally:
            unlock()


def due_events(study, now, *, lead_seconds=180):
    """Schedule from first observations; outcomes never control capture choice."""
    now = timestamp(now)
    due, missed = [], []
    for path in (study.path/"events").glob("*.json"):
        event = json.loads(path.read_text())
        start = timestamp(event["first_start"])
        if start.date().isoformat() < study.protocol["first_eligible_date"] or study.selected(event["event_id"]):
            continue
        decision = start-timedelta(seconds=study.protocol["horizon_seconds"])
        if now > decision:
            missed.append(event)
        elif now >= decision-timedelta(seconds=lead_seconds):
            due.append(event)
    return sorted(due, key=lambda v:(v["first_start"], v["event_id"])), missed


class Recorder:
    def __init__(self, study, *, client=None, clock=utcnow, discover=discover_events,
                 capture=capture_event, workers=2, max_attempts=3, source_capture=None):
        if not 1 <= workers <= 2 or not 1 <= max_attempts <= 3:
            raise ValueError("Recorder permits one/two workers and at most three attempts per event")
        self.study, self.client, self.clock = study, client, clock
        self.discover, self.capture, self.max_attempts = discover, capture, max_attempts
        self.executor = ThreadPoolExecutor(max_workers=workers)
        self.workers, self.pending = workers, {}
        self.source_capture = source_capture
        self.source_executor = ThreadPoolExecutor(max_workers=1) if source_capture else None
        self.source_future, self.last_source_capture = None, None
        self.source_status = None
        self.last_discovery, self.failures, self.completed = None, [], []
        self.state_path = study.path/"recorder_state.json"
        self.state = json.loads(self.state_path.read_text()) if self.state_path.exists() else {"attempts":{}, "missed":[]}
        # A restart counts interrupted launches as attempts; it does not launch
        # unlimited retries or overwrite any incomplete scientific evidence.

    def tick(self, *, discover_only=False):
        self.study.verify()
        now = timestamp(self.clock())
        # Expiring RTV context is independent of the timed forecast workers.
        # Its outage or latency must never delay a forecast's issue window.
        if self.source_future is not None and self.source_future.done():
            try:
                self.source_status = self.source_future.result()
            except Exception as error:
                self.source_status = {"state": "error", "error": str(error)[:500]}
                self.failures.append({"stage": "rtv_snapshot", "error": str(error)[:500]})
            self.source_future = None
        if (self.source_capture and self.source_future is None
                and (self.last_source_capture is None or (now-self.last_source_capture).total_seconds() >= 900)):
            self.source_future = self.source_executor.submit(self.source_capture)
            self.last_source_capture = now
        for event_id, future in list(self.pending.items()):
            if future.done():
                try:
                    self.completed.append(future.result())
                except Exception as error:
                    self.failures.append({"event_id":event_id, "error":str(error)[:500]})
                del self.pending[event_id]
        if self.last_discovery is None or (now-self.last_discovery).total_seconds() >= 900:
            try:
                self.discover(self.study, client=self.client, hours=24, clock=self.clock)
                self.last_discovery = now
                self.state["last_discovery"] = now.isoformat()
            except Exception as error:
                self.failures.append({"stage":"discovery", "error":str(error)[:500]})
                # Retry after one minute rather than hammering a blocked feed.
                self.last_discovery = now-timedelta(seconds=840)
        due, missed = due_events(self.study, now)
        self.state["missed"] = sorted(set(self.state["missed"])|{v["event_id"] for v in missed})
        if not discover_only:
            for event in due:
                identity = event["event_id"]
                if len(self.pending) >= self.workers:
                    break
                if identity in self.pending or self.state["attempts"].get(identity,0) >= self.max_attempts:
                    continue
                self.state["attempts"][identity] = self.state["attempts"].get(identity,0)+1
                replace_json(self.state_path, self.state)
                self.pending[identity] = self.executor.submit(self.capture, self.study, identity,
                    client=self.client, clock=self.clock, wait_for_issue_window=True)
        replace_json(self.state_path, self.state)
        heartbeat = {"observed_at":timestamp(self.clock()).isoformat(), "study":str(self.study.path),
            "model_id":self.study.protocol["model_id"], "last_discovery":self.state.get("last_discovery"),
            "active_events":list(self.pending), "launched_attempts":sum(self.state["attempts"].values()),
            "events_with_missed_window":len(self.state["missed"]),
            "selected_pairs":len(list(self.study.selected_records())), "recent_errors":self.failures[-10:],
            "rtv_capture": self.source_status,
            "mode":"discovery_only" if discover_only else "recording", "pid":os.getpid()}
        replace_json(self.study.path/"recorder_heartbeat.json", heartbeat)
        return heartbeat

    def close(self):
        self.executor.shutdown(wait=True, cancel_futures=True)
        if self.source_executor:
            self.source_executor.shutdown(wait=True, cancel_futures=True)


def refresh_history(study):
    """Normal settled-history refresh; never update the pinned model's weights."""
    study.verify()
    from src.source_pipeline import sync_database
    from src.data_processor import process_data
    from src.store_refresh import append_current_store
    sync_database(days_back=3, enrich=False)
    # Match the ordinary CSV ingest contract (including numeric IDs, missing
    # strings and float parsing), rather than fingerprinting SQLite TEXT IDs.
    processed = process_data(save=False)
    if not append_current_store(processed):
        # A recorder must not silently start a full historical rebuild. Such
        # changes require maintenance and possibly a newly qualified model.
        raise RuntimeError("History cannot be appended safely; a maintenance rebuild is required before recording")
    study.verify()


def collect_withdrawals(study, *, client, clock=utcnow):
    """Save explicit later withdrawal flags and original API receipts."""
    from src.utils import normalise_horse_key
    from src.matchbook_signals import clean_runner_name
    from src.deployment_validation import file_sha256
    rows = []
    destination = study.path/"outcomes"
    destination.mkdir(exist_ok=True)
    for record in study.selected_records():
        if timestamp(record["first_start"]) >= timestamp(clock()):
            continue
        receipt = destination/(record["event_id"]+".json")
        if receipt.exists():
            saved = json.loads(receipt.read_text())
            if not saved.get("terminal"):
                # A partial earlier lookup must not permanently suppress a
                # later settled withdrawal observation.
                receipt = destination/(record["event_id"]+"_"+uuid4().hex[:8]+".json")
        else:
            saved = None
        if saved is None or not saved.get("terminal"):
            event = client.get_event(record["event_id"])
            payload = client.get_event_markets(record["event_id"], names="WIN", include_prices=False,
                                               per_page=20, currency="GBP")
            markets = [m for m in payload.get("markets",[]) if str(m.get("id")) == record["market_id"]]
            if len(markets) != 1 or str(event.get("id")) != record["event_id"]:
                raise ValueError("Settlement market identity differs from the issued market")
            market = markets[0]
            saved = {"event_id":record["event_id"], "market_id":record["market_id"],
                "observed_at":timestamp(clock()).isoformat(), "event":event, "payload":payload,
                "terminal":market.get("status") in {"closed", "settled", "graded"}}
            publish_json(receipt, saved)
        # Use the exact issued exchange runner IDs as well as normalized names.
        issued = pd.read_parquet(study.path/"records"/record["capture_id"]/"forecasts.parquet")
        by_id = dict(zip(issued.runner_id.astype(str), issued.horse_key))
        for market in saved["payload"].get("markets",[]):
            if str(market.get("id")) != record["market_id"]:
                continue
            for runner in market.get("runners",[]):
                key = by_id.get(str(runner.get("id")))
                if runner.get("withdrawn") is True and key:
                    source_key = normalise_horse_key(clean_runner_name(runner.get("name", "")))
                    if source_key != key:
                        raise ValueError("Withdrawn runner name and issued identity disagree")
                    rows.append({"event_id":record["event_id"], "horse_key":key, "confirmed_withdrawn":True,
                        "runner_id":str(runner["id"]), "market_id":record["market_id"],
                        "source":str(receipt.relative_to(study.path)), "source_sha256":file_sha256(receipt),
                        "observed_at":saved["observed_at"]})
    frame = pd.DataFrame(rows, columns=["event_id", "horse_key", "confirmed_withdrawn", "runner_id", "market_id",
                                       "source", "source_sha256", "observed_at"])
    target = destination/"withdrawals.parquet"
    if target.exists():
        # A later API response may no longer expose an old market. Preserve
        # previously verified source observations instead of deleting labels.
        previous = pd.read_parquet(target)
        frame = previous if frame.empty else (frame if previous.empty else pd.concat([previous,frame],ignore_index=True))
    if not frame.empty:
        frame = frame.drop_duplicates(["event_id", "horse_key"])
    temporary = target.with_suffix(".tmp.parquet")
    frame.to_parquet(temporary, index=False)
    os.replace(temporary, target)
    return frame


def run_recorder(study, *, client, duration_minutes=1080, refresh=False, discover_only=False,
                 clock=utcnow, sleeper=time.sleep):
    if not 0 < duration_minutes <= 1440:
        raise ValueError("A recorder session must last at most24hours")
    with recorder_lock(study):
        if refresh:
            refresh_history(study)
        from src.source_pipeline import capture_rtv_snapshot
        recorder = Recorder(study, client=client, clock=clock, source_capture=capture_rtv_snapshot)
        deadline = time.monotonic()+duration_minutes*60
        last_assessment = None
        try:
            while True:
                heartbeat = recorder.tick(discover_only=discover_only)
                print(json.dumps(heartbeat), flush=True)
                moment = timestamp(clock())
                if not recorder.pending and (last_assessment is None or (moment-last_assessment).total_seconds() >= 3600):
                    try:
                        collect_withdrawals(study, client=client, clock=clock)
                    except Exception as error:
                        recorder.failures.append({"stage":"withdrawal_observations", "error":str(error)[:500]})
                    # Unavailable withdrawal evidence must not suppress scoring
                    # unchanged fields from already validated local results.
                    try:
                        from src.forward_assessment import save_assessment
                        from src.database import DB_PATH
                        save_assessment(study, DB_PATH)
                        last_assessment = moment
                    except Exception as error:
                        recorder.failures.append({"stage":"assessment", "error":str(error)[:500]})
                        last_assessment = moment-timedelta(seconds=3300)
                if discover_only or time.monotonic() >= deadline:
                    break
                sleeper(min(15., max(0., deadline-time.monotonic())))
        except BaseException as error:
            replace_json(study.path/"recorder_heartbeat.json", {"observed_at":timestamp(clock()).isoformat(),
                "mode":"stopped", "error":str(error)[:500], "pid":os.getpid()})
            raise
        finally:
            recorder.close()
            previous = study.path/"recorder_heartbeat.json"
            final = json.loads(previous.read_text()) if previous.exists() else {}
            replace_json(previous, {**final, "mode":"stopped", "active_events":[],
                "observed_at":timestamp(clock()).isoformat(), "pid":os.getpid()})
        return heartbeat
