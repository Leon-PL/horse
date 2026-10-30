"""Bounded waiting within one manually started forecast attempt."""
from datetime import timedelta
import time

from src.forward_store import timestamp, utcnow


def plan_window(protocol, first, event, *, now=None):
    moment = timestamp(now or utcnow())
    start = timestamp(first["first_start"])
    close = start-timedelta(seconds=protocol["horizon_seconds"])
    opening = close-timedelta(seconds=protocol["capture_window_seconds"])
    target = opening+timedelta(seconds=2)
    maximum = protocol.get("timed_max_wait_seconds", 600)
    if maximum <= 0 or target > close:
        raise ValueError("invalid_timed_capture_window")
    if timestamp(event["start"]) != start:
        raise ValueError("schedule_changed")
    if timestamp(first["first_seen_at"]) > opening:
        raise ValueError("schedule_first_seen_too_late")
    if moment > close:
        raise ValueError("decision_window_missed")
    if (target-moment).total_seconds() > maximum:
        raise ValueError("too_early_for_bounded_capture_start_within_ten_minutes_of_window")
    return {"opening": opening.isoformat(), "target": target.isoformat(), "close": close.isoformat(),
            "planned_at": moment.isoformat(), "maximum_wait_seconds": maximum,
            "clock_skew_tolerance_seconds": protocol["clock_skew_tolerance_seconds"]}


def wait_for_window(plan, *, clock=utcnow, sleeper=time.sleep, monotonic=time.monotonic):
    target, close = timestamp(plan["target"]), timestamp(plan["close"])
    began = timestamp(clock())
    steady_start = monotonic()
    moment = began
    while moment < target:
        remaining = (target-moment).total_seconds()
        if remaining > plan["maximum_wait_seconds"]:
            raise ValueError("timed_capture_wait_limit_exceeded")
        sleeper(min(1., remaining))
        moment = timestamp(clock())
        elapsed = monotonic()-steady_start
        if moment > close:
            raise ValueError("decision_window_missed")
        if elapsed < 0 or elapsed > plan["maximum_wait_seconds"]+2:
            raise ValueError("timed_capture_wait_limit_exceeded")
        if abs((moment-began).total_seconds()-elapsed) > plan["clock_skew_tolerance_seconds"]:
            raise ValueError("clock_changed_during_wait")
    if moment > close:
        raise ValueError("decision_window_missed")
    return {"started_at": began.isoformat(), "finished_at": moment.isoformat(),
            "wait_seconds": (moment-began).total_seconds(), "target": plan["target"],
            "scope": "One manually started attempt; fresh card and market reads follow this wait"}
