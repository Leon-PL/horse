from datetime import datetime, timedelta, timezone

import pytest

from src.forward_timing import plan_window, wait_for_window


class Clock:
    def __init__(self, now):
        self.now = now
        self.elapsed = 0.
        self.sleeps = []

    def read(self):
        return self.now

    def steady(self):
        return self.elapsed

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.elapsed += seconds
        self.now += timedelta(seconds=seconds)


def specification(now=None):
    protocol = {"horizon_seconds": 300, "capture_window_seconds": 30,
                "clock_skew_tolerance_seconds": 2, "timed_max_wait_seconds": 600}
    first = {"first_start": "2026-10-28T17:00:00Z", "first_seen_at": "2026-10-28T15:00:00Z"}
    event = {"start": first["first_start"]}
    return protocol, first, event


def test_bounded_wait_ends_inside_window_before_any_fresh_final_reads():
    clock = Clock(datetime(2026, 10, 28, 16, 54, tzinfo=timezone.utc))
    plan = plan_window(*specification(), now=clock.read())
    receipt = wait_for_window(plan, clock=clock.read, sleeper=clock.sleep, monotonic=clock.steady)
    assert receipt["finished_at"] == "2026-10-28T16:54:32+00:00"
    assert receipt["wait_seconds"] == 32
    assert max(clock.sleeps) <= 1
    assert plan["close"] == "2026-10-28T16:55:00+00:00"


@pytest.mark.parametrize("problem,expected", [("early", "too_early"), ("late", "decision_window_missed"),
    ("schedule", "schedule_changed"), ("late_seen", "schedule_first_seen_too_late")])
def test_wait_preflight_rejects_unusable_windows(problem, expected):
    protocol, first, event = specification()
    now = "2026-10-28T16:54:00Z"
    if problem == "early": now = "2026-10-28T16:40:00Z"
    elif problem == "late": now = "2026-10-28T16:55:01Z"
    elif problem == "schedule": event["start"] = "2026-10-28T17:01:00Z"
    else: first["first_seen_at"] = "2026-10-28T16:54:31Z"
    with pytest.raises(ValueError, match=expected): plan_window(protocol, first, event, now=now)


@pytest.mark.parametrize("jump", [-5, 5, 120])
def test_clock_jumps_and_machine_delays_cannot_publish_a_timed_pair(jump):
    clock = Clock(datetime(2026, 10, 28, 16, 54, tzinfo=timezone.utc))
    plan = plan_window(*specification(), now=clock.read())
    def changed(seconds):
        clock.sleep(seconds)
        clock.now += timedelta(seconds=jump)
    with pytest.raises(ValueError, match="clock_changed|decision_window_missed"):
        wait_for_window(plan, clock=clock.read, sleeper=changed, monotonic=clock.steady)


def test_late_native_computation_fails_without_waiting():
    plan = plan_window(*specification(), now="2026-10-28T16:54:00Z")
    clock = Clock(datetime(2026, 10, 28, 16, 55, 1, tzinfo=timezone.utc))
    with pytest.raises(ValueError, match="decision_window_missed"):
        wait_for_window(plan, clock=clock.read, sleeper=clock.sleep, monotonic=clock.steady)
    assert clock.sleeps == []
