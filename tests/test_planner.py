"""Unit tests for the read-only itinerary planning engine."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from app.planner import busy_intervals, earliest_open_slot, plan_itinerary
from app.scheduler import (
    MakeupRule,
    RuleSet,
    WeeklyRule,
)
from app.planner import StopSpec

UTC = timezone.utc
T = lambda h, m=0: datetime(2026, 10, 2, h, m, tzinfo=UTC)  # noqa: E731


def test_busy_intervals_capacity_one_touching_chained_full():
    # One visit ends exactly when another starts: capacity stays 1 at 10:00
    # (the first is released but the second already holds the slot), so the
    # whole [09:00,11:00) is full — a third visitor must wait until 11:00.
    busy = busy_intervals([(T(9), T(10)), (T(10), T(11))], capacity=1)
    assert busy == [(T(9), T(11))]
    # With capacity 2 the chain never saturates anything.
    assert busy_intervals([(T(9), T(10)), (T(10), T(11))], capacity=2) == []


def test_busy_intervals_overlapping_merge():
    busy = busy_intervals(
        [(T(9), T(10, 30)), (T(10), T(11))], capacity=2
    )
    assert busy == [(T(10), T(10, 30))]


def test_busy_intervals_touching_full_runs_merge():
    busy = busy_intervals(
        [(T(9), T(10)), (T(10), T(11)), (T(9, 30), T(10, 30))],
        capacity=2,
    )
    # Two visits overlap on [09:30,10:30); after 10:30 only the third visit
    # (10:00-11:00) remains, so the full run ends at 10:30.
    assert busy == [(T(9, 30), T(10, 30))]


def test_earliest_slot_simple():
    s = earliest_open_slot([(T(9), T(17))], [], timedelta(minutes=30), T(9), T(17))
    assert s == T(9)


def test_earliest_slot_respects_lower_bound():
    s = earliest_open_slot(
        [(T(9), T(17))], [], timedelta(minutes=30), T(10), T(17)
    )
    assert s == T(10)


def test_earliest_slot_jumps_past_busy_and_touches_free():
    busy = busy_intervals([(T(9), T(10))], 1)
    s = earliest_open_slot(
        [(T(9), T(17))], busy, timedelta(minutes=30), T(9), T(17)
    )
    assert s == T(10)  # touching at 10:00 is free


def test_earliest_slot_busy_at_open_then_open_again():
    busy = busy_intervals([(T(10), T(11))], 1)
    s = earliest_open_slot(
        [(T(9), T(12))], busy, timedelta(minutes=60), T(9), T(12)
    )
    # 09:00 start ends at 10:00 — touches the full run, free.
    assert s == T(9)


def test_earliest_slot_backtracks_to_touch_earlier_run():
    # Two full runs: a 90-minute window that reaches into the second run
    # moves back to 09:00 (ends 10:30 touching run 2); that window overlaps
    # run 1 (09:00-09:45), so it must move to 11:00.  A naive single pass
    # that does not rescan after moving backwards would wrongly accept 09:00.
    busy = busy_intervals(
        [(T(9), T(9, 45)), (T(10, 30), T(11))], 1
    )
    s = earliest_open_slot(
        [(T(8), T(13))], busy, timedelta(minutes=90), T(8, 30), T(13)
    )
    assert s == T(11)
    # Sanity: a short window at 08:30 clears both runs.
    s = earliest_open_slot(
        [(T(8), T(13))], busy, timedelta(minutes=15), T(8, 30), T(13)
    )
    assert s == T(8, 30)



def test_earliest_slot_must_not_overlap_busy():
    busy = busy_intervals([(T(9, 30), T(11))], 1)
    s = earliest_open_slot(
        [(T(9), T(12))], busy, timedelta(minutes=60), T(9), T(12)
    )
    assert s == T(11)


def test_earliest_slot_no_room_before_horizon():
    # A full run ending exactly at the horizon leaves exactly one 60-minute
    # slot touching both boundaries: start at 10:30, end at 11:30.
    busy = busy_intervals([(T(9, 30), T(10, 30))], 1)
    s = earliest_open_slot(
        [(T(9), T(12))], busy, timedelta(minutes=60), T(9), T(11, 30)
    )
    assert s == T(10, 30)
    # Push the horizon one minute earlier: no room for a 60-minute visit.
    s = earliest_open_slot(
        [(T(9), T(12))], busy, timedelta(minutes=60), T(9), T(11, 29)
    )
    assert s is None
    # A full run that ends at the horizon itself leaves nothing after it.
    busy = busy_intervals([(T(9, 30), T(12))], 1)
    s = earliest_open_slot(
        [(T(9), T(12))], busy, timedelta(minutes=60), T(9), T(12)
    )
    assert s is None


def test_earliest_slot_spans_multiple_open_intervals():
    s = earliest_open_slot(
        [(T(9), T(9, 30)), (T(10), T(17))], [],
        timedelta(minutes=60), T(9), T(17),
    )
    assert s == T(10)


def spec(point_id, tz, ruleset, reservations=(), capacity=1, version=1):
    return StopSpec(
        point_id=point_id,
        timezone=tz,
        capacity=capacity,
        rules_version=version,
        ruleset=ruleset,
        reservations=tuple(reservations),
    )


DAILY = None  # placeholder removed below


def open_all_ruleset():
    # A weekly rule covering Monday 00:00-24:00 is not expressible (end ==
    # start), so use a makeup-free long Monday session 00:00->23:59 plus
    # overnight Monday 23:59->23:58 would be odd; instead use makeup on the
    # specific day 2026-10-02 (a Friday) 00:00-23:59.
    return RuleSet(
        weekly=(),
        closed_dates=frozenset(),
        makeup=(MakeupRule(datetime(2026, 10, 2, tzinfo=UTC).date(),
                           datetime(2026, 10, 2, 0, 0).time(),
                           datetime(2026, 10, 2, 23, 59).time()),),
    )


def test_plan_lexicographically_earliest_two_stops():
    rs = open_all_ruleset()
    planned = plan_itinerary(
        [spec(1, "UTC", rs), spec(2, "UTC", rs)],
        [timedelta(minutes=30), timedelta(minutes=30)],
        timedelta(minutes=15),
        search_from=T(8),
        search_to=T(20),
        now=T(8),
        horizon=T(23),
    )
    assert [(p.start, p.end) for p in planned] == [
        (T(8), T(8, 30)),
        (T(8, 45), T(9, 15)),
    ]


def test_plan_second_stop_waits_for_open():
    # Point 2 is open only 12:00-13:00.
    rs1 = open_all_ruleset()
    rs2 = RuleSet(
        weekly=(),
        closed_dates=frozenset(),
        makeup=(MakeupRule(datetime(2026, 10, 2, tzinfo=UTC).date(),
                           datetime(2026, 10, 2, 12, 0).time(),
                           datetime(2026, 10, 2, 13, 0).time()),),
    )
    planned = plan_itinerary(
        [spec(1, "UTC", rs1), spec(2, "UTC", rs2)],
        [timedelta(minutes=30), timedelta(minutes=30)],
        timedelta(0),
        search_from=T(8),
        search_to=T(20),
        now=T(8),
        horizon=T(23),
    )
    assert planned[0].start == T(8)
    assert planned[1].start == T(12)


def test_plan_first_stop_busy_pushes_whole_chain():
    rs = open_all_ruleset()
    planned = plan_itinerary(
        [spec(1, "UTC", rs, reservations=((T(8), T(9)),)),
         spec(2, "UTC", rs)],
        [timedelta(minutes=30), timedelta(minutes=30)],
        timedelta(minutes=10),
        search_from=T(8),
        search_to=T(20),
        now=T(8),
        horizon=T(23),
    )
    assert planned[0].start == T(9)
    assert planned[1].start == T(9, 40)


def test_plan_no_solution_when_first_stop_never_open():
    rs1 = RuleSet(
        weekly=(), closed_dates=frozenset(), makeup=()
    )
    rs2 = open_all_ruleset()
    planned = plan_itinerary(
        [spec(1, "UTC", rs1), spec(2, "UTC", rs2)],
        [timedelta(minutes=30), timedelta(minutes=30)],
        timedelta(0),
        search_from=T(8),
        search_to=T(20),
        now=T(8),
        horizon=T(23),
    )
    assert planned is None


def test_plan_search_window_clips_group():
    rs = open_all_ruleset()
    planned = plan_itinerary(
        [spec(1, "UTC", rs), spec(2, "UTC", rs)],
        [timedelta(minutes=30), timedelta(minutes=30)],
        timedelta(0),
        search_from=T(8),
        search_to=T(9),  # second stop must end by 09:00
        now=T(8),
        horizon=T(23),
    )
    assert planned[1].end == T(9)
    planned = plan_itinerary(
        [spec(1, "UTC", rs), spec(2, "UTC", rs)],
        [timedelta(minutes=30), timedelta(minutes=30)],
        timedelta(0),
        search_from=T(8),
        search_to=T(8, 59),
        now=T(8),
        horizon=T(23),
    )
    assert planned is None


def test_plan_respects_now_floor_and_horizon():
    rs = open_all_ruleset()
    planned = plan_itinerary(
        [spec(1, "UTC", rs), spec(2, "UTC", rs)],
        [timedelta(minutes=30), timedelta(minutes=30)],
        timedelta(0),
        search_from=T(7),
        search_to=T(20),
        now=T(9),
        horizon=T(23),
    )
    assert planned[0].start == T(9)
    planned = plan_itinerary(
        [spec(1, "UTC", rs), spec(2, "UTC", rs)],
        [timedelta(minutes=30), timedelta(minutes=30)],
        timedelta(0),
        search_from=T(7),
        search_to=T(23),
        now=T(8),
        horizon=T(8, 45),
    )
    assert planned is None


def test_plan_reservation_started_before_lower_still_holds_capacity():
    # A reservation that began before the search window still occupies
    # [08:00,09:00): the planner must not ignore its missing start event.
    rs = open_all_ruleset()
    planned = plan_itinerary(
        [spec(1, "UTC", rs, reservations=((T(7), T(9)),)),
         spec(2, "UTC", rs)],
        [timedelta(minutes=30), timedelta(minutes=30)],
        timedelta(0),
        search_from=T(8),
        search_to=T(20),
        now=T(8),
        horizon=T(23),
    )
    assert planned[0].start == T(9)
    assert planned[1].start == T(9, 30)


def test_plan_capacity_two_allows_overlap():
    # With capacity 2 a single visitor never saturates the point, so every
    # start inside the open interval is available.
    rs = open_all_ruleset()
    planned = plan_itinerary(
        [spec(1, "UTC", rs, reservations=((T(8), T(10),),), capacity=2),
         spec(2, "UTC", rs)],
        [timedelta(minutes=30), timedelta(minutes=30)],
        timedelta(0),
        search_from=T(8),
        search_to=T(20),
        now=T(8),
        horizon=T(23),
    )
    assert planned[0].start == T(8)
    assert busy_intervals([(T(8), T(10))], 2) == []
