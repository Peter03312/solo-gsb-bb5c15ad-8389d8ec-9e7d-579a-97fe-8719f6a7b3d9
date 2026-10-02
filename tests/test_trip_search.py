"""End-to-end tests for read-only earliest-itinerary search (行程可行性搜索).

Covers ``POST /trips/search``:
* earliest feasible itinerary under open intervals, per-point capacity and
  the minimum inter-stop gap; half-open ``[start, end)`` occupancy including
  touching boundaries;
* whole group inside the (timezone-aware) search range and the existing
  reservation lead-time horizon; cross-midnight sessions, skipped closed
  days and DST;
* distinguishable per-stop errors (missing point / unpublished rules /
  over-long stay) and structural parameter errors;
* read-only semantics: nothing is booked, the result is a snapshot;
* the planner's greedy result is actually submittable to ``POST /trips``.
"""

from __future__ import annotations

import itertools
from datetime import datetime, timedelta, timezone, date

import pytest

from app import trip_planner
from app.trip_planner import StopPlanInput, hot_intervals, plan_earliest_trip

UTC = timezone.utc

_keys = itertools.count(1)


def unique_key(prefix="search"):
    return f"{prefix}-{next(_keys)}"


def iso(dt):
    return dt.isoformat().replace("+00:00", "Z")


def day(day_offset):
    return (datetime.now(UTC) + timedelta(days=day_offset)).date()


def at(hour, minute=0, day_offset=1):
    d = day(day_offset)
    return datetime(d.year, d.month, d.day, hour, minute, tzinfo=UTC)


# ---------------------------------------------------------------------------
# API helpers
# ---------------------------------------------------------------------------


def create_point(client, tz="UTC", capacity=None):
    payload = {"name": "搜索测试点", "timezone": tz}
    if capacity is not None:
        payload["capacity"] = capacity
    r = client.post("/points", json=payload)
    assert r.status_code == 201, r.text
    return r.json()["id"]


def publish(client, pid, weekly, closed=None, makeup=None, expected_version=0):
    r = client.post(
        f"/points/{pid}/rules",
        json={
            "expected_version": expected_version,
            "rules": {
                "weekly": weekly,
                "closed_dates": closed or [],
                "makeup": makeup or [],
            },
        },
    )
    assert r.status_code == 201, r.text
    return r.json()["version"]


def weekly(day_offset, start, end):
    return {
        "weekday": day(day_offset).isoweekday(),
        "start_time": start,
        "end_time": end,
    }


def open_point(client, start="08:00", end="12:00", day_offset=1, tz="UTC",
               capacity=None):
    pid = create_point(client, tz=tz, capacity=capacity)
    publish(client, pid, [weekly(day_offset, start, end)])
    return pid


def search(client, stops, start, end, gap=0):
    return client.post(
        "/trips/search",
        json={
            "stops": stops,
            "min_gap_minutes": gap,
            "search_start": iso(start),
            "search_end": iso(end),
        },
    )


def stop(pid, minutes):
    return {"point_id": pid, "duration_minutes": minutes}


def reserve(client, pid, start, end, key=None):
    return client.post(
        f"/points/{pid}/reservations",
        json={
            "idempotency_key": key or unique_key("resv"),
            "start": iso(start),
            "end": iso(end),
        },
    )


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


def test_search_returns_earliest_itinerary_back_to_back(client):
    p1 = open_point(client)
    p2 = open_point(client)
    r = search(
        client,
        [stop(p1, 30), stop(p2, 30)],
        at(0), at(23, 59), gap=0,
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["trip"] is not None
    got = body["trip"]
    assert [s["point_id"] for s in got] == [p1, p2]
    assert got[0]["start"] == iso(at(8))
    assert got[0]["end"] == iso(at(8, 30))
    assert got[1]["start"] == iso(at(8, 30))  # touching is allowed
    assert got[1]["end"] == iso(at(9))
    assert {s["rules_version"] for s in got} == {1}
    assert body["min_gap_minutes"] == 0
    assert body["search_start"].endswith("Z") and body["search_end"].endswith("Z")


def test_search_respects_min_gap(client):
    p1, p2 = open_point(client), open_point(client)
    r = search(
        client, [stop(p1, 30), stop(p2, 30)], at(0), at(23, 59), gap=60
    )
    assert r.status_code == 200, r.text
    got = r.json()["trip"]
    assert got[0]["end"] == iso(at(8, 30))
    assert got[1]["start"] == iso(at(9, 30))  # 60 minutes after stop 1 ends


def test_search_skips_occupied_window_but_allows_touching(client):
    p1, p2 = open_point(client), open_point(client)
    # [08:00, 09:00) occupied: a 30-minute stop may not overlap, so its start
    # is pushed to 09:00 (the [07:30, 08:00) morning slot is before opening).
    assert reserve(client, p1, at(8), at(9)).status_code == 201
    r = search(
        client, [stop(p1, 30), stop(p2, 30)], at(0), at(23, 59), gap=0
    )
    got = r.json()["trip"]
    assert got[0]["start"] == iso(at(9))
    assert got[0]["end"] == iso(at(9, 30))
    assert got[1]["start"] == iso(at(9, 30))


def test_search_touching_reservation_is_feasible(client):
    p1, p2 = open_point(client), open_point(client)
    # [08:30, 09:00) occupied: a [08:00, 08:30) stop merely touches it.
    assert reserve(client, p1, at(8, 30), at(9)).status_code == 201
    r = search(client, [stop(p1, 30), stop(p2, 30)], at(0), at(23, 59))
    got = r.json()["trip"]
    assert got[0]["start"] == iso(at(8))
    assert got[0]["end"] == iso(at(8, 30))


def test_search_capacity_two_peak_concurrency(client):
    p1 = open_point(client, capacity=2)
    p2 = open_point(client)
    # One carrier leaves room: the earliest 30-minute stop is still at 08:00.
    assert reserve(client, p1, at(8, 30), at(9, 30)).status_code == 201
    r = search(client, [stop(p1, 30), stop(p2, 30)], at(0), at(23, 59))
    assert r.json()["trip"][0]["start"] == iso(at(8))

    # A second carrier covering [08:00,10:00) saturates capacity 2 over
    # [08:30,09:30): the stop moves to the first touching instant after.
    assert reserve(client, p1, at(8), at(10)).status_code == 201
    r = search(client, [stop(p1, 30), stop(p2, 30)], at(0), at(23, 59))
    got = r.json()["trip"]
    assert got[0]["start"] == iso(at(8))  # [08:00,08:30) still has room
    assert got[0]["end"] == iso(at(8, 30))

    # Fill that too: only the post-peak slot remains.
    assert reserve(client, p1, at(8), at(8, 30)).status_code == 201
    r = search(client, [stop(p1, 30), stop(p2, 30)], at(0), at(23, 59))
    got = r.json()["trip"]
    assert got[0]["start"] == iso(at(9, 30))


def test_search_no_solution_returns_null(client):
    # Open only 08:00-08:45; two 30-minute stops back-to-back cannot fit.
    p1 = open_point(client, start="08:00", end="08:45")
    p2 = open_point(client, start="08:00", end="08:45")
    r = search(client, [stop(p1, 30), stop(p2, 30)], at(0), at(23, 59))
    assert r.status_code == 200
    assert r.json()["trip"] is None


def test_search_end_boundary_is_inclusive_via_half_open_window(client):
    # The last stop's end may equal search_end (the group window is inside
    # the search range; reservations occupy [start, end)).
    p1, p2 = open_point(client), open_point(client)
    r = search(
        client, [stop(p1, 30), stop(p2, 30)], at(0), at(9), gap=0
    )
    got = r.json()["trip"]
    assert [s["start"] for s in got] == [iso(at(8)), iso(at(8, 30))]
    assert got[1]["end"] == iso(at(9))


def test_search_group_extending_past_search_end_is_null(client):
    p1, p2 = open_point(client), open_point(client)
    r = search(
        client, [stop(p1, 30), stop(p2, 30)], at(0), at(8, 59), gap=0
    )
    assert r.status_code == 200
    assert r.json()["trip"] is None


def test_search_start_lower_bounds_first_stop(client):
    p1, p2 = open_point(client), open_point(client)
    r = search(
        client, [stop(p1, 30), stop(p2, 30)], at(9, 15), at(23, 59)
    )
    got = r.json()["trip"]
    assert got[0]["start"] == iso(at(9, 15))
    assert got[1]["start"] == iso(at(9, 45))


def test_search_picks_earliest_across_multiple_open_sessions(client):
    p1 = create_point(client)
    publish(client, p1, [weekly(1, "08:00", "09:00"), weekly(1, "13:00", "14:00")])
    p2 = open_point(client, start="09:00", end="10:00")
    # Greedy: stop 1 at 08:00 (first session), stop 2 earliest >= 08:30 in
    # its own 09:00-10:00 session -> 09:00.
    r = search(client, [stop(p1, 30), stop(p2, 30)], at(0), at(23, 59))
    got = r.json()["trip"]
    assert got[0]["start"] == iso(at(8))
    assert got[1]["start"] == iso(at(9))


def test_search_waits_for_later_opening_day(client):
    # Point 1 is shut tomorrow and open the day after; the itinerary starts
    # on the later day even though point 2 is open tomorrow.
    p1 = create_point(client)
    publish(
        client, p1,
        [weekly(1, "08:00", "12:00"), weekly(2, "08:00", "12:00")],
        closed=[day(1).isoformat()],
    )
    p2 = create_point(client)
    publish(
        client, p2,
        [weekly(1, "08:00", "12:00"), weekly(2, "08:00", "12:00")],
    )
    r = search(
        client, [stop(p1, 30), stop(p2, 30)], at(0, day_offset=1),
        at(23, 59, day_offset=3),
    )
    assert r.status_code == 200, r.text
    got = r.json()["trip"]
    assert got[0]["start"] == iso(at(8, day_offset=2))
    assert got[1]["start"] == iso(at(8, 30, day_offset=2))


def test_search_closed_date_is_skipped(client):
    p1 = create_point(client)
    # Same weekday rules for tomorrow and the same weekday next week would
    # duplicate; instead open tomorrow and day+2 (different weekdays).
    publish(
        client, p1,
        [weekly(1, "08:00", "12:00"), weekly(2, "08:00", "12:00")],
        closed=[day(1).isoformat()],
    )
    p2 = open_point(client, day_offset=2)
    r = search(
        client, [stop(p1, 30), stop(p2, 30)],
        at(0, day_offset=1), at(23, 59, day_offset=2),
    )
    got = r.json()["trip"]
    assert got[0]["start"] == iso(at(8, day_offset=2))


def test_search_overnight_session_spans_midnight(client):
    # Friday 20:00 to Saturday 02:00 local/UTC; the stop may straddle
    # midnight because it uses continuous (unclipped) intervals.
    fri = (datetime.now(UTC) + timedelta(days=4)).date()  # nearest Friday
    while fri.isoweekday() != 5:
        fri += timedelta(days=1)
    p1 = create_point(client)
    pid = p1
    r = client.post(
        f"/points/{pid}/rules",
        json={
            "expected_version": 0,
            "rules": {
                "weekly": [
                    {"weekday": 5, "start_time": "20:00", "end_time": "02:00"}
                ],
                "closed_dates": [],
                "makeup": [],
            },
        },
    )
    assert r.status_code == 201, r.text
    p2 = create_point(client)
    r = client.post(
        f"/points/{p2}/rules",
        json={
            "expected_version": 0,
            "rules": {
                "weekly": [
                    {"weekday": 6, "start_time": "00:30", "end_time": "05:00"}
                ],
                "closed_dates": [],
                "makeup": [],
            },
        },
    )
    assert r.status_code == 201, r.text

    def dt(d, h, m=0):
        return datetime(d.year, d.month, d.day, h, m, tzinfo=UTC)

    r = search(
        client,
        [stop(p1, 60), stop(p2, 30)],
        dt(fri, 19), dt(fri + timedelta(days=1), 6), gap=0,
    )
    got = r.json()["trip"]
    assert got[0]["start"] == iso(dt(fri, 20))
    assert got[0]["end"] == iso(dt(fri, 21))
    assert got[1]["start"] == iso(dt(fri + timedelta(days=1), 0, 30))


def test_search_dst_fall_back_api_gaza(client):
    # Asia/Gaza falls back in the early Saturday local morning of
    # 2026-10-24 (01:00 local exists twice).  A 00:30-03:00 wall session
    # expands to [21:30Z, 01:00Z] (start early instance, end late), a 3h30
    # continuous span that counts the repeated hour.  This is within the
    # 30-day booking horizon, so it exercises the real HTTP + scheduler path.
    gaza = "Asia/Gaza"
    p1 = create_point(client, tz=gaza)
    publish(client, p1, [{"weekday": 6, "start_time": "00:30",
                          "end_time": "03:00"}])
    p2 = create_point(client, tz=gaza)
    publish(client, p2, [{"weekday": 6, "start_time": "00:30",
                          "end_time": "04:00"}])
    r = search(
        client, [stop(p1, 60), stop(p2, 60)],
        datetime(2026, 10, 23, 20, tzinfo=UTC),
        datetime(2026, 10, 24, 5, tzinfo=UTC), gap=0,
    )
    assert r.status_code == 200, r.text
    got = r.json()["trip"]
    # Early 00:30 (EEST, UTC+3) -> 21:30Z the previous day.
    assert got[0]["start"] == "2026-10-23T21:30:00Z"
    assert got[0]["end"] == "2026-10-23T22:30:00Z"
    assert got[1]["start"] == "2026-10-23T22:30:00Z"
    assert got[1]["end"] == "2026-10-23T23:30:00Z"


def test_search_dst_fall_back_uses_expanded_repeated_hour(client):
    # The calendar engine expands a New-York fall-back 01:00-05:00 wall
    # session as [05:00,10:00) UTC (early instance start, late instance end —
    # the repeated hour counts).  Such UTC intervals are all the planner sees,
    # so this verifies the trip search treats the expanded 5-hour stretch as
    # one continuous interval across the DST repeat.  (Fall-back Sunday
    # 2026-11-01 is beyond the 30-day booking horizon for an API call; the
    # engine's own DST resolution is covered by the scheduler test suite.)
    from app.scheduler import build_open_intervals, RuleSet, WeeklyRule
    from datetime import time as dtime

    ruleset = RuleSet(
        weekly=(WeeklyRule(7, dtime(1, 0), dtime(5, 0)),),
        closed_dates=frozenset(),
        makeup=(),
    )
    p1_intervals = tuple(
        build_open_intervals(
            "America/New_York", ruleset, date(2026, 10, 31), date(2026, 11, 1)
        )
    )
    assert p1_intervals == (
        (datetime(2026, 11, 1, 5, tzinfo=UTC),
         datetime(2026, 11, 1, 10, tzinfo=UTC)),
    )

    s1 = StopPlanInput(
        1, timedelta(minutes=60), 3, 1, p1_intervals, (),
    )
    s2 = StopPlanInput(
        2, timedelta(minutes=60), 3, 1,
        ((datetime(2026, 11, 1, 4, tzinfo=UTC),
          datetime(2026, 11, 1, 12, tzinfo=UTC)),),
        (),
    )
    planned = plan_earliest_trip(
        [s1, s2],
        search_start=datetime(2026, 10, 31, tzinfo=UTC),
        search_end=datetime(2026, 11, 1, 12, tzinfo=UTC),
        min_gap=timedelta(0),
        now=datetime(2026, 10, 1, tzinfo=UTC),
        max_days_ahead=365,
    )
    assert [(p.start, p.end) for p in planned] == [
        (datetime(2026, 11, 1, 5, tzinfo=UTC),
         datetime(2026, 11, 1, 6, tzinfo=UTC)),
        (datetime(2026, 11, 1, 6, tzinfo=UTC),
         datetime(2026, 11, 1, 7, tzinfo=UTC)),
    ]


def test_search_rules_version_refresh_after_republish(client):
    p1, p2 = open_point(client), open_point(client)
    publish(
        client, p1, [weekly(1, "09:00", "12:00")], expected_version=1
    )
    r = search(client, [stop(p1, 30), stop(p2, 30)], at(0), at(23, 59))
    got = r.json()["trip"]
    assert got[0]["rules_version"] == 2
    assert got[0]["start"] == iso(at(9))
    assert got[1]["rules_version"] == 1


def test_search_past_range_clamped_to_now(client):
    p1, p2 = open_point(client), open_point(client)
    # A range starting in the past must not yield past stops; the earliest
    # feasible itinerary is still tomorrow at opening.
    r = search(
        client, [stop(p1, 30), stop(p2, 30)],
        at(0, day_offset=-2), at(23, 59),
    )
    got = r.json()["trip"]
    assert got[0]["start"] == iso(at(8))


def test_search_entirely_in_past_is_null(client):
    p1, p2 = open_point(client), open_point(client)
    r = search(
        client, [stop(p1, 30), stop(p2, 30)],
        at(0, day_offset=-2), at(23, 59, day_offset=-1),
    )
    assert r.json()["trip"] is None


def test_search_beyond_bookable_horizon_is_null(client):
    # Nothing open within the 30-day lead limit: null, not a far-future trip.
    # The search window itself stays inside the 7-day query-span limit; the
    # bookable horizon is the binding constraint.  Rules repeat weekly, so
    # the first opening at +35 days has no earlier occurrence within the
    # 28..34-day search window (35 - 7 = 28 is just before its start edge).
    far = day(35)
    p1 = create_point(client)
    publish(client, p1, [{"weekday": far.isoweekday(), "start_time": "08:00",
                          "end_time": "12:00"}])
    p2 = create_point(client)
    publish(client, p2, [{"weekday": far.isoweekday(), "start_time": "08:00",
                          "end_time": "12:00"}])
    r = search(
        client, [stop(p1, 30), stop(p2, 30)],
        at(0, day_offset=29), at(23, 59, day_offset=34),
    )
    assert r.status_code == 200
    assert r.json()["trip"] is None


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


def test_search_point_not_found_names_stop(client):
    p1 = open_point(client)
    r = search(client, [stop(p1, 30), stop(999999, 30)], at(0), at(23, 59))
    assert r.status_code == 404
    err = r.json()["error"]
    assert err["code"] == "point_not_found"
    assert err["details"]["stop_index"] == 1
    assert err["details"]["point_id"] == 999999


def test_search_no_published_rules_names_stop(client):
    p1 = open_point(client)
    p2 = create_point(client)  # no rules
    r = search(client, [stop(p1, 30), stop(p2, 30)], at(0), at(23, 59))
    assert r.status_code == 409
    err = r.json()["error"]
    assert err["code"] == "no_published_version"
    assert err["details"]["stop_index"] == 1
    assert err["details"]["point_id"] == p2


def test_search_duration_over_cap_names_stop(client):
    p1, p2 = open_point(client), open_point(client)
    r = search(client, [stop(p1, 30), stop(p2, 121)], at(0), at(23, 59))
    assert r.status_code == 422
    err = r.json()["error"]
    assert err["code"] == "invalid_period"
    assert err["details"]["stop_index"] == 1
    assert err["details"]["point_id"] == p2
    assert err["details"]["duration_minutes"] == 121


def test_search_negative_duration_is_invalid_request(client):
    p1, p2 = open_point(client), open_point(client)
    r = client.post(
        "/trips/search",
        json={
            "stops": [stop(p1, -5), stop(p2, 30)],
            "search_start": iso(at(0)),
            "search_end": iso(at(23, 59)),
        },
    )
    assert r.status_code == 422
    assert r.json()["error"]["code"] == "invalid_request"


def test_search_duplicate_point_rejected(client):
    p1 = open_point(client)
    r = search(client, [stop(p1, 30), stop(p1, 30)], at(0), at(23, 59))
    assert r.status_code == 422
    err = r.json()["error"]
    assert err["code"] == "invalid_request"
    assert err["details"]["stop_indexes"] == [0, 1]


@pytest.mark.parametrize("count", [1, 6])
def test_search_stop_count_out_of_range(client, count):
    points = [open_point(client) for _ in range(6)]
    r = search(
        client, [stop(p, 30) for p in points[:count]], at(0), at(23, 59)
    )
    assert r.status_code == 422
    assert r.json()["error"]["code"] == "invalid_request"


def test_search_inverted_range_rejected(client):
    p1, p2 = open_point(client), open_point(client)
    r = search(client, [stop(p1, 30), stop(p2, 30)], at(12), at(8))
    assert r.status_code == 422
    assert r.json()["error"]["code"] == "invalid_period"


def test_search_span_over_seven_days_rejected(client):
    p1, p2 = open_point(client), open_point(client)
    r = search(
        client, [stop(p1, 30), stop(p2, 30)],
        at(0, day_offset=1), at(0, 1, day_offset=9),
    )
    assert r.status_code == 422
    err = r.json()["error"]
    assert err["code"] == "invalid_period"
    assert err["details"]["max_search_days"] == 7


def test_search_naive_timestamps_rejected(client):
    p1 = open_point(client)
    r = client.post(
        "/trips/search",
        json={
            "stops": [stop(p1, 30), stop(p1 + 1, 30)],
            "search_start": "2026-10-05T00:00:00",
            "search_end": "2026-10-05T23:59:00",
        },
    )
    assert r.status_code == 422
    assert r.json()["error"]["code"] == "invalid_request"


def test_search_negative_gap_rejected(client):
    p1, p2 = open_point(client), open_point(client)
    r = client.post(
        "/trips/search",
        json={
            "stops": [stop(p1, 30), stop(p2, 30)],
            "min_gap_minutes": -1,
            "search_start": iso(at(0)),
            "search_end": iso(at(23, 59)),
        },
    )
    assert r.status_code == 422
    assert r.json()["error"]["code"] == "invalid_request"


# ---------------------------------------------------------------------------
# Read-only / snapshot semantics
# ---------------------------------------------------------------------------


def test_search_independent_capacity_per_point(client):
    # Capacity is counted per point: even with both points at capacity 1,
    # stops at distinct points may share the same wall-clock window.
    p1, p2 = open_point(client, capacity=1), open_point(client, capacity=1)
    assert reserve(client, p1, at(8), at(9)).status_code == 201
    assert reserve(client, p2, at(8), at(9)).status_code == 201
    # Simultaneous 30-minute stops at 08:15 across the two distinct points
    # are both feasible; the minimum gap 0 applies end->next start, so the
    # second stop must still follow the first: place it at the touching 08:45
    # window of point 2? [08:45,09:15) would overlap point 2's reservation
    # [08,09). Instead expect the greedy result to move both stops past 09:00
    # independently but identically: 09:00 then 09:30 (touching gap 0).
    r = search(
        client, [stop(p1, 30), stop(p2, 30)], at(0), at(23, 59), gap=0
    )
    got = r.json()["trip"]
    assert [(s["start"], s["end"]) for s in got] == [
        (iso(at(9)), iso(at(9, 30))),
        (iso(at(9, 30)), iso(at(10))),
    ]
    # With a gap of -0 not allowed; with gap 0 the stops merely touch, and
    # crucially stop 2's 09:30 window does not count stop 1's reservation.


def test_search_creates_nothing_and_is_bookable(client):
    p1, p2 = open_point(client), open_point(client)
    r = search(
        client, [stop(p1, 30), stop(p2, 30)], at(0), at(23, 59), gap=15
    )
    assert r.status_code == 200
    got = r.json()["trip"]

    # Nothing was booked: the same window remains available afterwards.
    r2 = search(
        client, [stop(p1, 30), stop(p2, 30)], at(0), at(23, 59), gap=15
    )
    assert r2.json()["trip"] == got

    # The planned itinerary satisfies the booking endpoint exactly.
    booked = client.post(
        "/trips",
        json={
            "idempotency_key": unique_key("book-after-search"),
            "stops": [
                {"point_id": s["point_id"], "start": s["start"],
                 "end": s["end"]}
                for s in got
            ],
        },
    )
    assert booked.status_code == 201, booked.text

    # After booking, a fresh search for the same itinerary must move away
    # from the now-occupied first slot.
    r3 = search(
        client, [stop(p1, 30), stop(p2, 30)], at(0), at(23, 59), gap=15
    )
    assert r3.json()["trip"][0]["start"] != got[0]["start"]


def test_search_does_not_consume_idempotency_namespace(client):
    # Searches carry no idempotency key and create no trip rows.
    p1, p2 = open_point(client), open_point(client)
    search(client, [stop(p1, 30), stop(p2, 30)], at(0), at(23, 59))
    search(client, [stop(p1, 30), stop(p2, 30)], at(0), at(23, 59))
    booked = client.post(
        "/trips",
        json={
            "idempotency_key": "trip-after-two-searches",
            "stops": [
                {"point_id": p1, "start": iso(at(8)), "end": iso(at(8, 30))},
                {"point_id": p2, "start": iso(at(9)), "end": iso(at(9, 30))},
            ],
        },
    )
    assert booked.status_code == 201, booked.text


# ---------------------------------------------------------------------------
# Planner unit tests — half-open boundaries and the sweep
# ---------------------------------------------------------------------------


def _t(h, m=0):
    return datetime(2026, 10, 5, h, m, tzinfo=UTC)


def test_hot_intervals_end_before_start_on_touch():
    # Two touching reservations never overlap: no instant reaches capacity 2.
    hot = hot_intervals(((_t(8), _t(9)), (_t(9), _t(10))), capacity=2)
    assert hot == []


def test_hot_intervals_overlap_peak():
    hot = hot_intervals(
        ((_t(8), _t(9, 30)), (_t(8, 30), _t(10))), capacity=2
    )
    assert hot == [(_t(8, 30), _t(9, 30))]


def test_feasible_starts_keep_touching_boundaries():
    # Open 08-12, hot [09,10), duration 60: forbidden starts are the open
    # interval (08, 10).  Starting at 08 occupies [08,09), which only touches
    # the hot stretch at 09; starting at 10 begins exactly when it releases.
    stop_input = StopPlanInput(
        point_id=1,
        duration=timedelta(minutes=60),
        rules_version=1,
        capacity=1,
        open_intervals=((_t(8), _t(12)),),
        reservations=((_t(9), _t(10)),),
    )
    components = trip_planner.feasible_start_components(stop_input)
    assert components == [(_t(8), _t(8)), (_t(10), _t(11))]


def test_feasible_start_exactly_fills_cool_gap():
    # Hot [08,09) and [09:30,10:30); a 30-minute stop exactly fills the cool
    # gap at [09,09:30); the single touching start instant 09:00 survives.
    stop_input = StopPlanInput(
        point_id=1,
        duration=timedelta(minutes=30),
        rules_version=1,
        capacity=1,
        open_intervals=((_t(8), _t(12)),),
        reservations=((_t(8), _t(9)), (_t(9, 30), _t(10, 30))),
    )
    components = trip_planner.feasible_start_components(stop_input)
    assert (_t(9), _t(9)) in components  # degenerate single-instant component


def test_plan_greedy_lexicographic_minimum():
    # Stop 1 open 08-09 and 13-14; stop 2 open 09:30-10 only; 30-minute
    # stops, no gap.  Earliest tuple is (08:00, 09:30).
    s1 = StopPlanInput(
        1, timedelta(minutes=30), 1, 1,
        ((_t(8), _t(9)), (_t(13), _t(14))), (),
    )
    s2 = StopPlanInput(
        2, timedelta(minutes=30), 1, 1, ((_t(9, 30), _t(10)),), (),
    )
    planned = plan_earliest_trip(
        [s1, s2], search_start=_t(0), search_end=_t(23),
        min_gap=timedelta(0), now=datetime(2026, 10, 4, tzinfo=UTC),
        max_days_ahead=30,
    )
    assert [(p.start, p.end) for p in planned] == [
        (_t(8), _t(8, 30)),
        (_t(9, 30), _t(10)),
    ]


def test_plan_unsatisfiable_second_stop_returns_none():
    s1 = StopPlanInput(
        1, timedelta(minutes=30), 1, 1, ((_t(8), _t(12)),), (),
    )
    s2 = StopPlanInput(
        2, timedelta(minutes=30), 1, 1, ((_t(8), _t(8, 15)),), (),
    )
    planned = plan_earliest_trip(
        [s1, s2], search_start=_t(0), search_end=_t(23),
        min_gap=timedelta(0), now=datetime(2026, 10, 4, tzinfo=UTC),
        max_days_ahead=30,
    )
    assert planned is None
