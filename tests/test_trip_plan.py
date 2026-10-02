"""End-to-end tests for read-only trip planning (行程规划搜索) POST /trips/plan."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

UTC = timezone.utc


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def iso(dt):
    return dt.isoformat().replace("+00:00", "Z")


def create_point(client, tz="UTC", capacity=1):
    r = client.post(
        "/points",
        json={"name": "规划测试点", "timezone": tz, "capacity": capacity},
    )
    assert r.status_code == 201, r.text
    return r.json()["id"]


def publish_weekly(client, pid, weekday, start, end, expected=0,
                   closed=None, makeup=None):
    body = {
        "expected_version": expected,
        "rules": {
            "weekly": [
                {"weekday": weekday, "start_time": start, "end_time": end}
            ] if start is not None else [],
            "closed_dates": closed or [],
            "makeup": makeup or [],
        },
    }
    r = client.post(f"/points/{pid}/rules", json=body)
    assert r.status_code == 201, r.text
    return r.json()["version"]


def open_point(client, start_time="08:00", end_time="18:00", tz="UTC",
               capacity=1, day_offset=1, weekday=None):
    pid = create_point(client, tz=tz, capacity=capacity)
    zone = ZoneInfo(tz)
    local_day = (
        datetime.now(UTC).astimezone(zone) + timedelta(days=day_offset)
    ).date()
    publish_weekly(
        client, pid, weekday or local_day.isoweekday(), start_time, end_time
    )
    return pid


def open_makeup_point(client, day, start_time, end_time, tz="UTC"):
    """Point with a same-day makeup opening covering ``day`` (local date)."""
    pid = create_point(client, tz=tz)
    r = client.post(
        f"/points/{pid}/rules",
        json={
            "expected_version": 0,
            "rules": {
                "weekly": [],
                "closed_dates": [],
                "makeup": [
                    {
                        "day": day.isoformat(),
                        "start_time": start_time,
                        "end_time": end_time,
                    }
                ],
            },
        },
    )
    assert r.status_code == 201, r.text
    return pid


def tomorrow_window(hour_from, hour_to):
    day = (datetime.now(UTC) + timedelta(days=1)).date()
    return (
        datetime(day.year, day.month, day.day, hour_from, tzinfo=UTC),
        datetime(day.year, day.month, day.day, hour_to, tzinfo=UTC),
    )


def plan(client, points, durations, search_from, search_to, gap=0):
    return client.post(
        "/trips/plan",
        json={
            "stops": [
                {"point_id": pid, "duration_minutes": d}
                for pid, d in zip(points, durations)
            ],
            "min_gap_minutes": gap,
            "search_from": iso(search_from),
            "search_to": iso(search_to),
        },
    )


def reserve(client, pid, key, start, end):
    return client.post(
        f"/points/{pid}/reservations",
        json={"idempotency_key": key, "start": iso(start), "end": iso(end)},
    )


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------

def test_plan_returns_earliest_two_stop_itinerary_with_versions(client):
    p1 = open_point(client)
    p2 = open_point(client)
    sf, st = tomorrow_window(0, 23)
    r = plan(client, [p1, p2], [30, 45], sf, st, gap=15)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["min_gap_minutes"] == 15
    assert [s["point_id"] for s in body["stops"]] == [p1, p2]
    s0, s1 = body["stops"]
    assert s0["start"] == iso(tomorrow_window(8, 8)[0])
    assert s0["end"] == iso(tomorrow_window(8, 8)[0] + timedelta(minutes=30))
    assert s0["duration_minutes"] == 30
    assert s0["rules_version"] == 1
    # 08:30 + 15 minute gap -> 08:45, 45 minutes -> 09:30.
    assert s1["start"].endswith("T08:45:00Z")
    assert s1["end"].endswith("T09:30:00Z")
    assert s1["rules_version"] == 1


def test_plan_respects_timezone_offset_on_search_bounds(client):
    # Point in Asia/Shanghai (UTC+8): 08:00 local = 00:00 UTC.
    sh = ZoneInfo("Asia/Shanghai")
    local_day = (
        datetime.now(UTC).astimezone(sh) + timedelta(days=1)
    ).date()
    pid = open_point(client, tz="Asia/Shanghai")
    p2 = open_point(client, tz="Asia/Shanghai")
    sf = datetime.combine(local_day, datetime.min.time()).replace(tzinfo=sh)
    st = sf + timedelta(hours=10)
    r = plan(client, [pid, p2], [30, 30], sf, st, gap=10)
    assert r.status_code == 200, r.text
    stops = r.json()["stops"]
    # Search starts at local midnight; the point opens at 08:00 local =
    # 00:00 UTC that same local day.
    expected = (sf + timedelta(hours=8)).astimezone(UTC)
    assert stops[0]["start"] == iso(expected)
    assert stops[0]["rules_version"] == 1


def test_plan_empty_when_no_feasible_group(client):
    p1 = open_point(client, "08:00", "08:40")  # too short for a 45-minute stop
    p2 = open_point(client)
    sf, st = tomorrow_window(0, 23)
    r = plan(client, [p1, p2], [45, 30], sf, st, gap=30)
    assert r.status_code == 200, r.text
    assert r.json()["stops"] == []


def test_plan_empty_when_search_window_ends_before_open(client):
    p1 = open_point(client)
    p2 = open_point(client)
    sf, _ = tomorrow_window(0, 0)
    r = plan(client, [p1, p2], [30, 30], sf, sf + timedelta(minutes=30))
    assert r.status_code == 200
    assert r.json()["stops"] == []


def test_plan_does_not_create_anything(client):
    p1 = open_point(client)
    p2 = open_point(client)
    sf, st = tomorrow_window(0, 23)
    plan(client, [p1, p2], [30, 30], sf, st)
    # The same window must still be bookable afterwards.
    r = reserve(client, p1, "after-plan-1",
                tomorrow_window(8, 8)[0],
                tomorrow_window(8, 8)[0] + timedelta(minutes=30))
    assert r.status_code == 201, r.text
    r = reserve(client, p2, "after-plan-2",
                tomorrow_window(8, 8)[0],
                tomorrow_window(8, 8)[0] + timedelta(minutes=30))
    assert r.status_code == 201, r.text


def test_plan_uses_newest_published_version(client):
    p1 = open_point(client, "08:00", "12:00")
    p2 = open_point(client)
    day = (datetime.now(UTC) + timedelta(days=1)).date()
    # Narrow point 1 to the afternoon in v2.
    publish_weekly(
        client, p1, day.isoweekday(), "13:00", "17:00", expected=1
    )
    sf, st = tomorrow_window(0, 23)
    r = plan(client, [p1, p2], [30, 30], sf, st)
    assert r.status_code == 200
    stops = r.json()["stops"]
    assert stops[0]["start"].endswith("T13:00:00Z")
    assert stops[0]["rules_version"] == 2


# ---------------------------------------------------------------------------
# Capacity: half-open intervals and touching
# ---------------------------------------------------------------------------

def test_plan_skips_occupied_capacity_and_touches_free(client):
    p1 = open_point(client)
    p2 = open_point(client)
    reserve(
        client, p1, "block-1",
        tomorrow_window(8, 8)[0],
        tomorrow_window(8, 8)[0] + timedelta(minutes=60),
    )
    sf, st = tomorrow_window(0, 23)
    r = plan(client, [p1, p2], [30, 30], sf, st)
    assert r.status_code == 200
    stops = r.json()["stops"]
    # Full [08:00,09:00); the visit may start exactly at 09:00.
    assert stops[0]["start"].endswith("T09:00:00Z")
    assert stops[1]["start"].endswith("T09:30:00Z")


def test_plan_capacity_two_lets_visit_overlap(client):
    p1 = open_point(client, capacity=2)
    p2 = open_point(client)
    reserve(
        client, p1, "one-visitor",
        tomorrow_window(8, 8)[0],
        tomorrow_window(8, 8)[0] + timedelta(minutes=60),
    )
    sf, st = tomorrow_window(0, 23)
    r = plan(client, [p1, p2], [30, 30], sf, st)
    assert r.status_code == 200
    assert r.json()["stops"][0]["start"].endswith("T08:00:00Z")


def test_plan_touching_reservations_keep_point_full(client):
    p1 = open_point(client)
    p2 = open_point(client)
    base = tomorrow_window(8, 8)[0]
    reserve(client, p1, "a", base, base + timedelta(minutes=30))
    reserve(client, p1, "b",
            base + timedelta(minutes=30), base + timedelta(minutes=60))
    sf, st = tomorrow_window(0, 23)
    r = plan(client, [p1, p2], [30, 30], sf, st)
    assert r.status_code == 200
    # Capacity stays 1 across 08:00-09:00; earliest start 09:00.
    assert r.json()["stops"][0]["start"].endswith("T09:00:00Z")


def test_plan_cancelled_reservation_does_not_block(client):
    p1 = open_point(client)
    p2 = open_point(client)
    base = tomorrow_window(8, 8)[0]
    r = reserve(client, p1, "gone", base, base + timedelta(minutes=30))
    token = r.json()["cancel_token"]
    cr = client.post(
        f"/points/{p1}/reservations/{r.json()['id']}/cancel",
        json={"cancel_token": token},
    )
    assert cr.status_code == 200
    sf, st = tomorrow_window(0, 23)
    r = plan(client, [p1, p2], [30, 30], sf, st)
    assert r.json()["stops"][0]["start"].endswith("T08:00:00Z")


def test_plan_window_dodges_busy_by_ending_before_it(client):
    # Busy [09:00,10:00); a 30-minute visit placed at 08:30 ends at 09:00 and
    # merely touches, so it beats jumping to 10:00.
    p1 = open_point(client)
    p2 = open_point(client)
    reserve(
        client, p1, "late-block",
        tomorrow_window(9, 9)[0],
        tomorrow_window(9, 9)[0] + timedelta(minutes=60),
    )
    sf, _ = tomorrow_window(0, 0)
    r = plan(client, [p1, p2], [30, 30], sf, sf + timedelta(hours=20))
    stops = r.json()["stops"]
    assert stops[0]["start"].endswith("T08:00:00Z")  # 08:00 is open & free


def test_plan_reservation_started_before_search_from_still_blocks(client):
    # search_from is 09:00, but a visitor already holds 08:00-10:00: the
    # earliest free 30-minute start is 10:00, not 09:00.
    p1 = open_point(client)
    p2 = open_point(client)
    base = tomorrow_window(8, 8)[0]
    reserve(client, p1, "early-holder", base, base + timedelta(hours=2))
    r = plan(
        client, [p1, p2], [30, 30],
        tomorrow_window(9, 9)[0],
        tomorrow_window(20, 20)[0],
    )
    assert r.status_code == 200, r.text
    assert r.json()["stops"][0]["start"].endswith("T10:00:00Z")


def test_plan_second_stop_only_open_afternoon_pushes_chain(client):
    p1 = open_point(client, "08:00", "18:00")
    p2 = create_point(client)
    day = (datetime.now(UTC) + timedelta(days=1)).date()
    publish_weekly(client, p2, day.isoweekday(), "13:00", "14:00")
    sf, st = tomorrow_window(0, 23)
    r = plan(client, [p1, p2], [30, 30], sf, st, gap=15)
    stops = r.json()["stops"]
    assert stops[0]["start"].endswith("T08:00:00Z")
    assert stops[1]["start"].endswith("T13:00:00Z")
    assert stops[1]["end"].endswith("T13:30:00Z")


# ---------------------------------------------------------------------------
# Overnight sessions and DST follow the existing scheduler rules
# ---------------------------------------------------------------------------

def test_plan_stop_fits_inside_overnight_session(client):
    # New York Friday 22:00-02:00 (overnight, EST = UTC-5): the Friday-owned
    # session spans 03:00-07:00 UTC Saturday as one continuous interval, so a
    # visit planned in the Saturday UTC early morning fits without clipping.
    p1 = create_point(client, tz="America/New_York")
    publish_weekly(client, p1, 5, "22:00", "02:00")
    p2 = open_point(client)
    ny = ZoneInfo("America/New_York")
    now_ny = datetime.now(UTC).astimezone(ny)
    days_ahead = (5 - now_ny.isoweekday()) % 7
    friday = (now_ny + timedelta(days=days_ahead)).date()
    tail_start = (
        datetime.combine(friday, datetime.min.time().replace(hour=22))
        .replace(tzinfo=ny)
        .astimezone(UTC)
    )
    utc_saturday = tail_start.date()
    # Search the UTC day that actually contains the session tail.
    sf = datetime.combine(utc_saturday, datetime.min.time(), tzinfo=UTC)
    st = sf + timedelta(days=1)
    # Point 2 only needs to be open when stop 1 ends: give it a UTC makeup
    # window on the Saturday UTC day of the tail.
    p2 = open_makeup_point(
        client, utc_saturday, "00:00", "23:59", tz="UTC"
    )
    r = plan(client, [p1, p2], [30, 30], sf, st)
    assert r.status_code == 200, r.text
    stops = r.json()["stops"]
    # Session start: Friday 22:00 America/New_York resolved to UTC (03:00 in
    # EST, 02:00 in EDT) — the scheduler handles DST, just as bookings do.
    expected_start = tail_start
    assert stops[0]["start"] == iso(expected_start)
    assert stops[0]["end"] == iso(expected_start + timedelta(minutes=30))
    # Stop 2 (UTC point) follows immediately after.
    assert stops[1]["start"] == iso(expected_start + timedelta(minutes=30))


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------

def test_plan_unknown_point_named_with_stop_index(client):
    p1 = open_point(client)
    sf, st = tomorrow_window(0, 23)
    r = client.post(
        "/trips/plan",
        json={
            "stops": [
                {"point_id": p1, "duration_minutes": 30},
                {"point_id": 999999, "duration_minutes": 30},
            ],
            "search_from": iso(sf),
            "search_to": iso(st),
        },
    )
    assert r.status_code == 404
    err = r.json()["error"]
    assert err["code"] == "point_not_found"
    assert err["details"]["stop_index"] == 1
    assert err["details"]["point_id"] == 999999


def test_plan_unpublished_rules_named_with_stop_index(client):
    p1 = open_point(client)
    p2 = create_point(client)  # version 0
    sf, st = tomorrow_window(0, 23)
    r = plan(client, [p1, p2], [30, 30], sf, st)
    assert r.status_code == 409
    err = r.json()["error"]
    assert err["code"] == "no_published_version"
    assert err["details"]["stop_index"] == 1
    assert err["details"]["point_id"] == p2


def test_plan_duration_above_limit_invalid_period_with_stop(client):
    p1 = open_point(client)
    p2 = open_point(client)
    sf, st = tomorrow_window(0, 23)
    r = plan(client, [p1, p2], [30, 121], sf, st)
    assert r.status_code == 422
    err = r.json()["error"]
    assert err["code"] == "invalid_period"
    assert err["details"]["stop_index"] == 1
    assert err["details"]["duration_minutes"] == 121


def test_plan_negative_gap_invalid_period(client):
    p1 = open_point(client)
    p2 = open_point(client)
    sf, st = tomorrow_window(0, 23)
    r = client.post(
        "/trips/plan",
        json={
            "stops": [
                {"point_id": p1, "duration_minutes": 30},
                {"point_id": p2, "duration_minutes": 30},
            ],
            "min_gap_minutes": -1,
            "search_from": iso(sf),
            "search_to": iso(st),
        },
    )
    assert r.status_code == 422
    assert r.json()["error"]["code"] == "invalid_request"


def test_plan_duplicate_points_invalid_request(client):
    p1 = open_point(client)
    sf, st = tomorrow_window(0, 23)
    r = plan(client, [p1, p1], [30, 30], sf, st)
    assert r.status_code == 422
    err = r.json()["error"]
    assert err["code"] == "invalid_request"
    assert err["details"]["stop_indexes"] == [0, 1]


def test_plan_span_over_seven_days_invalid_period(client):
    p1 = open_point(client)
    p2 = open_point(client)
    sf, st = tomorrow_window(0, 0)
    r = plan(client, [p1, p2], [30, 30], sf, st + timedelta(days=7, minutes=1))
    assert r.status_code == 422
    err = r.json()["error"]
    assert err["code"] == "invalid_period"
    assert err["details"]["max_trip_search_days"] == 7


def test_plan_seven_days_exactly_accepted(client):
    p1 = open_point(client)
    p2 = open_point(client)
    sf, st = tomorrow_window(0, 0)
    r = plan(client, [p1, p2], [30, 30], sf, st + timedelta(days=7))
    assert r.status_code == 200
    assert len(r.json()["stops"]) == 2


def test_plan_reversed_window_invalid_period(client):
    p1 = open_point(client)
    p2 = open_point(client)
    day_end, day_start = tomorrow_window(12, 8)
    r = plan(client, [p1, p2], [30, 30], day_end, day_start)  # to before from
    assert r.status_code == 422
    assert r.json()["error"]["code"] == "invalid_period"


def test_plan_naive_timestamp_invalid_request(client):
    p1 = open_point(client)
    p2 = open_point(client)
    r = client.post(
        "/trips/plan",
        json={
            "stops": [
                {"point_id": p1, "duration_minutes": 30},
                {"point_id": p2, "duration_minutes": 30},
            ],
            "search_from": "2026-10-02T00:00:00",
            "search_to": "2026-10-03T00:00:00Z",
        },
    )
    assert r.status_code == 422
    assert r.json()["error"]["code"] == "invalid_request"


def test_plan_wrong_stop_count_invalid_request(client):
    p1 = open_point(client)
    sf, st = tomorrow_window(0, 23)
    r = client.post(
        "/trips/plan",
        json={
            "stops": [{"point_id": p1, "duration_minutes": 30}],
            "search_from": iso(sf),
            "search_to": iso(st),
        },
    )
    assert r.status_code == 422
    assert r.json()["error"]["code"] == "invalid_request"
