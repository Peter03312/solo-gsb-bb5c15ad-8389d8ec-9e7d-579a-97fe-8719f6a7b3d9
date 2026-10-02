"""Read-only itinerary planning engine (行程规划搜索).

Given, per stop, the point's *currently published* rule set, its capacity and
the active (non-cancelled) reservations visible in one database snapshot, find
the lexicographically earliest feasible itinerary:

* each stop is a fixed-duration visit that must fit wholly inside one
  continuous open interval of the point's published calendar (intervals may
  cross local midnight; DST follows the existing scheduler rules);
* every instant of the stop's half-open window ``[start, end)`` must keep the
  point strictly below its simultaneous-reception capacity — touching
  end-to-end is free;
* successive starts respect a minimum gap (end of the previous stop to the
  start of the next);
* the whole group lies inside both the search window and the existing
  bookable horizon (now .. now + reservation-max-days-ahead).

"Earliest" compares the stops' start instants in visit order: minimize the
first start, then the second, and so on.  Because the stops visit distinct
points and never constrain one another backwards (a later stop may only be
pushed forward by an earlier one), greedily taking the earliest feasible
start at every stop yields exactly that lexicographic minimum: any itinerary
whose k-th start is earlier while the previous ones match would have been
reachable from the same earlier-stop prefix the greedy chose.

The module is pure Python — all database reads happen in the caller inside a
single REPEATABLE READ transaction, so one query answers from one snapshot.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

from .scheduler import RuleSet, build_open_intervals, get_zone


@dataclass(frozen=True)
class StopSpec:
    """Everything one planning stop needs from the database snapshot."""

    point_id: int
    timezone: str
    capacity: int
    rules_version: int
    ruleset: RuleSet
    # (start_utc, end_utc) of every active reservation that can matter.
    reservations: tuple[tuple[datetime, datetime], ...]


@dataclass(frozen=True)
class PlannedStop:
    point_id: int
    start: datetime
    end: datetime
    rules_version: int


def busy_intervals(
    windows: list[tuple[datetime, datetime]] | tuple[tuple[datetime, datetime], ...],
    capacity: int,
) -> list[tuple[datetime, datetime]]:
    """Merge the instants at which ``windows`` already hold ``capacity`` visits.

    Occupancy follows the half-open rule: a visit occupies ``[start, end)``,
    so at a shared timestamp the end event is counted before the start event
    (visits that merely touch coexist).  Consecutive full segments (including
    ones merely touching) are merged into single intervals; a start time is
    infeasible exactly when the candidate window overlaps one of them.
    """
    # delta -1 (end) sorts before delta +1 (start) at the same instant.
    events: list[tuple[datetime, int]] = []
    for start, end in windows:
        events.append((start, 1))
        events.append((end, -1))
    events.sort(key=lambda event: (event[0], event[1]))

    full: list[tuple[datetime, datetime]] = []
    count = 0
    seg_start: datetime | None = None
    i = 0
    while i < len(events):
        t = events[i][0]
        if seg_start is not None and count >= capacity and t > seg_start:
            full.append((seg_start, t))
        while i < len(events) and events[i][0] == t:
            count += events[i][1]
            i += 1
        seg_start = t

    # Merge full runs that touch each other.
    merged: list[tuple[datetime, datetime]] = []
    for start, end in full:
        if merged and start <= merged[-1][1]:
            if end > merged[-1][1]:
                merged[-1] = (merged[-1][0], end)
        else:
            merged.append((start, end))
    return merged


def earliest_open_slot(
    open_intervals: list[tuple[datetime, datetime]],
    busy: list[tuple[datetime, datetime]],
    duration: timedelta,
    lower: datetime,
    upper_end: datetime,
) -> datetime | None:
    """Earliest start ``s >= lower`` with ``[s, s+duration)`` placeable.

    The window must end no later than ``upper_end``, sit wholly inside one
    continuous open interval and not overlap any capacity-full interval.
    Returns ``None`` when no such start exists.  ``open_intervals`` and
    ``busy`` are both ascending and disjoint-from-touching respectively.
    """
    for open_start, open_end in open_intervals:
        if open_end <= lower:
            continue
        candidate = max(lower, open_start)
        if candidate + duration > upper_end:
            # Every later open interval starts even later; the horizon can
            # never be met again.
            return None
        if candidate + duration > open_end:
            continue
        i = 0
        while i < len(busy):
            busy_start, busy_end = busy[i]
            if busy_end <= candidate:
                i += 1
                continue
            if candidate + duration <= busy_start:
                # The window ends no later than the run starts — touching at
                # the boundary is free, and all later runs start even later.
                break
            # [candidate, candidate+duration) reaches into this full run.
            # The earliest start that ends before the run is busy_start -
            # duration (touching is free); when that start is already behind
            # the candidate, the run sits inside the window and the next
            # viable start is busy_end (a visit starting exactly at the run
            # end merely touches the run).
            before_run = busy_start - duration
            if before_run > candidate:
                candidate = before_run
                # The candidate moved backwards: a full run skipped earlier
                # could now be overlapped, so rescan from the beginning.
                i = 0
                continue
            candidate = busy_end
            i += 1
            if candidate + duration > upper_end:
                return None
            if candidate + duration > open_end:
                break
        if candidate + duration <= open_end and candidate + duration <= upper_end:
            return candidate
    return None


def plan_itinerary(
    specs: list[StopSpec],
    durations: list[timedelta],
    gap: timedelta,
    *,
    search_from: datetime,
    search_to: datetime,
    now: datetime,
    horizon: datetime,
) -> list[PlannedStop] | None:
    """Greedy lexicographically earliest itinerary, or ``None`` when infeasible.

    ``horizon`` is the existing reservation lead-time limit; every stop must
    also finish by ``min(search_to, horizon)`` and not start before
    ``max(search_from, now)``.
    """
    upper_end = min(search_to, horizon)
    lower = max(search_from, now)
    planned: list[PlannedStop] = []
    for spec, duration in zip(specs, durations):
        if lower > upper_end or duration <= timedelta(0):
            return None
        zone = get_zone(spec.timezone)
        # Pad by a local day on both sides so overnight sessions owning a
        # piece just outside the window are still expanded correctly.
        date_from = (lower - timedelta(days=1)).astimezone(zone).date()
        date_to = (upper_end + timedelta(days=1)).astimezone(zone).date()
        open_intervals = build_open_intervals(
            spec.timezone, spec.ruleset, date_from, date_to
        )
        # Reservations that began before ``lower`` still hold capacity over
        # the searched span: clamp their start to ``lower`` so the sweep
        # begins with the correct initial occupancy instead of an
        # off-by-one (or negative) count.
        windows = tuple(
            (max(start, lower), end) for start, end in spec.reservations
        )
        full = busy_intervals(windows, spec.capacity)
        start = earliest_open_slot(open_intervals, full, duration, lower, upper_end)
        if start is None:
            return None
        end = start + duration
        planned.append(
            PlannedStop(
                point_id=spec.point_id, start=start, end=end,
                rules_version=spec.rules_version,
            )
        )
        lower = end + gap
    return planned
