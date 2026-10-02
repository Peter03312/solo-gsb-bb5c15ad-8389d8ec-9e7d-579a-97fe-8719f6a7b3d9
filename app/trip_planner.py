"""Earliest feasible multi-stop itinerary planner (行程可行性搜索).

Pure-Python planning layer used by the read-only ``POST /trips/search``
endpoint.  It never writes anything: given a frozen database snapshot of the
currently published calendars, the per-point capacities and the non-cancelled
reservations, it returns the *lexicographically earliest* feasible itinerary
(comparing the stops' start instants in visit order), or ``None`` when no such
itinerary exists.

Why a plain greedy pass is optimal here
---------------------------------------

The stops visit distinct points, so per-point capacity is independent across
stops; the only coupling is the minimum inter-stop gap, which only couples
adjacent starts:

    s[i+1] >= s[i] + duration[i] + gap[i].

Feasibility of every stop is an "at or after a lower bound" property, and the
feasible-start set of a later stop never expands when its lower bound moves
later.  Therefore choosing the earliest feasible start at each stop is
optimal: if stop ``i`` has no feasible start at or after the lower bound
produced by the earliest feasible prefix, delaying any earlier stop only
raises that lower bound and can never create a solution.  No backtracking is
needed and the resulting tuple ``(s[0], ..., s[n-1])`` is the lexicographic
minimum over every feasible itinerary.

Capacity (half-open intervals)
-------------------------------

A reservation occupies ``[start, end)``.  For a stop of length ``d`` starting
at ``s`` (occupied window ``[s, s+d)``) to fit, every instant of that window
must hold fewer than ``capacity`` active reservations.  The set of
*forbidden start instants* is the *open* interval

    (a - d, b)   for every "hot" interval [a, b) of the search space where
                  the active concurrency already reaches the capacity.

Both ends are open: starting exactly at ``a - d`` still ends exactly at ``a``
(touching is free), and starting exactly at ``b`` begins after the hot stretch
has released.  Hot intervals are found with the classic sweep: end events are
applied before start events at equal timestamps, so touching reservations
never overlap.

Containment in one continuous open interval
-------------------------------------------

Every stop must be fully contained in one continuous open interval
``[o_start, o_end)``, so its start lies in ``[o_start, o_end - d]``.  The
feasible starts of a stop are therefore

    (open intervals shifted by ``-d``)  minus  (hot intervals shifted by ``-d``)

computed entirely in UTC instants, exactly like the reservation engine does
for individual bookings (cross-midnight sessions, DST gaps/repeats and
capacity changes are already reflected in the supplied open intervals and
reservations).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta


@dataclass(frozen=True)
class StopPlanInput:
    """Planner input for one stop in visit order.

    ``open_intervals`` are the merged continuous open intervals (UTC) of the
    point's currently published calendar; ``reservations`` are the
    non-cancelled reservations ``(start, end)`` relevant to the search window.
    """

    point_id: int
    duration: timedelta
    rules_version: int
    capacity: int
    open_intervals: tuple[tuple[datetime, datetime], ...]
    reservations: tuple[tuple[datetime, datetime], ...]


@dataclass(frozen=True)
class PlannedStop:
    point_id: int
    start: datetime
    end: datetime
    rules_version: int


def _merge_intervals(
    intervals: list[tuple[datetime, datetime]],
) -> list[tuple[datetime, datetime]]:
    """Merge forbidden-start intervals that *strictly* overlap.

    The spans describe forbidden *start instants* with both boundary instants
    themselves feasible (touching reservations coexist).  Two spans that only
    touch at one instant therefore must NOT merge: when hot stretches are
    separated by a cool gap exactly as long as the stop duration, that single
    touching start instant is precisely the one window that fits the cool gap.
    """
    if not intervals:
        return []
    ordered = sorted(intervals)
    merged: list[tuple[datetime, datetime]] = []
    cur_start, cur_end = ordered[0]
    for start, end in ordered[1:]:
        if start < cur_end:
            if end > cur_end:
                cur_end = end
        else:
            merged.append((cur_start, cur_end))
            cur_start, cur_end = start, end
    merged.append((cur_start, cur_end))
    return merged


def hot_intervals(
    reservations: tuple[tuple[datetime, datetime], ...],
    capacity: int,
) -> list[tuple[datetime, datetime]]:
    """Maximal instants at which active concurrency is >= ``capacity``.

    Events at the same timestamp are ordered end-before-start so a reservation
    ending at ``t`` releases before another one starting at ``t`` is counted —
    the half-open ``[start, end)`` rule under which touching never overlaps.
    A "hot" maximal stretch is exactly a window a new reservation cannot cover
    any instant of.
    """
    if capacity <= 0 or not reservations:
        return []
    events: list[tuple[datetime, int, int]] = []
    for index, (start, end) in enumerate(reservations):
        # Tuple order (time, kind, index): kind 0 = end, kind 1 = start.
        events.append((end, 0, index))
        events.append((start, 1, index))
    events.sort()

    active: set[int] = set()
    hot: list[tuple[datetime, datetime]] = []
    hot_start: datetime | None = None
    for timestamp, kind, index in events:
        if kind == 0:
            active.discard(index)
        else:
            active.add(index)
        count = len(active)
        if hot_start is None and count >= capacity:
            hot_start = timestamp
        elif hot_start is not None and count < capacity:
            hot.append((hot_start, timestamp))
            hot_start = None
    return hot


def feasible_start_components(
    stop: StopPlanInput,
) -> list[tuple[datetime, datetime]]:
    """Closed intervals of feasible *start* instants for one stop.

    Each continuous open interval ``[o_start, o_end)`` long enough to hold the
    stop contributes the closed interval ``[o_start, o_end - d]`` of feasible
    starts.  A hot stretch ``[a, b)`` forbids exactly the starts
    ``(a - d, b)`` — an *open* interval, because starting at ``a - d`` ends
    exactly at ``a`` and starting at ``b`` begins exactly when the hot stretch
    releases (touching is free under the half-open occupancy rule).  Both
    boundary instants are therefore retained, including degenerate components
    of a single feasible instant where a window exactly fills a cool gap.
    Result intervals are disjoint, closed, and sorted.
    """
    duration = stop.duration
    allowed: list[tuple[datetime, datetime]] = []
    for open_start, open_end in stop.open_intervals:
        latest_start = open_end - duration
        if latest_start >= open_start:
            allowed.append((open_start, latest_start))

    forbidden: list[tuple[datetime, datetime]] = [
        (hot_start - duration, hot_end)
        for hot_start, hot_end in hot_intervals(stop.reservations, stop.capacity)
    ]
    forbidden = _merge_intervals(forbidden)

    components: list[tuple[datetime, datetime]] = []
    for allowed_start, allowed_end in allowed:
        pieces = [(allowed_start, allowed_end)]
        for forbidden_start, forbidden_end in forbidden:
            next_pieces: list[tuple[datetime, datetime]] = []
            for piece_start, piece_end in pieces:
                # Forbidden spans are open (fs, fe): equal-boundary instants
                # are feasible and must not be cut away.
                if forbidden_end <= piece_start or forbidden_start >= piece_end:
                    next_pieces.append((piece_start, piece_end))
                    continue
                # Left remainder up to and including the forbidden start;
                # a == comparison keeps the boundary as a degenerate piece.
                if piece_start <= forbidden_start:
                    next_pieces.append((piece_start, forbidden_start))
                # Right remainder from the forbidden end (inclusive).
                if forbidden_end <= piece_end:
                    next_pieces.append((forbidden_end, piece_end))
            pieces = next_pieces
        components.extend(pieces)
    return sorted(components)


def earliest_feasible_start(
    components: list[tuple[datetime, datetime]],
    lower_bound: datetime | None,
) -> datetime | None:
    """Earliest feasible start at or after ``lower_bound``, if any."""
    for component_start, component_end in components:
        if lower_bound is not None and component_end < lower_bound:
            continue
        return component_start if lower_bound is None or component_start >= lower_bound else lower_bound
    return None


def plan_earliest_trip(
    stops: list[StopPlanInput],
    *,
    search_start: datetime,
    search_end: datetime,
    min_gap: timedelta,
    now: datetime,
    max_days_ahead: int,
) -> list[PlannedStop] | None:
    """Return the lexicographically earliest feasible itinerary, or ``None``.

    Parameters (all UTC)
    --------------------
    search_start, search_end:
        Inclusive bounds of the whole group window: every stop's start must be
        >= ``search_start`` and every stop's end must be <= ``search_end``.
    min_gap:
        Minimum separation between one stop's end and the next stop's start;
        ``timedelta(0)`` allows touching.
    now, max_days_ahead:
        The existing booking lead-time policy (identical to
        ``validate_reservation_window``): each stop's start must be in
        ``[now, now + max_days_ahead]``.  The stop end is not separately
        horizon-limited, exactly like ordinary reservations — only the stop
        duration is capped — so every itinerary returned here is directly
        submittable to the trip-booking endpoint.
    """
    if search_end <= search_start:
        return None

    horizon_end = now + timedelta(days=max_days_ahead)
    first_lower = max(search_start, now)
    if first_lower > search_end or first_lower > horizon_end:
        return None

    planned: list[PlannedStop] = []
    lower_bound: datetime | None = first_lower
    for stop in stops:
        components = feasible_start_components(stop)
        start = earliest_feasible_start(components, lower_bound)
        if start is None or start > horizon_end:
            return None
        end = start + stop.duration
        # The whole group must stay inside the (inclusive) search range.
        if end > search_end:
            return None
        planned.append(
            PlannedStop(
                point_id=stop.point_id,
                start=start,
                end=end,
                rules_version=stop.rules_version,
            )
        )
        lower_bound = end + min_gap
    return planned
