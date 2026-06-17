"""
Spatial helpers for the OHRC catalog.

Longitude is cyclic on [0, 360). Naive `max - min` of four corner
longitudes is *not* a safe bbox for polar strips: near the pole, a
single strip can legitimately span a wide longitude range without
crossing the antimeridian (longitude is degenerate there). The naive
heuristic also misclassifies wide-pole strips as wrap-around crossers,
inverting the footprint.

Instead we represent each strip's longitude coverage as an *arc* on the
[0, 360) circle: `(arc_start, arc_end)` where the arc goes counter-
clockwise from start to end. If `start <= end`, the arc is the closed
interval `[start, end]`. If `start > end`, the arc wraps the antimeridian
and covers `[start, 360) U [0, end]`.

We derive the arc from the four corner longitudes via the
*cyclic-gap-complement* construction: sort the corner longitudes, find
the largest cyclic gap between consecutive points (treating the list as
cyclic), the arc is the complement of that gap. If the largest gap is
the wrap-gap (between the highest and the lowest), the strip does NOT
cross the antimeridian. Otherwise it does.
"""

from __future__ import annotations

from typing import Sequence

__all__ = [
    "lon_arc_from_corners",
    "lon_arc_span",
    "point_in_lon_arc",
    "arcs_overlap",
    "is_polar_degenerate",
    "obs_and_station",
]


def lon_arc_from_corners(lons: Sequence[float]) -> tuple[float, float, bool]:
    """
    From the four corner longitudes (any order, in degrees, may be in
    `(-180, 360]`), return `(arc_start, arc_end, wraps_antimeridian)`.

    The arc is the cyclic-gap-complement: the shortest arc on the
    [0, 360) circle that contains all of the corner points.
    """
    if not lons:
        raise ValueError("lons is empty")
    pts = sorted((float(L) % 360.0) for L in lons)
    n = len(pts)
    if n == 1:
        return pts[0], pts[0], False

    # All consecutive cyclic gaps. Gap i = pts[i+1] - pts[i] for i < n-1.
    # Gap n-1 (the "wrap gap") = 360 - pts[n-1] + pts[0].
    gaps = [(pts[i + 1] - pts[i], i) for i in range(n - 1)]
    wrap_gap = 360.0 - pts[-1] + pts[0]
    gaps.append((wrap_gap, n - 1))

    # Largest gap = complement of the occupied arc.
    max_gap, max_idx = max(gaps, key=lambda g: g[0])

    if max_idx == n - 1:
        # Largest gap is the wrap gap: arc does NOT cross 0/360.
        # Arc spans from pts[0] up through pts[-1].
        return pts[0], pts[-1], False

    # Largest gap is interior: arc DOES cross 0/360. The arc starts at
    # the point right after the gap (pts[max_idx + 1]) and ends at the
    # point right before it (pts[max_idx]).
    return pts[max_idx + 1], pts[max_idx], True


def lon_arc_span(arc_start: float, arc_end: float) -> float:
    """Length of the arc in degrees, in [0, 360]."""
    if arc_start <= arc_end:
        return arc_end - arc_start
    return (360.0 - arc_start) + arc_end


def point_in_lon_arc(lon: float, arc_start: float, arc_end: float) -> bool:
    """Is `lon` inside the arc `[arc_start -> arc_end]` (counter-clockwise)?"""
    p = lon % 360.0
    if arc_start <= arc_end:
        return arc_start <= p <= arc_end
    return p >= arc_start or p <= arc_end


def arcs_overlap(s1: float, e1: float, s2: float, e2: float) -> bool:
    """
    Do two arcs on [0, 360) overlap at any point?

    An arc `[s, e]` is `[s, e]` if `s <= e`, otherwise `[s, 360) U [0, e]`.
    The arcs overlap iff at least one of:
      - one arc contains a boundary of the other, OR
      - both arcs are simple (non-wrapping) and their intervals overlap.

    Equivalently: arcs are disjoint iff each lies entirely within the
    *gap* of the other. So we check non-disjointness: an arc contains
    any endpoint of the other.
    """
    return (
        point_in_lon_arc(s2, s1, e1)
        or point_in_lon_arc(e2, s1, e1)
        or point_in_lon_arc(s1, s2, e2)
        or point_in_lon_arc(e1, s2, e2)
    )


def is_polar_degenerate(
    bbox_max_lat: float,
    arc_start: float,
    arc_end: float,
    *,
    lat_threshold: float = -88.0,
    span_threshold: float = 90.0,
) -> bool:
    """
    Flag strips where longitude becomes geometrically degenerate near
    the pole: the strip's small physical footprint can sprawl across
    a huge lon arc, making lon-based bbox queries unreliable. Lat-based
    queries remain meaningful.
    """
    return (bbox_max_lat < lat_threshold) and (lon_arc_span(arc_start, arc_end) > span_threshold)


def obs_and_station(product_id: str) -> tuple[str, str | None]:
    """
    Split an OHRC product_id into `(obs_id, station_id)`.

    `product_id` is structured as `ch2_ohr_<role>_<ts>_d_img_<station>`.
    The trailing token is the ground station that received the downlink;
    the rest identifies the unique observation.
    """
    if "_" not in product_id:
        return product_id, None
    obs_id, station_id = product_id.rsplit("_", 1)
    return obs_id, station_id
