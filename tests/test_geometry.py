"""
Tests for catalog.geometry — the arc-based longitude representation.

These are the foundation of every spatial query in the catalog. The
naive `max - min` bbox is wrong for both antimeridian crossers and
near-pole strips; we test both cases explicitly.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pytest

from catalog.geometry import (
    arcs_overlap,
    is_polar_degenerate,
    lon_arc_from_corners,
    lon_arc_span,
    obs_and_station,
    point_in_lon_arc,
)


# ---------------------------------------------------------------------------
# lon_arc_from_corners
# ---------------------------------------------------------------------------

class TestLonArcFromCorners:
    def test_cabeus_clean_no_wrap(self):
        """Four corners clustered near 320°E (Cabeus): straightforward arc."""
        start, end, wraps = lon_arc_from_corners([318.0, 322.0, 318.5, 321.5])
        assert wraps is False
        assert start == pytest.approx(318.0)
        assert end == pytest.approx(322.0)

    def test_antimeridian_crosser(self):
        """Corners straddle 0/360 — must be flagged as wrapping."""
        # Real strip from the diagnostic: lons = [359.10, 0.44, 3.27, 4.84]
        start, end, wraps = lon_arc_from_corners([359.10, 0.44, 3.27, 4.84])
        assert wraps is True
        # Arc goes from 359.10 (east of antimeridian, near 360) forward
        # through 0 to 4.84.
        assert start == pytest.approx(359.10)
        assert end == pytest.approx(4.84)

    def test_wide_near_pole_strip_is_not_wrapping(self):
        """
        Critical case: a near-pole strip can span >180° of longitude
        without crossing the antimeridian. Must NOT be flagged as wrap.

        Real strip from the diagnostic at lat ~-89.5°, naive span 207.
        Corners: 22.45°, 110.27°, 222.26°, 229.73°. The "middle" corner
        near 110° closes the interior gap so the largest gap is the
        wrap gap (229.73° -> 22.45° through 360), giving wraps=False.

        (Without a filler corner — e.g. just [22, 222, 230] — four
        corners spread into two clusters with a 180° interior gap are
        geometrically indistinguishable from a wrap-around, and the
        smallest-enclosing-arc rule picks the wrap interpretation. Real
        OHRC strips always have four corners and the geometry resolves
        cleanly.)
        """
        start, end, wraps = lon_arc_from_corners([222.259, 229.729, 110.268, 22.454])
        assert wraps is False
        assert start == pytest.approx(22.454)
        assert end == pytest.approx(229.729)

    def test_negative_longitudes_normalised(self):
        """Inputs in (-180, 0] should normalise into [0, 360)."""
        # -10 == 350; -5 == 355; 0; 5 → arc from 350 wrapping to 5.
        start, end, wraps = lon_arc_from_corners([-10.0, -5.0, 0.0, 5.0])
        assert wraps is True
        assert start == pytest.approx(350.0)
        assert end == pytest.approx(5.0)

    def test_single_corner_collapses(self):
        start, end, wraps = lon_arc_from_corners([42.0])
        assert (start, end, wraps) == (42.0, 42.0, False)

    def test_two_close_corners(self):
        start, end, wraps = lon_arc_from_corners([10.0, 11.0])
        assert (round(start, 6), round(end, 6), wraps) == (10.0, 11.0, False)

    def test_three_sixty_normalises_to_zero(self):
        """A corner at lon=360 should normalise to 0, not invent a wrap."""
        start, end, wraps = lon_arc_from_corners([0.0, 1.0, 2.0, 360.0])
        assert wraps is False
        assert start == pytest.approx(0.0)
        # All four normalise into [0, 2]
        assert end == pytest.approx(2.0)

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            lon_arc_from_corners([])


# ---------------------------------------------------------------------------
# lon_arc_span
# ---------------------------------------------------------------------------

class TestLonArcSpan:
    def test_simple_span(self):
        assert lon_arc_span(10.0, 30.0) == pytest.approx(20.0)

    def test_wrapping_span(self):
        # 350 -> 10 = 20 degrees of arc
        assert lon_arc_span(350.0, 10.0) == pytest.approx(20.0)

    def test_zero_arc(self):
        assert lon_arc_span(42.0, 42.0) == 0.0

    def test_full_circle_via_wrap(self):
        # If somehow start = epsilon and end = start - epsilon, span ~ 360.
        assert lon_arc_span(1.0, 0.999) == pytest.approx(359.999)


# ---------------------------------------------------------------------------
# point_in_lon_arc
# ---------------------------------------------------------------------------

class TestPointInLonArc:
    def test_simple_arc_contains_interior(self):
        assert point_in_lon_arc(20.0, 10.0, 30.0) is True

    def test_simple_arc_excludes_exterior(self):
        assert point_in_lon_arc(40.0, 10.0, 30.0) is False
        assert point_in_lon_arc(5.0, 10.0, 30.0) is False

    def test_simple_arc_endpoints_inclusive(self):
        assert point_in_lon_arc(10.0, 10.0, 30.0) is True
        assert point_in_lon_arc(30.0, 10.0, 30.0) is True

    def test_wrapping_arc_contains_zero_side(self):
        # arc 350 -> 10, point at 5 is inside.
        assert point_in_lon_arc(5.0, 350.0, 10.0) is True

    def test_wrapping_arc_contains_360_side(self):
        assert point_in_lon_arc(355.0, 350.0, 10.0) is True

    def test_wrapping_arc_excludes_far_side(self):
        # arc 350 -> 10 (a 20° band straddling 0), point at 180 is outside.
        assert point_in_lon_arc(180.0, 350.0, 10.0) is False

    def test_input_normalised(self):
        # Caller passes -5; should be treated as 355.
        assert point_in_lon_arc(-5.0, 350.0, 10.0) is True


# ---------------------------------------------------------------------------
# arcs_overlap
# ---------------------------------------------------------------------------

class TestArcsOverlap:
    def test_disjoint_simple(self):
        assert arcs_overlap(10.0, 30.0, 50.0, 60.0) is False

    def test_overlapping_simple(self):
        assert arcs_overlap(10.0, 30.0, 20.0, 40.0) is True

    def test_touching_boundary(self):
        # Arcs sharing exactly an endpoint count as overlapping.
        assert arcs_overlap(10.0, 30.0, 30.0, 40.0) is True

    def test_one_inside_other(self):
        assert arcs_overlap(10.0, 50.0, 20.0, 30.0) is True

    def test_wrapping_vs_simple_overlap(self):
        # arc1 wraps: [350, 10]; arc2 simple [0, 5] -> overlap.
        assert arcs_overlap(350.0, 10.0, 0.0, 5.0) is True

    def test_wrapping_vs_simple_disjoint(self):
        # arc1 wraps: [350, 10]; arc2 simple [100, 120] -> no overlap.
        assert arcs_overlap(350.0, 10.0, 100.0, 120.0) is False

    def test_two_wrappers(self):
        # Both wrap. arc1 [350, 10], arc2 [355, 5] -> overlap.
        assert arcs_overlap(350.0, 10.0, 355.0, 5.0) is True


# ---------------------------------------------------------------------------
# is_polar_degenerate
# ---------------------------------------------------------------------------

class TestIsPolarDegenerate:
    def test_normal_cabeus_strip_is_not_degenerate(self):
        # lat -85, arc span 4°
        assert is_polar_degenerate(bbox_max_lat=-84.5, arc_start=318.0, arc_end=322.0) is False

    def test_polar_wide_span_is_degenerate(self):
        # lat -89.5, arc span 200° -> degenerate
        assert is_polar_degenerate(bbox_max_lat=-89.5, arc_start=10.0, arc_end=217.0) is True

    def test_polar_narrow_span_is_not_degenerate(self):
        # lat -89, arc span 5° (rare but possible) -> not degenerate
        assert is_polar_degenerate(bbox_max_lat=-89.0, arc_start=100.0, arc_end=105.0) is False

    def test_wide_span_off_pole_is_not_degenerate(self):
        # lat -50, arc span 200°. Unusual but lon is well-defined here.
        assert is_polar_degenerate(bbox_max_lat=-50.0, arc_start=10.0, arc_end=217.0) is False


# ---------------------------------------------------------------------------
# obs_and_station
# ---------------------------------------------------------------------------

class TestObsAndStation:
    @pytest.mark.parametrize(
        "product_id,expected_obs,expected_stn",
        [
            ("ch2_ohr_nrp_20241115T1326321339_d_img_d18",
             "ch2_ohr_nrp_20241115T1326321339_d_img", "d18"),
            ("ch2_ohr_nrp_20240101T0000000000_d_img_d32",
             "ch2_ohr_nrp_20240101T0000000000_d_img", "d32"),
            ("ch2_ohr_nrp_20200229T0938004033_d_img_n18",
             "ch2_ohr_nrp_20200229T0938004033_d_img", "n18"),
            ("ch2_ohr_nrp_20210405T0047199117_d_img_gds",
             "ch2_ohr_nrp_20210405T0047199117_d_img", "gds"),
            ("ch2_ohr_nrp_20210401T2357376656_d_img_hw1",
             "ch2_ohr_nrp_20210401T2357376656_d_img", "hw1"),
            ("ch2_ohr_nrp_20190907T0438126359_d_img_g26",
             "ch2_ohr_nrp_20190907T0438126359_d_img", "g26"),
            ("ch2_ohr_ncp_20211223T0019163816_d_img_d32",
             "ch2_ohr_ncp_20211223T0019163816_d_img", "d32"),
        ],
    )
    def test_known_stations(self, product_id, expected_obs, expected_stn):
        obs, stn = obs_and_station(product_id)
        assert obs == expected_obs
        assert stn == expected_stn

    def test_no_underscore(self):
        assert obs_and_station("singleword") == ("singleword", None)
