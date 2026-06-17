"""
Bbox diagnostic over OHRC raw XMLs.

For each product XML found under DATA_ROOT, parse the four corner lat/lon
coordinates and characterise the longitude span. The point is to decide
between two bbox designs:

  (a) naive min..max lon (current scanner)
  (b) antimeridian-aware: detect wraparound and represent as two intervals

The trap: naive span > 180 is NOT the same as antimeridian crossing. A
strip near the lunar pole can legitimately span a wide longitude range
without crossing 0/360, because longitude is degenerate at the pole.

For each row we report:
  span_naive    = max(lon) - min(lon)
  largest_gap   = the biggest cyclic gap between sorted corner lons
                  (= 360 - span_arc on the cyclic circle)
  span_cyclic   = 360 - largest_gap   (= the actual arc the strip occupies)
  wraps         = (span_naive > 180) and the strip really does cross 0/360
                  -- inferred from where the largest gap lies

For the bbox question the right span is span_cyclic.
"""

from __future__ import annotations

import json
import statistics
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

DATA_ROOT = Path("/Volumes/lazarus/ohrc-data")

NS = {
    "pds": "http://pds.nasa.gov/pds4/pds/v1",
    "isda": "https://isda.issdc.gov.in/pds4/isda/v1",
}

CORNER_TAGS = [
    ("upper_left_latitude",  "upper_left_longitude"),
    ("upper_right_latitude", "upper_right_longitude"),
    ("lower_left_latitude",  "lower_left_longitude"),
    ("lower_right_latitude", "lower_right_longitude"),
]


def first_text(elem: ET.Element, local_name: str) -> str | None:
    for child in elem.iter():
        tag = child.tag.split("}", 1)[-1]
        if tag == local_name and child.text and child.text.strip():
            return child.text.strip()
    return None


def parse_corners(xml_path: Path) -> list[tuple[float, float]] | None:
    """Return [(lat, lon), ...] from System_Level_Coordinates."""
    try:
        root = ET.parse(xml_path).getroot()
    except ET.ParseError:
        return None
    # Find the System_Level_Coordinates element (avoid Refined block)
    slc = None
    for elem in root.iter():
        if elem.tag.split("}", 1)[-1] == "System_Level_Coordinates":
            slc = elem
            break
    if slc is None:
        return None
    coords = []
    for lat_tag, lon_tag in CORNER_TAGS:
        lat = first_text(slc, lat_tag)
        lon = first_text(slc, lon_tag)
        if lat is None or lon is None:
            return None
        try:
            coords.append((float(lat), float(lon)))
        except ValueError:
            return None
    return coords


def cyclic_span(lons: list[float]) -> tuple[float, float, bool]:
    """
    Given longitudes in [0, 360), return:
      span_naive    = max - min  (the wrong, current measure)
      span_cyclic   = arc length actually covered, accounting for wrap
      wraps_antimeridian = True iff the cyclic arc straddles 0/360

    Method: sort lons, compute consecutive cyclic gaps, the largest gap
    is the *complement* of the occupied arc. span_cyclic = 360 - max_gap.
    If the largest gap is between the highest and lowest lon (wrapping
    through 360->0), then the arc does NOT cross the antimeridian. If
    the largest gap is between two interior lons, the arc DOES cross
    0/360.
    """
    if not lons:
        return 0.0, 0.0, False
    lons_mod = sorted(L % 360.0 for L in lons)
    span_naive = max(lons_mod) - min(lons_mod)
    # consecutive cyclic gaps
    gaps = []
    for i in range(len(lons_mod) - 1):
        gaps.append((lons_mod[i + 1] - lons_mod[i], i))
    # wrap gap (from last back to first through 360)
    wrap_gap = 360.0 - lons_mod[-1] + lons_mod[0]
    gaps.append((wrap_gap, len(lons_mod) - 1))
    gaps.sort(key=lambda x: -x[0])
    max_gap, max_idx = gaps[0]
    span_cyclic = 360.0 - max_gap
    # crosses antimeridian iff the largest gap is NOT the wrap gap
    wraps = (max_idx != len(lons_mod) - 1)
    return span_naive, span_cyclic, wraps


def main() -> None:
    xmls = sorted(DATA_ROOT.rglob("data/raw/*/ch2_ohr_*.xml"))
    print(f"found {len(xmls)} raw product XMLs under {DATA_ROOT}")

    rows = []
    for xml in xmls:
        coords = parse_corners(xml)
        if coords is None:
            rows.append({"xml": str(xml), "ok": False})
            continue
        lats = [c[0] for c in coords]
        lons = [c[1] for c in coords]
        span_naive, span_cyclic, wraps = cyclic_span(lons)
        rows.append({
            "xml": xml.name,
            "ok": True,
            "lats": lats,
            "lons": lons,
            "min_lat": min(lats),
            "max_lat": max(lats),
            "span_naive": span_naive,
            "span_cyclic": span_cyclic,
            "wraps": wraps,
        })

    ok = [r for r in rows if r["ok"]]
    n = len(ok)
    print(f"parsed {n} / {len(rows)} successfully\n")

    # --- summary stats ---
    naive_spans = [r["span_naive"] for r in ok]
    cyclic_spans = [r["span_cyclic"] for r in ok]
    print("=== longitude span distribution ===")
    def buckets(values: list[float], edges: list[float]) -> str:
        counts = [0] * (len(edges) + 1)
        labels = [f"<{edges[0]}"]
        for lo, hi in zip(edges[:-1], edges[1:]):
            labels.append(f"{lo}-{hi}")
        labels.append(f">={edges[-1]}")
        for v in values:
            placed = False
            for i, e in enumerate(edges):
                if v < e:
                    counts[i] += 1
                    placed = True
                    break
            if not placed:
                counts[-1] += 1
        return "\n".join(f"  {l:>10s}: {c:>4d}" for l, c in zip(labels, counts))

    edges = [0.1, 1, 10, 30, 90, 180, 270, 359]
    print(f"\nspan_naive  (max(lon) - min(lon)):")
    print(buckets(naive_spans, edges))
    print(f"  median = {statistics.median(naive_spans):.3f}")
    print(f"  max    = {max(naive_spans):.3f}")

    print(f"\nspan_cyclic (actual arc occupied):")
    print(buckets(cyclic_spans, edges))
    print(f"  median = {statistics.median(cyclic_spans):.3f}")
    print(f"  max    = {max(cyclic_spans):.3f}")

    # --- antimeridian crossers vs wide-near-pole strips ---
    wraps_naive = [r for r in ok if r["span_naive"] > 180]
    truly_wrapping = [r for r in ok if r["wraps"]]
    big_span_not_wrapping = [r for r in ok if r["span_naive"] > 180 and not r["wraps"]]

    print(f"\n=== misclassification check ===")
    print(f"naive span > 180:                       {len(wraps_naive)} strips")
    print(f"truly wraps antimeridian (cyclic test): {len(truly_wrapping)} strips")
    print(f"naive flag WRONG (wide-pole not wrap):  {len(big_span_not_wrapping)} strips")

    # --- spread of latitudes for the false positives ---
    if big_span_not_wrapping:
        print(f"\nfalse-positive lat ranges (would be miscoded with the > 180 heuristic):")
        for r in big_span_not_wrapping[:10]:
            print(f"  {r['xml']:>60s}  lat=[{r['min_lat']:.2f},{r['max_lat']:.2f}]  "
                  f"naive={r['span_naive']:.1f}  cyclic={r['span_cyclic']:.1f}")
        if len(big_span_not_wrapping) > 10:
            print(f"  ... and {len(big_span_not_wrapping) - 10} more")

    if truly_wrapping:
        print(f"\ntrue antimeridian crossers:")
        for r in truly_wrapping[:10]:
            print(f"  {r['xml']:>60s}  lat=[{r['min_lat']:.2f},{r['max_lat']:.2f}]  "
                  f"cyclic={r['span_cyclic']:.2f}  lons={[f'{l:.2f}' for l in r['lons']]}")
        if len(truly_wrapping) > 10:
            print(f"  ... and {len(truly_wrapping) - 10} more")

    # --- proximity to pole ---
    near_pole = [r for r in ok if r["min_lat"] < -85]
    print(f"\nstrips reaching below -85 lat: {len(near_pole)}")
    print(f"strips reaching below -89 lat: {sum(1 for r in ok if r['min_lat'] < -89)}")

    # save the full table for follow-up
    out = Path("analysis/bbox_diag.jsonl")
    with out.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"\nwrote per-strip diagnostic to {out}")


if __name__ == "__main__":
    main()
