"""
XML scanner -- finds every OHRC product label under a data root and parses
all isda:* / pds:* fields into a flat dict, one row per product.

Designed to be schema-tolerant: any new isda:* leaf the labels grow will
land in the output dict with its raw name (snake_cased), so the index
keeps working even if ISRO adds fields to future products.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Iterable

from .geometry import (
    is_polar_degenerate,
    lon_arc_from_corners,
    obs_and_station,
)

NS = {
    "pds": "http://pds.nasa.gov/pds4/pds/v1",
    "isda": "https://isda.issdc.gov.in/pds4/isda/v1",
}

# tags we never want as columns (containers, big text blobs, refs)
_SKIP_TAGS = {
    "Identification_Area",
    "Modification_History",
    "Modification_Detail",
    "Investigation_Area",
    "Internal_Reference",
    "Observing_System",
    "Observing_System_Component",
    "Target_Identification",
    "Observation_Area",
    "Mission_Area",
    "Primary_Result_Summary",
    "Time_Coordinates",
    "Product_Parameters",
    "Geometry_Parameters",
    "System_Level_Coordinates",
    "Refined_Corner_Coordinates",
    "File_Area_Observational",
    "File",
    "Array_2D_Image",
    "Element_Array",
    "Axis_Array",
}

# tags whose text we DO want (typed scalars). Anything not in here that
# is also not in _SKIP_TAGS is captured as a generic string.
_FLOAT_TAGS = {
    "line_exposure_duration",
    "detector_pixel_width",
    "focal_length",
    "spacecraft_altitude",
    "pixel_resolution",
    "roll",
    "pitch",
    "yaw",
    "sun_azimuth",
    "sun_elevation",
    "solar_incidence",
    "upper_left_latitude",
    "upper_left_longitude",
    "upper_right_latitude",
    "upper_right_longitude",
    "lower_left_latitude",
    "lower_left_longitude",
    "lower_right_latitude",
    "lower_right_longitude",
    "file_size",
}
_INT_TAGS = {
    "imaging_orbit_number",
    "dumping_orbit_number",
    "elements",
    "offset",
}

# Recognized product role markers in the filename / path.
_NRP_RE = re.compile(r"_n[rc]p_", re.IGNORECASE)


def _local(tag: str) -> str:
    """Strip XML namespace from a tag name."""
    return tag.split("}", 1)[1] if "}" in tag else tag


def _to_number(name: str, text: str) -> Any:
    txt = (text or "").strip()
    if not txt:
        return None
    try:
        if name in _INT_TAGS:
            return int(txt)
        if name in _FLOAT_TAGS:
            return float(txt)
    except ValueError:
        return txt
    return txt


def _walk(elem: ET.Element, into: dict[str, Any], *, prefix: str = "") -> None:
    """
    Recursively flatten an element subtree into `into`.

    Container tags in _SKIP_TAGS are descended into but contribute no
    column; leaf tags become `into[name] = value`. If a tag repeats
    (e.g. upper_left_latitude appears once in System_Level_Coordinates
    and again in Refined_Corner_Coordinates), the refined version is
    written under `refined_<name>`.
    """
    name = _local(elem.tag)
    children = list(elem)

    if children:
        is_refined = (name == "Refined_Corner_Coordinates")
        local_prefix = "refined_" if is_refined else prefix
        for child in children:
            _walk(child, into, prefix=local_prefix)
        return

    if name in _SKIP_TAGS:
        return

    col = f"{prefix}{name}"
    val = _to_number(name, elem.text or "")
    unit = elem.attrib.get("unit")
    # Don't clobber an existing typed value with empty string
    if val is None and col in into:
        return
    into[col] = val
    if unit:
        into[f"{col}_unit"] = unit


def _peer_path(xml_path: Path, label: str) -> Path | None:
    """
    Given a raw (nrp) or calibrated (ncp) product XML, return the path to
    its counterpart variant (ncp <-> nrp). Returns None if not found.
    """
    name = xml_path.name
    stem = xml_path.stem
    if "_nrp_" in name:
        peer_name = name.replace("_nrp_", "_ncp_")
        peer_stem = stem.replace("_nrp_", "_ncp_")
    elif "_ncp_" in name:
        peer_name = name.replace("_ncp_", "_nrp_")
        peer_stem = stem.replace("_ncp_", "_nrp_")
    else:
        return None
    # try in the same directory first
    candidate = xml_path.with_name(peer_name)
    if candidate.exists():
        return candidate
    # otherwise glob under the data root
    return None


def parse_xml(xml_path: Path) -> dict[str, Any]:
    """Parse a single product XML into a flat row dict."""
    tree = ET.parse(xml_path)
    root = tree.getroot()

    row: dict[str, Any] = {}
    _walk(root, row)

    # add path / identity fields the walker doesn't capture
    row["xml_path"] = str(xml_path.resolve())
    row["product_dir"] = str(xml_path.parents[3].resolve()) if len(xml_path.parents) >= 4 else str(xml_path.parent)
    row["product_id"] = xml_path.stem  # e.g. ch2_ohr_nrp_2021..._d_img_d32

    # locate the .img sibling (file_name extracted from XML, lives in same dir)
    img_name = row.get("file_name")
    if img_name:
        img_path = xml_path.with_name(img_name)
        row["img_path"] = str(img_path) if img_path.exists() else None
        if img_path.exists():
            row["img_size_actual"] = img_path.stat().st_size

    # dimensions: Axis_Array has axis_name=Line and axis_name=Sample with elements
    # The walker captures the last "elements" only. Reparse axes explicitly.
    axes = root.findall(".//pds:Array_2D_Image/pds:Axis_Array", NS)
    for ax in axes:
        nm_el = ax.find("pds:axis_name", NS)
        el_el = ax.find("pds:elements", NS)
        if nm_el is None or el_el is None:
            continue
        nm = (nm_el.text or "").strip().lower()
        try:
            row[f"{nm}_count"] = int(el_el.text)
        except (TypeError, ValueError):
            pass
    # drop the noisy generic "elements" / "sequence_number" leftovers
    for k in ("elements", "sequence_number", "axis_name", "axis_index_order", "data_type", "axes"):
        row.pop(k, None)

    # tag the product role from path tokens
    role = None
    pl = row.get("processing_level")
    if pl:
        role = pl.strip().lower()
    elif "_nrp_" in xml_path.name.lower():
        role = "raw"
    elif "_ncp_" in xml_path.name.lower():
        role = "calibrated"
    row["role"] = role

    # Tag the artifact type from the path. The product layout is
    #   <product_dir>/<artifact>/<role>/<YYYYMMDD>/<file>
    # so the artifact is the directory three levels up from the file.
    artifact = None
    if len(xml_path.parts) >= 4:
        candidate = xml_path.parts[-4].lower()
        if candidate in ("data", "browse", "geometry", "miscellaneous"):
            artifact = candidate
    row["artifact"] = artifact or "data"

    # peer variant
    peer = _peer_path(xml_path, role or "")
    row["peer_xml_path"] = str(peer) if peer else None

    # obs_id / station_id from product_id
    obs_id, station_id = obs_and_station(row["product_id"])
    row["obs_id"] = obs_id
    row["station_id"] = station_id

    # Longitude arc (cyclic-gap method). Robust to antimeridian crossers
    # and to wide-near-pole strips. See catalog.geometry for the design
    # notes.
    corner_lons = [
        row.get(f"{c}_longitude") for c in
        ("upper_left", "upper_right", "lower_left", "lower_right")
    ]
    corner_lats = [
        row.get(f"{c}_latitude") for c in
        ("upper_left", "upper_right", "lower_left", "lower_right")
    ]
    if all(isinstance(L, (int, float)) for L in corner_lons) and \
       all(isinstance(L, (int, float)) for L in corner_lats):
        arc_start, arc_end, wraps = lon_arc_from_corners(corner_lons)
        row["lon_arc_start"] = arc_start
        row["lon_arc_end"] = arc_end
        row["lon_wraps"] = wraps
        row["bbox_min_lat"] = min(corner_lats)
        row["bbox_max_lat"] = max(corner_lats)
        row["polar_degenerate"] = is_polar_degenerate(
            bbox_max_lat=row["bbox_max_lat"],
            arc_start=arc_start,
            arc_end=arc_end,
        )
    else:
        row["lon_arc_start"] = None
        row["lon_arc_end"] = None
        row["lon_wraps"] = None
        row["bbox_min_lat"] = None
        row["bbox_max_lat"] = None
        row["polar_degenerate"] = None

    # Truncation flag: img missing -> truncated; declared size disagrees -> truncated.
    declared = row.get("file_size")
    actual = row.get("img_size_actual")
    if actual is None:
        row["is_truncated"] = True
    elif declared is None:
        row["is_truncated"] = None  # can't tell
    else:
        row["is_truncated"] = (declared != actual)

    return row


def scan_dir(root: Path | str, *, only_data_artifact: bool = True) -> Iterable[dict[str, Any]]:
    """
    Yield one row dict per product XML found under `root`.

    By default skips browse-thumbnail XMLs, geometry-grid XMLs, and
    miscellaneous metadata — only the main `data/.../*.xml` labels (the
    .img-bearing products) are returned. Set only_data_artifact=False
    to include every XML.
    """
    root = Path(root).resolve()
    for xml in sorted(root.rglob("*.xml")):
        row = parse_xml(xml)
        if only_data_artifact and row.get("artifact") != "data":
            continue
        yield row
