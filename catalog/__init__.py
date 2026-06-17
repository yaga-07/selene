"""
selene/catalog -- local metadata index over PRADAN OHRC products.

Workflow:
    from catalog import build_index, load, query

    build_index("data/")               # scans XMLs, writes index.parquet
    df = load()                        # pandas DataFrame
    hits = query(
        bits_selection="lsb",
        min_solar_incidence=88.0,
        lat_range=(-90, -84),
        area="South Pole",
        processing_level="Raw",
    )

CLI:
    python -m catalog build [DATA_DIR]
    python -m catalog summary
    python -m catalog query --bits lsb --min-sun-incidence 88
    python -m catalog sql "SELECT product_id, bits_selection FROM cat WHERE ..."
"""

from .geometry import (
    arcs_overlap,
    is_polar_degenerate,
    lon_arc_from_corners,
    lon_arc_span,
    obs_and_station,
    point_in_lon_arc,
)
from .index import INDEX_JSONL, INDEX_META, INDEX_PARQUET, build_index, load
from .query import query, summary
from .scanner import parse_xml, scan_dir

__all__ = [
    "scan_dir",
    "parse_xml",
    "build_index",
    "load",
    "query",
    "summary",
    "INDEX_PARQUET",
    "INDEX_JSONL",
    "INDEX_META",
    "lon_arc_from_corners",
    "lon_arc_span",
    "point_in_lon_arc",
    "arcs_overlap",
    "is_polar_degenerate",
    "obs_and_station",
]
