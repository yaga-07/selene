"""
Index persistence: scan -> Parquet + JSONL on disk.

Paths in the parquet are stored *relative* to the data_root that was used
at build time. The build records the data_root in `_meta.json` alongside
the parquet. `load()` re-anchors paths back to absolute on read, with the
data_root resolved in priority order:
    1. explicit `data_root` argument
    2. SELENE_DATA_ROOT env var
    3. the data_root recorded in _meta.json

This keeps the index portable: re-mount the SSD anywhere, point
SELENE_DATA_ROOT at the new mount, and the same index keeps working
without a rebuild.
"""

from __future__ import annotations

import datetime as dt
import json
import os
from pathlib import Path

import pandas as pd

from .scanner import scan_dir

REPO_ROOT = Path(__file__).resolve().parents[1]
INDEX_DIR = REPO_ROOT / "catalog" / "_index"
INDEX_PARQUET = INDEX_DIR / "index.parquet"
INDEX_JSONL = INDEX_DIR / "index.jsonl"
INDEX_META = INDEX_DIR / "_meta.json"

# Paths to re-anchor when loading. Each entry is (rel_col, absolute_col).
_PATH_COLS: tuple[tuple[str, str], ...] = (
    ("xml_path_rel", "xml_path"),
    ("img_path_rel", "img_path"),
    ("product_dir_rel", "product_dir"),
    ("peer_xml_path_rel", "peer_xml_path"),
)


def _to_relative(absolute: str | None, data_root: Path) -> str | None:
    if absolute is None or (isinstance(absolute, float) and pd.isna(absolute)):
        return None
    p = Path(absolute)
    try:
        return str(p.resolve().relative_to(data_root.resolve()))
    except ValueError:
        # path is outside data_root -- keep as-is (caller can detect)
        return str(p)


def build_index(
    data_dir: Path | str,
    *,
    only_data_artifact: bool = True,
    verbose: bool = True,
) -> pd.DataFrame:
    """Scan `data_dir`, build the index, write Parquet + JSONL + _meta.json."""
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    data_root = Path(data_dir).resolve()
    if not data_root.exists():
        raise FileNotFoundError(f"data_root does not exist: {data_root}")

    rows = list(scan_dir(data_root, only_data_artifact=only_data_artifact))
    if not rows:
        raise RuntimeError(f"no product XMLs found under {data_root}")

    df = pd.DataFrame(rows)
    df = df.sort_values(["product_id", "role"], na_position="last").reset_index(drop=True)

    # Convert absolute paths to relative-to-data_root; drop the absolute ones.
    for abs_col, _ in (("xml_path", "_"), ("img_path", "_"),
                      ("product_dir", "_"), ("peer_xml_path", "_")):
        if abs_col in df.columns:
            df[f"{abs_col}_rel"] = df[abs_col].apply(lambda v: _to_relative(v, data_root))
            df = df.drop(columns=[abs_col])

    # Stable column order.
    id_cols = ["product_id", "obs_id", "station_id", "role", "artifact",
               "processing_level", "area", "projection"]
    time_cols = ["start_date_time", "stop_date_time"]
    pp_cols = [
        "bits_selection", "tdi_stages",
        "line_exposure_duration", "spacecraft_altitude", "pixel_resolution",
        "detector_pixel_width", "focal_length",
        "imaging_orbit_number", "dumping_orbit_number",
        "roll", "pitch", "yaw",
        "sun_azimuth", "sun_elevation", "solar_incidence",
        "orbit_limb_direction", "spacecraft_yaw_direction", "reference_data_used",
    ]
    geom_cols = [
        "bbox_min_lat", "bbox_max_lat",
        "lon_arc_start", "lon_arc_end", "lon_wraps", "polar_degenerate",
        "upper_left_latitude", "upper_left_longitude",
        "upper_right_latitude", "upper_right_longitude",
        "lower_left_latitude", "lower_left_longitude",
        "lower_right_latitude", "lower_right_longitude",
        "refined_upper_left_latitude", "refined_upper_left_longitude",
        "refined_upper_right_latitude", "refined_upper_right_longitude",
        "refined_lower_left_latitude", "refined_lower_left_longitude",
        "refined_lower_right_latitude", "refined_lower_right_longitude",
    ]
    file_cols = [
        "file_name", "file_size", "img_size_actual", "is_truncated",
        "md5_checksum", "creation_date_time", "line_count", "sample_count",
        "xml_path_rel", "img_path_rel", "peer_xml_path_rel", "product_dir_rel",
        "logical_identifier", "version_id", "title",
        "job_id", "level0_dir_name",
    ]
    preferred = id_cols + time_cols + pp_cols + geom_cols + file_cols
    rest = [c for c in df.columns if c not in preferred]
    ordered = [c for c in preferred if c in df.columns] + sorted(rest)
    df = df[ordered]

    # Parse datetimes
    for c in ("start_date_time", "stop_date_time", "creation_date_time"):
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce", utc=True)

    df.to_parquet(INDEX_PARQUET, index=False)
    with INDEX_JSONL.open("w") as f:
        for rec in df.to_dict(orient="records"):
            f.write(json.dumps(rec, default=str) + "\n")

    meta = {
        "data_root": str(data_root),
        "build_time": dt.datetime.now(dt.timezone.utc).isoformat(),
        "n_products": int(len(df)),
        "n_columns": int(len(df.columns)),
        "scanner_version": 2,
    }
    INDEX_META.write_text(json.dumps(meta, indent=2))

    if verbose:
        print(f"indexed {len(df)} product(s) -> {INDEX_PARQUET}")
        print(f"jsonl mirror              -> {INDEX_JSONL}")
        print(f"build metadata            -> {INDEX_META}")
        print(f"data_root recorded        -> {data_root}")
    return df


def _resolve_data_root(data_root: str | Path | None) -> Path | None:
    """Resolution order: arg > env > meta.json."""
    if data_root is not None:
        return Path(data_root).expanduser().resolve()
    env = os.environ.get("SELENE_DATA_ROOT")
    if env:
        return Path(env).expanduser().resolve()
    if INDEX_META.exists():
        try:
            meta = json.loads(INDEX_META.read_text())
            recorded = meta.get("data_root")
            if recorded:
                return Path(recorded)
        except json.JSONDecodeError:
            pass
    return None


def load(
    parquet: Path | str = INDEX_PARQUET,
    *,
    data_root: str | Path | None = None,
) -> pd.DataFrame:
    """
    Load the on-disk index and re-anchor relative paths to a data_root.

    Adds absolute path columns (`xml_path`, `img_path`, `product_dir`,
    `peer_xml_path`) derived from the corresponding `_rel` columns. If
    no data_root can be resolved, the absolute columns are left as the
    relative strings and a warning is printed.
    """
    p = Path(parquet)
    if not p.exists():
        raise FileNotFoundError(
            f"no index at {p}. Run `python -m catalog build <DATA_DIR>` first."
        )
    df = pd.read_parquet(p)
    root = _resolve_data_root(data_root)
    if root is None:
        import warnings
        warnings.warn(
            "no data_root resolved (pass data_root=, set SELENE_DATA_ROOT, "
            "or rebuild the index). Paths in the loaded DataFrame will be "
            "the relative strings stored in the parquet.",
            stacklevel=2,
        )
        for rel_col, abs_col in _PATH_COLS:
            if rel_col in df.columns:
                df[abs_col] = df[rel_col]
        return df
    if not root.exists():
        import warnings
        warnings.warn(
            f"data_root does not exist on disk: {root}. Paths may be invalid "
            "until the SSD is remounted or SELENE_DATA_ROOT is updated.",
            stacklevel=2,
        )
    for rel_col, abs_col in _PATH_COLS:
        if rel_col in df.columns:
            df[abs_col] = df[rel_col].apply(
                lambda r: str(root / r) if isinstance(r, str) and r else None
            )
    return df
