"""
Query helpers over the catalog DataFrame.

The pandas API covers the common filters (bits_selection, sun-incidence
range, lat/lon bbox, area, processing_level, TDI, date range). Longitude
queries use the arc representation in catalog.geometry — they handle
antimeridian crossers and near-pole strips correctly. For arbitrary SQL,
use `catalog sql ...` from the CLI, which dispatches to DuckDB on the
Parquet file (install: `pip install duckdb`).
"""

from __future__ import annotations

from typing import Iterable

import pandas as pd

from .geometry import arcs_overlap, point_in_lon_arc
from .index import load as _load


def query(
    *,
    df: pd.DataFrame | None = None,
    bits_selection: str | Iterable[str] | None = None,
    tdi_stages: str | Iterable[str] | None = None,
    area: str | Iterable[str] | None = None,
    processing_level: str | Iterable[str] | None = None,
    role: str | Iterable[str] | None = None,
    station_id: str | Iterable[str] | None = None,
    min_solar_incidence: float | None = None,
    max_solar_incidence: float | None = None,
    min_sun_elevation: float | None = None,
    max_sun_elevation: float | None = None,
    min_altitude: float | None = None,
    max_altitude: float | None = None,
    min_pixel_resolution: float | None = None,
    max_pixel_resolution: float | None = None,
    lat_range: tuple[float, float] | None = None,
    lon_range: tuple[float, float] | None = None,
    bbox: tuple[float, float, float, float] | None = None,  # min_lat, min_lon, max_lat, max_lon
    point: tuple[float, float] | None = None,               # (lat, lon)
    start_after: str | pd.Timestamp | None = None,
    start_before: str | pd.Timestamp | None = None,
    orbit_number: int | Iterable[int] | None = None,
    exclude_truncated: bool = False,
    exclude_polar_degenerate: bool = False,
    unique_obs: bool = False,
    sort_by: str | Iterable[str] | None = None,
    ascending: bool = True,
    limit: int | None = None,
) -> pd.DataFrame:
    """
    Filter the catalog. All args are optional; the result is the AND of
    every constraint provided.

    Spatial filters use the longitude *arc* representation
    (`lon_arc_start`, `lon_arc_end`, `lon_wraps`):
      - `bbox` keeps products whose lat bbox AND lon arc overlap the query.
      - `point` keeps products whose lat bbox contains the lat AND whose
        lon arc contains the lon.
      - `lon_range = (lo, hi)` is treated as a simple (non-wrapping) arc
        on [0, 360); pass it normalised (e.g. (350, 10) for a wrap-style
        range is fine — it'll be interpreted as the wrap-arc).

    `unique_obs=True` collapses dual-station downlinks of the same
    observation. Tiebreaker: prefer rows with `is_truncated=False`, then
    alphabetical `station_id`.
    """
    d = df if df is not None else _load()

    def _isin(col: str, values) -> pd.Series:
        if isinstance(values, str):
            values = [values]
        return d[col].isin(list(values))

    mask = pd.Series(True, index=d.index)

    if bits_selection is not None and "bits_selection" in d.columns:
        mask &= _isin("bits_selection", bits_selection)
    if tdi_stages is not None and "tdi_stages" in d.columns:
        mask &= _isin("tdi_stages", tdi_stages)
    if area is not None and "area" in d.columns:
        mask &= _isin("area", area)
    if processing_level is not None and "processing_level" in d.columns:
        mask &= _isin("processing_level", processing_level)
    if role is not None and "role" in d.columns:
        mask &= _isin("role", role)
    if station_id is not None and "station_id" in d.columns:
        mask &= _isin("station_id", station_id)
    if orbit_number is not None and "imaging_orbit_number" in d.columns:
        mask &= _isin("imaging_orbit_number", orbit_number)

    if min_solar_incidence is not None:
        mask &= d["solar_incidence"] >= min_solar_incidence
    if max_solar_incidence is not None:
        mask &= d["solar_incidence"] <= max_solar_incidence
    if min_sun_elevation is not None:
        mask &= d["sun_elevation"] >= min_sun_elevation
    if max_sun_elevation is not None:
        mask &= d["sun_elevation"] <= max_sun_elevation
    if min_altitude is not None:
        mask &= d["spacecraft_altitude"] >= min_altitude
    if max_altitude is not None:
        mask &= d["spacecraft_altitude"] <= max_altitude
    if min_pixel_resolution is not None:
        mask &= d["pixel_resolution"] >= min_pixel_resolution
    if max_pixel_resolution is not None:
        mask &= d["pixel_resolution"] <= max_pixel_resolution

    if lat_range is not None:
        lo, hi = lat_range
        mask &= (d["bbox_max_lat"] >= lo) & (d["bbox_min_lat"] <= hi)
    if lon_range is not None:
        qlo, qhi = lon_range
        mask &= d.apply(
            lambda r: _lon_arc_overlap_safe(r, qlo, qhi), axis=1
        )
    if bbox is not None:
        qmin_lat, qmin_lon, qmax_lat, qmax_lon = bbox
        mask &= (d["bbox_max_lat"] >= qmin_lat) & (d["bbox_min_lat"] <= qmax_lat)
        mask &= d.apply(
            lambda r: _lon_arc_overlap_safe(r, qmin_lon, qmax_lon), axis=1
        )
    if point is not None:
        plat, plon = point
        mask &= (d["bbox_min_lat"] <= plat) & (d["bbox_max_lat"] >= plat)
        mask &= d.apply(lambda r: _point_in_arc_safe(r, plon), axis=1)

    if start_after is not None:
        mask &= d["start_date_time"] >= pd.to_datetime(start_after, utc=True)
    if start_before is not None:
        mask &= d["start_date_time"] <= pd.to_datetime(start_before, utc=True)

    if exclude_truncated and "is_truncated" in d.columns:
        mask &= ~d["is_truncated"].fillna(False).astype(bool)
    if exclude_polar_degenerate and "polar_degenerate" in d.columns:
        mask &= ~d["polar_degenerate"].fillna(False).astype(bool)

    out = d.loc[mask]

    if unique_obs and "obs_id" in out.columns:
        out = _collapse_to_unique_obs(out)

    if sort_by:
        out = out.sort_values(list([sort_by] if isinstance(sort_by, str) else sort_by),
                              ascending=ascending)
    if limit is not None:
        out = out.head(limit)
    return out


def _lon_arc_overlap_safe(row, qlo: float, qhi: float) -> bool:
    """Apply arcs_overlap to a row, treating NaN arc cols as no-match."""
    s = row.get("lon_arc_start")
    e = row.get("lon_arc_end")
    if pd.isna(s) or pd.isna(e):
        return False
    return arcs_overlap(float(s), float(e), float(qlo), float(qhi))


def _point_in_arc_safe(row, plon: float) -> bool:
    s = row.get("lon_arc_start")
    e = row.get("lon_arc_end")
    if pd.isna(s) or pd.isna(e):
        return False
    return point_in_lon_arc(float(plon), float(s), float(e))


def _collapse_to_unique_obs(df: pd.DataFrame) -> pd.DataFrame:
    """
    One row per (obs_id, role). Tiebreaker on dual-station groups:
    prefer non-truncated (False < True), then alphabetical station_id.
    """
    if "is_truncated" not in df.columns:
        # If we don't have the flag, just take the first by station.
        return df.sort_values(["obs_id", "role", "station_id"]).drop_duplicates(
            subset=["obs_id", "role"], keep="first"
        )
    # Sort so the preferred row sorts first per group.
    sorted_df = df.assign(
        _trunc_sort=df["is_truncated"].fillna(True).astype(bool).astype(int)
    ).sort_values(["obs_id", "role", "_trunc_sort", "station_id"])
    out = sorted_df.drop_duplicates(subset=["obs_id", "role"], keep="first")
    return out.drop(columns=["_trunc_sort"])


def summary(df: pd.DataFrame | None = None) -> dict:
    """Counts grouped by the dimensions you'll filter on most, with
    inventory-snapshot fields for Gate-0."""
    d = df if df is not None else _load()
    out: dict = {
        "n_products": int(len(d)),
    }
    if "obs_id" in d.columns:
        out["n_obs_unique"] = int(d["obs_id"].nunique())
        dual = d.groupby("obs_id").size()
        out["n_dual_station_observations"] = int((dual > 1).sum())
    if "is_truncated" in d.columns:
        out["n_truncated"] = int(d["is_truncated"].fillna(False).sum())
    if "polar_degenerate" in d.columns:
        out["n_polar_degenerate"] = int(d["polar_degenerate"].fillna(False).sum())
    if "lon_wraps" in d.columns:
        out["n_antimeridian_crossers"] = int(d["lon_wraps"].fillna(False).sum())
    if "role" in d.columns:
        out["by_role"] = d["role"].value_counts(dropna=False).to_dict()
    if "station_id" in d.columns:
        out["by_station"] = d["station_id"].value_counts(dropna=False).to_dict()
    if "bits_selection" in d.columns:
        out["by_bits_selection"] = d["bits_selection"].value_counts(dropna=False).to_dict()
    if "tdi_stages" in d.columns:
        out["by_tdi_stages"] = d["tdi_stages"].value_counts(dropna=False).to_dict()
    if "area" in d.columns:
        out["by_area"] = d["area"].value_counts(dropna=False).to_dict()
    if "role" in d.columns and "bits_selection" in d.columns:
        ct = pd.crosstab(d["role"], d["bits_selection"]).to_dict()
        out["by_role_x_bits"] = ct
    if "solar_incidence" in d.columns and d["solar_incidence"].notna().any():
        out["solar_incidence_range"] = (
            float(d["solar_incidence"].min()), float(d["solar_incidence"].max())
        )
    if "spacecraft_altitude" in d.columns and d["spacecraft_altitude"].notna().any():
        out["altitude_range"] = (
            float(d["spacecraft_altitude"].min()), float(d["spacecraft_altitude"].max())
        )
    if "bbox_min_lat" in d.columns and d["bbox_min_lat"].notna().any():
        out["lat_range"] = (
            float(d["bbox_min_lat"].min()), float(d["bbox_max_lat"].max())
        )
    if "start_date_time" in d.columns and d["start_date_time"].notna().any():
        out["date_range"] = (
            str(d["start_date_time"].min()), str(d["start_date_time"].max())
        )
    return out
