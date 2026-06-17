"""
Runnable query recipes for the catalog.

Assumes you have already built the index:
    python -m catalog build data

Run from the repo root:
    python -m catalog.examples.queries

Recipes that need extra dependencies (shapely, duckdb) check for them at
runtime and skip with a hint if missing.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pandas as pd

# Make `catalog` importable when this file is run as a script from the repo root.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import catalog  # noqa: E402


def header(title: str) -> None:
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def show(df: pd.DataFrame, cols: list[str] | None = None, limit: int = 20) -> None:
    if cols:
        cols = [c for c in cols if c in df.columns]
        df = df[cols]
    df = df.head(limit)
    with pd.option_context(
        "display.max_rows", 200,
        "display.max_columns", 200,
        "display.width", 220,
        "display.max_colwidth", 60,
    ):
        print(df.to_string(index=False) if len(df) else "(no rows)")


def recipe_psr_candidates() -> None:
    """1. True-PSR candidates: sun below the local horizon."""
    header("1. True-PSR candidates (solar_incidence >= 90)")
    hits = catalog.query(min_solar_incidence=90.0, sort_by="solar_incidence", ascending=False)
    show(hits, cols=["product_id", "role", "solar_incidence", "sun_elevation",
                     "bbox_min_lat", "bbox_max_lat", "area"])


def recipe_lsb_raw() -> None:
    """2. lsb-encoded raw products -- usable for low-signal denoising work."""
    header("2. lsb-encoded raw products")
    hits = catalog.query(bits_selection="lsb", role="raw",
                         sort_by="solar_incidence", ascending=False)
    show(hits, cols=["product_id", "bits_selection", "solar_incidence",
                     "tdi_stages", "spacecraft_altitude"])


def recipe_cabeus_bbox() -> None:
    """3. Products whose footprint bbox covers a point near Cabeus."""
    header("3. Footprint covers Cabeus (LCROSS impact: -84.675, 311.275)")
    hits = catalog.query(point=(-84.675, 311.275))
    show(hits, cols=["product_id", "role", "bbox_min_lat", "bbox_max_lat",
                     "bbox_min_lon", "bbox_max_lon", "solar_incidence"])


def recipe_raw_calibrated_pair() -> None:
    """4. Find raw + calibrated counterparts of the same observation."""
    header("4. Raw + calibrated pairs for the same observation (by timestamp stem)")
    df = catalog.load()
    # The timestamp portion is shared between nrp and ncp for the same observation.
    df = df.assign(stem=df["product_id"].str.extract(r"_(?:nrp|ncp)_([0-9T]+)_", expand=False))
    pair_counts = df.groupby("stem")["role"].nunique()
    paired_stems = pair_counts[pair_counts > 1].index.tolist()
    paired = df[df["stem"].isin(paired_stems)].sort_values(["stem", "role"])
    show(paired, cols=["stem", "role", "product_id", "bits_selection",
                       "solar_incidence", "img_path"])


def recipe_date_window() -> None:
    """5. South Pole observations in a date window."""
    header("5. South Pole observations between 2021-12-01 and 2022-01-01")
    hits = catalog.query(
        area="South Pole",
        start_after="2021-12-01",
        start_before="2022-01-01",
        sort_by="start_date_time",
    )
    show(hits, cols=["product_id", "role", "start_date_time",
                     "solar_incidence", "bits_selection"])


def recipe_integrity_check() -> None:
    """6. Sanity check: XML file_size matches the .img on disk."""
    header("6. Download integrity check (file_size vs .img on disk)")
    df = catalog.load()
    if "img_size_actual" not in df.columns:
        print("(rebuild the index -- this column was added recently)")
        return
    mismatched = df.loc[
        df["img_size_actual"].notna() & (df["file_size"] != df["img_size_actual"]),
        ["product_id", "file_size", "img_size_actual", "img_path"],
    ]
    if mismatched.empty:
        print("All indexed .img files match their declared file_size. Good.")
    else:
        show(mismatched)


def recipe_polygon_intersection() -> None:
    """7. Exact polygon point-in-quadrilateral (needs shapely)."""
    header("7. Exact polygon containment via shapely")
    if importlib.util.find_spec("shapely") is None:
        print("(skipped -- pip install shapely to enable this recipe)")
        return
    from shapely.geometry import Point, Polygon  # type: ignore
    df = catalog.load()

    def poly_for(row) -> Polygon:
        return Polygon([
            (row["upper_left_longitude"], row["upper_left_latitude"]),
            (row["upper_right_longitude"], row["upper_right_latitude"]),
            (row["lower_right_longitude"], row["lower_right_latitude"]),
            (row["lower_left_longitude"], row["lower_left_latitude"]),
        ])

    target = Point(311.0, -84.5)
    contains = df.apply(lambda r: poly_for(r).contains(target), axis=1)
    show(df[contains], cols=["product_id", "role", "upper_left_latitude",
                             "upper_left_longitude", "solar_incidence"])


def recipe_duckdb_sql() -> None:
    """8. Arbitrary SQL via DuckDB."""
    header("8. DuckDB SQL: bits + sun-incidence + bbox")
    if importlib.util.find_spec("duckdb") is None:
        print("(skipped -- pip install duckdb to enable this recipe)")
        return
    import duckdb  # type: ignore
    con = duckdb.connect()
    con.execute(f"CREATE VIEW cat AS SELECT * FROM read_parquet('{catalog.INDEX_PARQUET}')")
    res = con.execute("""
        SELECT product_id, role, bits_selection, solar_incidence,
               bbox_min_lat, bbox_max_lat
          FROM cat
         WHERE solar_incidence >= 83
           AND bbox_min_lat <= -84
         ORDER BY solar_incidence DESC
         LIMIT 20
    """).df()
    show(res)


def main() -> None:
    recipe_psr_candidates()
    recipe_lsb_raw()
    recipe_cabeus_bbox()
    recipe_raw_calibrated_pair()
    recipe_date_window()
    recipe_integrity_check()
    recipe_polygon_intersection()
    recipe_duckdb_sql()


if __name__ == "__main__":
    main()
