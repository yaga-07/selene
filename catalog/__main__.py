"""
CLI:

    python -m catalog build [DATA_DIR]
    python -m catalog summary
    python -m catalog show [--limit N] [--cols a,b,c] [--unique-obs]
    python -m catalog query --bits lsb --min-sun-incidence 88 --lat -90 -84
    python -m catalog sql "SELECT product_id, bits_selection FROM cat WHERE solar_incidence > 88"

The `sql` subcommand needs duckdb (`pip install duckdb`). All other
subcommands run on pandas alone.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from .index import INDEX_PARQUET, build_index, load
from .query import _collapse_to_unique_obs, query, summary

DEFAULT_SHOW_COLS = [
    "product_id", "station_id", "role", "bits_selection", "tdi_stages",
    "solar_incidence", "sun_elevation", "spacecraft_altitude",
    "pixel_resolution", "area",
    "bbox_min_lat", "bbox_max_lat", "lon_arc_start", "lon_arc_end", "lon_wraps",
    "start_date_time",
]


def _print_df(df: pd.DataFrame, cols: list[str] | None, limit: int | None) -> None:
    if cols:
        cols = [c for c in cols if c in df.columns]
        df = df[cols]
    if limit:
        df = df.head(limit)
    with pd.option_context(
        "display.max_rows", 200,
        "display.max_columns", 200,
        "display.width", 220,
        "display.max_colwidth", 60,
    ):
        print(df.to_string(index=False))


def _cmd_build(args: argparse.Namespace) -> int:
    df = build_index(args.data_dir, only_data_artifact=not args.all_artifacts)
    print(f"\n{len(df)} rows, {len(df.columns)} columns")
    return 0


def _cmd_summary(args: argparse.Namespace) -> int:
    s = summary()
    print(json.dumps(s, indent=2, default=str))
    return 0


def _cmd_show(args: argparse.Namespace) -> int:
    df = load()
    if args.unique_obs:
        df = _collapse_to_unique_obs(df)
    cols = args.cols.split(",") if args.cols else DEFAULT_SHOW_COLS
    _print_df(df, cols, args.limit)
    return 0


def _cmd_query(args: argparse.Namespace) -> int:
    kw: dict = {}
    if args.bits:
        kw["bits_selection"] = args.bits
    if args.tdi:
        kw["tdi_stages"] = args.tdi
    if args.area:
        kw["area"] = args.area
    if args.processing_level:
        kw["processing_level"] = args.processing_level
    if args.role:
        kw["role"] = args.role
    if args.station:
        kw["station_id"] = args.station
    if args.min_sun_incidence is not None:
        kw["min_solar_incidence"] = args.min_sun_incidence
    if args.max_sun_incidence is not None:
        kw["max_solar_incidence"] = args.max_sun_incidence
    if args.min_altitude is not None:
        kw["min_altitude"] = args.min_altitude
    if args.max_altitude is not None:
        kw["max_altitude"] = args.max_altitude
    if args.lat:
        kw["lat_range"] = tuple(args.lat)
    if args.lon:
        kw["lon_range"] = tuple(args.lon)
    if args.bbox:
        kw["bbox"] = tuple(args.bbox)
    if args.point:
        kw["point"] = tuple(args.point)
    if args.start_after:
        kw["start_after"] = args.start_after
    if args.start_before:
        kw["start_before"] = args.start_before
    if args.exclude_truncated:
        kw["exclude_truncated"] = True
    if args.exclude_polar_degenerate:
        kw["exclude_polar_degenerate"] = True
    if args.unique_obs:
        kw["unique_obs"] = True
    if args.sort:
        kw["sort_by"] = args.sort
        kw["ascending"] = not args.desc
    if args.limit:
        kw["limit"] = args.limit

    out = query(**kw)
    cols = args.cols.split(",") if args.cols else DEFAULT_SHOW_COLS
    print(f"# {len(out)} match(es)")
    _print_df(out, cols, None)
    if args.out:
        Path(args.out).write_text(out.to_csv(index=False))
        print(f"# wrote {args.out}")
    return 0


def _cmd_sql(args: argparse.Namespace) -> int:
    try:
        import duckdb  # noqa: F401
    except ImportError:
        print("error: duckdb is not installed. run: pip install duckdb", file=sys.stderr)
        return 2
    import duckdb  # type: ignore
    con = duckdb.connect()
    con.execute(f"CREATE VIEW cat AS SELECT * FROM read_parquet('{INDEX_PARQUET}')")
    res = con.execute(args.sql).df()
    _print_df(res, None, None)
    if args.out:
        res.to_csv(args.out, index=False)
        print(f"# wrote {args.out}")
    return 0


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="catalog", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("build", help="scan DATA_DIR and write Parquet+JSONL index")
    sp.add_argument("data_dir", help="path to the root of the PRADAN data tree")
    sp.add_argument("--all-artifacts", action="store_true",
                    help="include browse/geometry/misc XMLs (default: data only)")
    sp.set_defaults(func=_cmd_build)

    sp = sub.add_parser("summary", help="print counts and ranges")
    sp.set_defaults(func=_cmd_summary)

    sp = sub.add_parser("show", help="print rows (default cols, or --cols a,b,c)")
    sp.add_argument("--cols", help="comma-separated columns")
    sp.add_argument("--limit", type=int)
    sp.add_argument("--unique-obs", action="store_true",
                    help="collapse dual-station downlinks to one row per observation")
    sp.set_defaults(func=_cmd_show)

    sp = sub.add_parser("query", help="filter the catalog with typed flags")
    sp.add_argument("--bits", nargs="+", help="bits_selection: lsb mid msb")
    sp.add_argument("--tdi", nargs="+", help="tdi_stages, e.g. TDI256")
    sp.add_argument("--area", nargs="+")
    sp.add_argument("--processing-level", nargs="+", choices=["Raw", "Calibrated"])
    sp.add_argument("--role", nargs="+", choices=["raw", "calibrated"])
    sp.add_argument("--station", nargs="+",
                    help="ground station id (d18 d32 g26 gds hw1 n18)")
    sp.add_argument("--min-sun-incidence", type=float)
    sp.add_argument("--max-sun-incidence", type=float)
    sp.add_argument("--min-altitude", type=float)
    sp.add_argument("--max-altitude", type=float)
    sp.add_argument("--lat", nargs=2, type=float, metavar=("LO", "HI"))
    sp.add_argument("--lon", nargs=2, type=float, metavar=("LO", "HI"))
    sp.add_argument("--bbox", nargs=4, type=float,
                    metavar=("MINLAT", "MINLON", "MAXLAT", "MAXLON"))
    sp.add_argument("--point", nargs=2, type=float, metavar=("LAT", "LON"))
    sp.add_argument("--start-after", help="ISO datetime")
    sp.add_argument("--start-before", help="ISO datetime")
    sp.add_argument("--exclude-truncated", action="store_true",
                    help="drop products whose .img is missing or wrong size")
    sp.add_argument("--exclude-polar-degenerate", action="store_true",
                    help="drop near-pole strips where lon bbox is unreliable")
    sp.add_argument("--unique-obs", action="store_true",
                    help="collapse dual-station downlinks to one row per observation")
    sp.add_argument("--sort", help="column to sort by")
    sp.add_argument("--desc", action="store_true")
    sp.add_argument("--limit", type=int)
    sp.add_argument("--cols", help="comma-separated columns")
    sp.add_argument("--out", help="write matches to CSV")
    sp.set_defaults(func=_cmd_query)

    sp = sub.add_parser("sql", help="run arbitrary SQL via DuckDB (view name: cat)")
    sp.add_argument("sql", help="SQL string. The index is exposed as a view named `cat`.")
    sp.add_argument("--out", help="write result to CSV")
    sp.set_defaults(func=_cmd_sql)

    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
