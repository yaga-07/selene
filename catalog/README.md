# catalog

Local, queryable metadata index over Chandrayaan-2 Orbiter High Resolution
Camera (OHRC) products downloaded from ISRO's PRADAN archive.

PRADAN's web filters are limited (orbit number, date, a coarse area
selector). Once you have the data on disk, every PDS4 metadata field —
`bits_selection`, `solar_incidence`, footprint lat/lon, TDI stages,
altitude, exposure, orbit number, area, processing level, refined corner
coordinates, ... — becomes indexable. This component walks the directory
tree, parses each product label into a flat row, persists the result as
Parquet (plus a greppable JSONL mirror), and exposes a Python API and CLI
for arbitrary queries.

The scanner is schema-tolerant: any new `isda:*` leaf the labels grow will
land in the index as a new column automatically, so it keeps working when
ISRO publishes future product revisions. The typed filters in
`catalog.query()` are tuned for OHRC; the same machinery indexes other
PDS4 instruments (TMC-2, IIRS) with no parser changes, but you'll get
plain string columns for the instrument-specific fields rather than typed
ranges until you extend `_FLOAT_TAGS` / `_INT_TAGS`.

---

## Status

- Tested against PDS4 OHRC raw (`*_nrp_*`) and calibrated (`*_ncp_*`)
  products from the ISDA information model v1.11.0.0.
- DuckDB-based SQL is optional; the typed CLI and pandas API need only
  pandas + pyarrow.

---

## Install

Requires Python 3.10+.

From the repo root:

```bash
pip install -r catalog/requirements.txt
```

Optional SQL backend (enables `python -m catalog sql ...`):

```bash
pip install duckdb
```

Optional polygon-geometry recipe in `examples/queries.py`:

```bash
pip install shapely
```

The package runs as a module — no `pip install` of the project itself is
required as long as you invoke `python -m catalog ...` from the repo
root (or have the repo root on `PYTHONPATH`).

---

## Quickstart

Drop your PRADAN downloads under `data/` at the repo root. The expected
layout (PRADAN's default) is:

```
data/
  ch2_ohr_nrp_<timestamp>_d_img_d32/
    data/raw/<YYYYMMDD>/<product_id>.xml
    data/raw/<YYYYMMDD>/<product_id>.img
    browse/...
    miscellaneous/...
  ch2_ohr_ncp_<timestamp>_d_img_d32/
    data/calibrated/<YYYYMMDD>/<product_id>.xml
    data/calibrated/<YYYYMMDD>/<product_id>.img
    browse/...
    geometry/...
```

Build the index:

```bash
python -m catalog build data
```

Summarize what got indexed:

```bash
python -m catalog summary
```

Sample output on three products:

```json
{
  "n_products": 3,
  "by_role": {"raw": 2, "calibrated": 1},
  "by_bits_selection": {"mid": 3},
  "by_tdi_stages": {"TDI256": 3},
  "by_area": {"South Pole": 3},
  "solar_incidence_range": [83.73, 84.89],
  "altitude_range": [108.6, 109.14],
  "lat_range": [-84.98, -84.11],
  "lon_range": [309.96, 312.35]
}
```

---

## CLI reference

All subcommands are dispatched from `python -m catalog <subcommand>`.

### `build [DATA_DIR]`

Scans `DATA_DIR` (default `data`) recursively, parses every product XML,
and writes `catalog/_index/index.parquet` and `catalog/_index/index.jsonl`.

| Flag | Meaning |
| --- | --- |
| `--all-artifacts` | Include browse/geometry/misc XMLs (default: only the data-product XMLs that have a `.img` sibling) |

### `summary`

Prints counts and ranges for the most useful filter dimensions (role,
bits_selection, tdi_stages, area, solar_incidence range, altitude range,
lat/lon coverage, date range).

### `show [--cols ...] [--limit N]`

Prints rows from the index. Default columns are the ones you usually want
at a glance (product_id, role, bits_selection, tdi_stages, solar_incidence,
sun_elevation, spacecraft_altitude, pixel_resolution, area, footprint bbox,
start_date_time).

### `query [filters...] [--sort COL] [--desc] [--limit N] [--cols ...] [--out PATH.csv]`

Typed filters, AND'd together. All are optional.

| Flag | Type | Meaning |
| --- | --- | --- |
| `--bits` | str(s) | `lsb`, `mid`, `msb` |
| `--tdi` | str(s) | e.g. `TDI256` |
| `--area` | str(s) | e.g. `South Pole` |
| `--processing-level` | `Raw` / `Calibrated` | from the PDS4 label |
| `--role` | `raw` / `calibrated` | normalized version of processing_level |
| `--min-sun-incidence` / `--max-sun-incidence` | float (deg) | inclusive bounds on `solar_incidence` |
| `--min-altitude` / `--max-altitude` | float (km) | inclusive bounds on `spacecraft_altitude` |
| `--lat LO HI` | floats | products whose footprint bbox overlaps [LO,HI] in lat |
| `--lon LO HI` | floats | same for lon |
| `--bbox MINLAT MINLON MAXLAT MAXLON` | floats | bbox intersection |
| `--point LAT LON` | floats | point falls inside footprint bbox |
| `--start-after` / `--start-before` | ISO datetime | `start_date_time` window |
| `--sort` | column | sort the result |
| `--desc` | flag | descending sort |
| `--limit` | int | first N rows |
| `--cols` | comma-list | columns to print |
| `--out` | path | write the result to CSV |

Example:

```bash
python -m catalog query \
  --bits lsb \
  --role raw \
  --min-sun-incidence 88 \
  --bbox -90 -180 -84 180 \
  --sort solar_incidence --desc \
  --cols product_id,solar_incidence,bbox_min_lat,xml_path \
  --out psr_candidates.csv
```

### `sql "<SQL>"`

Arbitrary SQL via DuckDB on the Parquet index. The index is exposed as a
view named `cat`. Requires `pip install duckdb`.

```bash
python -m catalog sql "
  SELECT product_id, solar_incidence, bbox_min_lat
  FROM cat
  WHERE bits_selection = 'lsb'
    AND solar_incidence > 88
    AND bbox_min_lat < -85
  ORDER BY solar_incidence DESC"
```

---

## Python API

```python
import catalog

# Build / refresh
catalog.build_index("data")          # walks data/, writes Parquet+JSONL

# Load the on-disk index
df = catalog.load()                  # pandas DataFrame

# Typed query
hits = catalog.query(
    bits_selection="lsb",
    role="raw",
    min_solar_incidence=88.0,
    bbox=(-90, -180, -84, 180),      # min_lat, min_lon, max_lat, max_lon
    sort_by="solar_incidence",
    ascending=False,
    limit=50,
)

# Aggregate counts and ranges
catalog.summary()                    # dict

# Raw scanning (without persisting)
from catalog import scan_dir, parse_xml
for row in scan_dir("data"):
    print(row["product_id"], row["bits_selection"])
```

`catalog.query()` parameters are keyword-only and optional. Pass `df=` to
filter an already-loaded DataFrame instead of re-reading the Parquet.

---

## Schema

Every leaf element under `isda:Product_Parameters` and the corner-coordinate
blocks is captured as a column, snake_cased to its tag name. Numeric fields
are cast to `float` / `int`; the `unit=` attribute is stored as a sibling
column `<name>_unit`. The full schema today on OHRC is 102 columns; the
groups below are the ones you'll typically filter on.

### Identity

| Column | Meaning |
| --- | --- |
| `product_id` | XML stem, e.g. `ch2_ohr_nrp_20211223T0019163816_d_img_d32` |
| `role` | `raw` or `calibrated` (normalized from `processing_level`) |
| `artifact` | `data` / `browse` / `geometry` / `miscellaneous` |
| `processing_level` | raw label string |
| `area` | e.g. `South Pole` |
| `projection` | e.g. `Polar stereographic` |
| `logical_identifier` | PDS4 LIDVID |
| `job_id`, `level0_dir_name` | ISRO pipeline identifiers |

### Time

| Column | Meaning |
| --- | --- |
| `start_date_time`, `stop_date_time` | UTC, parsed to `datetime64[ns, UTC]` |
| `creation_date_time` | label creation time |

### Imaging parameters

| Column | Unit | Meaning |
| --- | --- | --- |
| `bits_selection` | enum | `lsb` / `mid` / `msb` — which 8 bits of the 10-bit detector were kept |
| `tdi_stages` | enum | `TDI256` etc. |
| `line_exposure_duration` | µs (XML labels it `ms`; see note below) | per-line integration time |
| `spacecraft_altitude` | km | |
| `pixel_resolution` | m/pixel | nominal ground sampling |
| `detector_pixel_width` | µm | |
| `focal_length` | mm | |
| `imaging_orbit_number`, `dumping_orbit_number` | int | |
| `roll`, `pitch`, `yaw` | deg | spacecraft attitude |
| `sun_azimuth`, `sun_elevation`, `solar_incidence` | deg | illumination geometry |
| `orbit_limb_direction` | `Ascending` / `Descending` | calibrated only |
| `spacecraft_yaw_direction` | string | calibrated only |
| `reference_data_used` | string | calibrated only |

> Unit note: ISRO's labels mark `line_exposure_duration unit="ms"` but the
> numeric value is actually microseconds — the per-line integration times
> reconcile with the `start`/`stop` timestamps and line counts only at µs.
> The unit string is preserved verbatim in `line_exposure_duration_unit`
> so you can re-check on future products.

### Identity (derived)

| Column | Meaning |
| --- | --- |
| `obs_id` | `product_id` minus the trailing `_<station>` token. Same observation downlinked through two stations shares an `obs_id`; `--unique-obs` collapses to one row per `(obs_id, role)`. |
| `station_id` | ground station that received the downlink (`d18`, `d32`, `g26`, `gds`, `hw1`, `n18`). Same scene from different stations is byte-identical — verified across 11 dual-station groups. |

### Geometry

Four corner lat/lon, in degrees, both the system-level and the refined
post-orthorectification values (the refined ones appear in calibrated
products):

- `upper_left_latitude`, `upper_left_longitude`
- `upper_right_latitude`, `upper_right_longitude`
- `lower_left_latitude`, `lower_left_longitude`
- `lower_right_latitude`, `lower_right_longitude`
- `refined_upper_left_latitude`, ... (same suffix list)

Plus a derived footprint bbox for fast spatial filtering:

- `bbox_min_lat`, `bbox_max_lat` — straightforward lat range over the four corners.
- `lon_arc_start`, `lon_arc_end` — longitude *arc* on [0, 360). If
  `start <= end` the arc is `[start, end]`; if `start > end` it wraps the
  antimeridian as `[start, 360) ∪ [0, end]`. Derived via the cyclic-gap
  method so antimeridian crossers (5 in the current 312-product corpus)
  and wide-near-pole strips are handled correctly.
- `lon_wraps` — convenience boolean: `start > end`.
- `polar_degenerate` — true when `bbox_max_lat < -88` and the lon arc
  spans more than 90°. Longitude becomes geometrically degenerate near
  the pole, so lon-bbox queries on these strips are unreliable (lat
  filters remain meaningful). 30 such strips in the current corpus;
  drop them from spatial queries with `--exclude-polar-degenerate`.

### File / array info

| Column | Meaning |
| --- | --- |
| `file_name` | name of the `.img` sibling |
| `file_size` | bytes (from XML) |
| `img_size_actual` | bytes (from `stat()` — catches truncated downloads) |
| `is_truncated` | `True` when `.img` is missing or its size disagrees with the XML `file_size`. 0 truncations across the current 312-product corpus; drop them from queries with `--exclude-truncated`. |
| `md5_checksum` | for integrity checks |
| `line_count`, `sample_count` | image dimensions for memmap loading |
| `xml_path_rel`, `img_path_rel`, `product_dir_rel`, `peer_xml_path_rel` | paths stored *relative* to `data_root` (recorded in `catalog/_index/_meta.json` at build time). |
| `xml_path`, `img_path`, `product_dir`, `peer_xml_path` | absolute paths reconstructed lazily on `load()` using `data_root`. The data root is resolved in priority order: explicit `data_root=` argument, then `SELENE_DATA_ROOT` env var, then `_meta.json`. Re-mount the SSD anywhere and point `SELENE_DATA_ROOT` at the new mount — no rebuild needed. |

### Unit columns

Every numeric field with a unit attribute also has a `<name>_unit` column
(e.g. `spacecraft_altitude_unit = "km"`). Useful as a defensive check
when mixing products that may declare different units.

---

## Spatial queries

The lat axis is straightforward (`bbox_min_lat`, `bbox_max_lat`). The lon
axis is an *arc* on [0, 360), not an interval — see the `lon_arc_*`
columns under Schema. Spatial predicates handle wrap-around correctly:

- `--lat -90 -84` keeps products whose footprint **overlaps** [-90, -84].
- `--bbox MINLAT MINLON MAXLAT MAXLON` — lat range overlaps the query
  lat range AND the strip's lon arc overlaps the query lon range
  (treated as a simple arc, possibly wrapping if `MINLON > MAXLON`).
- `--point LAT LON` keeps products whose lat range contains LAT AND
  whose lon arc contains LON. Near-pole strips are also kept when LON
  falls outside the geometrically-degenerate arc (drop them explicitly
  with `--exclude-polar-degenerate`).

Bbox queries remain conservative: a product whose true ortho-quadrilateral
does not actually contain the query point may still pass the bbox test.
For exact polygon-in-polygon intersection, use the four corner columns
with DuckDB's spatial extension or shapely:

```python
from shapely.geometry import Polygon, Point
import catalog
df = catalog.load()
def poly(row):
    return Polygon([
        (row["upper_left_longitude"], row["upper_left_latitude"]),
        (row["upper_right_longitude"], row["upper_right_latitude"]),
        (row["lower_right_longitude"], row["lower_right_latitude"]),
        (row["lower_left_longitude"], row["lower_left_latitude"]),
    ])
target = Point(311.0, -84.5)
hits = df[df.apply(lambda r: poly(r).contains(target), axis=1)]
```

`examples/queries.py` includes a runnable version of this recipe along
with several others.

---

## Common recipes

`examples/queries.py` (runnable):

```python
import catalog

# 1. True-PSR candidates (sun below the local horizon)
psr = catalog.query(min_solar_incidence=90.0)

# 2. lsb-encoded raw products only -- usable for low-signal denoising work
lsb_raw = catalog.query(bits_selection="lsb", role="raw")

# 3. Products covering Cabeus (LCROSS impact ~ 84.675°S, 311.275°E)
cabeus = catalog.query(point=(-84.675, 311.275))

# 4. One row per observation (collapse dual-station downlinks)
unique = catalog.query(unique_obs=True)

# 5. Raw + calibrated counterparts via obs_id
df = catalog.load()
both = df[df["obs_id"] == "ch2_ohr_ncp_20211223T0019163816_d_img"]

# 6. South Pole observations from a date window
window = catalog.query(
    area="South Pole",
    start_after="2021-12-01",
    start_before="2022-01-01",
)

# 7. Drop truncated / polar-degenerate strips from a query
clean = catalog.query(
    min_solar_incidence=90,
    exclude_truncated=True,
    exclude_polar_degenerate=True,
)
```

---

## How it works

1. **Scanner** (`catalog/scanner.py`) walks the data root, finds every
   product-data `*.xml` (artifact = `data` under the product directory),
   parses the XML into a flat dict, and tags each row with role / artifact
   / paths.
2. **Index builder** (`catalog/index.py:build_index`) collects rows,
   orders columns into a stable schema, derives footprint bbox from the
   four corner coordinates, parses datetimes to UTC, and writes Parquet
   + JSONL.
3. **Query layer** (`catalog/query.py:query`) provides typed pandas-style
   filters; `catalog sql` dispatches to DuckDB on the Parquet file for
   arbitrary SQL with no schema commitment.

The scanner does not load any image data — only XML. Building the index
over hundreds of products is seconds-scale.

---

## Extending

- **New PDS4 fields**: nothing to do. Any leaf in the `isda:` namespace
  shows up automatically as a column. Re-run `build`.
- **New numeric coercions**: add the tag name to `_INT_TAGS` or
  `_FLOAT_TAGS` in `scanner.py` to get typed columns instead of strings.
- **New typed CLI flag**: extend `query.py:query()` and the argparse block
  in `__main__.py:_build_parser`.
- **A different mission (TMC-2, IIRS, ...)**: point `build` at that data
  directory. The walker is generic; only the typed filters in
  `catalog.query()` assume OHRC-style field names.

---

## Limitations

- Footprint queries use the axis-aligned bbox of the four corners. For
  true polygon intersection use the explicit corner columns with shapely
  or DuckDB-spatial (see Spatial queries section).
- The exposure-duration unit attribute in ISRO's labels appears to be
  mislabeled (`ms`) for OHRC; the actual value is microseconds. The
  catalog preserves both the value and the label so you can audit
  per-product.
- The `peer_xml_path` link only resolves nrp <-> ncp counterparts that
  live in the **same directory** as the source XML. For PRADAN's default
  layout (raw and calibrated products downloaded as separate zips), the
  peer is in a sibling product directory; the column will be `None` and
  you should match on the timestamp portion of `product_id` instead (see
  recipe 4).

---

## Layout

```
catalog/
  __init__.py        # public API re-exports
  scanner.py         # XML -> flat dict (schema-tolerant)
  index.py           # build_index(), load() -- Parquet + JSONL
  query.py           # query(), summary() -- typed pandas filters
  __main__.py        # CLI: build / summary / show / query / sql
  requirements.txt
  examples/
    queries.py       # runnable recipes
  _index/            # generated on `build`; gitignore-friendly
    index.parquet
    index.jsonl
```

---

## Renaming for PyPI

The import name `catalog` is generic and may collide with other packages
in larger Python environments. If publishing standalone, consider
renaming the package directory to `ohrc_catalog` (or similar) and
updating the imports in `__init__.py` and `__main__.py`. The on-disk
Parquet stays the same.
