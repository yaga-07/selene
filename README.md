# SELENE

An exploratory research repository on **low-signal restoration for
Chandrayaan-2 OHRC imagery** at the lunar south pole.

This is an early-stage project and reiteration of MSc work. The eventual scope is still being
shaped by what the data actually supports. Whatever the final target
ends up being, the work here is the data engineering + audit + noise
modelling foundation that any version of the project will need.

---

## What I want to figure out

Lunar south-pole and low lit area imagery from OHRC contains terrain that is very
dimly lit — lit only by indirect sources like earthshine and
multiply-scattered sunlight from crater walls. Denoising these images
isn't a generic computer-vision problem: pixel values are at the
detector's noise floor, the noise itself is signal-dependent (TDI CCD
physics), and anything a denoiser invents that isn't really there is
worse than a noisy image because downstream science will read into it.

The working question is roughly: **can we build a useful denoiser
whose noise model and training data are grounded in the actual
instrument physics — and that quantifies its own uncertainty
honestly — rather than treating this as off-the-shelf image
restoration?**

Whether the practical scope settles on a specific crater, a wider
polar region, a benchmark dataset, or something else is open. The
audit findings in [`FINDINGS.md`](FINDINGS.md) will help drive that
decision.

---

## What's in this repo

A set of small, focused components rather than a single ML codebase.
Each is independently runnable and testable.

```
selene/
├── catalog/         # Metadata index over PRADAN OHRC products (Python pkg + CLI)
│   ├── geometry.py  # Arc-based longitude bbox (handles antimeridian + near-pole)
│   ├── scanner.py   # PDS4 XML parser
│   ├── index.py     # Parquet + JSONL persistence, portable paths
│   ├── query.py     # Typed pandas queries + dual-station dedup
│   ├── __main__.py  # `python -m catalog <cmd>` CLI
│   └── README.md    # Schema, CLI, spatial-query semantics
│
├── analysis/        # Diagnostic scripts and the figures they produce
│   ├── diagnose_bits_selection.py        # Dark-band DN histogram + USABLE/CRUSHED verdict
│   ├── diagnose_bbox.py                  # Longitude-span distribution + antimeridian check
│   ├── diagnose_truncation_and_dedup.py  # file_size sanity + dual-station md5 check
│   ├── check_msb_psr_usability.py        # Apply the bits diagnostic to MSB PSR strips
│   └── *.png                             # Generated diagnostic plots (referenced in FINDINGS.md)
│
├── tests/           # pytest suite (currently: geometry primitives)
│   └── test_geometry.py
│
└── FINDINGS.md      # Empirical results from the data audit
```

---

## What's been done so far

- **Catalog data engineering layer.** 312 OHRC raw products from
  PRADAN indexed into a queryable Parquet store with typed queries
  and a CLI. Spatial queries use an arc-based longitude
  representation (cyclic-gap method) that handles antimeridian
  crossers and near-pole degeneracy correctly. Dual-station
  downlinks (same observation received by different ground
  stations) are detected and dedup-able. 38 passing unit tests for
  the geometry primitives.
- **Data audit.** All 312 products inspected: integrity (0
  truncations), encoding distribution, illumination geometry, and a
  pixel-level diagnostic on a sample of strips. Results in
  [`FINDINGS.md`](FINDINGS.md).
- **Reproducible diagnostic scripts** for everything above, kept
  alongside the figures they produce.

Next: noise-model estimation (Photon Transfer Curves) on the
usable sunlit polar strips.

---

## Running things

```bash
# Set up environment
python -m venv .venv
.venv/bin/pip install -r requirements.txt

# Run tests
.venv/bin/python -m pytest tests/

# Build the catalog over your local OHRC data root
.venv/bin/python -m catalog build /path/to/ohrc-data
.venv/bin/python -m catalog summary

# Example: low-signal raw strips, PSR illumination, one row per observation
.venv/bin/python -m catalog query \
    --role raw \
    --bits lsb mid \
    --min-sun-incidence 90 \
    --unique-obs
```

The catalog CLI's `--help` covers the full set of flags. See
[`catalog/README.md`](catalog/README.md) for the schema and
spatial-query semantics.

---

## Tech

- Python 3.11, plain `venv` (no conda).
- `pandas` + `pyarrow` for the index, `duckdb` for optional SQL access.
- `numpy` + `scipy` for analysis.
- `pytest` for the test suite.
- `matplotlib` for diagnostic plots.

Anything heavier (deep-learning frameworks, ray tracing) is deferred
until the relevant phase needs it.

---

## Status / non-goals

Active research repo, not a product. Code is written for
correctness and clarity; APIs may shift as the project takes shape.
Generalisation across instruments is not a goal of this first phase
— OHRC's TDI physics is the whole point.

## License

MIT — see [LICENSE](LICENSE).
