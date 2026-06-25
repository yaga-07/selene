"""Time ``OHRCReferenceDataset.__getitem__`` against the real corpus.

Closes the §2.5 gate item: median per-item wall time must be < 50 ms
(manifest → memmap-slice → inject_noise → torch tensor) on a warm
cache. Reports min / median / p95 / max so cold-start outliers don't
hide a real regression.

Run (SSD must be mounted):
  .venv/bin/python -m analysis.benchmark_dataset [--split train] \
      [--n-warmup 32] [--n-samples 200] [--seed 0]

Writes ``analysis/_outputs/benchmark_dataset.txt`` for the audit log.
"""

from __future__ import annotations

import argparse
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from training_data.dataset import OHRCReferenceDataset

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = REPO_ROOT / "analysis" / "_outputs"
OUTPUT_FILE = OUTPUT_DIR / "benchmark_dataset.txt"

GATE_MEDIAN_MS = 50.0


def benchmark(split: str, n_warmup: int, n_samples: int, seed: int) -> dict:
    ds = OHRCReferenceDataset(split=split, seed=seed)
    n = len(ds)
    print(f"benchmark {ds.manifest_path.name} split={split}: {n} patches")

    rng = np.random.default_rng(seed)
    idxs = rng.integers(0, n, size=n_warmup + n_samples)

    print(f"  warming up: {n_warmup} samples")
    for i in idxs[:n_warmup]:
        ds[int(i)]

    times_ms: list[float] = []
    print(f"  timing {n_samples} samples")
    for i in idxs[n_warmup:]:
        t0 = time.perf_counter()
        ds[int(i)]
        times_ms.append((time.perf_counter() - t0) * 1e3)

    times_ms.sort()
    p = lambda q: times_ms[int(q * len(times_ms))]  # noqa: E731
    return {
        "split": split,
        "n_samples": n_samples,
        "n_warmup": n_warmup,
        "seed": seed,
        "min_ms": min(times_ms),
        "median_ms": statistics.median(times_ms),
        "p95_ms": p(0.95),
        "max_ms": max(times_ms),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--split", choices=("train", "val"), default="train")
    parser.add_argument("--n-warmup", type=int, default=32)
    parser.add_argument("--n-samples", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    stats = benchmark(args.split, args.n_warmup, args.n_samples, args.seed)
    pass_str = "PASS" if stats["median_ms"] < GATE_MEDIAN_MS else "FAIL"
    line = (
        f"{datetime.now(timezone.utc).isoformat(timespec='seconds')}  "
        f"split={stats['split']:5s}  "
        f"n={stats['n_samples']:>4d}  warmup={stats['n_warmup']:>3d}  "
        f"seed={stats['seed']:>3d}  "
        f"min={stats['min_ms']:>6.2f}ms  "
        f"median={stats['median_ms']:>6.2f}ms  "
        f"p95={stats['p95_ms']:>6.2f}ms  "
        f"max={stats['max_ms']:>7.2f}ms  "
        f"gate(<{GATE_MEDIAN_MS:.0f}ms)={pass_str}"
    )
    print(line)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with OUTPUT_FILE.open("a") as f:
        f.write(line + "\n")
    print(f"appended to {OUTPUT_FILE.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
