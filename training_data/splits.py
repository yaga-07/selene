"""Generate ``training_data/splits.json`` — the train/val/test lock.

The 6 test product IDs in ``LOCKED_TEST_STRIPS`` are curated by hand and
audited in code. Train/val partition is derived deterministically from
``catalog \\ test`` using ``RANDOM_SEED`` and stratified by
``STRATIFY_BY``. Re-running this script after a catalog rebuild
refreshes train/val; the test IDs stay pinned.

Run:  ``.venv/bin/python -m training_data.splits``
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import catalog

SCHEMA_VERSION = 1
RANDOM_SEED = 42
VAL_FRACTION = 0.15
STRATIFY_BY = "bits_selection"
ELIGIBLE_POOL_FILTER = "area == 'Equatorial'"

LOCKED_TEST_STRIPS: dict[str, list[str]] = {
    # Dagar 2024 sunlit-rim Cabeus passes; reserved for the cross-strip
    # overlap-consistency validator and the paper's qualitative figure.
    "cabeus_overlap_pair": [
        "ch2_ohr_nrp_20211222T2023166276_d_img_d32",
        "ch2_ohr_nrp_20211223T0019163816_d_img_d32",
    ],
    "polar_held_out": [
        # Cabeus-band lat (-85), TDI64, sol_inc 86.5°.
        "ch2_ohr_nrp_20241118T1613209337_d_img_d18",
        # Deep-pole (-89.5), TDI64, sol_inc 89.21° — toughest low-signal
        # scene in the catalog.
        "ch2_ohr_nrp_20250228T0429255172_d_img_d18",
    ],
    "equatorial_held_out": [
        # Brightest-signal MSB TDI64 eq strip (sol_inc 75.94°, lat -3°).
        # One of two products tied at 75.937° on the same orbit ~0.5s
        # apart; stable-sort by row order picks this one.
        "ch2_ohr_nrp_20210405T1606537227_d_img_d18",
        # Median sol_inc within the eligible MSB-TDI64 equatorial pool
        # (sol_inc 82.73°, lat -0.04°).
        "ch2_ohr_nrp_20240330T0035085365_d_img_d18",
    ],
}

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = REPO_ROOT / "training_data" / "splits.json"


def _git_sha() -> str:
    return subprocess.check_output(
        ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
        text=True,
    ).strip()


def _git_dirty() -> bool:
    out = subprocess.check_output(
        ["git", "-C", str(REPO_ROOT), "status", "--porcelain"],
        text=True,
    )
    return bool(out.strip())


def _catalog_index_sha256() -> str:
    p = REPO_ROOT / "catalog" / "_index" / "index.parquet"
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _stratified_val_pick(
    train_pool: pd.DataFrame,
    seed: int,
    val_fraction: float,
    stratify_by: str,
) -> tuple[list[str], list[str]]:
    rng = np.random.default_rng(seed)
    val_ids: list[str] = []
    for _, group in train_pool.groupby(stratify_by, sort=True):
        n = len(group)
        if n < 2:
            continue
        n_val = max(1, round(n * val_fraction))
        ordered = group.sort_values("product_id").reset_index(drop=True)
        picks = rng.choice(n, size=n_val, replace=False)
        val_ids.extend(ordered.iloc[picks]["product_id"].tolist())
    val_set = set(val_ids)
    train_ids = sorted(p for p in train_pool["product_id"] if p not in val_set)
    return train_ids, sorted(val_ids)


def build_splits() -> dict:
    df = catalog.load()
    test_ids_flat = [pid for group in LOCKED_TEST_STRIPS.values() for pid in group]
    missing = [pid for pid in test_ids_flat if (df["product_id"] == pid).sum() == 0]
    if missing:
        raise SystemExit(f"locked test IDs missing from catalog: {missing}")

    eligible = df.query(ELIGIBLE_POOL_FILTER)
    train_pool = eligible[~eligible["product_id"].isin(test_ids_flat)].copy()

    train_ids, val_ids = _stratified_val_pick(
        train_pool, RANDOM_SEED, VAL_FRACTION, STRATIFY_BY,
    )

    return {
        "schema_version": SCHEMA_VERSION,
        "generation_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_sha": _git_sha(),
        "git_dirty": _git_dirty(),
        "catalog_index_sha256": _catalog_index_sha256(),
        "params": {
            "random_seed": RANDOM_SEED,
            "val_fraction": VAL_FRACTION,
            "stratify_by": STRATIFY_BY,
            "eligible_pool_filter": ELIGIBLE_POOL_FILTER,
            "test_lock_source": "training_data/splits.py:LOCKED_TEST_STRIPS",
        },
        "test": LOCKED_TEST_STRIPS,
        "train": train_ids,
        "val": val_ids,
    }


def main() -> None:
    out = build_splits()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w") as f:
        json.dump(out, f, indent=2)
        f.write("\n")
    n_test = sum(len(v) for v in LOCKED_TEST_STRIPS.values())
    print(f"wrote {OUTPUT_PATH.relative_to(REPO_ROOT)}")
    print(f"  test:  {n_test} strips across {len(LOCKED_TEST_STRIPS)} groups")
    print(f"  train: {len(out['train'])} strips")
    print(f"  val:   {len(out['val'])} strips")


if __name__ == "__main__":
    main()
