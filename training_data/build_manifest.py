"""Scan train+val strips at non-overlapping 256×256 patches, apply the
§2.1 curation rules, and write ``training_data/reference_manifest.parquet``.

The manifest is a flat parquet table with one row per accepted patch.
Patches are tagged with their split (``train`` / ``val``), source
mode/TDI, and per-patch curation stats. The forward-noise injector
reads the source strip and corner coordinates at training time and
injects on the fly — patches themselves are *not* materialised here.

Run:  ``.venv/bin/python -m training_data.build_manifest``

Provenance (git_sha, splits sha, curation params) is embedded as
key-value metadata on the parquet schema.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

import catalog
from training_data import curation
from training_data.curation import (
    PATCH_SIZE,
    check_patch,
    compute_patch_stats,
)

SCHEMA_VERSION = 1
STRIDE = PATCH_SIZE  # non-overlapping

REPO_ROOT = Path(__file__).resolve().parents[1]
SPLITS_PATH = REPO_ROOT / "training_data" / "splits.json"
OUTPUT_PATH = REPO_ROOT / "training_data" / "reference_manifest.parquet"


def _git_sha() -> str:
    return subprocess.check_output(
        ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"], text=True,
    ).strip()


def _git_dirty() -> bool:
    out = subprocess.check_output(
        ["git", "-C", str(REPO_ROOT), "status", "--porcelain"], text=True,
    )
    return bool(out.strip())


def _file_sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _scan_strip(
    product_id: str, split: str, row: pd.Series,
) -> tuple[list[dict], int, int]:
    """Walk one strip, return (accepted_rows, n_candidates, n_accepted)."""
    img_path = Path(row["img_path"])
    lines = int(row["line_count"])
    samples = int(row["sample_count"])
    bits = str(row["bits_selection"])
    tdi = int(str(row["tdi_stages"]).replace("TDI", ""))

    arr = np.memmap(img_path, dtype=np.uint8, mode="r", shape=(lines, samples))

    accepted: list[dict] = []
    n_candidates = 0
    for r0 in range(0, lines - PATCH_SIZE + 1, STRIDE):
        for c0 in range(0, samples - PATCH_SIZE + 1, STRIDE):
            patch = np.asarray(arr[r0:r0 + PATCH_SIZE, c0:c0 + PATCH_SIZE])
            stats = compute_patch_stats(patch)
            n_candidates += 1
            passes, _ = check_patch(stats)
            if not passes:
                continue
            accepted.append({
                "product_id": product_id,
                "split": split,
                "row0": r0,
                "col0": c0,
                "source_bits": bits,
                "source_tdi": tdi,
                **stats.as_dict(),
            })
    return accepted, n_candidates, len(accepted)


def build_manifest() -> tuple[pd.DataFrame, dict]:
    splits = json.loads(SPLITS_PATH.read_text())
    df = catalog.load()

    target_strips: list[tuple[str, str]] = (
        [(pid, "train") for pid in splits["train"]]
        + [(pid, "val") for pid in splits["val"]]
    )

    rows: list[dict] = []
    summary: list[dict] = []
    print(f"scanning {len(target_strips)} strips at stride={STRIDE}, patch={PATCH_SIZE}")
    t0 = time.time()
    for i, (pid, split) in enumerate(target_strips, 1):
        cat_row = df[df["product_id"] == pid]
        if cat_row.empty:
            print(f"  [{i:>2}/{len(target_strips)}] {pid} — NOT IN CATALOG, skipping")
            continue
        cat_row = cat_row.iloc[0]
        ts = time.time()
        accepted, n_cand, n_acc = _scan_strip(pid, split, cat_row)
        dt = time.time() - ts
        rate = n_acc / n_cand if n_cand else 0.0
        print(f"  [{i:>2}/{len(target_strips)}] {pid} ({split:5s}) "
              f"{cat_row['bits_selection']}/TDI{int(str(cat_row['tdi_stages']).replace('TDI',''))} "
              f"→ {n_acc:>5d}/{n_cand:<5d} accepted ({rate:>5.1%})   "
              f"{dt:>5.1f}s")
        rows.extend(accepted)
        summary.append({
            "product_id": pid,
            "split": split,
            "bits": str(cat_row["bits_selection"]),
            "tdi": int(str(cat_row["tdi_stages"]).replace("TDI", "")),
            "candidates": n_cand,
            "accepted": n_acc,
            "accept_rate": rate,
            "scan_seconds": dt,
        })
    elapsed = time.time() - t0
    print(f"done in {elapsed:.1f}s, {len(rows)} total accepted patches")

    manifest_df = pd.DataFrame(rows, columns=[
        "product_id", "split", "row0", "col0",
        "source_bits", "source_tdi",
        "mean_dn", "frac_zero", "frac_sat", "sobel_99",
    ])
    meta = {
        "schema_version": SCHEMA_VERSION,
        "generation_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_sha": _git_sha(),
        "git_dirty": _git_dirty(),
        "splits_sha256": _file_sha256(SPLITS_PATH),
        "stride": STRIDE,
        "curation_params": curation.CURATION_PARAMS,
        "n_strips_scanned": len(summary),
        "n_patches": len(rows),
        "elapsed_seconds": round(elapsed, 1),
        "per_strip_summary": summary,
    }
    return manifest_df, meta


def main() -> None:
    manifest_df, meta = build_manifest()

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pandas(manifest_df, preserve_index=False)
    table = table.replace_schema_metadata({
        b"selene_meta": json.dumps(meta).encode("utf-8"),
    })
    pq.write_table(table, OUTPUT_PATH)

    print()
    print(f"wrote {OUTPUT_PATH.relative_to(REPO_ROOT)}")
    print(f"  {len(manifest_df)} patches across {meta['n_strips_scanned']} strips")
    by_split = manifest_df["split"].value_counts().to_dict()
    print(f"  by split: {by_split}")
    by_mode = (manifest_df.groupby(["source_bits", "source_tdi"]).size().to_dict())
    print(f"  by mode:  {by_mode}")


if __name__ == "__main__":
    main()
