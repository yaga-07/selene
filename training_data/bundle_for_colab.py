"""Bake `reference_manifest.parquet` patches into a single uint8 `.npy`
for ephemeral-cloud (Colab / Kaggle) training, where 50 GB of source
strips can't practically be uploaded. The bundle's row i is the
manifest's row i — preserving order — so a cloud Dataset that opens
the bundle with ``np.load(path, mmap_mode="r")`` is sample-for-sample
equivalent to the manifest-driven Dataset that slices on the host.

Run:
  build:   ``.venv/bin/python -m training_data.bundle_for_colab build [--out PATH]``
  verify:  ``.venv/bin/python -m training_data.bundle_for_colab verify [--k 64] [--seed 0]``

`build` writes:
  - ``training_data/patches.npy``       — uint8, shape (N, 256, 256)
  - ``training_data/patches.meta.json`` — provenance sidecar

`verify` re-slices K random manifest rows from the source memmaps and
asserts byte-equality with the corresponding rows of the bundle. This
closes the §2.5 / §2.6 gate item *(bundle ↔ manifest equivalence)*.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

import catalog
from training_data.curation import PATCH_SIZE

SCHEMA_VERSION = 1

REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = REPO_ROOT / "training_data" / "reference_manifest.parquet"
DEFAULT_OUT = REPO_ROOT / "training_data" / "patches.npy"
DEFAULT_META = REPO_ROOT / "training_data" / "patches.meta.json"


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


def _load_manifest() -> tuple["pd.DataFrame", dict]:  # noqa: F821
    table = pq.read_table(MANIFEST_PATH)
    df = table.to_pandas()
    raw = table.schema.metadata
    if not raw or b"selene_meta" not in raw:
        raise SystemExit(f"{MANIFEST_PATH} has no selene_meta — rebuild manifest first")
    manifest_meta = json.loads(raw[b"selene_meta"])
    return df, manifest_meta


def _strip_meta() -> dict[str, dict]:
    cat = catalog.load()
    rows = cat.set_index("product_id")[["img_path", "line_count", "sample_count"]]
    return {
        pid: {
            "img_path": Path(rows.at[pid, "img_path"]),
            "lines": int(rows.at[pid, "line_count"]),
            "samples": int(rows.at[pid, "sample_count"]),
        }
        for pid in rows.index
    }


def build(out_path: Path, meta_path: Path) -> None:
    df, manifest_meta = _load_manifest()
    n = len(df)
    if n == 0:
        raise SystemExit("manifest is empty")

    strips = _strip_meta()
    missing = [pid for pid in df["product_id"].unique() if pid not in strips]
    if missing:
        raise SystemExit(
            f"{len(missing)} product_ids missing from catalog (e.g. {missing[:3]}) — "
            f"rebuild the catalog over the current data root"
        )
    unreadable = [
        pid for pid in df["product_id"].unique()
        if not strips[pid]["img_path"].exists()
    ]
    if unreadable:
        raise SystemExit(
            f"{len(unreadable)} source strips not readable at recorded path "
            f"(e.g. {strips[unreadable[0]]['img_path']}) — is the SSD mounted?"
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    size_gb = n * PATCH_SIZE * PATCH_SIZE / 1024 ** 3
    print(f"baking {n} patches ({size_gb:.2f} GB uint8) → {out_path}")
    bundle = np.lib.format.open_memmap(
        out_path, mode="w+", dtype=np.uint8,
        shape=(n, PATCH_SIZE, PATCH_SIZE),
    )

    df_idx = df.reset_index().rename(columns={"index": "manifest_idx"})
    t0 = time.time()
    n_written = 0
    by_strip = list(df_idx.groupby("product_id", sort=False))
    n_strips = len(by_strip)
    for i, (pid, rows) in enumerate(by_strip, 1):
        meta = strips[pid]
        arr = np.memmap(
            meta["img_path"], dtype=np.uint8, mode="r",
            shape=(meta["lines"], meta["samples"]),
        )
        for r0, c0, mi in zip(
            rows["row0"].to_numpy(),
            rows["col0"].to_numpy(),
            rows["manifest_idx"].to_numpy(),
        ):
            bundle[int(mi)] = arr[int(r0):int(r0) + PATCH_SIZE,
                                  int(c0):int(c0) + PATCH_SIZE]
            n_written += 1
        del arr
        if i % 5 == 0 or i == n_strips:
            dt = time.time() - t0
            rate = n_written / dt if dt else 0.0
            remain = (n - n_written) / rate if rate else 0.0
            print(f"  [{i:>3}/{n_strips}] {pid}: +{len(rows):>5d}, "
                  f"total {n_written:>6d}/{n} ({n_written / n:>5.1%}), "
                  f"{rate:>6.0f} patches/s, ETA {remain:>5.0f}s")
    bundle.flush()
    elapsed = time.time() - t0
    del bundle
    print(f"bundle flushed; {n} patches in {elapsed:.1f}s "
          f"({n / elapsed:.0f} patches/s)")

    out_meta = {
        "schema_version": SCHEMA_VERSION,
        "generation_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_sha": _git_sha(),
        "git_dirty": _git_dirty(),
        "manifest_path": str(MANIFEST_PATH.relative_to(REPO_ROOT)),
        "manifest_parquet_sha256": _file_sha256(MANIFEST_PATH),
        "manifest_n_patches": manifest_meta["n_patches"],
        "manifest_generation_utc": manifest_meta["generation_utc"],
        "manifest_splits_sha256": manifest_meta["splits_sha256"],
        "manifest_git_sha": manifest_meta["git_sha"],
        "output_shape": [n, PATCH_SIZE, PATCH_SIZE],
        "output_dtype": "uint8",
        "row_order": "bundle[i] is reference_manifest.parquet row i",
        "elapsed_seconds": round(elapsed, 1),
    }
    meta_path.write_text(json.dumps(out_meta, indent=2) + "\n")
    print(f"wrote {meta_path.relative_to(REPO_ROOT)}")


def verify(out_path: Path, k: int, seed: int) -> None:
    if not out_path.exists():
        raise SystemExit(f"{out_path} does not exist — run `build` first")
    df, _ = _load_manifest()
    bundle = np.load(out_path, mmap_mode="r")
    expected_shape = (len(df), PATCH_SIZE, PATCH_SIZE)
    if bundle.shape != expected_shape:
        raise SystemExit(
            f"bundle shape {bundle.shape} != expected {expected_shape}"
        )
    if bundle.dtype != np.uint8:
        raise SystemExit(f"bundle dtype {bundle.dtype} != uint8")

    strips = _strip_meta()
    rng = np.random.default_rng(seed)
    k = min(k, len(df))
    idxs = np.sort(rng.choice(len(df), size=k, replace=False))

    arr_cache: dict[str, np.memmap] = {}
    n_pass = 0
    for i in idxs:
        row = df.iloc[int(i)]
        pid = row["product_id"]
        if pid not in arr_cache:
            meta = strips[pid]
            arr_cache[pid] = np.memmap(
                meta["img_path"], dtype=np.uint8, mode="r",
                shape=(meta["lines"], meta["samples"]),
            )
        arr = arr_cache[pid]
        r0, c0 = int(row["row0"]), int(row["col0"])
        expected = np.asarray(arr[r0:r0 + PATCH_SIZE, c0:c0 + PATCH_SIZE])
        actual = np.asarray(bundle[int(i)])
        if not np.array_equal(expected, actual):
            diff = (expected.astype(np.int16) - actual.astype(np.int16))
            raise SystemExit(
                f"MISMATCH at manifest row {i} (pid={pid}, row0={r0}, col0={c0}): "
                f"max|delta|={int(np.abs(diff).max())}, "
                f"n_diff_pixels={(diff != 0).sum()}"
            )
        n_pass += 1
    print(f"VERIFY pass: {n_pass}/{k} random rows match "
          f"(seed={seed}, bundle={out_path.name})")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)

    pb = sub.add_parser("build", help="bake patches.npy from the manifest")
    pb.add_argument("--out", type=Path, default=DEFAULT_OUT)
    pb.add_argument("--meta", type=Path, default=DEFAULT_META)

    pv = sub.add_parser("verify", help="re-slice K random rows and assert equality")
    pv.add_argument("--out", type=Path, default=DEFAULT_OUT)
    pv.add_argument("--k", type=int, default=64)
    pv.add_argument("--seed", type=int, default=0)

    args = p.parse_args()
    if args.cmd == "build":
        build(args.out, args.meta)
    elif args.cmd == "verify":
        verify(args.out, args.k, args.seed)


if __name__ == "__main__":
    main()
