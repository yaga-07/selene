"""Confirm the bundle Dataset path returns byte-identical (clean, noisy)
pairs to the catalog Dataset path on every checked idx.

Runs locally where both paths are available (SSD-mounted ``catalog/_index/``
+ local ``patches.npy``). Picks K random idxs per split, instantiates two
Datasets with the same seed (one bundle-mode, one catalog-mode), and
asserts byte-equality on both tensors and meta scalars.

This is the Colab-readiness check: if equivalence holds locally, the
Colab clone reading the same bundle will produce identical training data
to what was validated on the workstation.

Run:
    .venv/bin/python -m analysis.validate_bundle_dataset \\
        --bundle /Volumes/lazarus/selene-colab/patches.npy --k 32
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from training_data.dataset import OHRCReferenceDataset


def validate_split(split: str, bundle_path: Path, k: int, seed: int) -> None:
    print(f"\n[{split}] instantiating both paths …")
    ds_catalog = OHRCReferenceDataset(split=split, seed=seed)
    ds_bundle = OHRCReferenceDataset(split=split, seed=seed, bundle_path=bundle_path)

    n = len(ds_catalog)
    assert len(ds_bundle) == n, f"len mismatch: catalog={n} bundle={len(ds_bundle)}"

    rng = np.random.default_rng(seed)
    idxs = rng.choice(n, size=min(k, n), replace=False)

    print(f"[{split}] checking {len(idxs)} random idxs (n={n}) …")
    for i, idx in enumerate(idxs, 1):
        idx = int(idx)
        a = ds_catalog[idx]
        b = ds_bundle[idx]
        # Clean must be byte-identical (same source data, no rng involved
        # in reading — only in alpha/sim_mode draws, which are seeded).
        if not np.array_equal(a["clean"].numpy(), b["clean"].numpy()):
            diff = (a["clean"].numpy() - b["clean"].numpy())
            raise SystemExit(
                f"[{split}] CLEAN mismatch at idx={idx}: "
                f"max|Δ|={float(np.abs(diff).max()):.4f}"
            )
        # Noisy depends on rng + clean. Same seed + same clean + same
        # col_offset (we pass col_offset=col0 in dataset.py) → bit-exact.
        if not np.array_equal(a["noisy"].numpy(), b["noisy"].numpy()):
            diff = (a["noisy"].numpy() - b["noisy"].numpy())
            raise SystemExit(
                f"[{split}] NOISY mismatch at idx={idx}: "
                f"max|Δ|={float(np.abs(diff).max()):.4f}"
            )
        # Meta scalars
        for k_ in ("product_id", "row0", "col0", "source_bits",
                   "source_tdi", "sim_bits", "sim_tdi"):
            if a["meta"][k_] != b["meta"][k_]:
                raise SystemExit(
                    f"[{split}] meta[{k_!r}] mismatch at idx={idx}: "
                    f"catalog={a['meta'][k_]!r} bundle={b['meta'][k_]!r}"
                )
        if abs(a["meta"]["alpha"] - b["meta"]["alpha"]) > 1e-10:
            raise SystemExit(
                f"[{split}] alpha mismatch at idx={idx}: "
                f"{a['meta']['alpha']} vs {b['meta']['alpha']}"
            )
        if i % 8 == 0 or i == len(idxs):
            print(f"  [{i:>3}/{len(idxs)}] idx={idx}: OK")

    print(f"[{split}] PASS — all {len(idxs)} pairs byte-identical.")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--bundle", type=Path, required=True,
                   help="path to patches.npy (the Colab bundle)")
    p.add_argument("--k", type=int, default=32,
                   help="number of random idxs to check per split")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    if not args.bundle.exists():
        raise SystemExit(f"bundle not found: {args.bundle}")

    for split in ("train", "val"):
        validate_split(split, args.bundle, args.k, args.seed)

    print("\nALL SPLITS PASS — bundle-mode Dataset is Colab-ready.")


if __name__ == "__main__":
    main()
