"""Pre-flight checks before launching a real training run.

Validates the things that, if broken, would only surface mid-run:

  - manifest loads and has the columns and split coverage we expect
  - bundle ``patches.npy`` exists, dtype/shape matches ``len(manifest)``
  - FPN templates load for all four (bits, tdi) modes the dataset draws
  - one batch of ``(g, sigma_fpn)`` assembles on the target device
  - U-Net instantiates and forwards a synthetic batch on the device
  - PG-NLL and MSE losses both produce finite values on that batch
  - target run directory has enough free space for checkpoints + samples

Run:
    .venv/bin/python -m training.preflight --bundle path/to/patches.npy \\
        --device cuda --runs-dir runs --min-free-gb 5
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch

import noise_model as nm
from model import OHRCDenoiserUNet, pg_nll_loss
from noise_model.fpn_template import load_fpn_template
from training.loop import assemble_noise_constants
from training_data.curation import PATCH_SIZE
from training_data.dataset import (
    DEFAULT_MANIFEST,
    OHRCReferenceDataset,
)

# All modes the dataset can simulate (cross-product of sim_bits × sim_tdi).
EXPECTED_MODES: tuple[tuple[str, int], ...] = (
    ("lsb", 64), ("lsb", 128), ("msb", 64), ("msb", 128),
)

EXPECTED_MANIFEST_COLUMNS = {
    "product_id", "split", "row0", "col0",
    "source_bits", "source_tdi",
}


class PreflightError(RuntimeError):
    """Raised when a check fails."""


def _ok(label: str, msg: str = "") -> None:
    suffix = f" — {msg}" if msg else ""
    print(f"  [OK]  {label}{suffix}")


def _fail(label: str, msg: str) -> None:
    print(f"  [FAIL] {label} — {msg}")


def check_manifest(manifest_path: Path) -> dict:
    """Load manifest, verify schema + split coverage. Returns manifest df."""
    label = f"manifest schema ({manifest_path.name})"
    if not manifest_path.exists():
        raise PreflightError(f"manifest not found: {manifest_path}")
    df = pq.read_table(manifest_path).to_pandas()
    missing = EXPECTED_MANIFEST_COLUMNS - set(df.columns)
    if missing:
        raise PreflightError(
            f"manifest missing columns: {sorted(missing)}; "
            f"have {df.columns.tolist()}"
        )
    splits = set(df["split"].unique())
    if not {"train", "val"}.issubset(splits):
        raise PreflightError(
            f"manifest must have both 'train' and 'val' splits; got {splits}"
        )
    n_train = int((df["split"] == "train").sum())
    n_val = int((df["split"] == "val").sum())
    _ok(label, f"rows={len(df):,}  train={n_train:,}  val={n_val:,}")
    return df


def check_bundle(bundle_path: Path | None, n_manifest: int) -> None:
    label = "bundle"
    if bundle_path is None:
        _ok(label, "no bundle given (catalog path will be used)")
        return
    if not bundle_path.exists():
        raise PreflightError(f"bundle does not exist: {bundle_path}")
    arr = np.load(bundle_path, mmap_mode="r")
    if arr.dtype != np.uint8:
        raise PreflightError(
            f"bundle dtype must be uint8; got {arr.dtype}"
        )
    expected_shape = (n_manifest, PATCH_SIZE, PATCH_SIZE)
    if arr.shape != expected_shape:
        raise PreflightError(
            f"bundle shape {arr.shape} != expected {expected_shape}"
        )
    size_gb = bundle_path.stat().st_size / 1e9
    _ok(label, f"shape={arr.shape}  dtype={arr.dtype}  size={size_gb:.1f} GB")


def check_fpn_templates() -> None:
    label = "FPN templates"
    for bits, tdi in EXPECTED_MODES:
        try:
            tmpl = load_fpn_template(bits, tdi)
        except Exception as e:
            raise PreflightError(
                f"failed to load FPN template ({bits}, TDI{tdi}): {e}"
            ) from e
        if tmpl.sigma_fpn.ndim != 1 or tmpl.sigma_fpn.shape[0] < PATCH_SIZE:
            raise PreflightError(
                f"FPN template ({bits}, TDI{tdi}) σ_fpn shape "
                f"{tmpl.sigma_fpn.shape} too small for patch width {PATCH_SIZE}"
            )
    _ok(label, f"{len(EXPECTED_MODES)} modes loaded")


def check_dataset_and_model(
    bundle_path: Path | None,
    device: torch.device,
    seed: int = 0,
) -> None:
    """End-to-end: one batch from the Dataset → model → both losses."""
    ds = OHRCReferenceDataset(
        split="train", seed=seed, bundle_path=bundle_path,
    )
    if len(ds) == 0:
        raise PreflightError("train split is empty")

    # Tiny batch via default collate.
    from torch.utils.data import DataLoader
    loader = DataLoader(ds, batch_size=2, shuffle=False, num_workers=0)
    batch = next(iter(loader))

    # Shape sanity.
    expected = (2, 1, PATCH_SIZE, PATCH_SIZE)
    if tuple(batch["clean"].shape) != expected:
        raise PreflightError(
            f"clean shape {tuple(batch['clean'].shape)} != {expected}"
        )
    if tuple(batch["noisy"].shape) != expected:
        raise PreflightError(
            f"noisy shape {tuple(batch['noisy'].shape)} != {expected}"
        )
    _ok("Dataset batch", f"clean/noisy shape={expected}  "
        f"clean.mean={float(batch['clean'].mean()):.2f}  "
        f"noisy.std={float(batch['noisy'].std()):.2f}")

    # Noise constants.
    g, sigma_fpn = assemble_noise_constants(batch["meta"], device)
    if g.shape != (2, 1, 1, 1):
        raise PreflightError(f"g shape {g.shape} != (2,1,1,1)")
    if sigma_fpn.shape != (2, 1, 1, PATCH_SIZE):
        raise PreflightError(
            f"sigma_fpn shape {sigma_fpn.shape} != (2,1,1,{PATCH_SIZE})"
        )
    _ok("noise constants", f"g={g.view(-1).tolist()}  "
        f"σ_fpn.mean={float(sigma_fpn.mean()):.3f} DN")

    # Model forward + losses.
    model = OHRCDenoiserUNet().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    noisy = batch["noisy"].to(device)
    clean = batch["clean"].to(device)
    out = model(noisy)
    if set(out.keys()) != {"mu", "delta"}:
        raise PreflightError(f"model output keys {out.keys()} != {{mu,delta}}")
    if tuple(out["mu"].shape) != expected:
        raise PreflightError(
            f"mu shape {tuple(out['mu'].shape)} != {expected}"
        )

    import torch.nn.functional as F
    mse = F.mse_loss(out["mu"], clean)
    if not torch.isfinite(mse):
        raise PreflightError(f"MSE not finite: {mse}")
    pgnll = pg_nll_loss(
        out["mu"], out["delta"], clean, g,
        nm.SIGMA_FLOOR_EFF_E, sigma_fpn,
    )
    if not torch.isfinite(pgnll):
        raise PreflightError(f"PG-NLL not finite: {pgnll}")
    # Gradient flow check.
    pgnll.backward()
    grad_ok = any(
        p.grad is not None and torch.isfinite(p.grad).all()
        for p in model.parameters()
    )
    if not grad_ok:
        raise PreflightError("no finite gradients after PG-NLL backward")
    _ok("model + losses", f"params={n_params:,}  device={device}  "
        f"MSE={float(mse.detach()):.3f}  "
        f"PG-NLL={float(pgnll.detach()):.3f}  grads=finite")


def check_disk_space(runs_dir: Path, min_free_gb: float) -> None:
    runs_dir.mkdir(parents=True, exist_ok=True)
    usage = shutil.disk_usage(runs_dir)
    free_gb = usage.free / 1e9
    if free_gb < min_free_gb:
        raise PreflightError(
            f"only {free_gb:.1f} GB free at {runs_dir} "
            f"(need ≥ {min_free_gb} GB)"
        )
    _ok("disk space", f"{free_gb:.1f} GB free at {runs_dir}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--manifest", type=Path, default=DEFAULT_MANIFEST,
        help="path to reference_manifest.parquet",
    )
    p.add_argument(
        "--bundle", type=Path, default=None,
        help="path to patches.npy (omit to validate catalog/memmap path)",
    )
    p.add_argument(
        "--device", default="auto",
        help="'auto' (cuda→mps→cpu), 'cuda', 'mps', or 'cpu'",
    )
    p.add_argument(
        "--runs-dir", type=Path, default=Path("runs"),
        help="runs directory to disk-space check",
    )
    p.add_argument("--min-free-gb", type=float, default=5.0)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    if args.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    print(f"Pre-flight on device={device}")

    try:
        df = check_manifest(args.manifest)
        check_bundle(args.bundle, n_manifest=len(df))
        check_fpn_templates()
        check_dataset_and_model(args.bundle, device, seed=args.seed)
        check_disk_space(args.runs_dir, args.min_free_gb)
    except PreflightError as e:
        _fail("pre-flight", str(e))
        return 1

    print("\nALL CHECKS PASS — ready to launch training.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
