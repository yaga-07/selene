"""Minimal trainer skeleton — proves the U-Net + losses + Dataset wire
together and that gradients flow on MPS.

This is **not** the Colab training script. It exists so:
    1. The per-batch (g, σ_FPN) assembly is unit-tested in isolation.
    2. The MSE → PG-NLL phase switch (`PROJECT_ROADMAP.md` §3.4) has a
       reference implementation the full trainer can copy.
    3. A 100-step sanity run on 32 patches via MPS confirms loss
       decreases before we burn Colab time.

Run the sanity loop (SSD must be mounted for the Dataset):
    .venv/bin/python -m training.loop --n-steps 100 --device mps
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

import noise_model as nm
from model import OHRCDenoiserUNet, pg_nll_loss
from noise_model.fpn_template import load_fpn_template
from training_data.curation import PATCH_SIZE
from training_data.dataset import OHRCReferenceDataset

_FPN_TENSOR_CACHE: dict[tuple[str, int, torch.device], torch.Tensor] = {}


def _get_fpn_sigma(bits: str, tdi: int, device: torch.device) -> torch.Tensor:
    key = (bits, tdi, device)
    if key not in _FPN_TENSOR_CACHE:
        tmpl = load_fpn_template(bits, tdi)
        _FPN_TENSOR_CACHE[key] = torch.from_numpy(tmpl.sigma_fpn).to(device)
    return _FPN_TENSOR_CACHE[key]


def assemble_noise_constants(
    meta: dict,
    device: torch.device,
    patch_size: int = PATCH_SIZE,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build per-sample ``(g, sigma_fpn)`` tensors from a collated batch.

    The Dataset returns per-sample ``meta`` with ``sim_bits`` (str),
    ``sim_tdi`` (int), ``col0`` (int). The default DataLoader collate
    turns scalars into 1-D tensors and strings into Python lists.

    Args:
        meta: collated meta dict — ``sim_bits`` is ``list[str]`` of
            length N; ``sim_tdi`` and ``col0`` are 1-D int tensors.
        device: target device for the output tensors.
        patch_size: column-slice width for the FPN tensor.

    Returns:
        g: ``(N, 1, 1, 1)`` DN/e⁻ per sample.
        sigma_fpn: ``(N, 1, 1, patch_size)`` DN per-column slice
            from the per-(bits, tdi) FPN template at the sample's
            ``col0`` offset.
    """
    sim_bits: list[str] = meta["sim_bits"]
    sim_tdi = meta["sim_tdi"].tolist() if torch.is_tensor(meta["sim_tdi"]) \
        else list(meta["sim_tdi"])
    col0 = meta["col0"].tolist() if torch.is_tensor(meta["col0"]) \
        else list(meta["col0"])

    n = len(sim_bits)
    g = torch.tensor(
        [nm.gain_dn_per_e(b) for b in sim_bits],
        dtype=torch.float32, device=device,
    ).view(n, 1, 1, 1)

    sigma_fpn = torch.empty((n, 1, 1, patch_size), dtype=torch.float32, device=device)
    for i, (b, t, c) in enumerate(zip(sim_bits, sim_tdi, col0)):
        full = _get_fpn_sigma(b, int(t), device)
        sigma_fpn[i, 0, 0, :] = full[int(c):int(c) + patch_size]
    return g, sigma_fpn


def compute_loss(
    out: dict[str, torch.Tensor],
    batch: dict,
    loss_phase: str,
    device: torch.device,
) -> torch.Tensor:
    """Dispatch to MSE (vanilla) or PG-NLL (physics-prior) per ``loss_phase``."""
    target = batch["clean"].to(device)
    if loss_phase == "mse":
        return F.mse_loss(out["mu"], target)
    if loss_phase == "pg_nll":
        patch_w = out["mu"].shape[-1]
        g, sigma_fpn = assemble_noise_constants(batch["meta"], device,
                                                 patch_size=patch_w)
        return pg_nll_loss(
            out["mu"], out["delta"], target, g,
            nm.SIGMA_FLOOR_EFF_E, sigma_fpn,
        )
    raise ValueError(f"loss_phase must be 'mse' or 'pg_nll', got {loss_phase!r}")


def train_step(
    model: torch.nn.Module,
    batch: dict,
    optimizer: torch.optim.Optimizer,
    loss_phase: str,
    device: torch.device,
) -> float:
    """One forward+backward+step. Returns the loss value as a Python float."""
    model.train()
    optimizer.zero_grad()
    noisy = batch["noisy"].to(device)
    out = model(noisy)
    loss = compute_loss(out, batch, loss_phase, device)
    loss.backward()
    optimizer.step()
    return float(loss.detach().cpu())


def run_sanity_loop(
    n_steps: int = 100,
    n_patches: int = 32,
    batch_size: int = 4,
    mse_warmup_steps: int = 50,
    lr: float = 1e-4,
    weight_decay: float = 1e-5,
    device_str: str = "mps",
    seed: int = 0,
) -> dict:
    """Proof-of-life loop on a tiny Subset of the train split.

    Picks the first ``n_patches`` indices, builds a DataLoader, and
    runs ``n_steps`` train_step iterations — switching from MSE to
    PG-NLL at ``mse_warmup_steps`` to exercise both phases.

    Returns:
        ``{"losses": {"mse": [...], "pg_nll": [...]}, "elapsed_s": ...}``
    """
    device = torch.device(device_str)
    torch.manual_seed(seed)

    ds = OHRCReferenceDataset(split="train", seed=seed)
    subset = Subset(ds, list(range(min(n_patches, len(ds)))))
    loader = DataLoader(subset, batch_size=batch_size, shuffle=True, num_workers=0)

    model = OHRCDenoiserUNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    losses: dict[str, list[float]] = {"mse": [], "pg_nll": []}
    t0 = time.time()
    step = 0
    while step < n_steps:
        for batch in loader:
            phase = "mse" if step < mse_warmup_steps else "pg_nll"
            loss_val = train_step(model, batch, optimizer, phase, device)
            losses[phase].append(loss_val)
            step += 1
            if step % 10 == 0 or step == 1:
                print(f"  step {step:>4}  phase={phase:<7}  loss={loss_val:.4f}")
            if step >= n_steps:
                break

    elapsed = time.time() - t0
    print(f"\n{n_steps} steps in {elapsed:.1f}s "
          f"({n_steps / elapsed:.2f} steps/s on {device_str})")
    return {"losses": losses, "elapsed_s": elapsed}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--n-steps", type=int, default=100)
    p.add_argument("--n-patches", type=int, default=32)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--mse-warmup-steps", type=int, default=50)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--device", default="mps")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    print(f"sanity loop: {args.n_steps} steps on {args.n_patches} patches "
          f"(batch={args.batch_size}, mse→pg_nll @ step {args.mse_warmup_steps}, "
          f"device={args.device})")

    out = run_sanity_loop(
        n_steps=args.n_steps,
        n_patches=args.n_patches,
        batch_size=args.batch_size,
        mse_warmup_steps=args.mse_warmup_steps,
        lr=args.lr,
        device_str=args.device,
        seed=args.seed,
    )

    print("\nper-phase loss trend (first-half mean vs second-half mean):")
    for phase, ls in out["losses"].items():
        if len(ls) < 4:
            continue
        half = len(ls) // 2
        first = float(np.mean(ls[:half]))
        second = float(np.mean(ls[half:]))
        change = second - first
        verdict = "↓" if change < 0 else "↑"
        print(f"  {phase:<7} n={len(ls):>3}  first={first:.4f}  "
              f"second={second:.4f}  Δ={change:+.4f} {verdict}")


if __name__ == "__main__":
    main()
