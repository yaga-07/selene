"""Real trainer for SELENE Phase-3 — one ablation per invocation.

One run = one loss objective for the whole schedule (no MSE→PG-NLL
curriculum, dropped 2026-06-28). Each invocation produces an isolated
``runs/<run_id>/`` directory with:

  - ``config.json``      — snapshot of CLI args + git SHA + started_at
  - ``train_log.csv``    — per-step (step, epoch, lr, loss, time_s)
  - ``val_log.jsonl``    — per-val (step, epoch, val_loss, psnr_dark)
  - ``latest.pt``        — written every ``--ckpt-every`` steps
  - ``best.pt``          — written when val_loss improves
  - ``best.meta.json``   — sidecar with (step, epoch, val_loss, psnr_dark,
                            saved_at) — rewritten with each best.pt
  - ``samples/``         — denoised PNG triptychs each val eval, run on a
                            fixed showcase set spanning val mean_dn
                            percentiles so terrain texture is visible

**Resume.** Re-running with the same ``--run-dir`` and ``latest.pt``
present resumes from the last checkpointed step. Model, optimizer,
scheduler, RNG, and bookkeeping all restore.

**Drive sync (Colab).** ``--drive-mirror`` points at a Drive path; the
run dir is rsync'd there at every val cadence and again at epoch end.

**PSNR.** Pinned: ``peak=255``, masked by ``clean < dark_threshold``.
This isolates the low-signal regime that SELENE is designed for.

Run (Colab):
    .venv/bin/python -m training.train \\
        --bundle /content/patches.npy \\
        --loss pg_nll --epochs 80 --batch-size 32 --device cuda \\
        --run-dir runs/E2_pg_nll \\
        --drive-mirror /content/drive/MyDrive/selene-runs/E2_pg_nll
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import noise_model as nm
from model import OHRCDenoiserUNet, pg_nll_loss
from training.loop import assemble_noise_constants
from training.sampler import make_mode_balanced_sampler
from training_data.curation import PATCH_SIZE
from training_data.dataset import DEFAULT_MANIFEST, OHRCReferenceDataset

# ----- config ----------------------------------------------------------------

DEFAULT_DARK_THRESHOLD_DN = 20.0  # mask for PSNR — dark/PSR-like regime


@dataclass
class RunConfig:
    bundle: str | None
    manifest: str
    loss: str  # "mse" or "pg_nll"
    epochs: int
    batch_size: int
    lr: float
    weight_decay: float
    warmup_steps: int
    val_every_steps: int
    ckpt_every_steps: int
    val_batches: int
    num_workers: int
    device: str
    seed: int
    val_seed: int
    run_dir: str
    drive_mirror: str | None
    dark_threshold_dn: float
    n_sample_pngs: int


# ----- helpers ---------------------------------------------------------------


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return "unknown"


def _device(name: str) -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(name)


def _lr_lambda(step: int, warmup: int, total: int) -> float:
    """Linear warmup → cosine annealing to 0."""
    if step < warmup:
        return (step + 1) / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    progress = min(1.0, max(0.0, progress))
    return 0.5 * (1.0 + math.cos(math.pi * progress))


def _compute_loss(
    out: dict, batch: dict, loss_name: str, device: torch.device,
) -> torch.Tensor:
    target = batch["clean"].to(device)
    if loss_name == "mse":
        return F.mse_loss(out["mu"], target)
    patch_w = out["mu"].shape[-1]
    g, sigma_fpn = assemble_noise_constants(batch["meta"], device, patch_size=patch_w)
    return pg_nll_loss(
        out["mu"], out["delta"], target, g, nm.SIGMA_FLOOR_EFF_E, sigma_fpn,
    )


def _masked_psnr_dn(
    mu: torch.Tensor, clean: torch.Tensor, peak: float, threshold: float,
) -> float:
    """PSNR in DN-space, masked to clean < threshold (dark regime)."""
    mask = clean < threshold
    n = int(mask.sum())
    if n == 0:
        return float("nan")
    err = (mu - clean)[mask]
    mse = float((err * err).mean())
    if mse <= 0:
        return float("inf")
    return float(10.0 * math.log10((peak * peak) / mse))


def _save_sample_pngs(
    noisy: torch.Tensor, clean: torch.Tensor, mu: torch.Tensor,
    out_dir: Path, step: int, n: int,
    captions: list[str] | None = None,
) -> None:
    """Side-by-side noisy/clean/μ triptychs as PNGs. Uses matplotlib if
    available, falls back to PIL if not.

    For visual comparability across panels, vmin/vmax is the union range
    of (clean, μ) per sample — not a fixed 0/255 — so faint texture is
    actually visible in dark-regime patches.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    n = min(n, noisy.shape[0])
    captions = captions or [""] * n

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        from PIL import Image
        for i in range(n):
            row = []
            for t in (noisy[i, 0], clean[i, 0], mu[i, 0]):
                a = t.detach().cpu().numpy().clip(0, 255).astype(np.uint8)
                row.append(a)
            strip = np.concatenate(row, axis=1)
            Image.fromarray(strip).save(out_dir / f"step{step:08d}_s{i}.png")
        return

    for i in range(n):
        c = clean[i, 0].detach().cpu().numpy()
        nimg = noisy[i, 0].detach().cpu().numpy()
        m = mu[i, 0].detach().cpu().numpy()
        # Adaptive scale to whichever of (clean, μ) covers the wider
        # dynamic range — keeps the comparison fair while preventing
        # dark patches from rendering pure black.
        vmin = float(min(c.min(), m.min(), nimg.min()))
        vmax = float(max(c.max(), m.max(), nimg.max()))
        if vmax <= vmin:
            vmax = vmin + 1.0
        fig, axes = plt.subplots(1, 3, figsize=(9, 3.2))
        for ax, t, title in zip(
            axes, (nimg, c, m), ("noisy", "clean", "μ (denoised)"),
        ):
            im = ax.imshow(t, cmap="gray", vmin=vmin, vmax=vmax)
            ax.set_title(
                f"{title}\nmean={t.mean():.1f}  "
                f"range=[{t.min():.1f},{t.max():.1f}]"
            )
            ax.axis("off")
            fig.colorbar(im, ax=ax, fraction=0.046)
        suptitle = f"step {step}  sample {i}"
        if captions[i]:
            suptitle += f"  —  {captions[i]}"
        fig.suptitle(suptitle)
        fig.tight_layout()
        fig.savefig(out_dir / f"step{step:08d}_s{i}.png", dpi=80)
        plt.close(fig)


def _pick_showcase_indices(
    val_df, n: int, percentiles: tuple[float, ...] | None = None,
) -> list[int]:
    """Pick ``n`` per-split val idxs spanning the val ``mean_dn``
    distribution, so sample PNGs show a mix of dark / mid / bright
    terrain rather than only the first few rows of the val parquet.

    If ``mean_dn`` is missing or constant, falls back to evenly-spaced
    idxs across the df.
    """
    if percentiles is None:
        percentiles = tuple(np.linspace(15, 95, n))
    if "mean_dn" not in val_df.columns or val_df["mean_dn"].nunique() < 2:
        return [int(i) for i in np.linspace(0, len(val_df) - 1, n).round()]
    qs = val_df["mean_dn"].quantile([p / 100 for p in percentiles]).values
    idxs: list[int] = []
    for q in qs:
        i = int((val_df["mean_dn"] - q).abs().idxmin())
        idxs.append(i)
    return idxs


def _build_showcase_batch(
    manifest_path: Path, bundle_path: Path | None, val_seed: int,
    n: int,
) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
    """Pre-build the (clean, noisy) tensors for the sample-PNG set.

    Uses ``deterministic=True`` so each chosen patch is shown at its
    natural α=1.0 brightness with ``sim_mode = source_mode`` — i.e. the
    network is evaluated against the same source mode as the patch, with
    no artificial dimming. Captures real texture quality at every
    illumination level.
    """
    ds_det = OHRCReferenceDataset(
        split="val", manifest_path=manifest_path,
        bundle_path=bundle_path, seed=val_seed, deterministic=True,
    )
    idxs = _pick_showcase_indices(ds_det.df, n)
    cleans, noisies, captions = [], [], []
    for i in idxs:
        item = ds_det[i]
        cleans.append(item["clean"])
        noisies.append(item["noisy"])
        m = item["meta"]
        captions.append(
            f"idx={i}  {m['source_bits']}/TDI{m['source_tdi']}  "
            f"clean_mean={float(item['clean'].mean()):.1f} DN"
        )
    return torch.stack(cleans), torch.stack(noisies), captions


def _mirror_to_drive(run_dir: Path, drive_path: Path) -> None:
    """rsync run_dir → drive_path. Best-effort; logs failure but never raises."""
    drive_path.mkdir(parents=True, exist_ok=True)
    try:
        subprocess.run(
            ["rsync", "-a", "--delete", str(run_dir) + "/", str(drive_path) + "/"],
            check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
        )
        print(f"  [drive] mirrored {run_dir} → {drive_path}")
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"  [drive] mirror failed ({type(e).__name__}); continuing")


def _save_ckpt(
    path: Path, model, optimizer, scheduler, step: int, epoch: int,
    best_val: float | None, cfg: RunConfig, rng_state: dict,
) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "step": step,
        "epoch": epoch,
        "best_val": best_val,
        "config": asdict(cfg),
        "rng": rng_state,
    }, tmp)
    tmp.replace(path)  # atomic on POSIX


def _save_best_meta(
    path: Path, step: int, epoch: int, val_loss: float, psnr_dark: float,
    wall_s: float, started_at: str, git_sha: str,
) -> None:
    """Sidecar JSON beside ``best.pt`` — same info, human-readable, cheap
    to grep without ``torch.load``. Rewritten on every best update."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps({
        "step": step, "epoch": epoch,
        "val_loss": val_loss, "psnr_dark": psnr_dark,
        "wall_s": wall_s,
        "saved_at": datetime.now(timezone.utc).isoformat(),
        "started_at": started_at,
        "git_sha": git_sha,
    }, indent=2))
    tmp.replace(path)


def _load_ckpt(path: Path, model, optimizer, scheduler, device) -> dict:
    state = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    return state


def _capture_rng() -> dict:
    return {
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "numpy": np.random.get_state(),
    }


def _restore_rng(state: dict) -> None:
    torch.set_rng_state(state["torch"])
    if state.get("cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])
    np.random.set_state(state["numpy"])


# ----- validation ------------------------------------------------------------


@torch.no_grad()
def evaluate(
    model, val_loader, loss_name: str, device, n_batches: int,
    dark_threshold: float,
) -> tuple[float, float]:
    """Returns (val_loss, psnr_dark) averaged over up to ``n_batches`` val batches."""
    model.eval()
    losses, psnrs = [], []
    for i, batch in enumerate(val_loader):
        if i >= n_batches:
            break
        out = model(batch["noisy"].to(device))
        loss = _compute_loss(out, batch, loss_name, device)
        losses.append(float(loss.detach().cpu()))
        psnrs.append(_masked_psnr_dn(
            out["mu"], batch["clean"].to(device),
            peak=255.0, threshold=dark_threshold,
        ))
    psnrs_finite = [p for p in psnrs if math.isfinite(p)]
    psnr = float(np.mean(psnrs_finite)) if psnrs_finite else float("nan")
    return float(np.mean(losses)), psnr


# ----- main loop -------------------------------------------------------------


def train(cfg: RunConfig) -> None:
    device = _device(cfg.device)
    run_dir = Path(cfg.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    samples_dir = run_dir / "samples"
    train_csv_path = run_dir / "train_log.csv"
    val_jsonl_path = run_dir / "val_log.jsonl"
    latest_ckpt = run_dir / "latest.pt"
    best_ckpt = run_dir / "best.pt"
    config_path = run_dir / "config.json"

    # Datasets.
    bundle_path = Path(cfg.bundle) if cfg.bundle else None
    train_ds = OHRCReferenceDataset(
        split="train", manifest_path=Path(cfg.manifest),
        bundle_path=bundle_path, seed=cfg.seed,
    )
    val_ds = OHRCReferenceDataset(
        split="val", manifest_path=Path(cfg.manifest),
        bundle_path=bundle_path, seed=cfg.val_seed,
        deterministic=False,  # see roadmap §3.4 / locked decisions
    )

    sampler = make_mode_balanced_sampler(train_ds.df, seed=cfg.seed)
    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, sampler=sampler,
        num_workers=cfg.num_workers, pin_memory=(device.type == "cuda"),
        drop_last=True, persistent_workers=cfg.num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=(device.type == "cuda"),
        persistent_workers=cfg.num_workers > 0,
    )

    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * cfg.epochs

    model = OHRCDenoiserUNet().to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda s: _lr_lambda(s, cfg.warmup_steps, total_steps),
    )

    # Write config (overwritten each launch; resume reads it from ckpt).
    started_at = datetime.now(timezone.utc).isoformat()
    git_sha = _git_sha()
    cfg_dict = asdict(cfg)
    cfg_dict["git_sha"] = git_sha
    cfg_dict["total_steps"] = total_steps
    cfg_dict["steps_per_epoch"] = steps_per_epoch
    cfg_dict["started_at"] = started_at
    config_path.write_text(json.dumps(cfg_dict, indent=2))

    # Pre-build the showcase batch — varied terrain across mean_dn
    # percentiles, rendered at α=1.0 so faint texture stays visible.
    showcase_clean, showcase_noisy, showcase_caps = _build_showcase_batch(
        Path(cfg.manifest), bundle_path, cfg.val_seed, cfg.n_sample_pngs,
    )

    # Resume.
    start_step, start_epoch, best_val = 0, 0, None
    if latest_ckpt.exists():
        state = _load_ckpt(latest_ckpt, model, optimizer, scheduler, device)
        start_step = int(state["step"])
        start_epoch = int(state["epoch"])
        best_val = state.get("best_val")
        if "rng" in state and state["rng"] is not None:
            _restore_rng(state["rng"])
        print(f"resumed from {latest_ckpt}: step={start_step} epoch={start_epoch} "
              f"best_val={best_val}")
    else:
        torch.manual_seed(cfg.seed)
        np.random.seed(cfg.seed)
        print(f"fresh run at {run_dir}")
        with train_csv_path.open("w", newline="") as f:
            csv.writer(f).writerow(["step", "epoch", "lr", "loss", "wall_s"])

    print(f"device={device}  loss={cfg.loss}  epochs={cfg.epochs}  "
          f"batch={cfg.batch_size}  steps/epoch={steps_per_epoch}  total={total_steps}")
    print(f"warmup_steps={cfg.warmup_steps}  val_every={cfg.val_every_steps}  "
          f"ckpt_every={cfg.ckpt_every_steps}")

    step = start_step
    t0 = time.time()
    for epoch in range(start_epoch, cfg.epochs):
        model.train()
        for batch in train_loader:
            optimizer.zero_grad()
            out = model(batch["noisy"].to(device))
            loss = _compute_loss(out, batch, cfg.loss, device)
            loss.backward()
            optimizer.step()
            scheduler.step()
            step += 1

            loss_val = float(loss.detach().cpu())
            wall = time.time() - t0
            with train_csv_path.open("a", newline="") as f:
                csv.writer(f).writerow([
                    step, epoch, scheduler.get_last_lr()[0], loss_val, wall,
                ])
            if step == 1 or step % 50 == 0:
                print(f"  step {step:>7}/{total_steps}  ep{epoch:>3}  "
                      f"lr={scheduler.get_last_lr()[0]:.2e}  loss={loss_val:.4f}")

            # Checkpoint cadence.
            if step % cfg.ckpt_every_steps == 0:
                _save_ckpt(
                    latest_ckpt, model, optimizer, scheduler,
                    step, epoch, best_val, cfg, _capture_rng(),
                )

            # Val cadence.
            if step % cfg.val_every_steps == 0:
                val_loss, psnr = evaluate(
                    model, val_loader, cfg.loss, device,
                    n_batches=cfg.val_batches,
                    dark_threshold=cfg.dark_threshold_dn,
                )
                wall_s = time.time() - t0
                with val_jsonl_path.open("a") as f:
                    f.write(json.dumps({
                        "step": step, "epoch": epoch,
                        "val_loss": val_loss, "psnr_dark": psnr,
                        "wall_s": wall_s,
                    }) + "\n")
                print(f"  [val]   step {step}  val_loss={val_loss:.4f}  "
                      f"psnr_dark={psnr:.2f} dB")
                if best_val is None or val_loss < best_val:
                    best_val = val_loss
                    _save_ckpt(
                        best_ckpt, model, optimizer, scheduler,
                        step, epoch, best_val, cfg, _capture_rng(),
                    )
                    _save_best_meta(
                        run_dir / "best.meta.json", step=step, epoch=epoch,
                        val_loss=val_loss, psnr_dark=psnr,
                        wall_s=wall_s, started_at=started_at, git_sha=git_sha,
                    )
                    print(f"  [val]   new best — saved {best_ckpt.name} + best.meta.json")
                # Showcase samples — same idxs every val cycle, so PNGs
                # form a comparable time series across the run.
                with torch.no_grad():
                    showcase_out = model(showcase_noisy.to(device))
                _save_sample_pngs(
                    showcase_noisy, showcase_clean, showcase_out["mu"].cpu(),
                    samples_dir, step,
                    n=cfg.n_sample_pngs, captions=showcase_caps,
                )
                # Mirror at val cadence too, so a disconnect mid-epoch
                # still leaves a recent latest.pt on Drive.
                if cfg.drive_mirror:
                    _mirror_to_drive(run_dir, Path(cfg.drive_mirror))
                model.train()

        # End-of-epoch always checkpoints + mirrors.
        _save_ckpt(
            latest_ckpt, model, optimizer, scheduler,
            step, epoch + 1, best_val, cfg, _capture_rng(),
        )
        if cfg.drive_mirror:
            _mirror_to_drive(run_dir, Path(cfg.drive_mirror))

    print(f"\nTRAINING COMPLETE  step={step}  best_val={best_val}")


def parse_args(argv: list[str] | None = None) -> RunConfig:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--bundle", default=None)
    p.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    p.add_argument("--loss", choices=["mse", "pg_nll"], required=True)
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--warmup-steps", type=int, default=500)
    p.add_argument("--val-every-steps", type=int, default=1000)
    p.add_argument("--ckpt-every-steps", type=int, default=200)
    p.add_argument("--val-batches", type=int, default=32,
                   help="number of val batches per evaluate() call")
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--device", default="auto")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--val-seed", type=int, default=1234,
                   help="separate seed for val Dataset RNG")
    p.add_argument("--run-dir", required=True)
    p.add_argument("--drive-mirror", default=None,
                   help="optional Drive path to rsync the run dir into at epoch end")
    p.add_argument("--dark-threshold-dn", type=float, default=DEFAULT_DARK_THRESHOLD_DN)
    p.add_argument("--n-sample-pngs", type=int, default=4)
    args = p.parse_args(argv)
    return RunConfig(
        bundle=args.bundle,
        manifest=args.manifest,
        loss=args.loss,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        val_every_steps=args.val_every_steps,
        ckpt_every_steps=args.ckpt_every_steps,
        val_batches=args.val_batches,
        num_workers=args.num_workers,
        device=args.device,
        seed=args.seed,
        val_seed=args.val_seed,
        run_dir=args.run_dir,
        drive_mirror=args.drive_mirror,
        dark_threshold_dn=args.dark_threshold_dn,
        n_sample_pngs=args.n_sample_pngs,
    )


def main(argv: list[str] | None = None) -> int:
    cfg = parse_args(argv)
    train(cfg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
