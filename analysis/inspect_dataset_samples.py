"""Eyeball ``OHRCReferenceDataset`` output across the α distribution.

Picks ``N_PER_BIN`` samples in each of 4 log-spaced α bins (so the dim,
mid, bright regimes the trainer will see are all visible), then renders
one row per sample with 4 panels: ``clean | noisy | noisy − clean |
column-mean profile``. The column-mean panel is what makes per-column
FPN visible — if the noisy curve has rapid spatial variation absent in
the clean curve, FPN is being applied as intended.

Run (SSD must be mounted):
  .venv/bin/python -m analysis.inspect_dataset_samples

Writes ``analysis/_outputs/inspect_dataset_samples.png`` plus a short
text summary of (α, sim_mode, mean DN) per sample.
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from training_data.dataset import (
    DEFAULT_ALPHA_LOG_RANGE,
    DEFAULT_SIM_BITS_PROBS,
    DEFAULT_SIM_TDI_PROBS,
    OHRCReferenceDataset,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PNG = REPO_ROOT / "analysis" / "_outputs" / "inspect_dataset_samples.png"

ALPHA_BIN_EDGES = (0.05, 0.10, 0.22, 0.47, 1.0001)
N_PER_BIN = 2
SCAN_POOL = 800             # candidate idxs to compute α for
DATASET_SEED = 2026         # base seed for the Dataset's per-item rngs
SCAN_SEED = 0               # rng for picking candidate idxs


def _predict_alpha(base_seed: int, idx: int) -> float:
    """Mirror Dataset.__getitem__'s α draw — pure-math, no disk."""
    rng = np.random.default_rng(np.random.SeedSequence([base_seed, idx]))
    log_lo, log_hi = DEFAULT_ALPHA_LOG_RANGE
    return float(math.exp(rng.uniform(log_lo, log_hi)))


def _pick_idxs(n_items: int, base_seed: int) -> list[int]:
    """Pick N_PER_BIN idxs per α-bin so the eyeball grid spans the distribution."""
    rng = np.random.default_rng(SCAN_SEED)
    pool = rng.choice(n_items, size=min(SCAN_POOL, n_items), replace=False)
    alphas = np.array([_predict_alpha(base_seed, int(i)) for i in pool])

    chosen: list[int] = []
    for b in range(len(ALPHA_BIN_EDGES) - 1):
        lo, hi = ALPHA_BIN_EDGES[b], ALPHA_BIN_EDGES[b + 1]
        mask = (alphas >= lo) & (alphas < hi)
        candidates = pool[mask][:N_PER_BIN]
        if len(candidates) < N_PER_BIN:
            print(f"warning: α-bin [{lo:.3f}, {hi:.3f}) only filled "
                  f"{len(candidates)}/{N_PER_BIN}")
        chosen.extend(int(c) for c in candidates)
    return chosen


def _render(samples: list[dict], output_path: Path) -> None:
    nrow = len(samples)
    fig, axes = plt.subplots(nrow, 4, figsize=(15, 3.0 * nrow))
    if nrow == 1:
        axes = axes[None, :]

    for r, s in enumerate(samples):
        clean = s["clean"].squeeze(0).numpy()
        noisy = s["noisy"].squeeze(0).numpy()
        resid = noisy - clean
        m = s["meta"]
        vmax = max(float(clean.max()), float(noisy.max()), 1.0)
        rabs = float(max(abs(resid.min()), abs(resid.max()), 1.0))

        axes[r, 0].imshow(clean, cmap="gray", vmin=0, vmax=vmax)
        axes[r, 0].set_title(
            f"clean   α={m['alpha']:.3f}\n"
            f"mean={clean.mean():.2f}  max={clean.max():.1f}",
            fontsize=8,
        )
        axes[r, 1].imshow(noisy, cmap="gray", vmin=0, vmax=vmax)
        axes[r, 1].set_title(
            f"noisy   sim {m['sim_bits']}/TDI{m['sim_tdi']}\n"
            f"mean={noisy.mean():.2f}  std={noisy.std():.2f}",
            fontsize=8,
        )
        axes[r, 2].imshow(resid, cmap="RdBu_r", vmin=-rabs, vmax=rabs)
        axes[r, 2].set_title(
            f"noisy − clean\n"
            f"std={resid.std():.2f}  max|Δ|={rabs:.1f}",
            fontsize=8,
        )
        axes[r, 3].plot(clean.mean(axis=0), color="black", lw=0.8, label="clean")
        axes[r, 3].plot(noisy.mean(axis=0), color="tab:red", lw=0.8,
                        alpha=0.85, label="noisy")
        axes[r, 3].set_title("column-mean profile (FPN visible if striped)",
                             fontsize=8)
        axes[r, 3].set_xlim(0, clean.shape[1] - 1)
        axes[r, 3].legend(fontsize=6, loc="upper right")
        axes[r, 3].tick_params(labelsize=6)
        axes[r, 3].set_xlabel("column", fontsize=7)
        axes[r, 3].set_ylabel("mean DN", fontsize=7)

        for ax in axes[r, :3]:
            ax.set_xticks([])
            ax.set_yticks([])
        axes[r, 0].set_ylabel(
            f"src {m['source_bits']}/TDI{m['source_tdi']}\n"
            f"…{m['product_id'][-22:]}\nr0={m['row0']} c0={m['col0']}",
            fontsize=7, rotation=0, ha="right", va="center", labelpad=58,
        )

    fig.suptitle(
        "OHRCReferenceDataset.__getitem__ — samples stratified by α (log bins).\n"
        "α scales the reference patch before forward-noise injection. Read mean DN "
        "in panel titles to compare brightness across rows (shared vmax per row).",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ds = OHRCReferenceDataset(split="train", seed=DATASET_SEED)
    print(f"loaded train split: {len(ds)} patches")
    print(f"sampling distribution: α LogUniform{DEFAULT_ALPHA_LOG_RANGE}, "
          f"sim_bits={DEFAULT_SIM_BITS_PROBS}, sim_tdi={DEFAULT_SIM_TDI_PROBS}")

    chosen_idxs = _pick_idxs(len(ds), DATASET_SEED)
    print(f"picked {len(chosen_idxs)} idxs across {len(ALPHA_BIN_EDGES) - 1} α-bins")
    samples = [ds[i] for i in chosen_idxs]

    _render(samples, OUTPUT_PNG)
    print(f"wrote {OUTPUT_PNG.relative_to(REPO_ROOT)}")
    print()
    print(f"{'idx':>7s}  {'α':>6s}  {'src':>10s}  {'sim':>10s}  "
          f"{'clean.mean':>10s}  {'noisy.mean':>10s}  {'noisy.std':>10s}")
    for s in samples:
        m = s["meta"]
        clean = s["clean"].numpy()
        noisy = s["noisy"].numpy()
        print(f"{m['idx']:>7d}  {m['alpha']:>6.3f}  "
              f"{m['source_bits']:>3s}/TDI{m['source_tdi']:>3d}  "
              f"{m['sim_bits']:>3s}/TDI{m['sim_tdi']:>3d}  "
              f"{clean.mean():>10.2f}  {noisy.mean():>10.2f}  {noisy.std():>10.2f}")


if __name__ == "__main__":
    main()
