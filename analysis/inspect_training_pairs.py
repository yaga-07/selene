"""Sample candidate training patches from a few strips and render a
multi-panel diagnostic PNG + a curation-stat CSV per strip.

Purpose: calibrate the §2.1 curation thresholds and eyeball the
forward-noise injector on real OHRC patches before ``build_manifest.py``
applies the rules at scale.

Run:  ``.venv/bin/python -m analysis.inspect_training_pairs``
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import catalog
import noise_model as nm
from noise_model.fpn_template import load_fpn_template, template_available
from training_data.curation import (
    PATCH_SIZE,
    check_patch,
    compute_patch_stats,
)

GRID_ROWS = 4
GRID_COLS = 3
N_ACCEPT_TO_RENDER = 3
ALPHA_INSPECT = 0.2
RNG_SEED = 42

REPO_ROOT = Path(__file__).resolve().parents[1]
SPLITS_PATH = REPO_ROOT / "training_data" / "splits.json"
OUTPUT_DIR = REPO_ROOT / "analysis" / "_outputs"


@dataclass(frozen=True)
class StripInfo:
    product_id: str
    bits: str
    tdi: int
    line_count: int
    sample_count: int
    solar_incidence: float
    label: str  # "test/equatorial_held_out", etc.


def _pick_strips(splits: dict, df: pd.DataFrame) -> list[StripInfo]:
    """Three strips: brightest eq test, Cabeus-band polar LSB test, a bright eq train."""
    picks: list[tuple[str, str]] = [
        (splits["test"]["equatorial_held_out"][0], "test/equatorial_held_out[0]"),
        (splits["test"]["polar_held_out"][0],      "test/polar_held_out[0]"),
        # Bright train eq: pick deterministically — lowest sol_inc among train strips.
        (_brightest_train(splits, df),             "train/brightest"),
    ]
    out: list[StripInfo] = []
    for pid, label in picks:
        row = df[df["product_id"] == pid].iloc[0]
        out.append(StripInfo(
            product_id=pid,
            bits=str(row["bits_selection"]),
            tdi=int(str(row["tdi_stages"]).replace("TDI", "")),
            line_count=int(row["line_count"]),
            sample_count=int(row["sample_count"]),
            solar_incidence=float(row["solar_incidence"]),
            label=label,
        ))
    return out


def _brightest_train(splits: dict, df: pd.DataFrame) -> str:
    train = df[df["product_id"].isin(splits["train"])]
    return str(train.sort_values("solar_incidence").iloc[0]["product_id"])


def _grid_positions(strip: StripInfo) -> list[tuple[int, int]]:
    """Deterministic grid of (row0, col0) corners for 256×256 patches."""
    row_fracs = np.linspace(0.15, 0.85, GRID_ROWS)
    col_fracs = np.linspace(0.15, 0.85, GRID_COLS)
    max_r = strip.line_count - PATCH_SIZE
    max_c = strip.sample_count - PATCH_SIZE
    return [
        (int(rf * max_r), int(cf * max_c))
        for rf in row_fracs for cf in col_fracs
    ]


def _sigma_physics_dn(
    clean_dn: np.ndarray,
    bits: str,
    fpn_template,
    col_offset: int,
) -> np.ndarray:
    g = nm.gain_dn_per_e(bits)
    sigma_floor_dn = nm.SIGMA_FLOOR_EFF_E * g
    n_cols = clean_dn.shape[1]
    sigma_fpn = fpn_template.sigma_fpn[col_offset:col_offset + n_cols].astype(np.float32)
    var = (g * np.maximum(clean_dn, 0.0)
           + sigma_floor_dn ** 2
           + sigma_fpn[None, :] ** 2)
    return np.sqrt(var)


def _render_png(
    strip: StripInfo,
    accepted: list[tuple[np.ndarray, int, int, dict]],
    output_path: Path,
) -> None:
    rng = np.random.default_rng(RNG_SEED)
    has_template = template_available(strip.bits, strip.tdi)
    fpn = load_fpn_template(strip.bits, strip.tdi) if has_template else None

    n = min(N_ACCEPT_TO_RENDER, len(accepted))
    if n == 0:
        return
    fig, axes = plt.subplots(n, 4, figsize=(13, 3.2 * n))
    if n == 1:
        axes = axes[None, :]

    for i, (patch_u8, r0, c0, stats) in enumerate(accepted[:n]):
        clean = patch_u8.astype(np.float32)
        dim = clean * ALPHA_INSPECT
        if has_template:
            tmpl_n = fpn.n_cols
            patch_col_offset = min(c0, tmpl_n - PATCH_SIZE)
            noisy = nm.inject_noise(
                dim, bits_selection=strip.bits, tdi_stages=strip.tdi,
                rng=rng, fpn_template=fpn, col_offset=patch_col_offset,
            )
            sigma = _sigma_physics_dn(dim, strip.bits, fpn, patch_col_offset)
        else:
            noisy = nm.inject_noise(
                dim, bits_selection=strip.bits, tdi_stages=strip.tdi,
                rng=rng, fpn_template=nm._NO_FPN, prnu_frac=0,
            )
            sigma = np.full_like(dim, np.nan)

        vmax_dim = max(1.0, float(dim.max()))
        axes[i, 0].imshow(clean, cmap="gray", vmin=0, vmax=255)
        axes[i, 0].set_title(
            f"clean   r={r0} c={c0}\n"
            f"mean={clean.mean():.1f} DN  max={clean.max():.0f}  "
            f"sobel99={stats['sobel_99']:.0f}\n"
            f"[fixed scale 0–255]",
            fontsize=8,
        )
        axes[i, 1].imshow(dim, cmap="gray", vmin=0, vmax=vmax_dim)
        axes[i, 1].set_title(
            f"clean × α={ALPHA_INSPECT}\n"
            f"mean={dim.mean():.2f} DN  max={dim.max():.1f}\n"
            f"[auto-scaled 0–{vmax_dim:.0f}]",
            fontsize=8,
        )
        axes[i, 2].imshow(noisy, cmap="gray", vmin=0, vmax=vmax_dim)
        title3 = "noisy = inject_noise(clean × α)" if has_template \
            else "noisy (no FPN — template missing)"
        axes[i, 2].set_title(
            f"{title3}\n"
            f"mean={noisy.mean():.2f} DN  std={noisy.std():.2f}\n"
            f"[auto-scaled 0–{vmax_dim:.0f}]",
            fontsize=8,
        )
        im3 = axes[i, 3].imshow(sigma, cmap="viridis", vmin=0)
        axes[i, 3].set_title(
            f"σ_physics (DN)\n"
            f"mean={np.nanmean(sigma):.2f}  max={np.nanmax(sigma):.2f}",
            fontsize=8,
        )
        plt.colorbar(im3, ax=axes[i, 3], fraction=0.046, pad=0.04)
        for ax in axes[i]:
            ax.set_xticks([]); ax.set_yticks([])

    fig.suptitle(
        f"{strip.product_id}   [{strip.label}]   "
        f"{strip.bits}/TDI{strip.tdi}   sol_inc={strip.solar_incidence:.1f}°"
        + ("" if has_template else "   (no FPN template)")
        + "\nNote: cols 2–3 auto-scaled to their own range — see mean DN to compare actual brightness vs col 1.",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(output_path, dpi=110)
    plt.close(fig)


def _process_strip(strip: StripInfo, df: pd.DataFrame) -> pd.DataFrame:
    row = df[df["product_id"] == strip.product_id].iloc[0]
    img_path = Path(row["img_path"])
    arr = np.memmap(
        img_path, dtype=np.uint8, mode="r",
        shape=(strip.line_count, strip.sample_count),
    )
    positions = _grid_positions(strip)
    table_rows = []
    accepted: list[tuple[np.ndarray, int, int, dict]] = []
    for r0, c0 in positions:
        patch = np.asarray(arr[r0:r0 + PATCH_SIZE, c0:c0 + PATCH_SIZE])
        stats = compute_patch_stats(patch)
        passes, reason = check_patch(stats)
        table_rows.append({
            "product_id": strip.product_id,
            "row0": r0, "col0": c0,
            **stats.as_dict(),
            "passes": passes,
            "reject_reason": reason,
        })
        if passes:
            accepted.append((patch.copy(), r0, c0, stats.as_dict()))

    out_png = OUTPUT_DIR / f"inspect_{strip.product_id}.png"
    _render_png(strip, accepted, out_png)
    print(f"  -> {out_png.name}: {len(accepted)}/{len(positions)} accepted")
    return pd.DataFrame(table_rows)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    splits = json.loads(SPLITS_PATH.read_text())
    df = catalog.load()

    strips = _pick_strips(splits, df)
    all_rows: list[pd.DataFrame] = []
    for s in strips:
        print(f"strip {s.product_id} [{s.label}] {s.bits}/TDI{s.tdi} sol_inc={s.solar_incidence:.1f}°")
        all_rows.append(_process_strip(s, df))

    full = pd.concat(all_rows, ignore_index=True)
    csv_path = OUTPUT_DIR / "inspect_training_pairs.csv"
    full.to_csv(csv_path, index=False)
    print()
    print(f"wrote {csv_path.relative_to(REPO_ROOT)}")
    print()
    print("pass-rate summary:")
    print(full.groupby("product_id")["passes"].agg(["sum", "count"])
              .rename(columns={"sum": "accepted", "count": "candidates"}).to_string())


if __name__ == "__main__":
    main()
