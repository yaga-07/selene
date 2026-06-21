"""
Two sanity PNGs after Gate 2 closes:

  1. analysis/_outputs/snr_predicted_vs_published.png
       Predicted SNR curve over radiance with the two published points marked.

  2. analysis/_outputs/inject_noise_examples.png
       (reference, noisy) pairs on a real OHRC patch at descending α scales,
       per encoding mode. Visual check that the injector behaves.

Both are diagnostic only — the *test* of correctness lives in
tests/test_snr_validation.py.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import catalog  # noqa: E402
import noise_model as nm  # noqa: E402

OUT_DIR = REPO / "analysis" / "_outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def plot_snr_curve() -> Path:
    radiance = np.linspace(0.005, nm.SAT_RADIANCE * 1.05, 400)
    snr_curve = nm.snr(radiance)

    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    ax.plot(radiance, snr_curve, lw=1.6, color="tab:blue", label="seeded model")
    for p in nm.PUBLISHED_POINTS:
        pred = float(nm.snr(p.radiance))
        ax.scatter([p.radiance], [p.snr], s=70, color="black", zorder=5,
                   label=f"published {p.name} (SNR={p.snr:.0f})")
        ax.annotate(
            f"pred {pred:.1f}\n({(pred - p.snr) / p.snr * 100:+.1f}%)",
            xy=(p.radiance, p.snr), xytext=(8, -22),
            textcoords="offset points", fontsize=9,
        )
    ax.set_xlabel("radiance (mW / cm² / sr / µm)")
    ax.set_ylabel("SNR @ TDI256")
    ax.set_title("OHRC SNR — seeded predictor vs Chowdhury 2020 (Table 7 / Fig. 9)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="lower right")
    fig.tight_layout()
    out = OUT_DIR / "snr_predicted_vs_published.png"
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return out


def plot_inject_examples() -> Path:
    """Per-mode comparison plus a per-mode FPN-on vs FPN-off side-by-side."""
    df = catalog.load()
    pick = (
        df[(df["bits_selection"] == "lsb") & (df["solar_incidence"] < 80)]
        .sort_values("sample_count", ascending=False)
        .iloc[0]
    )
    arr = np.memmap(
        Path(pick["img_path"]),
        dtype=np.uint8,
        mode="r",
        shape=(int(pick["line_count"]), int(pick["sample_count"])),
    )
    row_means = arr[::100, :].mean(axis=1)
    bright_idx = int(np.argmax(row_means))
    r0 = max(0, bright_idx * 100 - 128)
    c0 = arr.shape[1] // 2 - 128
    ref = arr[r0:r0 + 256, c0:c0 + 256].astype(np.float32)

    modes = [("lsb", 64), ("lsb", 128), ("msb", 64), ("msb", 128)]
    alphas = [1.0, 0.5, 0.25, 0.1]

    # --- main grid: rows = α, cols = (mode, TDI) with measured FPN ---
    fig, axes = plt.subplots(len(alphas), len(modes), figsize=(11, 2.6 * len(alphas)))
    for i, alpha in enumerate(alphas):
        dim_ref = ref * alpha
        for j, (mode, n_tdi) in enumerate(modes):
            noisy = nm.inject_noise(
                dim_ref, bits_selection=mode, tdi_stages=n_tdi,
                rng=np.random.default_rng(1000 * i + j),
            )
            ax = axes[i, j]
            ax.imshow(noisy, cmap="gray", vmin=0, vmax=255)
            ax.set_title(
                f"α={alpha:g}  {mode}  TDI{n_tdi}\n"
                f"μ={noisy.mean():.1f}  σ={noisy.std():.1f}",
                fontsize=8,
            )
            ax.axis("off")
        axes[i, 0].set_ylabel(f"ref·α   μ={dim_ref.mean():.1f}", fontsize=8)
    fig.suptitle(
        f"inject_noise examples — measured FPN  —  ref: {pick['product_id']}",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = OUT_DIR / "inject_noise_examples.png"
    fig.savefig(out, dpi=110)
    plt.close(fig)

    # --- FPN-on vs FPN-off on a FLAT reference at full detector width ---
    # Flat reference removes scene confusion; full width captures the entire
    # bias profile including the periodic spikes seen in bias_profile_lsb_TDI64.png.
    cmp_mode, cmp_tdi = "lsb", 64
    flat_ref = np.full((512, 12000), 30.0, dtype=np.float32)
    seed = 99
    noisy_with = nm.inject_noise(
        flat_ref, bits_selection=cmp_mode, tdi_stages=cmp_tdi,
        rng=np.random.default_rng(seed), clip=False,
    )
    noisy_without = nm.inject_noise(
        flat_ref, bits_selection=cmp_mode, tdi_stages=cmp_tdi,
        rng=np.random.default_rng(seed), fpn_template=nm._NO_FPN, clip=False,
    )
    diff = noisy_with - noisy_without
    col_mean_with = noisy_with.mean(axis=0)
    col_mean_without = noisy_without.mean(axis=0)

    fig2, ax2 = plt.subplots(4, 1, figsize=(14, 9), gridspec_kw={"height_ratios": [2, 2, 2, 3]})
    vmin, vmax = 20, 50
    ax2[0].imshow(noisy_without, cmap="gray", vmin=vmin, vmax=vmax, aspect="auto")
    ax2[0].set_title(
        f"injector OFF-FPN  flat ref μ=30  ({cmp_mode}, TDI{cmp_tdi})  "
        f"σ_pix={noisy_without.std():.2f} DN",
        fontsize=10,
    )
    ax2[0].axis("off")
    ax2[1].imshow(noisy_with, cmap="gray", vmin=vmin, vmax=vmax, aspect="auto")
    ax2[1].set_title(
        f"injector ON-FPN  flat ref μ=30  ({cmp_mode}, TDI{cmp_tdi})  "
        f"σ_pix={noisy_with.std():.2f} DN  ← vertical stripes should appear",
        fontsize=10,
    )
    ax2[1].axis("off")
    ax2[2].imshow(diff, cmap="RdBu_r", vmin=-15, vmax=15, aspect="auto")
    ax2[2].set_title(
        f"difference (ON − OFF)  isolates the FPN contribution  "
        f"σ_diff_col_means={diff.mean(axis=0).std():.2f} DN",
        fontsize=10,
    )
    ax2[2].axis("off")
    ax2[3].plot(col_mean_without, lw=0.5, color="tab:orange",
                label=f"OFF  σ(col mean) = {col_mean_without.std():.2f} DN")
    ax2[3].plot(col_mean_with, lw=0.5, color="tab:blue",
                label=f"ON   σ(col mean) = {col_mean_with.std():.2f} DN")
    ax2[3].set_xlabel("column index c (0–11 999)")
    ax2[3].set_ylabel("column-mean DN")
    ax2[3].set_title("Per-column mean of the noisy patch (averaged over 512 rows)",
                     fontsize=10)
    ax2[3].grid(True, alpha=0.3)
    ax2[3].legend(loc="upper right", fontsize=9)
    fig2.tight_layout()
    cmp_out = OUT_DIR / "inject_noise_fpn_comparison.png"
    fig2.savefig(cmp_out, dpi=110)
    plt.close(fig2)
    print(f"wrote {cmp_out.relative_to(REPO)}")
    print(f"  col-mean σ — OFF: {col_mean_without.std():.3f} DN  "
          f"ON: {col_mean_with.std():.3f} DN  ratio: "
          f"{col_mean_with.std() / col_mean_without.std():.1f}×")

    return out


def main() -> int:
    snr_png = plot_snr_curve()
    print(f"wrote {snr_png.relative_to(REPO)}")
    inj_png = plot_inject_examples()
    print(f"wrote {inj_png.relative_to(REPO)}")
    print("\nresiduals (predicted vs published):")
    for name, info in nm.published_residuals().items():
        print(
            f"  {name:>10s}: predicted {info['predicted']:6.1f} | "
            f"published {info['published']:6.1f} | error {info['error_pct']:+5.1f}%"
        )
    print(f"\npasses Gate 2 (±15%): {nm.passes_gate_2()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
