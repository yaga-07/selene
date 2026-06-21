"""
Pilot PTC fit on a single LSB-encoded sunlit OHRC strip.

The point of this script is to confirm — before scaling to all modes
and all strips — that:

  - The PTC scatter on real OHRC data shows a clean linear shot-noise
    trend after our row-difference variance estimator.
  - The fitted gain lands near the expected ~0.04 DN/e⁻ for LSB encoding.
  - The fitted read noise lands near the expected ~1.6 DN (~40 e⁻).

If those three numbers land in the predicted band, the fitter is
trustworthy and we promote it. If they don't, we debug before scaling.

Pilot strip:
  ch2_ohr_nrp_20250516T1540191774_d_img_d18 — LSB, TDI64, solar_incidence ≈ 67°.
  Widest DN range LSB strip in the corpus; the cleanest gain + read-noise
  recovery candidate per the Gate-2 priors.

Usage (from repo root, with .venv activated):
    .venv/bin/python analysis/pilot_ptc.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import catalog  # noqa: E402
from noise_model.ptc import (  # noqa: E402
    extract_ptc_patches,
    fit_ptc_ransac,
    row_diff_binned_variance,
)

PILOT_PRODUCT_ID = "ch2_ohr_nrp_20250516T1540191774_d_img_d18"
OUT_DIR = REPO / "analysis"


# ---- Expected values from Chowdhury 2019 + bit-selection encoding ----------
# Full well 26.6 ke⁻, ADC 10-bit -> native gain ≈ 0.04 DN/e⁻.
# LSB encoding preserves the LSB scale, so g_lsb ≈ g_native.
# Read noise 40 e⁻ → ~1.6 DN on LSB.
EXPECTED_GAIN_LSB = 0.04          # DN / e⁻
EXPECTED_READ_NOISE_E = 40.0      # electrons
EXPECTED_READ_NOISE_DN_LSB = EXPECTED_GAIN_LSB * EXPECTED_READ_NOISE_E


def load_strip(product_id: str) -> tuple[np.ndarray, dict]:
    """Memmap the .img and return (image, meta-dict) for the named product."""
    df = catalog.load()
    row = df[df["product_id"] == product_id]
    if len(row) == 0:
        raise SystemExit(f"product_id {product_id!r} not in catalog index")
    r = row.iloc[0]
    img_path = Path(r["img_path"])
    if not img_path.exists():
        raise SystemExit(f"img file missing: {img_path}")
    expected = int(r["line_count"]) * int(r["sample_count"])
    actual = img_path.stat().st_size
    if actual != expected:
        raise SystemExit(
            f"size mismatch: file={actual}, expected={expected} "
            f"(lines × samples × uint8)"
        )
    arr = np.memmap(img_path, dtype=np.uint8, mode="r",
                    shape=(int(r["line_count"]), int(r["sample_count"])))
    meta = {
        "product_id": r["product_id"],
        "bits_selection": r["bits_selection"],
        "tdi_stages": r["tdi_stages"],
        "solar_incidence": float(r["solar_incidence"]),
        "spacecraft_altitude": float(r["spacecraft_altitude"]),
        "lines": int(r["line_count"]),
        "samples": int(r["sample_count"]),
    }
    return arr, meta


def main() -> int:
    arr, meta = load_strip(PILOT_PRODUCT_ID)
    print(f"loaded {meta['product_id']}")
    print(f"  bits_selection = {meta['bits_selection']}, "
          f"tdi = {meta['tdi_stages']}, "
          f"solar_incidence = {meta['solar_incidence']:.1f}°")
    print(f"  shape = {arr.shape}, size = {arr.nbytes/1e9:.2f} GB on disk")
    print(f"  DN min/median/max = {arr.min()}/{int(np.median(arr))}/{arr.max()}")

    # Limit the scan to a fast-but-representative subsample of the strip
    # (every product is ~1 GB; we don't need every row for a pilot).
    n_lines = arr.shape[0]
    band_size = min(20000, n_lines)            # ~20k lines × 12k samples ≈ 240 M pixels
    row_start = (n_lines - band_size) // 2     # middle band: avoid scan-start/end transients
    row_stop = row_start + band_size

    # Per-patch noise floor on |∇I|: for a flat patch with σ≈1.6 DN (LSB),
    # max(|∇I|) in 32×32 will be ~7 DN from noise alone. Threshold has to
    # clear that floor before it starts rejecting real scene texture.
    print(f"\nextracting patches in rows {row_start}..{row_stop} "
          f"({(row_stop - row_start) * arr.shape[1] / 1e6:.0f} Mpix)")

    # Sweep a few thresholds and report yields so we can see the texture
    # distribution before committing to one. Cheap with stride=64 on a sub-band.
    for thr in (5.0, 10.0, 20.0, 40.0):
        m, v = extract_ptc_patches(
            arr, patch_size=32, grad_threshold=thr, stride=128,
            row_range=(row_start, row_stop),
        )
        n_strict = int(np.sum(v < 50))      # 'noise-only' patches
        print(f"  grad_threshold={thr:>5.1f} → {len(m):>6d} patches, "
              f"{n_strict} with var<50 DN²")

    means, variances = extract_ptc_patches(
        arr,
        patch_size=32,
        grad_threshold=15.0,      # well above the noise floor, well below typical terrain texture
        stride=64,
        row_range=(row_start, row_stop),
        max_patches=20000,
        rng=np.random.default_rng(42),
    )
    print(f"\n  retained {len(means)} flat patches at grad_threshold=15")
    if len(means) < 30:
        print("  ERROR: not enough flat patches to fit. Inspect the strip first.")
        return 1

    # Diagnose dynamic-range coverage — the fit needs patches across the
    # DN axis, not just one mode.
    print(f"  patch mean DN:  "
          f"min={means.min():.1f}  median={np.median(means):.1f}  "
          f"max={means.max():.1f}  std={means.std():.1f}")
    print(f"  patch variance: "
          f"min={variances.min():.2f}  median={np.median(variances):.2f}  "
          f"max={variances.max():.2f}  std={variances.std():.2f}")
    # Histogram of means in coarse DN bins
    bins = [0, 10, 25, 50, 100, 150, 200, 255]
    hist, _ = np.histogram(means, bins=bins)
    print(f"  mean DN histogram (8 bins):")
    for lo, hi, n in zip(bins[:-1], bins[1:], hist):
        bar = "#" * min(60, int(60 * n / max(hist.max(), 1)))
        print(f"    [{lo:>3d}, {hi:>3d}): {n:>5d}  {bar}")

    # Secondary estimator: row-difference variance binned by DN.
    # Every adjacent-row pixel pair contributes, regardless of texture —
    # but texture-driven outliers within each bin are robustly clipped.
    # Gives full DN-axis coverage; complementary to the patch approach.
    print("\nbinned row-difference PTC (full DN axis)")
    bin_centers, bin_vars, bin_counts = row_diff_binned_variance(
        arr,
        row_range=(row_start, row_stop),
        col_stride=4,                  # 1/4 of columns = 3000 cols × 20k rows ≈ 60M pairs
        min_samples_per_bin=1000,
    )
    print(f"  {len(bin_centers)} DN bins populated "
          f"(range {int(bin_centers.min())}..{int(bin_centers.max())} DN)")
    print(f"  bin_counts: median={int(np.median(bin_counts))}, "
          f"min={int(bin_counts.min())}, max={int(bin_counts.max())}")

    # Fit on the binned values, weighted by sqrt(count) to emphasise
    # well-sampled bins.
    from sklearn.linear_model import LinearRegression
    weights = np.sqrt(bin_counts.astype(np.float64))
    lr = LinearRegression()
    lr.fit(bin_centers.reshape(-1, 1), bin_vars, sample_weight=weights)
    g_binned = float(lr.coef_[0])
    b_binned = float(lr.intercept_)
    sigma_read_dn_binned = float(np.sqrt(max(b_binned, 0.0)))
    sigma_read_e_binned = sigma_read_dn_binned / g_binned if g_binned > 0 else float("nan")
    ss_res = float(np.sum(weights * (bin_vars - lr.predict(bin_centers.reshape(-1, 1))) ** 2))
    ss_tot = float(np.sum(weights * (bin_vars - np.average(bin_vars, weights=weights)) ** 2))
    r2_binned = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    print(f"  fit: g_eff = {g_binned:.5f} DN/e⁻  "
          f"(expected ~{EXPECTED_GAIN_LSB:.3f}; "
          f"deviation = {(g_binned - EXPECTED_GAIN_LSB)/EXPECTED_GAIN_LSB*100:+.1f}%)")
    print(f"  fit: σ_read = {sigma_read_dn_binned:.3f} DN "
          f"= {sigma_read_e_binned:.1f} e⁻")
    print(f"  fit: R² (weighted) = {r2_binned:.3f}")

    fit = fit_ptc_ransac(means, variances, random_state=0)

    print(f"\nfit results (LSB pilot)")
    print(f"  g_eff       = {fit.g_eff:.5f} DN/e⁻   "
          f"(expected ~{EXPECTED_GAIN_LSB:.3f}; "
          f"deviation = {(fit.g_eff - EXPECTED_GAIN_LSB)/EXPECTED_GAIN_LSB*100:+.1f}%)")
    print(f"  σ_read      = {fit.sigma_read_dn:.3f} DN "
          f"= {fit.sigma_read_e:.1f} e⁻   "
          f"(expected ~{EXPECTED_READ_NOISE_DN_LSB:.2f} DN "
          f"= {EXPECTED_READ_NOISE_E:.0f} e⁻)")
    print(f"  R² (inliers)= {fit.r2:.3f}   "
          f"({fit.inlier_mask.sum()} / {fit.n_patches} patches inlier)")

    verdict_g = abs(fit.g_eff - EXPECTED_GAIN_LSB) / EXPECTED_GAIN_LSB < 0.25
    verdict_s = (abs(fit.sigma_read_e - EXPECTED_READ_NOISE_E) / EXPECTED_READ_NOISE_E < 0.25
                 and not np.isnan(fit.sigma_read_e))
    verdict_r = fit.r2 > 0.5
    print(f"\nverdict triplet: "
          f"gain {'PASS' if verdict_g else 'FAIL'} | "
          f"σ_read {'PASS' if verdict_s else 'FAIL'} | "
          f"R² {'PASS' if verdict_r else 'FAIL'}")

    # ---- Plot: two panels (patch-based vs binned row-diff) ----
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    ax = axes[0]
    ax.scatter(means[~fit.inlier_mask], variances[~fit.inlier_mask],
               s=4, alpha=0.3, color="grey", label=f"outliers ({(~fit.inlier_mask).sum()})")
    ax.scatter(means[fit.inlier_mask], variances[fit.inlier_mask],
               s=4, alpha=0.5, color="C0", label=f"inliers ({fit.inlier_mask.sum()})")
    xs = np.linspace(0, 255, 200)
    ys_fit = fit.g_eff * xs + (fit.sigma_read_dn ** 2)
    ax.plot(xs, ys_fit, "C3-", lw=2,
            label=f"fit: g={fit.g_eff:.4f}, σ={fit.sigma_read_dn:.2f}")
    ys_expected = EXPECTED_GAIN_LSB * xs + EXPECTED_READ_NOISE_DN_LSB ** 2
    ax.plot(xs, ys_expected, "C2--", lw=1.5, alpha=0.7, label="expected (Chowdhury 2019)")
    ax.set_xlim(0, 255)
    ax.set_xlabel("Mean DN")
    ax.set_ylabel("Var DN²")
    ax.set_title(f"Patch-based PTC (gradient-rejected)\n"
                 f"R²={fit.r2:.3f}, n={fit.n_patches}")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(alpha=0.3)

    ax = axes[1]
    sc = ax.scatter(bin_centers, bin_vars, c=np.log10(bin_counts),
                    cmap="viridis", s=30, edgecolors="black", linewidths=0.3)
    ys_binned = g_binned * xs + b_binned
    ax.plot(xs, ys_binned, "C3-", lw=2,
            label=f"fit: g={g_binned:.4f}, σ={sigma_read_dn_binned:.2f}")
    ax.plot(xs, ys_expected, "C2--", lw=1.5, alpha=0.7, label="expected (Chowdhury 2019)")
    ax.set_xlim(0, 255)
    ax.set_xlabel("Mean DN")
    ax.set_ylabel("Var DN²  (row-diff temporal estimator)")
    ax.set_title(f"Binned row-difference PTC (all rows)\n"
                 f"R²={r2_binned:.3f}, {len(bin_centers)} DN bins")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(alpha=0.3)
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label("log10(pixel-pair count)")

    fig.suptitle(
        f"PTC pilot — {meta['product_id']}  "
        f"({meta['bits_selection']}, {meta['tdi_stages']}, "
        f"solar_inc {meta['solar_incidence']:.1f}°)",
        fontsize=11,
    )
    fig.tight_layout()
    out = OUT_DIR / f"pilot_ptc_{meta['product_id']}.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
