"""
Photon Transfer Curve (PTC) estimation for OHRC.

The Poisson-Gaussian forward model predicts a linear relation between
the variance and mean of pixel values in a flat scene:

    Var(I) = g · Mean(I) + g² · σ_read²

where g is the effective gain in DN/e⁻ and σ_read is the read noise in
electrons. Fitting this line over many flat patches across the dynamic
range recovers (g, σ_read).

Two OHRC-specific design choices:

1. **Variance is computed from adjacent-row differences, not raw patch
   variance.** OHRC has column-wise fixed-pattern noise (the destriper's
   target). Naive Var(patch) conflates that column FPN with the temporal
   noise we want; row-difference variance cancels the FPN because column
   bias is identical in both rows.

2. **The fit is per `(bits_selection, tdi_stages)` mode.** Bit-selection
   rescales the DN axis by an exact factor (LSB:MID:MSB = 1:2:4 e⁻/DN),
   so pooling across modes would average over a real physical parameter.
   The ratio of fitted gains across modes (expected 4:2:1) is a free
   sanity check on the fit.

Read noise note: σ_read ≈ 40 e⁻ (Chowdhury 2019 Table 3, system-level)
translates to ~1.6 DN on LSB, ~0.8 DN on MID, ~0.4 DN on MSB. The MSB
intercept is below quantization step; PTC on MSB recovers gain but not
read noise.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class PTCFit:
    """Result of a single-strip PTC fit."""
    g_eff: float                 # DN per electron
    sigma_read_dn: float         # read noise in DN units (g_eff · σ_read_e)
    sigma_read_e: float          # read noise in electrons
    r2: float                    # goodness of linear fit in the shot-noise regime
    n_patches: int               # patches retained after gradient rejection
    means: np.ndarray            # per-patch mean DN
    variances: np.ndarray        # per-patch variance (temporal-only via row-diffs)
    inlier_mask: np.ndarray      # RANSAC inlier mask (bool, len == n_patches)


def extract_ptc_patches(
    img: np.ndarray,
    *,
    patch_size: int = 32,
    grad_threshold: float = 5.0,
    stride: int | None = None,
    row_range: tuple[int, int] | None = None,
    max_patches: int | None = None,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Walk `img` in `patch_size × patch_size` patches, reject textured
    patches by max gradient magnitude, and return (means, variances)
    where variance is estimated from adjacent-row differences (cancels
    column fixed-pattern noise; the resulting variance is twice the
    temporal noise, so we divide by 2).

    Parameters
    ----------
    img : 2D ndarray of shape (lines, samples), dtype uint8
    patch_size : edge length of the patches
    grad_threshold : reject patches whose max |∇I| exceeds this
    stride : step between patches (default = patch_size, i.e. non-overlapping)
    row_range : (row_start, row_stop) — restrict scan to a contiguous band of rows
    max_patches : if set, randomly subsample this many flat patches after extraction
    rng : numpy Generator for subsampling

    Returns
    -------
    means, variances : 1D ndarrays, same length
    """
    if stride is None:
        stride = patch_size
    if row_range is None:
        r0, r1 = 0, img.shape[0]
    else:
        r0, r1 = row_range

    means: list[float] = []
    variances: list[float] = []

    for row in range(r0, r1 - patch_size + 1, stride):
        for col in range(0, img.shape[1] - patch_size + 1, stride):
            patch = img[row:row + patch_size, col:col + patch_size].astype(np.float32)
            # Gradient rejection: skip textured patches.
            gy, gx = np.gradient(patch)
            if np.max(np.sqrt(gx * gx + gy * gy)) > grad_threshold:
                continue
            # Variance from adjacent-row diffs cancels column FPN.
            diffs = patch[1:, :] - patch[:-1, :]
            means.append(float(patch.mean()))
            variances.append(float(diffs.var()) / 2.0)

    means_arr = np.asarray(means, dtype=np.float64)
    variances_arr = np.asarray(variances, dtype=np.float64)

    if max_patches is not None and len(means_arr) > max_patches:
        if rng is None:
            rng = np.random.default_rng(0)
        idx = rng.choice(len(means_arr), size=max_patches, replace=False)
        means_arr = means_arr[idx]
        variances_arr = variances_arr[idx]

    return means_arr, variances_arr


def row_diff_binned_variance(
    img: np.ndarray,
    *,
    row_range: tuple[int, int] | None = None,
    col_stride: int = 1,
    dn_bins: np.ndarray | None = None,
    min_samples_per_bin: int = 200,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Estimate PTC by binning *every* adjacent-row pixel pair by its mean DN
    and computing the variance of the row-differences within each bin.

    For each (row r, column c) and its neighbour (row r+1, column c):
        bin = round((img[r,c] + img[r+1,c]) / 2)
        diff = img[r,c] - img[r+1,c]
    The variance of the diffs within a bin gives 2× the temporal noise
    variance at that DN level. Column fixed-pattern cancels (both pixels
    share the same column bias) and so do vertical ramps.

    This estimator gives full DN-axis coverage by construction — every
    pixel pair contributes, regardless of texture. The cost is that
    spatial-scale texture in the row direction (i.e. real scene
    features) inflates the variance. We mitigate by also clipping the
    bin via a robust quantile of the diff distribution.

    Returns
    -------
    bin_centers : 1D array of mean-DN bin centers
    variances   : 1D array of per-bin temporal noise variance (DN²)
    counts      : 1D array of pixel pairs in each bin (after quantile clip)
    """
    if row_range is None:
        r0, r1 = 0, img.shape[0]
    else:
        r0, r1 = row_range
    if dn_bins is None:
        dn_bins = np.arange(0, 256, dtype=np.int32)  # integer DN bins

    upper = img[r0:r1 - 1, ::col_stride].astype(np.int32)
    lower = img[r0 + 1:r1, ::col_stride].astype(np.int32)
    diff = (upper - lower).ravel()
    mean = ((upper + lower) / 2.0).ravel()

    # Map mean to integer bin index
    bin_idx = np.clip(np.round(mean).astype(np.int32), dn_bins[0], dn_bins[-1])

    bin_centers: list[int] = []
    variances: list[float] = []
    counts: list[int] = []
    for b in dn_bins:
        sel = bin_idx == b
        n = int(sel.sum())
        if n < min_samples_per_bin:
            continue
        diffs_b = diff[sel].astype(np.float64)
        # Robust variance: clip at 3.5×MAD to suppress texture-induced outliers.
        med = float(np.median(diffs_b))
        mad = float(np.median(np.abs(diffs_b - med)))
        cap = 3.5 * mad * 1.4826 + 1.0   # +1 to give breathing room at low MAD
        keep = np.abs(diffs_b - med) < cap
        if keep.sum() < min_samples_per_bin:
            continue
        var_temporal = float(diffs_b[keep].var()) / 2.0
        bin_centers.append(int(b))
        variances.append(var_temporal)
        counts.append(int(keep.sum()))

    return (
        np.asarray(bin_centers, dtype=np.float64),
        np.asarray(variances, dtype=np.float64),
        np.asarray(counts, dtype=np.int64),
    )


def fit_ptc_ransac(
    means: np.ndarray,
    variances: np.ndarray,
    *,
    g_native_prior: float = 0.04,
    min_samples: int = 30,
    residual_threshold: float | None = None,
    random_state: int = 0,
) -> PTCFit:
    """
    Robust linear fit to Var(I) = a · Mean(I) + b, where a = g_eff,
    b = g_eff² · σ_read².

    RANSAC is used because real PTC scatter contains outliers from
    sub-threshold scene texture, cosmic rays, and bright pixels near
    saturation. The shot-noise regime (high mean) should dominate the
    fit; the intercept is sensitive to outliers, so a robust regressor
    is non-optional.

    Parameters
    ----------
    means, variances : 1D arrays from extract_ptc_patches
    g_native_prior : expected gain (DN/e⁻); used only to set a sensible
        default RANSAC residual threshold when one is not provided.
    min_samples : minimum patches per RANSAC iteration
    residual_threshold : RANSAC inlier threshold in variance units
    random_state : RNG seed for reproducibility

    Returns
    -------
    PTCFit with fitted parameters and inlier mask.
    """
    from sklearn.linear_model import RANSACRegressor

    if len(means) < min_samples:
        raise ValueError(
            f"too few patches ({len(means)}) for RANSAC with min_samples={min_samples}"
        )

    X = means.reshape(-1, 1)
    y = variances

    if residual_threshold is None:
        # Default: 3× MAD of variances around the median predicted line.
        # This is loose enough to not over-prune, tight enough to remove
        # patches that survived gradient rejection but still carry texture.
        residual_threshold = 3.0 * np.median(np.abs(y - np.median(y)))
        residual_threshold = max(residual_threshold, 0.5)

    ransac = RANSACRegressor(
        min_samples=min_samples,
        residual_threshold=residual_threshold,
        random_state=random_state,
    )
    ransac.fit(X, y)
    g_eff = float(ransac.estimator_.coef_[0])
    intercept = float(ransac.estimator_.intercept_)

    if g_eff <= 0:
        # Negative slope is non-physical; the fit failed.
        sigma_read_dn = float("nan")
        sigma_read_e = float("nan")
    else:
        sigma_read_dn = float(np.sqrt(max(intercept, 0.0)))
        sigma_read_e = sigma_read_dn / g_eff

    # R² over inliers
    inliers = ransac.inlier_mask_
    y_pred = g_eff * X[inliers, 0] + intercept
    ss_res = float(np.sum((y[inliers] - y_pred) ** 2))
    ss_tot = float(np.sum((y[inliers] - y[inliers].mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    return PTCFit(
        g_eff=g_eff,
        sigma_read_dn=sigma_read_dn,
        sigma_read_e=sigma_read_e,
        r2=r2,
        n_patches=int(len(means)),
        means=means,
        variances=variances,
        inlier_mask=inliers,
    )
