"""
Lag-1 autocorrelation diagnostic on flat OHRC patches.

Truly raw detector data has near-zero spatial autocorrelation in flat
regions — adjacent pixels are independent photon counts plus
independent readout noise. After ground processing that involves any
resampling, interpolation, MTF restoration, or filtering, adjacent
pixels become correlated and lag-1 autocorrelation jumps to 0.2–0.5+.

This test is the decisive way to find out whether the OHRC raw products
(`nrp`) actually deliver raw pixels or whether they have been
geometrically resampled before archiving. It's independent of the
row-difference estimators used in the PTC fit — those estimators cancel
column FPN and vertical ramps, which can mask processing if the
processing is correlated row-wise.

Outputs:
  - Lag-1 autocorrelation coefficient in the row direction (line axis)
  - Lag-1 autocorrelation coefficient in the column direction (sample axis)
  - Raw patch variance for comparison
  - A printed 16×16 block of integer pixel values for visual inspection
  - The PDS4 `processing_level` field and anything resembling resampling
    metadata from the XML.

Pilot strip:
  ch2_ohr_nrp_20250516T1540191774_d_img_d18 — LSB, TDI64, solar_inc 67°.
"""

from __future__ import annotations

import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import catalog  # noqa: E402

PRODUCT_ID = "ch2_ohr_nrp_20250516T1540191774_d_img_d18"


def lag1_autocorr(arr: np.ndarray, axis: int) -> float:
    """
    Compute the normalized lag-1 autocorrelation of `arr` along `axis`.

    Definition: cov(arr[..., i], arr[..., i+1]) / var(arr).

    No row-differencing involved — this is a *direct* second-order
    statistic, computed independent of the PTC-fit estimators.
    """
    a = arr.astype(np.float64)
    a = a - a.mean()
    if axis == 0:
        num = float(np.mean(a[:-1, :] * a[1:, :]))
    elif axis == 1:
        num = float(np.mean(a[:, :-1] * a[:, 1:]))
    else:
        raise ValueError(f"axis must be 0 or 1, got {axis}")
    den = float(a.var())
    return num / den if den > 0 else float("nan")


def find_flat_patches(arr: np.ndarray,
                       target_mean: float,
                       patch_size: int = 64,
                       n_patches: int = 5,
                       grad_threshold: float = 8.0,
                       rng_seed: int = 0) -> list[tuple[int, int]]:
    """
    Walk a random subset of patches; keep those near `target_mean` and
    with max-gradient under threshold (i.e. flat enough to be useful
    for autocorrelation testing).
    """
    rng = np.random.default_rng(rng_seed)
    n_lines, n_samples = arr.shape
    found: list[tuple[int, int]] = []
    tried = 0
    while len(found) < n_patches and tried < 5000:
        tried += 1
        r = int(rng.integers(0, n_lines - patch_size))
        c = int(rng.integers(0, n_samples - patch_size))
        p = arr[r:r + patch_size, c:c + patch_size].astype(np.float32)
        if abs(p.mean() - target_mean) > 15:
            continue
        gy, gx = np.gradient(p)
        if np.max(np.sqrt(gx * gx + gy * gy)) > grad_threshold:
            continue
        found.append((r, c))
    return found


def main() -> int:
    df = catalog.load()
    row = df[df["product_id"] == PRODUCT_ID].iloc[0]
    img_path = Path(row["img_path"])
    lines, samples = int(row["line_count"]), int(row["sample_count"])
    arr = np.memmap(img_path, dtype=np.uint8, mode="r", shape=(lines, samples))
    print(f"loaded {row['product_id']}  shape={arr.shape}")
    print(f"  bits_selection={row['bits_selection']}  tdi_stages={row['tdi_stages']}  "
          f"solar_incidence={row['solar_incidence']:.1f}°")

    # --- check XML processing_level + any resampling-ish fields ---
    print("\n=== PDS4 label hints ===")
    xml_path = Path(row["xml_path"])
    tree = ET.parse(xml_path)
    root = tree.getroot()
    flag_tags = (
        "processing_level", "processing_method", "resampling",
        "interpolation_method", "encoding", "compression",
        "radiometric_correction", "geometric_correction",
        "reference_data_used",
    )
    for el in root.iter():
        local = el.tag.split("}", 1)[-1]
        if local in flag_tags and el.text and el.text.strip():
            print(f"  {local:>25s}: {el.text.strip()}")

    # --- find a few flat patches near mean DN 90 ---
    print("\n=== flat-patch autocorrelation ===")
    coords = find_flat_patches(arr, target_mean=90.0, patch_size=64,
                               n_patches=5, grad_threshold=8.0)
    if not coords:
        print("  no flat patches found — try a different target_mean")
        return 1

    print(f"{'patch':>10s}  {'mean':>6s}  {'std':>5s}  "
          f"{'autocorr_row':>13s}  {'autocorr_col':>13s}")
    for r, c in coords:
        p = arr[r:r + 64, c:c + 64]
        m = float(p.mean())
        s = float(p.std())
        a_row = lag1_autocorr(p, axis=0)
        a_col = lag1_autocorr(p, axis=1)
        print(f"  ({r:>5d},{c:>5d})  {m:>6.2f}  {s:>5.2f}  "
              f"{a_row:>13.4f}  {a_col:>13.4f}")

    # --- raw integer block dump (16x16) ---
    print("\n=== raw 16×16 integer dump (first flat patch) ===")
    r, c = coords[0]
    block = arr[r:r + 16, c:c + 16]
    for line in block:
        print("  " + " ".join(f"{int(v):>3d}" for v in line))

    # --- summary verdict ---
    a_row_med = float(np.median([
        lag1_autocorr(arr[r:r + 64, c:c + 64], axis=0) for r, c in coords
    ]))
    a_col_med = float(np.median([
        lag1_autocorr(arr[r:r + 64, c:c + 64], axis=1) for r, c in coords
    ]))
    print(f"\n=== verdict ===")
    print(f"  median lag-1 autocorr  row: {a_row_med:.3f}")
    print(f"  median lag-1 autocorr  col: {a_col_med:.3f}")
    print(f"  interpretation:")
    if abs(a_row_med) < 0.1 and abs(a_col_med) < 0.1:
        print(f"    → Both ≈ 0: data plausibly raw; PTC fits should work.")
    elif a_row_med > 0.1 or a_col_med > 0.1:
        print(f"    → Significantly > 0: pixels are correlated. The data has been")
        print(f"      processed (most likely interpolated/resampled) before archive.")
        print(f"      Shot noise is being smoothed out; PTC-from-data cannot recover")
        print(f"      gain. Seed the forward noise model from published constants.")
    else:
        print(f"    → Inconclusive (near zero but with some asymmetry). Re-run on")
        print(f"      more strips / different DN levels before deciding.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
