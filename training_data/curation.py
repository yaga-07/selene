"""Per-patch curation rules for the training reference pool (§2.1).

A candidate 256×256 patch passes if all of:
  - ``mean_dn >= MEAN_DN_FLOOR``                          (absolute signal floor)
  - if ``mean_dn < MEAN_DN_TEXTURELESS``: ``sobel_99 >= SOBEL_99_MIN``
    (a dim patch must still carry terrain texture; otherwise it's a
    flat-shadow / pure-FPN sample that's useless as a clean reference)
  - ``frac_zero < FRAC_ZERO_MAX``                         (no big data-void edge)
  - ``frac_sat < FRAC_SAT_MAX``                           (no saturated regions)

Calibrated 2026-06-22 from `analysis/inspect_training_pairs.py` after
visual inspection: the polar-LSB mid-shadow patches at mean ~9 DN with
sobel_99 ~10 are dominated by per-column FPN with no scene structure
and should not enter the reference pool. Bright equatorial patches all
pass; bright polar rim patches all pass.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import sobel

PATCH_SIZE = 256

MEAN_DN_FLOOR = 5.0
MEAN_DN_TEXTURELESS = 20.0
SOBEL_99_MIN = 25.0
FRAC_ZERO_MAX = 0.30
FRAC_SAT_MAX = 0.10

CURATION_PARAMS = {
    "patch_size": PATCH_SIZE,
    "mean_dn_floor": MEAN_DN_FLOOR,
    "mean_dn_textureless": MEAN_DN_TEXTURELESS,
    "sobel_99_min": SOBEL_99_MIN,
    "frac_zero_max": FRAC_ZERO_MAX,
    "frac_sat_max": FRAC_SAT_MAX,
}


@dataclass(frozen=True)
class PatchStats:
    mean_dn: float
    frac_zero: float
    frac_sat: float
    sobel_99: float

    def as_dict(self) -> dict:
        return {
            "mean_dn": self.mean_dn,
            "frac_zero": self.frac_zero,
            "frac_sat": self.frac_sat,
            "sobel_99": self.sobel_99,
        }


def sobel_magnitude(patch: np.ndarray) -> np.ndarray:
    p = patch.astype(np.float32)
    sx = sobel(p, axis=1)
    sy = sobel(p, axis=0)
    return np.sqrt(sx * sx + sy * sy)


def compute_patch_stats(patch: np.ndarray) -> PatchStats:
    return PatchStats(
        mean_dn=float(patch.mean()),
        frac_zero=float((patch == 0).mean()),
        frac_sat=float((patch >= 250).mean()),
        sobel_99=float(np.percentile(sobel_magnitude(patch), 99)),
    )


def check_patch(stats: PatchStats) -> tuple[bool, str]:
    reasons: list[str] = []
    if stats.mean_dn < MEAN_DN_FLOOR:
        reasons.append(f"mean_dn<{MEAN_DN_FLOOR}")
    elif (stats.mean_dn < MEAN_DN_TEXTURELESS
          and stats.sobel_99 < SOBEL_99_MIN):
        reasons.append(
            f"flat_shadow(mean<{MEAN_DN_TEXTURELESS}&sobel99<{SOBEL_99_MIN})"
        )
    if stats.frac_zero >= FRAC_ZERO_MAX:
        reasons.append(f"frac_zero>={FRAC_ZERO_MAX}")
    if stats.frac_sat >= FRAC_SAT_MAX:
        reasons.append(f"frac_sat>={FRAC_SAT_MAX}")
    return (len(reasons) == 0), ",".join(reasons)
