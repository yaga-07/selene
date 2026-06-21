"""
SNR-vs-radiance predictor for the seeded OHRC noise model.

This is the only quantitative cross-check available given the Gate-1 finding
that PTC-from-data is blocked. The predictor combines the seeded constants
from `forward_model.py` per NOISE_MODEL.md §5, and the unit test in
`tests/test_snr_validation.py` asserts the published reference and
saturation points within ±15 % — the close criterion for Gate 2.

The signal model assumes radiometric linearity (Chowdhury 2020, Fig. 8):
saturation radiance corresponds to the full well, and signal scales
linearly between zero and saturation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .forward_model import (
    FULL_WELL_E,
    PRNU_FRAC,
    REF_RADIANCE,
    SAT_RADIANCE,
    SIGMA_FLOOR_EFF_E,
    SIGMA_Q_E,
)


@dataclass(frozen=True)
class PublishedPoint:
    """A published (radiance, SNR) point from Chowdhury 2020."""

    name: str
    radiance: float
    snr: float


PUBLISHED_POINTS: tuple[PublishedPoint, ...] = (
    PublishedPoint("reference", REF_RADIANCE, 100.0),
    PublishedPoint("saturation", SAT_RADIANCE, 140.0),
)


def signal_e(radiance: float | np.ndarray) -> float | np.ndarray:
    """Electrons at a given radiance, assuming linearity to saturation = FWC."""
    return FULL_WELL_E * (np.asarray(radiance) / SAT_RADIANCE)


def snr(radiance: float | np.ndarray) -> float | np.ndarray:
    """
    Predicted SNR per NOISE_MODEL.md §5:

        SNR = S / sqrt( S + σ_floor² + (PRNU·S)² + σ_q² )

    where the lumped floor σ_floor (~95 e⁻) replaces the lab read noise
    (40 e⁻) because the published SNR=100 at reference radiance implies a
    higher effective additive floor than read noise alone explains.
    """
    S = signal_e(radiance)
    var = (
        S                                # shot
        + SIGMA_FLOOR_EFF_E ** 2         # lumped additive floor
        + (PRNU_FRAC * S) ** 2           # PRNU at bright end
        + SIGMA_Q_E ** 2                 # quantisation
    )
    return S / np.sqrt(var)


def published_residuals() -> dict[str, dict[str, float]]:
    """Return {point_name: {predicted, published, error_pct}} for diagnostics."""
    out: dict[str, dict[str, float]] = {}
    for p in PUBLISHED_POINTS:
        pred = float(snr(p.radiance))
        out[p.name] = {
            "predicted": pred,
            "published": p.snr,
            "error_pct": 100.0 * (pred - p.snr) / p.snr,
        }
    return out


def passes_gate_2(tolerance_pct: float = 15.0) -> bool:
    """True iff |predicted − published| / published ≤ tolerance at every point."""
    res = published_residuals()
    return all(abs(v["error_pct"]) <= tolerance_pct for v in res.values())
