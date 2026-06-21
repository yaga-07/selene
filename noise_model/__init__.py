"""
selene/noise_model — OHRC Poisson-Gaussian forward noise model.

This package estimates and exposes the per-mode noise parameters
{g_eff (DN/e⁻), σ_read (e⁻), dark_current (e⁻/stage), bias_profile (DN)}
that drive both the training-data noise injection (Phase 2) and the
PG-NLL loss function (Phase 3).
"""

from .forward_model import (
    ADC_MAX_DN,
    FULL_WELL_E,
    GAIN_DN_PER_E,
    NATIVE_GAIN_DN_PER_E,
    PRNU_FRAC,
    READ_NOISE_E,
    REF_RADIANCE,
    SAT_RADIANCE,
    SIGMA_FLOOR_EFF_E,
    SIGMA_Q_E,
    _NO_FPN,
    floor_noise_dn,
    gain_dn_per_e,
    inject_noise,
    read_noise_dn,
)
from .fpn_template import (
    FPNTemplate,
    load_fpn_template,
    template_available,
)
from .ptc import (
    PTCFit,
    extract_ptc_patches,
    fit_ptc_ransac,
    row_diff_binned_variance,
)
from .snr_validation import (
    PUBLISHED_POINTS,
    PublishedPoint,
    passes_gate_2,
    published_residuals,
    signal_e,
    snr,
)

__all__ = [
    "ADC_MAX_DN",
    "FPNTemplate",
    "FULL_WELL_E",
    "GAIN_DN_PER_E",
    "NATIVE_GAIN_DN_PER_E",
    "PRNU_FRAC",
    "PTCFit",
    "PUBLISHED_POINTS",
    "PublishedPoint",
    "READ_NOISE_E",
    "REF_RADIANCE",
    "SAT_RADIANCE",
    "SIGMA_FLOOR_EFF_E",
    "SIGMA_Q_E",
    "_NO_FPN",
    "extract_ptc_patches",
    "fit_ptc_ransac",
    "floor_noise_dn",
    "gain_dn_per_e",
    "inject_noise",
    "load_fpn_template",
    "passes_gate_2",
    "published_residuals",
    "read_noise_dn",
    "row_diff_binned_variance",
    "signal_e",
    "snr",
    "template_available",
]
