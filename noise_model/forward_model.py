"""
Seeded OHRC forward-noise constants.

These are the published instrument numbers (Chowdhury et al. 2019,
Table 3) projected onto the per-mode gain table. They are the *seed*
values for the forward noise model — Gate-1 found that an in-flight
PTC fit on the archive products is structurally impossible (lag-1
autocorr ≈ 0.81; see docs/GATE_1_FINDING.md §3), so these constants
play the role that a measured PTC slope would normally play.

Quantities
----------
NATIVE_GAIN_DN_PER_E
    The detector's native conversion: 1023 DN / 26_600 e⁻ ≈ 0.0385.
    Derived from the 10-bit ADC range and the full-well capacity.

GAIN_DN_PER_E[mode]
    Per-encoding gain after the on-board bit-shift. The encoding takes
    8 of the 10 ADC bits; LSB keeps bits [0:8], MID keeps [1:9], MSB
    keeps [2:10]. The effective gain ratio is *structurally* exactly
    4 : 2 : 1, because each upshift by 1 bit doubles the e⁻-per-DN.

READ_NOISE_E
    System-level read noise in electrons — Chowdhury 2019, Table 3.
    In DN this translates to roughly 1.6, 0.8, 0.4 DN for the three
    encodings (= READ_NOISE_E * GAIN_DN_PER_E[mode]).

FULL_WELL_E
    Saturation charge per pixel (single-stage equivalent) — used to
    derive NATIVE_GAIN_DN_PER_E.

The dark current, per-column bias profile, and residual σ_FPN(c) are
measured per-mode from data in Steps 1b.2 – 1b.4 and stored alongside
these constants in the eventual per-mode `noise_params` dict.

See docs/PROJECT_ROADMAP.md Phase 1b and docs/GATE_1_FINDING.md §10
and §12 for the full provenance and rationale.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np

from .fpn_template import (
    FPNTemplate,
    load_fpn_template,
    template_available,
)

ADC_MAX_DN: int = 1023
FULL_WELL_E: int = 26_600
READ_NOISE_E: float = 40.0

# Effective additive noise floor (electrons), back-calculated by inverting
# the published SNR=100 at reference radiance (Chowdhury 2020, Table 7 / Fig. 9).
# Lumps read + dark + TDI-transfer + bias-residual into one number — the
# decomposed pieces are filed as deferred (see NOISE_MODEL.md §6).
SIGMA_FLOOR_EFF_E: float = 95.0

# 10-bit quantisation noise, in electrons. Uniform distribution → σ = LSB/√12.
SIGMA_Q_E: float = (FULL_WELL_E / ADC_MAX_DN) / math.sqrt(12.0)

# Photo-response non-uniformity fraction, back-out from SNR=140 at saturation.
PRNU_FRAC: float = 0.003

# Saturation and reference radiance (mW / cm² / sr / µm) at 256 TDI.
SAT_RADIANCE: float = 0.8
REF_RADIANCE: float = 0.5

NATIVE_GAIN_DN_PER_E: float = ADC_MAX_DN / FULL_WELL_E

GAIN_DN_PER_E: dict[str, float] = {
    "lsb": NATIVE_GAIN_DN_PER_E,
    "mid": NATIVE_GAIN_DN_PER_E / 2.0,
    "msb": NATIVE_GAIN_DN_PER_E / 4.0,
}


def gain_dn_per_e(bits_selection: str) -> float:
    """Return DN-per-electron gain for an OHRC bits_selection mode."""
    key = bits_selection.lower()
    if key not in GAIN_DN_PER_E:
        raise ValueError(
            f"unknown bits_selection {bits_selection!r}; "
            f"expected one of {sorted(GAIN_DN_PER_E)}"
        )
    return GAIN_DN_PER_E[key]


def read_noise_dn(bits_selection: str) -> float:
    """Lab read noise in DN for the given encoding mode."""
    return READ_NOISE_E * gain_dn_per_e(bits_selection)


def floor_noise_dn(bits_selection: str) -> float:
    """Effective additive noise floor in DN for the given encoding mode."""
    return SIGMA_FLOOR_EFF_E * gain_dn_per_e(bits_selection)


def inject_noise(
    reference_dn: np.ndarray,
    bits_selection: str,
    tdi_stages: int,
    rng: np.random.Generator,
    *,
    fpn_template: Optional[FPNTemplate] = None,
    prnu_frac: float = PRNU_FRAC,
    col_offset: Optional[int] = None,
    clip: bool = True,
) -> np.ndarray:
    """
    Apply the OHRC forward noise model to a less-noisy reference patch.

    Per NOISE_MODEL.md §3, the model has both per-pixel and per-column terms:

      Per-pixel  (in electrons):
          g · sqrt(S) · ε_shot      shot noise
          g · σ_floor · ε_floor     lumped additive floor
      Per-column  (in DN, applied as a fixed offset/draw across all rows):
          b(c)                      measured bias_profile (deterministic)
          σ_FPN(c) · η(c)           measured per-column variability
      Per-pixel multiplicative  (bright-end only):
          PRNU · S · ζ_pix          residual photo-response non-uniformity

    Parameters
    ----------
    reference_dn
        Float array of reference signal in DN, in the same `bits_selection`
        encoding as the simulated output.
    bits_selection
        OHRC encoding of the simulated observation ("lsb" / "mid" / "msb").
        Gain is chosen from this — pass per-strip from the catalog.
    tdi_stages
        Number of TDI stages (64 / 128 / 256).
    rng
        Numpy default_rng for reproducibility.
    fpn_template
        Pre-loaded FPN template for this (mode, TDI). If None (default), the
        template is loaded from the canonical location; pass an explicit
        template to override, or set to a `_NO_FPN` sentinel to disable. If
        no template exists for the requested mode the call raises; pass
        `prnu_frac=0, fpn_template=_NO_FPN` if you really want a bare model.
    prnu_frac
        Multiplicative PRNU fraction (bright-end). Default PRNU_FRAC = 0.003.
    col_offset
        Column index in the full 12 000-wide detector that this patch starts
        at. If None and the patch is narrower than the template, a random
        offset is drawn (different "horizontal slice" of the detector per
        sample). If the patch matches template width, the full template is
        used. This makes the same patch see *different column biases* across
        training samples while keeping each sample's FPN spatially coherent.
    clip
        Clip the returned DN array to [0, 255] (archived OHRC is UnsignedByte).

    Returns
    -------
    Noisy DN array, same shape and dtype-promoted to float32.
    """
    if tdi_stages <= 0:
        raise ValueError(f"tdi_stages must be > 0; got {tdi_stages}")
    g = gain_dn_per_e(bits_selection)
    ref_dn = np.asarray(reference_dn, dtype=np.float32)
    n_rows, n_cols = ref_dn.shape[-2], ref_dn.shape[-1]

    # Resolve FPN template — load on demand unless explicitly disabled.
    use_fpn = fpn_template is not _NO_FPN
    if use_fpn and fpn_template is None:
        fpn_template = load_fpn_template(bits_selection, tdi_stages)

    # --- Per-pixel signal-domain noise (in electrons) ---
    signal_e = np.maximum(ref_dn, 0.0) / g
    shot_e = rng.normal(0.0, np.sqrt(np.maximum(signal_e, 0.0)))
    floor_e = rng.normal(0.0, SIGMA_FLOOR_EFF_E, size=ref_dn.shape)
    noisy_e = signal_e + shot_e + floor_e
    noisy_dn = g * noisy_e

    # --- Per-pixel multiplicative PRNU (matters only at bright signal) ---
    if prnu_frac > 0:
        prnu = rng.normal(0.0, prnu_frac, size=ref_dn.shape).astype(np.float32)
        noisy_dn = noisy_dn + prnu * ref_dn

    # --- Per-column structural FPN (DN-domain, constant across rows) ---
    if use_fpn:
        tmpl_n = fpn_template.n_cols
        if n_cols > tmpl_n:
            raise ValueError(
                f"patch width {n_cols} > template width {tmpl_n} "
                f"for ({bits_selection}, TDI{tdi_stages})"
            )
        if col_offset is None:
            col_offset = int(rng.integers(0, max(1, tmpl_n - n_cols + 1)))
        sl = slice(col_offset, col_offset + n_cols)
        bias_col = fpn_template.bias_profile[sl].astype(np.float32)
        sigma_fpn_col = fpn_template.sigma_fpn[sl].astype(np.float32)
        fpn_draw = rng.normal(0.0, 1.0, size=n_cols).astype(np.float32) * sigma_fpn_col
        col_offset_dn = bias_col + fpn_draw  # shape (n_cols,)
        noisy_dn = noisy_dn + col_offset_dn  # broadcast across rows

    if clip:
        noisy_dn = np.clip(noisy_dn, 0.0, 255.0)
    return noisy_dn.astype(np.float32)


# Sentinel to explicitly disable measured-FPN injection.
class _NoFPNType:
    pass


_NO_FPN = _NoFPNType()
