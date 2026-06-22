"""
Gate-2 close: SNR validator must reproduce the published OHRC SNR-vs-radiance
points (Chowdhury 2020, Table 7 / Fig. 9) within ±15 %, and the per-encoding
gain ratio must be exactly 4 : 2 : 1.

See docs/NOISE_MODEL.md §5 for the validator definition, §8 for the
acceptance criteria.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

import noise_model as nm


TOLERANCE_PCT = 15.0


@pytest.mark.parametrize("point", nm.PUBLISHED_POINTS, ids=lambda p: p.name)
def test_published_snr_points_within_tolerance(point: nm.PublishedPoint) -> None:
    """Each published (radiance, SNR) point must match the predictor within 15 %."""
    predicted = float(nm.snr(point.radiance))
    err_pct = 100.0 * (predicted - point.snr) / point.snr
    assert abs(err_pct) <= TOLERANCE_PCT, (
        f"{point.name}: predicted SNR {predicted:.1f} vs published {point.snr:.1f} "
        f"({err_pct:+.1f}% — exceeds ±{TOLERANCE_PCT}%)"
    )


def test_gate_2_closes() -> None:
    """passes_gate_2() agrees with the per-point assertions."""
    assert nm.passes_gate_2(tolerance_pct=TOLERANCE_PCT)


def test_gain_ratio_is_exactly_4_2_1() -> None:
    """Per-encoding gain ratio (lsb : mid : msb) must be exactly 4 : 2 : 1."""
    g_lsb = nm.gain_dn_per_e("lsb")
    g_mid = nm.gain_dn_per_e("mid")
    g_msb = nm.gain_dn_per_e("msb")
    assert g_lsb / g_msb == pytest.approx(4.0, rel=0, abs=1e-12)
    assert g_mid / g_msb == pytest.approx(2.0, rel=0, abs=1e-12)
    assert g_lsb / g_mid == pytest.approx(2.0, rel=0, abs=1e-12)


def test_native_gain_matches_published_constants() -> None:
    """Native gain = ADC_MAX / FULL_WELL = 1023 / 26_600."""
    expected = 1023.0 / 26_600.0
    assert nm.NATIVE_GAIN_DN_PER_E == pytest.approx(expected, rel=1e-12)
    assert nm.gain_dn_per_e("lsb") == pytest.approx(expected, rel=1e-12)


def test_effective_floor_inverts_reference_snr() -> None:
    """σ_floor ≈ 95 e⁻ must invert the SNR=100 reference within tolerance."""
    S_ref = nm.signal_e(nm.REF_RADIANCE)
    snr_ref_no_extras = S_ref / math.sqrt(S_ref + nm.SIGMA_FLOOR_EFF_E ** 2)
    assert snr_ref_no_extras == pytest.approx(100.0, abs=10.0)


def test_inject_noise_unknown_bits_raises() -> None:
    """Forward injector must reject unknown encoding modes."""
    rng = np.random.default_rng(0)
    ref = np.full((16, 64), 100.0, dtype=np.float32)
    with pytest.raises(ValueError, match="bits_selection"):
        nm.inject_noise(ref, "foo", tdi_stages=64, rng=rng,
                        fpn_template=nm._NO_FPN)


def test_inject_noise_missing_template_raises() -> None:
    """Asking for a (mode, TDI) without a measured template must fail loudly."""
    rng = np.random.default_rng(0)
    ref = np.full((16, 64), 50.0, dtype=np.float32)
    with pytest.raises(FileNotFoundError, match="no FPN template"):
        nm.inject_noise(ref, "mid", tdi_stages=64, rng=rng)


def test_inject_noise_increases_variance() -> None:
    """Injected noise must exceed the reference's intrinsic variance on a flat patch."""
    rng = np.random.default_rng(42)
    ref = np.full((128, 256), 50.0, dtype=np.float32)
    noisy = nm.inject_noise(ref, "lsb", tdi_stages=64, rng=rng,
                            fpn_template=nm._NO_FPN)
    assert noisy.shape == ref.shape
    assert noisy.dtype == np.float32
    assert noisy.std() > ref.std() + 0.5


def test_inject_noise_respects_bits_selection() -> None:
    """A noisier LSB patch must have larger DN-domain σ than an MSB patch
    of the same input — gain differs by 4×, and the (PRNU·signal) term
    scales with g·signal too."""
    ref = np.full((128, 256), 80.0, dtype=np.float32)
    noisy_lsb = nm.inject_noise(ref, "lsb", tdi_stages=64,
                                rng=np.random.default_rng(7),
                                fpn_template=nm._NO_FPN)
    noisy_msb = nm.inject_noise(ref, "msb", tdi_stages=64,
                                rng=np.random.default_rng(7),
                                fpn_template=nm._NO_FPN)
    # Same seed, same ref → the gain factor dominates the spread.
    assert noisy_lsb.std() > noisy_msb.std()


def test_fpn_template_carries_provenance_meta() -> None:
    """Regenerated templates must embed provenance metadata so the (cutoff,
    dedup flag, n_strips, git sha) that produced them is recoverable from
    the artifact alone. See feedback memory `feedback-data-provenance`."""
    tpl = nm.load_fpn_template("msb", tdi_stages=64)
    assert tpl.meta is not None, (
        "MSB TDI64 template has no embedded meta — regenerate with the "
        "current extract_noise_params.py (which writes a json-encoded meta "
        "field) before relying on this template."
    )
    assert tpl.meta["schema_version"] >= 1
    assert {"generation_utc", "script_path", "git_sha", "params", "inputs"} <= set(tpl.meta)
    params = tpl.meta["params"]
    assert params["bits_selection"] == "msb"
    assert params["tdi_stages"] == "TDI64"
    inputs = tpl.meta["inputs"]
    assert inputs["n_strips_included"] == tpl.n_strips
    assert len(inputs["included_product_ids"]) == tpl.n_strips


def test_inject_noise_with_fpn_template_adds_column_structure() -> None:
    """With the measured FPN template loaded, injected noise must have
    non-trivial column-direction structure (mean-of-column variance) on a
    flat reference patch."""
    rng = np.random.default_rng(1234)
    ref = np.full((512, 1024), 30.0, dtype=np.float32)
    # With FPN template
    with_fpn = nm.inject_noise(ref, "lsb", tdi_stages=64,
                               rng=np.random.default_rng(1234), clip=False)
    # Without FPN
    without_fpn = nm.inject_noise(ref, "lsb", tdi_stages=64,
                                  rng=np.random.default_rng(1234),
                                  fpn_template=nm._NO_FPN, clip=False)
    col_mean_with = with_fpn.mean(axis=0)
    col_mean_without = without_fpn.mean(axis=0)
    # With FPN, the column-mean profile should vary much more than without.
    assert col_mean_with.std() > 2.0 * col_mean_without.std()
