"""Shape + init checks for ``OHRCDenoiserUNet``.

Not a training/convergence test — just confirms the model wires up
correctly: the heads emit the right shapes, δ starts at zero (so the
MSE→PG-NLL curriculum hands off cleanly per `PROJECT_ROADMAP.md` §3.3),
μ is non-negative (softplus), and the model accepts the documented
input range without surprises.
"""

from __future__ import annotations

import pytest
import torch

from model import OHRCDenoiserUNet


@pytest.fixture
def model() -> OHRCDenoiserUNet:
    torch.manual_seed(0)
    return OHRCDenoiserUNet(in_channels=1, base_channels=8, num_levels=4)


def test_output_shapes_match_input(model: OHRCDenoiserUNet) -> None:
    x = torch.randn(2, 1, 64, 64) * 30 + 50  # DN-ish scale, can be negative pre-softplus
    out = model(x)
    assert set(out.keys()) == {"mu", "delta"}
    assert out["mu"].shape == (2, 1, 64, 64)
    assert out["delta"].shape == (2, 1, 64, 64)
    assert out["mu"].dtype == torch.float32
    assert out["delta"].dtype == torch.float32


def test_delta_starts_near_zero(model: OHRCDenoiserUNet) -> None:
    """δ-head must be zero-initialised so σ²_pred = σ²_physics at init.

    This is what makes the MSE warmup → PG-NLL switch a clean handoff
    (NOISE_MODEL.md §4 + PROJECT_ROADMAP.md §3.3).
    """
    model.eval()
    x = torch.randn(2, 1, 64, 64) * 30 + 50
    with torch.no_grad():
        out = model(x)
    assert torch.all(out["delta"] == 0.0), (
        f"δ-head should be zero at init; got max |δ| = {out['delta'].abs().max():.3e}"
    )


def test_mu_is_nonnegative(model: OHRCDenoiserUNet) -> None:
    """μ-head is softplus(linear) — DN-space, ≥ 0, no upper bound."""
    model.eval()
    x = torch.randn(2, 1, 64, 64) * 30 + 50
    with torch.no_grad():
        out = model(x)
    assert torch.all(out["mu"] >= 0.0)
    # softplus(0) = ln(2) ≈ 0.693, so the entire output shouldn't pin to 0
    # for arbitrary input — sanity-check the head isn't dead.
    assert out["mu"].max() > 0.1


def test_input_size_must_be_divisible(model: OHRCDenoiserUNet) -> None:
    """4-level U-Net requires H, W divisible by 16. Refuse otherwise
    instead of silently giving misaligned skip connections."""
    bad = torch.zeros(1, 1, 65, 64)
    with pytest.raises(ValueError, match="divisible by 16"):
        model(bad)


def test_accepts_256x256_real_patch_size(model: OHRCDenoiserUNet) -> None:
    """Sanity: the curated patch size (256×256, per training_data.curation)
    flows through without dimension errors."""
    x = torch.zeros(1, 1, 256, 256)
    out = model(x)
    assert out["mu"].shape == (1, 1, 256, 256)
    assert out["delta"].shape == (1, 1, 256, 256)


def test_parameter_count_is_reasonable(model: OHRCDenoiserUNet) -> None:
    """Tiny-config (base=8, 4 levels) is ~500k params — small enough
    for fast CI but big enough to confirm the channel ladder isn't
    collapsed. Production-config check is the separate test below."""
    n = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert 250_000 < n < 1_000_000, f"parameter count {n} outside expected band"


def test_default_base_channels_is_31m_params() -> None:
    """Pin the production-config parameter count so a refactor that
    inadvertently changes the channel ladder fails loudly."""
    m = OHRCDenoiserUNet()  # defaults: base=64, num_levels=4
    n = sum(p.numel() for p in m.parameters() if p.requires_grad)
    assert 25_000_000 < n < 35_000_000, f"production-config param count {n} drifted"
