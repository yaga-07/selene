"""Tests for ``model.losses.pg_nll_loss``.

The key behavioural check is the **known-σ calibration test**: if the
trainer hands the loss a target sampled from N(μ, σ²_pred), the loss
should converge to the Gaussian negative log-likelihood entropy
``0.5 · (1 + log σ²)`` (the constant ``log(2π)`` is dropped from the
formula in `NOISE_MODEL.md` §4, so the test mirrors that omission).
"""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F

from model.losses import pg_nll_loss

PHYSICS_G = 0.0385         # lsb gain (DN/e⁻)
PHYSICS_SIGMA_FLOOR = 95.0  # e⁻ (noise_model.SIGMA_FLOOR_EFF_E)


def _make_batch(
    n: int = 4, h: int = 32, w: int = 32, mu_dn: float = 50.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    mu = torch.full((n, 1, h, w), mu_dn)
    delta = torch.zeros((n, 1, h, w))
    sigma_fpn = torch.full((n, 1, 1, w), 1.3)  # DN; ~ measured lsb/TDI64
    return mu, delta, sigma_fpn


def test_returns_scalar_under_mean(_=None) -> None:
    mu, delta, sigma_fpn = _make_batch()
    loss = pg_nll_loss(mu, delta, mu, PHYSICS_G, PHYSICS_SIGMA_FLOOR, sigma_fpn)
    assert loss.dim() == 0
    assert torch.isfinite(loss)


def test_reduction_none_preserves_shape() -> None:
    mu, delta, sigma_fpn = _make_batch()
    loss = pg_nll_loss(
        mu, delta, mu, PHYSICS_G, PHYSICS_SIGMA_FLOOR, sigma_fpn, reduction="none",
    )
    assert loss.shape == mu.shape


def test_reduction_sum_equals_mean_times_n() -> None:
    mu, delta, sigma_fpn = _make_batch()
    target = mu + torch.randn_like(mu)
    loss_mean = pg_nll_loss(mu, delta, target, PHYSICS_G, PHYSICS_SIGMA_FLOOR, sigma_fpn,
                            reduction="mean")
    loss_sum = pg_nll_loss(mu, delta, target, PHYSICS_G, PHYSICS_SIGMA_FLOOR, sigma_fpn,
                           reduction="sum")
    assert torch.allclose(loss_sum, loss_mean * mu.numel(), rtol=1e-5)


def test_reduction_invalid_raises() -> None:
    mu, delta, sigma_fpn = _make_batch()
    with pytest.raises(ValueError, match="reduction must be"):
        pg_nll_loss(mu, delta, mu, PHYSICS_G, PHYSICS_SIGMA_FLOOR, sigma_fpn,
                    reduction="median")


def test_at_perfect_prediction_loss_is_entropy_term() -> None:
    """When y = μ exactly, the squared-error term vanishes and the loss
    reduces to ``0.5 · log σ²_pred``. Verify against a hand-computed
    σ²_physics."""
    mu, delta, sigma_fpn = _make_batch(n=1, h=4, w=4, mu_dn=100.0)
    expected_var = (
        PHYSICS_G * 100.0
        + (PHYSICS_G * PHYSICS_SIGMA_FLOOR) ** 2
        + 1.3 ** 2
    )
    expected_loss = 0.5 * math.log(expected_var)  # δ=0, perfect prediction
    loss = pg_nll_loss(mu, delta, mu.clone(), PHYSICS_G, PHYSICS_SIGMA_FLOOR, sigma_fpn)
    assert torch.allclose(loss, torch.tensor(expected_loss), rtol=1e-5), \
        f"got {loss.item():.6f}, expected {expected_loss:.6f}"


def test_calibrated_target_matches_gaussian_entropy() -> None:
    """The central correctness check: sample target ~ N(μ, σ²_pred) and
    confirm the loss converges to ``0.5 · (1 + log σ²)`` (the constant
    ``log(2π)`` is intentionally dropped — see NOISE_MODEL.md §4)."""
    torch.manual_seed(0)
    n, h, w = 4, 64, 64  # 16k samples per realisation
    mu = torch.full((n, 1, h, w), 50.0)
    delta = torch.zeros_like(mu)
    sigma_fpn = torch.full((n, 1, 1, w), 1.3)

    sigma2_pred = (
        PHYSICS_G * 50.0
        + (PHYSICS_G * PHYSICS_SIGMA_FLOOR) ** 2
        + 1.3 ** 2
    )
    sigma_pred = math.sqrt(sigma2_pred)
    noise = torch.randn_like(mu) * sigma_pred
    target = mu + noise

    loss = pg_nll_loss(mu, delta, target, PHYSICS_G, PHYSICS_SIGMA_FLOOR, sigma_fpn)
    expected = 0.5 * (1.0 + math.log(sigma2_pred))
    # 16k samples → std of mean of (y-μ)²/σ² (chi² with df=1, var=2) is √(2/16384) ≈ 0.011
    # the entropy term itself is deterministic, only the squared-error term fluctuates.
    assert abs(loss.item() - expected) < 0.02, \
        f"got {loss.item():.4f}, expected {expected:.4f}"


def test_delta_inflates_or_deflates_variance() -> None:
    """δ > 0 should raise σ²_pred (overshoots the prior); δ < 0 lowers
    it. With a calibrated target, positive δ overdisperses the model
    and the (y-μ)²/σ² term shrinks but the log-σ² term grows — the
    net effect is loss > entropy. Symmetric story for δ < 0 below the
    point where the squared-error term explodes."""
    torch.manual_seed(0)
    mu, _, sigma_fpn = _make_batch(n=4, h=64, w=64)
    sigma2_physics = (
        PHYSICS_G * 50.0 + (PHYSICS_G * PHYSICS_SIGMA_FLOOR) ** 2 + 1.3 ** 2
    )
    target = mu + torch.randn_like(mu) * math.sqrt(sigma2_physics)

    delta_zero = torch.zeros_like(mu)
    delta_pos = torch.full_like(mu, 0.5)
    delta_neg = torch.full_like(mu, -0.3)

    base_args = (target, PHYSICS_G, PHYSICS_SIGMA_FLOOR, sigma_fpn)
    loss_zero = pg_nll_loss(mu, delta_zero, *base_args)
    loss_pos = pg_nll_loss(mu, delta_pos, *base_args)
    loss_neg = pg_nll_loss(mu, delta_neg, *base_args)

    # δ=0 is the MLE choice for a calibrated target → minimum
    assert loss_zero < loss_pos, f"δ=0 loss {loss_zero:.4f} not < δ=+0.5 loss {loss_pos:.4f}"
    assert loss_zero < loss_neg, f"δ=0 loss {loss_zero:.4f} not < δ=-0.3 loss {loss_neg:.4f}"


def test_gradients_flow_through_mu_and_delta() -> None:
    mu, delta, sigma_fpn = _make_batch()
    mu = mu.detach().requires_grad_(True)
    delta = delta.detach().requires_grad_(True)
    target = mu.detach() + torch.randn_like(mu)
    loss = pg_nll_loss(mu, delta, target, PHYSICS_G, PHYSICS_SIGMA_FLOOR, sigma_fpn)
    loss.backward()
    assert mu.grad is not None and torch.isfinite(mu.grad).all()
    assert delta.grad is not None and torch.isfinite(delta.grad).all()
    assert mu.grad.abs().sum() > 0
    assert delta.grad.abs().sum() > 0


def test_per_sample_gain_broadcasts() -> None:
    """Trainer passes ``g`` as shape ``(N, 1, 1, 1)`` so each sample's
    bits_selection gives its own gain. Confirm broadcasting works and
    a lsb (g=0.0385) and msb (g=0.0096) sample in the same batch give
    different per-pixel σ² contributions."""
    n, h, w = 2, 16, 16
    mu = torch.full((n, 1, h, w), 100.0)
    delta = torch.zeros_like(mu)
    sigma_fpn = torch.full((n, 1, 1, w), 1.0)
    g = torch.tensor([[0.0385], [0.0096]]).view(n, 1, 1, 1)  # lsb, msb

    per_pixel = pg_nll_loss(mu, delta, mu.clone(), g, PHYSICS_SIGMA_FLOOR,
                            sigma_fpn, reduction="none")
    # At perfect prediction, loss = 0.5 · log σ²_physics; σ²_physics
    # differs between samples → loss values differ between samples.
    assert per_pixel[0].mean() != per_pixel[1].mean()


def test_works_with_unet_forward_dict() -> None:
    """End-to-end: feed a real ``OHRCDenoiserUNet`` output dict into
    the loss without unpacking gymnastics."""
    from model import OHRCDenoiserUNet
    torch.manual_seed(0)
    net = OHRCDenoiserUNet(base_channels=8, num_levels=4)
    x = torch.randn(2, 1, 64, 64) * 5 + 50
    out = net(x)
    sigma_fpn = torch.full((2, 1, 1, 64), 1.3)
    loss = pg_nll_loss(out["mu"], out["delta"], x.clamp(min=0),
                       PHYSICS_G, PHYSICS_SIGMA_FLOOR, sigma_fpn)
    assert torch.isfinite(loss)

    # Curriculum-handoff check: at δ=0 (network init), pg_nll and a
    # purely-physics-prior heteroscedastic NLL agree exactly.
    sigma2_physics = (
        PHYSICS_G * out["mu"].clamp(min=1e-6)
        + (PHYSICS_G * PHYSICS_SIGMA_FLOOR) ** 2
        + sigma_fpn ** 2
    )
    expected = 0.5 * ((x.clamp(min=0) - out["mu"]) ** 2 / sigma2_physics
                      + torch.log(sigma2_physics))
    assert torch.allclose(loss, expected.mean(), rtol=1e-5)


def test_mse_phase_uses_F_mse_loss_directly() -> None:
    """Documents the trainer contract: the MSE phase has no wrapper in
    this module — vanilla ``F.mse_loss(mu, target)`` is the call site.
    This test exists so a reader of the test file sees the convention."""
    mu = torch.full((2, 1, 8, 8), 50.0, requires_grad=True)
    target = mu.detach() + torch.randn_like(mu) * 2
    mse = F.mse_loss(mu, target)
    mse.backward()
    assert torch.isfinite(mse)
    assert mu.grad is not None
