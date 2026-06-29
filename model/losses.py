"""Physics-informed Poisson-Gaussian NLL with multiplicative δ residual.

Plain MSE for the curriculum warmup / E0 control is just
``torch.nn.functional.mse_loss(mu, target)`` — no wrapper here, see
`docs/PROJECT_ROADMAP.md` §3.3.

The single function this module exports implements the heteroscedastic
NLL form pinned in `docs/NOISE_MODEL.md` §4:

    σ²_physics(r,c) = g · μ(r,c) + (g · σ_floor)² + σ²_FPN(c)
    σ²_pred(r,c)    = exp( log σ²_physics + δ(r,c) )
    L(r,c)          = 0.5 · [ (y − μ)² / σ²_pred  +  log σ²_pred ]

Everything except `sigma_floor` is in DN-space. `sigma_floor` is in
electrons (the value `noise_model.SIGMA_FLOOR_EFF_E`); `g` then carries
it back to DN-space variance.

Per-patch noise-model constants (`g`, `sigma_fpn(c)`) are assembled by
the trainer from the Dataset's `meta` (bits, tdi, col_offset) and the
`noise_model` package — they're tensors here, not lookups, so the loss
stays a pure function and trains via vanilla autograd.
"""

from __future__ import annotations

import torch


def pg_nll_loss(
    pred_mean: torch.Tensor,
    pred_delta: torch.Tensor,
    target: torch.Tensor,
    g: torch.Tensor | float,
    sigma_floor: torch.Tensor | float,
    sigma_fpn: torch.Tensor,
    *,
    eps: float = 1e-6,
    reduction: str = "mean",
) -> torch.Tensor:
    """Per-pixel heteroscedastic NLL with a physics prior + multiplicative δ.

    Args:
        pred_mean: μ from the network, DN-space, shape ``(N, 1, H, W)``.
            Expected ≥ 0 (softplus head); a small clamp guards log
            arithmetic without changing well-behaved gradients.
        pred_delta: δ from the network, unitless log-residual on the
            physics-prior variance, shape ``(N, 1, H, W)``. Unconstrained.
        target: clean reference in DN-space, shape ``(N, 1, H, W)``.
        g: gain in DN/e⁻. Scalar or broadcastable per-sample tensor
            (e.g. shape ``(N, 1, 1, 1)``) — per-mode gain varies by
            `bits_selection`, so the trainer typically passes a vector.
        sigma_floor: effective additive noise floor in **electrons**.
            Scalar or per-sample tensor. Multiplied by ``g`` to give
            the DN-space floor variance term.
        sigma_fpn: per-column FPN std in DN, shape ``(N, 1, 1, W)``
            (or any shape broadcastable to ``pred_mean``). Comes from
            the per-(bits, tdi) FPN template sliced at the patch's
            ``col_offset``.
        eps: positive clamp for the variance terms — prevents
            ``log(0)`` if a pathological batch makes σ²_physics
            collapse. Default 1e-6 DN².
        reduction: ``"mean"`` (default), ``"sum"``, or ``"none"``.

    Returns:
        Scalar loss when reduction is ``mean``/``sum``; per-pixel
        loss tensor when ``"none"``.
    """
    sigma2_physics = (
        g * pred_mean.clamp(min=eps)
        + (g * sigma_floor) ** 2
        + sigma_fpn ** 2
    )
    sigma2_physics = sigma2_physics.clamp(min=eps)

    log_sigma2_pred = torch.log(sigma2_physics) + pred_delta
    sigma2_pred = torch.exp(log_sigma2_pred)

    sq_err = (target - pred_mean) ** 2
    nll = 0.5 * (sq_err / sigma2_pred + log_sigma2_pred)

    if reduction == "mean":
        return nll.mean()
    if reduction == "sum":
        return nll.sum()
    if reduction == "none":
        return nll
    raise ValueError(f"reduction must be 'mean'|'sum'|'none', got {reduction!r}")
