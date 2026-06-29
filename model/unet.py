"""Single U-Net with dual output heads for OHRC low-signal denoising.

Architecture (per `docs/PROJECT_ROADMAP.md` §3.2):
    - Input: 256×256 single-channel float32 patch in DN-space (no [0, 1]
      normalisation — the PG-NLL prior in `docs/NOISE_MODEL.md` §4 lives
      in DN-space and unit mismatch silently breaks the variance term).
    - 4-level encoder (double Conv → BN → ReLU, MaxPool downsample).
    - Bottleneck (double conv).
    - 4-level decoder (bilinear upsample, concat skip, double conv).
    - Two 1×1 output heads from the final decoder feature map:
        μ-head: softplus to enforce μ ≥ 0 (DN-space, no upper bound —
                ADC native max is 1023 DN at 10-bit and clean targets
                can exceed 255 after α-scaling + bias).
        δ-head: linear, zero-init so δ ≈ 0 at construction → σ²_pred
                starts exactly at σ²_physics. This gives the MSE → PG-NLL
                curriculum (§3.3) a clean handoff: the μ-only warmup
                phase doesn't perturb the variance prior.

Inputs and outputs are both single-channel `(N, 1, H, W)` tensors. The
spatial dims must be divisible by 2**(num_levels) — 16 for the default
4-level configuration. 256 → 128 → 64 → 32 → 16 at the bottleneck.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


def _double_conv(in_ch: int, out_ch: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class _DownBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.conv = _double_conv(in_ch, out_ch)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        skip = self.conv(x)
        return self.pool(skip), skip


class _UpBlock(nn.Module):
    """Bilinear upsample → concat skip → double conv. Bilinear chosen
    over `ConvTranspose2d` to avoid the checkerboard artifacts the
    transposed-conv path is known to produce on smooth low-signal scenes.
    """

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int) -> None:
        super().__init__()
        self.conv = _double_conv(in_ch + skip_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class OHRCDenoiserUNet(nn.Module):
    """U-Net with μ-head + multiplicative δ-head, DN-space throughout.

    Args:
        in_channels: number of input channels. OHRC patches are scalar,
            so 1 by default. Kept configurable for future cross-sensor
            (ShadowCam) experiments.
        base_channels: width of the first encoder stage. Subsequent
            stages double. Default 64 — a 4-level 64-base U-Net at
            256×256 fits comfortably in 16 GB VRAM at batch 16.
        num_levels: encoder depth. Spatial dims must be divisible by
            2**num_levels. Default 4.
    """

    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 64,
        num_levels: int = 4,
    ) -> None:
        super().__init__()
        self.num_levels = num_levels

        widths: Sequence[int] = [base_channels * (2 ** i) for i in range(num_levels + 1)]

        self.downs = nn.ModuleList()
        prev = in_channels
        for w in widths[:-1]:
            self.downs.append(_DownBlock(prev, w))
            prev = w

        self.bottleneck = _double_conv(prev, widths[-1])

        self.ups = nn.ModuleList()
        prev = widths[-1]
        for w in reversed(widths[:-1]):
            self.ups.append(_UpBlock(prev, w, w))
            prev = w

        self.mu_head = nn.Conv2d(prev, 1, kernel_size=1)
        self.delta_head = nn.Conv2d(prev, 1, kernel_size=1)

        nn.init.zeros_(self.delta_head.weight)
        nn.init.zeros_(self.delta_head.bias)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Run a forward pass.

        Returns a dict with:
            mu: (N, 1, H, W) — denoised mean in DN-space, ≥ 0.
            delta: (N, 1, H, W) — multiplicative log-variance residual,
                ≈ 0 at init.

        The σ²_physics prior and the final σ²_pred = exp(log σ²_physics + δ)
        are computed in `model.losses.pg_nll_loss` from these outputs +
        per-patch noise-model constants — kept out of the network so the
        net stays a pure feature predictor.
        """
        h, w = x.shape[-2:]
        div = 2 ** self.num_levels
        if h % div or w % div:
            raise ValueError(
                f"input H={h}, W={w} must each be divisible by {div} "
                f"for a {self.num_levels}-level U-Net"
            )

        skips: list[torch.Tensor] = []
        z = x
        for down in self.downs:
            z, skip = down(z)
            skips.append(skip)

        z = self.bottleneck(z)

        for up, skip in zip(self.ups, reversed(skips)):
            z = up(z, skip)

        mu = F.softplus(self.mu_head(z))
        delta = self.delta_head(z)
        return {"mu": mu, "delta": delta}
