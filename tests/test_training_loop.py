"""Unit tests for ``training/loop.py``.

The sanity loop itself needs the SSD-mounted Dataset, so it's exercised
by running the module's ``__main__`` rather than by pytest. These tests
cover the pieces that can stand alone:

    - per-batch noise-constant assembly (the load-bearing translation
      from Dataset ``meta`` to ``(g, σ_FPN)`` for the loss);
    - ``compute_loss`` dispatch between MSE and PG-NLL phases;
    - ``train_step`` on a hand-built batch — confirms gradients flow
      and loss decreases when over-fitting a single fixed batch.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

import noise_model as nm
from model import OHRCDenoiserUNet
from training.loop import (
    PATCH_SIZE,
    assemble_noise_constants,
    compute_loss,
    train_step,
)


def _fake_batch(
    n: int = 2,
    patch: int = 64,
    sim_bits: list[str] | None = None,
    sim_tdi: list[int] | None = None,
    col0: list[int] | None = None,
) -> dict:
    """Build a collated batch that mimics ``OHRCReferenceDataset``'s output."""
    sim_bits = sim_bits or ["lsb"] * n
    sim_tdi = sim_tdi or [64] * n
    col0 = col0 or [0] * n
    return {
        "noisy": torch.randn(n, 1, patch, patch) * 5 + 50,
        "clean": torch.zeros(n, 1, patch, patch) + 50,
        "meta": {
            "product_id": ["fake"] * n,
            "row0": torch.zeros(n, dtype=torch.long),
            "col0": torch.tensor(col0, dtype=torch.long),
            "source_bits": sim_bits,
            "source_tdi": torch.tensor(sim_tdi, dtype=torch.long),
            "sim_bits": sim_bits,
            "sim_tdi": torch.tensor(sim_tdi, dtype=torch.long),
            "alpha": torch.ones(n),
            "idx": torch.arange(n),
        },
    }


def test_assemble_noise_constants_shapes_and_device() -> None:
    batch = _fake_batch(n=3, sim_bits=["lsb", "msb", "lsb"],
                       sim_tdi=[64, 128, 64], col0=[0, 100, 500])
    g, sigma_fpn = assemble_noise_constants(batch["meta"], torch.device("cpu"),
                                             patch_size=256)
    assert g.shape == (3, 1, 1, 1)
    assert sigma_fpn.shape == (3, 1, 1, 256)
    assert g.dtype == torch.float32
    assert sigma_fpn.dtype == torch.float32


def test_assemble_noise_constants_uses_correct_per_sample_gain() -> None:
    """Two samples in one batch, different bits — their g values must
    follow the noise_model lookup, not be averaged or swapped."""
    batch = _fake_batch(n=2, sim_bits=["lsb", "msb"], sim_tdi=[64, 64])
    g, _ = assemble_noise_constants(batch["meta"], torch.device("cpu"))
    expected_lsb = nm.gain_dn_per_e("lsb")
    expected_msb = nm.gain_dn_per_e("msb")
    assert float(g[0]) == pytest.approx(expected_lsb)
    assert float(g[1]) == pytest.approx(expected_msb)
    # 4:1 ratio — a regression that hard-codes one gain would fail this.
    assert float(g[0] / g[1]) == pytest.approx(4.0, rel=1e-3)


def test_assemble_noise_constants_slices_fpn_at_col0() -> None:
    """Two samples with different col0 must get different σ_FPN slices
    (FPN templates are column-coherent — the whole point of slicing)."""
    batch = _fake_batch(n=2, sim_bits=["msb", "msb"], sim_tdi=[64, 64],
                       col0=[0, 6000])
    _, sigma_fpn = assemble_noise_constants(batch["meta"], torch.device("cpu"),
                                             patch_size=256)
    # The two slices come from disjoint column ranges of the same template
    # → should not be identical (would only happen with zero probability).
    assert not torch.allclose(sigma_fpn[0], sigma_fpn[1])


def test_compute_loss_mse_phase() -> None:
    torch.manual_seed(0)
    net = OHRCDenoiserUNet(base_channels=8, num_levels=4)
    batch = _fake_batch(n=2, patch=64)
    out = net(batch["noisy"])
    loss = compute_loss(out, batch, "mse", torch.device("cpu"))
    assert loss.dim() == 0 and torch.isfinite(loss)


def test_compute_loss_pg_nll_phase() -> None:
    torch.manual_seed(0)
    net = OHRCDenoiserUNet(base_channels=8, num_levels=4)
    batch = _fake_batch(n=2, patch=64)
    out = net(batch["noisy"])
    loss = compute_loss(out, batch, "pg_nll", torch.device("cpu"))
    assert loss.dim() == 0 and torch.isfinite(loss)


def test_compute_loss_rejects_unknown_phase() -> None:
    net = OHRCDenoiserUNet(base_channels=8, num_levels=4)
    batch = _fake_batch(n=2, patch=64)
    out = net(batch["noisy"])
    with pytest.raises(ValueError, match="loss_phase must be"):
        compute_loss(out, batch, "huber", torch.device("cpu"))


@pytest.mark.parametrize("phase", ["mse", "pg_nll"])
def test_train_step_decreases_loss_on_overfit_batch(phase: str) -> None:
    """Hand the model the same batch repeatedly with a high LR — both
    loss phases should drive the loss down. This is the central
    proof-of-life check for the full train_step path."""
    torch.manual_seed(0)
    np.random.seed(0)
    net = OHRCDenoiserUNet(base_channels=8, num_levels=4)
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-2)
    batch = _fake_batch(n=2, patch=64)

    losses = []
    for _ in range(30):
        losses.append(train_step(net, batch, optimizer, phase, torch.device("cpu")))

    # Allow small-batch noise: compare the mean of the last 5 steps to
    # the first 5 rather than relying on monotonic decrease.
    assert np.mean(losses[-5:]) < np.mean(losses[:5]), (
        f"phase={phase}: loss did not decrease — first-5 mean "
        f"{np.mean(losses[:5]):.4f}, last-5 mean {np.mean(losses[-5:]):.4f}"
    )
