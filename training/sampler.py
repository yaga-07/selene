"""Mode-balanced ``WeightedRandomSampler`` for the SELENE manifest.

The reference manifest is heavily skewed by detector mode (msb/TDI64 ≈ 83 %
of train, lsb/TDI128 ≈ 0.04 %). Uniform sampling would feed the trainer
almost exclusively bright-equatorial msb/TDI64 patches and starve it of
low-signal lsb scenes — the very regime SELENE is designed to denoise.

**What this is actually balancing.**  Noise-mode coverage (``sim_bits``,
``sim_tdi``) is *already* uniformised by the per-``__getitem__`` random
draws in ``OHRCReferenceDataset`` — every sample independently picks a
simulated mode from ``DEFAULT_SIM_*_PROBS`` regardless of the source
patch's mode. The sampler therefore does not need to fix noise-mode
coverage.

What the sampler *does* fix is **scene diversity**: source ``(bits, tdi)``
correlates strongly with where on the Moon the patch came from. lsb
patches are over-represented in polar scenes (low-signal, what we
denoise at inference) and msb in equatorial sunlit scenes (what we use
as clean reference). Up-weighting lsb sources gives the network more
polar-scene structure to learn from.

**Weight cap.** lsb/TDI128 has only 138 patches in train — sampling it
proportionally to its target weight would over-fit those 138 tiles. The
``weight_cap`` (default 5.0) bounds any group's per-row weight relative
to its uniform weight, accepting that lsb/TDI128 will be structurally
under-represented vs the target. Document this in the paper as a known
limitation of the corpus.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd
import torch
from torch.utils.data import WeightedRandomSampler

# Default target marginals — matches ``DEFAULT_SIM_*_PROBS`` in
# training_data/dataset.py (roadmap §2.3).
DEFAULT_TARGET_BITS: dict[str, float] = {"lsb": 0.4, "msb": 0.6}
DEFAULT_TARGET_TDI: dict[int, float] = {64: 0.7, 128: 0.3}


def _target_joint(
    target_bits: Mapping[str, float],
    target_tdi: Mapping[int, float],
) -> dict[tuple[str, int], float]:
    """Joint target over (bits, tdi) under independence of the marginals."""
    if not np.isclose(sum(target_bits.values()), 1.0):
        raise ValueError(
            f"target_bits must sum to 1.0; got {sum(target_bits.values())}"
        )
    if not np.isclose(sum(target_tdi.values()), 1.0):
        raise ValueError(
            f"target_tdi must sum to 1.0; got {sum(target_tdi.values())}"
        )
    return {
        (b, t): pb * pt
        for b, pb in target_bits.items()
        for t, pt in target_tdi.items()
    }


def make_mode_balanced_sampler(
    df: pd.DataFrame,
    *,
    target_bits: Mapping[str, float] | None = None,
    target_tdi: Mapping[int, float] | None = None,
    weight_cap: float = 5.0,
    num_samples: int | None = None,
    seed: int = 0,
) -> WeightedRandomSampler:
    """Build a ``WeightedRandomSampler`` that up-weights under-represented
    source ``(bits, tdi)`` groups toward the joint target.

    Parameters
    ----------
    df:
        Manifest slice (typically ``OHRCReferenceDataset.df`` for the
        train split). Must contain ``source_bits`` and ``source_tdi``
        columns. One sampler row corresponds to one ``df`` row, in order.
    target_bits, target_tdi:
        Marginals defining the joint target weight. Defaults are
        ``{lsb: 0.4, msb: 0.6}`` and ``{64: 0.7, 128: 0.3}``, matching
        the per-sample noise-mode distribution in
        ``training_data/dataset.py``.
    weight_cap:
        Maximum per-row weight, expressed as a multiple of the uniform
        weight (i.e. ``cap × 1/N``). Caps the lsb/TDI128 group, which is
        structurally thin (138 patches in train). Set ``inf`` to disable.
    num_samples:
        Length of one epoch. Defaults to ``len(df)``.
    seed:
        Deterministic seed for the sampler's internal generator.

    Returns
    -------
    WeightedRandomSampler with ``replacement=True``.
    """
    required = {"source_bits", "source_tdi"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"df missing required columns: {sorted(missing)}; "
            f"have {df.columns.tolist()}"
        )
    if len(df) == 0:
        raise ValueError("df is empty")

    target_bits = dict(target_bits or DEFAULT_TARGET_BITS)
    target_tdi = dict(target_tdi or DEFAULT_TARGET_TDI)
    target = _target_joint(target_bits, target_tdi)

    n = len(df)
    # Per-group source frequency in this df.
    counts = (
        df.groupby(["source_bits", "source_tdi"]).size().to_dict()
    )

    # Per-row weight = (target_freq / source_freq) for that row's group,
    # normalised so uniform weight = 1 (then capped at weight_cap).
    # Equivalently: per-row weight × N = (target / source_freq).
    # WeightedRandomSampler only cares about relative weights, so the
    # absolute scale doesn't matter — but the cap is interpreted as
    # "max k× the uniform weight 1/N", i.e. cap on (target / source_freq).
    bits_arr = df["source_bits"].to_numpy()
    tdi_arr = df["source_tdi"].to_numpy()
    weights = np.empty(n, dtype=np.float64)
    for (b, t), grp_target in target.items():
        source_count = counts.get((b, t), 0)
        if source_count == 0:
            # No patches in this group → target is unreachable; skip.
            continue
        source_freq = source_count / n
        raw = grp_target / source_freq
        capped = min(raw, weight_cap)
        mask = (bits_arr == b) & (tdi_arr == t)
        weights[mask] = capped

    # Any group not in target (shouldn't happen with default product
    # space) keeps weight 0 → never sampled. Guard against the all-zero
    # pathology:
    if weights.sum() == 0:
        raise ValueError(
            "no rows match any target group — check target_bits/target_tdi "
            f"vs df groups {list(counts.keys())}"
        )

    gen = torch.Generator()
    gen.manual_seed(int(seed))
    return WeightedRandomSampler(
        weights=torch.as_tensor(weights, dtype=torch.double),
        num_samples=int(num_samples) if num_samples is not None else n,
        replacement=True,
        generator=gen,
    )
