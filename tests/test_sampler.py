"""Tests for training.sampler.make_mode_balanced_sampler."""

from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd
import pytest

from training.sampler import (
    DEFAULT_TARGET_BITS,
    DEFAULT_TARGET_TDI,
    make_mode_balanced_sampler,
)


def _toy_manifest(counts: dict[tuple[str, int], int]) -> pd.DataFrame:
    rows = []
    for (b, t), n in counts.items():
        for _ in range(n):
            rows.append({"source_bits": b, "source_tdi": t})
    return pd.DataFrame(rows)


def test_missing_columns_raises() -> None:
    df = pd.DataFrame({"foo": [1, 2, 3]})
    with pytest.raises(ValueError, match="missing required columns"):
        make_mode_balanced_sampler(df)


def test_empty_df_raises() -> None:
    df = pd.DataFrame({"source_bits": [], "source_tdi": []})
    with pytest.raises(ValueError, match="df is empty"):
        make_mode_balanced_sampler(df)


def test_target_bits_must_sum_to_one() -> None:
    df = _toy_manifest({("msb", 64): 10})
    with pytest.raises(ValueError, match="target_bits must sum"):
        make_mode_balanced_sampler(df, target_bits={"msb": 0.5, "lsb": 0.4})


def test_target_tdi_must_sum_to_one() -> None:
    df = _toy_manifest({("msb", 64): 10})
    with pytest.raises(ValueError, match="target_tdi must sum"):
        make_mode_balanced_sampler(df, target_tdi={64: 0.5, 128: 0.4})


def test_balanced_df_uniform_weights() -> None:
    """If source already matches target, all rows get equal weight."""
    # Source freq = target freq = 0.42 msb64, 0.18 msb128, 0.28 lsb64, 0.12 lsb128.
    df = _toy_manifest({
        ("msb", 64): 42, ("msb", 128): 18,
        ("lsb", 64): 28, ("lsb", 128): 12,
    })
    sampler = make_mode_balanced_sampler(df, weight_cap=float("inf"))
    w = np.asarray(sampler.weights)
    # All weights should be ~1.0 (the target / source ratio).
    assert np.allclose(w, w[0], rtol=1e-9)


def test_underrepresented_group_gets_higher_weight() -> None:
    """msb64 dominates → its per-row weight should be < lsb64's."""
    df = _toy_manifest({("msb", 64): 1000, ("lsb", 64): 10, ("msb", 128): 1, ("lsb", 128): 1})
    sampler = make_mode_balanced_sampler(df, weight_cap=float("inf"))
    w = np.asarray(sampler.weights)
    bits = df["source_bits"].to_numpy()
    tdi = df["source_tdi"].to_numpy()
    w_msb64 = w[(bits == "msb") & (tdi == 64)][0]
    w_lsb64 = w[(bits == "lsb") & (tdi == 64)][0]
    assert w_lsb64 > w_msb64


def test_weight_cap_applied() -> None:
    """Very thin group (1 row out of N) gets capped, not its raw ratio."""
    df = _toy_manifest({
        ("msb", 64): 999,
        ("lsb", 128): 1,  # 1/1000 source vs 0.12 target → raw weight 120
    })
    sampler = make_mode_balanced_sampler(df, weight_cap=5.0)
    w = np.asarray(sampler.weights)
    # Find the lsb/128 row's weight
    bits = df["source_bits"].to_numpy()
    tdi = df["source_tdi"].to_numpy()
    w_thin = w[(bits == "lsb") & (tdi == 128)][0]
    assert w_thin == pytest.approx(5.0)


def test_sampling_over_represents_minority() -> None:
    """Drawing num_samples=N should pull more minority rows than uniform."""
    df = _toy_manifest({("msb", 64): 1000, ("lsb", 64): 10})
    sampler = make_mode_balanced_sampler(
        df, weight_cap=float("inf"), num_samples=10_000, seed=42,
    )
    idxs = list(iter(sampler))
    bits = df["source_bits"].to_numpy()
    drawn_lsb_frac = float(np.mean(bits[idxs] == "lsb"))
    # Uniform would give lsb fraction ≈ 10/1010 ≈ 0.0099.
    # Target marginal for lsb is 0.4, so we expect ~0.4 (within sampling noise).
    assert drawn_lsb_frac > 0.3, f"expected lsb upweighting, got {drawn_lsb_frac:.3f}"


def test_default_targets_match_dataset_module() -> None:
    """Sanity: the defaults here equal DEFAULT_SIM_*_PROBS in dataset.py."""
    from training_data.dataset import (
        DEFAULT_SIM_BITS_PROBS,
        DEFAULT_SIM_TDI_PROBS,
    )
    assert DEFAULT_TARGET_BITS == DEFAULT_SIM_BITS_PROBS
    assert DEFAULT_TARGET_TDI == DEFAULT_SIM_TDI_PROBS


def test_seed_determinism() -> None:
    """Same seed → same draw sequence."""
    df = _toy_manifest({("msb", 64): 50, ("lsb", 128): 50})
    s1 = make_mode_balanced_sampler(df, seed=7, num_samples=200)
    s2 = make_mode_balanced_sampler(df, seed=7, num_samples=200)
    assert list(iter(s1)) == list(iter(s2))


def test_different_seeds_diverge() -> None:
    df = _toy_manifest({("msb", 64): 50, ("lsb", 128): 50})
    s1 = make_mode_balanced_sampler(df, seed=1, num_samples=200)
    s2 = make_mode_balanced_sampler(df, seed=2, num_samples=200)
    assert list(iter(s1)) != list(iter(s2))


def test_default_num_samples_equals_len_df() -> None:
    df = _toy_manifest({("msb", 64): 100, ("lsb", 64): 10})
    sampler = make_mode_balanced_sampler(df)
    assert sampler.num_samples == len(df)


def test_missing_target_group_does_not_crash() -> None:
    """df has only msb/64; target asks for all four — sampler still works."""
    df = _toy_manifest({("msb", 64): 100})
    sampler = make_mode_balanced_sampler(df, weight_cap=float("inf"))
    w = np.asarray(sampler.weights)
    # All weight on the one present group; sampler draws only those idxs.
    assert (w > 0).sum() == 100
    idxs = Counter(iter(sampler))
    assert set(idxs.keys()).issubset(set(range(100)))
