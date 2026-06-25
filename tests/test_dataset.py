"""Unit tests for training_data.dataset.OHRCReferenceDataset.

These tests use synthetic fixtures (tmp_path strips + tiny in-memory
manifest + fake FPN templates) so they don't depend on the SSD or the
real corpus. The real-corpus timing check lives in
``analysis/benchmark_dataset.py``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

from noise_model.fpn_template import FPNTemplate
from training_data import dataset as ds_mod
from training_data.dataset import OHRCReferenceDataset


PATCH = ds_mod.PATCH_SIZE
STRIP_H = 4 * PATCH
STRIP_W = 4 * PATCH


@pytest.fixture(autouse=True)
def _reset_module_caches():
    """Each test starts with empty memmap/FPN caches."""
    ds_mod._MEMMAP_CACHE.clear()
    ds_mod._FPN_CACHE.clear()
    yield
    ds_mod._MEMMAP_CACHE.clear()
    ds_mod._FPN_CACHE.clear()


def _seed_fake_fpn():
    """Populate the module-level FPN cache with all 4 (bits, tdi) modes the
    dataset might sample, using small synthetic templates."""
    for bits in ("lsb", "msb"):
        for tdi in (64, 128):
            ds_mod._FPN_CACHE[(bits, tdi)] = FPNTemplate(
                bits_selection=bits,
                tdi_stages=tdi,
                bias_profile=np.zeros(2 * PATCH, dtype=np.float32),
                sigma_fpn=np.full(2 * PATCH, 0.5, dtype=np.float32),
                within_noise=np.zeros(2 * PATCH, dtype=np.float32),
                n_strips=1,
            )


def _make_fixture(tmp_path: Path) -> tuple[Path, dict]:
    """Write one synthetic strip + a manifest with a handful of patches."""
    pid = "synth_strip_0001"
    img_path = tmp_path / f"{pid}.img"
    rng = np.random.default_rng(0)
    strip = rng.integers(0, 255, size=(STRIP_H, STRIP_W), dtype=np.uint8)
    img_path.write_bytes(strip.tobytes())

    rows = []
    for r0 in (0, PATCH, 2 * PATCH):
        for c0 in (0, PATCH):
            rows.append({
                "product_id": pid,
                "split": "train",
                "row0": r0,
                "col0": c0,
                "source_bits": "msb",
                "source_tdi": 64,
                "mean_dn": float(strip[r0:r0 + PATCH, c0:c0 + PATCH].mean()),
                "frac_zero": 0.0,
                "frac_sat": 0.0,
                "sobel_99": 50.0,
            })
    # Add a val row so split filtering is exercised.
    rows.append({
        "product_id": pid,
        "split": "val",
        "row0": 3 * PATCH,
        "col0": 0,
        "source_bits": "msb",
        "source_tdi": 64,
        "mean_dn": float(strip[3 * PATCH:, :PATCH].mean()),
        "frac_zero": 0.0,
        "frac_sat": 0.0,
        "sobel_99": 50.0,
    })
    manifest_path = tmp_path / "manifest.parquet"
    pq.write_table(pa.Table.from_pandas(pd.DataFrame(rows), preserve_index=False),
                   manifest_path)

    strip_meta = {pid: {
        "img_path": img_path,
        "lines": STRIP_H,
        "samples": STRIP_W,
    }}
    return manifest_path, strip_meta


def test_len_and_split_filter(tmp_path):
    manifest_path, strip_meta = _make_fixture(tmp_path)
    _seed_fake_fpn()
    train = OHRCReferenceDataset("train", manifest_path=manifest_path,
                                 strip_meta=strip_meta)
    val = OHRCReferenceDataset("val", manifest_path=manifest_path,
                               strip_meta=strip_meta)
    assert len(train) == 6
    assert len(val) == 1


def test_getitem_shapes_and_dtypes(tmp_path):
    manifest_path, strip_meta = _make_fixture(tmp_path)
    _seed_fake_fpn()
    ds = OHRCReferenceDataset("train", manifest_path=manifest_path,
                              strip_meta=strip_meta, seed=42)
    sample = ds[0]
    assert sample["clean"].shape == (1, PATCH, PATCH)
    assert sample["noisy"].shape == (1, PATCH, PATCH)
    assert sample["clean"].dtype == torch.float32
    assert sample["noisy"].dtype == torch.float32
    m = sample["meta"]
    assert m["product_id"] == "synth_strip_0001"
    assert m["source_bits"] == "msb"
    assert m["source_tdi"] == 64
    assert m["sim_bits"] in {"lsb", "msb"}
    assert m["sim_tdi"] in {64, 128}
    assert 0.05 <= m["alpha"] <= 1.0
    assert m["idx"] == 0


def test_determinism_bit_exact(tmp_path):
    """Same seed + same idx → bit-exact noisy tensor."""
    manifest_path, strip_meta = _make_fixture(tmp_path)
    _seed_fake_fpn()
    ds1 = OHRCReferenceDataset("train", manifest_path=manifest_path,
                               strip_meta=strip_meta, seed=7)
    ds2 = OHRCReferenceDataset("train", manifest_path=manifest_path,
                               strip_meta=strip_meta, seed=7)
    for idx in (0, 2, 4):
        s1 = ds1[idx]
        s2 = ds2[idx]
        assert torch.equal(s1["noisy"], s2["noisy"])
        assert torch.equal(s1["clean"], s2["clean"])
        assert s1["meta"]["alpha"] == s2["meta"]["alpha"]
        assert s1["meta"]["sim_bits"] == s2["meta"]["sim_bits"]
        assert s1["meta"]["sim_tdi"] == s2["meta"]["sim_tdi"]


def test_different_seeds_differ(tmp_path):
    manifest_path, strip_meta = _make_fixture(tmp_path)
    _seed_fake_fpn()
    ds1 = OHRCReferenceDataset("train", manifest_path=manifest_path,
                               strip_meta=strip_meta, seed=1)
    ds2 = OHRCReferenceDataset("train", manifest_path=manifest_path,
                               strip_meta=strip_meta, seed=2)
    s1, s2 = ds1[0], ds2[0]
    assert not torch.equal(s1["noisy"], s2["noisy"])


def test_deterministic_mode_pins_alpha_and_sim(tmp_path):
    manifest_path, strip_meta = _make_fixture(tmp_path)
    _seed_fake_fpn()
    ds = OHRCReferenceDataset("train", manifest_path=manifest_path,
                              strip_meta=strip_meta, deterministic=True)
    s = ds[0]
    assert s["meta"]["alpha"] == 1.0
    assert s["meta"]["sim_bits"] == s["meta"]["source_bits"]
    assert s["meta"]["sim_tdi"] == s["meta"]["source_tdi"]
    # In deterministic mode clean == reference patch as float32.
    arr = np.memmap(strip_meta["synth_strip_0001"]["img_path"],
                    dtype=np.uint8, mode="r", shape=(STRIP_H, STRIP_W))
    row = s["meta"]["row0"]
    col = s["meta"]["col0"]
    expected_clean = arr[row:row + PATCH, col:col + PATCH].astype(np.float32)
    assert np.array_equal(s["clean"].squeeze(0).numpy(), expected_clean)


def test_clean_scales_with_alpha(tmp_path):
    """clean_dn = α · reference_uint8.astype(float32)."""
    manifest_path, strip_meta = _make_fixture(tmp_path)
    _seed_fake_fpn()
    ds = OHRCReferenceDataset("train", manifest_path=manifest_path,
                              strip_meta=strip_meta, seed=0)
    arr = np.memmap(strip_meta["synth_strip_0001"]["img_path"],
                    dtype=np.uint8, mode="r", shape=(STRIP_H, STRIP_W))
    for idx in range(len(ds)):
        s = ds[idx]
        row, col = s["meta"]["row0"], s["meta"]["col0"]
        ref = arr[row:row + PATCH, col:col + PATCH].astype(np.float32)
        expected = s["meta"]["alpha"] * ref
        actual = s["clean"].squeeze(0).numpy()
        assert np.allclose(actual, expected, atol=1e-5)


def test_index_out_of_range(tmp_path):
    manifest_path, strip_meta = _make_fixture(tmp_path)
    _seed_fake_fpn()
    ds = OHRCReferenceDataset("train", manifest_path=manifest_path,
                              strip_meta=strip_meta)
    with pytest.raises(IndexError):
        _ = ds[len(ds)]


def test_invalid_probs_raise(tmp_path):
    manifest_path, strip_meta = _make_fixture(tmp_path)
    _seed_fake_fpn()
    with pytest.raises(ValueError, match="sim_bits_probs"):
        OHRCReferenceDataset("train", manifest_path=manifest_path,
                             strip_meta=strip_meta,
                             sim_bits_probs={"lsb": 0.3, "msb": 0.5})
