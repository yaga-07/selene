"""PyTorch ``Dataset`` over ``reference_manifest.parquet`` with on-the-fly
forward-noise injection (§2.5).

Each ``__getitem__`` returns a dict with:

  - ``clean``: float32 tensor (1, 256, 256) — reference patch scaled by α
  - ``noisy``: float32 tensor (1, 256, 256) — clean + forward-noise (DN)
  - ``meta``:  dict {product_id, row0, col0, source_bits, source_tdi,
                     sim_bits, sim_tdi, alpha, idx}

The reference patch is in ``source_bits`` / ``source_tdi``; the simulated
observation can be a different (bits, tdi) mode. α ∈ [0.05, 1.0] log-
uniformly darkens the scene before noise injection — covering the
read-noise regime through to bright sunlit (roadmap §2.3).

Per-worker memmap cache: each DataLoader worker process keeps its own
``{product_id: memmap}`` dict; the first ``__getitem__`` per
(worker, strip) opens an FD, subsequent calls reuse it. Module-level
state is independent across forked workers — no `h5py`-style handle-
sharing problem.

Reproducibility: per-item rng is seeded from ``(base_seed, idx)`` via
``np.random.SeedSequence``. Two runs with the same ``seed`` produce
identical batches up to ``DataLoader`` shuffle order.

Mode imbalance: the manifest is heavily skewed (msb/TDI64 ≈ 85 %,
lsb/TDI128 ≈ 0.1 %). The Dataset itself does *not* re-weight — that
belongs to a ``WeightedRandomSampler`` constructed from
``df["source_bits"], df["source_tdi"]`` and passed to the DataLoader.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset

from noise_model import (
    PRNU_FRAC,
    inject_noise,
    load_fpn_template,
)
from training_data.curation import PATCH_SIZE

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = REPO_ROOT / "training_data" / "reference_manifest.parquet"

DEFAULT_ALPHA_LOG_RANGE: tuple[float, float] = (math.log(0.05), math.log(1.0))
DEFAULT_SIM_BITS_PROBS: dict[str, float] = {"lsb": 0.4, "msb": 0.6}
DEFAULT_SIM_TDI_PROBS: dict[int, float] = {64: 0.7, 128: 0.3}

# Module-level caches — each forked DataLoader worker has its own copy.
_MEMMAP_CACHE: dict[str, np.memmap] = {}
_FPN_CACHE: dict[tuple[str, int], object] = {}
_BUNDLE_CACHE: dict[Path, np.memmap] = {}


def _get_memmap(pid: str, strip_meta: dict) -> np.memmap:
    arr = _MEMMAP_CACHE.get(pid)
    if arr is None:
        info = strip_meta[pid]
        arr = np.memmap(
            info["img_path"], dtype=np.uint8, mode="r",
            shape=(info["lines"], info["samples"]),
        )
        _MEMMAP_CACHE[pid] = arr
    return arr


def _get_fpn(bits: str, tdi: int):
    key = (bits, tdi)
    tmpl = _FPN_CACHE.get(key)
    if tmpl is None:
        tmpl = load_fpn_template(bits, tdi)
        _FPN_CACHE[key] = tmpl
    return tmpl


def _get_bundle(bundle_path: Path) -> np.memmap:
    """Cached mmap of the Colab patches.npy bundle (per-worker)."""
    arr = _BUNDLE_CACHE.get(bundle_path)
    if arr is None:
        arr = np.load(bundle_path, mmap_mode="r")
        _BUNDLE_CACHE[bundle_path] = arr
    return arr


def strip_meta_from_catalog() -> dict[str, dict]:
    """Build {product_id: {img_path, lines, samples}} from the catalog.

    Imported lazily so the Colab path (which has no `catalog/_index/`)
    doesn't trigger a `catalog.load()` it can't satisfy.
    """
    import catalog  # noqa: PLC0415 — lazy, see docstring
    cat = catalog.load()
    meta_df = cat.set_index("product_id")[["img_path", "line_count", "sample_count"]]
    return {
        pid: {
            "img_path": Path(meta_df.at[pid, "img_path"]),
            "lines": int(meta_df.at[pid, "line_count"]),
            "samples": int(meta_df.at[pid, "sample_count"]),
        }
        for pid in meta_df.index
    }


class OHRCReferenceDataset(Dataset):
    """Manifest-driven (noisy, clean) pair Dataset for SELENE Phase 2."""

    def __init__(
        self,
        split: str,
        *,
        manifest_path: Path = DEFAULT_MANIFEST,
        bundle_path: Path | None = None,
        seed: int = 0,
        alpha_log_range: tuple[float, float] = DEFAULT_ALPHA_LOG_RANGE,
        sim_bits_probs: dict[str, float] | None = None,
        sim_tdi_probs: dict[int, float] | None = None,
        deterministic: bool = False,
        strip_meta: dict[str, dict] | None = None,
    ) -> None:
        """
        bundle_path:
            If given, read clean patches from the pre-baked ``patches.npy``
            via ``bundle[manifest_idx]`` and bypass the catalog/memmap path
            entirely (Colab path — no SSD, no ``catalog/_index/``).
            ``bundle[i]`` must equal ``manifest`` row ``i`` (held by
            ``training_data/bundle_for_colab.py``).
        """
        if split not in {"train", "val"}:
            raise ValueError(f"split must be 'train' or 'val'; got {split!r}")

        df = pq.read_table(manifest_path).to_pandas()
        # Preserve the full-manifest position as `manifest_idx`. The bundle
        # is indexed by this (bundle[i] == manifest row i), so per-split
        # idx → manifest_idx → bundle row.
        df = df.reset_index().rename(columns={"index": "manifest_idx"})
        self.df = df[df["split"] == split].reset_index(drop=True)
        if len(self.df) == 0:
            raise ValueError(
                f"no patches in split={split!r} in {manifest_path}"
            )

        self.split = split
        self.manifest_path = Path(manifest_path)
        self.bundle_path = Path(bundle_path) if bundle_path is not None else None
        if self.bundle_path is not None and not self.bundle_path.exists():
            raise FileNotFoundError(
                f"bundle_path does not exist: {self.bundle_path}"
            )
        self.seed = int(seed)
        self.alpha_log_range = tuple(alpha_log_range)
        self.deterministic = bool(deterministic)

        sim_bits = dict(sim_bits_probs or DEFAULT_SIM_BITS_PROBS)
        sim_tdi = dict(sim_tdi_probs or DEFAULT_SIM_TDI_PROBS)
        self._bits_keys = list(sim_bits.keys())
        self._bits_probs = np.asarray(
            [sim_bits[k] for k in self._bits_keys], dtype=np.float64,
        )
        if not np.isclose(self._bits_probs.sum(), 1.0):
            raise ValueError(
                f"sim_bits_probs must sum to 1.0; got {self._bits_probs.sum()}"
            )
        self._tdi_keys = list(sim_tdi.keys())
        self._tdi_probs = np.asarray(
            [sim_tdi[k] for k in self._tdi_keys], dtype=np.float64,
        )
        if not np.isclose(self._tdi_probs.sum(), 1.0):
            raise ValueError(
                f"sim_tdi_probs must sum to 1.0; got {self._tdi_probs.sum()}"
            )

        self._strip_meta_override = strip_meta
        self._strip_meta_resolved: dict[str, dict] | None = None

    def __len__(self) -> int:
        return len(self.df)

    def _strips(self) -> dict[str, dict]:
        """Resolve per-strip memmap metadata. NEVER called on the bundle
        path — that path skips the catalog entirely so Colab clones
        without ``catalog/_index/`` still work."""
        if self.bundle_path is not None:
            raise RuntimeError(
                "_strips() called on bundle-mode Dataset — should not happen; "
                "bundle path reads bundle[manifest_idx] directly."
            )
        if self._strip_meta_resolved is None:
            self._strip_meta_resolved = (
                self._strip_meta_override
                if self._strip_meta_override is not None
                else strip_meta_from_catalog()
            )
        return self._strip_meta_resolved

    def __getitem__(self, idx: int) -> dict:
        if not 0 <= idx < len(self.df):
            raise IndexError(idx)
        row = self.df.iloc[idx]
        pid = str(row["product_id"])
        r0 = int(row["row0"])
        c0 = int(row["col0"])
        source_bits = str(row["source_bits"])
        source_tdi = int(row["source_tdi"])

        if self.bundle_path is not None:
            manifest_idx = int(row["manifest_idx"])
            bundle = _get_bundle(self.bundle_path)
            ref_uint8 = np.asarray(bundle[manifest_idx])
        else:
            arr = _get_memmap(pid, self._strips())
            ref_uint8 = np.asarray(
                arr[r0:r0 + PATCH_SIZE, c0:c0 + PATCH_SIZE]
            )

        rng = np.random.default_rng(
            np.random.SeedSequence([self.seed, idx])
        )

        if self.deterministic:
            alpha = 1.0
            sim_bits = source_bits
            sim_tdi = source_tdi
        else:
            log_lo, log_hi = self.alpha_log_range
            alpha = float(math.exp(rng.uniform(log_lo, log_hi)))
            sim_bits = str(rng.choice(self._bits_keys, p=self._bits_probs))
            sim_tdi = int(rng.choice(self._tdi_keys, p=self._tdi_probs))

        clean_dn = (alpha * ref_uint8.astype(np.float32))
        fpn_tmpl = _get_fpn(sim_bits, sim_tdi)
        # Pin the FPN column slice to the patch's source col0 — at inference
        # on real strips the σ_FPN(c) for a patch IS its actual detector
        # columns, so training-time parity requires the same. See
        # docs/PAPER_NOTES.md (FPN train-test parity) for the rationale.
        noisy_dn = inject_noise(
            clean_dn,
            bits_selection=sim_bits,
            tdi_stages=sim_tdi,
            rng=rng,
            fpn_template=fpn_tmpl,
            prnu_frac=PRNU_FRAC,
            col_offset=c0,
            clip=True,
        )

        return {
            "clean": torch.from_numpy(clean_dn).unsqueeze(0),
            "noisy": torch.from_numpy(noisy_dn).unsqueeze(0),
            "meta": {
                "product_id": pid,
                "row0": r0,
                "col0": c0,
                "source_bits": source_bits,
                "source_tdi": source_tdi,
                "sim_bits": sim_bits,
                "sim_tdi": sim_tdi,
                "alpha": alpha,
                "idx": idx,
            },
        }
