"""
Per-(bits_selection, tdi_stages) FPN templates, measured from deep-shadow
rows of `nrp` PSR strips.

The templates are dictionaries with:

    bias_profile   : ndarray[float32] (12 000,)
        Per-column deterministic DC offset, DN in the given encoding.
        This is the source of the visible vertical stripes in browse PNGs.
    sigma_fpn      : ndarray[float32] (12 000,)
        Per-column std *across strips* of the per-column shadow mean, DN.
        Used as the additive per-column FPN draw at injection time.
    within_noise   : ndarray[float32] (12 000,)
        Per-column temporal noise within a single strip's deep-shadow rows,
        DN. Not used by inject_noise (the lumped σ_floor in the forward
        model already covers this); exposed for diagnostic comparison.
    n_strips       : int
        Number of contributing strips. Larger ⇒ more reliable template.

The .npz files were produced by `analysis/extract_noise_params.py`. See
docs/NOISE_MODEL.md §3 for how they enter the forward model.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TEMPLATE_DIR = REPO_ROOT / "analysis" / "_outputs"


@dataclass(frozen=True)
class FPNTemplate:
    bits_selection: str
    tdi_stages: int
    bias_profile: np.ndarray   # (n_cols,), DN
    sigma_fpn: np.ndarray      # (n_cols,), DN
    within_noise: np.ndarray   # (n_cols,), DN
    n_strips: int
    meta: dict[str, Any] | None = field(default=None)

    @property
    def n_cols(self) -> int:
        return int(self.bias_profile.shape[0])


_TDI_LABELS = {64: "TDI64", 128: "TDI128", 256: "TDI256"}


def _template_path(bits_selection: str, tdi_stages: int, source_dir: Path) -> Path:
    tdi_label = _TDI_LABELS.get(int(tdi_stages))
    if tdi_label is None:
        raise ValueError(
            f"unsupported tdi_stages={tdi_stages}; have templates only for "
            f"{sorted(_TDI_LABELS)}"
        )
    return source_dir / f"bias_arrays_{bits_selection.lower()}_{tdi_label}.npz"


def template_available(bits_selection: str, tdi_stages: int,
                       source_dir: Path | None = None) -> bool:
    """Whether a measured FPN template exists for this (mode, TDI)."""
    sd = source_dir or DEFAULT_TEMPLATE_DIR
    try:
        return _template_path(bits_selection, tdi_stages, sd).is_file()
    except ValueError:
        return False


def load_fpn_template(bits_selection: str, tdi_stages: int,
                      source_dir: Path | None = None) -> FPNTemplate:
    """
    Load the measured FPN template for one (bits_selection, tdi_stages) mode.

    Raises FileNotFoundError with a usable message if the template is
    missing (e.g. MID modes — we have no PSR-condition MID strips).
    """
    sd = source_dir or DEFAULT_TEMPLATE_DIR
    path = _template_path(bits_selection, tdi_stages, sd)
    if not path.is_file():
        raise FileNotFoundError(
            f"no FPN template for ({bits_selection}, TDI{tdi_stages}) at {path}. "
            "Either generate it with analysis/extract_noise_params.py, or pass "
            "inject_noise(..., fpn_template=None) to disable measured-FPN injection."
        )
    z = np.load(path)
    meta: dict[str, Any] | None = None
    if "meta" in z.files:
        try:
            meta = json.loads(str(z["meta"].item()))
        except (json.JSONDecodeError, ValueError):
            meta = None
    return FPNTemplate(
        bits_selection=bits_selection.lower(),
        tdi_stages=int(tdi_stages),
        bias_profile=z["bias_profile"].astype(np.float32),
        sigma_fpn=z["sigma_fpn"].astype(np.float32),
        within_noise=z["within_noise"].astype(np.float32),
        n_strips=int(z["n_strips"]),
        meta=meta,
    )
