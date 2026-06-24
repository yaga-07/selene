"""§2.6 sanity validator: forward-model σ matches empirical K-realisation σ
within ±15 % per mode.

For each mode (bits, tdi) with an FPN template, sample N reference patches
from the manifest and, at α ∈ {0.10, 0.30, 1.00}, run K=30 Monte-Carlo
realisations of ``noise_model.inject_noise``. Compare per-pixel empirical
σ across realisations to the analytic prediction:

    σ²_pred(r, c) = g · S_dn(r, c)              shot
                  + (g · σ_floor)²              additive floor
                  + (PRNU_FRAC · S_dn(r, c))²   residual PRNU
                  + σ_FPN(c)²                   column FPN (redrawn per call)

The bias b(c) is a DC offset (line 196 of ``forward_model.py``) and does
not enter a cross-realisation std, so it is absent from both sides.

We pass ``clip=False`` to the sampler to match the unclipped analytic
formula. The operational ``inject_noise`` default clips to [0, 255]; the
truncation mismatch at low S and at saturation is a known Phase-3
calibration issue absorbed by the multiplicative residual log-variance δ
(see ``docs/GATE_1_FINDING.md`` §12.1 and ``docs/NOISE_MODEL.md`` §4).

A perfect run reports ratio ≈ 0.991, not 1.000: the K=30 sample-std
estimator has a c4 bias factor of √(2/(K-1)) · Γ(K/2)/Γ((K-1)/2) ≈
0.9914 — sample std is unbiased for variance but biased low for σ.

Run:  ``.venv/bin/python -m analysis.validate_injection_stats``
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

import catalog
import noise_model as nm
from noise_model.fpn_template import load_fpn_template, template_available
from training_data.curation import PATCH_SIZE

N_PATCHES_PER_MODE = 80
K_REALIZATIONS = 30
ALPHA_GRID = (0.10, 0.30, 1.00)
RNG_SEED = 42
PASS_TOLERANCE = 0.15
SIGNAL_BIN_EDGES_DN = np.array(
    [0, 1, 2, 4, 8, 16, 32, 64, 128, 256], dtype=np.float32,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = REPO_ROOT / "training_data" / "reference_manifest.parquet"
OUTPUT_DIR = REPO_ROOT / "analysis" / "_outputs"

_MMAP_CACHE: dict[str, np.memmap] = {}


def _pred_sigma_dn(
    dim_dn: np.ndarray, bits: str, fpn, col_offset: int,
) -> np.ndarray:
    g = nm.gain_dn_per_e(bits)
    sigma_floor_dn = nm.SIGMA_FLOOR_EFF_E * g
    n_cols = dim_dn.shape[1]
    sigma_fpn = fpn.sigma_fpn[col_offset:col_offset + n_cols].astype(np.float32)
    s = np.maximum(dim_dn, 0.0).astype(np.float32)
    var = (
        g * s                          # shot
        + sigma_floor_dn ** 2          # additive floor
        + (nm.PRNU_FRAC * s) ** 2      # PRNU
        + sigma_fpn[None, :] ** 2      # column FPN η(c)
    )
    return np.sqrt(var)


def _emp_sigma_dn(
    dim_dn: np.ndarray, bits: str, tdi: int, fpn, col_offset: int,
    k: int, rng: np.random.Generator,
) -> np.ndarray:
    stack = np.empty((k,) + dim_dn.shape, dtype=np.float32)
    for i in range(k):
        stack[i] = nm.inject_noise(
            dim_dn, bits_selection=bits, tdi_stages=tdi, rng=rng,
            fpn_template=fpn, col_offset=col_offset, clip=False,
        )
    return stack.std(axis=0, ddof=1)


def _load_patch(
    catalog_df: pd.DataFrame, product_id: str, row0: int, col0: int,
) -> np.ndarray:
    if product_id not in _MMAP_CACHE:
        row = catalog_df[catalog_df["product_id"] == product_id].iloc[0]
        _MMAP_CACHE[product_id] = np.memmap(
            row["img_path"], dtype=np.uint8, mode="r",
            shape=(int(row["line_count"]), int(row["sample_count"])),
        )
    arr = _MMAP_CACHE[product_id]
    return np.asarray(
        arr[row0:row0 + PATCH_SIZE, col0:col0 + PATCH_SIZE]
    ).astype(np.float32)


def _run_mode(
    bits: str, tdi: int,
    manifest: pd.DataFrame, catalog_df: pd.DataFrame,
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not template_available(bits, tdi):
        print(f"  {bits}/TDI{tdi}: no FPN template — skipping")
        return pd.DataFrame(), pd.DataFrame()
    fpn = load_fpn_template(bits, tdi)
    pool = manifest[
        (manifest["source_bits"] == bits)
        & (manifest["source_tdi"] == tdi)
    ]
    if pool.empty:
        print(f"  {bits}/TDI{tdi}: 0 patches in manifest — skipping")
        return pd.DataFrame(), pd.DataFrame()
    n = min(N_PATCHES_PER_MODE, len(pool))
    picks = pool.sample(n=n, random_state=RNG_SEED).reset_index(drop=True)
    print(f"  {bits}/TDI{tdi}: {n} patches × K={K_REALIZATIONS} × |α|={len(ALPHA_GRID)}")

    patch_rows: list[dict] = []
    binned: dict[float, list[tuple[np.ndarray, np.ndarray, np.ndarray]]] = {
        a: [] for a in ALPHA_GRID
    }
    for _, p in picks.iterrows():
        clean = _load_patch(
            catalog_df, p["product_id"], int(p["row0"]), int(p["col0"]),
        )
        col_offset = min(int(p["col0"]), fpn.n_cols - PATCH_SIZE)
        for alpha in ALPHA_GRID:
            dim = clean * alpha
            pred = _pred_sigma_dn(dim, bits, fpn, col_offset)
            emp = _emp_sigma_dn(
                dim, bits, tdi, fpn, col_offset, K_REALIZATIONS, rng,
            )
            patch_rows.append({
                "bits": bits, "tdi": tdi, "alpha": alpha,
                "product_id": p["product_id"],
                "row0": int(p["row0"]), "col0": int(p["col0"]),
                "mean_clean_dn": float(clean.mean()),
                "mean_dim_dn": float(dim.mean()),
                "pred_sigma_mean": float(pred.mean()),
                "emp_sigma_mean": float(emp.mean()),
                "ratio": float(emp.mean() / max(pred.mean(), 1e-6)),
            })
            binned[alpha].append((dim.ravel(), pred.ravel(), emp.ravel()))

    patch_df = pd.DataFrame(patch_rows)

    bin_rows: list[dict] = []
    for alpha, parts in binned.items():
        if not parts:
            continue
        dim_all = np.concatenate([d for d, _, _ in parts])
        pred_all = np.concatenate([p for _, p, _ in parts])
        emp_all = np.concatenate([e for _, _, e in parts])
        for lo, hi in zip(SIGNAL_BIN_EDGES_DN[:-1], SIGNAL_BIN_EDGES_DN[1:]):
            sel = (dim_all >= lo) & (dim_all < hi)
            if int(sel.sum()) < 200:
                continue
            bin_rows.append({
                "bits": bits, "tdi": tdi, "alpha": alpha,
                "bin_lo": float(lo), "bin_hi": float(hi),
                "n_pixels": int(sel.sum()),
                "mean_signal_dn": float(dim_all[sel].mean()),
                "pred_sigma": float(pred_all[sel].mean()),
                "emp_sigma": float(emp_all[sel].mean()),
                "ratio": float(emp_all[sel].mean() / max(pred_all[sel].mean(), 1e-6)),
            })
    bin_df = pd.DataFrame(bin_rows)
    return patch_df, bin_df


def _summarise(patch_all: pd.DataFrame) -> list[dict]:
    rows: list[dict] = []
    for (bits, tdi), grp in patch_all.groupby(["bits", "tdi"]):
        med = float(grp["ratio"].median())
        p5, p95 = grp["ratio"].quantile([0.05, 0.95]).tolist()
        worst = float((grp["ratio"] - 1.0).abs().quantile(0.95))
        passes = (abs(med - 1.0) <= PASS_TOLERANCE) and (worst <= 2 * PASS_TOLERANCE)
        rows.append({
            "bits": bits, "tdi": tdi, "n_patches": int(len(grp)),
            "ratio_median": med,
            "ratio_p5": float(p5), "ratio_p95": float(p95),
            "worst_95pct_abs_dev": worst,
            "passes": passes,
        })
        print(
            f"  {bits}/TDI{tdi}: median={med:.3f}  "
            f"5–95%={p5:.3f}–{p95:.3f}  worst_95pct=|{worst:.1%}|  "
            f"{'PASS' if passes else 'FAIL'}"
        )
    return rows


def _plot(
    patch_all: pd.DataFrame, bin_all: pd.DataFrame, out_path: Path,
) -> None:
    palette = {
        ("msb", 64): "C0", ("msb", 128): "C1",
        ("lsb", 64): "C2", ("lsb", 128): "C3",
    }
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for (b, t), grp in patch_all.groupby(["bits", "tdi"]):
        axes[0].scatter(
            grp["pred_sigma_mean"], grp["emp_sigma_mean"],
            s=12, alpha=0.5, c=palette.get((b, t)), label=f"{b}/TDI{t}",
        )
    mx = float(
        max(patch_all["pred_sigma_mean"].max(), patch_all["emp_sigma_mean"].max())
    ) * 1.05
    xs = np.linspace(0, mx, 50)
    axes[0].plot(xs, xs, "k--", lw=1, label="y = x")
    axes[0].plot(xs, xs * (1 + PASS_TOLERANCE), "r:", lw=1,
                 label=f"±{PASS_TOLERANCE:.0%}")
    axes[0].plot(xs, xs * (1 - PASS_TOLERANCE), "r:", lw=1)
    axes[0].set_xlim(0, mx); axes[0].set_ylim(0, mx)
    axes[0].set_xlabel("predicted σ (DN, analytic)")
    axes[0].set_ylabel(f"empirical σ (DN, K={K_REALIZATIONS} MC)")
    axes[0].set_title("Per-patch mean σ — empirical vs predicted")
    axes[0].legend(fontsize=8)

    for (b, t), grp in bin_all.groupby(["bits", "tdi"]):
        grp_sorted = grp.sort_values("mean_signal_dn")
        axes[1].plot(
            grp_sorted["mean_signal_dn"], grp_sorted["ratio"], "o-",
            c=palette.get((b, t)), alpha=0.7, label=f"{b}/TDI{t}",
        )
    axes[1].axhline(1.0, color="k", lw=1)
    axes[1].axhline(1 + PASS_TOLERANCE, color="r", ls=":", lw=1,
                    label=f"±{PASS_TOLERANCE:.0%}")
    axes[1].axhline(1 - PASS_TOLERANCE, color="r", ls=":", lw=1)
    axes[1].set_xscale("log")
    axes[1].set_xlabel("mean signal DN (log)")
    axes[1].set_ylabel("ratio empirical / predicted")
    axes[1].set_title("Per-bin ratio vs signal — term-mismatch fingerprint")
    axes[1].legend(fontsize=8, loc="best")

    fig.suptitle(
        f"§2.6 sanity: empirical vs forward-model σ "
        f"(N≤{N_PATCHES_PER_MODE}/mode, K={K_REALIZATIONS}, α∈{list(ALPHA_GRID)})"
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest = pq.read_table(MANIFEST_PATH).to_pandas()
    catalog_df = catalog.load()
    rng = np.random.default_rng(RNG_SEED)

    modes = [("msb", 64), ("msb", 128), ("lsb", 64), ("lsb", 128)]
    print(f"validating modes={modes}  N≤{N_PATCHES_PER_MODE}  K={K_REALIZATIONS}  α={list(ALPHA_GRID)}")
    patch_chunks: list[pd.DataFrame] = []
    bin_chunks: list[pd.DataFrame] = []
    for bits, tdi in modes:
        patch_df, bin_df = _run_mode(bits, tdi, manifest, catalog_df, rng)
        if not patch_df.empty:
            patch_chunks.append(patch_df)
            bin_chunks.append(bin_df)

    patch_all = pd.concat(patch_chunks, ignore_index=True)
    bin_all = pd.concat(bin_chunks, ignore_index=True)

    csv_patch = OUTPUT_DIR / "validate_injection_stats_per_patch.csv"
    csv_bin = OUTPUT_DIR / "validate_injection_stats_per_bin.csv"
    patch_all.to_csv(csv_patch, index=False)
    bin_all.to_csv(csv_bin, index=False)

    print()
    print("per-mode ratio summary (patch-level):")
    _summarise(patch_all)

    png = OUTPUT_DIR / "validate_injection_stats.png"
    _plot(patch_all, bin_all, png)
    print()
    print(f"wrote {csv_patch.relative_to(REPO_ROOT)}")
    print(f"wrote {csv_bin.relative_to(REPO_ROOT)}")
    print(f"wrote {png.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
