"""
Combined Phase-1b extraction of bias profile, σ_FPN(c), and dark current
from deep-shadow rows of OHRC `nrp` strips.

Per strip s:
  - Identify deep-shadow rows: rows whose mean DN is in the bottom q-quantile
    (default 1%). For PSR-condition strips, essentially every row qualifies.
    For sunlit strips, crater shadows / very dark scene regions qualify.
  - Per-column mean over those rows: μ_s(c).
  - Per-column std over those rows: σ_within_s(c). (Within-strip temporal noise.)
  - Per-strip global shadow mean: M_s = mean_c μ_s(c).

Per mode (bits_selection, tdi_stages):
  - bias_profile(c) = mean_s μ_s(c)              [DN]   (deterministic per-column DC offset)
  - σ_FPN(c)        = std_s μ_s(c)               [DN]   (across-strip residual; the σ_FPN
                                                         term in the §12.1 variance prior)
  - M_avg           = mean_s M_s                 [DN]   (= <bias> + g·dark·N_tdi)
  - within_noise(c) = mean_s σ_within_s(c)       [DN]   (per-column temporal noise floor)

Dark-current fit across modes & TDI levels:
  M_avg(mode, N_tdi) = bias_global(mode) + g(mode) · dark_current · N_tdi
  Linear regression of M_avg against (g(mode) · N_tdi) with per-mode intercept
  gives one shared dark_current slope (e⁻/stage). Needs ≥2 distinct N_tdi
  values within at least one mode for the slope to be identifiable.

Outputs:
  analysis/_outputs/per_strip_shadow_stats.parquet
  analysis/_outputs/per_mode_summary.parquet
  analysis/_outputs/bias_profile_<mode>_TDI<N>.png
  analysis/_outputs/dark_current_fit.png   (when ≥2 N_tdi points)

Usage:
  .venv/bin/python -m analysis.extract_noise_params [--modes lsb,msb] [--max-strips-per-mode 20]
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import catalog  # noqa: E402
import noise_model as nm  # noqa: E402

OUT_DIR = REPO / "analysis" / "_outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class StripShadowStats:
    product_id: str
    bits_selection: str
    tdi_stages: str
    n_tdi: int
    solar_incidence: float
    line_count: int
    sample_count: int
    n_shadow_rows: int
    shadow_row_threshold: float
    strip_shadow_mean_dn: float  # mean over (shadow rows × all columns)
    strip_shadow_std_dn: float   # std over (shadow rows × all columns)
    # column-vector stats stored separately in a 2D array per strip


TDI_STAGES = {"TDI64": 64, "TDI128": 128, "TDI256": 256}


def deep_shadow_rows(
    arr: np.ndarray,
    row_quantile: float = 0.01,
    min_rows: int = 50,
) -> tuple[np.ndarray, float]:
    """Return indices of the bottom-`row_quantile` rows by mean DN.

    Returns
    -------
    idx : indices into `arr`'s row axis, sorted ascending.
    threshold : the row-mean cutoff used.
    """
    row_means = arr.mean(axis=1, dtype=np.float64)
    threshold = float(np.quantile(row_means, row_quantile))
    idx = np.where(row_means <= threshold)[0]
    if idx.size < min_rows:
        # widen to at least min_rows by raising the threshold
        order = np.argsort(row_means)
        idx = np.sort(order[:min_rows])
        threshold = float(row_means[idx[-1]])
    return idx, threshold


def per_strip_shadow_stats(
    row: pd.Series,
    row_quantile: float = 0.01,
) -> tuple[StripShadowStats, np.ndarray, np.ndarray] | None:
    """Compute shadow-row stats for one strip.

    Returns (stats, col_mean[c], col_std[c]) — or None if the file is missing.
    """
    img_path = Path(row["img_path"])
    if not img_path.exists():
        return None
    lines = int(row["line_count"])
    samples = int(row["sample_count"])
    try:
        arr = np.memmap(img_path, dtype=np.uint8, mode="r", shape=(lines, samples))
    except (OSError, ValueError) as e:
        print(f"  ! failed to memmap {row['product_id']}: {e}")
        return None

    idx, threshold = deep_shadow_rows(arr, row_quantile=row_quantile)
    shadow = arr[idx, :].astype(np.float32)
    col_mean = shadow.mean(axis=0)
    col_std = shadow.std(axis=0)

    stats = StripShadowStats(
        product_id=str(row["product_id"]),
        bits_selection=str(row["bits_selection"]),
        tdi_stages=str(row["tdi_stages"]),
        n_tdi=TDI_STAGES[str(row["tdi_stages"])],
        solar_incidence=float(row["solar_incidence"]),
        line_count=lines,
        sample_count=samples,
        n_shadow_rows=int(idx.size),
        shadow_row_threshold=threshold,
        strip_shadow_mean_dn=float(shadow.mean()),
        strip_shadow_std_dn=float(shadow.std()),
    )
    return stats, col_mean, col_std


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--modes",
        default="lsb,msb",
        help="comma-separated bits_selection values to process",
    )
    p.add_argument(
        "--max-strips-per-mode",
        type=int,
        default=20,
        help="cap on strips processed per (mode, N_tdi) group",
    )
    p.add_argument(
        "--row-quantile",
        type=float,
        default=0.01,
        help="row-mean quantile defining 'deep shadow' (default 1%)",
    )
    p.add_argument(
        "--only-psr",
        action="store_true",
        help="restrict to solar_incidence ≥ 90° (clean shadow rows)",
    )
    p.add_argument(
        "--max-shadow-mean-dn",
        type=float,
        default=10.0,
        help=(
            "strips whose deep-shadow mean exceeds this DN value are computed "
            "but excluded from per-mode aggregation (they have no real shadow "
            "content). Default 10 DN; scale loosely by 0.5× for mid, 0.25× for msb."
        ),
    )
    p.add_argument(
        "--unique-obs",
        action="store_true",
        help=(
            "Deduplicate dual-station downlinks of the same observation. "
            "Defaults off for backward compatibility; pass for honest n on MSB TDI64."
        ),
    )
    args = p.parse_args()

    requested_modes = [m.strip() for m in args.modes.split(",")]
    df = catalog.load()
    raw = df[df["role"] == "raw"].copy()
    if args.only_psr:
        raw = raw[raw["solar_incidence"] >= 90.0]

    # 12_000 samples is the full detector width; the project assumes it.
    raw = raw[raw["sample_count"] == 12_000]

    if args.unique_obs and "obs_id" in raw.columns:
        before = len(raw)
        raw = raw.drop_duplicates(subset="obs_id", keep="first")
        print(f"[dedup] {before} → {len(raw)} unique observations")

    per_strip_records: list[dict] = []
    per_mode_records: list[dict] = []
    col_mean_buffer: dict[tuple[str, str], list[np.ndarray]] = {}
    col_std_buffer: dict[tuple[str, str], list[np.ndarray]] = {}

    mode_cutoff = {
        "lsb": args.max_shadow_mean_dn,
        "mid": args.max_shadow_mean_dn * 0.5,
        "msb": args.max_shadow_mean_dn * 0.25,
    }

    for mode in requested_modes:
        mode_strips = raw[raw["bits_selection"] == mode]
        if mode_strips.empty:
            print(f"[{mode}] no strips matching filters; skipping")
            continue

        for n_tdi_label, group in mode_strips.groupby("tdi_stages"):
            n_tdi = TDI_STAGES.get(n_tdi_label)
            if n_tdi is None:
                continue
            picks = group.head(args.max_strips_per_mode)
            print(f"[{mode} {n_tdi_label}]  processing {len(picks)}/{len(group)} strips")

            mode_key = (mode, n_tdi_label)
            col_mean_buffer.setdefault(mode_key, [])
            col_std_buffer.setdefault(mode_key, [])

            cutoff = mode_cutoff[mode]
            for _, row in picks.iterrows():
                result = per_strip_shadow_stats(row, row_quantile=args.row_quantile)
                if result is None:
                    print(f"  · {row['product_id']}: skipped (missing/unreadable)")
                    continue
                stats, col_mean, col_std = result
                included = stats.strip_shadow_mean_dn <= cutoff
                rec = asdict(stats)
                rec["included_in_aggregate"] = bool(included)
                rec["shadow_cutoff_dn"] = float(cutoff)
                per_strip_records.append(rec)
                if included:
                    col_mean_buffer[mode_key].append(col_mean)
                    col_std_buffer[mode_key].append(col_std)
                flag = "" if included else "  [EXCLUDED — no real shadow]"
                print(
                    f"  · {stats.product_id}: "
                    f"n_shadow={stats.n_shadow_rows}, "
                    f"shadow_thr={stats.shadow_row_threshold:.2f}, "
                    f"mean={stats.strip_shadow_mean_dn:.3f} DN, "
                    f"std={stats.strip_shadow_std_dn:.3f} DN{flag}"
                )

            if not col_mean_buffer[mode_key]:
                continue
            stack_mean = np.stack(col_mean_buffer[mode_key], axis=0)  # (n_strips, n_cols)
            stack_std = np.stack(col_std_buffer[mode_key], axis=0)
            bias_profile = stack_mean.mean(axis=0)
            sigma_fpn = stack_mean.std(axis=0)
            within_noise = stack_std.mean(axis=0)
            m_avg = float(bias_profile.mean())
            sigma_fpn_global = float(sigma_fpn.mean())
            g = nm.gain_dn_per_e(mode)
            per_mode_records.append(
                {
                    "bits_selection": mode,
                    "tdi_stages": n_tdi_label,
                    "n_tdi": n_tdi,
                    "n_strips": int(stack_mean.shape[0]),
                    "gain_dn_per_e": g,
                    "M_avg_dn": m_avg,
                    "sigma_fpn_mean_dn": sigma_fpn_global,
                    "within_noise_mean_dn": float(within_noise.mean()),
                    "bias_profile_min_dn": float(bias_profile.min()),
                    "bias_profile_max_dn": float(bias_profile.max()),
                    "bias_profile_p2p_dn": float(bias_profile.max() - bias_profile.min()),
                }
            )

            # Save full per-column arrays as a .npz for downstream use.
            np.savez_compressed(
                OUT_DIR / f"bias_arrays_{mode}_{n_tdi_label}.npz",
                bias_profile=bias_profile,
                sigma_fpn=sigma_fpn,
                within_noise=within_noise,
                n_strips=stack_mean.shape[0],
            )

            # Diagnostic PNG.
            try:
                import matplotlib

                matplotlib.use("Agg")
                import matplotlib.pyplot as plt

                fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
                axes[0].plot(bias_profile, lw=0.6, color="tab:blue")
                axes[0].set_ylabel(f"bias(c) [DN, {mode}]")
                axes[0].set_title(
                    f"{mode} {n_tdi_label} — bias profile (n={stack_mean.shape[0]} strips)"
                )
                axes[1].plot(sigma_fpn, lw=0.6, color="tab:orange")
                axes[1].set_ylabel(f"σ_FPN(c) [DN, {mode}]")
                axes[2].plot(within_noise, lw=0.6, color="tab:green")
                axes[2].set_ylabel(f"<σ_within(c)>_s [DN]")
                axes[2].set_xlabel("column index c (0–11 999)")
                fig.tight_layout()
                fig.savefig(OUT_DIR / f"bias_profile_{mode}_{n_tdi_label}.png", dpi=110)
                plt.close(fig)
            except ImportError:
                pass

    # Save per-strip and per-mode tables.
    if per_strip_records:
        ps = pd.DataFrame(per_strip_records)
        ps.to_parquet(OUT_DIR / "per_strip_shadow_stats.parquet", index=False)
    if per_mode_records:
        pm = pd.DataFrame(per_mode_records)
        pm.to_parquet(OUT_DIR / "per_mode_summary.parquet", index=False)

        print("\n=== per-mode summary ===")
        cols = [
            "bits_selection",
            "tdi_stages",
            "n_strips",
            "M_avg_dn",
            "sigma_fpn_mean_dn",
            "within_noise_mean_dn",
            "bias_profile_p2p_dn",
        ]
        print(pm[cols].to_string(index=False))

        # Dark-current fit: fit one slope + per-mode intercepts at the *per-strip*
        # level, so the degrees-of-freedom reflect the actual sample size.
        ps = pd.DataFrame(per_strip_records)
        ps_in = ps[ps["included_in_aggregate"]].copy() if "included_in_aggregate" in ps.columns else ps
        if ps_in["n_tdi"].nunique() >= 2 and not ps_in.empty:
            print("\n=== dark-current fit (per-strip points) ===")
            ps_in["g"] = ps_in["bits_selection"].map(nm.GAIN_DN_PER_E)
            x = (ps_in["g"] * ps_in["n_tdi"]).to_numpy(dtype=np.float64)
            y = ps_in["strip_shadow_mean_dn"].to_numpy(dtype=np.float64)
            mode_cat = ps_in["bits_selection"].astype("category")
            mode_codes = mode_cat.cat.codes.to_numpy()
            n_modes = int(mode_codes.max()) + 1

            # M_s = slope·(g·N_tdi)_s + bias_intercept[mode(s)]
            X = np.zeros((len(ps_in), 1 + n_modes))
            X[:, 0] = x
            for i, c in enumerate(mode_codes):
                X[i, 1 + c] = 1.0
            coef, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
            slope = float(coef[0])
            intercepts = {
                str(mode_cat.cat.categories[i]): float(coef[1 + i])
                for i in range(n_modes)
            }
            yhat = X @ coef
            resid = y - yhat
            ss_res = float((resid ** 2).sum())
            ss_tot = float(((y - y.mean()) ** 2).sum())
            r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
            dof = len(y) - (1 + n_modes)
            rmse_dn = float(np.sqrt(ss_res / max(dof, 1)))
            print(f"  n_points = {len(y)}, dof = {dof}, R² = {r2:.4f}, RMSE = {rmse_dn:.3f} DN")
            print(f"  dark_current  = {slope:.4f} e⁻/stage")
            for m, b in intercepts.items():
                print(f"  bias_<{m}> intercept = {b:.3f} DN  (native DN: {b / nm.GAIN_DN_PER_E[m] * nm.NATIVE_GAIN_DN_PER_E:.3f})")

            # Diagnostic: M vs g·N_tdi, colour by mode.
            try:
                import matplotlib

                matplotlib.use("Agg")
                import matplotlib.pyplot as plt

                fig, ax = plt.subplots(figsize=(7, 5))
                for m_name, m_idx in zip(mode_cat.cat.categories, range(n_modes)):
                    sel = mode_codes == m_idx
                    ax.scatter(x[sel], y[sel], label=f"{m_name} (n={sel.sum()})", s=24, alpha=0.7)
                xx = np.linspace(0, x.max() * 1.05, 100)
                for m_name, m_idx in zip(mode_cat.cat.categories, range(n_modes)):
                    ax.plot(xx, intercepts[str(m_name)] + slope * xx, lw=0.8, alpha=0.6)
                ax.set_xlabel("g · N_tdi  (DN / e⁻/stage · stages)")
                ax.set_ylabel("strip shadow mean DN")
                ax.set_title(
                    f"dark current fit  slope = {slope:.4f} e⁻/stage  R² = {r2:.3f}  n={len(y)}"
                )
                ax.legend()
                fig.tight_layout()
                fig.savefig(OUT_DIR / "dark_current_fit.png", dpi=110)
                plt.close(fig)
            except ImportError:
                pass
        else:
            print(
                "\n=== dark-current fit skipped — need ≥2 distinct N_tdi values "
                "across modes ==="
            )

    print(f"\nartifacts in {OUT_DIR.relative_to(REPO)}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
