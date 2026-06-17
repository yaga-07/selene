"""
Phase 0 diagnostic for OHRC raw products.

Reads each raw .img as a uint8 memmap (no full load into RAM), computes a
DN histogram over the full frame and over the darkest band (proxy for the
PSR region), and writes a side-by-side report into analysis/.

Verdict logic comes from task.md:
- If most of the dark-band mass piles up at DN <= 2, the msb-style crush
  has happened and the data is not usable for PSR work.
- If the dark band shows a meaningful spread (roughly DN ~2..25), the data
  is usable.
"""

from __future__ import annotations

import os
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path("/Users/yashgajjar/workspace/yash/selene")
OUT = ROOT / "analysis"
OUT.mkdir(parents=True, exist_ok=True)

NS = {
    "pds": "http://pds.nasa.gov/pds4/pds/v1",
    "isda": "https://isda.issdc.gov.in/pds4/isda/v1",
}


@dataclass
class Product:
    name: str
    xml_path: Path
    img_path: Path
    lines: int
    samples: int
    bits_selection: str
    tdi_stages: str
    line_exposure_duration: str
    spacecraft_altitude: str
    pixel_resolution: str
    solar_incidence: str


def parse_xml(xml_path: Path) -> Product:
    tree = ET.parse(xml_path)
    root = tree.getroot()

    def t(xpath: str) -> str:
        el = root.find(xpath, NS)
        return "" if el is None or el.text is None else el.text.strip()

    file_name = t(".//pds:File_Area_Observational/pds:File/pds:file_name")
    axes = root.findall(".//pds:Array_2D_Image/pds:Axis_Array", NS)
    dims = {}
    for ax in axes:
        nm = ax.find("pds:axis_name", NS).text
        el = int(ax.find("pds:elements", NS).text)
        dims[nm] = el

    return Product(
        name=xml_path.stem,
        xml_path=xml_path,
        img_path=xml_path.with_name(file_name),
        lines=dims["Line"],
        samples=dims["Sample"],
        bits_selection=t(".//isda:bits_selection"),
        tdi_stages=t(".//isda:tdi_stages"),
        line_exposure_duration=t(".//isda:line_exposure_duration"),
        spacecraft_altitude=t(".//isda:spacecraft_altitude"),
        pixel_resolution=t(".//isda:pixel_resolution"),
        solar_incidence=t(".//isda:solar_incidence"),
    )


def load_memmap(p: Product) -> np.ndarray:
    expected = p.lines * p.samples
    actual = p.img_path.stat().st_size
    assert actual == expected, (
        f"size mismatch for {p.img_path.name}: "
        f"file={actual} bytes, expected={expected} (Lines*Samples uint8)"
    )
    return np.memmap(p.img_path, dtype=np.uint8, mode="r", shape=(p.lines, p.samples))


def full_histogram(arr: np.ndarray, sample_every: int = 8) -> np.ndarray:
    """Histogram of DN 0..255 over a strided subsample (fast, low memory)."""
    sub = arr[::sample_every, ::sample_every]
    return np.bincount(sub.ravel(), minlength=256).astype(np.int64)


def dark_band_stats(arr: np.ndarray, band_rows: int = 4096) -> tuple[np.ndarray, tuple[int, int]]:
    """
    Find the darkest contiguous band of `band_rows` rows by mean DN.
    Returns (histogram_over_that_band, (row_start, row_stop)).
    The darkest band is the best proxy for PSR / heavily shadowed terrain
    without doing georeferencing.
    """
    n_rows = arr.shape[0]
    if n_rows <= band_rows:
        means = arr.mean(axis=1)
        return np.bincount(arr.ravel(), minlength=256).astype(np.int64), (0, n_rows)

    # Coarse row-mean using a stride to keep it fast on ~1GB arrays.
    stride = max(1, n_rows // 4000)
    row_means = arr[::stride, ::16].mean(axis=1)
    # Convolve with a box of width `band_rows/stride` to find darkest window.
    win = max(1, band_rows // stride)
    kernel = np.ones(win) / win
    smoothed = np.convolve(row_means, kernel, mode="valid")
    start_in_strided = int(np.argmin(smoothed))
    row_start = start_in_strided * stride
    row_stop = min(row_start + band_rows, n_rows)

    band = arr[row_start:row_stop, :]
    hist = np.bincount(band.ravel(), minlength=256).astype(np.int64)
    return hist, (row_start, row_stop)


def verdict(dark_hist: np.ndarray) -> str:
    total = dark_hist.sum()
    if total == 0:
        return "UNKNOWN (empty band)"
    low_mass = dark_hist[:3].sum() / total          # DN 0..2
    spread_mass = dark_hist[2:26].sum() / total     # DN 2..25
    mean_dn = (np.arange(256) * dark_hist).sum() / total

    if low_mass > 0.95 and mean_dn < 1.5:
        return (
            f"CRUSHED -- {low_mass*100:.1f}% of dark-band pixels at DN<=2, "
            f"mean DN={mean_dn:.2f}. Data not usable for PSR work."
        )
    if spread_mass > 0.5 and mean_dn > 2:
        return (
            f"USABLE -- dark-band mean DN={mean_dn:.2f}, "
            f"{spread_mass*100:.1f}% of mass in DN 2..25. Low-signal detail preserved."
        )
    return (
        f"MARGINAL -- dark-band mean DN={mean_dn:.2f}, "
        f"DN<=2 mass={low_mass*100:.1f}%, DN 2..25 mass={spread_mass*100:.1f}%."
    )


def render(p: Product, full_hist: np.ndarray, dark_hist: np.ndarray,
           band: tuple[int, int], arr: np.ndarray) -> Path:
    fig = plt.figure(figsize=(14, 9))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.2])

    ax_meta = fig.add_subplot(gs[0, 0])
    ax_meta.axis("off")
    meta_lines = [
        f"Product: {p.name}",
        f"Lines x Samples: {p.lines} x {p.samples}",
        "",
        f"bits_selection      : {p.bits_selection}",
        f"tdi_stages          : {p.tdi_stages}",
        f"line_exposure (ms)  : {p.line_exposure_duration}",
        f"spacecraft_alt (km) : {p.spacecraft_altitude}",
        f"pixel_res (m/pix)   : {p.pixel_resolution}",
        f"solar_incidence (d) : {p.solar_incidence}",
        "",
        f"dark band rows      : {band[0]} .. {band[1]}",
        f"verdict             : {verdict(dark_hist)}",
    ]
    ax_meta.text(0, 1, "\n".join(meta_lines), va="top", family="monospace", fontsize=10)

    ax_full = fig.add_subplot(gs[0, 1])
    ax_full.bar(np.arange(256), full_hist, width=1.0, color="#1f77b4")
    ax_full.set_yscale("log")
    ax_full.set_xlim(0, 255)
    ax_full.set_xlabel("DN (8-bit)")
    ax_full.set_ylabel("count (log)")
    ax_full.set_title("Full-frame DN histogram (strided sample)")

    ax_dark = fig.add_subplot(gs[1, 0])
    ax_dark.bar(np.arange(64), dark_hist[:64], width=1.0, color="#d62728")
    ax_dark.set_xlim(-0.5, 63.5)
    ax_dark.set_xlabel("DN (8-bit), 0..63")
    ax_dark.set_ylabel("count")
    ax_dark.set_title(f"Dark-band DN histogram (rows {band[0]}..{band[1]}) -- PSR proxy")

    ax_img = fig.add_subplot(gs[1, 1])
    thumb = arr[::max(1, p.lines // 800), ::max(1, p.samples // 800)]
    ax_img.imshow(thumb, cmap="gray", vmin=0, vmax=max(8, int(thumb.mean() * 3)))
    ax_img.set_title("Thumbnail (auto-stretched to dark end)")
    ax_img.axhline(band[0] / max(1, p.lines // 800), color="red", lw=0.5)
    ax_img.axhline(band[1] / max(1, p.lines // 800), color="red", lw=0.5)
    ax_img.set_xticks([])
    ax_img.set_yticks([])

    fig.tight_layout()
    out_path = OUT / f"{p.name}_diagnostic.png"
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def main() -> None:
    xmls = [
        ROOT / "data/ch2_ohr_nrp_20211223T0019163816_d_img_d32/data/raw/20211223/ch2_ohr_nrp_20211223T0019163816_d_img_d32.xml",
        ROOT / "data/ch2_ohr_nrp_20211222T2023166276_d_img_d32/data/raw/20211222/ch2_ohr_nrp_20211222T2023166276_d_img_d32.xml",
    ]

    report_lines = []
    for xml in xmls:
        p = parse_xml(xml)
        arr = load_memmap(p)
        full_hist = full_histogram(arr)
        dark_hist, band = dark_band_stats(arr)
        v = verdict(dark_hist)
        out = render(p, full_hist, dark_hist, band, arr)

        block = [
            "=" * 78,
            f"product            : {p.name}",
            f"img dims           : {p.lines} x {p.samples} ({arr.nbytes/1e6:.1f} MB on disk)",
            f"bits_selection     : {p.bits_selection}",
            f"tdi_stages         : {p.tdi_stages}",
            f"line_exposure      : {p.line_exposure_duration} ms",
            f"spacecraft_altitude: {p.spacecraft_altitude} km",
            f"pixel_resolution   : {p.pixel_resolution} m/pix",
            f"solar_incidence    : {p.solar_incidence} deg",
            f"dark band rows     : {band[0]} .. {band[1]}",
            f"verdict            : {v}",
            f"plot               : {out}",
            "DN<=2 count (full)   : " + str(int(full_hist[:3].sum())),
            "DN<=2 count (dark)   : " + str(int(dark_hist[:3].sum())),
            "DN 3..25 count (dark): " + str(int(dark_hist[3:26].sum())),
            "DN>=250 count (full) : " + str(int(full_hist[250:].sum())),
        ]
        report_lines.extend(block)
        print("\n".join(block))

    (OUT / "report.txt").write_text("\n".join(report_lines) + "\n")
    print(f"\nReport written to {OUT/'report.txt'}")


if __name__ == "__main__":
    main()
