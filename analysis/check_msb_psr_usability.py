"""
Run the bits_selection diagnostic against MSB PSR strips.

The Gate-0 inventory shows the corpus is 97% MSB-encoded. Before
concluding "MSB crushes the dark band and PSR work is blocked," we need
to look at the actual pixels. This script imports the diagnostic
functions from `diagnose_bits_selection.py` and runs them on three MSB
strips with solar_incidence >= 90 (true PSR condition) at polar
latitudes — exactly the strips the project would otherwise need.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from analysis.diagnose_bits_selection import (  # noqa: E402
    dark_band_stats,
    full_histogram,
    load_memmap,
    parse_xml,
    render,
    verdict,
)

DATA_ROOT = Path("/Volumes/lazarus/ohrc-data")
PICKS = [
    DATA_ROOT / "ch2_ohr_nrp_20251007T0502258606_d_img_d18/data/raw/20251007/ch2_ohr_nrp_20251007T0502258606_d_img_d18.xml",
    DATA_ROOT / "ch2_ohr_nrp_20240229T0921574793_d_img_d18/data/raw/20240229/ch2_ohr_nrp_20240229T0921574793_d_img_d18.xml",
    DATA_ROOT / "ch2_ohr_nrp_20240229T0921593215_d_img_d18/data/raw/20240229/ch2_ohr_nrp_20240229T0921593215_d_img_d18.xml",
]


def main() -> None:
    out_lines = []
    for xml in PICKS:
        xml = xml.resolve()
        print(f"\n=== {xml.name} ===")
        p = parse_xml(xml)
        arr = load_memmap(p)
        full_hist = full_histogram(arr)
        dark_hist, band = dark_band_stats(arr)
        v = verdict(dark_hist)
        out = render(p, full_hist, dark_hist, band, arr)

        block = [
            f"product            : {p.name}",
            f"bits_selection     : {p.bits_selection}",
            f"tdi_stages         : {p.tdi_stages}",
            f"solar_incidence    : {p.solar_incidence} deg",
            f"dark band rows     : {band[0]} .. {band[1]}",
            f"DN<=2 count (dark) : {int(dark_hist[:3].sum())}",
            f"DN 3..25 count (dark): {int(dark_hist[3:26].sum())}",
            f"DN 26..63 count (dark): {int(dark_hist[26:64].sum())}",
            f"DN<=2 count (full) : {int(full_hist[:3].sum())}",
            f"DN>=250 count (full): {int(full_hist[250:].sum())}",
            f"verdict            : {v}",
            f"plot               : {out}",
        ]
        out_lines.extend(block + [""])
        print("\n".join(block))

    (REPO / "analysis" / "msb_psr_report.txt").write_text("\n".join(out_lines))


if __name__ == "__main__":
    main()
