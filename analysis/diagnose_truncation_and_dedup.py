"""
Two pre-design checks:

1. Is XML <file_size> actually equal to the .img file size on disk for
   known-good downloads? If they differ by a constant (header) we need to
   correct for it before using `file_size != img_size_actual` as a
   truncation signal.

2. For dual-station observations (same timestamp, different ground
   station suffix), are the .img bytes identical? If not, "same
   observation" is only a catalog-convenience label, and we should keep
   the better-quality copy rather than dropping one arbitrarily.

Outputs a short report. Designed to run against the SSD where the 312
unzipped products live.
"""

from __future__ import annotations

import hashlib
import re
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

DATA_ROOT = Path("/Volumes/lazarus/ohrc-data")
NS = {"isda": "https://isda.issdc.gov.in/pds4/isda/v1"}


def parse_xml_size(xml_path: Path) -> tuple[int | None, str | None, str | None]:
    """Return (declared_file_size, declared_md5, declared_file_name)."""
    try:
        root = ET.parse(xml_path).getroot()
    except ET.ParseError:
        return None, None, None
    declared_size = None
    declared_md5 = None
    declared_name = None
    for elem in root.iter():
        tag = elem.tag.split("}", 1)[-1]
        if tag == "file_size" and elem.text:
            try:
                declared_size = int(elem.text.strip())
            except ValueError:
                pass
        elif tag == "md5_checksum" and elem.text:
            declared_md5 = elem.text.strip()
        elif tag == "file_name" and elem.text:
            declared_name = elem.text.strip()
    return declared_size, declared_md5, declared_name


def obs_id(product_id: str) -> str:
    """Strip the trailing _<station> token. e.g. ch2_ohr_nrp_<ts>_d_img_d18 -> ch2_ohr_nrp_<ts>_d_img."""
    return re.sub(r"_(d\d{2}|n\d{2}|g\d{2}|gds|cnb|hw\d|hb\d|m\d{2}|[a-z]{3}\d?)$", "", product_id)


def md5_of(path: Path, max_bytes: int | None = None) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        while True:
            chunk = f.read(1 << 20)
            if not chunk:
                break
            h.update(chunk)
            if max_bytes is not None and f.tell() >= max_bytes:
                break
    return h.hexdigest()


def main() -> None:
    xmls = sorted(DATA_ROOT.rglob("data/raw/*/ch2_ohr_*.xml"))
    print(f"found {len(xmls)} XMLs")

    products = []
    for xml in xmls:
        declared_size, declared_md5, declared_name = parse_xml_size(xml)
        if declared_name is None:
            continue
        img = xml.with_name(declared_name)
        actual = img.stat().st_size if img.exists() else None
        products.append({
            "xml": xml,
            "product_id": xml.stem,
            "declared_size": declared_size,
            "declared_md5": declared_md5,
            "img_path": img if img.exists() else None,
            "actual_size": actual,
        })

    print(f"parsed {len(products)} products with declared file_name\n")

    # --- Check 1: file_size == actual size? ---
    print("=== check 1: declared file_size vs actual .img size ===")
    mismatches = []
    missing = []
    for p in products:
        if p["img_path"] is None:
            missing.append(p)
            continue
        if p["declared_size"] != p["actual_size"]:
            mismatches.append(p)
    print(f"products with no .img on disk:                 {len(missing)}")
    print(f"products where declared_size != actual_size:   {len(mismatches)}")
    if mismatches:
        for p in mismatches[:10]:
            diff = (p["actual_size"] or 0) - (p["declared_size"] or 0)
            print(f"  {p['product_id']}: declared={p['declared_size']} actual={p['actual_size']} diff={diff:+d}")
        # Is the diff constant? (header offset)
        diffs = {(p["actual_size"] or 0) - (p["declared_size"] or 0) for p in mismatches}
        print(f"  unique diff values across {len(mismatches)} mismatches: {sorted(diffs)[:5]}{'...' if len(diffs) > 5 else ''}")
    else:
        print("  good: file_size in XML matches the .img on disk exactly.")

    # --- Check 2: dual-station observations, same bytes? ---
    print("\n=== check 2: dual-station observations ===")
    by_obs = defaultdict(list)
    for p in products:
        if p["img_path"] is None:
            continue
        by_obs[obs_id(p["product_id"])].append(p)
    dupes = [(obs, plist) for obs, plist in by_obs.items() if len(plist) > 1]
    print(f"total unique observations (by timestamp+role): {len(by_obs)}")
    print(f"observations downlinked through 2+ stations:   {len(dupes)}")

    if dupes:
        # For each duplicate group, compare sizes and (for a sample) md5s
        same_size = 0
        diff_size = 0
        md5_checked = 0
        md5_match = 0
        for obs, plist in dupes:
            sizes = {p["actual_size"] for p in plist}
            if len(sizes) == 1:
                same_size += 1
            else:
                diff_size += 1
        print(f"  groups with all-equal byte sizes:            {same_size}")
        print(f"  groups with differing byte sizes:            {diff_size}")

        # MD5-compare the first 5 same-size groups
        sample = [pl for _, pl in dupes if len({p['actual_size'] for p in pl}) == 1][:5]
        print(f"\n  hashing first .img of each variant in {len(sample)} same-size groups (full md5):")
        for plist in sample:
            md5s = []
            for p in plist:
                md5s.append((p["product_id"].split("_")[-1], md5_of(p["img_path"])))
            match = len({m for _, m in md5s}) == 1
            mark = "EQUAL" if match else "DIFFER"
            print(f"    {plist[0]['product_id'][:50]}... {[s for s, _ in md5s]}  -> {mark}")

        # Show a couple of different-size groups for inspection
        diff_groups = [pl for _, pl in dupes if len({p['actual_size'] for p in pl}) > 1]
        if diff_groups:
            print(f"\n  examples of different-size dual-station groups:")
            for plist in diff_groups[:5]:
                print(f"    obs_id={obs_id(plist[0]['product_id'])}")
                for p in plist:
                    print(f"      {p['product_id']:>55s}  size={p['actual_size']}  declared={p['declared_size']}")


if __name__ == "__main__":
    main()
