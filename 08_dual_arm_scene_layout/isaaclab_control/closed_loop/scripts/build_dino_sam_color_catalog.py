#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

HERE = Path(__file__).resolve()
CLOSED_LOOP = HERE.parents[1]
if str(CLOSED_LOOP) not in sys.path:
    sys.path.insert(0, str(CLOSED_LOOP))

from color_sort_dino_sam.catalog import build_trusted_color_catalog

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--capture-root", type=Path, required=True)
    p.add_argument("--color-root", type=Path, required=True)
    p.add_argument("--grounded-sam-result", type=Path, required=True)
    p.add_argument("--source-zone-depth-mask", type=Path, required=True)
    p.add_argument("--requested-color", choices=("red","blue"), required=True)
    p.add_argument("--duplicate-iou-threshold", type=float, default=0.85)
    p.add_argument("--minimum-source-valid-depth-pixels", type=int, default=100)
    p.add_argument("--minimum-source-fraction", type=float, default=0.80)
    p.add_argument("--removal-expand-px", type=int, default=2)
    a = p.parse_args()

    result = build_trusted_color_catalog(
        capture_root=a.capture_root,
        color_root=a.color_root,
        grounded_sam_result_path=a.grounded_sam_result,
        source_zone_depth_mask_path=a.source_zone_depth_mask,
        requested_color=a.requested_color,
        duplicate_iou_threshold=a.duplicate_iou_threshold,
        minimum_source_valid_depth_pixels=a.minimum_source_valid_depth_pixels,
        minimum_source_fraction=a.minimum_source_fraction,
        removal_expand_px=a.removal_expand_px,
    )
    print(json.dumps({
        "status": result["status"],
        "trusted_object_count": result["trusted_object_count"],
        "catalog": result["catalog"],
    }, ensure_ascii=False))

if __name__ == "__main__":
    main()
