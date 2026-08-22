#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

HERE = Path(__file__).resolve()
CLOSED_LOOP = HERE.parents[1]
if str(CLOSED_LOOP) not in sys.path:
    sys.path.insert(0, str(CLOSED_LOOP))

from color_sort_dino_sam.spatial_context import build_spatial_context

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--capture-root", type=Path, required=True)
    p.add_argument("--filtered-depth", type=Path, required=True)
    p.add_argument("--rgb-no-robot", type=Path, required=True)
    p.add_argument("--layout", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--context-margin-xy-m", type=float, default=0.20)
    a = p.parse_args()
    artifacts = build_spatial_context(
        capture_root=a.capture_root,
        filtered_depth_path=a.filtered_depth,
        rgb_no_robot_path=a.rgb_no_robot,
        layout_path=a.layout,
        output_dir=a.output_dir,
        context_margin_xy_m=a.context_margin_xy_m,
    )
    print(json.dumps({"status":"PASS", **artifacts.to_jsonable()}, ensure_ascii=False))

if __name__ == "__main__":
    main()
