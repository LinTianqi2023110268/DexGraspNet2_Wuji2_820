#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np


HERE = Path(__file__).resolve()
CLOSED_LOOP = HERE.parents[1]
if str(CLOSED_LOOP) not in sys.path:
    sys.path.insert(0, str(CLOSED_LOOP))

from perception.target_removal_mask import write_target_removal_artifacts  # noqa: E402


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_sam_mask_from_source(source: str) -> np.ndarray:
    if "#" not in source:
        return np.load(Path(source).expanduser().resolve()).astype(bool)

    archive_text, row_text = source.rsplit("#", 1)
    archive_path = Path(archive_text).expanduser().resolve()
    if not archive_path.is_file():
        raise FileNotFoundError(archive_path)

    row = int(row_text)
    with np.load(archive_path, allow_pickle=False) as z:
        masks = np.asarray(z["masks"], dtype=bool)
    if row < 0 or row >= len(masks):
        raise IndexError(f"mask archive row out of range: {archive_path}#{row}")
    return np.asarray(masks[row], dtype=bool)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Offline replay for color-sort grasp/removal mask split."
    )
    parser.add_argument(
        "--cycle-root",
        type=Path,
        required=True,
        help="closed_loop_sessions/<session>/cycle_XXX",
    )
    parser.add_argument(
        "--target-id",
        default=None,
        help="color target id, e.g. red_target_000_red_003. Defaults to selected_target.json.",
    )
    parser.add_argument("--sam-expand-radius", type=int, default=0)
    parser.add_argument("--hsv-expand-radius", type=int, default=12)
    parser.add_argument("--depth-threshold-m", type=float, default=0.03)
    parser.add_argument("--depth-percentile-low", type=float, default=1.0)
    parser.add_argument("--depth-percentile-high", type=float, default=99.0)
    parser.add_argument("--depth-percentile-padding-m", type=float, default=0.01)
    parser.add_argument("--supplement-expand-radius-px", type=int, default=12)
    args = parser.parse_args()

    cycle_root = args.cycle_root.expanduser().resolve()
    capture_root = cycle_root / "capture"
    color_root = cycle_root / "color_sort"
    selected_path = color_root / "selected_target.json"
    target_pool_path = color_root / "target_pool.json"
    if not target_pool_path.is_file():
        raise FileNotFoundError(target_pool_path)

    target_id = args.target_id
    if target_id is None:
        if not selected_path.is_file():
            raise FileNotFoundError(selected_path)
        selected = _load_json(selected_path)
        target_id = str(selected["instance_id"])

    pool = _load_json(target_pool_path)
    rows = [row for row in pool["targets"] if str(row["target_id"]) == target_id]
    if not rows:
        raise RuntimeError(f"target id not found in pool: {target_id}")
    target = rows[0]
    metrics = dict(target.get("metrics", {}))

    sam_source = str(metrics["proposal_mask_source"])
    hsv_source_value = (
        metrics.get("hsv_mask_path")
        or metrics.get("hsv_instance_mask_path")
        or target.get("mask_path")
    )
    if not hsv_source_value:
        raise RuntimeError("target pool row does not contain an HSV mask path")
    hsv_source = Path(str(hsv_source_value)).expanduser().resolve()
    depth_source = capture_root / "planning/filtered_depth.npy"
    if not hsv_source.is_file():
        raise FileNotFoundError(hsv_source)
    if not depth_source.is_file():
        raise FileNotFoundError(depth_source)

    sam_mask = _load_sam_mask_from_source(sam_source)
    hsv_mask = np.load(hsv_source).astype(bool)
    depth = np.load(depth_source).astype(np.float32)

    target_dir = color_root / "targets" / target_id
    target_dir.mkdir(parents=True, exist_ok=True)
    target_audit = write_target_removal_artifacts(
        output_dir=target_dir,
        sam_mask=sam_mask,
        hsv_mask=hsv_mask,
        depth_m=depth,
        sam_source=sam_source,
        hsv_source=hsv_source,
        depth_source=depth_source,
        sam_expand_radius=args.sam_expand_radius,
        hsv_expand_radius=args.hsv_expand_radius,
        depth_threshold_m=args.depth_threshold_m,
        depth_percentile_low=args.depth_percentile_low,
        depth_percentile_high=args.depth_percentile_high,
        depth_percentile_padding_m=args.depth_percentile_padding_m,
        supplement_expand_radius_px=args.supplement_expand_radius_px,
    )

    planning_dir = capture_root / "planning"
    planning_audit = write_target_removal_artifacts(
        output_dir=planning_dir,
        sam_mask=sam_mask,
        hsv_mask=hsv_mask,
        depth_m=depth,
        sam_source=sam_source,
        hsv_source=hsv_source,
        depth_source=depth_source,
        sam_expand_radius=args.sam_expand_radius,
        hsv_expand_radius=args.hsv_expand_radius,
        depth_threshold_m=args.depth_threshold_m,
        depth_percentile_low=args.depth_percentile_low,
        depth_percentile_high=args.depth_percentile_high,
        depth_percentile_padding_m=args.depth_percentile_padding_m,
        supplement_expand_radius_px=args.supplement_expand_radius_px,
    )

    legacy_hsv_removal = hsv_mask
    legacy_leftover = int(np.count_nonzero(sam_mask & ~legacy_hsv_removal))
    legacy_fraction = float(legacy_leftover / max(1, np.count_nonzero(sam_mask)))
    old_core_pixels = int(planning_audit["core_removal_pixels"])
    old_core_leftover = int(planning_audit["core_sam_leftover_pixels"])
    old_core_fraction = float(planning_audit["core_sam_leftover_fraction"])

    report = {
        "schema_version": 1,
        "cycle_root": str(cycle_root),
        "target_id": target_id,
        "legacy_hsv_behavior": {
            "removal_mask": "HSV selected instance mask",
            "removal_pixels": int(np.count_nonzero(legacy_hsv_removal)),
            "sam_leftover_pixels": legacy_leftover,
            "sam_leftover_fraction": legacy_fraction,
        },
        "old_behavior": {
            "removal_mask": "core = SAM & HSV_neighbourhood & adaptive_depth_gate",
            "removal_pixels": old_core_pixels,
            "sam_leftover_pixels": old_core_leftover,
            "sam_leftover_fraction": old_core_fraction,
        },
        "new_behavior": {
            "target_grasp_mask": planning_audit["target_grasp_mask"],
            "target_removal_mask": planning_audit["target_removal_mask"],
            "target_removal_audit": planning_audit["target_removal_audit"],
            "supplement_pixels": int(planning_audit["supplement_pixels"]),
            "sam_leftover_pixels": int(planning_audit["sam_leftover_pixels"]),
            "sam_leftover_fraction": float(planning_audit["sam_leftover_fraction"]),
            "removal_pixels": int(planning_audit["removal_pixels"]),
            "removal_over_sam_fraction": float(planning_audit["removal_over_sam_fraction"]),
        },
        "target_dir_audit": target_audit["target_removal_audit"],
    }
    report_path = planning_dir / "target_removal_replay_report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print("TARGET REMOVAL MASK REPLAY")
    print("==========================")
    print(f"cycle             : {cycle_root}")
    print(f"target            : {target_id}")
    print(f"SAM pixels        : {planning_audit['sam_pixels']}")
    print(f"HSV pixels        : {planning_audit['hsv_pixels']}")
    print(
        f"old core removal  : {old_core_pixels} px | "
        f"SAM leftover={old_core_leftover} ({old_core_fraction:.3%})"
    )
    print(
        "new removal       : "
        f"{planning_audit['removal_pixels']} px | "
        f"SAM leftover={planning_audit['sam_leftover_pixels']} "
        f"({planning_audit['sam_leftover_fraction']:.3%})"
    )
    print(f"supplement        : {planning_audit['supplement_pixels']} px")
    print(f"grasp mask        : {planning_audit['target_grasp_mask']}")
    print(f"removal mask      : {planning_audit['target_removal_mask']}")
    print(f"audit             : {planning_audit['target_removal_audit']}")
    print(f"report            : {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
