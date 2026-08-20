#!/usr/bin/env python3
"""Compare source scene manifest poses against a Persistent Isaac capture.

This is an offline diagnostic.  It does not start Isaac and does not modify
the scene.  It reports whether objects moved during the initial HOME hold after
Persistent Isaac spawned them into the calibrated SourceZone.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def load_json(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def transform_from_layout_source(layout: dict) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, 3] = np.asarray(
        layout["transforms"]["source_zone"]["position_world_m"], dtype=np.float64
    )
    return transform


def pose_xyz(row: dict) -> np.ndarray:
    value = row.get("T_world_centered_object", row.get("pose_world_object"))
    if value is None:
        raise KeyError("object row has no T_world_centered_object/pose_world_object")
    return np.asarray(value, dtype=np.float64)[:3, 3]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-manifest", type=Path, required=True)
    parser.add_argument("--capture-dir", type=Path, required=True)
    parser.add_argument(
        "--layout",
        type=Path,
        default=Path("08_dual_arm_scene_layout/config/manual_layout_calibrated.json"),
    )
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    source = load_json(args.scene_manifest)
    settled = load_json(args.capture_dir / "settled_scene_manifest.json")
    layout = load_json(args.layout)
    world_from_source = transform_from_layout_source(layout)
    table_center = np.asarray(layout["transforms"]["table"]["position_world_m"], dtype=np.float64)
    table_size = np.asarray(layout["geometry"]["table_size_m"], dtype=np.float64)
    table_top = float(table_center[2] + 0.5 * table_size[2])

    settled_by_seg = {
        int(row["segmentation_id"]): row for row in settled.get("objects", [])
    }
    rows = []
    print("[SPAWN VS SETTLED]")
    print(f"table_top_world_z = {table_top:.9f}")
    print("seg | dz_root_mm | dxy_root_mm | initial_world_z | settled_world_z | object")
    for row in source.get("objects", []):
        seg = int(row["segmentation_id"])
        if seg not in settled_by_seg:
            continue
        initial_local = pose_xyz(row)
        settled_local = pose_xyz(settled_by_seg[seg])
        initial_world = (world_from_source @ np.r_[initial_local, 1.0])[:3]
        settled_world = (world_from_source @ np.r_[settled_local, 1.0])[:3]
        delta = settled_world - initial_world
        dz_mm = 1000.0 * float(delta[2])
        dxy_mm = 1000.0 * float(np.linalg.norm(delta[:2]))
        out = {
            "segmentation_id": seg,
            "object_code": row.get("object_code"),
            "initial_local_xyz_m": initial_local.tolist(),
            "settled_local_xyz_m": settled_local.tolist(),
            "initial_world_xyz_m": initial_world.tolist(),
            "settled_world_xyz_m": settled_world.tolist(),
            "delta_world_xyz_m": delta.tolist(),
            "delta_z_mm": dz_mm,
            "delta_xy_mm": dxy_mm,
            "settled_root_minus_table_top_mm": 1000.0 * float(settled_world[2] - table_top),
        }
        rows.append(out)
        print(
            f"{seg:3d} | {dz_mm:+9.2f} | {dxy_mm:10.2f} | "
            f"{initial_world[2]:.6f} | {settled_world[2]:.6f} | "
            f"{row.get('object_code')}"
        )

    report = {
        "schema_version": 1,
        "scene_manifest": str(Path(args.scene_manifest).resolve()),
        "capture_dir": str(Path(args.capture_dir).resolve()),
        "layout": str(Path(args.layout).resolve()),
        "table_top_world_z_m": table_top,
        "world_from_source_zone": world_from_source.tolist(),
        "objects": rows,
    }
    output = args.output or (args.capture_dir / "spawn_vs_settled_audit.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"report = {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
