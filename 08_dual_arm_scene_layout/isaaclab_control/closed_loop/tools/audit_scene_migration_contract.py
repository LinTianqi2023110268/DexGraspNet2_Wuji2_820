#!/usr/bin/env python3
"""Read-only four-layer audit for training scene -> Persistent Isaac migration.

This tool never starts Isaac.  It compares a source ``scene_manifest.json``
with the zero-physics-step and settled artifacts emitted by the persistent
worker.  The live worker remains the authority for USD prim transforms and
collision AABBs; this script makes its contract easy to review afterwards.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def load(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def pose(row: dict, key: str) -> np.ndarray:
    return np.asarray(row[key], dtype=np.float64)


def pose_error(expected: np.ndarray, actual: np.ndarray) -> tuple[float, float]:
    expected = np.asarray(expected, dtype=np.float64)
    actual = np.asarray(actual, dtype=np.float64)
    position_mm = 1000.0 * float(np.linalg.norm(expected[:3, 3] - actual[:3, 3]))
    relative = expected[:3, :3].T @ actual[:3, :3]
    cosine = float(np.clip((np.trace(relative) - 1.0) * 0.5, -1.0, 1.0))
    return position_mm, float(np.rad2deg(np.arccos(cosine)))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-manifest", type=Path, required=True)
    parser.add_argument("--capture-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--task-object-collision-policy",
        choices=("persistent_filtered", "training_default"),
        default=None,
        help="Required only for legacy audit artifacts which predate the policy field.",
    )
    args = parser.parse_args()

    source = load(args.scene_manifest)
    capture = Path(args.capture_dir)
    initial_path = capture / "initial_spawn_scene_manifest.json"
    audit_path = capture / "scene_migration_spawn_audit.json"
    settled_path = capture / "settled_scene_manifest.json"
    settle_trace_path = capture / "physics_settle_trace.json"
    missing = [str(path) for path in (initial_path, audit_path) if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Missing zero-step migration artifacts. Re-run capture with the audited worker: " + ", ".join(missing)
        )
    initial = load(initial_path)
    spawn = load(audit_path)
    settled = load(settled_path) if settled_path.is_file() else None
    settle_trace = load(settle_trace_path) if settle_trace_path.is_file() else None
    source_by_seg = {int(row["segmentation_id"]): row for row in source["objects"]}
    initial_by_seg = {int(row["segmentation_id"]): row for row in initial["objects"]}
    settled_by_seg = {} if settled is None else {int(row["segmentation_id"]): row for row in settled["objects"]}

    rows = []
    expected_xyz, actual_xyz = [], []
    for seg, source_row in source_by_seg.items():
        current = initial_by_seg.get(seg)
        if current is None:
            rows.append({"segmentation_id": seg, "status": "MISSING_INITIAL_OBJECT"})
            continue
        expected = pose(current, "expected_world_pose")
        # The migration contract is an authored USD frame contract.  Use the
        # rigid-root transform evaluated from USD, matching the live worker's
        # pre-physics guard.  Legacy artifacts can fall back to their runtime
        # cache, but new artifacts always provide the authoritative field.
        actual = pose(
            current,
            "actual_rigid_body_world_pose"
            if "actual_rigid_body_world_pose" in current
            else "actual_rigid_body_runtime_pose",
        )
        runtime_initial = pose(current, "actual_rigid_body_runtime_pose")
        pre_physics_mm, pre_physics_deg = pose_error(expected, actual)
        runtime_diagnostic_mm, runtime_diagnostic_deg = pose_error(expected, runtime_initial)
        expected_xyz.append(expected[:3, 3])
        actual_xyz.append(actual[:3, 3])
        settled_row = settled_by_seg.get(seg)
        drift_mm = None
        drift_at_final_mm = None
        drift_samples_mm = []
        if settle_trace is not None:
            for sample in settle_trace.get("samples", []):
                sample_row = next(
                    (item for item in sample.get("objects", []) if int(item["segmentation_id"]) == seg),
                    None,
                )
                if sample_row is not None:
                    xyz = np.asarray(sample_row["position_world_m"], dtype=np.float64)
                    drift_samples_mm.append({
                        "time_s": float(sample["time_s"]),
                        "translation_drift_mm": 1000.0 * float(np.linalg.norm(xyz - runtime_initial[:3, 3])),
                    })
        if drift_samples_mm:
            drift_mm = max(item["translation_drift_mm"] for item in drift_samples_mm)
            drift_at_final_mm = drift_samples_mm[-1]["translation_drift_mm"]
        elif settled_row is not None:
            settled_world = pose(settled_row, "actual_rigid_body_runtime_pose")
            drift_mm = 1000.0 * float(np.linalg.norm(settled_world[:3, 3] - runtime_initial[:3, 3]))
        rows.append({
            "segmentation_id": seg,
            "object_code": source_row.get("object_code"),
            "reference_root": current.get("root_path"),
            "rigid_body_root": current.get("rigid_path"),
            "T_reference_root_rigid_body": current.get("T_reference_root_rigid_body"),
            "pre_physics_pose_error_mm": pre_physics_mm,
            "pre_physics_pose_error_deg": pre_physics_deg,
            "runtime_pose_diagnostic_error_mm": runtime_diagnostic_mm,
            "runtime_pose_diagnostic_error_deg": runtime_diagnostic_deg,
            "visual_AABB_world": current.get("visual_AABB_world"),
            "collision_AABB_world": current.get("collision_AABB_world"),
            "asset_resolution": current.get("asset_resolution"),
            "source_mesh_hash_verified": current.get("source_mesh_hash_verified"),
            "settling_translation_drift_mm": drift_mm,
            "settling_translation_drift_at_final_sample_mm": drift_at_final_mm,
            "settling_translation_trace_mm": drift_samples_mm,
        })

    pairwise_mm = None
    if len(expected_xyz) >= 2:
        expected_xyz = np.stack(expected_xyz)
        actual_xyz = np.stack(actual_xyz)
        a = np.linalg.norm(expected_xyz[:, None] - expected_xyz[None, :], axis=-1)
        b = np.linalg.norm(actual_xyz[:, None] - actual_xyz[None, :], axis=-1)
        pairwise_mm = 1000.0 * float(np.max(np.abs(a - b)))

    report = {
        "schema_version": 1,
        "scene_manifest": str(args.scene_manifest.resolve()),
        "capture_dir": str(capture.resolve()),
        "training_scene_size_xy_m": source["table"]["paper_size_m"],
        "source_zone_visual_size_xy_m": spawn["source_zone_visual_size_xy_m"],
        "source_zone_scale_used_for_scene_mapping": False,
        "source_zone_physical_collision_support": False,
        "table_physical_collision_support": True,
        "task_object_collision_policy": args.task_object_collision_policy or spawn.get(
            "task_object_collision_policy",
            initial.get("task_object_collision_policy"),
        ),
        "pre_physics": {
            "status": spawn["status"],
            "errors": spawn["errors"],
            "max_pairwise_distance_error_mm": pairwise_mm,
        },
        "settled_available": settled is not None,
        "physics_settle_trace_available": settle_trace is not None,
        "max_settling_drift_mm": max(
            (row["settling_translation_drift_mm"] or 0.0 for row in rows),
            default=None,
        ),
        "max_final_settling_drift_mm": max(
            (row["settling_translation_drift_at_final_sample_mm"] or 0.0 for row in rows),
            default=None,
        ),
        "objects": rows,
    }
    output = args.output or capture / "scene_migration_contract_report.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("=" * 50)
    print(" SCENE MIGRATION CONTRACT")
    print("=" * 50)
    print(f"scene                  : {Path(args.scene_manifest).parent.name}")
    print("scene size             : " + " x ".join(f"{float(v):.3f}" for v in report["training_scene_size_xy_m"]) + " m")
    print("SourceZone size        : " + " x ".join(f"{float(v):.3f}" for v in report["source_zone_visual_size_xy_m"]) + " m")
    print("SourceZone scale used  : NO")
    print(f"pre-physics            : {spawn['status']}")
    print(f"object collision policy: {report['task_object_collision_policy']}")
    print(f"max pairwise error     : {pairwise_mm if pairwise_mm is not None else float('nan'):.6f} mm")
    print(f"report                 : {output}")
    return 0 if spawn["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
