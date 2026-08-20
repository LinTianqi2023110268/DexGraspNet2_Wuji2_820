#!/usr/bin/env python3
"""Build one generic live RGB-D LEAP candidate case without legacy mesh collision.

Selection has already been made by:
  GroundedSAM target membership -> descending official DGN2 score.

This adapter composes the official LEAP PREGRASP/COVER/GRASP/SQUEEZE/LIFT
waypoints for exactly one candidate, converts them from calibrated layout world
to SourceZone coordinates, and writes the existing case contract consumed by
the reviewed LEAP->Wuji2 retarget scripts.

NO reachability/collision PASS is asserted here.  Those are decided later by
cuRobo V2 using the retargeted Wuji2 geometry and observed RGB-D ESDF.
"""
from __future__ import annotations
import argparse, json, os, shutil, sys
from pathlib import Path
import numpy as np
import torch
from pytorch3d.transforms import matrix_to_euler_angles

PROJECT_ROOT = Path(__file__).resolve().parents[4]
PIPELINE_ROOT = PROJECT_ROOT / "06_leap_to_wuji2_final_pipeline"
CASES_ROOT = PIPELINE_ROOT / "01_cases/active"
OFFICIAL_ROOT = PROJECT_ROOT / "03_prediction_network/official_core"
ROOT_JOINT_NAMES = np.asarray(
    ["x_joint","y_joint","z_joint","x_rotation_joint","y_rotation_joint","z_rotation_joint"]
)
WAYPOINT_NAMES = np.asarray(["pregrasp","cover","grasp","squeeze","lift"])
OFFICIAL_CONTROL_STEPS = np.asarray([40,20,20,60], dtype=np.int64)

def transform_pose_left(T_left, poses):
    return np.asarray(T_left)[None] @ np.asarray(poses)

def sample_perception_proxy_surface(geometry: dict, output_path: Path) -> None:
    lower = np.asarray(geometry["robust_aabb_world_min_m"], dtype=np.float64)
    upper = np.asarray(geometry["robust_aabb_world_max_m"], dtype=np.float64)
    center = 0.5 * (lower + upper)
    dims = np.maximum(upper - lower, 0.01)
    rng = np.random.default_rng(0)
    unit = rng.uniform(-0.5, 0.5, size=(4096, 3))
    points = center[None, :] + unit * dims[None, :]
    np.save(output_path, np.asarray(points, dtype=np.float32))

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--case-id", required=True)
    p.add_argument(
        "--case-root",
        type=Path,
        help=(
            "Optional explicit case directory for closed-loop scratch screening. "
            "If omitted, writes under 01_cases/active/<case-id> for legacy compatibility."
        ),
    )
    p.add_argument("--candidate-index", type=int, required=True)
    p.add_argument("--prediction", type=Path, required=True)
    p.add_argument("--network-input", type=Path, required=True)
    p.add_argument("--capture-root", type=Path, required=True)
    p.add_argument("--settled-manifest", type=Path, required=True)
    p.add_argument("--sim-target-segmentation-id", type=int, default=None)
    p.add_argument("--target-geometry", type=Path, required=True)
    p.add_argument("--replace", action="store_true")
    a = p.parse_args()

    if "/" in a.case_id or a.case_id in {"",".",".."}:
        raise ValueError("invalid case id")
    if a.case_root is None:
        case_root = (CASES_ROOT / a.case_id).resolve()
        if case_root.parent != CASES_ROOT.resolve():
            raise RuntimeError("case escaped active root")
    else:
        case_root = a.case_root.expanduser().resolve()
        if case_root.name != a.case_id:
            raise RuntimeError(
                f"explicit --case-root must end with --case-id: {case_root} vs {a.case_id}"
            )
    if case_root.exists():
        if not a.replace:
            raise FileExistsError(case_root)
        shutil.rmtree(case_root)
    for rel in ("00_config","01_input","02_retargeting","03_root_alignment","04_squeeze",
                "05_visualization","06_isaacsim","07_arm_execution"):
        (case_root/rel).mkdir(parents=True, exist_ok=True)
    shutil.copy2(
        PIPELINE_ROOT/"00_shared/config/wuji2_native_width_mapper.json",
        case_root/"00_config/wuji2_native_width_mapper.json"
    )

    with np.load(a.prediction, allow_pickle=False) as z:
        pred = {k:z[k] for k in z.files}
    idx = int(a.candidate_index)
    target_order = set(map(int, np.asarray(pred["target_score_descending_candidate_index"]).tolist()))
    if idx not in target_order:
        raise RuntimeError(f"candidate {idx} is not a GroundedSAM target-seed proposal")
    joint_order = [str(x) for x in pred["leap_joint_order"].tolist()]
    qpos = np.asarray(pred["leap_qpos_rad"][idx], dtype=np.float32)
    grasp = {
        "rotation": np.asarray(pred["rotation_world"][[idx]], dtype=np.float32),
        "translation": np.asarray(pred["translation_world"][[idx]], dtype=np.float32),
        **{name: np.asarray([qpos[j]], dtype=np.float32) for j,name in enumerate(joint_order)}
    }

    old_cwd = Path.cwd()
    os.chdir(OFFICIAL_ROOT)
    sys.path.insert(0, str(OFFICIAL_ROOT))
    try:
        from src.eval.prepare_isaacsim5_job import ROOT_JOINT_NAMES as OFFICIAL_ROOT_NAMES, compose_waypoints
        from src.utils.robot_model import RobotModel
        from src.utils.width_mapper import WidthMapper
        robot = RobotModel(
            "robot_models/urdf/leap_hand_simplified.urdf",
            "robot_models/meta/leap_hand/meta.yaml",
        )
        if list(robot.joint_names) != joint_order:
            raise RuntimeError("LEAP joint order mismatch")
        mapper = WidthMapper(robot, "robot_models/meta/leap_hand/width_mapper_meta.yaml")
        waypoint_world, waypoint_qpos, _ = compose_waypoints(grasp, robot, mapper)
        official_root_names = np.asarray(OFFICIAL_ROOT_NAMES)
    finally:
        os.chdir(old_cwd)

    with np.load(a.network_input, allow_pickle=False) as z:
        source_from_world = np.asarray(z["source_from_world"], dtype=np.float64)
    waypoint_source = np.asarray(waypoint_world, dtype=np.float64).copy()
    for j in range(waypoint_source.shape[1]):
        waypoint_source[0,j] = source_from_world @ waypoint_source[0,j]
    pose = waypoint_source[0]
    root_dofs = np.concatenate([
        pose[:,:3,3],
        matrix_to_euler_angles(torch.as_tensor(pose[:,:3,:3]), "XYZ").numpy()
    ], axis=1).astype(np.float32)

    settled = json.loads(a.settled_manifest.read_text(encoding="utf-8"))
    target_geometry = json.loads(a.target_geometry.read_text(encoding="utf-8"))
    if bool(target_geometry.get("simulator_identity_used", True)):
        raise RuntimeError("target geometry used simulator identity")
    target_id = -1
    surface_path = case_root/"01_input/perception_target_surface_points.npy"
    sample_perception_proxy_surface(target_geometry, surface_path)
    T_world_target_anchor = np.asarray(
        target_geometry["T_world_target_anchor"], dtype=np.float64
    )
    T_source_target_anchor = source_from_world @ T_world_target_anchor

    frozen_input = case_root/"01_input/live_top_camera_network_input.npz"
    shutil.copy2(a.network_input, frozen_input)
    leap_job = case_root/"01_input/leap_official_waypoints.npz"
    np.savez_compressed(
        leap_job,
        waypoint_pose_world=waypoint_source.astype(np.float32),
        waypoint_pose_frame=np.asarray("SourceZone"),
        coordinate_convention=np.asarray("T_A_B maps coordinates from frame B into frame A"),
        waypoint_root_dofs=root_dofs[None],
        waypoint_joint_positions=np.asarray(waypoint_qpos, dtype=np.float32),
        waypoint_names=WAYPOINT_NAMES,
        waypoint_steps=OFFICIAL_CONTROL_STEPS,
        finger_joint_names=np.asarray(joint_order),
        root_joint_names=official_root_names,
        pregrasp_valid=np.asarray([False]),
        scene_penetration=np.asarray([np.nan], dtype=np.float32),
        table_penetration=np.asarray([np.nan], dtype=np.float32),
        source_candidate_index=np.asarray([idx], dtype=np.int64),
        score=np.asarray([pred["score"][idx]], dtype=np.float32),
        graspness=np.asarray([pred["graspness"][idx]], dtype=np.float32),
        log_prob=np.asarray([pred["log_prob"][idx]], dtype=np.float32),
        seed_point_world=np.asarray(pred["seed_point_world"][[idx]], dtype=np.float32),
        target_segmentation_id=np.asarray([target_id], dtype=np.int64),
    )

    objects = []
    for r in settled["objects"]:
        rr = dict(r)
        rr["code"] = str(r.get("code", r.get("object_code", "")))
        rr["pose_world_object"] = r["pose_world_object"]
        rr["surface_points"] = None
        objects.append(rr)
    objects.append({
        "segmentation_id": target_id,
        "object_pool_index": -1,
        "object_code": "perception_target_proxy",
        "code": "perception_target_proxy",
        "pose_world_object": T_source_target_anchor.tolist(),
        "surface_points": str(surface_path.resolve()),
        "perception_target_geometry": str(a.target_geometry.resolve()),
    })
    scene_manifest = {
        "schema_version": 2,
        "experiment": "closed-loop live RGB-D semantic selection",
        "source_scene_manifest": str(a.settled_manifest.resolve()),
        "coordinate_contract": settled.get("coordinate_contract", {}),
        "table": settled["table"],
        "world_from_source_zone": settled["world_from_source_zone"],
        "objects": objects,
    }
    manifest_path = case_root/"01_input/scene_closed_loop_manifest.json"
    manifest_path.write_text(json.dumps(scene_manifest, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")

    case = {
        "schema_version": 2,
        "case_id": a.case_id,
        "scene_id": "closed_loop",
        "view_id": "live_rgbd",
        "target_segmentation_id": target_id,
        "target_object_code": "perception_target_proxy",
        "source_candidate_index": idx,
        "official_score": float(pred["score"][idx]),
        "selection_policy": "perception target mask -> official DGN2 score order -> route full gate",
        "source_hand": "LEAP Hand",
        "target_hand": "Wuji Hand 2 Beta1 right",
        "pipeline_status": "official_leap_waypoints_ready_no_legacy_collision_claim",
        "physics_status": "not_tested",
        "live_capture_root": str(a.capture_root.resolve()),
        "perception_target_geometry": str(a.target_geometry.resolve()),
        "simulator_target_identity_used_for_planning": False,
        "legacy_mesh_collision_used": False,
    }
    (case_root/"case.json").write_text(json.dumps(case, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    print(json.dumps({"status":"PASS","case_root":str(case_root),"candidate_index":idx,
                      "score":float(pred["score"][idx]),"target_segmentation_id":target_id},
                     ensure_ascii=False))

if __name__ == "__main__":
    main()
