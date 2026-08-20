#!/usr/bin/env python3
"""Build a trajectory_visualizer bundle for the validated Route B right-arm path."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[3]
ISAACLAB_CONTROL_ROOT = PROJECT_ROOT / "08_dual_arm_scene_layout/isaaclab_control"
THIS_DIR = Path(__file__).resolve().parent
RIGHT_ARM_CORE_ROOT = THIS_DIR / "routeB_right_arm_only_core_v1"
for path in (ISAACLAB_CONTROL_ROOT, THIS_DIR, RIGHT_ARM_CORE_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from core.perception_collision.esdf_collision import query_spheres
from core.perception_collision.rgbd_mapper import RGBDFrame
from curobo_motion_planning_routeB import RouteBMotionPlannerAdapter
from curobo_motion_planning_routeB.routeB_adapter import (
    DEFAULT_ENABLE_GRAPH_ATTEMPT,
    DEFAULT_LAYOUT_JSON,
    DEFAULT_ROBOT_FILE,
)
from right_arm_only_core.contract import rebuild_robot_cfg_with_lock_joints
from trajectory_visualizer.bundle import VisualizationBundle, save_bundle, validate_bundle


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_yaml(path: Path) -> dict[str, Any]:
    import yaml

    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _robot_kinematics_dict(robot_yaml: dict[str, Any]) -> dict[str, Any]:
    if "robot_cfg" in robot_yaml:
        return robot_yaml["robot_cfg"]["kinematics"]
    return robot_yaml["kinematics"]


def _sphere_link_names_from_state(robot_yaml: dict[str, Any], state: Any) -> np.ndarray:
    kin = _robot_kinematics_dict(robot_yaml)
    collision_link_names = [str(x) for x in kin.get("collision_link_names", [])]
    idx_map = getattr(state.robot_collision_geometry, "link_sphere_idx_map", None)
    if idx_map is None:
        return np.asarray([f"sphere_{i}" for i in range(int(state.robot_spheres.shape[-2]))], dtype="U")
    if hasattr(idx_map, "detach"):
        idx_map = idx_map.detach().cpu().numpy()
    idx_map = np.asarray(idx_map, dtype=np.int64).reshape(-1)
    names: list[str] = []
    for idx in idx_map:
        if 0 <= int(idx) < len(collision_link_names):
            names.append(collision_link_names[int(idx)])
        else:
            names.append(f"link_index_{int(idx)}")
    return np.asarray(names, dtype="U")


def _load_scene_points_base(
    adapter: RouteBMotionPlannerAdapter,
    *,
    filtered_depth: Path,
    intrinsics: Path,
    T_world_camera: Path,
    rgb_path: Path | None,
    max_points: int,
) -> tuple[np.ndarray, np.ndarray | None]:
    frame = RGBDFrame.from_npy(filtered_depth, intrinsics, T_world_camera).validated()
    T_base_camera = adapter._base_from_world() @ frame.T_world_camera
    depth = np.asarray(frame.depth_m, dtype=np.float32)
    valid = np.isfinite(depth) & (depth > 0.0)
    ys, xs = np.nonzero(valid)
    z = depth[ys, xs].astype(np.float64)
    K = np.asarray(frame.intrinsics, dtype=np.float64)
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    x = (xs.astype(np.float64) - cx) * z / fx
    y = (ys.astype(np.float64) - cy) * z / fy
    pts_cam = np.stack([x, y, z, np.ones_like(z)], axis=1)
    pts_base = (np.asarray(T_base_camera, dtype=np.float64) @ pts_cam.T).T[:, :3]

    colors = None
    if rgb_path is not None and rgb_path.is_file():
        try:
            from PIL import Image

            rgb = np.asarray(Image.open(rgb_path).convert("RGB"))
            if rgb.shape[:2] == depth.shape:
                colors = rgb[ys, xs]
        except Exception:
            colors = None

    if pts_base.shape[0] > max_points:
        idx = np.linspace(0, pts_base.shape[0] - 1, max_points).astype(np.int64)
        pts_base = pts_base[idx]
        if colors is not None:
            colors = colors[idx]
    return pts_base.astype(np.float32), colors


def _make_locked_planner(
    *,
    adapter: RouteBMotionPlannerAdapter,
    scene,
    robot_source: dict[str, Any],
    lock_joints: dict[str, float],
):
    import torch
    from curobo.motion_planner import MotionPlanner, MotionPlannerCfg
    from curobo.types import DeviceCfg

    locked_robot_cfg = rebuild_robot_cfg_with_lock_joints(robot_source, lock_joints)
    device_cfg = DeviceCfg(device=torch.device(adapter.device), dtype=torch.float32)
    cfg = MotionPlannerCfg.create(
        robot=locked_robot_cfg,
        scene_model=scene,
        device_cfg=device_cfg,
        self_collision_check=adapter.self_collision_check,
        num_ik_seeds=adapter.num_ik_seeds,
        num_trajopt_seeds=adapter.num_trajopt_seeds,
        use_cuda_graph=adapter.use_cuda_graph,
        interpolation_dt=adapter.interpolation_dt_s,
    )
    collision_policy = adapter._apply_collision_policy_to_motion_cfg(cfg)
    planner = MotionPlanner(cfg)
    voxel_shape_contract = adapter._normalize_voxel_shape_contract(planner, scene)
    planner.warmup(enable_graph=adapter.use_cuda_graph, num_warmup_iterations=adapter.warmup_iterations)
    return planner, {"collision_policy": collision_policy, "voxel_shape_contract": voxel_shape_contract}


def _fk_spheres_and_ee(planner, q_right: np.ndarray, ee_frame: str):
    import torch
    from curobo.types import JointState

    q = torch.as_tensor(
        q_right,
        device=planner.device_cfg.device,
        dtype=planner.device_cfg.dtype,
    ).contiguous()
    state = JointState.from_position(q, joint_names=list(planner.joint_names))
    kin = planner.compute_kinematics(state)
    spheres = kin.robot_spheres.detach().cpu().numpy()
    while spheres.ndim > 3 and spheres.shape[0] == 1:
        spheres = spheres[0]
    if spheres.ndim == 4 and spheres.shape[1] == 1:
        spheres = spheres[:, 0]
    if spheres.ndim != 3 or spheres.shape[-1] != 4:
        raise RuntimeError(f"unexpected robot_spheres shape: {spheres.shape}")
    frames = [str(x) for x in kin.tool_frames]
    if ee_frame not in frames:
        raise RuntimeError(f"EE frame {ee_frame} not in planner tool_frames={frames}")
    ee_idx = frames.index(ee_frame)
    ee_pos = kin.tool_poses.position.detach().cpu().numpy()
    while ee_pos.ndim > 3 and ee_pos.shape[0] == 1:
        ee_pos = ee_pos[0]
    if ee_pos.ndim == 4 and ee_pos.shape[1] == 1:
        ee_pos = ee_pos[:, 0]
    if ee_pos.ndim != 3:
        raise RuntimeError(f"unexpected tool_poses.position shape: {ee_pos.shape}")
    return spheres[:, :, :3], spheres[0, :, 3], ee_pos[:, ee_idx, :], kin


def run(args: argparse.Namespace) -> int:
    capture_dir = Path(args.capture_dir).expanduser().resolve()
    output_dir = capture_dir / "curobo_test_result" if args.output_dir is None else Path(args.output_dir).expanduser().resolve()
    trajectory_path = output_dir / "trajectory_right_arm.npz" if args.trajectory is None else Path(args.trajectory).expanduser().resolve()
    report_path = output_dir / "report_right_arm.json" if args.report is None else Path(args.report).expanduser().resolve()
    bundle_path = output_dir / "visualization_right_arm_bundle.npz" if args.bundle is None else Path(args.bundle).expanduser().resolve()

    inputs = {
        "filtered_depth": capture_dir / "planning/filtered_depth.npy",
        "intrinsics": capture_dir / "intrinsics.npy",
        "T_world_camera": capture_dir / "T_world_camera.npy",
        "trajectory": trajectory_path,
        "report": report_path,
    }
    missing = [str(p) for p in inputs.values() if not p.is_file()]
    if missing:
        raise FileNotFoundError("missing visualization inputs: " + ", ".join(missing))

    cfg = {
        "routeB": {
            "device": args.device,
            "robot_file": str(args.robot_file),
            "layout_json": str(args.layout_json),
            "collision": {"environment_collision": True, "self_collision": False},
            "use_cuda_graph": args.use_cuda_graph,
            "num_ik_seeds": args.num_ik_seeds,
            "num_trajopt_seeds": args.num_trajopt_seeds,
            "max_attempts": args.max_attempts,
            "enable_graph_attempt": args.enable_graph_attempt,
            "warmup_iterations": args.warmup_iterations,
            "interpolation_dt_s": args.interpolation_dt_s,
        }
    }
    adapter = RouteBMotionPlannerAdapter(cfg)
    scene = adapter.build_pick_scene(inputs["filtered_depth"], inputs["intrinsics"], inputs["T_world_camera"])

    traj_npz = np.load(trajectory_path, allow_pickle=False)
    q_right = np.asarray(traj_npz["q_rad"], dtype=np.float32)
    time_s = np.asarray(traj_npz["time_s"], dtype=np.float64)
    joint_names = np.asarray(traj_npz["joint_names"]).astype("U")
    report = load_json(report_path)
    lock_joints = report["locked_joint_contract"]["lock_joints"]
    robot_source = load_yaml(Path(args.robot_file).expanduser().resolve())
    planner, planner_fixup = _make_locked_planner(
        adapter=adapter,
        scene=scene,
        robot_source=robot_source,
        lock_joints=lock_joints,
    )
    if int(getattr(planner, "action_dim")) != 7:
        raise RuntimeError(f"expected locked planner.action_dim=7, got {planner.action_dim}")
    if [str(x) for x in planner.joint_names] != [str(x) for x in joint_names.tolist()]:
        raise RuntimeError(f"planner joint_names {planner.joint_names} != trajectory joints {joint_names.tolist()}")

    sphere_centers, sphere_radii, ee_positions, kin_state = _fk_spheres_and_ee(
        planner, q_right, args.ee_frame
    )
    sphere_link_names = _sphere_link_names_from_state(robot_source, kin_state)
    if sphere_link_names.shape[0] != sphere_radii.shape[0]:
        raise RuntimeError(
            f"sphere link count {sphere_link_names.shape[0]} != sphere count {sphere_radii.shape[0]}"
        )

    movement = np.linalg.norm(sphere_centers - sphere_centers[0:1], axis=2).max(axis=0)
    sphere_active_mask = movement > 1.0e-7

    frame_min = []
    frame_worst = []
    for centers, radii in zip(sphere_centers, np.repeat(sphere_radii[None, :], sphere_centers.shape[0], axis=0)):
        batch = query_spheres(scene.voxel[0], centers, radii, margin_m=0.0)
        clearance = np.asarray(batch.distance_m, dtype=np.float64) - radii
        idx = int(np.argmin(clearance))
        frame_min.append(float(clearance[idx]))
        frame_worst.append(idx)
    frame_min_arr = np.asarray(frame_min, dtype=np.float32)
    frame_worst_arr = np.asarray(frame_worst, dtype=np.int32)

    scene_points, scene_colors = _load_scene_points_base(
        adapter,
        filtered_depth=inputs["filtered_depth"],
        intrinsics=inputs["intrinsics"],
        T_world_camera=inputs["T_world_camera"],
        rgb_path=capture_dir / "rgb.png",
        max_points=int(args.max_scene_points),
    )

    metadata = {
        "schema_version": 1,
        "route": "RouteB",
        "stage": "current_to_pregrasp_right_arm_only_visualization",
        "capture_dir": str(capture_dir),
        "trajectory_right_arm": str(trajectory_path),
        "source_report": str(report_path),
        "frame": "arm_base_link",
        "ee_frame": args.ee_frame,
        "planner_action_dim": int(planner.action_dim),
        "planner_joint_names": [str(x) for x in planner.joint_names],
        "planner_fixup": planner_fixup,
        "global_min_clearance_m": float(np.min(frame_min_arr)),
        "global_worst_frame": int(np.argmin(frame_min_arr)),
        "global_worst_sphere": int(frame_worst_arr[int(np.argmin(frame_min_arr))]),
        "global_worst_link": str(sphere_link_names[int(frame_worst_arr[int(np.argmin(frame_min_arr))])]),
    }
    bundle = VisualizationBundle(
        scene_points_base=scene_points,
        sphere_centers_base=sphere_centers,
        sphere_radii_m=sphere_radii,
        sphere_link_names=sphere_link_names,
        sphere_active_mask=sphere_active_mask,
        ee_positions_base=ee_positions,
        time_s=time_s,
        q_rad=q_right,
        joint_names=joint_names,
        frame_min_clearance_m=frame_min_arr,
        frame_worst_sphere_index=frame_worst_arr,
        scene_colors_rgb=scene_colors,
        metadata_json=json.dumps(metadata, ensure_ascii=False),
    )
    validate_bundle(bundle)
    saved = save_bundle(bundle_path, bundle)
    summary_path = output_dir / "visualization_right_arm_summary.json"
    summary = {
        **metadata,
        "bundle": str(saved),
        "scene_point_count": int(scene_points.shape[0]),
        "sphere_count": int(sphere_radii.shape[0]),
        "moving_sphere_count": int(np.count_nonzero(sphere_active_mask)),
        "static_sphere_count": int(sphere_active_mask.shape[0] - np.count_nonzero(sphere_active_mask)),
        "trajectory_frames": int(q_right.shape[0]),
        "frame0_clearance_m": float(frame_min_arr[0]),
        "frame_last_clearance_m": float(frame_min_arr[-1]),
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print("[Route B visualization bundle]")
    print(f"bundle={saved}")
    print(f"scene_points={scene_points.shape[0]}")
    print(f"spheres={sphere_radii.shape[0]} moving={int(np.count_nonzero(sphere_active_mask))} static={int(sphere_active_mask.shape[0] - np.count_nonzero(sphere_active_mask))}")
    print(f"ee_frame={args.ee_frame}")
    print(f"frames={q_right.shape[0]}")
    print(f"global_min_clearance_m={float(np.min(frame_min_arr)):.6f}")
    print(f"worst_frame={metadata['global_worst_frame']} worst_sphere={metadata['global_worst_sphere']} link={metadata['global_worst_link']}")
    print(f"summary={summary_path}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture-dir", required=True)
    parser.add_argument("--output-dir")
    parser.add_argument("--trajectory")
    parser.add_argument("--report")
    parser.add_argument("--bundle")
    parser.add_argument("--robot-file", default=str(DEFAULT_ROBOT_FILE))
    parser.add_argument("--layout-json", default=str(DEFAULT_LAYOUT_JSON))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--use-cuda-graph", action="store_true")
    parser.add_argument("--num-ik-seeds", type=int, default=32)
    parser.add_argument("--num-trajopt-seeds", type=int, default=4)
    parser.add_argument("--max-attempts", type=int, default=2)
    parser.add_argument("--enable-graph-attempt", type=int, default=DEFAULT_ENABLE_GRAPH_ATTEMPT)
    parser.add_argument("--warmup-iterations", type=int, default=1)
    parser.add_argument("--interpolation-dt-s", type=float, default=0.025)
    parser.add_argument("--max-scene-points", type=int, default=30000)
    parser.add_argument("--ee-frame", default="arm_r_link_tf")
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
