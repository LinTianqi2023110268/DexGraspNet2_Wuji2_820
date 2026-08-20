#!/usr/bin/env python3
from __future__ import annotations

"""Offline Route B attachment audit at the LIFT endpoint.

This is diagnostic-only.  It does not launch Isaac, does not change Route A,
and does not alter planner thresholds.  It visualizes and audits the carried
target proxy that is attached for LIFT_TO_TRANSFER.
"""

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np


THIS = Path(__file__).resolve()
PROJECT_ROOT = THIS.parents[5]
CONTROL_ROOT = PROJECT_ROOT / "08_dual_arm_scene_layout/isaaclab_control"
ROUTEB_ROOT = CONTROL_ROOT / "curobo_motion_planning_routeB"
RIGHT_CORE_ROOT = ROUTEB_ROOT / "routeB_right_arm_only_core_v1"
for path in (CONTROL_ROOT, ROUTEB_ROOT, RIGHT_CORE_ROOT, THIS.parent):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from core.perception_collision.esdf_collision import query_spheres  # noqa: E402
from core.perception_collision.rgbd_mapper import RGBDFrame  # noqa: E402
from curobo_motion_planning_routeB import RouteBMotionPlannerAdapter  # noqa: E402
from curobo_motion_planning_routeB.routeB_adapter import (  # noqa: E402
    DEFAULT_LAYOUT_JSON,
    DEFAULT_ROBOT_FILE,
)
from right_arm_only_core.contract import rebuild_robot_cfg_with_lock_joints  # noqa: E402

from attachment_proxy import build_target_proxy_from_capture  # noqa: E402
from full_motion_backend import (  # noqa: E402
    case_target_segmentation_id,
    hand_states,
    load_json,
    load_yaml,
    make_joint_state,
    matrix_to_pose_list,
    pose_to_matrix,
)
from robot_config import (  # noqa: E402
    ATTACHED_LINK,
    build_locked_joint_values,
    with_attachment_link,
)


def _kinematics_dict(robot_yaml: dict[str, Any]) -> dict[str, Any]:
    if "robot_cfg" in robot_yaml:
        return robot_yaml["robot_cfg"]["kinematics"]
    return robot_yaml["kinematics"]


def _sphere_link_names_from_state(robot_yaml: dict[str, Any], state: Any) -> np.ndarray:
    kin = _kinematics_dict(robot_yaml)
    collision_link_names = [str(x) for x in kin.get("collision_link_names", [])]
    idx_map = getattr(state.robot_collision_geometry, "link_sphere_idx_map", None)
    if idx_map is None:
        return np.asarray(
            [f"sphere_{i}" for i in range(int(state.robot_spheres.shape[-2]))],
            dtype="U",
        )
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
    max_points: int,
) -> np.ndarray:
    frame = RGBDFrame.from_npy(filtered_depth, intrinsics, T_world_camera).validated()
    T_base_camera = adapter._base_from_world() @ frame.T_world_camera
    depth = np.asarray(frame.depth_m, dtype=np.float32)
    valid = np.isfinite(depth) & (depth > 0.0)
    ys, xs = np.nonzero(valid)
    z = depth[ys, xs].astype(np.float64)
    K = np.asarray(frame.intrinsics, dtype=np.float64)
    x = (xs.astype(np.float64) - float(K[0, 2])) * z / float(K[0, 0])
    y = (ys.astype(np.float64) - float(K[1, 2])) * z / float(K[1, 1])
    pts_cam = np.stack([x, y, z, np.ones_like(z)], axis=1)
    pts_base = (np.asarray(T_base_camera, dtype=np.float64) @ pts_cam.T).T[:, :3]
    if pts_base.shape[0] > max_points:
        idx = np.linspace(0, pts_base.shape[0] - 1, max_points).astype(np.int64)
        pts_base = pts_base[idx]
    return pts_base.astype(np.float32)


def _make_locked_attachment_planner(
    *,
    adapter: RouteBMotionPlannerAdapter,
    scene,
    robot_source: dict[str, Any],
    full_joint_names: list[str],
    measured_by_name: dict[str, float],
    hand_joint_names: list[str],
    hand_q20: np.ndarray,
    sphere_slots: int,
):
    import torch
    from curobo.motion_planner import MotionPlanner, MotionPlannerCfg
    from curobo.types import DeviceCfg

    lock = build_locked_joint_values(
        full_joint_names=full_joint_names,
        measured_by_name=measured_by_name,
        hand_joint_names=hand_joint_names,
        hand_q20=hand_q20,
    )
    source = with_attachment_link(
        robot_source,
        sphere_slots=int(sphere_slots),
    )
    locked_robot_cfg = rebuild_robot_cfg_with_lock_joints(source, lock)
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
    planner.warmup(
        enable_graph=adapter.use_cuda_graph,
        num_warmup_iterations=adapter.warmup_iterations,
    )
    return planner, {
        "collision_policy": collision_policy,
        "voxel_shape_contract": voxel_shape_contract,
    }


def _as_np(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    return np.asarray(value)


def _plot_png(
    *,
    path: Path,
    scene_points: np.ndarray,
    proxy_center: np.ndarray,
    proxy_dims: np.ndarray,
    attached_centers: np.ndarray,
    attached_radii: np.ndarray,
    worst_index: int | None,
    lift_xyz: np.ndarray,
    transfer_xyz: np.ndarray,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")
    if len(scene_points):
        ax.scatter(
            scene_points[:, 0],
            scene_points[:, 1],
            scene_points[:, 2],
            s=1,
            c="0.65",
            alpha=0.25,
            label="filtered_depth_no_target",
        )
    sizes = np.maximum((attached_radii * 1800.0) ** 2, 12.0)
    colors = np.full((len(attached_centers),), "tab:orange", dtype=object)
    if worst_index is not None and 0 <= worst_index < len(colors):
        colors[worst_index] = "red"
    ax.scatter(
        attached_centers[:, 0],
        attached_centers[:, 1],
        attached_centers[:, 2],
        s=sizes,
        c=colors,
        alpha=0.55,
        label="attached spheres",
    )
    ax.scatter(*proxy_center.tolist(), s=90, c="blue", marker="x", label="proxy center")
    ax.scatter(*lift_xyz.tolist(), s=70, c="green", marker="o", label="LIFT flange")
    ax.scatter(*transfer_xyz.tolist(), s=70, c="purple", marker="^", label="TRANSFER flange")
    ax.plot(
        [lift_xyz[0], transfer_xyz[0]],
        [lift_xyz[1], transfer_xyz[1]],
        [lift_xyz[2], transfer_xyz[2]],
        c="purple",
        linewidth=2,
        label="LIFT->TRANSFER chord",
    )

    # AABB wireframe for the target proxy.
    half = 0.5 * proxy_dims
    lo = proxy_center - half
    hi = proxy_center + half
    corners = np.asarray(
        [
            [x, y, z]
            for x in (lo[0], hi[0])
            for y in (lo[1], hi[1])
            for z in (lo[2], hi[2])
        ],
        dtype=np.float64,
    )
    edges = [
        (0, 1),
        (0, 2),
        (0, 4),
        (3, 1),
        (3, 2),
        (3, 7),
        (5, 1),
        (5, 4),
        (5, 7),
        (6, 2),
        (6, 4),
        (6, 7),
    ]
    for a, b in edges:
        ax.plot(*zip(corners[a], corners[b]), c="blue", alpha=0.7)

    all_pts = np.vstack(
        [
            scene_points[: min(len(scene_points), 5000)] if len(scene_points) else np.zeros((0, 3)),
            attached_centers,
            proxy_center.reshape(1, 3),
            lift_xyz.reshape(1, 3),
            transfer_xyz.reshape(1, 3),
        ]
    )
    mid = all_pts.mean(axis=0)
    span = max(float(np.ptp(all_pts[:, 0])), float(np.ptp(all_pts[:, 1])), float(np.ptp(all_pts[:, 2])), 0.2)
    for axis, c in zip((ax.set_xlim, ax.set_ylim, ax.set_zlim), mid):
        axis(c - 0.55 * span, c + 0.55 * span)
    ax.set_xlabel("base x (m)")
    ax.set_ylabel("base y (m)")
    ax.set_zlabel("base z (m)")
    ax.set_title("Route B attachment audit at LIFT")
    ax.legend(loc="best")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--capture-dir", type=Path, required=True)
    parser.add_argument("--query", required=True)
    parser.add_argument("--case-root", type=Path, required=True)
    parser.add_argument("--backhalf-pool", type=Path, required=True)
    parser.add_argument("--chain-index", type=int, default=0)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--attachment-padding-m", type=float, default=0.005)
    parser.add_argument("--attachment-min-dim-m", type=float, default=0.02)
    parser.add_argument("--attachment-sphere-slots", type=int, default=48)
    parser.add_argument("--attachment-sphere-count", type=int, default=32)
    parser.add_argument("--max-scene-points", type=int, default=30000)
    args = parser.parse_args()

    root = args.project_root.resolve()
    capture = args.capture_dir.resolve()
    case_root = args.case_root.resolve()
    output = (
        capture / "curobo_test_result/attachment_audit"
        if args.output_dir is None
        else args.output_dir.resolve()
    )
    output.mkdir(parents=True, exist_ok=True)

    control = root / "08_dual_arm_scene_layout/isaaclab_control"
    robot_state_path = capture / "robot_state.json"
    filtered_no_target = capture / "planning/filtered_depth_no_target.npy"
    if not filtered_no_target.is_file():
        filtered_no_target = capture / "planning/filtered_depth.npy"
    intrinsics = capture / "intrinsics.npy"
    T_world_camera = capture / "T_world_camera.npy"
    for path in (robot_state_path, filtered_no_target, intrinsics, T_world_camera):
        if not path.is_file():
            raise FileNotFoundError(path)

    adapter_cfg = {
        "routeB": {
            "device": args.device,
            "robot_file": str(DEFAULT_ROBOT_FILE),
            "layout_json": str(DEFAULT_LAYOUT_JSON),
            "collision": {"environment_collision": True, "self_collision": False},
            "use_cuda_graph": False,
            "num_ik_seeds": 32,
            "num_trajopt_seeds": 4,
            "max_attempts": 2,
            "enable_graph_attempt": 1000000,
            "warmup_iterations": 1,
            "interpolation_dt_s": 0.025,
        }
    }
    adapter = RouteBMotionPlannerAdapter(adapter_cfg)
    scene = adapter.build_pick_scene(filtered_no_target, intrinsics, T_world_camera)
    adapter.create_planner(scene)
    full_joint_names = list(adapter.joint_names)
    robot_source = load_yaml(DEFAULT_ROBOT_FILE)
    measured = {
        str(k): float(v)
        for k, v in load_json(robot_state_path)["joint_positions_by_name"].items()
    }
    hand_names, hand = hand_states(case_root)

    with np.load(args.backhalf_pool.resolve(), allow_pickle=False) as z:
        chain_i = int(args.chain_index)
        q_lift = np.asarray(z["q_lift_rad"][chain_i], dtype=np.float64)
        q_transfer = np.asarray(z["q_transfer_rad"][chain_i], dtype=np.float64)
        lift_pose_world = np.asarray(z["lift_pose_world"][chain_i], dtype=np.float64)
        transfer_pose_world = np.asarray(z["transfer_pose_world"][chain_i], dtype=np.float64)

    target_segmentation_id = case_target_segmentation_id(case_root)
    proxy = build_target_proxy_from_capture(
        project_root=root,
        capture_dir=capture,
        target_segmentation_id=target_segmentation_id,
        padding_m=args.attachment_padding_m,
        minimum_dim_m=args.attachment_min_dim_m,
    )

    planner, fixup = _make_locked_attachment_planner(
        adapter=adapter,
        scene=scene,
        robot_source=robot_source,
        full_joint_names=full_joint_names,
        measured_by_name=measured,
        hand_joint_names=hand_names,
        hand_q20=hand["squeeze"],
        sphere_slots=args.attachment_sphere_slots,
    )

    try:
        from curobo.geom.types import Cuboid
    except ModuleNotFoundError:
        from curobo._src.geom.types import Cuboid
    from curobo.types import Pose
    from curobo._src.geom.sphere_fit.types import SphereFitType

    local_proxy = Cuboid(
        name="routeB_target_proxy_local",
        pose=[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        dims=proxy.dims_base_m.tolist(),
    )
    fitted = planner.attachment_manager.fit_spheres(
        [local_proxy],
        num_spheres=int(args.attachment_sphere_count),
        sphere_fit_type=SphereFitType.VOXEL,
    )

    cover_q = np.load(args.backhalf_pool.resolve(), allow_pickle=False)["q_lift_rad"][chain_i]
    # Use the same object-to-flange contract as full_motion_backend: infer
    # flange_from_proxy from exact COVER, then place the proxy at LIFT.
    # q_cover is stored in the front-half report.
    front_report_path = (
        args.backhalf_pool.resolve().parents[1]
        / "routeB_front_half/planning/routeB_front_half_report.json"
    )
    if not front_report_path.is_file():
        # Current standard layout: cycle_root/routeB_full/backhalf_chain_pool.npz
        front_report_path = (
            args.backhalf_pool.resolve().parents[1]
            / "../routeB_front_half/planning/routeB_front_half_report.json"
        ).resolve()
    front_report = load_json(front_report_path)
    q_cover = np.asarray(front_report["selected"]["q_cover_rad"], dtype=np.float64)
    cover_flange = pose_to_matrix(
        planner.compute_kinematics(make_joint_state(q_cover, planner)).tool_poses.get_link_pose("arm_r_link_tf")
    )
    proxy_initial = np.eye(4, dtype=np.float64)
    proxy_initial[:3, 3] = proxy.center_base_m
    flange_from_proxy = np.linalg.inv(cover_flange) @ proxy_initial
    lift_flange = pose_to_matrix(
        planner.compute_kinematics(make_joint_state(q_lift, planner)).tool_poses.get_link_pose("arm_r_link_tf")
    )
    proxy_at_lift = lift_flange @ flange_from_proxy

    planner.attachment_manager.update(
        fitted,
        make_joint_state(q_lift, planner),
        link_name=ATTACHED_LINK,
        world_objects_pose_offset=Pose.from_list(
            matrix_to_pose_list(proxy_at_lift),
            device_cfg=planner.device_cfg,
        ),
    )
    kin = planner.compute_kinematics(make_joint_state(q_lift, planner))
    spheres = _as_np(kin.robot_spheres)
    while spheres.ndim > 3 and spheres.shape[0] == 1:
        spheres = spheres[0]
    if spheres.ndim == 3 and spheres.shape[0] == 1:
        spheres = spheres[0]
    if spheres.ndim != 2 or spheres.shape[-1] != 4:
        raise RuntimeError(f"unexpected attached robot_spheres shape: {spheres.shape}")
    robot_source_with_attachment = with_attachment_link(
        robot_source,
        sphere_slots=int(args.attachment_sphere_slots),
    )
    names = _sphere_link_names_from_state(robot_source_with_attachment, kin)
    attached_mask = names == ATTACHED_LINK
    attached = spheres[attached_mask]
    attached_indices = np.nonzero(attached_mask)[0]
    if attached.size == 0:
        raise RuntimeError("no attached spheres found in robot_spheres")

    batch = query_spheres(scene.voxel[0], attached[:, :3], attached[:, 3], margin_m=0.0)
    signed_distance = np.asarray(batch.distance_m, dtype=np.float64)
    clearance = signed_distance - attached[:, 3]
    collision = np.asarray(batch.collision, dtype=bool)
    worst_local = int(np.argmin(clearance))

    scene_points = _load_scene_points_base(
        adapter,
        filtered_depth=filtered_no_target,
        intrinsics=intrinsics,
        T_world_camera=T_world_camera,
        max_points=int(args.max_scene_points),
    )
    png_path = output / f"attachment_lift_chain{chain_i:02d}.png"
    npz_path = output / f"attachment_lift_chain{chain_i:02d}.npz"
    json_path = output / f"attachment_audit_chain{chain_i:02d}.json"

    np.savez_compressed(
        npz_path,
        scene_points_base=scene_points,
        attached_sphere_centers_base=attached[:, :3],
        attached_sphere_radii_m=attached[:, 3],
        attached_sphere_indices=attached_indices,
        attached_sphere_clearance_m=clearance,
        lift_flange_base=lift_flange,
        transfer_flange_world=transfer_pose_world,
        lift_flange_world=lift_pose_world,
        proxy_at_lift_base=proxy_at_lift,
        proxy_dims_base_m=proxy.dims_base_m,
    )
    _plot_png(
        path=png_path,
        scene_points=scene_points,
        proxy_center=proxy_at_lift[:3, 3],
        proxy_dims=proxy.dims_base_m,
        attached_centers=attached[:, :3],
        attached_radii=attached[:, 3],
        worst_index=worst_local,
        lift_xyz=lift_flange[:3, 3],
        transfer_xyz=(adapter._base_from_world() @ np.r_[transfer_pose_world[:3, 3], 1.0])[:3],
    )

    rows = []
    for i, idx in enumerate(attached_indices):
        rows.append(
            {
                "local_attachment_index": int(i),
                "global_sphere_index": int(idx),
                "center_base_m": attached[i, :3].tolist(),
                "radius_m": float(attached[i, 3]),
                "signed_distance_m": float(signed_distance[i]),
                "clearance_m": float(clearance[i]),
                "collision": bool(collision[i]),
            }
        )
    report = {
        "schema_version": 1,
        "stage": "ROUTEB_ATTACHMENT_AUDIT_LIFT",
        "query": args.query,
        "case_root": str(case_root),
        "capture_dir": str(capture),
        "chain_index": int(chain_i),
        "collision_policy": {
            "environment_collision": True,
            "self_collision": False,
            "attachment": True,
        },
        "planner_fixup": fixup,
        "target_proxy": proxy.to_jsonable(),
        "proxy_at_lift_base": proxy_at_lift.tolist(),
        "flange_from_proxy": flange_from_proxy.tolist(),
        "lift_flange_base": lift_flange.tolist(),
        "lift_flange_world_from_pool": lift_pose_world.tolist(),
        "transfer_flange_world_from_pool": transfer_pose_world.tolist(),
        "q_lift_rad": q_lift.tolist(),
        "q_transfer_rad": q_transfer.tolist(),
        "attachment": {
            "attached_link": ATTACHED_LINK,
            "sphere_count": int(attached.shape[0]),
            "fitted_shape": list(_as_np(fitted).shape),
            "collision_count": int(np.count_nonzero(collision)),
            "min_clearance_m": float(np.min(clearance)),
            "worst_local_attachment_index": int(worst_local),
            "worst_global_sphere_index": int(attached_indices[worst_local]),
            "worst_clearance_m": float(clearance[worst_local]),
            "worst_signed_distance_m": float(signed_distance[worst_local]),
            "worst_radius_m": float(attached[worst_local, 3]),
            "spheres": rows,
        },
        "outputs": {
            "json": str(json_path),
            "npz": str(npz_path),
            "png": str(png_path),
        },
    }
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("[Route B attachment audit]")
    print(f"json={json_path}")
    print(f"npz={npz_path}")
    print(f"png={png_path}")
    print(f"proxy_dims_m={np.round(proxy.dims_base_m, 5).tolist()}")
    print(f"attached_spheres={attached.shape[0]}")
    print(f"collision_count={int(np.count_nonzero(collision))}")
    print(f"min_clearance_m={float(np.min(clearance)):.6f}")
    print(
        "worst="
        f"local:{worst_local} global:{int(attached_indices[worst_local])} "
        f"center={np.round(attached[worst_local, :3], 6).tolist()} "
        f"radius={float(attached[worst_local, 3]):.6f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
