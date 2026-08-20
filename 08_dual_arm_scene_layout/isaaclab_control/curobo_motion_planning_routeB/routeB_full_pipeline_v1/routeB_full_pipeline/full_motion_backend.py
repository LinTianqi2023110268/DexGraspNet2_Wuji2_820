#!/usr/bin/env python3
from __future__ import annotations

"""Complete Route B motion-plan backend.

This backend receives a PASS front-half selection and a pool of post-retarget
LIFT/TRANSFER/PLACE/RETREAT endpoint IK chains.  It creates true 7DOF cuRobo
MotionPlanner instances and produces all remaining dense arm trajectories.

Intentional-contact policy:
- PREGRASP->COVER uses RobotSegmenter depth with the target mask removed.
- Carry planning uses the same non-target ESDF plus an attached target proxy.
- Post-release planning uses the non-target ESDF plus the predicted placed proxy.
- self collision stays OFF; environment collision stays ON.
"""

import argparse
import copy
import json
import math
from pathlib import Path
import sys
import time
from typing import Any
from collections import Counter

import numpy as np

try:
    from .attachment_proxy import (
        TargetProxy,
        build_target_proxy_from_capture,
        remove_target_mask_from_filtered_depth,
    )
    from .metrics import raw_constraint_summary
    from .robot_config import (
        ATTACHED_LINK,
        RIGHT_ARM_JOINTS,
        build_locked_joint_values,
        with_attachment_link,
    )
except ImportError:
    from attachment_proxy import (  # type: ignore
        TargetProxy,
        build_target_proxy_from_capture,
        remove_target_mask_from_filtered_depth,
    )
    from metrics import raw_constraint_summary  # type: ignore
    from robot_config import (  # type: ignore
        ATTACHED_LINK,
        RIGHT_ARM_JOINTS,
        build_locked_joint_values,
        with_attachment_link,
    )


def load_json(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_yaml(path: Path) -> dict:
    import yaml

    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def safe_slug(text: str) -> str:
    import re

    out = re.sub(r"[^0-9A-Za-z._-]+", "_", str(text).strip()).strip("._")
    return out[:64] or "target"


def tensor_success(value: Any) -> bool:
    if value is None:
        return False
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    return bool(np.asarray(value, dtype=bool).any())


def summarize_trials(trials: list[dict[str, Any]]) -> dict[str, Any]:
    failed = [row for row in trials if not bool(row.get("success"))]
    counts = Counter(str(row.get("failed_stage", "UNKNOWN")) for row in failed)
    return {
        "trial_count": int(len(trials)),
        "success_count": int(sum(1 for row in trials if bool(row.get("success")))),
        "failure_count": int(len(failed)),
        "failed_stage_counts": dict(sorted(counts.items())),
        "dominant_failed_stage": counts.most_common(1)[0][0] if counts else None,
    }


def hand_states(case_root: Path) -> tuple[list[str], dict[str, np.ndarray]]:
    path = Path(case_root) / "06_isaacsim/final_waypoints.npz"
    with np.load(path, allow_pickle=False) as z:
        names = [str(x) for x in z["finger_joint_names"].tolist()]
        stages = [str(x) for x in z["waypoint_names"].tolist()]
        q = np.asarray(z["waypoint_joint_positions"][0], dtype=np.float64)
    idx = {name: i for i, name in enumerate(stages)}
    return names, {name: q[idx[name]].copy() for name in stages}


def case_target_segmentation_id(case_root: Path) -> int:
    data = load_json(Path(case_root) / "case.json")
    return int(data["target_segmentation_id"])


def quaternion_to_matrix(q) -> np.ndarray:
    w, x, y, z = np.asarray(q, dtype=np.float64).reshape(4)
    n = max(float(np.linalg.norm([w, x, y, z])), 1e-12)
    w, x, y, z = np.asarray([w, x, y, z]) / n
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def pose_to_matrix(pose) -> np.ndarray:
    pos = pose.position.detach().cpu().numpy().reshape(-1, 3)[0]
    quat = pose.quaternion.detach().cpu().numpy().reshape(-1, 4)[0]
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = quaternion_to_matrix(quat)
    T[:3, 3] = pos
    return T


def matrix_to_pose_list(T: np.ndarray) -> list[float]:
    T = np.asarray(T, dtype=np.float64)
    R = T[:3, :3]
    tr = float(np.trace(R))
    if tr > 0:
        s = math.sqrt(tr + 1.0) * 2.0
        q = np.asarray(
            [
                0.25 * s,
                (R[2, 1] - R[1, 2]) / s,
                (R[0, 2] - R[2, 0]) / s,
                (R[1, 0] - R[0, 1]) / s,
            ]
        )
    else:
        i = int(np.argmax(np.diag(R)))
        if i == 0:
            s = math.sqrt(max(1 + R[0, 0] - R[1, 1] - R[2, 2], 1e-16)) * 2
            q = np.asarray(
                [
                    (R[2, 1] - R[1, 2]) / s,
                    0.25 * s,
                    (R[0, 1] + R[1, 0]) / s,
                    (R[0, 2] + R[2, 0]) / s,
                ]
            )
        elif i == 1:
            s = math.sqrt(max(1 + R[1, 1] - R[0, 0] - R[2, 2], 1e-16)) * 2
            q = np.asarray(
                [
                    (R[0, 2] - R[2, 0]) / s,
                    (R[0, 1] + R[1, 0]) / s,
                    0.25 * s,
                    (R[1, 2] + R[2, 1]) / s,
                ]
            )
        else:
            s = math.sqrt(max(1 + R[2, 2] - R[0, 0] - R[1, 1], 1e-16)) * 2
            q = np.asarray(
                [
                    (R[1, 0] - R[0, 1]) / s,
                    (R[0, 2] + R[2, 0]) / s,
                    (R[1, 2] + R[2, 1]) / s,
                    0.25 * s,
                ]
            )
    q /= max(float(np.linalg.norm(q)), 1e-12)
    if q[0] < 0:
        q = -q
    return T[:3, 3].tolist() + q.tolist()


def make_joint_state(q: np.ndarray, planner: Any):
    import torch
    from curobo.types import JointState

    t = torch.as_tensor(
        np.asarray(q, dtype=np.float32),
        device=planner.device_cfg.device,
        dtype=planner.device_cfg.dtype,
    ).view(1, -1)
    return JointState.from_position(t, joint_names=planner.joint_names)


def sanitize_active_q(adapter, planner, q: np.ndarray) -> np.ndarray:
    lower, upper = adapter._motion_planner_position_bounds_for(planner)
    lower = np.asarray(lower, dtype=np.float64).reshape(-1)[:7]
    upper = np.asarray(upper, dtype=np.float64).reshape(-1)[:7]
    out = np.asarray(q, dtype=np.float64).reshape(7).copy()
    tol = 1e-5
    margin = 1e-6
    for i in range(7):
        if out[i] < lower[i]:
            if lower[i] - out[i] > tol:
                raise RuntimeError(
                    f"active q[{i}] below lower limit by {lower[i]-out[i]}"
                )
            out[i] = lower[i] + margin
        elif out[i] > upper[i]:
            if out[i] - upper[i] > tol:
                raise RuntimeError(
                    f"active q[{i}] above upper limit by {out[i]-upper[i]}"
                )
            out[i] = upper[i] - margin
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--capture-dir", type=Path, required=True)
    parser.add_argument("--query", required=True)
    parser.add_argument("--case-root", type=Path, required=True)
    parser.add_argument("--front-half-dir", type=Path, required=True)
    parser.add_argument("--backhalf-pool", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--target-mask-path", type=Path, default=None)
    parser.add_argument("--target-removal-mask-path", type=Path, default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--attachment-padding-m", type=float, default=0.005)
    parser.add_argument("--attachment-min-dim-m", type=float, default=0.02)
    parser.add_argument("--attachment-sphere-slots", type=int, default=48)
    parser.add_argument("--attachment-sphere-count", type=int, default=32)
    parser.add_argument("--max-chain-trials", type=int, default=16)
    parser.add_argument("--max-attempts", type=int, default=2)
    parser.add_argument("--enable-graph-attempt", type=int, default=1000000)
    parser.add_argument("--num-ik-seeds", type=int, default=32)
    parser.add_argument("--num-trajopt-seeds", type=int, default=4)
    parser.add_argument("--interpolation-dt-s", type=float, default=0.025)
    parser.add_argument("--warmup-iterations", type=int, default=1)
    parser.add_argument(
        "--diagnostic-disable-transfer-attachment",
        action="store_true",
        help="Backward-compatible alias: plan LIFT_TO_TRANSFER without the carried target proxy.",
    )
    parser.add_argument(
        "--disable-transfer-attachment",
        action="store_true",
        help="Plan LIFT_TO_TRANSFER without the carried target proxy while keeping environment collision ON.",
    )
    args = parser.parse_args()
    transfer_attachment_enabled = not (
        bool(args.disable_transfer_attachment)
        or bool(args.diagnostic_disable_transfer_attachment)
    )

    started = time.perf_counter()
    root = args.project_root.resolve()
    capture = args.capture_dir.resolve()
    case_root = args.case_root.resolve()
    front_dir = args.front_half_dir.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)

    control = root / "08_dual_arm_scene_layout/isaaclab_control"
    routeb = control / "curobo_motion_planning_routeB"
    right_core = routeb / "routeB_right_arm_only_core_v1"
    for path in (control, routeb, right_core):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))

    from curobo_motion_planning_routeB import RouteBMotionPlannerAdapter
    from curobo_motion_planning_routeB.routeB_adapter import (
        DEFAULT_LAYOUT_JSON,
        DEFAULT_ROBOT_FILE,
    )
    from right_arm_only_core import rebuild_robot_cfg_with_lock_joints
    from right_arm_only_core.trajectory import (
        extract_dense_right_arm_trajectory,
        save_right_arm_npz,
        validate_dense_trajectory,
    )
    from core.perception_collision.esdf_collision import query_spheres
    from curobo.motion_planner import MotionPlanner, MotionPlannerCfg
    from curobo.types import DeviceCfg, Pose
    try:
        from curobo.geom.types import Cuboid
    except ModuleNotFoundError:
        from curobo._src.geom.types import Cuboid
    from curobo._src.geom.sphere_fit.types import SphereFitType
    import torch

    front_report = load_json(front_dir / "routeB_front_half_report.json")
    if not bool(front_report.get("success")):
        raise RuntimeError("front-half report is not PASS")
    selected = front_report["selected"]
    q_pre = np.asarray(selected["q_pregrasp_rad"], dtype=np.float64)
    q_cover = np.asarray(selected["q_cover_rad"], dtype=np.float64)
    front_traj = front_dir / "trajectory_right_arm.npz"
    if not front_traj.is_file():
        raise FileNotFoundError(front_traj)

    with np.load(args.backhalf_pool.resolve(), allow_pickle=False) as z:
        chains = {key: np.asarray(z[key]) for key in z.files}
    chain_count = int(len(chains["score"]))
    if chain_count <= 0:
        raise RuntimeError("back-half endpoint chain pool is empty")

    target_segmentation_id = case_target_segmentation_id(case_root)
    explicit_target_mask = (
        args.target_mask_path.expanduser().resolve()
        if args.target_mask_path is not None
        else None
    )
    explicit_target_removal_mask = (
        args.target_removal_mask_path.expanduser().resolve()
        if args.target_removal_mask_path is not None
        else explicit_target_mask
    )
    proxy = build_target_proxy_from_capture(
        project_root=root,
        capture_dir=capture,
        target_segmentation_id=target_segmentation_id,
        target_mask_path=explicit_target_mask,
        padding_m=args.attachment_padding_m,
        minimum_dim_m=args.attachment_min_dim_m,
    )

    slug = safe_slug(args.query)
    mask_path = explicit_target_mask or (capture / "grounded_sam" / slug / "mask.npy")
    removal_mask_path = explicit_target_removal_mask or mask_path
    filtered_depth = capture / "planning/filtered_depth.npy"
    intrinsics = capture / "intrinsics.npy"
    T_world_camera = capture / "T_world_camera.npy"
    robot_state_path = capture / "robot_state.json"
    for path in (
        mask_path,
        filtered_depth,
        intrinsics,
        T_world_camera,
        robot_state_path,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    no_target_depth = remove_target_mask_from_filtered_depth(
        filtered_depth_path=filtered_depth,
        target_mask_path=removal_mask_path,
        output_path=capture / "planning/filtered_depth_no_target.npy",
    )

    adapter_cfg = {
        "routeB": {
            "device": args.device,
            "robot_file": str(DEFAULT_ROBOT_FILE),
            "layout_json": str(DEFAULT_LAYOUT_JSON),
            "collision": {
                "environment_collision": True,
                "self_collision": False,
            },
            "use_cuda_graph": False,
            "num_ik_seeds": int(args.num_ik_seeds),
            "num_trajopt_seeds": int(args.num_trajopt_seeds),
            "max_attempts": int(args.max_attempts),
            "enable_graph_attempt": int(args.enable_graph_attempt),
            "warmup_iterations": int(args.warmup_iterations),
            "interpolation_dt_s": float(args.interpolation_dt_s),
        }
    }
    adapter = RouteBMotionPlannerAdapter(adapter_cfg)
    scene = adapter.build_pick_scene(
        no_target_depth, intrinsics, T_world_camera
    )
    adapter.create_planner(scene)
    full_joint_names = list(adapter.joint_names)
    robot_source = load_yaml(adapter.robot_file)
    measured = {
        str(k): float(v)
        for k, v in load_json(robot_state_path)[
            "joint_positions_by_name"
        ].items()
    }
    hand_names, hand = hand_states(case_root)

    def build_planner(
        hand_q20: np.ndarray,
        *,
        attachment: bool = False,
        scene_override=None,
    ):
        lock = build_locked_joint_values(
            full_joint_names=full_joint_names,
            measured_by_name=measured,
            hand_joint_names=hand_names,
            hand_q20=hand_q20,
        )
        source = (
            with_attachment_link(
                robot_source,
                sphere_slots=int(args.attachment_sphere_slots),
            )
            if attachment
            else copy.deepcopy(robot_source)
        )
        robot_cfg = rebuild_robot_cfg_with_lock_joints(source, lock)
        scene_model = scene if scene_override is None else scene_override
        device_cfg = DeviceCfg(
            device=torch.device(adapter.device), dtype=torch.float32
        )
        motion_cfg = MotionPlannerCfg.create(
            robot=robot_cfg,
            scene_model=scene_model,
            device_cfg=device_cfg,
            self_collision_check=False,
            num_ik_seeds=int(args.num_ik_seeds),
            num_trajopt_seeds=int(args.num_trajopt_seeds),
            use_cuda_graph=False,
            interpolation_dt=float(args.interpolation_dt_s),
        )
        adapter._apply_collision_policy_to_motion_cfg(motion_cfg)
        planner = MotionPlanner(motion_cfg)
        adapter._normalize_voxel_shape_contract(planner, scene_model)
        planner.warmup(
            enable_graph=False,
            num_warmup_iterations=int(args.warmup_iterations),
        )
        if int(planner.action_dim) != 7:
            raise RuntimeError(
                f"locked planner action_dim={planner.action_dim}, expected 7"
            )
        if tuple(str(x) for x in planner.joint_names) != RIGHT_ARM_JOINTS:
            raise RuntimeError(
                f"active joints mismatch: {planner.joint_names}"
            )
        return planner

    def external_voxel_clearance(planner, scene_model, q_path):
        import torch
        from curobo.types import JointState

        q = torch.as_tensor(
            np.asarray(q_path, dtype=np.float32),
            device=planner.device_cfg.device,
            dtype=planner.device_cfg.dtype,
        ).unsqueeze(0).contiguous()
        kin = planner.compute_kinematics(
            JointState.from_position(q, joint_names=planner.joint_names)
        )
        spheres = (
            kin.robot_spheres.detach()
            .cpu()
            .numpy()
            .reshape(len(q_path), -1, 4)
        )
        minimum = float("inf")
        collisions = 0
        worst = None
        for t, sph in enumerate(spheres):
            batch = query_spheres(
                scene_model.voxel[0], sph[:, :3], sph[:, 3], margin_m=0.0
            )
            clearance = np.asarray(batch.distance_m) - sph[:, 3]
            i = int(np.argmin(clearance))
            if float(clearance[i]) < minimum:
                minimum = float(clearance[i])
                worst = {
                    "timestep": int(t),
                    "sphere_index": int(i),
                    "clearance_m": minimum,
                }
            if bool(np.any(batch.collision)):
                collisions += 1
        return {
            "voxel_collision": bool(collisions > 0),
            "voxel_collision_samples": int(collisions),
            "voxel_min_clearance_m": float(minimum),
            "voxel_worst_sample": worst,
        }

    class Contract:
        pass

    def plan_segment(planner, scene_model, q0, q1, stage, artifact_path, *, verbose: bool = False):
        if verbose:
            print(f"[Route B][PLAN] {stage} ...", flush=True)
        q0s = sanitize_active_q(adapter, planner, q0)
        q1s = sanitize_active_q(adapter, planner, q1)
        raw = planner.plan_cspace(
            goal_state=make_joint_state(q1s, planner),
            current_state=make_joint_state(q0s, planner),
            max_attempts=int(args.max_attempts),
            enable_graph_attempt=int(args.enable_graph_attempt),
        )
        if raw is None or not tensor_success(getattr(raw, "success", None)):
            if verbose:
                print(f"[Route B][PLAN] FAIL {stage}: MotionPlanner success=false", flush=True)
            return None, {"stage": stage, "success": False}
        if int(raw.solution.shape[-1]) != 7:
            raise RuntimeError(f"{stage}: result is not true 7DOF")
        contract = Contract()
        contract.q_current_active = q0s
        contract.q_goal_active = q1s
        contract.locked_joint_names = tuple()
        contract.lock_joints = {}
        traj = extract_dense_right_arm_trajectory(
            raw, planner.joint_names, contract
        )
        validation = validate_dense_trajectory(
            traj,
            contract,
            start_tolerance_rad=1e-6,
            goal_tolerance_rad=1e-4,
            locked_tolerance_rad=1e-7,
        )
        constraints = raw_constraint_summary(planner, raw)
        if constraints["scene_collision_max"] is None or constraints[
            "cspace_max"
        ] is None:
            raise RuntimeError(f"{stage}: constraint metrics unavailable")
        voxel = external_voxel_clearance(
            planner, scene_model, traj.q_rad
        )
        hard_pass = (
            float(constraints["scene_collision_max"]) == 0.0
            and int(constraints["scene_collision_positive_count"] or 0) == 0
            and float(constraints["cspace_max"]) == 0.0
            and int(constraints["cspace_positive_count"] or 0) == 0
            and not bool(voxel["voxel_collision"])
        )
        if not hard_pass:
            if verbose:
                print(
                    f"[Route B][PLAN] FAIL {stage}: scene={constraints['scene_collision_max']} "
                    f"cspace={constraints['cspace_max']} voxel_collision={voxel['voxel_collision']}",
                    flush=True,
                )
            return None, {
                "stage": stage,
                "success": False,
                "constraints": constraints,
                "voxel": voxel,
        }
        save_right_arm_npz(artifact_path, traj)
        if verbose:
            print(
                f"[Route B][PLAN] PASS {stage}: points={traj.point_count} "
                f"dt={float(traj.dt_s):.4f}s duration={float(traj.duration_s):.3f}s "
                f"min_clearance={voxel['voxel_min_clearance_m']:.4f}m",
                flush=True,
            )
        return traj, {
            "stage": stage,
            "success": True,
            "point_count": int(traj.point_count),
            "duration_s": float(traj.duration_s),
            "dt_s": float(traj.dt_s),
            "constraints": constraints,
            "voxel": voxel,
            "validation": validation,
            "artifact": str(Path(artifact_path).resolve()),
        }

    # 1) PREGRASP -> COVER with open/pregrasp hand; target is absent from ESDF.
    print("[Route B full 1/5] PREGRASP -> COVER", flush=True)
    approach_planner = build_planner(hand["pregrasp"], attachment=False)
    approach_path = output / "traj_pregrasp_to_cover.npz"
    approach_traj, approach_report = plan_segment(
        approach_planner,
        scene,
        q_pre,
        q_cover,
        "PREGRASP_TO_COVER",
        approach_path,
        verbose=True,
    )
    approach_planner.destroy()
    if approach_traj is None:
        trials = [
            {
                "chain_index": None,
                "success": False,
                "failed_stage": "PREGRASP_TO_COVER",
            }
        ]
        summary = summarize_trials(trials)
        failure_report = {
            "schema_version": 1,
            "route": "RouteB",
            "stage": "FULL_MOTION_PLAN",
            "success": False,
            "selected_candidate": int(selected["candidate_index"]),
            "selected_case": str(case_root),
            "failure_type": "PREGRASP_TO_COVER_MOTIONPLANNER_FAILED",
            "failure_stage": "PREGRASP_TO_COVER",
            "trial_summary": summary,
            "trials": trials,
            "approach_report": approach_report,
            "target_proxy": proxy.to_jsonable(),
            "transfer_attachment": bool(transfer_attachment_enabled),
            "diagnostic_disable_transfer_attachment": bool(
                args.diagnostic_disable_transfer_attachment
            ),
            "front_half_trajectory": str(front_traj.resolve()),
            "isaac_execution_started": False,
            "total_wall_time_s": float(time.perf_counter() - started),
        }
        report_path = output / "routeB_full_plan_report.json"
        report_path.write_text(
            json.dumps(failure_report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(
            "[Route B][FULL] FAIL "
            f"candidate={failure_report['selected_candidate']} "
            "dominant_stage=PREGRASP_TO_COVER trials=1/1 "
            f"report={report_path}",
            flush=True,
        )
        return 0

    # 2) Build contact/free-space planners for the post-grasp route.
    # Intentional support contact is handled like PREGRASP->COVER: target is absent
    # from ESDF and the target proxy is NOT attached on COVER->LIFT or TRANSFER->PLACE.
    # The attachment is active only on the true free-space carry LIFT->TRANSFER.
    print("[Route B full 2/5] build lift / attached-transfer / place planners", flush=True)
    lift_planner = build_planner(hand["squeeze"], attachment=False)
    transfer_planner = build_planner(
        hand["squeeze"],
        attachment=transfer_attachment_enabled,
    )
    place_planner = build_planner(hand["squeeze"], attachment=False)

    local_proxy = Cuboid(
        name="routeB_target_proxy_local",
        pose=[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        dims=proxy.dims_base_m.tolist(),
    )
    fitted = transfer_planner.attachment_manager.fit_spheres(
        [local_proxy],
        num_spheres=int(args.attachment_sphere_count),
        sphere_fit_type=SphereFitType.VOXEL,
    )

    # Preserve the measured object-to-flange transform at exact COVER.  We use
    # it later to place the attachment at each candidate LIFT and to predict the
    # object proxy at PLACE.
    cover_flange = pose_to_matrix(
        lift_planner.compute_kinematics(
            make_joint_state(q_cover, lift_planner)
        ).tool_poses.get_link_pose("arm_r_link_tf")
    )
    proxy_initial = np.eye(4, dtype=np.float64)
    proxy_initial[:3, 3] = proxy.center_base_m
    flange_from_proxy = np.linalg.inv(cover_flange) @ proxy_initial

    # 3) Test endpoint chains with true MotionPlanner.
    print("[Route B full 3/5] test back-half endpoint chains", flush=True)
    trials = []
    selected_chain = None
    selected_reports = None
    max_trials = min(chain_count, int(args.max_chain_trials))
    for chain_i in range(max_trials):
        q_lift = np.asarray(chains["q_lift_rad"][chain_i], dtype=np.float64)
        q_transfer = np.asarray(
            chains["q_transfer_rad"][chain_i], dtype=np.float64
        )
        q_place = np.asarray(chains["q_place_rad"][chain_i], dtype=np.float64)
        q_retreat = np.asarray(
            chains["q_retreat_rad"][chain_i], dtype=np.float64
        )
        trial_dir = output / f"_trial_{chain_i:02d}"
        trial_dir.mkdir(parents=True, exist_ok=True)
        reports = []

        # COVER->LIFT: intentional target/table support contact phase.  Robot
        # collision is checked against non-target ESDF; the carried target proxy
        # is not attached until the LIFT endpoint is clear of the support.
        lift_traj, lift_rep = plan_segment(
            lift_planner,
            scene,
            q_cover,
            q_lift,
            "COVER_TO_LIFT",
            trial_dir / "traj_cover_to_lift.npz",
            verbose=False,
        )
        reports.append(lift_rep)
        if lift_traj is None:
            trials.append(
                {
                    "chain_index": chain_i,
                    "success": False,
                    "failed_stage": "COVER_TO_LIFT",
                }
            )
            continue

        # Attach at the candidate LIFT pose and keep attachment active only for
        # the free-space transfer segment.
        lift_flange = pose_to_matrix(
            transfer_planner.compute_kinematics(
                make_joint_state(q_lift, transfer_planner)
            ).tool_poses.get_link_pose("arm_r_link_tf")
        )
        proxy_at_lift = lift_flange @ flange_from_proxy
        if transfer_attachment_enabled:
            transfer_planner.attachment_manager.update(
                fitted,
                make_joint_state(q_lift, transfer_planner),
                link_name=ATTACHED_LINK,
                world_objects_pose_offset=Pose.from_list(
                    matrix_to_pose_list(proxy_at_lift),
                    device_cfg=transfer_planner.device_cfg,
                ),
            )
        transfer_traj, transfer_rep = plan_segment(
            transfer_planner,
            scene,
            q_lift,
            q_transfer,
            "LIFT_TO_TRANSFER",
            trial_dir / "traj_lift_to_transfer.npz",
            verbose=False,
        )
        reports.append(transfer_rep)
        if transfer_traj is None:
            trials.append(
                {
                    "chain_index": chain_i,
                    "success": False,
                    "failed_stage": "LIFT_TO_TRANSFER",
                }
            )
            continue

        # TRANSFER->PLACE: the target proxy is deliberately detached because
        # target/table contact at the placement endpoint is intentional.  The
        # right arm/hand still plans against the non-target ESDF.
        place_traj, place_rep = plan_segment(
            place_planner,
            scene,
            q_transfer,
            q_place,
            "TRANSFER_TO_PLACE",
            trial_dir / "traj_transfer_to_place.npz",
            verbose=False,
        )
        reports.append(place_rep)
        if place_traj is None:
            trials.append(
                {
                    "chain_index": chain_i,
                    "success": False,
                    "failed_stage": "TRANSFER_TO_PLACE",
                }
            )
            continue

        # Predict the placed proxy at planned q_place.  The immediate
        # PLACE->RETREAT segment starts from intentional target/hand contact, so
        # it still uses the non-target ESDF.  The placed proxy becomes a static
        # obstacle after the hand has retreated and is used for RETREAT->HOME.
        place_flange = pose_to_matrix(
            place_planner.compute_kinematics(
                make_joint_state(q_place, place_planner)
            ).tool_poses.get_link_pose("arm_r_link_tf")
        )
        placed_proxy = place_flange @ flange_from_proxy
        placed_pose = matrix_to_pose_list(placed_proxy)
        post_scene = scene.clone()
        post_scene.cuboid.append(
            Cuboid(
                name="routeB_placed_target_proxy",
                pose=placed_pose,
                dims=proxy.dims_base_m.tolist(),
            )
        )
        post_scene.objects = (
            post_scene.sphere
            + post_scene.cuboid
            + post_scene.capsule
            + post_scene.mesh
            + post_scene.cylinder
            + post_scene.voxel
        )
        home_q = np.deg2rad(
            np.asarray(
                load_json(
                    root
                    / "08_dual_arm_scene_layout/isaaclab_control/closed_loop/config/closed_loop.json"
                ).get("home_q_deg", [50, -70, 0, 40, 35, 0, 25]),
                dtype=np.float64,
            )
        )
        retreat_traj, retreat_rep = plan_segment(
            place_planner,
            scene,
            q_place,
            q_retreat,
            "PLACE_TO_RETREAT",
            trial_dir / "traj_place_to_retreat.npz",
            verbose=False,
        )
        if retreat_traj is None:
            trials.append(
                {
                    "chain_index": chain_i,
                    "success": False,
                    "failed_stage": "PLACE_TO_RETREAT",
                }
            )
            continue
        post_planner = build_planner(
            hand["pregrasp"], attachment=False, scene_override=post_scene
        )
        home_traj, home_rep = plan_segment(
            post_planner,
            post_scene,
            q_retreat,
            home_q,
            "RETREAT_TO_HOME",
            trial_dir / "traj_retreat_to_home.npz",
            verbose=False,
        )
        post_planner.destroy()
        if home_traj is None:
            trials.append(
                {
                    "chain_index": chain_i,
                    "success": False,
                    "failed_stage": "RETREAT_TO_HOME",
                }
            )
            continue

        selected_chain = {
            "chain_index": int(chain_i),
            "score": float(chains["score"][chain_i]),
            "q_cover_rad": q_cover.tolist(),
            "q_lift_rad": q_lift.tolist(),
            "q_transfer_rad": q_transfer.tolist(),
            "q_place_rad": q_place.tolist(),
            "q_retreat_rad": q_retreat.tolist(),
            "q_home_rad": home_q.tolist(),
            "placed_proxy_pose_base_wxyz": placed_pose,
        }
        selected_reports = reports + [retreat_rep, home_rep]
        trials.append({"chain_index": chain_i, "success": True})
        break

    lift_planner.destroy()
    transfer_planner.destroy()
    place_planner.destroy()
    if selected_chain is None:
        summary = summarize_trials(trials)
        failure_report = {
            "schema_version": 1,
            "route": "RouteB",
            "stage": "FULL_MOTION_PLAN",
            "success": False,
            "selected_candidate": int(selected["candidate_index"]),
            "selected_case": str(case_root),
            "failure_type": "NO_BACKHALF_CHAIN_PASSED_TRUE_MOTIONPLANNER",
            "failure_stage": summary.get("dominant_failed_stage"),
            "trial_summary": summary,
            "trials": trials,
            "target_proxy": proxy.to_jsonable(),
            "transfer_attachment": bool(transfer_attachment_enabled),
            "diagnostic_disable_transfer_attachment": bool(
                args.diagnostic_disable_transfer_attachment
            ),
            "front_half_trajectory": str(front_traj.resolve()),
            "isaac_execution_started": False,
            "total_wall_time_s": float(time.perf_counter() - started),
        }
        report_path = output / "routeB_full_plan_report.json"
        report_path.write_text(
            json.dumps(failure_report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(
            "[Route B][FULL] FAIL "
            f"candidate={failure_report['selected_candidate']} "
            f"dominant_stage={summary.get('dominant_failed_stage')} "
            f"trials={summary['failure_count']}/{summary['trial_count']} "
            f"report={report_path}",
            flush=True,
        )
        return 0

    # 4) Promote selected trial trajectories.
    import shutil

    trial_dir = output / f"_trial_{selected_chain['chain_index']:02d}"
    canonical = [
        "traj_cover_to_lift.npz",
        "traj_lift_to_transfer.npz",
        "traj_transfer_to_place.npz",
        "traj_place_to_retreat.npz",
        "traj_retreat_to_home.npz",
    ]
    for filename in canonical:
        shutil.copy2(trial_dir / filename, output / filename)

    # 5) Execution manifest.
    print("[Route B full 4/5] write execution manifest", flush=True)
    manifest = {
        "schema_version": 1,
        "route": "RouteB",
        "query": args.query,
        "case_root": str(case_root),
        "candidate_index": int(selected["candidate_index"]),
        "target_segmentation_id": int(target_segmentation_id),
        "hand_waypoints": str(
            (case_root / "06_isaacsim/final_waypoints.npz").resolve()
        ),
        "front_half_report": str(
            (front_dir / "routeB_front_half_report.json").resolve()
        ),
        "target_proxy": proxy.to_jsonable(),
        "selected_chain": selected_chain,
        "segments": [
            {
                "state": "CURRENT_TO_PREGRASP",
                "trajectory": str(front_traj.resolve()),
            },
            {
                "state": "PREGRASP_TO_COVER",
                "trajectory": str(approach_path.resolve()),
            },
            {
                "state": "COVER_TO_LIFT",
                "trajectory": str((output / "traj_cover_to_lift.npz").resolve()),
            },
            {
                "state": "LIFT_TO_TRANSFER",
                "trajectory": str(
                    (output / "traj_lift_to_transfer.npz").resolve()
                ),
            },
            {
                "state": "TRANSFER_TO_PLACE",
                "trajectory": str(
                    (output / "traj_transfer_to_place.npz").resolve()
                ),
            },
            {
                "state": "PLACE_TO_RETREAT",
                "trajectory": str(
                    (output / "traj_place_to_retreat.npz").resolve()
                ),
            },
            {
                "state": "RETREAT_TO_HOME",
                "trajectory": str(
                    (output / "traj_retreat_to_home.npz").resolve()
                ),
            },
        ],
        "collision_policy": {
            "LEAP_reach": "OFF",
            "exact_cover_observed_ESDF": "OFF",
            "pregrasp_endpoint_collision": "OFF",
            "current_to_pregrasp_environment": "ON",
            "pregrasp_to_cover_non_target_environment": "ON",
            "cover_to_lift_non_target_environment": "ON (target proxy detached for intentional support contact)",
            "lift_to_transfer_environment": "ON",
            "lift_to_transfer_attachment": "ON" if transfer_attachment_enabled else "OFF",
            "transfer_to_place_non_target_environment": "ON (target proxy detached for intentional placement contact)",
            "post_release_environment_plus_placed_target": "ON",
            "self_collision": "OFF",
            "Isaac_PhysX": "ON",
        },
        "arm_dense_execution": "linear time interpolation only; no quintic arm regeneration",
    }
    manifest_path = output / "routeB_execution_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    # Include the already-validated front-half segment in the full summary.
    front_traj_data = np.load(front_traj, allow_pickle=False)
    front_q = np.asarray(front_traj_data["q_rad"], dtype=np.float64)
    front_time = np.asarray(front_traj_data["time_s"], dtype=np.float64).reshape(-1)
    front_post = front_report.get("postcheck", {})
    front_segment_report = {
        "stage": "CURRENT_TO_PREGRASP",
        "success": True,
        "point_count": int(len(front_q)),
        "duration_s": float(front_time[-1]) if len(front_time) else 0.0,
        "dt_s": float(np.asarray(front_traj_data["dt_s"]).reshape(-1)[0]) if "dt_s" in front_traj_data.files else None,
        "constraints": {
            "scene_collision_max": front_post.get("scene_collision_max"),
            "scene_collision_positive_count": front_post.get("scene_collision_positive_count"),
            "cspace_max": front_post.get("cspace_max"),
            "cspace_positive_count": front_post.get("cspace_positive_count"),
        },
        "voxel": {
            "voxel_collision": bool(front_post.get("environment_collision", False)),
            "voxel_min_clearance_m": front_post.get("min_environment_clearance_m"),
        },
        "artifact": str(front_traj.resolve()),
    }

    full_report = {
        "schema_version": 1,
        "route": "RouteB",
        "stage": "FULL_MOTION_PLAN",
        "success": True,
        "selected_candidate": int(selected["candidate_index"]),
        "selected_case": str(case_root),
        "selected_chain": selected_chain,
        "target_proxy": proxy.to_jsonable(),
        "transfer_attachment": bool(transfer_attachment_enabled),
        "diagnostic_disable_transfer_attachment": bool(
            args.diagnostic_disable_transfer_attachment
        ),
        "attachment_fitted_sphere_count": int(fitted.shape[0]),
        "segments": [front_segment_report, approach_report] + list(selected_reports or []),
        "front_half_trajectory": str(front_traj.resolve()),
        "execution_manifest": str(manifest_path),
        "trials": trials,
        "isaac_execution_started": False,
        "total_wall_time_s": float(time.perf_counter() - started),
    }
    report_path = output / "routeB_full_plan_report.json"
    report_path.write_text(
        json.dumps(full_report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print("[Route B full 5/5] FULL MOTION PLAN PASS", flush=True)
    print("========================================", flush=True)
    print(" Route B FULL MOTION PLAN PASS", flush=True)
    print("========================================", flush=True)
    print(f"candidate        : {full_report['selected_candidate']}", flush=True)
    print(f"backhalf chain   : {selected_chain['chain_index']}", flush=True)
    print(f"attachment proxy : {np.round(proxy.dims_base_m, 4).tolist()} m / {int(fitted.shape[0])} spheres", flush=True)
    for row in full_report['segments']:
        vox = row.get('voxel', {})
        con = row.get('constraints', {})
        print(
            f"{row['stage']:<22}: {row.get('point_count','?')} pts "
            f"{row.get('duration_s','?')} s | clear={vox.get('voxel_min_clearance_m')} "
            f"scene={con.get('scene_collision_max')} cspace={con.get('cspace_max')}",
            flush=True,
        )
    print("Isaac execution  : NOT STARTED", flush=True)
    print("========================================", flush=True)
    print(f"report           : {report_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
