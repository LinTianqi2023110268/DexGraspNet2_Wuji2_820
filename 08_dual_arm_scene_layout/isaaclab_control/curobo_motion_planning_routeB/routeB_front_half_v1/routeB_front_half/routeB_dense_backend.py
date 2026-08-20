#!/usr/bin/env python3
from __future__ import annotations

"""Production Route B current->PREGRASP multi-goal backend.

Build the RobotSegmenter-cleaned ESDF once, build one true 7DOF locked-joint
MotionPlanner once, then try the ordered post-retarget PREGRASP goals until
one collision-free dense trajectory succeeds.

No Isaac execution happens here.
"""

import argparse
import copy
import json
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np

try:
    from .metrics import raw_constraint_summary
    from .pregrasp_pool import load_front_half_goal_pool
except ImportError:
    from metrics import raw_constraint_summary  # type: ignore
    from pregrasp_pool import load_front_half_goal_pool  # type: ignore


RIGHT_ARM_JOINTS = tuple(f"arm_r_joint_{i}" for i in range(1, 8))


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_yaml(path: Path) -> dict[str, Any]:
    import yaml
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def json_ready(value: Any) -> Any:
    """Convert numpy values in reports to standard JSON types."""
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(v) for v in value]
    if isinstance(value, np.ndarray):
        return json_ready(value.tolist())
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def tensor_success(value: Any) -> bool:
    if value is None:
        return False
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    return bool(np.asarray(value, dtype=bool).any())


def make_joint_state(q: np.ndarray, planner: Any):
    import torch
    from curobo.types import JointState
    t = torch.as_tensor(
        np.asarray(q, dtype=np.float32),
        device=planner.device_cfg.device,
        dtype=planner.device_cfg.dtype,
    ).view(1, -1)
    return JointState.from_position(t)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--capture-dir", type=Path, required=True)
    parser.add_argument("--goal-pool", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-attempts", type=int, default=2)
    parser.add_argument("--enable-graph-attempt", type=int, default=1000000)
    parser.add_argument("--num-ik-seeds", type=int, default=32)
    parser.add_argument("--num-trajopt-seeds", type=int, default=4)
    parser.add_argument("--interpolation-dt-s", type=float, default=0.025)
    parser.add_argument("--warmup-iterations", type=int, default=1)
    parser.add_argument("--use-cuda-graph", action="store_true")
    parser.add_argument("--max-goals-to-try", type=int, default=128)
    args = parser.parse_args()

    started = time.perf_counter()
    project_root = args.project_root.expanduser().resolve()
    capture_dir = args.capture_dir.expanduser().resolve()
    goal_pool_path = args.goal_pool.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    control_root = project_root / "08_dual_arm_scene_layout/isaaclab_control"
    routeb_dir = control_root / "curobo_motion_planning_routeB"
    right_arm_core_root = routeb_dir / "routeB_right_arm_only_core_v1"
    for path in (control_root, routeb_dir, right_arm_core_root):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))

    from curobo_motion_planning_routeB import RouteBMotionPlannerAdapter
    from curobo_motion_planning_routeB.routeB_adapter import (
        DEFAULT_LAYOUT_JSON,
        DEFAULT_ROBOT_FILE,
    )
    from right_arm_only_core import (
        RIGHT_ARM_JOINTS as CORE_RIGHT_ARM_JOINTS,
        build_locked_joint_contract,
        rebuild_robot_cfg_with_lock_joints,
    )
    from right_arm_only_core.trajectory import (
        extract_dense_right_arm_trajectory,
        save_right_arm_npz,
        validate_dense_trajectory,
    )
    from core.perception_collision.esdf_collision import query_spheres
    from core.perception_collision.robot_spheres import CuroboRobotSphereModel

    if tuple(CORE_RIGHT_ARM_JOINTS) != RIGHT_ARM_JOINTS:
        raise RuntimeError("right_arm_only_core joint contract mismatch")

    inputs = {
        "filtered_depth": capture_dir / "planning/filtered_depth.npy",
        "intrinsics": capture_dir / "intrinsics.npy",
        "T_world_camera": capture_dir / "T_world_camera.npy",
        "robot_state": capture_dir / "robot_state.json",
        "goal_pool": goal_pool_path,
    }
    missing = [str(p) for p in inputs.values() if not p.is_file()]
    if missing:
        raise FileNotFoundError(
            "missing Route B front-half inputs: " + ", ".join(missing)
        )

    pool = load_front_half_goal_pool(goal_pool_path)
    q_pre_all = np.asarray(pool["q_pregrasp_rad"], dtype=np.float64)
    q_cover_all = np.asarray(pool["q_cover_rad"], dtype=np.float64)
    case_root_all = pool["case_root"].astype(str)
    candidate_index_all = np.asarray(
        pool["candidate_index"], dtype=np.int64
    )
    official_score_all = np.asarray(
        pool["official_score"], dtype=np.float64
    )
    pair_score_all = np.asarray(pool["pair_score"], dtype=np.float64)

    cfg = {
        "routeB": {
            "device": args.device,
            "robot_file": str(DEFAULT_ROBOT_FILE),
            "layout_json": str(DEFAULT_LAYOUT_JSON),
            "collision": {
                "environment_collision": True,
                "self_collision": False,
            },
            "use_cuda_graph": bool(args.use_cuda_graph),
            "num_ik_seeds": int(args.num_ik_seeds),
            "num_trajopt_seeds": int(args.num_trajopt_seeds),
            "max_attempts": int(args.max_attempts),
            "enable_graph_attempt": int(args.enable_graph_attempt),
            "warmup_iterations": int(args.warmup_iterations),
            "interpolation_dt_s": float(args.interpolation_dt_s),
        }
    }
    adapter = RouteBMotionPlannerAdapter(cfg)

    print("[Route B front-half 1/4] build RobotSegmenter-cleaned scene", flush=True)
    scene = adapter.build_pick_scene(
        inputs["filtered_depth"],
        inputs["intrinsics"],
        inputs["T_world_camera"],
    )

    # Create the existing full adapter model only to recover the canonical
    # full joint order/bounds and apply the already-validated numerical
    # sanitization contract. It does NOT generate a full-DOF trajectory.
    adapter.create_planner(scene)
    robot_state = load_json(inputs["robot_state"])
    measured = {
        str(k): float(v)
        for k, v in robot_state["joint_positions_by_name"].items()
    }
    q_current_raw = adapter._coerce_q(measured)
    full_joint_names = list(adapter.joint_names)

    sanitized_goals: list[np.ndarray] = []
    sanitization_reports: list[dict[str, Any]] = []
    q_current_planning_ref = None
    for q_pre in q_pre_all:
        q_goal_raw = adapter._coerce_q(
            q_pre.astype(np.float32),
            base_q=q_current_raw,
        )
        q_cur_plan, q_goal_plan, san_report = (
            adapter.sanitize_planning_joint_states(
                q_current_raw,
                q_goal_raw,
            )
        )
        if q_current_planning_ref is None:
            q_current_planning_ref = q_cur_plan
        elif float(
            np.max(
                np.abs(
                    np.asarray(q_cur_plan)
                    - np.asarray(q_current_planning_ref)
                )
            )
        ) > 1e-10:
            raise RuntimeError(
                "q_current planning sanitization changed across goals"
            )
        sanitized_goals.append(np.asarray(q_goal_plan, dtype=np.float64))
        sanitization_reports.append(san_report)

    assert q_current_planning_ref is not None
    q_current_planning = np.asarray(
        q_current_planning_ref, dtype=np.float64
    )

    first_contract = build_locked_joint_contract(
        full_joint_names,
        q_current_planning,
        sanitized_goals[0],
    )
    robot_source = load_yaml(adapter.robot_file)
    locked_robot_cfg = rebuild_robot_cfg_with_lock_joints(
        robot_source,
        first_contract.lock_joints,
    )

    print(
        "[Route B front-half 2/4] build one true 7DOF locked planner | "
        f"locked={first_contract.locked_joint_count}",
        flush=True,
    )
    import torch
    from curobo.motion_planner import MotionPlanner, MotionPlannerCfg
    from curobo.types import DeviceCfg

    device_cfg = DeviceCfg(
        device=torch.device(adapter.device),
        dtype=torch.float32,
    )
    motion_cfg = MotionPlannerCfg.create(
        robot=locked_robot_cfg,
        scene_model=scene,
        device_cfg=device_cfg,
        self_collision_check=adapter.self_collision_check,
        num_ik_seeds=adapter.num_ik_seeds,
        num_trajopt_seeds=adapter.num_trajopt_seeds,
        use_cuda_graph=adapter.use_cuda_graph,
        interpolation_dt=adapter.interpolation_dt_s,
    )
    collision_policy_report = (
        adapter._apply_collision_policy_to_motion_cfg(motion_cfg)
    )
    planner = MotionPlanner(motion_cfg)
    voxel_contract_report = adapter._normalize_voxel_shape_contract(
        planner,
        scene,
    )
    planner.warmup(
        enable_graph=adapter.use_cuda_graph,
        num_warmup_iterations=adapter.warmup_iterations,
    )

    if int(planner.action_dim) != 7:
        raise RuntimeError(
            f"Route B locked planner action_dim={planner.action_dim}, expected 7"
        )
    if tuple(str(x) for x in planner.joint_names) != RIGHT_ARM_JOINTS:
        raise RuntimeError(
            "Route B active joint order mismatch: "
            f"{planner.joint_names}"
        )

    sphere_model = CuroboRobotSphereModel(
        adapter.robot_file,
        device=args.device,
    )
    full_index = {
        name: i for i, name in enumerate(full_joint_names)
    }

    def postcheck(traj, contract, raw_result):
        min_clearance = float("inf")
        collision_samples = 0
        worst = None
        for t, q_right in enumerate(traj.q_rad):
            q_map = {
                name: float(contract.lock_joints[name])
                for name in contract.locked_joint_names
            }
            for name, value in zip(RIGHT_ARM_JOINTS, q_right):
                q_map[name] = float(value)
            spheres = sphere_model.spheres_from_named_joints(q_map)
            batch = query_spheres(
                scene.voxel[0],
                spheres[:, :3],
                spheres[:, 3],
                margin_m=0.0,
            )
            clearance = (
                np.asarray(batch.distance_m, dtype=np.float64)
                - spheres[:, 3]
            )
            i = int(np.argmin(clearance))
            if float(clearance[i]) < min_clearance:
                min_clearance = float(clearance[i])
                worst = {
                    "timestep": int(t),
                    "sphere_index": int(i),
                    "link_name": (
                        sphere_model.sphere_link_names[i]
                        if i < len(sphere_model.sphere_link_names)
                        else None
                    ),
                    "clearance_m": float(clearance[i]),
                }
            if bool(np.any(batch.collision)):
                collision_samples += 1

        lower, upper = adapter._motion_planner_position_bounds_for(planner)
        lower = np.asarray(lower, dtype=np.float64)[:7]
        upper = np.asarray(upper, dtype=np.float64)[:7]
        q = np.asarray(traj.q_rad, dtype=np.float64)
        violation = np.maximum(
            lower[None] - q,
            q - upper[None],
        )
        joint_limit_pass = bool(
            np.count_nonzero(violation > 0.0) == 0
        )
        constraints = raw_constraint_summary(
            planner,
            raw_result,
            sphere_link_names=list(sphere_model.sphere_link_names),
        )
        return {
            "environment_collision": bool(
                collision_samples > 0
            ),
            "environment_collision_pass": bool(
                collision_samples == 0
            ),
            "environment_collision_sample_count": int(
                collision_samples
            ),
            "min_environment_clearance_m": float(
                min_clearance
            ),
            "environment_worst_sample": worst,
            "joint_limit_pass": joint_limit_pass,
            "joint_limit_violation_count": int(
                np.count_nonzero(violation > 0.0)
            ),
            "velocity_max_abs_rad_s": float(
                np.max(np.abs(traj.qd_rad_s))
            ),
            "acceleration_max_abs_rad_s2": float(
                np.max(np.abs(traj.qdd_rad_s2))
            ),
            "jerk_max_abs_rad_s3": float(
                np.max(np.abs(traj.jerk_rad_s3))
            ),
            "velocity_pass": bool(
                np.isfinite(traj.qd_rad_s).all()
            ),
            "acceleration_pass": bool(
                np.isfinite(traj.qdd_rad_s2).all()
            ),
            "jerk_pass": bool(
                np.isfinite(traj.jerk_rad_s3).all()
            ),
            **constraints,
        }

    q_current_active = np.asarray(
        first_contract.q_current_active,
        dtype=np.float64,
    )
    current_state = make_joint_state(q_current_active, planner)

    trials: list[dict[str, Any]] = []
    selected = None
    trajectory_obj = None
    selected_contract = None
    selected_post = None
    max_goals = min(
        len(q_pre_all),
        max(1, int(args.max_goals_to_try)),
    )

    print(
        "[Route B front-half 3/4] try ordered PREGRASP goals | "
        f"goals={max_goals}",
        flush=True,
    )
    for goal_i in range(max_goals):
        contract = build_locked_joint_contract(
            full_joint_names,
            q_current_planning,
            sanitized_goals[goal_i],
        )
        if contract.lock_joints != first_contract.lock_joints:
            raise RuntimeError(
                "locked joint values changed across PREGRASP goals"
            )

        goal_state = make_joint_state(
            np.asarray(contract.q_goal_active, dtype=np.float64),
            planner,
        )
        solve_started = time.perf_counter()
        raw = planner.plan_cspace(
            goal_state=goal_state,
            current_state=current_state,
            max_attempts=int(args.max_attempts),
            enable_graph_attempt=int(args.enable_graph_attempt),
        )
        solve_wall = time.perf_counter() - solve_started
        success = tensor_success(getattr(raw, "success", None))
        trial = {
            "goal_index": int(goal_i),
            "case_root": str(case_root_all[goal_i]),
            "candidate_index": int(candidate_index_all[goal_i]),
            "official_score": float(official_score_all[goal_i]),
            "pair_score": float(pair_score_all[goal_i]),
            "success": bool(success),
            "solve_wall_time_s": float(solve_wall),
            "result_solution_shape": (
                None
                if getattr(raw, "solution", None) is None
                else [
                    int(x)
                    for x in getattr(raw, "solution").shape
                ]
            ),
        }
        if not success:
            trials.append(trial)
            print(
                f"  goal {goal_i+1}/{max_goals} "
                f"cand={trial['candidate_index']} FAIL "
                f"{solve_wall:.2f}s",
                flush=True,
            )
            continue

        solution = getattr(raw, "solution", None)
        if solution is None or int(solution.shape[-1]) != 7:
            raise RuntimeError(
                "planner success but optimizer result is not true 7DOF"
            )

        traj = extract_dense_right_arm_trajectory(
            raw,
            planner.joint_names,
            contract,
        )
        validation = validate_dense_trajectory(
            traj,
            contract,
            start_tolerance_rad=1e-6,
            goal_tolerance_rad=1e-4,
            locked_tolerance_rad=1e-7,
        )
        post = postcheck(traj, contract, raw)

        scene_max = post.get("scene_collision_max")
        scene_pos = post.get("scene_collision_positive_count")
        cspace_max = post.get("cspace_max")
        cspace_pos = post.get("cspace_positive_count")
        constraints_available = (
            scene_max is not None
            and scene_pos is not None
            and cspace_max is not None
            and cspace_pos is not None
        )
        hard_pass = (
            constraints_available
            and not bool(post["environment_collision"])
            and bool(post["joint_limit_pass"])
            and bool(post["velocity_pass"])
            and bool(post["acceleration_pass"])
            and bool(post["jerk_pass"])
            and float(scene_max) == 0.0
            and int(scene_pos) == 0
            and float(cspace_max) == 0.0
            and int(cspace_pos) == 0
        )
        trial.update(
            {
                "dense_point_count": int(traj.point_count),
                "validation": validation,
                "postcheck": post,
                "hard_pass": bool(hard_pass),
            }
        )
        trials.append(trial)
        if not hard_pass:
            print(
                f"  goal {goal_i+1}/{max_goals} "
                f"cand={trial['candidate_index']} "
                "planner success but postcheck FAIL",
                flush=True,
            )
            continue

        selected = {
            "goal_index": int(goal_i),
            "case_root": str(case_root_all[goal_i]),
            "candidate_index": int(candidate_index_all[goal_i]),
            "official_score": float(official_score_all[goal_i]),
            "pair_score": float(pair_score_all[goal_i]),
            "q_pregrasp_rad": q_pre_all[goal_i].tolist(),
            "q_cover_rad": q_cover_all[goal_i].tolist(),
            "sanitization": sanitization_reports[goal_i],
            "validation": validation,
        }
        trajectory_obj = traj
        selected_contract = contract
        selected_post = post
        print(
            f"  ✓ goal {goal_i+1}/{max_goals} "
            f"cand={selected['candidate_index']} "
            f"points={traj.point_count} "
            f"clearance={post['min_environment_clearance_m']:.4f}m",
            flush=True,
        )
        break

    if selected is None or trajectory_obj is None:
        report = {
            "schema_version": 1,
            "route": "RouteB",
            "stage": "front_half_current_to_pregrasp",
            "success": False,
            "reason": "NO_PREGRASP_GOAL_WITH_VALID_CUROBO_TRAJECTORY",
            "planner": {
                "action_dim": int(planner.action_dim),
                "active_joint_names": list(planner.joint_names),
                "locked_joint_count": int(
                    first_contract.locked_joint_count
                ),
            },
            "goal_pool": str(goal_pool_path),
            "trials": trials,
            "total_wall_time_s": float(
                time.perf_counter() - started
            ),
        }
        report = json_ready(report)
        report_path = output_dir / "routeB_front_half_report.json"
        report_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(
            "[Route B front-half] FAIL | "
            f"reason={report['reason']} | report={report_path}",
            flush=True,
        )
        return 3

    print(
        "[Route B front-half 4/4] save selected dense trajectory",
        flush=True,
    )
    trajectory_path = save_right_arm_npz(
        output_dir / "trajectory_right_arm.npz",
        trajectory_obj,
    )
    plan_path = output_dir / "routeB_front_half_plan.npz"
    np.savez_compressed(
        plan_path,
        arm_joint_names=np.asarray(RIGHT_ARM_JOINTS, dtype="U"),
        q_current_rad=np.asarray(
            selected_contract.q_current_active,
            dtype=np.float32,
        ),
        q_pregrasp_rad=np.asarray(
            selected["q_pregrasp_rad"], dtype=np.float32
        ),
        q_cover_rad=np.asarray(
            selected["q_cover_rad"], dtype=np.float32
        ),
        case_root=np.asarray(selected["case_root"], dtype="U"),
        candidate_index=np.asarray(
            selected["candidate_index"], dtype=np.int64
        ),
        goal_index=np.asarray(selected["goal_index"], dtype=np.int64),
    )

    report = {
        "schema_version": 1,
        "route": "RouteB",
        "stage": "front_half_current_to_pregrasp",
        "success": True,
        "selected": selected,
        "planner": {
            "action_dim": int(planner.action_dim),
            "active_joint_names": list(planner.joint_names),
            "locked_joint_count": int(
                first_contract.locked_joint_count
            ),
            "environment_collision": True,
            "self_collision": False,
            "collision_policy_report": collision_policy_report,
            "voxel_shape_contract": voxel_contract_report,
        },
        "trajectory": {
            **trajectory_obj.to_report(),
            "artifact": str(trajectory_path),
        },
        "postcheck": selected_post,
        "goal_pool": str(goal_pool_path),
        "front_half_plan": str(plan_path),
        "trials": trials,
        "routeA_modified": False,
        "isaac_execution_started": False,
        "total_wall_time_s": float(
            time.perf_counter() - started
        ),
    }
    report = json_ready(report)
    report_path = output_dir / "routeB_front_half_report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        "[Route B front-half] PASS | "
        f"candidate={selected['candidate_index']} | "
        f"trajectory={list(trajectory_obj.q_rad.shape)} | "
        f"min_clearance={selected_post.get('min_environment_clearance_m')} | "
        f"report={report_path}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
