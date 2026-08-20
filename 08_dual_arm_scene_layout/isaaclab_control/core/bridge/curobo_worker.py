#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
import traceback
from pathlib import Path

import numpy as np

_PROTOCOL = "__CUROBO_CORE__"


def emit(payload: dict) -> None:
    print(_PROTOCOL + json.dumps(payload, separators=(",", ":")), flush=True)


def jsonable(value):
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(x) for x in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return value.item()
    return value


def selected_collision_records_for_independent_targets(
    selected: list,
    feasible_solutions: list[list[dict]],
) -> list[dict | None]:
    """Match selected independent-target IK solutions to collision records.

    ``select_chain=False`` means each target is a separate candidate, not a
    waypoint in one path.  A target may legitimately have no selected solution,
    or may have IK accepted solutions but no collision-feasible solution.  Both
    cases must be serialized as ``null`` instead of crashing or forcing a path
    collision check across unrelated candidates.
    """
    records: list[dict | None] = []
    for pick in selected:
        if pick is None:
            records.append(None)
            continue
        target_records = feasible_solutions[pick.target_index]
        match = next(
            (
                row
                for row in target_records
                if int(row["solution_index"]) == int(pick.solution_index)
            ),
            None,
        )
        records.append(match)
    return records


def selected_collision_records_for_chain(
    selected: list,
    feasible_solutions: list[list[dict]],
) -> list[dict] | None:
    """Match a full waypoint chain to collision records.

    Missing records mean the chain is not collision-feasible.
    """
    records = selected_collision_records_for_independent_targets(
        selected, feasible_solutions
    )
    if any(row is None for row in records):
        return None
    return records


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seeds", type=int, default=48)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--stdio", action="store_true")
    args = parser.parse_args()

    core_parent = args.project_root / "08_dual_arm_scene_layout/isaaclab_control"
    sys.path.insert(0, str(core_parent))
    from core.config import (
        IKConfig,
        MapperConfig,
        DEFAULT_INITIAL_RIGHT_ARM_DEG,
        RIGHT_ARM_NAMES,
    )
    from core.ik import CuroboGpuIK, select_waypoint_chain, select_solution
    from core.ik.stage_acceptance import StageAcceptancePolicy, acceptance_mask_from_result
    from core.perception_collision import (
        RGBDFrame, CuroboRGBDMapper, CuroboRobotSphereModel,
    )

    robot_urdf = (
        args.project_root
        / "01_environment/vendor/wuji-description/dual_arm_right_wuji2/urdf/dual_arm_right_wuji2.urdf"
    )
    ik = CuroboGpuIK(
        robot_urdf,
        IKConfig(
            device=args.device,
            num_seeds=args.seeds,
            batch_size=args.batch_size,
            return_seeds=args.seeds,
        ),
    )
    mapper = CuroboRGBDMapper(MapperConfig(device=args.device))
    observed_map = None
    robot_sphere_model = None
    robot_collision_config = (
        args.project_root
        / "08_dual_arm_scene_layout/isaaclab_control/core/generated/dual_arm_right_wuji2_curobo.yml"
    )

    def apply_acceptance_policy(result, payload: dict | None) -> dict:
        if payload is None:
            return {
                "mode": "solver_default_strict",
                "position_tolerance_m": float(ik.config.position_tolerance_m),
                "orientation_tolerance_rad": float(ik.config.orientation_tolerance_rad),
                "minimum_inner_limit_margin_rad": float(
                    ik.config.minimum_inner_limit_margin_rad
                ),
                "require_raw_success": True,
            }
        policy = StageAcceptancePolicy.from_payload(payload)
        result.accepted = acceptance_mask_from_result(result, policy)
        return {"mode": "stage_override", **policy.to_jsonable()}

    def get_robot_sphere_model():
        nonlocal robot_sphere_model
        if robot_sphere_model is None:
            robot_sphere_model = CuroboRobotSphereModel(
                robot_collision_config, device=args.device
            )
        return robot_sphere_model

    def ik_solution_record(result, target_index: int, solution_index: int) -> dict:
        i, k = int(target_index), int(solution_index)
        return {
            "target_index": i,
            "solution_index": k,
            "raw_success": bool(result.raw_success[i, k]),
            "q_rad": result.q_rad[i, k].tolist(),
            "position_error_m": float(result.position_error_m[i, k]),
            "orientation_error_rad": float(result.orientation_error_rad[i, k]),
            "inner_limit_margin_rad": float(result.inner_limit_margin_rad[i, k]),
        }

    def residual_summary_per_target(result, payload: dict | None) -> list[dict]:
        policy = (
            StageAcceptancePolicy.from_payload(payload)
            if payload is not None
            else StageAcceptancePolicy(
                name="solver_default_strict",
                position_tolerance_m=float(ik.config.position_tolerance_m),
                orientation_tolerance_rad=float(ik.config.orientation_tolerance_rad),
                minimum_inner_limit_margin_rad=float(
                    ik.config.minimum_inner_limit_margin_rad
                ),
                require_raw_success=True,
            )
        )
        pos = np.asarray(result.position_error_m, dtype=np.float64)
        rot = np.asarray(result.orientation_error_rad, dtype=np.float64)
        margin = np.asarray(result.inner_limit_margin_rad, dtype=np.float64)
        raw = np.asarray(result.raw_success, dtype=bool)
        q = np.asarray(result.q_rad, dtype=np.float64)
        finite = (
            np.isfinite(q).all(axis=-1)
            & np.isfinite(pos)
            & np.isfinite(rot)
            & np.isfinite(margin)
        )
        rows = []
        for i in range(result.batch_size):
            finite_i = finite[i]
            if np.any(finite_i):
                best_pos = float(np.min(pos[i][finite_i]))
                best_rot = float(np.min(rot[i][finite_i]))
                best_margin = float(np.max(margin[i][finite_i]))
            else:
                best_pos = float("nan")
                best_rot = float("nan")
                best_margin = float("nan")
            pos_ok = bool(np.any(finite_i & (pos[i] <= policy.position_tolerance_m)))
            rot_ok = bool(np.any(finite_i & (rot[i] <= policy.orientation_tolerance_rad)))
            margin_ok = bool(
                np.any(finite_i & (margin[i] >= policy.minimum_inner_limit_margin_rad))
            )
            raw_ok = bool(np.any(raw[i]))
            rows.append(
                {
                    "target_index": int(i),
                    "raw_success_count": int(np.count_nonzero(raw[i])),
                    "finite_seed_count": int(np.count_nonzero(finite_i)),
                    "best_position_error_m": best_pos,
                    "best_orientation_error_rad": best_rot,
                    "best_inner_limit_margin_rad": best_margin,
                    "any_position_ok": pos_ok,
                    "any_orientation_ok": rot_ok,
                    "any_margin_ok": margin_ok,
                    "any_raw_success": raw_ok,
                    "require_raw_success": bool(policy.require_raw_success),
                    "position_fail": not pos_ok,
                    "orientation_fail": not rot_ok,
                    "margin_fail": not margin_ok,
                    "raw_success_fail": bool(policy.require_raw_success) and not raw_ok,
                }
            )
        return rows

    def select_index_chain(result, target_indices: list[int], q_reference: np.ndarray):
        """Select a continuous branch through explicit target indices.

        Unlike ``select_waypoint_chain``, this supports chunked candidate
        screening where one cuRobo solve contains many independent candidate
        waypoint groups in a single flattened batch.
        """
        q_ref = np.asarray(q_reference, dtype=np.float64).reshape(7)
        selected = []
        for target_index in target_indices:
            pick = select_solution(result, int(target_index), q_ref)
            if pick is None:
                return None
            selected.append(pick)
            q_ref = pick.q_rad
        return selected

    def collision_filter_ik(result, context: dict) -> tuple[list[list[dict]], list[list[dict]]]:
        if observed_map is None:
            raise RuntimeError("build_map must be called before collision-aware solve_ik")
        phases = [str(x) for x in context["phases"]]
        if len(phases) != result.batch_size:
            raise ValueError(
                f"collision phases must match IK batch: {len(phases)} != {result.batch_size}"
            )
        states = context.get("joint_positions_by_target")
        if states is None:
            baseline = context["joint_positions_by_name"]
            states = [baseline for _ in range(result.batch_size)]
        if len(states) != result.batch_size:
            raise ValueError(
                "joint_positions_by_target must contain one named state per IK target"
            )
        T_world_base = np.asarray(context["T_world_base"], dtype=np.float64)
        margin_m = float(context.get("margin_m", 0.0))
        model = get_robot_sphere_model()
        required_names = set(model.joint_names)
        for i, state in enumerate(states):
            missing = sorted(required_names - set(state))
            if missing:
                raise KeyError(
                    "production collision state must provide every active joint; "
                    f"target_index={i}, missing={missing}"
                )

        ik_accepted = result.accepted.copy()
        audited: list[list[dict]] = []
        feasible: list[list[dict]] = []
        for i in range(result.batch_size):
            target_audit: list[dict] = []
            target_feasible: list[dict] = []
            for k in np.flatnonzero(ik_accepted[i]):
                named = {str(name): float(value) for name, value in states[i].items()}
                for name, value in zip(RIGHT_ARM_NAMES, result.q_rad[i, k]):
                    named[name] = float(value)
                self_collision = model.check_self_collision(named)
                spheres = model.spheres_from_named_joints(named, T_world_base)
                collision = observed_map.check_spheres(
                    spheres[:, :3], spheres[:, 3], phases[i], margin_m
                )
                scene_count = int(np.count_nonzero(collision["scene_collision"]))
                target_count = int(np.count_nonzero(collision["target_collision"]))
                blocking_count = int(np.count_nonzero(collision["blocking_collision"]))
                unknown_count = int(np.count_nonzero(collision["unknown"]))
                record = ik_solution_record(result, i, int(k))
                record.update({
                    "phase": phases[i],
                    "observed_scene_collision_pass": blocking_count == 0,
                    **self_collision,
                    "unknown_space_exposure": unknown_count > 0,
                    "blocking_collision_sphere_count": blocking_count,
                    "scene_collision_sphere_count": scene_count,
                    "target_collision_sphere_count": target_count,
                    "unknown_sphere_count": unknown_count,
                    "robot_sphere_count": int(len(spheres)),
                })
                target_audit.append(record)
                if blocking_count == 0:
                    target_feasible.append(record)
                else:
                    result.accepted[i, k] = False
            audited.append(target_audit)
            feasible.append(target_feasible)
        return audited, feasible

    def _worst_sphere_records(
        spheres: np.ndarray,
        collision: dict,
        *,
        phase: str,
        top_k: int,
    ) -> list[dict]:
        records = []
        model = get_robot_sphere_model()
        link_names = getattr(model, "sphere_link_names", tuple())
        scene_distance = np.asarray(collision["scene_distance_m"], dtype=np.float64)
        target_distance = (
            None if collision.get("target_distance_m") is None
            else np.asarray(collision["target_distance_m"], dtype=np.float64)
        )
        scene_collision = np.asarray(collision["scene_collision"], dtype=bool)
        target_collision = np.asarray(collision["target_collision"], dtype=bool)
        for layer, mask, distance in (
            ("non_target_scene", scene_collision, scene_distance),
            ("target", target_collision, target_distance),
        ):
            if distance is None:
                continue
            indices = np.flatnonzero(mask)
            if indices.size == 0:
                continue
            severity = np.asarray(spheres[indices, 3], dtype=np.float64) - distance[indices]
            order = indices[np.argsort(-severity)[:top_k]]
            for sphere_index in order:
                i = int(sphere_index)
                records.append({
                    "layer": layer,
                    "phase": phase,
                    "sphere_index": i,
                    "robot_link": link_names[i] if i < len(link_names) else None,
                    "sphere_center_world_m": spheres[i, :3].tolist(),
                    "sphere_radius_m": float(spheres[i, 3]),
                    "esdf_signed_distance_m": float(distance[i]),
                    "penetration_margin_m": float(spheres[i, 3] - distance[i]),
                })
        records.sort(key=lambda x: x["penetration_margin_m"], reverse=True)
        return records[:top_k]

    def diagnose_ik_collisions(req: dict) -> dict:
        if observed_map is None:
            raise RuntimeError("build_map must be called before diagnose_ik_collisions")
        targets = np.asarray(req["targets"], dtype=np.float64)
        q_ref = np.asarray(
            req.get("q_reference_rad", np.deg2rad(DEFAULT_INITIAL_RIGHT_ARM_DEG)),
            dtype=np.float64,
        )
        context = req["collision_context"]
        top_k = int(req.get("top_k", 5))
        result = ik.solve(targets)
        phases = [str(x) for x in context["phases"]]
        states = context.get("joint_positions_by_target")
        if states is None:
            states = [context["joint_positions_by_name"] for _ in range(result.batch_size)]
        T_world_base = np.asarray(context["T_world_base"], dtype=np.float64)
        margin_m = float(context.get("margin_m", 0.0))
        model = get_robot_sphere_model()
        stage_reports = []
        for i in range(result.batch_size):
            raw_indices = set(int(x) for x in np.flatnonzero(result.raw_success[i]))
            threshold_indices = set(int(x) for x in np.flatnonzero(result.accepted[i]))
            rejected_by_self = set()
            rejected_by_scene = set()
            rejected_by_target = set()
            final_survivors = []
            solution_records = []
            worst_records = []
            for k in sorted(threshold_indices):
                named = {str(name): float(value) for name, value in states[i].items()}
                for name, value in zip(RIGHT_ARM_NAMES, result.q_rad[i, k]):
                    named[name] = float(value)
                self_collision = model.check_self_collision(named)
                spheres = model.spheres_from_named_joints(named, T_world_base)
                collision = observed_map.check_spheres(
                    spheres[:, :3], spheres[:, 3], phases[i], margin_m
                )
                scene_count = int(np.count_nonzero(collision["scene_collision"]))
                target_count = int(np.count_nonzero(collision["target_collision"]))
                blocking_count = int(np.count_nonzero(collision["blocking_collision"]))
                causes = []
                if not self_collision["self_collision_pass"]:
                    rejected_by_self.add(k)
                    causes.append("self_collision")
                if scene_count:
                    rejected_by_scene.add(k)
                    causes.append("scene_esdf")
                if target_count:
                    rejected_by_target.add(k)
                    causes.append("target_esdf")
                if not causes and blocking_count == 0:
                    final_survivors.append(k)
                worst = _worst_sphere_records(spheres, collision, phase=phases[i], top_k=top_k)
                worst_records.extend([
                    {**record, "solution_index": k} for record in worst
                ])
                solution_records.append({
                    **ik_solution_record(result, i, k),
                    "rejection_causes": causes,
                    "self_collision": self_collision,
                    "scene_collision_sphere_count": scene_count,
                    "target_collision_sphere_count": target_count,
                    "blocking_collision_sphere_count": blocking_count,
                    "worst_collisions": worst,
                })
            multiple = set()
            for k in threshold_indices:
                count = int(k in rejected_by_self) + int(k in rejected_by_scene) + int(k in rejected_by_target)
                if count >= 2:
                    multiple.add(k)
            worst_records.sort(key=lambda x: x["penetration_margin_m"], reverse=True)
            stage_reports.append({
                "target_index": i,
                "phase": phases[i],
                "raw_ik_solutions": int(len(raw_indices)),
                "threshold_accepted": int(len(threshold_indices)),
                "rejected_by_self_collision": int(len(rejected_by_self)),
                "rejected_by_scene_esdf": int(len(rejected_by_scene)),
                "rejected_by_target_esdf": int(len(rejected_by_target)),
                "rejected_by_multiple_causes": int(len(multiple)),
                "final_surviving_solutions": int(len(final_survivors)),
                "final_survivor_solution_indices": sorted(final_survivors),
                "worst_collisions": worst_records[:top_k],
                "solutions": solution_records,
            })
        return {
            "solve_time_s": result.solve_time_s,
            "q_reference_rad": q_ref.tolist(),
            "stage_reports": stage_reports,
        }

    def coarse_prefilter(req: dict) -> dict:
        targets = np.asarray(req["targets"], dtype=np.float64)
        q_ref = np.asarray(
            req.get("q_reference_rad", np.deg2rad(DEFAULT_INITIAL_RIGHT_ARM_DEG)),
            dtype=np.float64,
        )
        result = ik.solve(targets)
        raw_counts = result.raw_success.sum(axis=1).astype(np.int64)
        threshold_counts = result.accepted.sum(axis=1).astype(np.int64)
        selected = [select_solution(result, i, q_ref) for i in range(result.batch_size)]
        collision_checked = observed_map is not None and "joint_positions_by_name" in req
        coarse_collision_pass = np.zeros(result.batch_size, dtype=bool)
        self_collision_pass = np.zeros(result.batch_size, dtype=bool)
        scene_collision_pass = np.zeros(result.batch_size, dtype=bool)
        if collision_checked:
            model = get_robot_sphere_model()
            baseline = {str(k): float(v) for k, v in req["joint_positions_by_name"].items()}
            T_world_base = np.asarray(req["T_world_base"], dtype=np.float64)
            phase = str(req.get("phase", "pregrasp"))
            margin_m = float(req.get("margin_m", 0.0))
            prefixes = tuple(req.get("arm_link_prefixes") or ["arm_r_link", "arm_base_link"])
            link_names = getattr(model, "sphere_link_names", tuple())
            if link_names:
                arm_mask = np.asarray(
                    [any(str(name).startswith(prefix) for prefix in prefixes) for name in link_names],
                    dtype=bool,
                )
            else:
                arm_mask = None
            for i, pick in enumerate(selected):
                if pick is None:
                    continue
                named = _with_arm_q(baseline, pick.q_rad)
                self_collision = model.check_self_collision(named)
                self_collision_pass[i] = bool(self_collision["self_collision_pass"])
                spheres = model.spheres_from_named_joints(named, T_world_base)
                if arm_mask is not None and len(arm_mask) == len(spheres):
                    spheres = spheres[arm_mask]
                collision = observed_map.check_spheres(
                    spheres[:, :3], spheres[:, 3], phase, margin_m
                )
                # Level-2 is arm-only coarse observed scene filtering.  Do not
                # use target-layer or unretargeted finger collision as a final
                # hand decision here.
                scene_ok = int(np.count_nonzero(collision["scene_collision"])) == 0
                scene_collision_pass[i] = scene_ok
                coarse_collision_pass[i] = bool(scene_ok)
        return {
            "pose_count": int(result.batch_size),
            "raw_success_per_target": raw_counts.tolist(),
            "threshold_accepted_per_target": threshold_counts.tolist(),
            "raw_reachable_indices": np.flatnonzero(raw_counts > 0).astype(np.int64).tolist(),
            "threshold_accepted_indices": np.flatnonzero(threshold_counts > 0).astype(np.int64).tolist(),
            "coarse_collision_checked": bool(collision_checked),
            "self_collision_pass_indices": np.flatnonzero(self_collision_pass).astype(np.int64).tolist(),
            "scene_collision_pass_indices": np.flatnonzero(scene_collision_pass).astype(np.int64).tolist(),
            "coarse_collision_pass_indices": np.flatnonzero(coarse_collision_pass).astype(np.int64).tolist(),
            "selected": [None if x is None else x.to_jsonable() for x in selected],
            "solve_time_s": result.solve_time_s,
        }

    def _arm_sphere_mask(model, prefixes):
        link_names = getattr(model, "sphere_link_names", tuple())
        if not link_names:
            return None
        prefixes = tuple(prefixes or ["arm_r_link", "arm_base_link"])
        return np.asarray(
            [any(str(name).startswith(prefix) for prefix in prefixes) for name in link_names],
            dtype=bool,
        )

    def coarse_approach_prefilter(req: dict) -> dict:
        if observed_map is None:
            raise RuntimeError("build_map must be called before coarse_approach_prefilter")
        pre_targets = np.asarray(req["pregrasp_targets"], dtype=np.float64)
        grasp_targets = np.asarray(req["grasp_targets"], dtype=np.float64)
        if pre_targets.shape != grasp_targets.shape:
            raise ValueError(f"pregrasp/grasp target shape mismatch: {pre_targets.shape} vs {grasp_targets.shape}")
        n = int(len(grasp_targets))
        q_ref = np.asarray(
            req.get("q_reference_rad", np.deg2rad(DEFAULT_INITIAL_RIGHT_ARM_DEG)),
            dtype=np.float64,
        )
        combined = np.concatenate([pre_targets, grasp_targets], axis=0)
        result = ik.solve(combined)
        pre_raw = result.raw_success[:n].sum(axis=1).astype(np.int64)
        grasp_raw = result.raw_success[n:].sum(axis=1).astype(np.int64)
        pre_acc = result.accepted[:n].sum(axis=1).astype(np.int64)
        grasp_acc = result.accepted[n:].sum(axis=1).astype(np.int64)
        model = get_robot_sphere_model()
        baseline = {str(k): float(v) for k, v in req["joint_positions_by_name"].items()}
        T_world_base = np.asarray(req["T_world_base"], dtype=np.float64)
        margin_m = float(req.get("margin_m", 0.0))
        max_step = float(req.get("path_max_joint_step_rad", math.radians(3.0)))
        arm_mask = _arm_sphere_mask(model, req.get("arm_link_prefixes"))
        pre_scene_pass = np.zeros(n, dtype=bool)
        approach_path_pass = np.zeros(n, dtype=bool)
        self_collision_any = np.zeros(n, dtype=bool)
        selected_pre = []
        selected_grasp = []
        for i in range(n):
            pre_pick = select_solution(result, i, q_ref)
            grasp_pick = select_solution(result, n + i, pre_pick.q_rad if pre_pick is not None else q_ref)
            selected_pre.append(None if pre_pick is None else pre_pick.to_jsonable())
            selected_grasp.append(None if grasp_pick is None else grasp_pick.to_jsonable())
            if pre_pick is None or grasp_pick is None:
                continue
            q_nodes = [q_ref, pre_pick.q_rad, grasp_pick.q_rad]
            ok = True
            for seg_index in range(2):
                start_q = np.asarray(q_nodes[seg_index], dtype=np.float64)
                end_q = np.asarray(q_nodes[seg_index + 1], dtype=np.float64)
                sample_count = max(2, int(math.ceil(float(np.max(np.abs(end_q - start_q))) / max_step)) + 1)
                for sample_index in range(sample_count):
                    alpha = sample_index / (sample_count - 1)
                    q = (1.0 - alpha) * start_q + alpha * end_q
                    named = _with_arm_q(baseline, q)
                    self_collision = model.check_self_collision(named)
                    if not self_collision["self_collision_pass"]:
                        self_collision_any[i] = True
                    spheres = model.spheres_from_named_joints(named, T_world_base)
                    if arm_mask is not None and len(arm_mask) == len(spheres):
                        spheres = spheres[arm_mask]
                    phase = "pregrasp" if seg_index == 0 else "approach"
                    collision = observed_map.check_spheres(spheres[:, :3], spheres[:, 3], phase, margin_m)
                    if int(np.count_nonzero(collision["blocking_collision"])) != 0:
                        ok = False
                        break
                if not ok:
                    break
            if ok:
                approach_path_pass[i] = True
            # Endpoint PREGRASP scene check is included in the sampled path, but
            # keep this separate count for report readability.
            named = _with_arm_q(baseline, pre_pick.q_rad)
            spheres = model.spheres_from_named_joints(named, T_world_base)
            if arm_mask is not None and len(arm_mask) == len(spheres):
                spheres = spheres[arm_mask]
            collision = observed_map.check_spheres(spheres[:, :3], spheres[:, 3], "pregrasp", margin_m)
            pre_scene_pass[i] = int(np.count_nonzero(collision["blocking_collision"])) == 0
        both_threshold = (pre_acc > 0) & (grasp_acc > 0)
        survivors = both_threshold & approach_path_pass
        return {
            "candidate_count": n,
            "pregrasp_raw_success_per_target": pre_raw.tolist(),
            "grasp_raw_success_per_target": grasp_raw.tolist(),
            "pregrasp_threshold_accepted_per_target": pre_acc.tolist(),
            "grasp_threshold_accepted_per_target": grasp_acc.tolist(),
            "pregrasp_raw_reachable_indices": np.flatnonzero(pre_raw > 0).astype(np.int64).tolist(),
            "grasp_raw_reachable_indices": np.flatnonzero(grasp_raw > 0).astype(np.int64).tolist(),
            "both_threshold_accepted_indices": np.flatnonzero(both_threshold).astype(np.int64).tolist(),
            "pregrasp_scene_pass_indices": np.flatnonzero(pre_scene_pass).astype(np.int64).tolist(),
            "approach_path_pass_indices": np.flatnonzero(approach_path_pass).astype(np.int64).tolist(),
            "survivor_indices": np.flatnonzero(survivors).astype(np.int64).tolist(),
            "self_collision_reported_indices": np.flatnonzero(self_collision_any).astype(np.int64).tolist(),
            "selected_pregrasp": selected_pre,
            "selected_grasp": selected_grasp,
            "solve_time_s": result.solve_time_s,
            "SELF_COLLISION_POLICY": "REPORT_ONLY_UNRESOLVED",
        }

    def _with_arm_q(state: dict, q_rad: np.ndarray) -> dict:
        named = {str(name): float(value) for name, value in state.items()}
        for name, value in zip(RIGHT_ARM_NAMES, np.asarray(q_rad, dtype=np.float64)):
            named[name] = float(value)
        return named

    def _interpolate_named(a: dict, b: dict, alpha: float, names: tuple[str, ...]) -> dict:
        out = {}
        for name in names:
            out[name] = (1.0 - alpha) * float(a[name]) + alpha * float(b[name])
        return out

    def path_collision_check(
        selected: list,
        q_reference: np.ndarray,
        context: dict,
    ) -> dict:
        if observed_map is None:
            raise RuntimeError("build_map must be called before path checking")
        model = get_robot_sphere_model()
        states = context["joint_positions_by_target"]
        phases = [str(x) for x in context["phases"]]
        baseline = context.get("joint_positions_by_name")
        if baseline is None:
            baseline = states[0]
        T_world_base = np.asarray(context["T_world_base"], dtype=np.float64)
        margin_m = float(context.get("margin_m", 0.0))
        max_step = float(context.get("path_max_joint_step_rad", math.radians(3.0)))
        if max_step <= 0.0:
            raise ValueError("path_max_joint_step_rad must be positive")

        node_q = [np.asarray(q_reference, dtype=np.float64).reshape(7)]
        node_states = [_with_arm_q(baseline, node_q[0])]
        node_labels = ["q_current"]
        node_phase_after = []
        for pick in selected:
            q = np.asarray(pick.q_rad, dtype=np.float64).reshape(7)
            node_q.append(q)
            node_states.append(_with_arm_q(states[pick.target_index], q))
            node_labels.append(f"target_{pick.target_index}")
            node_phase_after.append(phases[pick.target_index])
        include_return_to_reference = bool(context.get("include_return_to_reference", True))
        if include_return_to_reference:
            node_q.append(np.asarray(q_reference, dtype=np.float64).reshape(7))
            node_states.append(_with_arm_q(baseline, node_q[-1]))
            node_labels.append("return_reference")
            if node_phase_after:
                node_phase_after.append(node_phase_after[-1])
            else:
                node_phase_after.append("pregrasp")

        required = set(model.joint_names)
        for index, state in enumerate(node_states):
            missing = sorted(required - set(state))
            if missing:
                raise KeyError(f"path node {index} missing active joints: {missing}")

        segments = []
        path_pass = True
        first_failure = None
        for seg_index in range(len(node_q) - 1):
            start_q = node_q[seg_index]
            end_q = node_q[seg_index + 1]
            max_delta = float(np.max(np.abs(end_q - start_q)))
            sample_count = max(2, int(math.ceil(max_delta / max_step)) + 1)
            phase = node_phase_after[seg_index] if seg_index < len(node_phase_after) else "pregrasp"
            segment = {
                "segment_index": seg_index,
                "from": node_labels[seg_index],
                "to": node_labels[seg_index + 1],
                "phase": phase,
                "max_joint_delta_rad": max_delta,
                "sample_count": sample_count,
                "pass": True,
                "first_failure": None,
            }
            for sample_index in range(sample_count):
                alpha = sample_index / (sample_count - 1)
                named = _interpolate_named(
                    node_states[seg_index],
                    node_states[seg_index + 1],
                    alpha,
                    model.joint_names,
                )
                self_collision = model.check_self_collision(named)
                spheres = model.spheres_from_named_joints(named, T_world_base)
                collision = observed_map.check_spheres(
                    spheres[:, :3], spheres[:, 3], phase, margin_m
                )
                blocking_count = int(np.count_nonzero(collision["blocking_collision"]))
                unknown_count = int(np.count_nonzero(collision["unknown"]))
                if blocking_count != 0:
                    failure = {
                        "sample_index": sample_index,
                        "alpha": alpha,
                        "blocking_collision_sphere_count": blocking_count,
                        "unknown_sphere_count": unknown_count,
                        **self_collision,
                    }
                    segment["pass"] = False
                    segment["first_failure"] = failure
                    if first_failure is None:
                        first_failure = {**failure, "segment_index": seg_index}
                    path_pass = False
                    break
            segments.append(segment)
        return {
            "path_pass": bool(path_pass),
            "path_max_joint_step_rad": max_step,
            "path_segments": segments,
            "path_first_failure": first_failure,
        }

    def generic_joint_path_check(req: dict) -> dict:
        """Validate an explicit arm joint path for regression tests/tools.

        Production candidate selection uses ``path_collision_check`` above after
        grouped IK selection.  This explicit op exists so regression can verify
        continuous-path pass/fail semantics without fabricating an IK target.
        """
        model = get_robot_sphere_model()
        q_nodes = np.asarray(req["q_nodes_rad"], dtype=np.float64)
        if q_nodes.ndim != 2 or q_nodes.shape[1] != 7 or q_nodes.shape[0] < 2:
            raise ValueError(f"q_nodes_rad must be [N>=2,7], got {q_nodes.shape}")
        baseline = {str(k): float(v) for k, v in req["joint_positions_by_name"].items()}
        states = req.get("joint_positions_by_node")
        if states is None:
            states = [baseline for _ in range(len(q_nodes))]
        if len(states) != len(q_nodes):
            raise ValueError("joint_positions_by_node must match q_nodes_rad length")
        node_states = [
            _with_arm_q({str(k): float(v) for k, v in state.items()}, q)
            for state, q in zip(states, q_nodes)
        ]
        required = set(model.joint_names)
        for index, state in enumerate(node_states):
            missing = sorted(required - set(state))
            if missing:
                raise KeyError(f"path node {index} missing active joints: {missing}")
        max_step = float(req.get("path_max_joint_step_rad", math.radians(3.0)))
        if max_step <= 0.0:
            raise ValueError("path_max_joint_step_rad must be positive")
        check_observed = bool(req.get("check_observed_map", False))
        check_self = bool(req.get("check_self_collision", False))
        if check_observed and observed_map is None:
            raise RuntimeError("build_map must be called before observed path checking")
        T_world_base = req.get("T_world_base")
        if T_world_base is not None:
            T_world_base = np.asarray(T_world_base, dtype=np.float64)
        margin_m = float(req.get("margin_m", 0.0))
        phases = req.get("phases") or ["pregrasp"] * (len(q_nodes) - 1)
        if len(phases) != len(q_nodes) - 1:
            raise ValueError("phases must contain one entry per path segment")

        segments = []
        path_pass = True
        first_failure = None
        for seg_index in range(len(q_nodes) - 1):
            start_q = q_nodes[seg_index]
            end_q = q_nodes[seg_index + 1]
            max_delta = float(np.max(np.abs(end_q - start_q)))
            sample_count = max(2, int(math.ceil(max_delta / max_step)) + 1)
            segment = {
                "segment_index": seg_index,
                "phase": str(phases[seg_index]),
                "max_joint_delta_rad": max_delta,
                "sample_count": sample_count,
                "pass": True,
                "first_failure": None,
            }
            for sample_index in range(sample_count):
                alpha = sample_index / (sample_count - 1)
                named = _interpolate_named(
                    node_states[seg_index],
                    node_states[seg_index + 1],
                    alpha,
                    model.joint_names,
                )
                self_collision = model.check_self_collision(named)
                blocking_count = 0
                unknown_count = 0
                if check_observed:
                    spheres = model.spheres_from_named_joints(named, T_world_base)
                    collision = observed_map.check_spheres(
                        spheres[:, :3], spheres[:, 3], str(phases[seg_index]), margin_m
                    )
                    blocking_count = int(np.count_nonzero(collision["blocking_collision"]))
                    unknown_count = int(np.count_nonzero(collision["unknown"]))
                self_blocked = bool(
                    check_self and not bool(self_collision["self_collision_pass"])
                )
                if self_blocked or blocking_count != 0:
                    failure = {
                        "self_collision_blocked": self_blocked,
                        "sample_index": sample_index,
                        "alpha": alpha,
                        "blocking_collision_sphere_count": blocking_count,
                        "unknown_sphere_count": unknown_count,
                        **self_collision,
                    }
                    segment["pass"] = False
                    segment["first_failure"] = failure
                    if first_failure is None:
                        first_failure = {**failure, "segment_index": seg_index}
                    path_pass = False
                    break
            segments.append(segment)
        return {
            "path_pass": bool(path_pass),
            "path_max_joint_step_rad": max_step,
            "path_segments": segments,
            "path_first_failure": first_failure,
            "check_observed_map": check_observed,
            "check_self_collision": check_self,
        }

    if not args.stdio:
        print(f"cuRobo={ik.version}; joints={ik.joint_names}")
        return 0

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
            op = req.get("op")
            if op == "ping":
                emit({"ok": True, "op": "pong", "curobo_version": ik.version, "joint_names": list(ik.joint_names)})
            elif op == "shutdown":
                emit({"ok": True, "op": "shutdown"})
                return 0
            elif op == "solve_ik":
                targets = np.asarray(req["targets"], dtype=np.float64)
                q_ref = np.asarray(req.get("q_reference_rad", np.deg2rad(DEFAULT_INITIAL_RIGHT_ARM_DEG)), dtype=np.float64)
                result = ik.solve(targets)
                acceptance_policy = apply_acceptance_policy(
                    result, req.get("acceptance_policy")
                )
                ik_accepted_per_target = result.accepted.sum(axis=1).tolist()
                collision_context = req.get("collision_context")
                if collision_context is None:
                    ik_accepted_solutions = [
                        [ik_solution_record(result, i, int(k)) for k in np.flatnonzero(result.accepted[i])]
                        for i in range(result.batch_size)
                    ]
                    feasible_solutions = ik_accepted_solutions
                else:
                    ik_accepted_solutions, feasible_solutions = collision_filter_ik(
                        result, collision_context
                    )
                if bool(req.get("select_chain", True)):
                    selected = select_waypoint_chain(result, q_ref)
                else:
                    selected = [select_solution(result, i, q_ref) for i in range(result.batch_size)]
                selected_collision = None
                path_report = None
                select_chain = bool(req.get("select_chain", True))
                if selected is not None and collision_context is not None:
                    if select_chain:
                        selected_collision = selected_collision_records_for_chain(
                            selected, feasible_solutions
                        )
                        if selected_collision is not None:
                            path_report = path_collision_check(
                                selected, q_ref, collision_context
                            )
                    else:
                        selected_collision = selected_collision_records_for_independent_targets(
                            selected, feasible_solutions
                        )
                        # Independent Exact-COVER targets are separate
                        # candidates, not one waypoint chain.
                        path_report = None
                self_collision_pass = (
                    None if selected_collision is None
                    else bool(
                        all(
                            x is not None and x["self_collision_pass"]
                            for x in selected_collision
                        )
                    )
                )
                emit({
                    "ok": True,
                    "op": "solve_ik",
                    "acceptance_policy": acceptance_policy,
                    "accepted_per_target": result.accepted.sum(axis=1).tolist(),
                    "ik_accepted_per_target": ik_accepted_per_target,
                    "raw_success_per_target": result.raw_success.sum(axis=1).tolist(),
                    "residual_summary_per_target": residual_summary_per_target(
                        result, req.get("acceptance_policy")
                    ),
                    "ik_accepted_solutions": ik_accepted_solutions,
                    "feasible_solutions": feasible_solutions,
                    "selected": None if selected is None else [None if x is None else x.to_jsonable() for x in selected],
                    "selected_collision": selected_collision,
                    "ik_pass": bool(all(int(x) > 0 for x in ik_accepted_per_target)),
                    "observed_scene_collision_pass": (
                        None if collision_context is None
                        else (
                            bool(
                                all(
                                    x is not None and x["observed_scene_collision_pass"]
                                    for x in selected_collision
                                )
                            )
                            if selected_collision is not None
                            else False
                        )
                    ),
                    "self_collision_pass": self_collision_pass,
                    "path_pass": None if path_report is None else path_report["path_pass"],
                    "path_segments": None if path_report is None else path_report["path_segments"],
                    "path_first_failure": None if path_report is None else path_report["path_first_failure"],
                    "unknown_space_exposure": (
                        None if selected_collision is None
                        else [
                            None if x is None else bool(x["unknown_space_exposure"])
                            for x in selected_collision
                        ]
                    ),
                    "solve_time_s": result.solve_time_s,
                })
            elif op == "solve_ik_groups":
                targets = np.asarray(req["targets"], dtype=np.float64)
                group_sizes = [int(x) for x in req["group_sizes"]]
                if any(x <= 0 for x in group_sizes):
                    raise ValueError(f"group_sizes must be positive: {group_sizes}")
                if int(np.sum(group_sizes)) != int(len(targets)):
                    raise ValueError(
                        f"group_sizes sum {int(np.sum(group_sizes))} != target count {int(len(targets))}"
                    )
                q_ref = np.asarray(
                    req.get("q_reference_rad", np.deg2rad(DEFAULT_INITIAL_RIGHT_ARM_DEG)),
                    dtype=np.float64,
                )
                result = ik.solve(targets)
                acceptance_policy = apply_acceptance_policy(
                    result, req.get("acceptance_policy")
                )
                ik_accepted_per_target = result.accepted.sum(axis=1).tolist()
                collision_context = req.get("collision_context")
                if collision_context is None:
                    ik_accepted_solutions = [
                        [ik_solution_record(result, i, int(k)) for k in np.flatnonzero(result.accepted[i])]
                        for i in range(result.batch_size)
                    ]
                    feasible_solutions = ik_accepted_solutions
                else:
                    ik_accepted_solutions, feasible_solutions = collision_filter_ik(
                        result, collision_context
                    )
                groups = []
                offset = 0
                for group_index, group_size in enumerate(group_sizes):
                    indices = list(range(offset, offset + group_size))
                    selected = (
                        select_index_chain(result, indices, q_ref)
                        if bool(req.get("select_chain", True))
                        else [select_solution(result, i, q_ref) for i in indices]
                    )
                    selected_collision = None
                    path_report = None
                    group_pass = selected is not None and all(x is not None for x in selected)
                    if group_pass and collision_context is not None:
                        selected_collision = []
                        for pick in selected:
                            match = next(
                                x for x in feasible_solutions[pick.target_index]
                                if x["solution_index"] == pick.solution_index
                            )
                            selected_collision.append(match)
                        path_report = path_collision_check(selected, q_ref, collision_context)
                        group_pass = bool(
                            path_report["path_pass"]
                            and all(x["observed_scene_collision_pass"] for x in selected_collision)
                        )
                    groups.append({
                        "group_index": group_index,
                        "target_indices": indices,
                        "raw_success_per_target": result.raw_success[indices].sum(axis=1).tolist(),
                        "accepted_per_target": result.accepted[indices].sum(axis=1).tolist(),
                        "ik_accepted_per_target": [ik_accepted_per_target[i] for i in indices],
                        "selected": None if selected is None else [
                            None if x is None else x.to_jsonable() for x in selected
                        ],
                        "selected_collision": selected_collision,
                        "ik_pass": bool(all(int(ik_accepted_per_target[i]) > 0 for i in indices)),
                        "observed_scene_collision_pass": (
                            None if selected_collision is None
                            else bool(all(x["observed_scene_collision_pass"] for x in selected_collision))
                        ),
                        "self_collision_pass": (
                            None if selected_collision is None
                            else bool(all(x["self_collision_pass"] for x in selected_collision))
                        ),
                        "path_pass": None if path_report is None else path_report["path_pass"],
                        "path_segments": None if path_report is None else path_report["path_segments"],
                        "path_first_failure": None if path_report is None else path_report["path_first_failure"],
                        "unknown_space_exposure": (
                            None if selected_collision is None
                            else [bool(x["unknown_space_exposure"]) for x in selected_collision]
                        ),
                        "pass": bool(group_pass),
                    })
                    offset += group_size
                emit({
                    "ok": True,
                    "op": "solve_ik_groups",
                    "acceptance_policy": acceptance_policy,
                    "group_count": len(groups),
                    "pose_count": int(len(targets)),
                    "group_sizes": group_sizes,
                    "raw_success_per_target": result.raw_success.sum(axis=1).tolist(),
                    "accepted_per_target": result.accepted.sum(axis=1).tolist(),
                    "ik_accepted_per_target": ik_accepted_per_target,
                    "ik_accepted_solutions": ik_accepted_solutions,
                    "feasible_solutions": feasible_solutions,
                    "groups": groups,
                    "solve_time_s": result.solve_time_s,
                })
            elif op == "diagnose_ik_collisions":
                emit({"ok": True, "op": "diagnose_ik_collisions", **jsonable(diagnose_ik_collisions(req))})
            elif op == "coarse_prefilter":
                emit({"ok": True, "op": "coarse_prefilter", **jsonable(coarse_prefilter(req))})
            elif op == "coarse_approach_prefilter":
                emit({"ok": True, "op": "coarse_approach_prefilter", **jsonable(coarse_approach_prefilter(req))})
            elif op == "build_map":
                frame = RGBDFrame.from_npy(
                    req["depth_path"],
                    req["intrinsics_path"],
                    req["T_world_camera_path"],
                    req.get("target_mask_path"),
                )
                observed_map = mapper.build(frame)
                emit({
                    "ok": True,
                    "op": "build_map",
                    "map_id": observed_map.map_id,
                    "grid_center_world": observed_map.grid_center_world.tolist(),
                    "extent_meters_xyz": observed_map.extent_meters_xyz.tolist(),
                    "has_target_layer": observed_map.target_grid is not None,
                })
            elif op == "query_spheres":
                if observed_map is None:
                    raise RuntimeError("build_map must be called before query_spheres")
                result = observed_map.check_spheres(
                    np.asarray(req["centers_world"], dtype=np.float64),
                    np.asarray(req["radii_m"], dtype=np.float64),
                    req["phase"],
                    float(req.get("margin_m", 0.0)),
                )
                emit({"ok": True, "op": "query_spheres", "result": jsonable(result)})
            elif op == "robot_spheres":
                model = get_robot_sphere_model()
                T_world_base = req.get("T_world_base")
                spheres = model.spheres_from_named_joints(
                    {str(k): float(v) for k, v in req["joint_positions_by_name"].items()},
                    None if T_world_base is None else np.asarray(T_world_base, dtype=np.float64),
                )
                emit({
                    "ok": True,
                    "op": "robot_spheres",
                    "joint_names": list(model.joint_names),
                    "default_joint_positions_by_name": {
                        name: float(value)
                        for name, value in zip(
                            model.joint_names,
                            model.robot.default_joint_position.detach().cpu().numpy().reshape(-1),
                        )
                    },
                    "sphere_count": int(len(spheres)),
                    "spheres_world_xyzw_radius": spheres.tolist(),
                })
            elif op == "check_self_collision":
                model = get_robot_sphere_model()
                result = model.check_self_collision(
                    {str(k): float(v) for k, v in req["joint_positions_by_name"].items()}
                )
                emit({"ok": True, "op": "check_self_collision", **jsonable(result)})
            elif op == "check_joint_path":
                emit({"ok": True, "op": "check_joint_path", **jsonable(generic_joint_path_check(req))})
            elif op == "check_robot_state":
                if observed_map is None:
                    raise RuntimeError("build_map must be called before check_robot_state")
                model = get_robot_sphere_model()
                spheres = model.spheres_from_named_joints(
                    {str(k): float(v) for k, v in req["joint_positions_by_name"].items()},
                    np.asarray(req["T_world_base"], dtype=np.float64),
                )
                result = observed_map.check_spheres(
                    spheres[:, :3],
                    spheres[:, 3],
                    req["phase"],
                    float(req.get("margin_m", 0.0)),
                )
                emit({
                    "ok": True,
                    "op": "check_robot_state",
                    "sphere_count": int(len(spheres)),
                    "result": jsonable(result),
                })
            else:
                raise ValueError(f"unknown worker op: {op}")
        except Exception as exc:
            emit({
                "ok": False,
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(limit=8),
            })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
