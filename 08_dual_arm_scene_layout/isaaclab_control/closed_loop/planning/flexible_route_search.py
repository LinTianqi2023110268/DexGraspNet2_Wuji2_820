#!/usr/bin/env python3
"""Flexible GPU-IK search around one exact Wuji2 COVER pose.

The planner intentionally separates two concepts:

* IK accuracy stays strict (the existing cuRobo 5 mm / 5 deg / joint-margin
  contract is untouched).
* Non-contact task goals are sets of legal 6D poses rather than one fixed pose.

COVER is the only exact arm pose used as the grasp-root hard gate.  GRASP and
SQUEEZE reuse the same arm q7 while only the Wuji2 q20 changes.  PREGRASP,
LIFT, TRANSFER, PLACE and RETREAT are sampled task regions and solved in GPU
batches.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np

from .flexible_pose_sampling import (
    PoseSampleSet,
    free_placement_centres_xy,
    placement_zone_bounds,
    sample_lift,
    sample_place_from_centres,
    sample_pregrasp,
    sample_retreat,
    sample_transfer,
)


ROUTE_STAGES = [
    "pregrasp", "cover", "grasp", "squeeze", "lift",
    "transfer", "place", "release", "retreat",
]

HAND_FOR_STAGE = {
    "pregrasp": "pregrasp",
    "cover": "cover",
    "grasp": "grasp",
    "squeeze": "squeeze",
    "lift": "squeeze",
    "transfer": "squeeze",
    "place": "squeeze",
    "release": "pregrasp",
    "retreat": "pregrasp",
}

PHASE_FOR_STAGE = {
    "pregrasp": "pregrasp",
    "cover": "cover",
    "grasp": "grasp",
    "squeeze": "squeeze",
    "lift": "lift",
    "transfer": "lift",
    "place": "lift",
    "release": "lift",
    "retreat": "lift",
}


@dataclass
class IKNode:
    stage: str
    q_rad: np.ndarray
    target_index: int
    solution_index: int
    target_pose_world: np.ndarray
    metadata: dict
    inner_limit_margin_rad: float
    intrinsic_penalty: float


@dataclass
class BeamState:
    node: IKNode
    cost: float
    parent: "BeamState | None"

    @property
    def q_rad(self) -> np.ndarray:
        return self.node.q_rad


def load_json(path: Path) -> dict:
    return json.loads(Path(path).resolve().read_text(encoding="utf-8"))


def _world_from_base(project_root: Path) -> np.ndarray:
    layout = load_json(project_root / "08_dual_arm_scene_layout/config/manual_layout_calibrated.json")
    return np.asarray(
        layout["transforms"]["dual_arm_mount"]["Gf_local_to_world_row_major"],
        dtype=np.float64,
    ).T


def _candidate_geometry(case_root: Path) -> dict:
    case_root = Path(case_root).resolve()
    arm_path = case_root / "07_arm_execution/arm_flange_targets.npz"
    hand_path = case_root / "06_isaacsim/final_waypoints.npz"
    case_meta = load_json(case_root / "case.json")
    manifests = sorted((case_root / "01_input").glob("scene_*_manifest.json"))
    if len(manifests) != 1:
        raise RuntimeError(f"{case_root}: expected one scene manifest, got {manifests}")
    scene = load_json(manifests[0])
    target_id = int(case_meta["target_segmentation_id"])
    target_record = next(
        row for row in scene["objects"] if int(row["segmentation_id"]) == target_id
    )

    with np.load(arm_path, allow_pickle=False) as z:
        names = [str(x) for x in z["waypoint_names"].tolist()]
        flange_world = np.asarray(z["world_from_right_flange"], dtype=np.float64)
        wrist_world = np.asarray(z["world_from_wuji2_wrist"], dtype=np.float64)
        flange_from_wrist = np.asarray(z["flange_from_wuji2_wrist"], dtype=np.float64)
        world_from_source = np.asarray(z["world_from_source_zone"], dtype=np.float64)
    expected = ["pregrasp", "cover", "grasp", "squeeze", "lift"]
    if names != expected:
        raise RuntimeError(f"{case_root}: unexpected pick stages {names}")

    with np.load(hand_path, allow_pickle=False) as z:
        hand_names = [str(x) for x in z["finger_joint_names"].tolist()]
        hand_stage_names = [str(x) for x in z["waypoint_names"].tolist()]
        hand_q = np.asarray(z["waypoint_joint_positions"][0], dtype=np.float64)
        if "wuji2_semantic_palm_approach_axis_source" in z.files:
            approach_axis = (
                world_from_source[:3, :3]
                @ np.asarray(z["wuji2_semantic_palm_approach_axis_source"], dtype=np.float64)
            )
        else:
            approach_axis = np.asarray(z["wuji2_semantic_palm_approach_axis_world"], dtype=np.float64)
        is_top = bool(np.asarray(z["is_top_grasp"]).item())
    hand_index = {name: i for i, name in enumerate(hand_stage_names)}

    object_world_initial = world_from_source @ np.asarray(
        target_record["pose_world_object"], dtype=np.float64
    )
    cover_index = names.index("cover")
    grasp_index = names.index("grasp")
    cover_flange = flange_world[cover_index]
    cover_wrist = wrist_world[cover_index]
    grasp_flange = flange_world[grasp_index]
    flange_from_object_grasp = np.linalg.inv(grasp_flange) @ object_world_initial

    return {
        "case_root": case_root,
        "target_segmentation_id": target_id,
        "target_object_code": str(case_meta.get("target_object_code", "")),
        "source_candidate_index": int(case_meta["source_candidate_index"]),
        "official_score": float(case_meta.get("official_score", float("nan"))),
        "flange_from_wrist": flange_from_wrist,
        "cover_flange_world": cover_flange,
        "cover_wrist_world": cover_wrist,
        "nominal_lift_wrist_world": wrist_world[names.index("lift")],
        "approach_axis_world": approach_axis / np.linalg.norm(approach_axis),
        "is_top_grasp": is_top,
        "object_world_initial": object_world_initial,
        "flange_from_object_grasp": flange_from_object_grasp,
        "hand_names": hand_names,
        "hand_q": hand_q,
        "hand_index": hand_index,
    }


def _named_state(geometry: dict, measured: dict, stage: str) -> dict:
    named = {str(k): float(v) for k, v in measured.items()}
    hand_stage = HAND_FOR_STAGE[stage]
    q20 = geometry["hand_q"][geometry["hand_index"][hand_stage]]
    for name, value in zip(geometry["hand_names"], q20):
        named[name] = float(value)
    return named


def _filter_unknown(records: Iterable[dict], *, block_unknown: bool) -> list[dict]:
    rows = list(records)
    if not block_unknown:
        return rows
    return [row for row in rows if not bool(row.get("unknown_space_exposure", False))]


def _solution_pool_for_target(report: dict, target_index: int, *, collision_enabled: bool, block_unknown: bool) -> list[dict]:
    key = "feasible_solutions" if collision_enabled else "ik_accepted_solutions"
    rows = report.get(key) or []
    if target_index >= len(rows):
        return []
    return _filter_unknown(rows[target_index], block_unknown=block_unknown)


def _exact_cover_candidate_subfunnel(
    report: dict,
    target_index: int,
    *,
    collision_enabled: bool,
    block_unknown: bool,
    diagnostic_disable_cover_esdf: bool,
    cover_collision_bypass_reason: str | None,
    final_solution_count: int,
) -> dict:
    raw_counts = report.get("raw_success_per_target") or []
    raw_count = int(raw_counts[target_index]) if target_index < len(raw_counts) else 0
    strict_all = report.get("ik_accepted_solutions") or []
    feasible_all = report.get("feasible_solutions") or []
    strict_rows = list(strict_all[target_index]) if target_index < len(strict_all) else []
    feasible_rows = list(feasible_all[target_index]) if target_index < len(feasible_all) else []
    audited_rows = strict_rows if collision_enabled else []
    scene_reject = 0
    target_reject = 0
    blocking_reject = 0
    unknown_exposure = 0
    for row in audited_rows:
        scene_reject += int(int(row.get("scene_collision_sphere_count", 0)) > 0)
        target_reject += int(int(row.get("target_collision_sphere_count", 0)) > 0)
        blocking_reject += int(int(row.get("blocking_collision_sphere_count", 0)) > 0)
        unknown_exposure += int(bool(row.get("unknown_space_exposure", False)))
    return {
        "raw_success_solution_count": raw_count,
        "raw_success_target_pass": bool(raw_count > 0),
        "strict_ik_accepted_solution_count": int(len(strict_rows)),
        "strict_ik_target_pass": bool(len(strict_rows) > 0),
        "cover_esdf_bypassed": bool(diagnostic_disable_cover_esdf or cover_collision_bypass_reason),
        "cover_collision_bypass_reason": cover_collision_bypass_reason,
        "collision_audited_solution_count": int(len(audited_rows)),
        "scene_collision_rejected_solution_count": int(scene_reject),
        "target_collision_rejected_solution_count": int(target_reject),
        "blocking_collision_rejected_solution_count": int(blocking_reject),
        "unknown_exposure_solution_count": int(unknown_exposure),
        "feasible_solution_count": int(len(feasible_rows)),
        "block_unknown": bool(block_unknown),
        "final_exact_cover_solution_count": int(final_solution_count),
        "final_exact_cover_pass": bool(final_solution_count > 0),
    }


def summarize_exact_cover_subfunnel(rows: list[dict]) -> dict:
    sub = [row.get("exact_cover_subfunnel", {}) for row in rows]
    total = int(len(rows))
    raw_targets = sum(1 for row in sub if row.get("raw_success_target_pass"))
    strict_targets = sum(1 for row in sub if row.get("strict_ik_target_pass"))
    feasible_targets = sum(1 for row in sub if int(row.get("feasible_solution_count", 0)) > 0)
    final_targets = sum(1 for row in sub if row.get("final_exact_cover_pass"))
    raw_solutions = sum(int(row.get("raw_success_solution_count", 0)) for row in sub)
    strict_solutions = sum(int(row.get("strict_ik_accepted_solution_count", 0)) for row in sub)
    audited_solutions = sum(int(row.get("collision_audited_solution_count", 0)) for row in sub)
    scene_rejected = sum(int(row.get("scene_collision_rejected_solution_count", 0)) for row in sub)
    target_rejected = sum(int(row.get("target_collision_rejected_solution_count", 0)) for row in sub)
    blocking_rejected = sum(int(row.get("blocking_collision_rejected_solution_count", 0)) for row in sub)
    unknown_exposure = sum(int(row.get("unknown_exposure_solution_count", 0)) for row in sub)
    feasible_solutions = sum(int(row.get("feasible_solution_count", 0)) for row in sub)
    cover_esdf_bypassed = any(bool(row.get("cover_esdf_bypassed", False)) for row in sub)
    bypass_reasons = sorted({
        str(row.get("cover_collision_bypass_reason"))
        for row in sub
        if row.get("cover_collision_bypass_reason")
    })
    return {
        "input_candidates": total,
        "cover_esdf_bypassed": bool(cover_esdf_bypassed),
        "cover_collision_bypass_reason": bypass_reasons[0] if len(bypass_reasons) == 1 else (
            bypass_reasons if bypass_reasons else None
        ),
        "raw_curobo_reachable_targets": int(raw_targets),
        "strict_ik_targets": int(strict_targets),
        "post_collision_targets": int(feasible_targets),
        "final_exact_cover_pass_targets": int(final_targets),
        "raw_success_solution_count": int(raw_solutions),
        "strict_ik_accepted_solution_count": int(strict_solutions),
        "collision_audited_solution_count": int(audited_solutions),
        "scene_collision_rejected_solution_count": int(scene_rejected),
        "target_collision_rejected_solution_count": int(target_rejected),
        "blocking_collision_rejected_solution_count": int(blocking_rejected),
        "unknown_exposure_solution_count": int(unknown_exposure),
        "collision_rejected_solution_count": int(blocking_rejected),
        "feasible_solution_count": int(feasible_solutions),
    }


def _cap_solutions(records: list[dict], q_reference: np.ndarray, limit: int) -> list[dict]:
    q_ref = np.asarray(q_reference, dtype=np.float64).reshape(7)
    ordered = sorted(
        records,
        key=lambda row: (
            float(np.linalg.norm(np.asarray(row["q_rad"], dtype=np.float64) - q_ref)),
            -float(row.get("inner_limit_margin_rad", 0.0)),
        ),
    )
    return ordered[: max(1, int(limit))]


def _solve_pose_set(
    *,
    client,
    stage: str,
    pose_set: PoseSampleSet,
    q_reference: np.ndarray,
    measured: dict,
    geometry: dict,
    T_base_from_world: np.ndarray,
    T_world_base: np.ndarray,
    no_planner_collision_check: bool,
    block_unknown: bool,
    solutions_per_pose: int,
) -> tuple[list[IKNode], dict]:
    poses_world = np.asarray(pose_set.poses_world, dtype=np.float64)
    targets_base = np.stack([T_base_from_world @ pose for pose in poses_world])
    collision_context = None
    if not no_planner_collision_check:
        state = _named_state(geometry, measured, stage)
        collision_context = {
            "phases": [PHASE_FOR_STAGE[stage]] * len(poses_world),
            "joint_positions_by_name": measured,
            "joint_positions_by_target": [state] * len(poses_world),
            "T_world_base": T_world_base,
            "margin_m": 0.0,
            "include_return_to_reference": False,
        }
    report = client.solve_ik(
        targets_base,
        q_reference,
        select_chain=False,
        collision_context=collision_context,
    )
    nodes: list[IKNode] = []
    for target_index, pose in enumerate(poses_world):
        records = _solution_pool_for_target(
            report,
            target_index,
            collision_enabled=collision_context is not None,
            block_unknown=block_unknown,
        )
        for row in _cap_solutions(records, q_reference, solutions_per_pose):
            nodes.append(IKNode(
                stage=stage,
                q_rad=np.asarray(row["q_rad"], dtype=np.float64),
                target_index=target_index,
                solution_index=int(row["solution_index"]),
                target_pose_world=pose.copy(),
                metadata=dict(pose_set.metadata[target_index]),
                inner_limit_margin_rad=float(row.get("inner_limit_margin_rad", 0.0)),
                intrinsic_penalty=float(pose_set.metadata[target_index].get("nominal_penalty", 0.0)),
            ))
    summary = {
        "stage": stage,
        "target_count": int(len(poses_world)),
        "reachable_target_count": int(sum(1 for i in range(len(poses_world)) if _solution_pool_for_target(
            report, i, collision_enabled=collision_context is not None, block_unknown=block_unknown
        ))),
        "node_count": int(len(nodes)),
        "worker_solve_time_s": float(report.get("solve_time_s", 0.0)),
        "planner_collision_check": not no_planner_collision_check,
    }
    return nodes, summary


def _node_from_record(stage: str, pose_world: np.ndarray, record: dict, metadata: dict | None = None) -> IKNode:
    return IKNode(
        stage=stage,
        q_rad=np.asarray(record["q_rad"], dtype=np.float64),
        target_index=int(record.get("target_index", 0)),
        solution_index=int(record["solution_index"]),
        target_pose_world=np.asarray(pose_world, dtype=np.float64).copy(),
        metadata={} if metadata is None else dict(metadata),
        inner_limit_margin_rad=float(record.get("inner_limit_margin_rad", 0.0)),
        intrinsic_penalty=float((metadata or {}).get("nominal_penalty", 0.0)),
    )


def _transition_cost(q_from: np.ndarray, q_to: np.ndarray, cfg: dict) -> float:
    delta = np.asarray(q_to, dtype=np.float64) - np.asarray(q_from, dtype=np.float64)
    return (
        float(cfg.get("joint_l2_weight", 1.0)) * float(np.linalg.norm(delta))
        + float(cfg.get("joint_max_weight", 0.25)) * float(np.max(np.abs(delta)))
    )


def _node_extra_cost(node: IKNode, cfg: dict) -> float:
    return (
        float(cfg.get("nominal_penalty_weight", 0.05)) * float(node.intrinsic_penalty)
        - float(cfg.get("joint_margin_reward", 0.02)) * float(node.inner_limit_margin_rad)
    )


def _expand_beam(
    parents: list[BeamState], nodes: list[IKNode], *, beam_width: int, selection_cfg: dict,
) -> list[BeamState]:
    if not parents or not nodes:
        return []
    parent_q = np.stack([state.q_rad for state in parents])
    node_q = np.stack([node.q_rad for node in nodes])
    # [P,N,7]
    delta = node_q[None, :, :] - parent_q[:, None, :]
    l2 = np.linalg.norm(delta, axis=2)
    mx = np.max(np.abs(delta), axis=2)
    parent_cost = np.asarray([state.cost for state in parents], dtype=np.float64)[:, None]
    node_extra = np.asarray([_node_extra_cost(node, selection_cfg) for node in nodes], dtype=np.float64)[None, :]
    total = (
        parent_cost
        + float(selection_cfg.get("joint_l2_weight", 1.0)) * l2
        + float(selection_cfg.get("joint_max_weight", 0.25)) * mx
        + node_extra
    )
    best_parent_for_node = np.argmin(total, axis=0)
    best_cost_for_node = total[best_parent_for_node, np.arange(len(nodes))]
    order = np.argsort(best_cost_for_node)[: max(1, int(beam_width))]
    return [
        BeamState(
            node=nodes[int(node_index)],
            cost=float(best_cost_for_node[int(node_index)]),
            parent=parents[int(best_parent_for_node[int(node_index)])],
        )
        for node_index in order
    ]


def _home_parent(q_current: np.ndarray) -> BeamState:
    node = IKNode(
        stage="q_current",
        q_rad=np.asarray(q_current, dtype=np.float64).copy(),
        target_index=-1,
        solution_index=-1,
        target_pose_world=np.eye(4, dtype=np.float64),
        metadata={},
        inner_limit_margin_rad=0.0,
        intrinsic_penalty=0.0,
    )
    return BeamState(node=node, cost=0.0, parent=None)


def _ancestry(state: BeamState) -> list[BeamState]:
    chain = []
    cursor: BeamState | None = state
    while cursor is not None:
        chain.append(cursor)
        cursor = cursor.parent
    return list(reversed(chain))


def read_occupied_centres(registry_path: Path) -> list[np.ndarray]:
    path = Path(registry_path)
    if not path.is_file():
        return []
    registry = load_json(path)
    centres = []
    for row in registry.get("placements", []):
        if "center_world_xy_m" in row:
            centres.append(np.asarray(row["center_world_xy_m"], dtype=np.float64))
        elif "object_root_world_m" in row:
            centres.append(np.asarray(row["object_root_world_m"][:2], dtype=np.float64))
        elif "footprint_world_xy_min_m" in row and "footprint_world_xy_max_m" in row:
            lower = np.asarray(row["footprint_world_xy_min_m"], dtype=np.float64)
            upper = np.asarray(row["footprint_world_xy_max_m"], dtype=np.float64)
            centres.append(0.5 * (lower + upper))
    return centres


def screen_exact_cover_batch(
    *,
    client,
    case_roots: list[Path],
    q_current: np.ndarray,
    measured: dict,
    T_base_from_world: np.ndarray,
    T_world_base: np.ndarray,
    no_planner_collision_check: bool,
    block_unknown: bool,
    solutions_per_candidate: int,
    diagnostic_disable_cover_esdf: bool = False,
    cover_collision_bypass_reason: str | None = None,
) -> list[dict]:
    """One batched hard gate: exact Wuji2 COVER pose only."""
    geometries = [_candidate_geometry(Path(root)) for root in case_roots]
    cover_world = np.stack([geometry["cover_flange_world"] for geometry in geometries])
    cover_base = np.stack([T_base_from_world @ pose for pose in cover_world])
    collision_context = None
    if not no_planner_collision_check and not diagnostic_disable_cover_esdf:
        states = [_named_state(geometry, measured, "cover") for geometry in geometries]
        collision_context = {
            "phases": ["cover"] * len(geometries),
            "joint_positions_by_name": measured,
            "joint_positions_by_target": states,
            "T_world_base": T_world_base,
            "margin_m": 0.0,
            "include_return_to_reference": False,
        }
    report = client.solve_ik(
        cover_base,
        q_current,
        select_chain=False,
        collision_context=collision_context,
    )
    rows = []
    for index, (case_root, geometry) in enumerate(zip(case_roots, geometries)):
        records = _solution_pool_for_target(
            report,
            index,
            collision_enabled=collision_context is not None,
            block_unknown=block_unknown,
        )
        records = _cap_solutions(records, q_current, solutions_per_candidate)
        subfunnel = _exact_cover_candidate_subfunnel(
            report,
            index,
            collision_enabled=collision_context is not None,
            block_unknown=block_unknown,
            diagnostic_disable_cover_esdf=bool(diagnostic_disable_cover_esdf),
            cover_collision_bypass_reason=cover_collision_bypass_reason,
            final_solution_count=len(records),
        )
        rows.append({
            "case_root": str(Path(case_root).resolve()),
            "candidate_index": geometry["source_candidate_index"],
            "official_score": geometry["official_score"],
            "pass": bool(records),
            "solution_count": len(records),
            "diagnostic_cover_esdf_bypassed": bool(diagnostic_disable_cover_esdf),
            "cover_collision_bypass_reason": cover_collision_bypass_reason,
            "exact_cover_subfunnel": subfunnel,
            "cover_solutions": records,
        })
    return rows


def _write_plan(
    *,
    geometry: dict,
    q_current: np.ndarray,
    chosen: dict[str, BeamState],
    output_npz: Path,
    summaries: list[dict],
    final_path_report: dict | None,
    placement_registry: Path,
) -> dict:
    cover_state = chosen["cover"]
    place_state = chosen["place"]
    arm_q = np.stack([
        chosen["pregrasp"].q_rad,
        cover_state.q_rad,
        cover_state.q_rad,
        cover_state.q_rad,
        chosen["lift"].q_rad,
        chosen["transfer"].q_rad,
        place_state.q_rad,
        place_state.q_rad,
        chosen["retreat"].q_rad,
    ]).astype(np.float64)
    flange_world = np.stack([
        chosen["pregrasp"].node.target_pose_world,
        cover_state.node.target_pose_world,
        cover_state.node.target_pose_world,
        cover_state.node.target_pose_world,
        chosen["lift"].node.target_pose_world,
        chosen["transfer"].node.target_pose_world,
        place_state.node.target_pose_world,
        place_state.node.target_pose_world,
        chosen["retreat"].node.target_pose_world,
    ]).astype(np.float64)
    flange_from_wrist = geometry["flange_from_wrist"]
    wrist_world = flange_world @ flange_from_wrist[None]
    placement_meta = dict(place_state.node.metadata)
    metadata = {
        "schema_version": 1,
        "status": "PASS",
        "planner": "flexible_task_set_gpu_ik",
        "candidate_index": geometry["source_candidate_index"],
        "official_score": geometry["official_score"],
        "target_segmentation_id": geometry["target_segmentation_id"],
        "target_object_code": geometry["target_object_code"],
        "route_stages": ROUTE_STAGES,
        "cover_is_exact": True,
        "grasp_squeeze_reuse_cover_q": True,
        "release_reuses_place_q": True,
        "q_current_rad": np.asarray(q_current, dtype=np.float64).tolist(),
        "placement": placement_meta,
        "placement_registry": str(Path(placement_registry).resolve()),
        "stage_summaries": summaries,
        "final_path_report": final_path_report,
    }
    output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_npz,
        waypoint_names=np.asarray(ROUTE_STAGES),
        arm_q_rad=arm_q,
        world_from_right_flange=flange_world,
        world_from_wuji2_wrist=wrist_world,
        q_current_rad=np.asarray(q_current, dtype=np.float64),
        target_segmentation_id=np.asarray(geometry["target_segmentation_id"], dtype=np.int64),
        source_candidate_index=np.asarray(geometry["source_candidate_index"], dtype=np.int64),
        placement_object_center_world_xy_m=np.asarray(
            placement_meta["object_center_world_xy_m"], dtype=np.float64
        ),
        metadata_json=np.asarray(json.dumps(metadata, ensure_ascii=False)),
    )
    report_path = output_npz.with_suffix(".json")
    report_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {**metadata, "output_npz": str(output_npz), "report_json": str(report_path)}


def plan_flexible_route(
    *,
    client,
    project_root: Path,
    case_root: Path,
    cover_solutions: list[dict],
    q_current: np.ndarray,
    measured: dict,
    placement_registry: Path,
    config: dict,
    no_planner_collision_check: bool,
    block_unknown: bool,
    output_npz: Path | None = None,
) -> dict:
    """Search one full route around an already-passed exact COVER candidate."""
    if not cover_solutions:
        return {"status": "FAIL", "reason": "no exact COVER IK solution"}
    project_root = Path(project_root).resolve()
    case_root = Path(case_root).resolve()
    geometry = _candidate_geometry(case_root)
    layout = load_json(project_root / "08_dual_arm_scene_layout/config/manual_layout_calibrated.json")
    T_world_base = _world_from_base(project_root)
    T_base_from_world = np.linalg.inv(T_world_base)
    cfg = config["flexible_ik"]
    selection_cfg = cfg["selection"]
    beam_width = int(selection_cfg.get("beam_width", 64))
    solutions_per_pose = int(selection_cfg.get("solutions_per_pose", 4))
    summaries: list[dict] = []

    cover_nodes = [
        _node_from_record("cover", geometry["cover_flange_world"], row, {"nominal_penalty": 0.0})
        for row in cover_solutions
    ]

    # PREGRASP: broad region around exact COVER.
    pre_cfg = cfg["pregrasp"]
    pre_wrist = sample_pregrasp(
        cover_wrist_world=geometry["cover_wrist_world"],
        approach_axis_world=geometry["approach_axis_world"],
        count=int(pre_cfg["samples"]),
        distance_range_m=tuple(pre_cfg["distance_range_m"]),
        lateral_half_width_m=float(pre_cfg["lateral_half_width_m"]),
        rotation_half_range_deg_xyz=tuple(pre_cfg["rotation_half_range_deg_xyz"]),
        nominal_distance_m=float(pre_cfg.get("nominal_distance_m", 0.10)),
    )
    wrist_from_flange = np.linalg.inv(geometry["flange_from_wrist"])
    pre_flange = PoseSampleSet(
        pre_wrist.poses_world @ wrist_from_flange[None], pre_wrist.metadata
    )
    pre_nodes, summary = _solve_pose_set(
        client=client, stage="pregrasp", pose_set=pre_flange, q_reference=q_current,
        measured=measured, geometry=geometry, T_base_from_world=T_base_from_world,
        T_world_base=T_world_base, no_planner_collision_check=no_planner_collision_check,
        block_unknown=block_unknown, solutions_per_pose=solutions_per_pose,
    )
    summaries.append(summary)
    if not pre_nodes:
        return {"status": "FAIL", "reason": "PREGRASP region has no IK", "stage_summaries": summaries}
    pre_beam = _expand_beam([_home_parent(q_current)], pre_nodes, beam_width=beam_width, selection_cfg=selection_cfg)
    cover_beam = _expand_beam(pre_beam, cover_nodes, beam_width=beam_width, selection_cfg=selection_cfg)
    if not cover_beam:
        return {"status": "FAIL", "reason": "PREGRASP cannot connect to exact COVER", "stage_summaries": summaries}
    summaries.append({"stage": "cover", "target_count": 1, "solution_count": len(cover_nodes), "beam_count": len(cover_beam)})

    # LIFT: retain official direction policy but turn 200 mm into a broad task set.
    lift_cfg = cfg["lift"]
    lift_axis = -geometry["approach_axis_world"] if geometry["is_top_grasp"] else np.asarray([0.0, 0.0, 1.0])
    lift_wrist = sample_lift(
        cover_wrist_world=geometry["cover_wrist_world"],
        lift_axis_world=lift_axis,
        count=int(lift_cfg["samples"]),
        distance_range_m=tuple(lift_cfg["distance_range_m"]),
        lateral_half_width_m=float(lift_cfg["lateral_half_width_m"]),
        rotation_half_range_deg_xyz=tuple(lift_cfg["rotation_half_range_deg_xyz"]),
        nominal_distance_m=float(lift_cfg.get("nominal_distance_m", 0.20)),
    )
    lift_flange = PoseSampleSet(lift_wrist.poses_world @ wrist_from_flange[None], lift_wrist.metadata)
    lift_nodes, summary = _solve_pose_set(
        client=client, stage="lift", pose_set=lift_flange, q_reference=cover_beam[0].q_rad,
        measured=measured, geometry=geometry, T_base_from_world=T_base_from_world,
        T_world_base=T_world_base, no_planner_collision_check=no_planner_collision_check,
        block_unknown=block_unknown, solutions_per_pose=solutions_per_pose,
    )
    summaries.append(summary)
    lift_beam = _expand_beam(cover_beam, lift_nodes, beam_width=beam_width, selection_cfg=selection_cfg)
    if not lift_beam:
        return {"status": "FAIL", "reason": "LIFT region has no route", "stage_summaries": summaries}

    # TRANSFER: broad corridor toward placement zone, not one fixed +180 mm point.
    zone_min, zone_max, table_top = placement_zone_bounds(layout)
    zone_center = 0.5 * (zone_min + zone_max)
    place_cfg = cfg["place"]
    nominal_height = float(place_cfg["nominal_object_size_xyz_m"][2])
    # The old runtime placed the wrist near grasp height.  Use that as the
    # corridor's base height, then sample 120-260 mm above it.
    place_wrist_nominal_z = float(geometry["cover_wrist_world"][2, 3] + place_cfg.get("release_wrist_height_delta_m", 0.01))
    transfer_cfg = cfg["transfer"]
    transfer_wrist = sample_transfer(
        lift_wrist_world_nominal=geometry["nominal_lift_wrist_world"],
        place_zone_center_xy_m=zone_center,
        place_wrist_nominal_z_m=place_wrist_nominal_z,
        count=int(transfer_cfg["samples"]),
        lambda_range=tuple(transfer_cfg["lambda_range"]),
        height_above_place_range_m=tuple(transfer_cfg["height_above_place_range_m"]),
        lateral_xy_half_width_m=float(transfer_cfg["lateral_xy_half_width_m"]),
        rotation_half_range_deg_xyz=tuple(transfer_cfg["rotation_half_range_deg_xyz"]),
        nominal_lambda=float(transfer_cfg.get("nominal_lambda", 0.65)),
        nominal_height_above_place_m=float(transfer_cfg.get("nominal_height_above_place_m", 0.18)),
    )
    transfer_flange = PoseSampleSet(
        transfer_wrist.poses_world @ wrist_from_flange[None], transfer_wrist.metadata
    )
    transfer_nodes, summary = _solve_pose_set(
        client=client, stage="transfer", pose_set=transfer_flange, q_reference=lift_beam[0].q_rad,
        measured=measured, geometry=geometry, T_base_from_world=T_base_from_world,
        T_world_base=T_world_base, no_planner_collision_check=no_planner_collision_check,
        block_unknown=block_unknown, solutions_per_pose=solutions_per_pose,
    )
    summaries.append(summary)
    transfer_beam = _expand_beam(lift_beam, transfer_nodes, beam_width=beam_width, selection_cfg=selection_cfg)
    if not transfer_beam:
        return {"status": "FAIL", "reason": "TRANSFER corridor has no route", "stage_summaries": summaries}

    # PLACE: all legal green-zone centres, nominal 120 mm object footprint,
    # multiple orientation/height variants per XY point.
    occupied = read_occupied_centres(placement_registry)
    centres = free_placement_centres_xy(
        layout=layout,
        nominal_object_size_xy_m=tuple(place_cfg["nominal_object_size_xyz_m"][:2]),
        edge_margin_m=float(place_cfg["edge_margin_m"]),
        grid_step_xy_m=tuple(place_cfg["grid_step_xy_m"]),
        occupied_centres_xy_m=occupied,
        minimum_center_spacing_m=float(place_cfg["minimum_center_spacing_m"]),
        preferred_world_y_m=float(place_cfg["preferred_world_y_m"]),
    )
    place_flange = sample_place_from_centres(
        centres_xy_m=centres,
        object_world_initial=geometry["object_world_initial"],
        flange_from_object_grasp=geometry["flange_from_object_grasp"],
        samples_per_xy=int(place_cfg["samples_per_xy"]),
        table_top_world_z_m=table_top,
        nominal_object_height_m=nominal_height,
        z_extra_range_m=tuple(place_cfg["z_extra_range_m"]),
        object_rotation_half_range_deg_xyz=tuple(place_cfg["object_rotation_half_range_deg_xyz"]),
    )
    place_nodes, summary = _solve_pose_set(
        client=client, stage="place", pose_set=place_flange, q_reference=transfer_beam[0].q_rad,
        measured=measured, geometry=geometry, T_base_from_world=T_base_from_world,
        T_world_base=T_world_base, no_planner_collision_check=no_planner_collision_check,
        block_unknown=block_unknown, solutions_per_pose=solutions_per_pose,
    )
    summary["free_xy_count"] = int(len(centres))
    summaries.append(summary)
    place_beam = _expand_beam(transfer_beam, place_nodes, beam_width=beam_width, selection_cfg=selection_cfg)
    if not place_beam:
        return {"status": "FAIL", "reason": "PLACE region has no route", "stage_summaries": summaries}

    # RETREAT depends on the chosen PLACE.  Try several best PLACE parents;
    # this avoids a massive PLACE x RETREAT Cartesian product while still
    # preventing one unfortunate PLACE from killing the grasp candidate.
    retreat_cfg = cfg["retreat"]
    home_q = np.asarray(config.get("home_q_deg", [50, -70, 0, 40, 35, 0, 25]), dtype=np.float64)
    home_q = np.deg2rad(home_q)
    best_final: BeamState | None = None
    best_parent_chain: list[BeamState] | None = None
    best_retreat_summary = None
    parent_trials = min(int(selection_cfg.get("retreat_parent_trials", 8)), len(place_beam))
    for trial_index, place_parent in enumerate(place_beam[:parent_trials]):
        place_wrist = place_parent.node.target_pose_world @ geometry["flange_from_wrist"]
        retreat_wrist = sample_retreat(
            place_wrist_world=place_wrist,
            count=int(retreat_cfg["samples"]),
            upward_range_m=tuple(retreat_cfg["upward_range_m"]),
            xy_half_width_m=float(retreat_cfg["xy_half_width_m"]),
            rotation_half_range_deg_xyz=tuple(retreat_cfg["rotation_half_range_deg_xyz"]),
            nominal_upward_m=float(retreat_cfg.get("nominal_upward_m", 0.12)),
            start_index=5001 + trial_index * max(1, int(retreat_cfg["samples"])),
        )
        retreat_flange = PoseSampleSet(
            retreat_wrist.poses_world @ wrist_from_flange[None], retreat_wrist.metadata
        )
        retreat_nodes, retreat_summary = _solve_pose_set(
            client=client, stage="retreat", pose_set=retreat_flange, q_reference=place_parent.q_rad,
            measured=measured, geometry=geometry, T_base_from_world=T_base_from_world,
            T_world_base=T_world_base, no_planner_collision_check=no_planner_collision_check,
            block_unknown=block_unknown, solutions_per_pose=solutions_per_pose,
        )
        if not retreat_nodes:
            continue
        retreat_beam = _expand_beam([place_parent], retreat_nodes, beam_width=beam_width, selection_cfg=selection_cfg)
        for state in retreat_beam:
            final_cost = state.cost + float(selection_cfg.get("home_return_weight", 0.25)) * _transition_cost(
                state.q_rad, home_q, selection_cfg
            )
            if best_final is None or final_cost < best_final.cost:
                best_final = BeamState(node=state.node, cost=float(final_cost), parent=state.parent)
                best_parent_chain = _ancestry(state)
                best_retreat_summary = retreat_summary
    if best_final is None or best_parent_chain is None:
        return {"status": "FAIL", "reason": "RETREAT region has no route", "stage_summaries": summaries}
    summaries.append(best_retreat_summary or {"stage": "retreat"})

    # Decode q_current -> pregrasp -> cover -> lift -> transfer -> place -> retreat.
    chain = best_parent_chain
    by_stage = {state.node.stage: state for state in chain if state.node.stage != "q_current"}
    by_stage["retreat"] = best_final
    required = {"pregrasp", "cover", "lift", "transfer", "place", "retreat"}
    if required.difference(by_stage):
        raise RuntimeError(f"internal beam chain missing stages: {sorted(required.difference(by_stage))}")

    # Mandatory far-field HOME -> PREGRASP safety gate.
    # HOME and PREGRASP both use PREGRASP q20, matching FORM_PREGRASP
    # before arm motion in the real executor.
    home_gate_cfg = config.get("home_pregrasp_collision_gate", {})
    if bool(home_gate_cfg.get("enabled", True)):
        pregrasp_named = _named_state(geometry, measured, "pregrasp")
        home_pregrasp_path_report = client.check_joint_path(
            np.stack([q_current, by_stage["pregrasp"].q_rad]),
            measured,
            joint_positions_by_node=[pregrasp_named, pregrasp_named],
            T_world_base=T_world_base,
            phases=["pregrasp"],
            margin_m=float(home_gate_cfg.get("margin_m", 0.0)),
            path_max_joint_step_rad=math.radians(
                float(
                    home_gate_cfg.get(
                        "path_max_joint_step_deg",
                        config.get("approach_path_max_joint_step_deg", 3.0),
                    )
                )
            ),
            check_observed_map=True,
        )
        summaries.append({
            "stage": "home_pregrasp_collision_gate",
            "path_pass": bool(home_pregrasp_path_report.get("path_pass")),
            "path_max_joint_step_rad": home_pregrasp_path_report.get("path_max_joint_step_rad"),
            "path_first_failure": home_pregrasp_path_report.get("path_first_failure"),
            "path_segments": home_pregrasp_path_report.get("path_segments"),
        })
        if not bool(home_pregrasp_path_report.get("path_pass")):
            return {
                "status": "FAIL",
                "reason": "HOME->PREGRASP observed-map collision gate failed",
                "stage_summaries": summaries,
                "home_pregrasp_path_report": home_pregrasp_path_report,
            }

    # Optional final continuous observed-map check.  In the user's current
    # diagnostic mode this is skipped exactly like the existing flag requests.
    final_path_report = None
    if not no_planner_collision_check:
        q_nodes = np.stack([
            q_current,
            by_stage["pregrasp"].q_rad,
            by_stage["cover"].q_rad,
            by_stage["lift"].q_rad,
            by_stage["transfer"].q_rad,
            by_stage["place"].q_rad,
            by_stage["retreat"].q_rad,
            home_q,
        ])
        states = [
            {str(k): float(v) for k, v in measured.items()},
            _named_state(geometry, measured, "pregrasp"),
            _named_state(geometry, measured, "cover"),
            _named_state(geometry, measured, "lift"),
            _named_state(geometry, measured, "transfer"),
            _named_state(geometry, measured, "place"),
            _named_state(geometry, measured, "retreat"),
            _named_state(geometry, measured, "retreat"),  # HOME keeps the released/open hand
        ]
        final_path_report = client.check_joint_path(
            q_nodes,
            measured,
            joint_positions_by_node=states,
            T_world_base=T_world_base,
            phases=["pregrasp", "cover", "lift", "lift", "lift", "lift", "lift"],
            margin_m=0.0,
            path_max_joint_step_rad=math.radians(float(config.get("approach_path_max_joint_step_deg", 3.0))),
            check_observed_map=True,
        )
        if not bool(final_path_report.get("path_pass")):
            return {
                "status": "FAIL",
                "reason": "selected flexible route failed final observed-map path check",
                "stage_summaries": summaries,
                "final_path_report": final_path_report,
            }

    output_npz = (
        case_root / "07_arm_execution/flexible_route_plan.npz"
        if output_npz is None else Path(output_npz).resolve()
    )
    report = _write_plan(
        geometry=geometry,
        q_current=q_current,
        chosen=by_stage,
        output_npz=output_npz,
        summaries=summaries,
        final_path_report=final_path_report,
        placement_registry=placement_registry,
    )
    return report
