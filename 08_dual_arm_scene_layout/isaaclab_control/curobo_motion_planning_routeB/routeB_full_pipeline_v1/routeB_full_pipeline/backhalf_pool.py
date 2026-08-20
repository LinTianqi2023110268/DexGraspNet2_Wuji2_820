from __future__ import annotations

"""Build post-retarget Route B endpoint chains without old path gates.

Existing project samplers define legal LIFT/TRANSFER/PLACE/RETREAT task regions.
This module reuses those samplers and GPU IK, but does not call check_joint_path.
The resulting endpoint chains are only proposals; the true Route B MotionPlanner
is the final path-feasibility authority.
"""

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class BackhalfChainPool:
    chains: list[dict[str, Any]]
    summaries: list[dict[str, Any]]

    @property
    def chain_count(self) -> int:
        return len(self.chains)


def _helpers():
    from planning.flexible_pose_sampling import (
        PoseSampleSet,
        free_placement_centres_xy,
        placement_zone_bounds,
        sample_lift,
        sample_place_from_centres,
        sample_retreat,
        sample_transfer,
    )
    from planning.ordered_place_policy import (
        build_release_place_flange_poses,
        ordered_near_to_far_slots,
    )
    from planning.flexible_route_search import (
        BeamState,
        IKNode,
        _candidate_geometry,
        _expand_beam,
        _transition_cost,
        _world_from_base,
        read_occupied_centres,
    )
    from planning.simplified_route_search import (
        _route_tuning,
        _solve_relaxed_pose_set,
    )
    return locals()


def _load_perception_target_geometry(geometry: dict[str, Any]) -> dict[str, Any]:
    case_root = Path(geometry["case_root"]).resolve()
    case_meta = json.loads((case_root / "case.json").read_text(encoding="utf-8"))
    path_value = case_meta.get("perception_target_geometry")
    if not path_value:
        raise RuntimeError(
            "PLACE V3 requires perception_target_geometry in case.json; "
            f"case={case_root}"
        )
    target_geometry_path = Path(str(path_value)).expanduser()
    if not target_geometry_path.is_absolute():
        target_geometry_path = (case_root / target_geometry_path).resolve()
    else:
        target_geometry_path = target_geometry_path.resolve()
    target_geometry = json.loads(target_geometry_path.read_text(encoding="utf-8"))
    if bool(target_geometry.get("simulator_identity_used", True)):
        raise RuntimeError(
            "PLACE V3 rejected target geometry that used simulator identity: "
            f"{target_geometry_path}"
        )
    return target_geometry


def _quantiles(values: list[float] | np.ndarray) -> dict[str, float | None]:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"best": None, "p10": None, "p50": None, "p90": None}
    return {
        "best": float(np.min(arr)),
        "p10": float(np.quantile(arr, 0.10)),
        "p50": float(np.quantile(arr, 0.50)),
        "p90": float(np.quantile(arr, 0.90)),
    }


def _place_v3_residual_audit(summary: dict[str, Any], metadata: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize available IK residuals for zero-PASS PLACE diagnosis.

    The current worker exposes full residuals only for accepted records.  When
    there are no accepted records, this report still preserves the V3 target
    generation contract and the acceptance policy.  If all targets fail, the
    next diagnostic step is to run a worker-level raw residual dump.
    """
    if "residual_audit" in summary:
        return dict(summary["residual_audit"])
    accepted = []
    for row in summary.get("accepted_solutions", []) or []:
        accepted.append(row)
    pos = [float(row.get("position_error_m", float("nan"))) for row in accepted]
    rot_deg = [
        float(np.degrees(float(row.get("orientation_error_rad", float("nan")))))
        for row in accepted
    ]
    margin_deg = [
        float(np.degrees(float(row.get("inner_limit_margin_rad", float("nan")))))
        for row in accepted
    ]
    return {
        "accepted_record_count": int(len(accepted)),
        "target_count": int(summary.get("target_count", len(metadata))),
        "accepted_residuals_only": True,
        "note": (
            "Worker response currently exposes residuals for accepted seeds. "
            "If PLACE accepted count is zero, run worker raw-residual audit "
            "before loosening tolerances."
        ),
        "position_error_m": _quantiles(pos),
        "orientation_error_deg": _quantiles(rot_deg),
        "inner_joint_margin_deg": _quantiles(margin_deg),
    }


def _chain_by_stage(final_state) -> dict[str, Any]:
    chain = []
    cursor = final_state
    while cursor is not None:
        chain.append(cursor)
        cursor = cursor.parent
    return {
        state.node.stage: state
        for state in reversed(chain)
        if state.node.stage != "q_current"
    }


def build_backhalf_chain_pool(
    *,
    client: Any,
    project_root: Path,
    case_root: Path,
    q_cover_rad: np.ndarray,
    measured: dict[str, float],
    placement_registry: Path,
    config: dict[str, Any],
    chain_limit: int = 32,
    placement_zone_override: dict[str, Any] | None = None,
) -> BackhalfChainPool:
    h = _helpers()
    PoseSampleSet = h["PoseSampleSet"]
    BeamState = h["BeamState"]
    IKNode = h["IKNode"]

    project_root = Path(project_root).resolve()
    case_root = Path(case_root).resolve()
    geometry = h["_candidate_geometry"](case_root)
    T_world_base = h["_world_from_base"](project_root)
    T_base_from_world = np.linalg.inv(T_world_base)
    tuning = h["_route_tuning"](config)
    selection = tuning["selection"]
    beam_width = int(selection.get("beam_width", 64))
    solutions_per_pose = int(selection.get("solutions_per_pose", 4))
    summaries: list[dict[str, Any]] = []

    cover_node = IKNode(
        stage="cover",
        q_rad=np.asarray(q_cover_rad, dtype=np.float64).reshape(7),
        target_index=0,
        solution_index=-1,
        target_pose_world=np.asarray(
            geometry["cover_flange_world"], dtype=np.float64
        ),
        metadata={"nominal_penalty": 0.0},
        inner_limit_margin_rad=0.0,
        intrinsic_penalty=0.0,
    )
    cover_state = BeamState(node=cover_node, cost=0.0, parent=None)
    wrist_from_flange = np.linalg.inv(geometry["flange_from_wrist"])

    # LIFT endpoint IK.
    lift_cfg = tuning["lift"]
    lift_axis = (
        -geometry["approach_axis_world"]
        if geometry["is_top_grasp"]
        else np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
    )
    lift_wrist = h["sample_lift"](
        cover_wrist_world=geometry["cover_wrist_world"],
        lift_axis_world=lift_axis,
        count=int(lift_cfg["samples"]),
        distance_range_m=tuple(lift_cfg["distance_range_m"]),
        lateral_half_width_m=float(lift_cfg["lateral_half_width_m"]),
        rotation_half_range_deg_xyz=tuple(
            lift_cfg["rotation_half_range_deg_xyz"]
        ),
        nominal_distance_m=float(lift_cfg.get("nominal_distance_m", 0.20)),
    )
    lift_flange = PoseSampleSet(
        lift_wrist.poses_world @ wrist_from_flange[None], lift_wrist.metadata
    )
    lift_nodes, summary = h["_solve_relaxed_pose_set"](
        client=client,
        stage="lift",
        pose_set=lift_flange,
        q_reference=cover_state.q_rad,
        config=config,
        T_base_from_world=T_base_from_world,
        solutions_per_pose=solutions_per_pose,
    )
    summaries.append(summary)
    lift_beam = h["_expand_beam"](
        [cover_state],
        lift_nodes,
        beam_width=beam_width,
        selection_cfg=selection,
    )
    if not lift_beam:
        return BackhalfChainPool([], summaries)

    # TRANSFER endpoint IK.
    layout = json.loads(
        (
            project_root
            / "08_dual_arm_scene_layout/config/manual_layout_calibrated.json"
        ).read_text(encoding="utf-8")
    )
    if placement_zone_override is not None:
        layout = json.loads(json.dumps(layout))
        layout["transforms"]["placement_zone"]["position_world_m"] = [
            float(v) for v in placement_zone_override["center_world_m"]
        ]
        layout["geometry"]["placement_zone_size_m"] = [
            float(v) for v in placement_zone_override["size_m"]
        ]
    zone_min, zone_max, table_top = h["placement_zone_bounds"](layout)
    zone_center = 0.5 * (zone_min + zone_max)
    place_cfg = tuning["place"]
    nominal_height = float(place_cfg["nominal_object_size_xyz_m"][2])
    place_wrist_nominal_z = float(
        geometry["cover_wrist_world"][2, 3]
        + place_cfg.get("release_wrist_height_delta_m", 0.01)
    )
    transfer_cfg = tuning["transfer"]
    transfer_wrist = h["sample_transfer"](
        lift_wrist_world_nominal=geometry["nominal_lift_wrist_world"],
        place_zone_center_xy_m=zone_center,
        place_wrist_nominal_z_m=place_wrist_nominal_z,
        count=int(transfer_cfg["samples"]),
        lambda_range=tuple(transfer_cfg["lambda_range"]),
        height_above_place_range_m=tuple(
            transfer_cfg["height_above_place_range_m"]
        ),
        lateral_xy_half_width_m=float(
            transfer_cfg["lateral_xy_half_width_m"]
        ),
        rotation_half_range_deg_xyz=tuple(
            transfer_cfg["rotation_half_range_deg_xyz"]
        ),
        nominal_lambda=float(transfer_cfg.get("nominal_lambda", 0.65)),
        nominal_height_above_place_m=float(
            transfer_cfg.get("nominal_height_above_place_m", 0.18)
        ),
    )
    transfer_flange = PoseSampleSet(
        transfer_wrist.poses_world @ wrist_from_flange[None],
        transfer_wrist.metadata,
    )
    transfer_nodes, summary = h["_solve_relaxed_pose_set"](
        client=client,
        stage="transfer",
        pose_set=transfer_flange,
        q_reference=lift_beam[0].q_rad,
        config=config,
        T_base_from_world=T_base_from_world,
        solutions_per_pose=solutions_per_pose,
    )
    summaries.append(summary)
    transfer_beam = h["_expand_beam"](
        lift_beam,
        transfer_nodes,
        beam_width=beam_width,
        selection_cfg=selection,
    )
    if not transfer_beam:
        return BackhalfChainPool([], summaries)

    # PLACE endpoint IK.
    occupied = h["read_occupied_centres"](Path(placement_registry))
    place_v3_slots = []
    place_v3_metadata = []
    if placement_zone_override is not None:
        target_geometry = _load_perception_target_geometry(geometry)
        target_anchor_world_initial = np.asarray(
            target_geometry["T_world_target_anchor"], dtype=np.float64
        )
        robust_min = np.asarray(
            target_geometry["robust_aabb_world_min_m"], dtype=np.float64
        )
        anchor_to_visible_bottom_m = float(
            target_anchor_world_initial[2, 3] - robust_min[2]
        )
        source_center = np.asarray(
            layout["transforms"]["source_zone"]["position_world_m"],
            dtype=np.float64,
        )
        place_v3_slots = h["ordered_near_to_far_slots"](
            zone_center_world_m=placement_zone_override["center_world_m"],
            zone_size_m=placement_zone_override["size_m"],
            source_center_world_m=source_center,
            nominal_object_size_xy_m=tuple(
                place_cfg["nominal_object_size_xyz_m"][:2]
            ),
            edge_margin_m=float(place_cfg["edge_margin_m"]),
            longitudinal_spacing_m=float(
                place_cfg.get(
                    "longitudinal_spacing_m",
                    place_cfg.get(
                        "minimum_center_spacing_m",
                        place_cfg["grid_step_xy_m"][0],
                    ),
                )
            ),
            minimum_center_spacing_m=float(
                place_cfg["minimum_center_spacing_m"]
            ),
            transverse_slots=int(place_cfg.get("transverse_slots", 2)),
            occupied_centres_xy_m=occupied,
        )
        poses, place_v3_metadata = h["build_release_place_flange_poses"](
            slots=place_v3_slots,
            target_anchor_world_initial=target_anchor_world_initial,
            grasp_flange_world=geometry["object_world_initial"]
            @ np.linalg.inv(geometry["flange_from_object_grasp"]),
            table_top_world_z_m=table_top,
            anchor_to_visible_bottom_m=anchor_to_visible_bottom_m,
            release_clearance_values_m=tuple(
                place_cfg.get(
                    "release_clearance_values_m",
                    (0.08, 0.10, 0.12),
                )
            ),
        )
        place_flange = PoseSampleSet(poses, place_v3_metadata)
    else:
        centres = h["free_placement_centres_xy"](
            layout=layout,
            nominal_object_size_xy_m=tuple(
                place_cfg["nominal_object_size_xyz_m"][:2]
            ),
            edge_margin_m=float(place_cfg["edge_margin_m"]),
            grid_step_xy_m=tuple(place_cfg["grid_step_xy_m"]),
            occupied_centres_xy_m=occupied,
            minimum_center_spacing_m=float(
                place_cfg["minimum_center_spacing_m"]
            ),
            preferred_world_y_m=float(place_cfg["preferred_world_y_m"]),
        )
        place_flange = h["sample_place_from_centres"](
            centres_xy_m=centres,
            object_world_initial=geometry["object_world_initial"],
            flange_from_object_grasp=geometry["flange_from_object_grasp"],
            samples_per_xy=int(place_cfg["samples_per_xy"]),
            table_top_world_z_m=table_top,
            nominal_object_height_m=nominal_height,
            z_extra_range_m=tuple(place_cfg["z_extra_range_m"]),
            object_rotation_half_range_deg_xyz=tuple(
                place_cfg["object_rotation_half_range_deg_xyz"]
            ),
        )
    place_nodes, summary = h["_solve_relaxed_pose_set"](
        client=client,
        stage="place",
        pose_set=place_flange,
        q_reference=transfer_beam[0].q_rad,
        config=config,
        T_base_from_world=T_base_from_world,
        solutions_per_pose=solutions_per_pose,
    )
    if placement_zone_override is not None:
        slot_counts: dict[str, int] = {}
        for row in place_v3_metadata:
            slot_counts[str(row["slot_id"])] = slot_counts.get(str(row["slot_id"]), 0) + 1
        summary["place_v3"] = {
            "enabled": True,
            "semantics": "ordered slot -> PLACE_RELEASE about 8/10/12 cm above slot -> physics settle",
            "simulator_target_identity_used": False,
            "source_zone_center_world_m": [
                float(v) for v in layout["transforms"]["source_zone"]["position_world_m"]
            ],
            "placement_zone_override": placement_zone_override,
            "slot_count": int(len(place_v3_slots)),
            "ordered_slots": [slot.to_jsonable() for slot in place_v3_slots],
            "release_pose_count": int(len(place_v3_metadata)),
            "release_pose_count_by_slot": slot_counts,
            "release_clearance_values_m": list(
                place_cfg.get("release_clearance_values_m", (0.08, 0.10, 0.12))
            ),
            "anchor_to_visible_bottom_m": float(anchor_to_visible_bottom_m),
            "target_geometry": target_geometry,
        }
    else:
        summary["free_xy_count"] = int(len(centres))
    if placement_zone_override is not None:
        summary["placement_zone_override"] = placement_zone_override
    if not place_nodes:
        summary["place_v3_residual_audit"] = _place_v3_residual_audit(
            summary, place_v3_metadata
        )
    summaries.append(summary)
    place_beam = h["_expand_beam"](
        transfer_beam,
        place_nodes,
        beam_width=beam_width,
        selection_cfg=selection,
    )
    if not place_beam:
        return BackhalfChainPool([], summaries)

    # RETREAT endpoint IK for several PLACE parents.
    retreat_cfg = tuning["retreat"]
    home_q = np.deg2rad(
        np.asarray(
            config.get("home_q_deg", [50, -70, 0, 40, 35, 0, 25]),
            dtype=np.float64,
        )
    )
    complete: list[tuple[float, Any]] = []
    parent_trials = min(
        int(selection.get("retreat_parent_trials", 8)), len(place_beam)
    )
    for trial_i, place_parent in enumerate(place_beam[:parent_trials]):
        place_wrist = (
            place_parent.node.target_pose_world @ geometry["flange_from_wrist"]
        )
        retreat_wrist = h["sample_retreat"](
            place_wrist_world=place_wrist,
            count=int(retreat_cfg["samples"]),
            upward_range_m=tuple(retreat_cfg["upward_range_m"]),
            xy_half_width_m=float(retreat_cfg["xy_half_width_m"]),
            rotation_half_range_deg_xyz=tuple(
                retreat_cfg["rotation_half_range_deg_xyz"]
            ),
            nominal_upward_m=float(
                retreat_cfg.get("nominal_upward_m", 0.12)
            ),
            start_index=5001
            + trial_i * max(1, int(retreat_cfg["samples"])),
        )
        retreat_flange = PoseSampleSet(
            retreat_wrist.poses_world @ wrist_from_flange[None],
            retreat_wrist.metadata,
        )
        retreat_nodes, summary = h["_solve_relaxed_pose_set"](
            client=client,
            stage="retreat",
            pose_set=retreat_flange,
            q_reference=place_parent.q_rad,
            config=config,
            T_base_from_world=T_base_from_world,
            solutions_per_pose=solutions_per_pose,
        )
        summary["place_parent_trial"] = int(trial_i)
        summaries.append(summary)
        if not retreat_nodes:
            continue
        retreat_beam = h["_expand_beam"](
            [place_parent],
            retreat_nodes,
            beam_width=beam_width,
            selection_cfg=selection,
        )
        for state in retreat_beam:
            total = float(state.cost) + float(
                selection.get("home_return_weight", 0.25)
            ) * h["_transition_cost"](
                state.q_rad, home_q, selection
            )
            complete.append((total, state))

    complete.sort(key=lambda row: row[0])
    chains: list[dict[str, Any]] = []
    for score, final_state in complete[: max(1, int(chain_limit))]:
        by = _chain_by_stage(final_state)
        if not {"lift", "transfer", "place", "retreat"}.issubset(by):
            continue
        chains.append(
            {
                "score": float(score),
                "q_lift_rad": by["lift"].q_rad.tolist(),
                "q_transfer_rad": by["transfer"].q_rad.tolist(),
                "q_place_rad": by["place"].q_rad.tolist(),
                "q_retreat_rad": by["retreat"].q_rad.tolist(),
                "lift_pose_world": by["lift"].node.target_pose_world.tolist(),
                "transfer_pose_world": by[
                    "transfer"
                ].node.target_pose_world.tolist(),
                "place_pose_world": by[
                    "place"
                ].node.target_pose_world.tolist(),
                "retreat_pose_world": by[
                    "retreat"
                ].node.target_pose_world.tolist(),
            }
        )
    return BackhalfChainPool(chains, summaries)


def save_backhalf_chain_pool(
    path: Path, pool: BackhalfChainPool
) -> tuple[Path, Path]:
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    if not pool.chains:
        raise RuntimeError("cannot save an empty Route B back-half chain pool")

    def arr(key: str, dtype=np.float32):
        return np.asarray([row[key] for row in pool.chains], dtype=dtype)

    np.savez_compressed(
        path,
        score=arr("score", np.float64),
        q_lift_rad=arr("q_lift_rad"),
        q_transfer_rad=arr("q_transfer_rad"),
        q_place_rad=arr("q_place_rad"),
        q_retreat_rad=arr("q_retreat_rad"),
        lift_pose_world=arr("lift_pose_world", np.float64),
        transfer_pose_world=arr("transfer_pose_world", np.float64),
        place_pose_world=arr("place_pose_world", np.float64),
        retreat_pose_world=arr("retreat_pose_world", np.float64),
    )
    report = path.with_suffix(".json")
    report.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "stage": "ROUTEB_BACKHALF_ENDPOINT_CHAIN_POOL",
                "chain_count": int(pool.chain_count),
                "collision_checks": False,
                "path_checks": False,
                "authoritative_next_gate": "true 7DOF Route B MotionPlanner",
                "summaries": pool.summaries,
                "artifact_npz": str(path),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return path, report
