#!/usr/bin/env python3
from __future__ import annotations

"""Offline PLACE endpoint contract audit for Route B candidates.

This is a diagnostic script only.  It reuses the production Route A samplers
and the Route B back-half pool builder to compare PLACE target contracts and
endpoint IK counts without launching Isaac or running MotionPlanner.
"""

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def _load_json(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _parse_placement_zone_usda(path: Path) -> dict:
    text = Path(path).read_text(encoding="utf-8")
    m = re.search(r'def Cube "PlacementZone"(?P<body>.*?uniform token\[\] xformOpOrder.*?)\n\s*}', text, re.S)
    if not m:
        raise RuntimeError(f"{path}: PlacementZone prim not found")
    body = m.group("body")
    translate = re.search(r"xformOp:translate\s*=\s*\(([^)]+)\)", body)
    scale = re.search(r"xformOp:scale\s*=\s*\(([^)]+)\)", body)
    if not translate or not scale:
        raise RuntimeError(f"{path}: PlacementZone translate/scale not found")
    def vals(match):
        return [float(x.strip()) for x in match.group(1).split(",")]
    centre = np.asarray(vals(translate), dtype=np.float64)
    size = np.asarray(vals(scale), dtype=np.float64)
    return {
        "path": str(path),
        "center_world": centre.tolist(),
        "size": size.tolist(),
        "world_aabb_min": (centre - 0.5 * size).tolist(),
        "world_aabb_max": (centre + 0.5 * size).tolist(),
    }


def _pose_stats(poses_world: np.ndarray, T_base_from_world: np.ndarray) -> dict:
    poses_world = np.asarray(poses_world, dtype=np.float64)
    xyz_world = poses_world[:, :3, 3]
    xyz_base = np.stack([(T_base_from_world @ pose)[:3, 3] for pose in poses_world])
    dist_base = np.linalg.norm(xyz_base, axis=1)
    zyx = []
    for T in poses_world:
        R = T[:3, :3]
        sy = math.sqrt(float(R[0, 0] ** 2 + R[1, 0] ** 2))
        singular = sy < 1e-9
        if not singular:
            x = math.atan2(float(R[2, 1]), float(R[2, 2]))
            y = math.atan2(float(-R[2, 0]), sy)
            z = math.atan2(float(R[1, 0]), float(R[0, 0]))
        else:
            x = math.atan2(float(-R[1, 2]), float(R[1, 1]))
            y = math.atan2(float(-R[2, 0]), sy)
            z = 0.0
        zyx.append([math.degrees(x), math.degrees(y), math.degrees(z)])
    rpy = np.asarray(zyx, dtype=np.float64)
    closest = np.argsort(dist_base)[:5]
    center = np.median(xyz_world, axis=0)
    centerish = np.argsort(np.linalg.norm(xyz_world[:, :2] - center[:2], axis=1))[:5]
    return {
        "count": int(len(poses_world)),
        "world_xyz_min": xyz_world.min(axis=0).tolist(),
        "world_xyz_max": xyz_world.max(axis=0).tolist(),
        "world_xyz_median": np.median(xyz_world, axis=0).tolist(),
        "base_xyz_min": xyz_base.min(axis=0).tolist(),
        "base_xyz_max": xyz_base.max(axis=0).tolist(),
        "base_xyz_median": np.median(xyz_base, axis=0).tolist(),
        "distance_from_arm_base_minmax": [float(dist_base.min()), float(dist_base.max())],
        "distance_from_arm_base_median": float(np.median(dist_base)),
        "orientation_rpy_deg_min": rpy.min(axis=0).tolist(),
        "orientation_rpy_deg_max": rpy.max(axis=0).tolist(),
        "orientation_rpy_deg_median": np.median(rpy, axis=0).tolist(),
        "nominal_pose_world": poses_world[0].tolist(),
        "closest_to_arm_base": [
            {"index": int(i), "distance_m": float(dist_base[i]), "pose_world": poses_world[i].tolist()}
            for i in closest
        ],
        "zone_centerish": [
            {"index": int(i), "pose_world": poses_world[i].tolist()}
            for i in centerish
        ],
    }


def _stage_summary(rows: list[dict], stage: str) -> dict | None:
    matches = [r for r in rows if r.get("stage") == stage]
    return matches[-1] if matches else None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--cycle-root", type=Path, required=True)
    parser.add_argument("--case-root", type=Path, required=True)
    parser.add_argument("--front-half-report", type=Path, required=True)
    parser.add_argument("--placement-registry", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scan-goal-pool", type=Path)
    parser.add_argument("--scan-limit", type=int, default=0)
    args = parser.parse_args()

    root = args.project_root.resolve()
    control = root / "08_dual_arm_scene_layout/isaaclab_control"
    sys.path.insert(0, str(control))
    sys.path.insert(0, str(control / "closed_loop"))
    sys.path.insert(0, str(control / "curobo_motion_planning_routeB/routeB_full_pipeline_v1"))

    from core.bridge import CuroboWorkerClient  # noqa: WPS433
    from core.config import WorkerConfig  # noqa: WPS433
    from planning.simplified_route_search import plan_flexible_route  # noqa: WPS433
    from routeB_full_pipeline.backhalf_pool import (  # noqa: WPS433
        _helpers,
        build_backhalf_chain_pool,
    )

    cfg = _load_json(control / "closed_loop/config/closed_loop.json")
    layout = _load_json(root / "08_dual_arm_scene_layout/config/manual_layout_calibrated.json")
    robot_state = _load_json(args.cycle_root / "capture/robot_state.json")
    measured = {
        str(key): float(value)
        for key, value in robot_state["joint_positions_by_name"].items()
    }
    q_current = np.asarray(robot_state["right_arm_q_current_rad"], dtype=np.float64)
    front = _load_json(args.front_half_report)
    selected = front["selected"]
    q_cover = np.asarray(selected["q_cover_rad"], dtype=np.float64)

    h = _helpers()
    geometry = h["_candidate_geometry"](args.case_root)
    T_world_base = h["_world_from_base"](root)
    T_base_from_world = np.linalg.inv(T_world_base)
    tuning = h["_route_tuning"](cfg)
    place_cfg = tuning["place"]
    zone_min, zone_max, table_top = h["placement_zone_bounds"](layout)
    centres = h["free_placement_centres_xy"](
        layout=layout,
        nominal_object_size_xy_m=tuple(place_cfg["nominal_object_size_xyz_m"][:2]),
        edge_margin_m=float(place_cfg["edge_margin_m"]),
        grid_step_xy_m=tuple(place_cfg["grid_step_xy_m"]),
        occupied_centres_xy_m=h["read_occupied_centres"](args.placement_registry),
        minimum_center_spacing_m=float(place_cfg["minimum_center_spacing_m"]),
        preferred_world_y_m=float(place_cfg["preferred_world_y_m"]),
    )
    place_flange = h["sample_place_from_centres"](
        centres_xy_m=centres,
        object_world_initial=geometry["object_world_initial"],
        flange_from_object_grasp=geometry["flange_from_object_grasp"],
        samples_per_xy=int(place_cfg["samples_per_xy"]),
        table_top_world_z_m=table_top,
        nominal_object_height_m=float(place_cfg["nominal_object_size_xyz_m"][2]),
        z_extra_range_m=tuple(place_cfg["z_extra_range_m"]),
        object_rotation_half_range_deg_xyz=tuple(place_cfg["object_rotation_half_range_deg_xyz"]),
    )

    print("[PLACE CONTRACT]")
    print(f"config center world = {layout['transforms']['placement_zone']['position_world_m']}")
    print(f"config size         = {layout['geometry']['placement_zone_size_m']}")
    print(f"table top           = {table_top:.12f}")
    print("[PLACE TARGET AUDIT]")
    stats = _pose_stats(place_flange.poses_world, T_base_from_world)
    print(f"count        = {stats['count']}")
    print(f"world x      = {[stats['world_xyz_min'][0], stats['world_xyz_max'][0]]}")
    print(f"world y      = {[stats['world_xyz_min'][1], stats['world_xyz_max'][1]]}")
    print(f"world z      = {[stats['world_xyz_min'][2], stats['world_xyz_max'][2]]}")
    print(f"base x       = {[stats['base_xyz_min'][0], stats['base_xyz_max'][0]]}")
    print(f"base y       = {[stats['base_xyz_min'][1], stats['base_xyz_max'][1]]}")
    print(f"base z       = {[stats['base_xyz_min'][2], stats['base_xyz_max'][2]]}")
    print(f"distance from arm_base range = {stats['distance_from_arm_base_minmax']}")

    worker_cfg = WorkerConfig(
        startup_timeout_s=float(cfg.get("worker_startup_timeout_s", 180.0)),
        request_timeout_s=float(cfg.get("worker_request_timeout_s", 600.0)),
    )
    with CuroboWorkerClient(
        root,
        worker_config=worker_cfg,
        seeds=int(cfg.get("gpu_ik_seeds", 48)),
        batch_size=int(cfg.get("gpu_ik_batch_size", 512)),
    ) as client:
        routeb_pool = build_backhalf_chain_pool(
            client=client,
            project_root=root,
            case_root=args.case_root,
            q_cover_rad=q_cover,
            measured=measured,
            placement_registry=args.placement_registry,
            config=cfg,
            chain_limit=int(cfg["routeB_full_pipeline"]["backhalf_chain_limit"]),
        )
        cover_solution = {
            "q_rad": q_cover.tolist(),
            "target_index": 0,
            "solution_index": 0,
            "inner_limit_margin_rad": 0.0,
        }
        routea = plan_flexible_route(
            client=client,
            project_root=root,
            case_root=args.case_root,
            cover_solutions=[cover_solution],
            q_current=q_current,
            measured=measured,
            placement_registry=args.placement_registry,
            config=cfg,
            no_planner_collision_check=True,
            block_unknown=False,
            diagnostic_disable_home_pre_esdf=True,
            diagnostic_disable_home_pre_self_collision=True,
            diagnostic_disable_pre_cover_esdf=True,
            diagnostic_disable_pre_cover_self_collision=True,
            output_npz=None,
        )
        scan_rows = []
        if args.scan_goal_pool is not None and int(args.scan_limit) > 0:
            with np.load(args.scan_goal_pool, allow_pickle=False) as z:
                case_roots = np.asarray(z["case_root"]).astype(str)
                candidates = np.asarray(z["candidate_index"], dtype=np.int64)
                q_covers = np.asarray(z["q_cover_rad"], dtype=np.float64)
            seen: set[str] = set()
            for case_root_str, candidate, q_cover_row in zip(case_roots, candidates, q_covers):
                resolved = str(Path(case_root_str).resolve())
                if resolved in seen:
                    continue
                seen.add(resolved)
                if len(scan_rows) >= int(args.scan_limit):
                    break
                pool_i = build_backhalf_chain_pool(
                    client=client,
                    project_root=root,
                    case_root=Path(case_root_str),
                    q_cover_rad=np.asarray(q_cover_row, dtype=np.float64),
                    measured=measured,
                    placement_registry=args.placement_registry,
                    config=cfg,
                    chain_limit=int(cfg["routeB_full_pipeline"]["backhalf_chain_limit"]),
                )
                by_stage = {row.get("stage"): row for row in pool_i.summaries}
                row = {
                    "case_root": resolved,
                    "candidate_index": int(candidate),
                    "target_rank": int(re.search(r"rank_(\d+)", resolved).group(1))
                    if re.search(r"rank_(\d+)", resolved)
                    else None,
                    "chain_count": int(pool_i.chain_count),
                }
                for stage in ("lift", "transfer", "place", "retreat"):
                    s = by_stage.get(stage) or {}
                    row[stage] = {
                        "target_count": int(s.get("target_count", 0)),
                        "raw_success_target_count": int(s.get("raw_success_target_count", 0)),
                        "reachable_target_count": int(s.get("reachable_target_count", 0)),
                        "node_count": int(s.get("node_count", 0)),
                    }
                scan_rows.append(row)
                print(
                    "[PLACE SCAN] "
                    f"rank={row['target_rank']} cand={row['candidate_index']} "
                    f"LIFT={row['lift']['raw_success_target_count']} "
                    f"TRANSFER={row['transfer']['raw_success_target_count']} "
                    f"PLACE={row['place']['raw_success_target_count']} "
                    f"RETREAT={row['retreat']['raw_success_target_count']} "
                    f"chains={row['chain_count']}",
                    flush=True,
                )

    routeb_place = _stage_summary(routeb_pool.summaries, "place")
    routea_place = _stage_summary(routea.get("stage_summaries", []), "place")
    result = {
        "schema_version": 1,
        "stage": "PLACE_ENDPOINT_AUDIT",
        "case_root": str(args.case_root.resolve()),
        "candidate_index": int(selected["candidate_index"]),
        "target_rank": int(re.search(r"rank_(\d+)", str(args.case_root)).group(1))
        if re.search(r"rank_(\d+)", str(args.case_root))
        else None,
        "placement_contract": {
            "config_center_world": layout["transforms"]["placement_zone"]["position_world_m"],
            "config_size": layout["geometry"]["placement_zone_size_m"],
            "zone_min_xy": zone_min.tolist(),
            "zone_max_xy": zone_max.tolist(),
            "table_top": float(table_top),
            "formal_usda": _parse_placement_zone_usda(root / "08_dual_arm_scene_layout/scenes/manual_layout_calibrated.usda"),
            "production_usda": _parse_placement_zone_usda(root / "08_dual_arm_scene_layout/scenes/manual_layout_calibrated_mass_fixed.usda"),
        },
        "free_placement_centres": {
            "count": int(len(centres)),
            "xy_min": centres.min(axis=0).tolist(),
            "xy_max": centres.max(axis=0).tolist(),
            "xy_median": np.median(centres, axis=0).tolist(),
        },
        "place_target_audit": stats,
        "routeB_backhalf_pool": {
            "chain_count": int(routeb_pool.chain_count),
            "summaries": routeb_pool.summaries,
            "place_summary": routeb_place,
        },
        "routeA_same_candidate": {
            "status": routea.get("status"),
            "failed_stage": routea.get("failed_stage"),
            "reason": routea.get("reason"),
            "summaries": routea.get("stage_summaries", []),
            "place_summary": routea_place,
        },
        "A_B_place_target_contract": {
            "same_sampler_functions": True,
            "routeB_imports_routeA_functions": [
                "placement_zone_bounds",
                "free_placement_centres_xy",
                "sample_place_from_centres",
                "_solve_relaxed_pose_set",
                "_expand_beam",
            ],
            "note": (
                "Route B backhalf_pool directly imports and calls the Route A "
                "placement samplers; generated target matrices are therefore "
                "the same function contract for the same geometry/layout/registry."
            ),
        },
        "scan_goal_pool": scan_rows if "scan_rows" in locals() else [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(_jsonable(result), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[PLACE AUDIT REPORT] {args.output}")
    if routeb_place:
        print(f"[Route B PLACE] raw={routeb_place.get('raw_success_target_count')} reachable={routeb_place.get('reachable_target_count')} nodes={routeb_place.get('node_count')}")
    if routea_place:
        print(f"[Route A PLACE] raw={routea_place.get('raw_success_target_count')} reachable={routea_place.get('reachable_target_count')} nodes={routea_place.get('node_count')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
