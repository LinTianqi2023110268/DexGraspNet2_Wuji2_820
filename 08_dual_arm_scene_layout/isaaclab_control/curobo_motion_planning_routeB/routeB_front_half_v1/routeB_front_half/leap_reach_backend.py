#!/usr/bin/env python3
from __future__ import annotations

"""Route B LEAP Target Reach Region backend.

Only one question is answered here:

    "Is this DGN2 LEAP grasp candidate inside or near a region where nearby
     LEAP grasp roots admit fast coarse right-arm IK?"

Deliberately NOT included:
- PREGRASP coarse IK
- observed RGB-D / ESDF collision checks
- self collision checks
- HOME->PREGRASP path checks
- PREGRASP->GRASP path checks
- corridor / graph construction
- Isaac Sim

The output is a cheap candidate-priority prior before LEAP->Wuji2 retargeting.
Downstream exact Wuji2 COVER/PREGRASP IK and Route B MotionPlanner remain the
authoritative gates.
"""

import argparse
import json
import math
from pathlib import Path
import re
import sys
import time
from typing import Any

import numpy as np

try:
    from .reach_contract import (
        PASS_DIRECT,
        PASS_NEAR_REGION,
        REJECT_OUTSIDE_REACH_REGION,
        pose_region_membership,
    )
except ImportError:
    from reach_contract import (  # type: ignore
        PASS_DIRECT,
        PASS_NEAR_REGION,
        REJECT_OUTSIDE_REACH_REGION,
        pose_region_membership,
    )


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def safe_slug(text: str) -> str:
    slug = re.sub(r"[^0-9A-Za-z._-]+", "_", str(text).strip()).strip("._")
    return slug[:64] or "target"


def world_from_base(project_root: Path) -> np.ndarray:
    layout = load_json(
        project_root / "08_dual_arm_scene_layout/config/manual_layout_calibrated.json"
    )
    return np.asarray(
        layout["transforms"]["dual_arm_mount"]["Gf_local_to_world_row_major"],
        dtype=np.float64,
    ).T


def build_approx_flange_grasps(
    *,
    rotation_world: np.ndarray,
    translation_world: np.ndarray,
    target_order: np.ndarray,
    T_leap_from_wuji2_wrist_mean: np.ndarray,
    wrist_from_flange: np.ndarray,
) -> np.ndarray:
    order = np.asarray(target_order, dtype=np.int64).reshape(-1)
    result = np.repeat(np.eye(4, dtype=np.float64)[None], len(order), axis=0)
    for rank, candidate_index in enumerate(order):
        i = int(candidate_index)
        T_world_leap = np.eye(4, dtype=np.float64)
        T_world_leap[:3, :3] = rotation_world[i]
        T_world_leap[:3, 3] = translation_world[i]
        result[rank] = (
            T_world_leap
            @ T_leap_from_wuji2_wrist_mean
            @ wrist_from_flange
        )
    return result


def _load_production_candidates(
    path: Path | None,
    target_order: np.ndarray,
    score: np.ndarray,
) -> list[dict[str, Any]]:
    if path is None:
        return [
            {
                "target_rank": int(rank),
                "candidate_index": int(idx),
                "score": float(score[int(idx)]),
            }
            for rank, idx in enumerate(target_order)
        ]
    payload = load_json(path)
    rows = payload.get("candidates")
    if not isinstance(rows, list):
        raise RuntimeError(f"{path}: expected top-level candidates list")
    return [dict(row) for row in rows]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--cycle-root", type=Path, required=True)
    parser.add_argument("--query", required=True)
    parser.add_argument("--bridge-npz", type=Path, required=True)
    parser.add_argument("--input-candidates-json", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--endpoint-ik-seeds", type=int, default=24)
    parser.add_argument("--endpoint-ik-batch-size", type=int, default=512)
    parser.add_argument("--coarse-joint-margin-deg", type=float, default=0.0)
    parser.add_argument("--extra-position-inflation-m", type=float, default=0.0)
    parser.add_argument("--extra-orientation-inflation-deg", type=float, default=0.0)
    args = parser.parse_args()

    started = time.perf_counter()
    project_root = args.project_root.expanduser().resolve()
    cycle_root = args.cycle_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    capture = cycle_root / "capture"
    query_slug = safe_slug(args.query)

    prediction = (
        capture / "dgn2" / query_slug / "official_leap_1024_target_ranked.npz"
    )
    robot_state_path = capture / "robot_state.json"
    robot_urdf = (
        project_root
        / "01_environment/vendor/wuji-description/dual_arm_right_wuji2/urdf/"
        "dual_arm_right_wuji2.urdf"
    )
    bridge_path = args.bridge_npz.expanduser().resolve()
    for path in (prediction, robot_state_path, robot_urdf, bridge_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    control_root = project_root / "08_dual_arm_scene_layout/isaaclab_control"
    sys.path.insert(0, str(control_root))
    from core.config import IKConfig  # noqa: E402
    from core.ik import CuroboGpuIK  # noqa: E402

    with np.load(prediction, allow_pickle=False) as z:
        target_order = np.asarray(
            z["target_score_descending_candidate_index"], dtype=np.int64
        )
        rotation_world = np.asarray(z["rotation_world"], dtype=np.float64)
        translation_world = np.asarray(z["translation_world"], dtype=np.float64)
        score = np.asarray(z["score"], dtype=np.float64)

    with np.load(bridge_path, allow_pickle=False) as z:
        T_leap_from_wuji2 = np.asarray(
            z["T_leap_from_wuji2_wrist_mean"], dtype=np.float64
        )
        flange_from_wrist = np.asarray(
            z["flange_from_wuji2_wrist"], dtype=np.float64
        )
        bridge_position_inflation_m = float(
            np.asarray(z["recommended_position_inflation_m"]).reshape(())
        )
        bridge_orientation_inflation_deg = float(
            np.asarray(z["recommended_orientation_inflation_deg"]).reshape(())
        )

    production_candidates = _load_production_candidates(
        None if args.input_candidates_json is None
        else args.input_candidates_json.expanduser().resolve(),
        target_order,
        score,
    )

    # Validate production identity against the DGN2 target ranking.
    global_rank_by_candidate = {
        int(candidate_index): int(rank)
        for rank, candidate_index in enumerate(target_order)
    }
    for row in production_candidates:
        idx = int(row["candidate_index"])
        rank = int(row["target_rank"])
        if idx not in global_rank_by_candidate:
            raise RuntimeError(
                f"production candidate {idx} does not exist in DGN2 target order"
            )
        if global_rank_by_candidate[idx] != rank:
            raise RuntimeError(
                f"target_rank mismatch for candidate {idx}: "
                f"production={rank}, dgn2={global_rank_by_candidate[idx]}"
            )

    wrist_from_flange = np.linalg.inv(flange_from_wrist)
    all_approx_world = build_approx_flange_grasps(
        rotation_world=rotation_world,
        translation_world=translation_world,
        target_order=target_order,
        T_leap_from_wuji2_wrist_mean=T_leap_from_wuji2,
        wrist_from_flange=wrist_from_flange,
    )

    T_base_world = np.linalg.inv(world_from_base(project_root))
    all_approx_base = np.asarray(
        [T_base_world @ T for T in all_approx_world], dtype=np.float64
    )

    print(
        "[LEAP REACH 1/3] batch coarse GRASP IK only | "
        f"target_candidates={len(target_order)} | "
        "collision=OFF | path=OFF",
        flush=True,
    )
    ik_cfg = IKConfig(
        device=args.device,
        num_seeds=int(args.endpoint_ik_seeds),
        batch_size=int(args.endpoint_ik_batch_size),
        return_seeds=int(args.endpoint_ik_seeds),
        minimum_inner_limit_margin_rad=math.radians(
            float(args.coarse_joint_margin_deg)
        ),
    )
    solver = CuroboGpuIK(robot_urdf, ik_cfg)
    ik_started = time.perf_counter()
    result = solver.solve(
        all_approx_base,
        return_seeds=int(args.endpoint_ik_seeds),
    )
    direct = np.any(np.asarray(result.accepted, dtype=bool), axis=1)
    ik_wall = time.perf_counter() - ik_started

    pos_radius = (
        bridge_position_inflation_m
        + float(args.extra_position_inflation_m)
    )
    rot_radius_deg = (
        bridge_orientation_inflation_deg
        + float(args.extra_orientation_inflation_deg)
    )
    rot_radius_rad = math.radians(rot_radius_deg)

    print(
        "[LEAP REACH 2/3] build inflated target reach region | "
        f"direct={int(np.count_nonzero(direct))}/{len(direct)} | "
        f"radius={1000.0*pos_radius:.1f} mm / {rot_radius_deg:.1f} deg",
        flush=True,
    )
    refs = all_approx_world[direct]
    region, nearest_pos, nearest_rot, nearest_ref_local = pose_region_membership(
        all_approx_world,
        refs,
        pos_radius,
        rot_radius_rad,
    )
    # A directly reachable sample must always remain in the region.
    region |= direct
    direct_global_ranks = np.flatnonzero(direct)
    nearest_ref_global_rank = np.full(len(target_order), -1, dtype=np.int64)
    valid = nearest_ref_local >= 0
    nearest_ref_global_rank[valid] = direct_global_ranks[
        nearest_ref_local[valid]
    ]

    rows: list[dict[str, Any]] = []
    direct_count = 0
    near_count = 0
    reject_count = 0
    for production_row in production_candidates:
        rank = int(production_row["target_rank"])
        idx = int(production_row["candidate_index"])
        if bool(direct[rank]):
            status = PASS_DIRECT
            direct_count += 1
        elif bool(region[rank]):
            status = PASS_NEAR_REGION
            near_count += 1
        else:
            status = REJECT_OUTSIDE_REACH_REGION
            reject_count += 1
        rows.append(
            {
                "target_rank": rank,
                "candidate_index": idx,
                "score": float(production_row.get("score", score[idx])),
                "status": status,
                "direct_coarse_ik": bool(direct[rank]),
                "reach_region_pass": bool(region[rank]),
                "nearest_reachable_position_distance_m": (
                    None if not np.isfinite(nearest_pos[rank])
                    else float(nearest_pos[rank])
                ),
                "nearest_reachable_orientation_distance_deg": (
                    None if not np.isfinite(nearest_rot[rank])
                    else float(math.degrees(nearest_rot[rank]))
                ),
                "nearest_reachable_target_rank": int(
                    nearest_ref_global_rank[rank]
                ),
            }
        )

    filter_payload = {
        "schema_version": 1,
        "filter": "LEAP_TARGET_REACH_REGION_ONLY",
        "query": args.query,
        "semantics": {
            "coarse_ik_subject": "DGN2 LEAP GRASP root mapped through calibrated LEAP->Wuji2 wrist bridge to approximate right-arm flange",
            "pregrasp_coarse_ik": False,
            "environment_collision_check": False,
            "self_collision_check": False,
            "rough_trajectory_space": False,
            "path_check": False,
            "authoritative_downstream_gates": [
                "post-retarget exact Wuji2 COVER IK",
                "post-retarget relaxed Wuji2 PREGRASP IK",
                "Route B true 7DOF current->PREGRASP MotionPlanner",
            ],
        },
        "rows": rows,
    }
    filter_path = output_dir / "leap_target_reach_filter.json"
    filter_path.write_text(
        json.dumps(filter_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    map_path = output_dir / "leap_target_reach_map.npz"
    np.savez_compressed(
        map_path,
        target_order=target_order.astype(np.int64),
        approx_flange_grasp_world=all_approx_world.astype(np.float32),
        direct_coarse_ik=direct.astype(np.uint8),
        reach_region_pass=region.astype(np.uint8),
        nearest_position_distance_m=nearest_pos.astype(np.float32),
        nearest_orientation_distance_rad=nearest_rot.astype(np.float32),
        nearest_reachable_target_rank=nearest_ref_global_rank.astype(np.int64),
        position_radius_m=np.asarray(pos_radius, dtype=np.float64),
        orientation_radius_rad=np.asarray(rot_radius_rad, dtype=np.float64),
    )

    report = {
        "schema_version": 1,
        "stage": "LEAP_TARGET_REACH_REGION_ONLY",
        "success": True,
        "query": args.query,
        "prediction": str(prediction),
        "bridge_npz": str(bridge_path),
        "all_target_candidate_count": int(len(target_order)),
        "production_candidate_count": int(len(production_candidates)),
        "all_direct_coarse_ik_count": int(np.count_nonzero(direct)),
        "all_reach_region_count": int(np.count_nonzero(region)),
        "production_direct_count": int(direct_count),
        "production_near_region_count": int(near_count),
        "production_reject_count": int(reject_count),
        "position_radius_m": float(pos_radius),
        "orientation_radius_deg": float(rot_radius_deg),
        "endpoint_ik_seeds": int(args.endpoint_ik_seeds),
        "endpoint_ik_batch_size": int(args.endpoint_ik_batch_size),
        "coarse_joint_margin_deg": float(args.coarse_joint_margin_deg),
        "ik_wall_time_s": float(ik_wall),
        "total_wall_time_s": float(time.perf_counter() - started),
        "filter_json": str(filter_path),
        "map_npz": str(map_path),
        "collision_checks": False,
        "rough_trajectory_space": False,
        "pregrasp_coarse_ik": False,
    }
    report_path = output_dir / "leap_target_reach_report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(
        "[LEAP REACH 3/3] "
        f"production PASS={direct_count+near_count}/{len(production_candidates)} "
        f"(direct={direct_count}, near={near_count}) | "
        f"reject={reject_count} | wall={report['total_wall_time_s']:.2f}s | "
        f"report={report_path}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
