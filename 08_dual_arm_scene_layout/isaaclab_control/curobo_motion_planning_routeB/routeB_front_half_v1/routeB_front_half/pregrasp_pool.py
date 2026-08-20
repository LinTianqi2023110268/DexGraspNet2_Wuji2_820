from __future__ import annotations

"""Build Route B's post-retarget PREGRASP goal pool.

This module deliberately stops at endpoint IK:
- exact COVER solutions are already provided by the existing strict screen;
- PREGRASP is sampled in the existing relaxed legal 6D region;
- no HOME->PREGRASP joint interpolation check is run;
- no PREGRASP->COVER path check is run here.

The real current->PREGRASP path is delegated to the true 7DOF Route B
MotionPlanner backend.
"""

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np


RIGHT_ARM_JOINTS = tuple(f"arm_r_joint_{i}" for i in range(1, 8))


@dataclass
class FrontHalfGoal:
    case_root: str
    candidate_index: int
    official_score: float
    candidate_order: int
    q_pregrasp_rad: np.ndarray
    q_cover_rad: np.ndarray
    pregrasp_pose_world: np.ndarray
    pair_score: float
    pregrasp_target_index: int
    pregrasp_solution_index: int
    cover_solution_index: int
    pregrasp_inner_limit_margin_rad: float
    cover_inner_limit_margin_rad: float


@dataclass
class FrontHalfGoalPool:
    goals: list[FrontHalfGoal]
    case_summaries: list[dict[str, Any]]
    q_current_rad: np.ndarray

    @property
    def goal_count(self) -> int:
        return len(self.goals)

    @property
    def case_count(self) -> int:
        return len({g.case_root for g in self.goals})


def _load_project_helpers():
    # Import lazily so pure artifact/unit tests do not require the project tree.
    from planning.flexible_pose_sampling import PoseSampleSet, sample_pregrasp
    from planning.flexible_route_search import _candidate_geometry, _node_from_record
    from planning.simplified_route_search import (
        _pair_score,
        _route_tuning,
        _solve_relaxed_pose_set,
    )
    return (
        PoseSampleSet,
        sample_pregrasp,
        _candidate_geometry,
        _node_from_record,
        _pair_score,
        _route_tuning,
        _solve_relaxed_pose_set,
    )


def build_front_half_goal_pool(
    *,
    client: Any,
    project_root: Path,
    passed_cover_rows: Sequence[dict[str, Any]],
    q_current: np.ndarray,
    config: dict[str, Any],
    max_candidate_cases: int = 32,
    goals_per_case: int = 8,
    max_total_goals: int = 128,
) -> FrontHalfGoalPool:
    """Create ordered PREGRASP goals for true Route B planning.

    Candidate order remains the upstream DGN2 / reach-priority order.
    Within one candidate, q_PREGRASP/q_COVER pairs are sorted using the
    existing joint transition + nominal pose cost, but are NOT path-checked.
    """
    (
        PoseSampleSet,
        sample_pregrasp,
        _candidate_geometry,
        _node_from_record,
        _pair_score,
        _route_tuning,
        _solve_relaxed_pose_set,
    ) = _load_project_helpers()

    q_current = np.asarray(q_current, dtype=np.float64).reshape(7)
    project_root = Path(project_root).resolve()
    tuning = _route_tuning(config)
    pre_cfg = tuning["pregrasp"]
    selection_cfg = tuning["selection"]
    solutions_per_pose = int(selection_cfg.get("solutions_per_pose", 4))

    goals: list[FrontHalfGoal] = []
    summaries: list[dict[str, Any]] = []

    eligible_rows = [
        row for row in passed_cover_rows
        if bool(row.get("pass")) and row.get("cover_solutions")
    ][: max(1, int(max_candidate_cases))]

    for candidate_order, cover_row in enumerate(eligible_rows):
        if len(goals) >= int(max_total_goals):
            break
        case_root = Path(cover_row["case_root"]).resolve()
        geometry = _candidate_geometry(case_root)

        cover_nodes = [
            _node_from_record(
                "cover",
                geometry["cover_flange_world"],
                row,
                {"nominal_penalty": 0.0},
            )
            for row in cover_row["cover_solutions"]
        ]

        pre_wrist = sample_pregrasp(
            cover_wrist_world=geometry["cover_wrist_world"],
            approach_axis_world=geometry["approach_axis_world"],
            count=int(pre_cfg["samples"]),
            distance_range_m=tuple(pre_cfg["distance_range_m"]),
            lateral_half_width_m=float(pre_cfg["lateral_half_width_m"]),
            rotation_half_range_deg_xyz=tuple(
                pre_cfg["rotation_half_range_deg_xyz"]
            ),
            nominal_distance_m=float(
                pre_cfg.get("nominal_distance_m", 0.10)
            ),
        )
        wrist_from_flange = np.linalg.inv(geometry["flange_from_wrist"])
        pre_flange = PoseSampleSet(
            pre_wrist.poses_world @ wrist_from_flange[None],
            pre_wrist.metadata,
        )
        pre_nodes, ik_summary = _solve_relaxed_pose_set(
            client=client,
            stage="pregrasp",
            pose_set=pre_flange,
            q_reference=q_current,
            config=config,
            T_base_from_world=np.linalg.inv(
                __import__(
                    "planning.flexible_route_search",
                    fromlist=["_world_from_base"],
                )._world_from_base(project_root)
            ),
            solutions_per_pose=solutions_per_pose,
        )

        summary = {
            "candidate_order": int(candidate_order),
            "case_root": str(case_root),
            "candidate_index": int(cover_row["candidate_index"]),
            "official_score": float(cover_row.get("official_score", float("nan"))),
            "exact_cover_solution_count": int(len(cover_nodes)),
            "pregrasp_ik": ik_summary,
            "old_home_pregrasp_path_check": False,
            "old_pregrasp_cover_path_check": False,
            "status": "PASS" if pre_nodes else "NO_PREGRASP_IK",
        }

        if not pre_nodes:
            summaries.append(summary)
            continue

        pairs = [
            (
                float(
                    _pair_score(
                        pre,
                        cover,
                        q_current,
                        selection_cfg,
                    )
                ),
                pre,
                cover,
            )
            for pre in pre_nodes
            for cover in cover_nodes
        ]
        pairs.sort(key=lambda row: row[0])

        kept = 0
        seen_pre: set[tuple[float, ...]] = set()
        for pair_score, pre, cover in pairs:
            key = tuple(np.round(pre.q_rad, 7).tolist())
            if key in seen_pre:
                continue
            seen_pre.add(key)
            goals.append(
                FrontHalfGoal(
                    case_root=str(case_root),
                    candidate_index=int(cover_row["candidate_index"]),
                    official_score=float(
                        cover_row.get("official_score", float("nan"))
                    ),
                    candidate_order=int(candidate_order),
                    q_pregrasp_rad=np.asarray(
                        pre.q_rad, dtype=np.float64
                    ).copy(),
                    q_cover_rad=np.asarray(
                        cover.q_rad, dtype=np.float64
                    ).copy(),
                    pregrasp_pose_world=np.asarray(
                        pre.target_pose_world, dtype=np.float64
                    ).copy(),
                    pair_score=float(pair_score),
                    pregrasp_target_index=int(pre.target_index),
                    pregrasp_solution_index=int(pre.solution_index),
                    cover_solution_index=int(cover.solution_index),
                    pregrasp_inner_limit_margin_rad=float(
                        pre.inner_limit_margin_rad
                    ),
                    cover_inner_limit_margin_rad=float(
                        cover.inner_limit_margin_rad
                    ),
                )
            )
            kept += 1
            if kept >= int(goals_per_case):
                break
            if len(goals) >= int(max_total_goals):
                break

        summary["kept_routeB_pregrasp_goals"] = int(kept)
        summaries.append(summary)

    return FrontHalfGoalPool(
        goals=goals,
        case_summaries=summaries,
        q_current_rad=q_current.copy(),
    )


def save_front_half_goal_pool(
    path: str | Path,
    pool: FrontHalfGoalPool,
) -> tuple[Path, Path]:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not pool.goals:
        raise RuntimeError("cannot save an empty Route B front-half goal pool")

    goals = pool.goals
    np.savez_compressed(
        path,
        arm_joint_names=np.asarray(RIGHT_ARM_JOINTS, dtype="U"),
        q_current_rad=np.asarray(pool.q_current_rad, dtype=np.float32),
        q_pregrasp_rad=np.stack(
            [g.q_pregrasp_rad for g in goals]
        ).astype(np.float32),
        q_cover_rad=np.stack(
            [g.q_cover_rad for g in goals]
        ).astype(np.float32),
        pregrasp_pose_world=np.stack(
            [g.pregrasp_pose_world for g in goals]
        ).astype(np.float64),
        case_root=np.asarray([g.case_root for g in goals], dtype="U"),
        candidate_index=np.asarray(
            [g.candidate_index for g in goals], dtype=np.int64
        ),
        official_score=np.asarray(
            [g.official_score for g in goals], dtype=np.float64
        ),
        candidate_order=np.asarray(
            [g.candidate_order for g in goals], dtype=np.int64
        ),
        pair_score=np.asarray(
            [g.pair_score for g in goals], dtype=np.float64
        ),
        pregrasp_target_index=np.asarray(
            [g.pregrasp_target_index for g in goals], dtype=np.int64
        ),
        pregrasp_solution_index=np.asarray(
            [g.pregrasp_solution_index for g in goals], dtype=np.int64
        ),
        cover_solution_index=np.asarray(
            [g.cover_solution_index for g in goals], dtype=np.int64
        ),
        pregrasp_inner_limit_margin_rad=np.asarray(
            [g.pregrasp_inner_limit_margin_rad for g in goals],
            dtype=np.float64,
        ),
        cover_inner_limit_margin_rad=np.asarray(
            [g.cover_inner_limit_margin_rad for g in goals],
            dtype=np.float64,
        ),
    )

    report_path = path.with_suffix(".json")
    report = {
        "schema_version": 1,
        "stage": "ROUTEB_PREGRASP_GOAL_POOL",
        "goal_count": int(pool.goal_count),
        "case_count": int(pool.case_count),
        "old_home_pregrasp_path_check": False,
        "old_pregrasp_cover_path_check": False,
        "authoritative_next_gate": (
            "true 7DOF Route B MotionPlanner current->PREGRASP"
        ),
        "case_summaries": pool.case_summaries,
        "artifact_npz": str(path),
    }
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return path, report_path


def load_front_half_goal_pool(path: str | Path) -> dict[str, np.ndarray]:
    path = Path(path)
    with np.load(path, allow_pickle=False) as z:
        out = {key: np.asarray(z[key]) for key in z.files}
    if tuple(out["arm_joint_names"].astype(str).tolist()) != RIGHT_ARM_JOINTS:
        raise RuntimeError(
            "front-half goal pool arm joint contract is not right-arm 7DOF"
        )
    q_pre = np.asarray(out["q_pregrasp_rad"])
    q_cover = np.asarray(out["q_cover_rad"])
    if q_pre.ndim != 2 or q_pre.shape[1] != 7:
        raise RuntimeError(f"q_pregrasp_rad must be [M,7], got {q_pre.shape}")
    if q_cover.shape != q_pre.shape:
        raise RuntimeError(
            f"q_cover_rad shape mismatch: {q_cover.shape} vs {q_pre.shape}"
        )
    return out
