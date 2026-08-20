from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Iterable, Sequence

import numpy as np


PASS_DIRECT = "PASS_DIRECT_IK"
PASS_NEAR_REGION = "PASS_NEAR_REACH_REGION"
REJECT_OUTSIDE_REACH_REGION = "REJECT_OUTSIDE_REACH_REGION"


def rotation_angle_rad(rotation: np.ndarray) -> float:
    R = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    x = np.clip((float(np.trace(R)) - 1.0) * 0.5, -1.0, 1.0)
    return float(math.acos(x))


def pose_region_membership(
    query_T: np.ndarray,
    reference_T: np.ndarray,
    position_radius_m: float,
    orientation_radius_rad: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Membership in an inflated union of reachable SE(3) samples.

    This is intentionally a reachability prior only.  There is no environment
    collision test, no self-collision test, and no path/corridor test here.
    """
    query_T = np.asarray(query_T, dtype=np.float64)
    reference_T = np.asarray(reference_T, dtype=np.float64)
    if query_T.ndim != 3 or query_T.shape[1:] != (4, 4):
        raise ValueError(f"query_T must be [N,4,4], got {query_T.shape}")
    if reference_T.ndim != 3 or reference_T.shape[1:] != (4, 4):
        raise ValueError(f"reference_T must be [M,4,4], got {reference_T.shape}")
    if position_radius_m < 0 or orientation_radius_rad < 0:
        raise ValueError("region radii must be non-negative")

    n = len(query_T)
    keep = np.zeros(n, dtype=bool)
    best_pos = np.full(n, np.inf, dtype=np.float64)
    best_rot = np.full(n, np.inf, dtype=np.float64)
    best_ref = np.full(n, -1, dtype=np.int64)
    if len(reference_T) == 0:
        return keep, best_pos, best_rot, best_ref

    qpos = query_T[:, :3, 3]
    rpos = reference_T[:, :3, 3]
    pos_radius = float(position_radius_m)
    rot_radius = float(orientation_radius_rad)

    # cKDTree is an acceleration only; the exact rule below is unchanged if
    # SciPy is unavailable.
    try:
        from scipy.spatial import cKDTree
        tree = cKDTree(rpos)
        neighborhoods = tree.query_ball_point(qpos, r=pos_radius)
        nearest_dist, nearest_idx = tree.query(qpos, k=1)
        for i, nbrs in enumerate(neighborhoods):
            if not nbrs:
                j = int(nearest_idx[i])
                best_pos[i] = float(nearest_dist[i])
                best_ref[i] = j
                best_rot[i] = rotation_angle_rad(
                    reference_T[j, :3, :3].T @ query_T[i, :3, :3]
                )
                continue
            best = None
            for j0 in nbrs:
                j = int(j0)
                dp = float(np.linalg.norm(qpos[i] - rpos[j]))
                dr = rotation_angle_rad(
                    reference_T[j, :3, :3].T @ query_T[i, :3, :3]
                )
                score = max(
                    dp / max(pos_radius, 1e-12),
                    dr / max(rot_radius, 1e-12),
                )
                row = (score, dp, dr, j)
                if best is None or row < best:
                    best = row
            assert best is not None
            _, dp, dr, j = best
            best_pos[i], best_rot[i], best_ref[i] = dp, dr, j
            keep[i] = bool(dp <= pos_radius and dr <= rot_radius)
    except Exception:
        # Chunked exact fallback.
        block = 256
        for start in range(0, n, block):
            stop = min(n, start + block)
            for i in range(start, stop):
                dp_all = np.linalg.norm(rpos - qpos[i][None, :], axis=1)
                close = np.flatnonzero(dp_all <= pos_radius)
                if len(close) == 0:
                    pool = np.asarray([int(np.argmin(dp_all))], dtype=np.int64)
                else:
                    pool = close
                best = None
                for j0 in pool:
                    j = int(j0)
                    dp = float(dp_all[j])
                    dr = rotation_angle_rad(
                        reference_T[j, :3, :3].T @ query_T[i, :3, :3]
                    )
                    score = max(
                        dp / max(pos_radius, 1e-12),
                        dr / max(rot_radius, 1e-12),
                    )
                    row = (score, dp, dr, j)
                    if best is None or row < best:
                        best = row
                assert best is not None
                _, dp, dr, j = best
                best_pos[i], best_rot[i], best_ref[i] = dp, dr, j
                keep[i] = bool(dp <= pos_radius and dr <= rot_radius)

    return keep, best_pos, best_rot, best_ref


def _candidate_identity(row: dict[str, Any]) -> tuple[int, int]:
    return int(row["target_rank"]), int(row["candidate_index"])


def _is_pass_status(status: str) -> bool:
    return str(status).upper() in {PASS_DIRECT, PASS_NEAR_REGION, "PASS"}


@dataclass(frozen=True)
class ReachOrdering:
    ordered_indices: list[int]
    pass_indices: list[int]
    reject_indices: list[int]
    direct_indices: list[int]
    near_region_indices: list[int]
    mode: str

    @property
    def pass_count(self) -> int:
        return len(self.pass_indices)

    @property
    def reject_count(self) -> int:
        return len(self.reject_indices)


def order_candidates_from_filter(
    production_candidates: Sequence[dict[str, Any]],
    filter_rows: Sequence[dict[str, Any]],
    *,
    mode: str = "priority_then_rescue",
) -> ReachOrdering:
    """Validate filter identity and produce production order.

    `priority_then_rescue` is the recommended rollout mode: candidates in the
    LEAP reach region are tried first, but outside-region candidates remain as
    a rescue tier so the approximate LEAP->Wuji2 bridge never gains hard veto
    power over downstream exact Wuji2 IK.
    """
    by_identity: dict[tuple[int, int], dict[str, Any]] = {}
    for row in filter_rows:
        key = _candidate_identity(row)
        if key in by_identity:
            raise RuntimeError(f"duplicate reach-filter row: {key}")
        by_identity[key] = dict(row)

    if len(by_identity) != len(production_candidates):
        raise RuntimeError(
            "reach-filter/candidate count mismatch: "
            f"filter={len(by_identity)} production={len(production_candidates)}"
        )

    passed: list[int] = []
    rejected: list[int] = []
    direct: list[int] = []
    near: list[int] = []
    for local_i, candidate in enumerate(production_candidates):
        key = _candidate_identity(candidate)
        row = by_identity.get(key)
        if row is None:
            raise RuntimeError(f"reach-filter missing production candidate {key}")
        status = str(row.get("status", "")).upper()
        if status == PASS_DIRECT:
            passed.append(local_i)
            direct.append(local_i)
        elif status in {PASS_NEAR_REGION, "PASS"}:
            passed.append(local_i)
            near.append(local_i)
        else:
            rejected.append(local_i)

    if mode == "priority_then_rescue":
        ordered = list(passed) + list(rejected)
    elif mode == "hard_filter":
        ordered = list(passed)
    else:
        raise ValueError(f"unsupported LEAP reach prefilter mode: {mode}")

    return ReachOrdering(
        ordered_indices=ordered,
        pass_indices=passed,
        reject_indices=rejected,
        direct_indices=direct,
        near_region_indices=near,
        mode=mode,
    )
