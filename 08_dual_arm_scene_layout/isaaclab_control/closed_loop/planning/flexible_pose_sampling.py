#!/usr/bin/env python3
"""Deterministic 6D task-space sampling for the flexible closed-loop route.

Design rule
-----------
Keep cuRobo IK accuracy strict.  Relax *task targets* by sampling many legal
6D poses for non-contact stages (PREGRASP/LIFT/TRANSFER/PLACE/RETREAT).
COVER remains the exact grasp-root target produced by the reviewed
LEAP->Wuji2 pipeline.

The module depends only on NumPy so it is cheap to unit-test outside Isaac Sim
and outside the cuRobo environment.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

import numpy as np


_EPS = 1.0e-12


@dataclass(frozen=True)
class PoseSampleSet:
    """One deterministic pose pool plus per-pose scalar metadata."""

    poses_world: np.ndarray  # [N,4,4]
    metadata: list[dict]

    def __post_init__(self) -> None:
        poses = np.asarray(self.poses_world)
        if poses.ndim != 3 or poses.shape[1:] != (4, 4):
            raise ValueError(f"poses_world must be [N,4,4], got {poses.shape}")
        if len(self.metadata) != len(poses):
            raise ValueError("metadata length must match pose count")


def _van_der_corput(index: int, base: int) -> float:
    """Return one radical-inverse value in [0,1)."""
    result = 0.0
    denominator = 1.0
    n = int(index)
    while n:
        n, remainder = divmod(n, base)
        denominator *= base
        result += remainder / denominator
    return result


def halton(count: int, dimension: int, *, start_index: int = 1) -> np.ndarray:
    """Small dependency-free Halton low-discrepancy sequence.

    Sobol would also work, but Halton avoids adding SciPy as a runtime
    dependency to the Isaac/retarget/planner environments.
    """
    if count < 0:
        raise ValueError("count must be non-negative")
    if dimension <= 0:
        raise ValueError("dimension must be positive")
    primes = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37)
    if dimension > len(primes):
        raise ValueError(f"dimension {dimension} exceeds supported {len(primes)}")
    out = np.empty((count, dimension), dtype=np.float64)
    for row in range(count):
        idx = start_index + row
        for column, base in enumerate(primes[:dimension]):
            out[row, column] = _van_der_corput(idx, base)
    return out


def rotation_x(angle: float) -> np.ndarray:
    c, s = math.cos(float(angle)), math.sin(float(angle))
    return np.asarray([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]], dtype=np.float64)


def rotation_y(angle: float) -> np.ndarray:
    c, s = math.cos(float(angle)), math.sin(float(angle))
    return np.asarray([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=np.float64)


def rotation_z(angle: float) -> np.ndarray:
    c, s = math.cos(float(angle)), math.sin(float(angle))
    return np.asarray([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)


def local_rpy(rotation_deg_xyz: Iterable[float]) -> np.ndarray:
    rx, ry, rz = [math.radians(float(value)) for value in rotation_deg_xyz]
    return rotation_z(rz) @ rotation_y(ry) @ rotation_x(rx)


def pose_with_local_rotation(pose: np.ndarray, delta_deg_xyz: Iterable[float]) -> np.ndarray:
    result = np.asarray(pose, dtype=np.float64).copy()
    result[:3, :3] = result[:3, :3] @ local_rpy(delta_deg_xyz)
    return result


def orthogonal_plane_basis(axis_world: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (axis, u, v) with u/v spanning the plane perpendicular to axis."""
    axis = np.asarray(axis_world, dtype=np.float64).reshape(3)
    norm = float(np.linalg.norm(axis))
    if norm <= _EPS:
        raise ValueError("zero axis")
    axis = axis / norm
    helper = np.asarray([0.0, 0.0, 1.0])
    if abs(float(np.dot(axis, helper))) > 0.90:
        helper = np.asarray([1.0, 0.0, 0.0])
    u = np.cross(axis, helper)
    u /= np.linalg.norm(u)
    v = np.cross(axis, u)
    v /= np.linalg.norm(v)
    return axis, u, v


def _symmetric(unit: np.ndarray, half_width: float) -> np.ndarray:
    return (2.0 * np.asarray(unit, dtype=np.float64) - 1.0) * float(half_width)


def _interval(unit: np.ndarray, lower: float, upper: float) -> np.ndarray:
    lo, hi = float(lower), float(upper)
    if hi < lo:
        raise ValueError(f"invalid interval [{lo}, {hi}]")
    return lo + (hi - lo) * np.asarray(unit, dtype=np.float64)


def _ensure_nominal_first(
    poses: list[np.ndarray], metadata: list[dict], nominal_pose: np.ndarray, nominal_meta: dict,
) -> None:
    poses.insert(0, np.asarray(nominal_pose, dtype=np.float64).copy())
    metadata.insert(0, dict(nominal_meta, nominal=True, nominal_penalty=0.0))


def sample_pregrasp(
    *,
    cover_wrist_world: np.ndarray,
    approach_axis_world: np.ndarray,
    count: int,
    distance_range_m: tuple[float, float],
    lateral_half_width_m: float,
    rotation_half_range_deg_xyz: tuple[float, float, float],
    nominal_distance_m: float = 0.10,
    start_index: int = 1,
) -> PoseSampleSet:
    """Sample PREGRASP around the exact COVER/GRASP root.

    The reviewed Wuji2 builder uses:
        pregrasp_position = grasp_position - retreat * approach_axis
    so ``approach_axis_world`` points from PREGRASP toward the object.
    """
    cover = np.asarray(cover_wrist_world, dtype=np.float64)
    axis, u, v = orthogonal_plane_basis(approach_axis_world)
    nominal = cover.copy()
    nominal[:3, 3] = cover[:3, 3] - float(nominal_distance_m) * axis

    sample_count = max(0, int(count) - 1)
    h = halton(sample_count, 6, start_index=start_index)
    distances = _interval(h[:, 0], *distance_range_m) if sample_count else np.empty(0)
    du = _symmetric(h[:, 1], lateral_half_width_m) if sample_count else np.empty(0)
    dv = _symmetric(h[:, 2], lateral_half_width_m) if sample_count else np.empty(0)
    rot_half = np.asarray(rotation_half_range_deg_xyz, dtype=np.float64)

    poses: list[np.ndarray] = []
    meta: list[dict] = []
    for row in range(sample_count):
        delta_deg = _symmetric(h[row, 3:6], 1.0) * rot_half
        pose = cover.copy()
        pose[:3, 3] = cover[:3, 3] - distances[row] * axis + du[row] * u + dv[row] * v
        pose[:3, :3] = cover[:3, :3] @ local_rpy(delta_deg)
        penalty = (
            abs(float(distances[row]) - float(nominal_distance_m)) / max(1.0e-6, distance_range_m[1] - distance_range_m[0])
            + 0.5 * (abs(float(du[row])) + abs(float(dv[row]))) / max(1.0e-6, lateral_half_width_m)
            + 0.15 * float(np.linalg.norm(delta_deg / np.maximum(rot_half, 1.0)))
        )
        poses.append(pose)
        meta.append({
            "distance_m": float(distances[row]),
            "lateral_u_m": float(du[row]),
            "lateral_v_m": float(dv[row]),
            "rotation_delta_deg_xyz": delta_deg.tolist(),
            "nominal_penalty": float(penalty),
        })
    _ensure_nominal_first(
        poses,
        meta,
        nominal,
        {
            "distance_m": float(nominal_distance_m),
            "lateral_u_m": 0.0,
            "lateral_v_m": 0.0,
            "rotation_delta_deg_xyz": [0.0, 0.0, 0.0],
        },
    )
    return PoseSampleSet(np.stack(poses), meta)


def sample_lift(
    *,
    cover_wrist_world: np.ndarray,
    lift_axis_world: np.ndarray,
    count: int,
    distance_range_m: tuple[float, float],
    lateral_half_width_m: float,
    rotation_half_range_deg_xyz: tuple[float, float, float],
    nominal_distance_m: float = 0.20,
    start_index: int = 101,
) -> PoseSampleSet:
    """Sample LIFT around the original 200 mm DGN2/Wuji2 policy."""
    cover = np.asarray(cover_wrist_world, dtype=np.float64)
    axis, u, v = orthogonal_plane_basis(lift_axis_world)
    nominal = cover.copy()
    nominal[:3, 3] = cover[:3, 3] + float(nominal_distance_m) * axis

    sample_count = max(0, int(count) - 1)
    h = halton(sample_count, 6, start_index=start_index)
    distances = _interval(h[:, 0], *distance_range_m) if sample_count else np.empty(0)
    du = _symmetric(h[:, 1], lateral_half_width_m) if sample_count else np.empty(0)
    dv = _symmetric(h[:, 2], lateral_half_width_m) if sample_count else np.empty(0)
    rot_half = np.asarray(rotation_half_range_deg_xyz, dtype=np.float64)

    poses: list[np.ndarray] = []
    meta: list[dict] = []
    for row in range(sample_count):
        delta_deg = _symmetric(h[row, 3:6], 1.0) * rot_half
        pose = cover.copy()
        pose[:3, 3] = cover[:3, 3] + distances[row] * axis + du[row] * u + dv[row] * v
        pose[:3, :3] = cover[:3, :3] @ local_rpy(delta_deg)
        penalty = (
            abs(float(distances[row]) - float(nominal_distance_m)) / max(1.0e-6, distance_range_m[1] - distance_range_m[0])
            + 0.4 * (abs(float(du[row])) + abs(float(dv[row]))) / max(1.0e-6, lateral_half_width_m)
            + 0.15 * float(np.linalg.norm(delta_deg / np.maximum(rot_half, 1.0)))
        )
        poses.append(pose)
        meta.append({
            "distance_m": float(distances[row]),
            "lateral_u_m": float(du[row]),
            "lateral_v_m": float(dv[row]),
            "rotation_delta_deg_xyz": delta_deg.tolist(),
            "nominal_penalty": float(penalty),
        })
    _ensure_nominal_first(
        poses,
        meta,
        nominal,
        {
            "distance_m": float(nominal_distance_m),
            "lateral_u_m": 0.0,
            "lateral_v_m": 0.0,
            "rotation_delta_deg_xyz": [0.0, 0.0, 0.0],
        },
    )
    return PoseSampleSet(np.stack(poses), meta)


def placement_zone_bounds(layout: dict) -> tuple[np.ndarray, np.ndarray, float]:
    centre = np.asarray(layout["transforms"]["placement_zone"]["position_world_m"], dtype=np.float64)
    size = np.asarray(layout["geometry"]["placement_zone_size_m"], dtype=np.float64)
    table_centre = np.asarray(layout["transforms"]["table"]["position_world_m"], dtype=np.float64)
    table_size = np.asarray(layout["geometry"]["table_size_m"], dtype=np.float64)
    zone_min = centre[:2] - 0.5 * size[:2]
    zone_max = centre[:2] + 0.5 * size[:2]
    table_top = float(table_centre[2] + 0.5 * table_size[2])
    return zone_min, zone_max, table_top


def _axis_grid(lower: float, upper: float, step: float) -> np.ndarray:
    lower, upper, step = float(lower), float(upper), float(step)
    if step <= 0.0:
        raise ValueError("grid step must be positive")
    if upper < lower:
        return np.empty(0, dtype=np.float64)
    count = int(math.floor((upper - lower) / step)) + 1
    values = lower + step * np.arange(max(count, 1), dtype=np.float64)
    if len(values) and upper - values[-1] > 0.5 * step:
        values = np.append(values, upper)
    return values


def free_placement_centres_xy(
    *,
    layout: dict,
    nominal_object_size_xy_m: tuple[float, float],
    edge_margin_m: float,
    grid_step_xy_m: tuple[float, float],
    occupied_centres_xy_m: Iterable[Iterable[float]],
    minimum_center_spacing_m: float,
    preferred_world_y_m: float | None = None,
) -> np.ndarray:
    """Return all legal object-centre XY grid points.

    The object footprint is intentionally *nominal*, not asset-derived.  This
    keeps simulation and later real-robot placement policy consistent when the
    exact object dimensions are unknown.
    """
    zone_min, zone_max, _ = placement_zone_bounds(layout)
    half = 0.5 * np.asarray(nominal_object_size_xy_m, dtype=np.float64)
    lower = zone_min + float(edge_margin_m) + half
    upper = zone_max - float(edge_margin_m) - half
    step = np.asarray(grid_step_xy_m, dtype=np.float64)
    xs = _axis_grid(lower[0], upper[0], step[0])
    ys = _axis_grid(lower[1], upper[1], step[1])
    if not len(xs) or not len(ys):
        raise RuntimeError("nominal object does not fit inside placement zone")

    preferred_y = float(np.mean([lower[1], upper[1]])) if preferred_world_y_m is None else float(preferred_world_y_m)
    ys = np.asarray(sorted(ys.tolist(), key=lambda y: (abs(y - preferred_y), y)), dtype=np.float64)
    occupied = [np.asarray(value, dtype=np.float64).reshape(2) for value in occupied_centres_xy_m]
    free = []
    for x in xs:
        for y in ys:
            xy = np.asarray([x, y], dtype=np.float64)
            if any(float(np.linalg.norm(xy - used)) < float(minimum_center_spacing_m) for used in occupied):
                continue
            free.append(xy)
    if not free:
        raise RuntimeError("no free nominal placement centre remains")
    return np.stack(free)


def sample_place_from_centres(
    *,
    centres_xy_m: np.ndarray,
    object_world_initial: np.ndarray,
    flange_from_object_grasp: np.ndarray,
    samples_per_xy: int,
    table_top_world_z_m: float,
    nominal_object_height_m: float,
    z_extra_range_m: tuple[float, float],
    object_rotation_half_range_deg_xyz: tuple[float, float, float],
    start_index: int = 1001,
) -> PoseSampleSet:
    """Build PLACE flange poses that preserve the rigid grasp transform.

    The object is treated as a nominal-size body for final height/spacing.  Its
    *current measured orientation* is retained as the nominal orientation, then
    small task-legal rotation variations are sampled around it.
    """
    centres = np.asarray(centres_xy_m, dtype=np.float64)
    if centres.ndim != 2 or centres.shape[1] != 2:
        raise ValueError(f"centres_xy_m must be [N,2], got {centres.shape}")
    initial = np.asarray(object_world_initial, dtype=np.float64)
    flange_from_object = np.asarray(flange_from_object_grasp, dtype=np.float64)
    if samples_per_xy <= 0:
        raise ValueError("samples_per_xy must be positive")

    poses: list[np.ndarray] = []
    metadata: list[dict] = []
    rot_half = np.asarray(object_rotation_half_range_deg_xyz, dtype=np.float64)
    h = halton(len(centres) * samples_per_xy, 4, start_index=start_index)
    cursor = 0
    for centre_index, xy in enumerate(centres):
        for local_index in range(samples_per_xy):
            unit = h[cursor]
            cursor += 1
            if local_index == 0:
                z_extra = 0.0
                delta_deg = np.zeros(3, dtype=np.float64)
            else:
                z_extra = float(_interval(np.asarray([unit[0]]), *z_extra_range_m)[0])
                delta_deg = _symmetric(unit[1:4], 1.0) * rot_half
            object_place = initial.copy()
            object_place[:3, 3] = np.asarray(
                [
                    float(xy[0]),
                    float(xy[1]),
                    float(table_top_world_z_m + 0.5 * nominal_object_height_m + z_extra),
                ],
                dtype=np.float64,
            )
            object_place[:3, :3] = initial[:3, :3] @ local_rpy(delta_deg)
            flange_place = object_place @ np.linalg.inv(flange_from_object)
            penalty = (
                0.4 * z_extra / max(1.0e-6, z_extra_range_m[1] - z_extra_range_m[0])
                + 0.15 * float(np.linalg.norm(delta_deg / np.maximum(rot_half, 1.0)))
            )
            poses.append(flange_place)
            metadata.append({
                "centre_index": int(centre_index),
                "object_center_world_xy_m": xy.tolist(),
                "object_center_world_z_m": float(object_place[2, 3]),
                "object_rotation_delta_deg_xyz": delta_deg.tolist(),
                "z_extra_m": float(z_extra),
                "nominal_penalty": float(penalty),
            })
    return PoseSampleSet(np.stack(poses), metadata)


def sample_transfer(
    *,
    lift_wrist_world_nominal: np.ndarray,
    place_zone_center_xy_m: np.ndarray,
    place_wrist_nominal_z_m: float,
    count: int,
    lambda_range: tuple[float, float],
    height_above_place_range_m: tuple[float, float],
    lateral_xy_half_width_m: float,
    rotation_half_range_deg_xyz: tuple[float, float, float],
    nominal_lambda: float = 0.65,
    nominal_height_above_place_m: float = 0.18,
    start_index: int = 3001,
) -> PoseSampleSet:
    """Sample a broad transfer corridor between LIFT and the placement zone.

    The first target is an explicit nominal corridor point (180 mm above the
    place-wrist reference by default), so increasing the sample count never
    removes the old/current design centre; the remaining poses fill the 6D
    interval with a deterministic low-discrepancy sequence.
    """
    lift = np.asarray(lift_wrist_world_nominal, dtype=np.float64)
    zone_xy = np.asarray(place_zone_center_xy_m, dtype=np.float64).reshape(2)
    total_count = max(1, int(count))
    sample_count = total_count - 1
    h = halton(sample_count, 7, start_index=start_index)
    lambdas = _interval(h[:, 0], *lambda_range) if sample_count else np.empty(0)
    heights = _interval(h[:, 1], *height_above_place_range_m) if sample_count else np.empty(0)
    dx = _symmetric(h[:, 2], lateral_xy_half_width_m) if sample_count else np.empty(0)
    dy = _symmetric(h[:, 3], lateral_xy_half_width_m) if sample_count else np.empty(0)
    rot_half = np.asarray(rotation_half_range_deg_xyz, dtype=np.float64)

    nominal_lambda = float(np.clip(nominal_lambda, lambda_range[0], lambda_range[1]))
    nominal_height = float(np.clip(
        nominal_height_above_place_m,
        height_above_place_range_m[0],
        height_above_place_range_m[1],
    ))
    nominal = lift.copy()
    nominal[:2, 3] = (1.0 - nominal_lambda) * lift[:2, 3] + nominal_lambda * zone_xy
    nominal[2, 3] = float(place_wrist_nominal_z_m + nominal_height)

    poses: list[np.ndarray] = [nominal]
    meta: list[dict] = [{
        "lambda": nominal_lambda,
        "height_above_place_m": nominal_height,
        "xy_offset_m": [0.0, 0.0],
        "rotation_delta_deg_xyz": [0.0, 0.0, 0.0],
        "nominal": True,
        "nominal_penalty": 0.0,
    }]
    for row in range(sample_count):
        delta_deg = _symmetric(h[row, 4:7], 1.0) * rot_half
        pose = lift.copy()
        pose[:2, 3] = (1.0 - lambdas[row]) * lift[:2, 3] + lambdas[row] * zone_xy + np.asarray([dx[row], dy[row]])
        pose[2, 3] = float(place_wrist_nominal_z_m + heights[row])
        pose[:3, :3] = lift[:3, :3] @ local_rpy(delta_deg)
        penalty = (
            0.2 * abs(float(lambdas[row]) - nominal_lambda)
            + 0.2 * (abs(float(dx[row])) + abs(float(dy[row]))) / max(1.0e-6, lateral_xy_half_width_m)
            + 0.1 * float(np.linalg.norm(delta_deg / np.maximum(rot_half, 1.0)))
            + 0.1 * abs(float(heights[row]) - nominal_height)
                / max(1.0e-6, height_above_place_range_m[1] - height_above_place_range_m[0])
        )
        poses.append(pose)
        meta.append({
            "lambda": float(lambdas[row]),
            "height_above_place_m": float(heights[row]),
            "xy_offset_m": [float(dx[row]), float(dy[row])],
            "rotation_delta_deg_xyz": delta_deg.tolist(),
            "nominal_penalty": float(penalty),
        })
    return PoseSampleSet(np.stack(poses), meta)


def sample_retreat(
    *,
    place_wrist_world: np.ndarray,
    count: int,
    upward_range_m: tuple[float, float],
    xy_half_width_m: float,
    rotation_half_range_deg_xyz: tuple[float, float, float],
    nominal_upward_m: float = 0.12,
    start_index: int = 5001,
) -> PoseSampleSet:
    """Sample a release-retreat region above/around the chosen PLACE."""
    place = np.asarray(place_wrist_world, dtype=np.float64)
    nominal = place.copy()
    nominal[2, 3] += float(nominal_upward_m)
    sample_count = max(0, int(count) - 1)
    h = halton(sample_count, 6, start_index=start_index)
    up = _interval(h[:, 0], *upward_range_m) if sample_count else np.empty(0)
    dx = _symmetric(h[:, 1], xy_half_width_m) if sample_count else np.empty(0)
    dy = _symmetric(h[:, 2], xy_half_width_m) if sample_count else np.empty(0)
    rot_half = np.asarray(rotation_half_range_deg_xyz, dtype=np.float64)

    poses: list[np.ndarray] = []
    meta: list[dict] = []
    for row in range(sample_count):
        delta_deg = _symmetric(h[row, 3:6], 1.0) * rot_half
        pose = place.copy()
        pose[:3, 3] += np.asarray([dx[row], dy[row], up[row]], dtype=np.float64)
        pose[:3, :3] = place[:3, :3] @ local_rpy(delta_deg)
        penalty = (
            abs(float(up[row]) - nominal_upward_m) / max(1.0e-6, upward_range_m[1] - upward_range_m[0])
            + 0.25 * (abs(float(dx[row])) + abs(float(dy[row]))) / max(1.0e-6, xy_half_width_m)
            + 0.1 * float(np.linalg.norm(delta_deg / np.maximum(rot_half, 1.0)))
        )
        poses.append(pose)
        meta.append({
            "upward_m": float(up[row]),
            "xy_offset_m": [float(dx[row]), float(dy[row])],
            "rotation_delta_deg_xyz": delta_deg.tolist(),
            "nominal_penalty": float(penalty),
        })
    _ensure_nominal_first(
        poses,
        meta,
        nominal,
        {
            "upward_m": float(nominal_upward_m),
            "xy_offset_m": [0.0, 0.0],
            "rotation_delta_deg_xyz": [0.0, 0.0, 0.0],
        },
    )
    return PoseSampleSet(np.stack(poses), meta)


def sample_place_from_perception_anchor(
    *,
    centres_xy_m: np.ndarray,
    target_anchor_world_initial: np.ndarray,
    flange_world_grasp: np.ndarray,
    samples_per_xy: int,
    table_top_world_z_m: float,
    nominal_object_height_m: float,
    z_extra_range_m: tuple[float, float],
    object_rotation_half_range_deg_xyz: tuple[float, float, float],
    start_index: int = 1001,
) -> PoseSampleSet:
    """
    PLACE without simulator object pose.

    The target reference is a translation-only anchor derived from the current
    visible target point cloud. The grasp flange-to-anchor transform is frozen
    and translated to each legal placement centre.
    """
    centres = np.asarray(centres_xy_m, dtype=np.float64)
    if centres.ndim != 2 or centres.shape[1] != 2:
        raise ValueError(
            f"centres_xy_m must be [N,2], got {centres.shape}"
        )
    if samples_per_xy <= 0:
        raise ValueError("samples_per_xy must be positive")

    anchor_initial = np.asarray(
        target_anchor_world_initial, dtype=np.float64
    )
    grasp_flange = np.asarray(
        flange_world_grasp, dtype=np.float64
    )
    if anchor_initial.shape != (4, 4):
        raise ValueError(
            "target_anchor_world_initial must be 4x4"
        )
    if grasp_flange.shape != (4, 4):
        raise ValueError("flange_world_grasp must be 4x4")

    flange_from_anchor = np.linalg.inv(grasp_flange) @ anchor_initial

    poses: list[np.ndarray] = []
    metadata: list[dict] = []

    rot_half = np.asarray(
        object_rotation_half_range_deg_xyz,
        dtype=np.float64,
    )
    h = halton(
        len(centres) * samples_per_xy,
        4,
        start_index=start_index,
    )

    cursor = 0
    for centre_index, xy in enumerate(centres):
        for local_index in range(samples_per_xy):
            unit = h[cursor]
            cursor += 1

            if local_index == 0:
                z_extra = 0.0
                delta_deg = np.zeros(3, dtype=np.float64)
            else:
                z_extra = float(
                    _interval(
                        np.asarray([unit[0]]),
                        *z_extra_range_m,
                    )[0]
                )
                delta_deg = (
                    _symmetric(unit[1:4], 1.0) * rot_half
                )

            anchor_place = anchor_initial.copy()
            anchor_place[:3, 3] = np.asarray(
                [
                    float(xy[0]),
                    float(xy[1]),
                    float(
                        table_top_world_z_m
                        + 0.5 * nominal_object_height_m
                        + z_extra
                    ),
                ],
                dtype=np.float64,
            )
            anchor_place[:3, :3] = (
                anchor_initial[:3, :3]
                @ local_rpy(delta_deg)
            )

            flange_place = (
                anchor_place
                @ np.linalg.inv(flange_from_anchor)
            )

            penalty = (
                0.4
                * z_extra
                / max(
                    1.0e-6,
                    z_extra_range_m[1] - z_extra_range_m[0],
                )
                + 0.15
                * float(
                    np.linalg.norm(
                        delta_deg / np.maximum(rot_half, 1.0)
                    )
                )
            )

            poses.append(flange_place)
            metadata.append(
                {
                    "centre_index": int(centre_index),
                    "target_anchor_center_world_xy_m": xy.tolist(),
                    "target_anchor_center_world_z_m":
                        float(anchor_place[2, 3]),
                    "target_anchor_rotation_delta_deg_xyz":
                        delta_deg.tolist(),
                    "z_extra_m": float(z_extra),
                    "nominal_penalty": float(penalty),
                    "geometry_source": "perception_target_anchor",
                }
            )

    return PoseSampleSet(np.stack(poses), metadata)
