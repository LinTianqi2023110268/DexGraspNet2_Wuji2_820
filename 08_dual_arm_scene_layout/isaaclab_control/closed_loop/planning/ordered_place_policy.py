from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Any

import numpy as np


@dataclass(frozen=True)
class OrderedPlaceSlot:
    slot_id: str
    column_index: int
    lane_index: int
    center_world_xy_m: tuple[float, float]
    distance_from_near_edge_m: float

    def to_jsonable(self) -> dict[str, Any]:
        return {
            "slot_id": self.slot_id,
            "column_index": int(self.column_index),
            "lane_index": int(self.lane_index),
            "center_world_xy_m": [
                float(self.center_world_xy_m[0]),
                float(self.center_world_xy_m[1]),
            ],
            "distance_from_near_edge_m": float(
                self.distance_from_near_edge_m
            ),
        }


def _axis_feasible_interval(
    *,
    center: float,
    size: float,
    object_size: float,
    edge_margin: float,
) -> tuple[float, float]:
    half_zone = 0.5 * float(size)
    half_obj = 0.5 * float(object_size)
    lower = float(center) - half_zone + half_obj + float(edge_margin)
    upper = float(center) + half_zone - half_obj - float(edge_margin)
    if upper < lower:
        raise RuntimeError(
            "nominal object footprint does not fit placement zone: "
            f"center={center}, size={size}, object={object_size}, "
            f"edge_margin={edge_margin}"
        )
    return lower, upper


def _ordered_axis_values(
    *,
    lower: float,
    upper: float,
    near_positive: bool,
    spacing: float,
) -> list[float]:
    """
    Return center coordinates from the edge nearest SourceZone to the far edge.
    """
    spacing = float(spacing)
    if spacing <= 0.0:
        raise ValueError("spacing must be positive")

    if near_positive:
        near = float(upper)
        far = float(lower)
        step = -spacing
        values = []
        value = near
        while value >= far - 1.0e-9:
            values.append(float(value))
            value += step
    else:
        near = float(lower)
        far = float(upper)
        step = spacing
        values = []
        value = near
        while value <= far + 1.0e-9:
            values.append(float(value))
            value += step

    if not values:
        values = [0.5 * (float(lower) + float(upper))]
    return values


def _transverse_lane_values(
    *,
    lower: float,
    upper: float,
    requested_slots: int,
    minimum_center_spacing_m: float,
    source_coordinate: float,
) -> list[float]:
    """
    Prefer two lanes across the short side when the width allows it.

    With the current 0.30 m-wide placement zone and 0.12 m nominal object,
    this normally yields two lanes.
    """
    lower = float(lower)
    upper = float(upper)
    count = max(1, int(requested_slots))
    center = 0.5 * (lower + upper)

    if count <= 1:
        values = [center]
    else:
        available = upper - lower
        desired = float(minimum_center_spacing_m)
        if available + 1.0e-9 < desired:
            values = [center]
        else:
            half_sep = min(0.5 * desired, 0.5 * available)
            values = [center - half_sep, center + half_sep]

    # Within the same near/far column, try the lane closer to SourceZone first.
    values.sort(key=lambda value: abs(float(value) - float(source_coordinate)))
    return [float(value) for value in values]


def ordered_near_to_far_slots(
    *,
    zone_center_world_m: Iterable[float],
    zone_size_m: Iterable[float],
    source_center_world_m: Iterable[float],
    nominal_object_size_xy_m: tuple[float, float],
    edge_margin_m: float,
    longitudinal_spacing_m: float,
    minimum_center_spacing_m: float,
    transverse_slots: int = 2,
    occupied_centres_xy_m: Iterable[Iterable[float]] = (),
) -> list[OrderedPlaceSlot]:
    """
    Build deterministic placement slots ordered for reachability.

    Policy:
      1. Determine which zone axis points most directly toward SourceZone.
      2. That axis is near->far.
      3. The perpendicular axis is the width/lane axis.
      4. Fill up to two lanes in the nearest column.
      5. Only after that column is full, advance one column farther away.
      6. Skip slots too close to already occupied centers.

    No simulator target identity is used.
    """
    zone_center = np.asarray(
        list(zone_center_world_m), dtype=np.float64
    ).reshape(3)
    zone_size = np.asarray(
        list(zone_size_m), dtype=np.float64
    ).reshape(3)
    source_center = np.asarray(
        list(source_center_world_m), dtype=np.float64
    ).reshape(3)
    object_xy = np.asarray(
        nominal_object_size_xy_m, dtype=np.float64
    ).reshape(2)

    delta = source_center[:2] - zone_center[:2]
    longitudinal_axis = int(np.argmax(np.abs(delta)))
    transverse_axis = 1 - longitudinal_axis

    feasible = [
        _axis_feasible_interval(
            center=float(zone_center[axis]),
            size=float(zone_size[axis]),
            object_size=float(object_xy[axis]),
            edge_margin=float(edge_margin_m),
        )
        for axis in (0, 1)
    ]

    # If SourceZone coordinate is above zone center along the longitudinal axis,
    # the positive bound is the near edge; otherwise the negative bound is near.
    near_positive = (
        float(source_center[longitudinal_axis])
        >= float(zone_center[longitudinal_axis])
    )

    long_values = _ordered_axis_values(
        lower=feasible[longitudinal_axis][0],
        upper=feasible[longitudinal_axis][1],
        near_positive=near_positive,
        spacing=float(longitudinal_spacing_m),
    )

    lane_values = _transverse_lane_values(
        lower=feasible[transverse_axis][0],
        upper=feasible[transverse_axis][1],
        requested_slots=int(transverse_slots),
        minimum_center_spacing_m=float(minimum_center_spacing_m),
        source_coordinate=float(source_center[transverse_axis]),
    )

    occupied = [
        np.asarray(value, dtype=np.float64).reshape(2)
        for value in occupied_centres_xy_m
    ]

    near_coordinate = float(long_values[0])
    slots: list[OrderedPlaceSlot] = []

    for column_index, long_value in enumerate(long_values):
        for lane_index, lane_value in enumerate(lane_values):
            xy = np.zeros(2, dtype=np.float64)
            xy[longitudinal_axis] = float(long_value)
            xy[transverse_axis] = float(lane_value)

            if any(
                float(np.linalg.norm(xy - used))
                < float(minimum_center_spacing_m)
                for used in occupied
            ):
                continue

            slots.append(
                OrderedPlaceSlot(
                    slot_id=(
                        f"column_{column_index:02d}_"
                        f"lane_{lane_index:02d}"
                    ),
                    column_index=int(column_index),
                    lane_index=int(lane_index),
                    center_world_xy_m=(
                        float(xy[0]),
                        float(xy[1]),
                    ),
                    distance_from_near_edge_m=float(
                        abs(float(long_value) - near_coordinate)
                    ),
                )
            )

    return slots


def build_release_place_flange_poses(
    *,
    slots: list[OrderedPlaceSlot],
    target_anchor_world_initial: np.ndarray,
    grasp_flange_world: np.ndarray,
    table_top_world_z_m: float,
    anchor_to_visible_bottom_m: float,
    release_clearance_values_m: tuple[float, ...] = (0.08, 0.10, 0.12),
    orientation_delta_deg_xyz: tuple[tuple[float, float, float], ...] = (
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 30.0),
        (0.0, 0.0, -30.0),
        (0.0, 0.0, 60.0),
        (0.0, 0.0, -60.0),
        (15.0, 0.0, 0.0),
        (-15.0, 0.0, 0.0),
        (0.0, 15.0, 0.0),
        (0.0, -15.0, 0.0),
    ),
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """
    Build relaxed release/pre-place flange targets.

    The target anchor is the perception-derived point-cloud anchor.
    The endpoint means:

        "keep the grasped target anchor over this placement slot,
         approximately 10 cm above the final support surface"

    It does NOT require preserving simulator object orientation.

    The rigid flange->anchor translation measured at grasp is preserved.
    For each allowed flange orientation, translation is solved so the target
    anchor remains over the requested release slot.
    """
    anchor_world = np.asarray(
        target_anchor_world_initial, dtype=np.float64
    )
    grasp_flange = np.asarray(
        grasp_flange_world, dtype=np.float64
    )
    if anchor_world.shape != (4, 4):
        raise ValueError("target_anchor_world_initial must be 4x4")
    if grasp_flange.shape != (4, 4):
        raise ValueError("grasp_flange_world must be 4x4")

    flange_from_anchor = np.linalg.inv(grasp_flange) @ anchor_world
    anchor_in_flange = flange_from_anchor[:3, 3].copy()
    base_rotation = grasp_flange[:3, :3].copy()

    def rot_x(a: float) -> np.ndarray:
        c, s = np.cos(a), np.sin(a)
        return np.asarray(
            [[1, 0, 0], [0, c, -s], [0, s, c]],
            dtype=np.float64,
        )

    def rot_y(a: float) -> np.ndarray:
        c, s = np.cos(a), np.sin(a)
        return np.asarray(
            [[c, 0, s], [0, 1, 0], [-s, 0, c]],
            dtype=np.float64,
        )

    def rot_z(a: float) -> np.ndarray:
        c, s = np.cos(a), np.sin(a)
        return np.asarray(
            [[c, -s, 0], [s, c, 0], [0, 0, 1]],
            dtype=np.float64,
        )

    def local_delta(delta_deg: tuple[float, float, float]) -> np.ndarray:
        rx, ry, rz = np.deg2rad(
            np.asarray(delta_deg, dtype=np.float64)
        )
        return rot_z(rz) @ rot_y(ry) @ rot_x(rx)

    poses: list[np.ndarray] = []
    metadata: list[dict[str, Any]] = []

    for slot_rank, slot in enumerate(slots):
        for clearance in release_clearance_values_m:
            # Keep the lowest visible target point approximately `clearance`
            # above the table.  This uses only point-cloud geometry.
            anchor_target = np.asarray(
                [
                    float(slot.center_world_xy_m[0]),
                    float(slot.center_world_xy_m[1]),
                    float(
                        table_top_world_z_m
                        + float(clearance)
                        + float(anchor_to_visible_bottom_m)
                    ),
                ],
                dtype=np.float64,
            )

            for delta_deg in orientation_delta_deg_xyz:
                rotation = base_rotation @ local_delta(delta_deg)
                translation = anchor_target - rotation @ anchor_in_flange

                pose = np.eye(4, dtype=np.float64)
                pose[:3, :3] = rotation
                pose[:3, 3] = translation
                poses.append(pose)

                metadata.append(
                    {
                        **slot.to_jsonable(),
                        "slot_rank": int(slot_rank),
                        "release_clearance_m": float(clearance),
                        "orientation_delta_deg_xyz": [
                            float(v) for v in delta_deg
                        ],
                        "target_anchor_world_m":
                            anchor_target.tolist(),
                        "task_semantics":
                            "release/pre-place over ordered color zone",
                        "simulator_target_identity_used": False,
                    }
                )

    if not poses:
        return np.empty((0, 4, 4), dtype=np.float64), []
    return np.stack(poses), metadata
