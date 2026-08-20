"""Final semantic-target safety gate.

This module does not create a second robot segmentation method.  Its only
robot input is the current capture's authoritative cuRobo RobotSegmenter mask.
GroundingDINO proposals remain advisory until this gate accepts a SAM mask.
"""
from __future__ import annotations

from typing import Any
from pathlib import Path

import numpy as np


# Evidence: 12 historical accepted semantic masks had 0.0 robot overlap;
# current robot false positives measured 0.781--1.000.  This threshold is for
# a mask that is predominantly robot.  Smaller overlap is removed from the
# target mask and the residual must still pass the 3D SourceZone gate.
ROBOT_DOMINANT_OVERLAP_FRACTION = 0.5


def assert_current_capture_robot_mask(*, robot_report_capture_dir: str | Path, capture_dir: str | Path) -> None:
    if Path(robot_report_capture_dir).resolve() != Path(capture_dir).resolve():
        raise RuntimeError("STALE_ROBOT_MASK: robot_mask capture_dir does not match RGB capture")


def evaluate_dino_box(*, xyxy: np.ndarray, robot_mask: np.ndarray) -> dict[str, Any]:
    """Coarse hard gate before SAM; uses the same mask and threshold as SAM."""
    mask = np.asarray(robot_mask, dtype=bool)
    height, width = mask.shape
    x1, y1, x2, y2 = np.asarray(xyxy, dtype=np.float64)
    x1 = max(0, min(width, int(np.floor(x1))))
    y1 = max(0, min(height, int(np.floor(y1))))
    x2 = max(0, min(width, int(np.ceil(x2))))
    y2 = max(0, min(height, int(np.ceil(y2))))
    area = max(0, (x2 - x1) * (y2 - y1))
    overlap_px = int(mask[y1:y2, x1:x2].sum()) if area else 0
    overlap_fraction = float(overlap_px / max(1, area))
    reject = overlap_fraction > ROBOT_DOMINANT_OVERLAP_FRACTION
    return {
        "robot_overlap_px_box": overlap_px,
        "robot_overlap_fraction_box": overlap_fraction,
        "dino_box_legal": not reject,
        "dino_box_reject_reason": "REJECT_ROBOT_OVERLAP" if reject else None,
    }


def source_zone_membership(
    *,
    depth_m: np.ndarray,
    intrinsics: np.ndarray,
    T_world_camera: np.ndarray,
    T_world_source_zone: np.ndarray,
    source_zone_size_xy_m: np.ndarray,
    source_zone_z_bounds_m: tuple[float, float] = (-0.01, 0.30),
) -> tuple[np.ndarray, np.ndarray]:
    """Return valid-depth and valid points in the rigid SourceZone volume."""
    depth = np.asarray(depth_m, dtype=np.float64)
    K = np.asarray(intrinsics, dtype=np.float64)
    T_wc = np.asarray(T_world_camera, dtype=np.float64)
    T_sw = np.linalg.inv(np.asarray(T_world_source_zone, dtype=np.float64))
    size_xy = np.asarray(source_zone_size_xy_m, dtype=np.float64).reshape(2)
    rows, cols = np.indices(depth.shape, dtype=np.float64)
    with np.errstate(invalid="ignore"):
        camera = np.stack(
            ((cols - K[0, 2]) * depth / K[0, 0], (rows - K[1, 2]) * depth / K[1, 1], depth),
            axis=-1,
        )
        world = camera @ T_wc[:3, :3].T + T_wc[:3, 3]
        source = world @ T_sw[:3, :3].T + T_sw[:3, 3]
    valid = np.isfinite(depth) & (depth > 0.0)
    inside = (
        valid
        & (np.abs(source[..., 0]) <= 0.5 * size_xy[0] + 1.0e-4)
        & (np.abs(source[..., 1]) <= 0.5 * size_xy[1] + 1.0e-4)
        & (source[..., 2] >= float(source_zone_z_bounds_m[0]))
        & (source[..., 2] <= float(source_zone_z_bounds_m[1]))
    )
    return valid, inside


def evaluate_sam_proposal(
    *,
    sam_mask: np.ndarray,
    robot_mask: np.ndarray,
    valid_depth: np.ndarray,
    inside_source_zone: np.ndarray,
) -> tuple[dict[str, Any], np.ndarray]:
    """Return auditable hard-gate decision and the non-robot target residual."""
    sam = np.asarray(sam_mask, dtype=bool)
    robot = np.asarray(robot_mask, dtype=bool)
    valid = np.asarray(valid_depth, dtype=bool)
    source = np.asarray(inside_source_zone, dtype=bool)
    if sam.shape != robot.shape or sam.shape != valid.shape or sam.shape != source.shape:
        raise ValueError("SAM/robot/depth/SourceZone masks must have identical HxW shape")
    raw_area = int(np.count_nonzero(sam))
    robot_px = int(np.count_nonzero(sam & robot))
    robot_fraction = float(robot_px / max(1, raw_area))
    residual = sam & ~robot
    residual_valid = residual & valid
    source_valid = residual_valid & source
    valid_fraction = float(np.count_nonzero(residual_valid) / max(1, np.count_nonzero(residual)))
    source_fraction = float(np.count_nonzero(source_valid) / max(1, np.count_nonzero(residual_valid)))
    if robot_fraction > ROBOT_DOMINANT_OVERLAP_FRACTION:
        reject_reason = "REJECT_ROBOT_OVERLAP"
    elif not np.any(residual_valid):
        reject_reason = "REJECT_NO_VALID_DEPTH"
    elif source_fraction < 0.5:
        reject_reason = "REJECT_OUTSIDE_SOURCE_ZONE"
    else:
        reject_reason = None
    return {
        "robot_overlap_px": robot_px,
        "robot_overlap_fraction": robot_fraction,
        "valid_depth_fraction": valid_fraction,
        "source_zone_overlap_fraction": source_fraction,
        "residual_mask_px": int(np.count_nonzero(residual)),
        "legal": reject_reason is None,
        "reject_reason": reject_reason,
    }, residual


def select_legal_proposal(rows: list[dict[str, Any]]) -> int | None:
    legal = [row for row in rows if bool(row.get("legal", False))]
    if not legal:
        return None
    legal.sort(key=lambda row: (-float(row["score"]), int(row["idx"])))
    return int(legal[0]["idx"])
