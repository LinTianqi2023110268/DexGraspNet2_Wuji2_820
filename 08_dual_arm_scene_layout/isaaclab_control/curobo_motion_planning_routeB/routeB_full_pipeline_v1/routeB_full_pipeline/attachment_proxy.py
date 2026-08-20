from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class TargetProxy:
    target_segmentation_id: int
    center_base_m: np.ndarray
    dims_base_m: np.ndarray
    pose_base_wxyz: np.ndarray
    source: str
    source_aabb_world_min_m: np.ndarray
    source_aabb_world_max_m: np.ndarray

    def to_jsonable(self) -> dict[str, Any]:
        return {
            "target_segmentation_id": int(self.target_segmentation_id),
            "center_base_m": self.center_base_m.tolist(),
            "dims_base_m": self.dims_base_m.tolist(),
            "pose_base_wxyz": self.pose_base_wxyz.tolist(),
            "source": self.source,
            "source_aabb_world_min_m": self.source_aabb_world_min_m.tolist(),
            "source_aabb_world_max_m": self.source_aabb_world_max_m.tolist(),
        }


def _load_json(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _world_from_base(project_root: Path) -> np.ndarray:
    layout = _load_json(
        Path(project_root)
        / "08_dual_arm_scene_layout/config/manual_layout_calibrated.json"
    )
    return np.asarray(
        layout["transforms"]["dual_arm_mount"]["Gf_local_to_world_row_major"],
        dtype=np.float64,
    ).T


def _aabb_corners(lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    lo = np.asarray(lower, dtype=np.float64).reshape(3)
    hi = np.asarray(upper, dtype=np.float64).reshape(3)
    return np.asarray(
        [
            [x, y, z]
            for x in (lo[0], hi[0])
            for y in (lo[1], hi[1])
            for z in (lo[2], hi[2])
        ],
        dtype=np.float64,
    )


def _transform_points(T: np.ndarray, points: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    ph = np.concatenate(
        [pts, np.ones((len(pts), 1), dtype=np.float64)], axis=1
    )
    return (np.asarray(T, dtype=np.float64) @ ph.T).T[:, :3]


def _is_valid_aabb(lower: np.ndarray, upper: np.ndarray) -> bool:
    lower = np.asarray(lower, dtype=np.float64).reshape(3)
    upper = np.asarray(upper, dtype=np.float64).reshape(3)
    return bool(
        np.all(np.isfinite(lower))
        and np.all(np.isfinite(upper))
        and np.all(upper > lower)
        and np.all(np.abs(lower) < 100.0)
        and np.all(np.abs(upper) < 100.0)
    )


def _target_aabb_from_mask_depth(
    *,
    project_root: Path,
    capture_dir: Path,
    target_mask_path: Path | None,
    padding_m: float,
    minimum_dim_m: float,
) -> TargetProxy:
    if target_mask_path is None:
        mask_candidates = sorted((capture_dir / "grounded_sam").glob("*/mask.npy"))
        if not mask_candidates:
            raise FileNotFoundError(
                f"no GroundedSAM mask found under {capture_dir / 'grounded_sam'}"
            )
        target_mask_path = mask_candidates[0]
    target_mask_path = Path(target_mask_path).resolve()
    mask = np.load(target_mask_path).astype(bool)
    depth = np.load(capture_dir / "depth_m.npy").astype(np.float64)
    K = np.load(capture_dir / "intrinsics.npy").astype(np.float64)
    T_world_camera = np.load(capture_dir / "T_world_camera.npy").astype(np.float64)
    if mask.shape != depth.shape:
        raise ValueError(f"mask/depth shape mismatch: {mask.shape} vs {depth.shape}")
    valid = mask & np.isfinite(depth) & (depth > 0.0)
    if int(np.count_nonzero(valid)) < 8:
        raise RuntimeError("target mask/depth fallback has too few valid pixels")

    ys, xs = np.nonzero(valid)
    z = depth[ys, xs]
    x = (xs.astype(np.float64) - float(K[0, 2])) * z / float(K[0, 0])
    y = (ys.astype(np.float64) - float(K[1, 2])) * z / float(K[1, 1])
    points_camera = np.stack([x, y, z], axis=1)
    points_world = _transform_points(T_world_camera, points_camera)
    T_base_world = np.linalg.inv(_world_from_base(Path(project_root)))
    points_base = _transform_points(T_base_world, points_world)
    lo_b = points_base.min(axis=0)
    hi_b = points_base.max(axis=0)
    center_b = 0.5 * (lo_b + hi_b)
    dims_b = np.maximum(
        (hi_b - lo_b) + 2.0 * float(padding_m), float(minimum_dim_m)
    )
    pose = np.asarray(
        [center_b[0], center_b[1], center_b[2], 1.0, 0.0, 0.0, 0.0],
        dtype=np.float64,
    )
    return TargetProxy(
        target_segmentation_id=-1,
        center_base_m=center_b,
        dims_base_m=dims_b,
        pose_base_wxyz=pose,
        source=str(target_mask_path),
        source_aabb_world_min_m=points_world.min(axis=0),
        source_aabb_world_max_m=points_world.max(axis=0),
    )


def build_target_proxy_from_capture(
    *,
    project_root: Path,
    capture_dir: Path,
    target_segmentation_id: int,
    target_mask_path: Path | None = None,
    padding_m: float = 0.005,
    minimum_dim_m: float = 0.02,
) -> TargetProxy:
    """Build a planner proxy from Isaac's exact collision AABB audit.

    The persistent Isaac capture already writes object_physics_audit.json from
    the actual simulation collision prims.  Reusing it is stronger than trying
    to infer unseen object extent from a single-view RGB-D mask.

    The world AABB's 8 corners are transformed to arm_base_link and enclosed by
    a base-frame AABB.  This can be rigidly attached to the flange during carry.
    """
    capture_dir = Path(capture_dir).resolve()
    audit_path = capture_dir / "object_physics_audit.json"
    if not audit_path.is_file():
        raise FileNotFoundError(audit_path)
    audit = _load_json(audit_path)
    row = next(
        (
            x
            for x in audit.get("objects", [])
            if int(x["segmentation_id"]) == int(target_segmentation_id)
        ),
        None,
    )
    if row is None:
        raise KeyError(
            f"target segmentation id {target_segmentation_id} missing from {audit_path}"
        )
    lo_w = np.asarray(row["collision_aabb_world_min_m"], dtype=np.float64)
    hi_w = np.asarray(row["collision_aabb_world_max_m"], dtype=np.float64)
    if lo_w.shape != (3,) or hi_w.shape != (3,):
        raise RuntimeError("invalid Isaac collision AABB")
    if not _is_valid_aabb(lo_w, hi_w):
        fallback = _target_aabb_from_mask_depth(
            project_root=project_root,
            capture_dir=capture_dir,
            target_mask_path=target_mask_path,
            padding_m=padding_m,
            minimum_dim_m=minimum_dim_m,
        )
        return TargetProxy(
            target_segmentation_id=int(target_segmentation_id),
            center_base_m=fallback.center_base_m,
            dims_base_m=fallback.dims_base_m,
            pose_base_wxyz=fallback.pose_base_wxyz,
            source=(
                "fallback_mask_depth_due_invalid_isaac_collision_aabb:"
                f"{fallback.source}"
            ),
            source_aabb_world_min_m=fallback.source_aabb_world_min_m,
            source_aabb_world_max_m=fallback.source_aabb_world_max_m,
        )

    T_base_world = np.linalg.inv(_world_from_base(Path(project_root)))
    corners_b = _transform_points(T_base_world, _aabb_corners(lo_w, hi_w))
    lo_b = corners_b.min(axis=0)
    hi_b = corners_b.max(axis=0)
    center_b = 0.5 * (lo_b + hi_b)
    dims_b = np.maximum(
        (hi_b - lo_b) + 2.0 * float(padding_m), float(minimum_dim_m)
    )
    pose = np.asarray(
        [center_b[0], center_b[1], center_b[2], 1.0, 0.0, 0.0, 0.0],
        dtype=np.float64,
    )
    return TargetProxy(
        target_segmentation_id=int(target_segmentation_id),
        center_base_m=center_b,
        dims_base_m=dims_b,
        pose_base_wxyz=pose,
        source=str(audit_path),
        source_aabb_world_min_m=lo_w,
        source_aabb_world_max_m=hi_w,
    )


def remove_target_mask_from_filtered_depth(
    *,
    filtered_depth_path: Path,
    target_mask_path: Path,
    output_path: Path,
) -> Path:
    """Remove only target pixels from RobotSegmenter-cleaned depth.

    This is required for intentional PREGRASP->COVER contact and for carry,
    where the target is represented separately as attached collision geometry.
    """
    depth = np.load(filtered_depth_path).astype(np.float32, copy=True)
    mask = np.load(target_mask_path).astype(bool)
    if depth.shape != mask.shape:
        raise ValueError(
            f"filtered depth / target mask mismatch: {depth.shape} vs {mask.shape}"
        )
    depth[mask] = 0.0
    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, depth)
    meta = {
        "schema_version": 1,
        "source_filtered_depth": str(Path(filtered_depth_path).resolve()),
        "target_mask": str(Path(target_mask_path).resolve()),
        "target_pixels_removed": int(np.count_nonzero(mask)),
        "intrinsics_unchanged": True,
        "T_world_camera_unchanged": True,
        "purpose": "Route B non-target ESDF for approach/carry",
    }
    output_path.with_suffix(".json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return output_path
