from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def build_perception_target_geometry(
    *,
    capture_root: Path,
    target_mask_path: Path,
    output_path: Path,
    depth_path: Path | None = None,
    minimum_valid_pixels: int = 50,
) -> dict:
    """
    Derive target geometry only from the current capture mask + RGB-D.

    No simulator target identity, mesh, pose, collision AABB, or object code
    is accepted by this function.
    """
    capture_root = Path(capture_root).resolve()
    target_mask_path = Path(target_mask_path).resolve()
    output_path = Path(output_path).resolve()

    if depth_path is None:
        depth_path = capture_root / "planning/filtered_depth.npy"
    depth_path = Path(depth_path).resolve()

    mask = np.load(target_mask_path).astype(bool)
    depth = np.load(depth_path).astype(np.float64)
    K = np.load(capture_root / "intrinsics.npy").astype(np.float64)
    T_world_camera = np.load(
        capture_root / "T_world_camera.npy"
    ).astype(np.float64)

    if mask.shape != depth.shape:
        raise ValueError(
            f"mask/depth shape mismatch: {mask.shape} vs {depth.shape}"
        )

    valid = mask & np.isfinite(depth) & (depth > 0.0)

    valid_count = int(np.count_nonzero(valid))
    if valid_count < int(minimum_valid_pixels):
        raise RuntimeError(
            "PERCEPTION_TARGET_GEOMETRY_TOO_SPARSE: "
            f"{valid_count} valid pixels"
        )

    rows, cols = np.nonzero(valid)
    z = depth[rows, cols]
    x = (
        cols.astype(np.float64) - float(K[0, 2])
    ) * z / float(K[0, 0])
    y = (
        rows.astype(np.float64) - float(K[1, 2])
    ) * z / float(K[1, 1])

    camera_points = np.stack([x, y, z], axis=1)
    world_points = (
        camera_points @ T_world_camera[:3, :3].T
        + T_world_camera[:3, 3]
    )

    centroid = np.median(world_points, axis=0)
    aabb_min = np.min(world_points, axis=0)
    aabb_max = np.max(world_points, axis=0)
    robust_min = np.quantile(world_points, 0.02, axis=0)
    robust_max = np.quantile(world_points, 0.98, axis=0)

    # Translation-only anchor.  Orientation is intentionally unknown.
    anchor = np.eye(4, dtype=np.float64)
    anchor[:3, 3] = centroid

    report = {
        "schema_version": 1,
        "source": "current perception mask + current filtered RGB-D",
        "capture_root": str(capture_root),
        "target_mask": str(target_mask_path),
        "depth": str(depth_path),
        "mask_pixels": int(np.count_nonzero(mask)),
        "valid_target_depth_pixels": valid_count,
        "centroid_world_m": centroid.tolist(),
        "aabb_world_min_m": aabb_min.tolist(),
        "aabb_world_max_m": aabb_max.tolist(),
        "robust_aabb_world_min_m": robust_min.tolist(),
        "robust_aabb_world_max_m": robust_max.tolist(),
        "visible_extent_xyz_m": (aabb_max - aabb_min).tolist(),
        "T_world_target_anchor": anchor.tolist(),
        "anchor_orientation_known": False,
        "simulator_identity_used": False,
        "object_mesh_used": False,
        "object_pose_gt_used": False,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return report
