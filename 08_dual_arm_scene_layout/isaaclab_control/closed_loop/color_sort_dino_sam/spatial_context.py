from __future__ import annotations

from dataclasses import dataclass, asdict
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class SpatialContextArtifacts:
    source_zone_depth_mask: str
    source_zone_rgb_mask: str
    rgb_source_only: str
    grasp_context_depth_mask: str
    filtered_depth_grasp_context: str
    report: str

    def to_jsonable(self) -> dict[str, Any]:
        return asdict(self)


def _backproject_world(
    depth_m: np.ndarray,
    K: np.ndarray,
    T_world_camera: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    depth = np.asarray(depth_m, dtype=np.float64)
    rows, cols = np.indices(depth.shape, dtype=np.float64)
    valid = np.isfinite(depth) & (depth > 0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        x = (cols - float(K[0, 2])) * depth / float(K[0, 0])
        y = (rows - float(K[1, 2])) * depth / float(K[1, 1])
    camera = np.stack((x, y, depth), axis=-1)
    world = camera @ np.asarray(T_world_camera, dtype=np.float64)[:3, :3].T
    world += np.asarray(T_world_camera, dtype=np.float64)[:3, 3]
    return world, valid


def _filled_source_rgb_mask(source_depth_mask: np.ndarray) -> np.ndarray:
    """Fill SourceZone RGB support so invalid-depth highlights are not punched out."""
    mask = np.asarray(source_depth_mask, dtype=bool)
    ys, xs = np.nonzero(mask)
    if len(xs) < 4:
        raise RuntimeError(
            f"SOURCE_ZONE_RGB_MASK_TOO_SPARSE: valid source pixels={len(xs)}"
        )
    try:
        import cv2
        points = np.stack((xs, ys), axis=1).astype(np.int32)
        hull = cv2.convexHull(points)
        out = np.zeros(mask.shape, dtype=np.uint8)
        cv2.fillConvexPoly(out, hull, 1)
        return out.astype(bool)
    except Exception:
        out = np.zeros_like(mask, dtype=bool)
        out[int(ys.min()):int(ys.max()) + 1, int(xs.min()):int(xs.max()) + 1] = True
        return out


def build_spatial_context(
    *,
    capture_root: Path,
    filtered_depth_path: Path,
    rgb_no_robot_path: Path,
    layout_path: Path,
    output_dir: Path,
    context_margin_xy_m: float = 0.20,
    source_z_bounds_m: tuple[float, float] = (-0.01, 0.30),
    outside_rgb_value: int = 127,
) -> SpatialContextArtifacts:
    """Build SourceZone target support and enlarged DGN2 context support."""
    capture_root = Path(capture_root).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    depth = np.load(Path(filtered_depth_path).resolve()).astype(np.float32)
    rgb = np.asarray(
        Image.open(Path(rgb_no_robot_path).resolve()).convert("RGB"),
        dtype=np.uint8,
    )
    K = np.load(capture_root / "intrinsics.npy").astype(np.float64)
    T_world_camera = np.load(capture_root / "T_world_camera.npy").astype(np.float64)
    if depth.shape != rgb.shape[:2]:
        raise ValueError(f"RGB/depth shape mismatch: {rgb.shape[:2]} vs {depth.shape}")

    settled_path = capture_root / "settled_scene_manifest.json"
    if not settled_path.is_file():
        raise FileNotFoundError(settled_path)
    settled = json.loads(settled_path.read_text(encoding="utf-8"))

    layout = json.loads(Path(layout_path).resolve().read_text(encoding="utf-8"))
    source_size_xy = np.asarray(
        layout["geometry"]["source_zone_size_m"][:2], dtype=np.float64
    )
    T_world_source = np.asarray(settled["world_from_source_zone"], dtype=np.float64)
    if T_world_source.shape != (4, 4):
        raise RuntimeError(
            f"invalid world_from_source_zone shape: {T_world_source.shape}"
        )
    T_source_world = np.linalg.inv(T_world_source)

    world, valid = _backproject_world(depth, K, T_world_camera)
    source = world @ T_source_world[:3, :3].T + T_source_world[:3, 3]

    z_lo, z_hi = [float(v) for v in source_z_bounds_m]
    half_source = 0.5 * source_size_xy

    source_depth_mask = (
        valid
        & (np.abs(source[..., 0]) <= half_source[0] + 1.0e-4)
        & (np.abs(source[..., 1]) <= half_source[1] + 1.0e-4)
        & (source[..., 2] >= z_lo)
        & (source[..., 2] <= z_hi)
    )

    margin = float(context_margin_xy_m)
    if margin < 0.0:
        raise ValueError("context_margin_xy_m must be non-negative")
    half_context = half_source + margin
    grasp_context_mask = (
        valid
        & (np.abs(source[..., 0]) <= half_context[0] + 1.0e-4)
        & (np.abs(source[..., 1]) <= half_context[1] + 1.0e-4)
        & (source[..., 2] >= z_lo)
        & (source[..., 2] <= z_hi)
    )

    if int(np.count_nonzero(source_depth_mask)) < 100:
        raise RuntimeError(
            "SOURCE_ZONE_DEPTH_SUPPORT_TOO_SPARSE: "
            f"{int(np.count_nonzero(source_depth_mask))} pixels"
        )
    if int(np.count_nonzero(grasp_context_mask)) < 100:
        raise RuntimeError(
            "GRASP_CONTEXT_DEPTH_SUPPORT_TOO_SPARSE: "
            f"{int(np.count_nonzero(grasp_context_mask))} pixels"
        )

    source_rgb_mask = _filled_source_rgb_mask(source_depth_mask)
    rgb_source = rgb.copy()
    rgb_source[~source_rgb_mask] = np.uint8(np.clip(outside_rgb_value, 0, 255))

    context_depth = depth.copy()
    context_depth[~grasp_context_mask] = 0.0

    source_depth_path = output_dir / "source_zone_depth_mask.npy"
    source_rgb_mask_path = output_dir / "source_zone_rgb_mask.npy"
    rgb_source_path = output_dir / "rgb_source_only.png"
    context_mask_path = output_dir / "grasp_context_depth_mask.npy"
    context_depth_path = output_dir / "filtered_depth_grasp_context.npy"
    report_path = output_dir / "spatial_context.json"

    np.save(source_depth_path, source_depth_mask)
    np.save(source_rgb_mask_path, source_rgb_mask)
    np.save(context_mask_path, grasp_context_mask)
    np.save(context_depth_path, context_depth)

    Image.fromarray((source_depth_mask.astype(np.uint8) * 255), mode="L").save(
        output_dir / "source_zone_depth_mask.png"
    )
    Image.fromarray((source_rgb_mask.astype(np.uint8) * 255), mode="L").save(
        output_dir / "source_zone_rgb_mask.png"
    )
    Image.fromarray((grasp_context_mask.astype(np.uint8) * 255), mode="L").save(
        output_dir / "grasp_context_depth_mask.png"
    )
    Image.fromarray(rgb_source, mode="RGB").save(rgb_source_path)

    report = {
        "schema_version": 1,
        "contract": {
            "target_eligibility_zone": "SourceZone",
            "dgn2_context_zone": "SourceZone enlarged in XY",
            "robot_pixels_removed_upstream": True,
            "rgb_perception_scope": "SourceZone only",
        },
        "capture_root": str(capture_root),
        "filtered_depth": str(Path(filtered_depth_path).resolve()),
        "rgb_no_robot": str(Path(rgb_no_robot_path).resolve()),
        "layout": str(Path(layout_path).resolve()),
        "source_zone_size_xy_m": source_size_xy.tolist(),
        "context_margin_xy_m": margin,
        "context_size_xy_m": (source_size_xy + 2.0 * margin).tolist(),
        "source_z_bounds_m": [z_lo, z_hi],
        "source_zone_valid_depth_pixels": int(np.count_nonzero(source_depth_mask)),
        "grasp_context_valid_depth_pixels": int(np.count_nonzero(grasp_context_mask)),
        "source_zone_depth_mask": str(source_depth_path),
        "source_zone_rgb_mask": str(source_rgb_mask_path),
        "rgb_source_only": str(rgb_source_path),
        "grasp_context_depth_mask": str(context_mask_path),
        "filtered_depth_grasp_context": str(context_depth_path),
    }
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    return SpatialContextArtifacts(
        source_zone_depth_mask=str(source_depth_path),
        source_zone_rgb_mask=str(source_rgb_mask_path),
        rgb_source_only=str(rgb_source_path),
        grasp_context_depth_mask=str(context_mask_path),
        filtered_depth_grasp_context=str(context_depth_path),
        report=str(report_path),
    )
