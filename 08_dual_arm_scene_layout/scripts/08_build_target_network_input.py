#!/usr/bin/env python3
"""Build one official DexGraspNet 2.0 input from the latest RGB-D capture.

Inputs
------
``captures/latest/depth_m.npy``
    Full single-view depth image in metres.
``captures/latest/intrinsics.npy``
    OpenCV camera intrinsic matrix.
``captures/latest/T_world_camera.npy``
    OpenCV camera-to-world transform.
``captures/latest/grounded_sam/<target>/mask.npy``
    GroundingDINO + SAM target mask.  It is used only to identify target
    membership; the network still receives the complete SourceZone scene.

Outputs
-------
``captures/latest/dgn2/<target>/network_input.npz``
    ``pc=(1,40000,3)``, ``seg/edge=(1,40000)`` and camera extrinsics.
``sampled_scene_target.glb``
    Grey full-scene samples plus orange target samples for auditing.

The edge calculation deliberately reproduces the official implementation:
normalize depth to uint8 range 0..200, Canny(10,20), then expand five pixels.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import cv2 as cv
import numpy as np
import trimesh


PROJECT_ROOT = Path(__file__).resolve().parents[2]
LAYOUT_ROOT = PROJECT_ROOT / "08_dual_arm_scene_layout"
DEFAULT_CAPTURE_ROOT = LAYOUT_ROOT / "captures/latest"
IMPORT_REPORT = LAYOUT_ROOT / "outputs/test_scene0000_import.json"
POINT_COUNT = 40_000
RANDOM_SEED = 0
TARGET_SEGMENTATION_ID = 14


def backproject(depth_m: np.ndarray, intrinsic: np.ndarray) -> np.ndarray:
    """Return dense HxWx3 points in OpenCV optical coordinates."""

    depth = np.asarray(depth_m, dtype=np.float32)
    depth = np.where(np.isfinite(depth) & (depth > 0.0), depth, np.nan)
    height, width = depth.shape
    rows, columns = np.indices((height, width), dtype=np.float32)
    fx, fy = float(intrinsic[0, 0]), float(intrinsic[1, 1])
    cx, cy = float(intrinsic[0, 2]), float(intrinsic[1, 2])
    with np.errstate(invalid="ignore"):
        x = (columns - cx) * depth / fx
        y = (rows - cy) * depth / fy
    return np.stack((x, y, depth), axis=-1)


def official_depth_edges(depth_m: np.ndarray) -> np.ndarray:
    """Exact algorithm from official ``src/utils/edge.py`` and preprocessor."""

    finite = np.isfinite(depth_m) & (depth_m > 0.0)
    image = np.zeros(depth_m.shape, dtype=np.uint8)
    if np.any(finite):
        maximum = float(np.max(depth_m[finite]))
        image[finite] = np.clip(depth_m[finite] / maximum * 200.0, 0, 200).astype(np.uint8)
    edges = cv.Canny(image, 10, 20)
    mask = edges > 0
    mask[:, 0] = True
    mask[:, -1] = True
    mask[0, :] = True
    mask[-1, :] = True
    for _ in range(5):
        new_mask = mask.copy()
        new_mask[1:, :] |= mask[:-1, :]
        new_mask[:-1, :] |= mask[1:, :]
        new_mask[:, 1:] |= mask[:, :-1]
        new_mask[:, :-1] |= mask[:, 1:]
        mask = new_mask
    edges[mask] = 255
    return edges


def export_audit_glb(path: Path, points_world: np.ndarray, target: np.ndarray) -> None:
    colors = np.empty((len(points_world), 4), dtype=np.uint8)
    colors[:] = (150, 150, 150, 210)
    colors[target] = (255, 145, 35, 255)
    cloud = trimesh.points.PointCloud(points_world, colors=colors)
    scene = trimesh.Scene()
    scene.add_geometry(cloud, geom_name="sampled_full_scene_target_orange")
    scene.export(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", default="ashtray")
    parser.add_argument("--target-segmentation-id", type=int, default=TARGET_SEGMENTATION_ID)
    parser.add_argument(
        "--capture-root",
        type=Path,
        default=DEFAULT_CAPTURE_ROOT,
        help="Directory containing one aligned RGB-D capture.",
    )
    parser.add_argument(
        "--mask",
        type=Path,
        default=None,
        help="Optional explicit Grounded-SAM mask; defaults to grounded_sam/<target>/mask.npy.",
    )
    parser.add_argument(
        "--depth-path",
        type=Path,
        default=None,
        help=(
            "Optional aligned depth image. Route B supplies RobotSegmenter-filtered "
            "depth here so robot pixels never enter the DGN2 scene point cloud."
        ),
    )
    args = parser.parse_args()

    capture_root = args.capture_root.resolve()
    depth_path = (args.depth_path or (capture_root / "depth_m.npy")).resolve()
    intrinsic_path = capture_root / "intrinsics.npy"
    extrinsic_path = capture_root / "T_world_camera.npy"
    mask_path = (
        args.mask.resolve()
        if args.mask is not None
        else capture_root / "grounded_sam" / args.target / "mask.npy"
    )
    capture_manifest_path = capture_root / "capture_manifest.json"
    for path in (depth_path, intrinsic_path, extrinsic_path, mask_path, capture_manifest_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    depth = np.load(depth_path).astype(np.float32)
    intrinsic = np.load(intrinsic_path).astype(np.float64)
    world_from_camera = np.load(extrinsic_path).astype(np.float64)
    target_mask = np.load(mask_path).astype(bool)
    if target_mask.shape != depth.shape:
        raise ValueError(f"mask/depth shape mismatch: {target_mask.shape} vs {depth.shape}")

    dense_camera = backproject(depth, intrinsic)
    flat_camera = dense_camera.reshape(-1, 3)
    with np.errstate(invalid="ignore"):
        flat_world = flat_camera @ world_from_camera[:3, :3].T + world_from_camera[:3, 3]

    # SourceZone transform is recorded by the persistent Isaac capture path.
    # Do not use the visual SourceZone marker matrix as a rigid transform: that
    # prim is display geometry and includes scale (0.5, 0.3, 0.001).  Without a
    # settled manifest from persistent_isaac.worker.rigid_world_transform we
    # cannot prove the frame, so fail loudly instead of silently using scale.
    capture_manifest = json.loads(capture_manifest_path.read_text(encoding="utf-8"))
    settled_manifest_path = capture_manifest.get("settled_scene_manifest")
    if settled_manifest_path and Path(settled_manifest_path).is_file():
        settled_manifest = json.loads(Path(settled_manifest_path).read_text(encoding="utf-8"))
        world_from_source = np.asarray(settled_manifest["world_from_source_zone"], dtype=np.float64)
        table = settled_manifest["table"]
        source_size = np.asarray(
            table.get("size_xy_m", table["size_m"][:2]), dtype=np.float64
        )
    else:
        raise RuntimeError(
            "capture_manifest.json must contain a valid settled_scene_manifest "
            "with world_from_source_zone; refusing to derive T_world_source from "
            "SourceZone display geometry scale"
        )
    source_from_world = np.linalg.inv(world_from_source)
    with np.errstate(invalid="ignore"):
        flat_source = flat_world @ source_from_world[:3, :3].T + source_from_world[:3, 3]

    size = source_size
    finite = np.isfinite(flat_camera).all(axis=1) & (flat_camera[:, 2] > 0.0)
    # Keep the complete visible source area: tabletop plus all objects.  The
    # 30 cm vertical allowance is deliberately expressed in SourceZone frame.
    inside = (
        finite
        & (np.abs(flat_source[:, 0]) <= size[0] / 2.0 + 1.0e-4)
        & (np.abs(flat_source[:, 1]) <= size[1] / 2.0 + 1.0e-4)
        & (flat_source[:, 2] >= -0.01)
        & (flat_source[:, 2] <= 0.30)
    )
    candidates = np.flatnonzero(inside)
    if len(candidates) == 0:
        raise RuntimeError("No valid depth point lies inside SourceZone")

    rng = np.random.default_rng(RANDOM_SEED)
    selected = rng.choice(candidates, size=POINT_COUNT, replace=len(candidates) < POINT_COUNT)
    edge_image = official_depth_edges(depth)
    flat_target = target_mask.reshape(-1)
    sampled_target = flat_target[selected]
    target_sample_count = int(sampled_target.sum())
    if target_sample_count < 100:
        raise RuntimeError(
            f"Only {target_sample_count} target samples survived; target/network crop is invalid"
        )

    segmentation = np.where(sampled_target, args.target_segmentation_id, 0).astype(np.int64)
    output_root = capture_root / "dgn2" / args.target
    output_root.mkdir(parents=True, exist_ok=True)
    network_path = output_root / "network_input.npz"
    np.savez_compressed(
        network_path,
        pc=flat_camera[selected][None].astype(np.float32),
        seg=segmentation[None],
        edge=edge_image.reshape(-1)[selected][None].astype(np.int64),
        extrinsics=world_from_camera[None].astype(np.float32),
        pixel_indices=selected[None].astype(np.int64),
        target_membership=sampled_target[None],
        target_segmentation_id=np.asarray(args.target_segmentation_id, dtype=np.int64),
        source_from_world=source_from_world.astype(np.float64),
        world_from_source=world_from_source.astype(np.float64),
    )

    sampled_world = flat_world[selected]
    glb_path = output_root / "sampled_scene_target.glb"
    export_audit_glb(glb_path, sampled_world, sampled_target)
    report = {
        "schema_version": 1,
        "status": "official_network_input_ready",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "target_query": args.target,
        "capture_root": str(capture_root),
        "depth_input": str(depth_path),
        "target_mask": str(mask_path),
        "target_segmentation_id": args.target_segmentation_id,
        "point_count": POINT_COUNT,
        "random_seed": RANDOM_SEED,
        "source_crop_valid_pixel_count": int(len(candidates)),
        "sampled_target_point_count": target_sample_count,
        "sampled_target_fraction": float(target_sample_count / POINT_COUNT),
        "edge_algorithm": "official depth uint8*200 -> Canny(10,20) -> expand 5 px",
        "network_input": str(network_path),
        "audit_glb": str(glb_path),
    }
    (output_root / "network_input.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"[PASS] complete SourceZone candidates: {len(candidates)}")
    print(f"[PASS] target samples: {target_sample_count}/{POINT_COUNT}")
    print(f"[OK] {network_path}")
    print(f"[AUDIT] {glb_path}")


if __name__ == "__main__":
    main()
