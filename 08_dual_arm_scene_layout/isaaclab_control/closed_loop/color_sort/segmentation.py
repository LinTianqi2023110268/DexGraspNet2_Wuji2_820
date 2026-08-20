#!/usr/bin/env python3
"""HSV red/blue instance segmentation for color-sort captures.

Input is a closed-loop capture directory.  Output is written under:

  <cycle>/color_sort/

Ground truth color assignment, when present, is used only for audit metadata.
Target selection is based on rendered RGB -> HSV -> connected components.
"""

from __future__ import annotations

import argparse
from collections import deque
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
from PIL import Image


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))
from perception_target_safety import assert_current_capture_robot_mask


PROJECT_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_CONFIG = (
    PROJECT_ROOT
    / "08_dual_arm_scene_layout/isaaclab_control/closed_loop/config/color_sort.json"
)
DEFAULT_LAYOUT = PROJECT_ROOT / "08_dual_arm_scene_layout/config/manual_layout_calibrated.json"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def rgb_to_hsv01(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    arr = np.asarray(rgb, dtype=np.float32) / 255.0
    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
    maxc = np.max(arr, axis=-1)
    minc = np.min(arr, axis=-1)
    delta = maxc - minc
    hue = np.zeros_like(maxc)
    nonzero = delta > 1.0e-6
    mask = nonzero & (maxc == r)
    hue[mask] = ((g[mask] - b[mask]) / delta[mask]) % 6.0
    mask = nonzero & (maxc == g)
    hue[mask] = ((b[mask] - r[mask]) / delta[mask]) + 2.0
    mask = nonzero & (maxc == b)
    hue[mask] = ((r[mask] - g[mask]) / delta[mask]) + 4.0
    hue_deg = 60.0 * hue
    sat = np.zeros_like(maxc)
    valid = maxc > 1.0e-6
    sat[valid] = delta[valid] / maxc[valid]
    val = maxc
    return hue_deg, sat, val


def threshold_color(hue: np.ndarray, sat: np.ndarray, val: np.ndarray, spec: dict[str, Any]) -> np.ndarray:
    mask = np.zeros(hue.shape, dtype=bool)
    for lo, hi in spec["hue_ranges_deg"]:
        lo = float(lo) % 360.0
        hi = float(hi) % 360.0
        if lo <= hi:
            mask |= (hue >= lo) & (hue <= hi)
        else:
            mask |= (hue >= lo) | (hue <= hi)
    mask &= sat >= float(spec["s_min"])
    mask &= val >= float(spec["v_min"])
    return mask


def binary_erode(mask: np.ndarray) -> np.ndarray:
    m = np.asarray(mask, dtype=bool)
    out = m.copy()
    out[1:, :] &= m[:-1, :]
    out[:-1, :] &= m[1:, :]
    out[:, 1:] &= m[:, :-1]
    out[:, :-1] &= m[:, 1:]
    return out


def binary_dilate(mask: np.ndarray) -> np.ndarray:
    m = np.asarray(mask, dtype=bool)
    out = m.copy()
    out[1:, :] |= m[:-1, :]
    out[:-1, :] |= m[1:, :]
    out[:, 1:] |= m[:, :-1]
    out[:, :-1] |= m[:, 1:]
    return out


def morph(mask: np.ndarray, *, open_iterations: int, close_iterations: int) -> np.ndarray:
    out = np.asarray(mask, dtype=bool)
    for _ in range(max(0, int(open_iterations))):
        out = binary_dilate(binary_erode(out))
    for _ in range(max(0, int(close_iterations))):
        out = binary_erode(binary_dilate(out))
    return out


def connected_components(mask: np.ndarray) -> list[np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    seen = np.zeros(mask.shape, dtype=bool)
    components: list[np.ndarray] = []
    h, w = mask.shape
    ys, xs = np.nonzero(mask)
    for y0, x0 in zip(ys.tolist(), xs.tolist()):
        if seen[y0, x0]:
            continue
        q: deque[tuple[int, int]] = deque([(y0, x0)])
        seen[y0, x0] = True
        coords: list[tuple[int, int]] = []
        while q:
            y, x = q.popleft()
            coords.append((y, x))
            for yy, xx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                if 0 <= yy < h and 0 <= xx < w and mask[yy, xx] and not seen[yy, xx]:
                    seen[yy, xx] = True
                    q.append((yy, xx))
        comp = np.zeros(mask.shape, dtype=bool)
        yy, xx = np.asarray(coords, dtype=np.int64).T
        comp[yy, xx] = True
        components.append(comp)
    return components


def backproject(depth_m: np.ndarray, intrinsic: np.ndarray) -> np.ndarray:
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


def source_zone_bounds(layout: dict[str, Any], z_tolerance_m: float) -> tuple[np.ndarray, np.ndarray]:
    center = np.asarray(layout["transforms"]["source_zone"]["position_world_m"], dtype=np.float64)
    size = np.asarray(layout["geometry"]["source_zone_size_m"], dtype=np.float64)
    lower = center - 0.5 * size
    upper = center + 0.5 * size
    lower[2] -= float(z_tolerance_m)
    upper[2] += float(z_tolerance_m)
    return lower, upper


def summarize_component(
    *,
    component: np.ndarray,
    depth: np.ndarray,
    world_points: np.ndarray,
    source_lower: np.ndarray,
    source_upper: np.ndarray,
) -> dict[str, Any]:
    ys, xs = np.nonzero(component)
    area = int(len(xs))
    valid = component & np.isfinite(depth) & (depth > 0.0)
    valid_count = int(valid.sum())
    valid_fraction = 0.0 if area <= 0 else valid_count / area
    bbox = [int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)]
    centroid_uv = [float(xs.mean()), float(ys.mean())]
    centroid_world = None
    inside_source = False
    if valid_count > 0:
        pts = world_points[valid]
        finite = np.all(np.isfinite(pts), axis=1)
        pts = pts[finite]
        if len(pts):
            centroid = np.median(pts, axis=0)
            centroid_world = centroid.tolist()
            inside_source = bool(np.all(centroid >= source_lower) and np.all(centroid <= source_upper))
    return {
        "area_px": area,
        "bbox_xyxy": bbox,
        "centroid_uv": centroid_uv,
        "valid_depth_fraction": float(valid_fraction),
        "centroid_world_m": centroid_world,
        "inside_source_zone": bool(inside_source),
    }


def overlay_image(rgb: np.ndarray, red_mask: np.ndarray, blue_mask: np.ndarray) -> np.ndarray:
    out = np.asarray(rgb, dtype=np.float32).copy()
    red = np.asarray([255.0, 35.0, 35.0], dtype=np.float32)
    blue = np.asarray([35.0, 75.0, 255.0], dtype=np.float32)
    out[red_mask] = 0.55 * out[red_mask] + 0.45 * red
    out[blue_mask] = 0.55 * out[blue_mask] + 0.45 * blue
    return np.clip(out, 0, 255).astype(np.uint8)


def run_color_segmentation(
    *,
    capture_root: Path,
    output_root: Path,
    config_path: Path = DEFAULT_CONFIG,
    layout_path: Path = DEFAULT_LAYOUT,
    assignment_path: Path | None = None,
    exclude_instance_ids: set[str] | None = None,
    preferred_color: str | None = None,
) -> dict[str, Any]:
    capture_root = Path(capture_root).resolve()
    output_root = Path(output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    cfg = load_json(config_path)
    layout = load_json(layout_path)
    if preferred_color is not None and preferred_color not in {"red", "blue"}:
        raise ValueError(f"preferred_color must be red/blue, got {preferred_color!r}")

    rgb = np.asarray(Image.open(capture_root / "rgb.png").convert("RGB"), dtype=np.uint8)
    depth = np.load(capture_root / "depth_m.npy").astype(np.float32)
    robot_mask_path = capture_root / "planning/robot_mask.npy"
    robot_report_path = capture_root / "planning/robot_segmentation_report.json"
    if not robot_mask_path.is_file() or not robot_report_path.is_file():
        raise FileNotFoundError("RobotSegmenter current-cycle robot mask/report is required for color-sort")
    robot_mask = np.load(robot_mask_path).astype(bool)
    robot_report = load_json(robot_report_path)
    assert_current_capture_robot_mask(
        robot_report_capture_dir=robot_report["capture_dir"],
        capture_dir=capture_root,
    )
    intrinsic = np.load(capture_root / "intrinsics.npy").astype(np.float64)
    world_from_camera = np.load(capture_root / "T_world_camera.npy").astype(np.float64)
    if depth.shape != rgb.shape[:2]:
        raise ValueError(f"RGB/depth shape mismatch: {rgb.shape[:2]} vs {depth.shape}")
    if robot_mask.shape != depth.shape:
        raise ValueError(f"STALE_ROBOT_MASK: robot mask/depth shape mismatch: {robot_mask.shape} vs {depth.shape}")

    camera_points = backproject(depth, intrinsic)
    flat_camera = camera_points.reshape(-1, 3)
    with np.errstate(invalid="ignore"):
        flat_world = flat_camera @ world_from_camera[:3, :3].T + world_from_camera[:3, 3]
    world_points = flat_world.reshape((*depth.shape, 3))
    source_lower, source_upper = source_zone_bounds(
        layout, cfg["source_zone"]["z_tolerance_m"]
    )

    hue, sat, val = rgb_to_hsv01(rgb)
    morph_cfg = cfg["morphology"]
    comp_cfg = cfg["components"]
    all_instances: list[dict[str, Any]] = []
    color_reports: dict[str, Any] = {}
    instances_dir = output_root / "instances"
    instances_dir.mkdir(exist_ok=True)
    masks_for_overlay: dict[str, np.ndarray] = {}

    for color in ("red", "blue"):
        raw_mask = threshold_color(hue, sat, val, cfg["hsv"][color]) & ~robot_mask
        mask = morph(
            raw_mask,
            open_iterations=int(morph_cfg["open_iterations"]),
            close_iterations=int(morph_cfg["close_iterations"]),
        )
        masks_for_overlay[color] = mask
        np.save(output_root / f"{color}_mask_all.npy", mask)
        Image.fromarray((mask.astype(np.uint8) * 255), mode="L").save(output_root / f"{color}_mask_all.png")

        instances = []
        for component in connected_components(mask):
            summary = summarize_component(
                component=component,
                depth=depth,
                world_points=world_points,
                source_lower=source_lower,
                source_upper=source_upper,
            )
            if summary["area_px"] < int(comp_cfg["min_area_px"]):
                continue
            if summary["valid_depth_fraction"] < float(comp_cfg["min_valid_depth_fraction"]):
                continue
            instance_id = f"{color}_{len(instances):03d}"
            mask_path = instances_dir / f"{instance_id}_mask.npy"
            png_path = instances_dir / f"{instance_id}_mask.png"
            np.save(mask_path, component)
            Image.fromarray((component.astype(np.uint8) * 255), mode="L").save(png_path)
            row = {
                "instance_id": instance_id,
                "color": color,
                "mask_path": str(mask_path),
                "mask_png": str(png_path),
                **summary,
            }
            instances.append(row)
            all_instances.append(row)
        color_reports[color] = {
            "visible_components": int(len(instances)),
            "inside_source_zone": int(sum(1 for item in instances if item["inside_source_zone"])),
            "instances": instances,
        }

    overlay = overlay_image(rgb, masks_for_overlay["red"], masks_for_overlay["blue"])
    overlay_path = output_root / "overlay.png"
    Image.fromarray(overlay, mode="RGB").save(overlay_path)

    excluded = set(exclude_instance_ids or set())
    selected = None
    selection_order = (preferred_color,) if preferred_color is not None else ("red", "blue")
    for color in selection_order:
        candidates = [
            item for item in all_instances
            if (
                item["color"] == color
                and item["inside_source_zone"]
                and str(item["instance_id"]) not in excluded
            )
        ]
        candidates.sort(
            key=lambda item: (
                -float(item["valid_depth_fraction"]),
                -int(item["area_px"]),
                float(item["centroid_uv"][1]),
            )
        )
        if candidates:
            selected = candidates[0]
            break

    assignment = None
    if assignment_path is not None and Path(assignment_path).is_file():
        assignment = str(Path(assignment_path).resolve())
    report = {
        "schema_version": 1,
        "task": "color-sort",
        "capture_root": str(capture_root),
        "config": str(Path(config_path).resolve()),
        "layout": str(Path(layout_path).resolve()),
        "assignment": assignment,
        "preferred_color": preferred_color,
        "robot_mask": str(robot_mask_path),
        "robot_mask_capture_dir": str(capture_root),
        "robot_exclusion": "ON",
        "stale_mask_check": "PASS",
        "excluded_instance_ids": sorted(excluded),
        "source_zone_bounds_world_m": {
            "min": source_lower.tolist(),
            "max": source_upper.tolist(),
        },
        "red": color_reports["red"],
        "blue": color_reports["blue"],
        "selected": selected,
        "overlay": str(overlay_path),
    }
    report_path = output_root / "detection_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if selected is not None:
        (output_root / "selected_target.json").write_text(
            json.dumps({
                "schema_version": 1,
                "task_type": "color-sort",
                "target_source": "hsv_instance",
                "target_mask_path": selected["mask_path"],
                "query_original": None,
                "query_canonical": selected["color"],
                "color": selected["color"],
                "instance_id": selected["instance_id"],
                "bbox": selected["bbox_xyxy"],
                "centroid_uv": selected["centroid_uv"],
                "centroid_world": selected["centroid_world_m"],
                "placement_zone_override": f"{selected['color']}_zone",
                "metadata": selected,
            }, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--layout", type=Path, default=DEFAULT_LAYOUT)
    parser.add_argument("--assignment", type=Path, default=None)
    parser.add_argument("--exclude-instance-id", action="append", default=[])
    parser.add_argument("--preferred-color", choices=["red", "blue"], default=None)
    args = parser.parse_args()
    report = run_color_segmentation(
        capture_root=args.capture_root,
        output_root=args.output_root,
        config_path=args.config,
        layout_path=args.layout,
        assignment_path=args.assignment,
        exclude_instance_ids={str(value) for value in args.exclude_instance_id},
        preferred_color=args.preferred_color,
    )
    print(
        "[COLOR SORT][HSV] "
        f"red source={report['red']['inside_source_zone']} "
        f"blue source={report['blue']['inside_source_zone']} "
        f"selected={(report['selected'] or {}).get('instance_id')}",
        flush=True,
    )
    print(f"[COLOR SORT][HSV] report={Path(args.output_root).resolve() / 'detection_report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
