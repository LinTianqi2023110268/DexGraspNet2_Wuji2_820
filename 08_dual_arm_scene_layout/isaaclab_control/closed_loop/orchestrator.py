#!/usr/bin/env python3
"""One-command persistent semantic dexterous grasp loop.

V2 architecture
---------------
* Isaac Lab/Sim starts once and keeps the same physical world for every capture
  and every grasp cycle.
* cuRobo starts once per candidate batch and is released before Isaac execution.
* legacy approximate GRASP/PREGRASP coarse IK gates are configurable and OFF by
  default.
* after LEAP->Wuji2, exact COVER is the hard grasp-root IK gate.
* PREGRASP/LIFT/TRANSFER/PLACE/RETREAT use large configurable 6D task sets;
  strict IK accuracy is unchanged.
* the q7 route produced by planning is executed directly in the same Isaac
  world: no second runtime IK and no pre-execution FK gate.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import math
import os
from pathlib import Path
import re
import shlex
import struct
import subprocess
import sys
import time
import zlib
from datetime import datetime

import numpy as np


HERE = Path(__file__).resolve().parent
CONTROL_ROOT = HERE.parent
SCRIPTS = HERE / "scripts"
DEFAULT_CONFIG = HERE / "config/closed_loop.json"
ROUTEB_FRONT_HALF_ROOT = (
    CONTROL_ROOT
    / "curobo_motion_planning_routeB/routeB_front_half_v1"
)
ROUTEB_FULL_ROOT = (
    CONTROL_ROOT
    / "curobo_motion_planning_routeB/routeB_full_pipeline_v1"
)
sys.path.insert(0, str(CONTROL_ROOT))
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(ROUTEB_FRONT_HALF_ROOT))
sys.path.insert(0, str(ROUTEB_FULL_ROOT))

from core.bridge import CuroboWorkerClient  # noqa: E402
from core.config import WorkerConfig  # noqa: E402
from persistent_isaac import PersistentIsaacClient  # noqa: E402
from planning.flexible_route_search import (  # noqa: E402
    screen_exact_cover_batch,
    summarize_exact_cover_subfunnel,
)
from planning.simplified_route_search import plan_flexible_route  # noqa: E402
from planning.candidate_rfs_v2_runtime import run_candidate_rfs_v2  # noqa: E402
from all_candidate_gpu_prefilter import load_targets  # noqa: E402
from routeB_front_half import (  # noqa: E402
    build_front_half_goal_pool,
    ensure_robot_segmented_depth,
    run_leap_reach_prefilter_runtime,
    run_routeB_dense_backend,
    save_front_half_goal_pool,
)
from routeB_full_pipeline import (  # noqa: E402
    build_backhalf_chain_pool,
    run_full_motion_backend,
    save_backhalf_chain_pool,
)


VERBOSE = False
DEBUG_LOG: Path | None = None
WORKSPACE_ROI_XYXY = (170, 0, 970, 700)


def load_json(path: Path) -> dict:
    return json.loads(Path(path).resolve().read_text(encoding="utf-8"))


def build_color_sort_zone_specs(*, root: Path, session_root: Path) -> dict[str, dict]:
    layout_path = root / "08_dual_arm_scene_layout/config/manual_layout_calibrated.json"
    color_cfg_path = (
        root
        / "08_dual_arm_scene_layout/isaaclab_control/closed_loop/config/color_sort.json"
    )
    layout = load_json(layout_path)
    color_cfg = load_json(color_cfg_path)
    centre = np.asarray(
        layout["transforms"]["placement_zone"]["position_world_m"], dtype=np.float64
    )
    size = np.asarray(layout["geometry"]["placement_zone_size_m"], dtype=np.float64)
    gap = float(color_cfg.get("placement_split_gap_m", 0.04))
    axis = 0 if float(size[0]) >= float(size[1]) else 1
    if float(size[axis]) <= gap:
        raise RuntimeError(
            f"PlacementZone too small to split: axis_size={size[axis]} gap={gap}"
        )
    child_size = size.copy()
    child_size[axis] = 0.5 * (float(size[axis]) - gap)
    offset = 0.5 * (float(child_size[axis]) + gap)
    red_center = centre.copy()
    blue_center = centre.copy()
    red_center[axis] -= offset
    blue_center[axis] += offset

    def spec(zone_id: str, color: str, c: np.ndarray) -> dict:
        lower = c[:2] - 0.5 * child_size[:2]
        upper = c[:2] + 0.5 * child_size[:2]
        return {
            "zone_id": zone_id,
            "color": color,
            "source": "split_from_manual_layout_calibrated.PlacementZone",
            "split_axis": "x" if axis == 0 else "y",
            "split_gap_m": gap,
            "center_world_m": [float(v) for v in c.tolist()],
            "size_m": [float(v) for v in child_size.tolist()],
            "bounds_xy_min_m": [float(v) for v in lower.tolist()],
            "bounds_xy_max_m": [float(v) for v in upper.tolist()],
        }

    specs = {
        "red_zone": spec("red_zone", "red", red_center),
        "blue_zone": spec("blue_zone", "blue", blue_center),
    }
    artifact = session_root / "color_sort_zones.json"
    artifact.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "layout": str(layout_path),
                "parent_placement_zone": {
                    "center_world_m": [float(v) for v in centre.tolist()],
                    "size_m": [float(v) for v in size.tolist()],
                },
                "zones": specs,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print("\n[COLOR SORT ZONES]")
    for key in ("red_zone", "blue_zone"):
        row = specs[key]
        print(
            f"    {key}: center={row['center_world_m']} size={row['size_m']} "
            f"bounds_xy={row['bounds_xy_min_m']}..{row['bounds_xy_max_m']}"
        )
    print(f"    artifact={artifact}")
    return specs


def write_npz_goal_pool_subset(
    source: Path,
    output: Path,
    *,
    excluded_case_roots: set[str],
) -> int:
    source = Path(source).resolve()
    output = Path(output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with np.load(source, allow_pickle=False) as z:
        payload = {key: np.asarray(z[key]) for key in z.files}
    case_roots = np.asarray(payload["case_root"]).astype(str)
    keep = np.asarray(
        [str(Path(value).resolve()) not in excluded_case_roots for value in case_roots],
        dtype=bool,
    )
    goal_count = int(len(case_roots))
    if int(np.count_nonzero(keep)) <= 0:
        return 0
    filtered = {}
    for key, value in payload.items():
        if getattr(value, "shape", ()) and int(value.shape[0]) == goal_count:
            filtered[key] = value[keep]
        else:
            filtered[key] = value
    np.savez_compressed(output, **filtered)
    return int(np.count_nonzero(keep))


def write_npz_goal_pool_indices(
    source: Path,
    output: Path,
    *,
    indices: list[int],
) -> int:
    source = Path(source).resolve()
    output = Path(output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    wanted = np.asarray(indices, dtype=np.int64).reshape(-1)
    with np.load(source, allow_pickle=False) as z:
        payload = {key: np.asarray(z[key]) for key in z.files}
    goal_count = int(len(payload["case_root"]))
    if np.any(wanted < 0) or np.any(wanted >= goal_count):
        raise IndexError(
            f"goal pool indices out of range: {wanted.tolist()} of {goal_count}"
        )
    filtered = {}
    for key, value in payload.items():
        if getattr(value, "shape", ()) and int(value.shape[0]) == goal_count:
            filtered[key] = value[wanted]
        else:
            filtered[key] = value
    np.savez_compressed(output, **filtered)
    return int(len(wanted))


def resolve(root: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def debug_write(text: str) -> None:
    if DEBUG_LOG is None:
        return
    DEBUG_LOG.parent.mkdir(parents=True, exist_ok=True)
    with DEBUG_LOG.open("a", encoding="utf-8") as stream:
        stream.write(str(text))
        if text and not str(text).endswith("\n"):
            stream.write("\n")


def gpu_memory_snapshot() -> str:
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.used,memory.free",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=5.0,
        )
    except Exception as exc:
        return f"unavailable ({type(exc).__name__}: {exc})"
    if completed.returncode != 0:
        reason = (completed.stderr or completed.stdout or "").strip()
        return f"unavailable ({reason})"
    rows = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    return "; ".join(f"gpu{index}: used/free MiB={row}" for index, row in enumerate(rows)) or "unavailable"


def prepare_roi_depth_for_esdf(capture_root: Path) -> tuple[Path, Path]:
    """Keep full K/T/depth shape, but invalidate depth outside the workspace ROI."""
    depth_path = Path(capture_root) / "depth_m.npy"
    depth = np.load(depth_path).astype(np.float32, copy=True)
    height, width = depth.shape
    x1, y1, x2, y2 = WORKSPACE_ROI_XYXY
    if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
        raise ValueError(f"Invalid workspace ROI {WORKSPACE_ROI_XYXY} for depth shape {(height, width)}")
    roi_depth = np.zeros_like(depth, dtype=np.float32)
    roi_depth[y1:y2, x1:x2] = depth[y1:y2, x1:x2]
    out = Path(capture_root) / "depth_m_workspace_roi.npy"
    np.save(out, roi_depth)
    metadata = {
        "schema_version": 1,
        "purpose": "planner ESDF workspace ROI; DGN2 40k input still uses full depth_m.npy",
        "roi_xyxy_pixels": [x1, y1, x2, y2],
        "full_depth_shape_hw": [height, width],
        "invalidated_outside_roi": True,
        "intrinsics_unchanged": True,
        "T_world_camera_unchanged": True,
    }
    meta_path = Path(capture_root) / "depth_workspace_roi_metadata.json"
    meta_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return out, meta_path


def safe_slug(text: str) -> str:
    slug = re.sub(r"[^0-9A-Za-z._-]+", "_", text.strip()).strip("._")
    return slug[:64] or "target"


def run(label: str, cmd: list, *, cwd: Path, env=None, capture_json: bool = False):
    command_line = " ".join(shlex.quote(str(value)) for value in cmd)
    debug_write(f"\n{'='*18} {label} {'='*18}\n$ {command_line}\n")
    if VERBOSE:
        print(f"\n{'='*18} {label} {'='*18}")
        print("$", command_line, flush=True)
    completed = subprocess.run(
        [str(value) for value in cmd],
        cwd=cwd,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    debug_write(completed.stdout or "")
    if VERBOSE and completed.stdout:
        print(completed.stdout, end="")
    if completed.returncode:
        tail = "\n".join((completed.stdout or "").splitlines()[-30:])
        raise RuntimeError(f"{label} failed: {completed.returncode}\n{tail}")
    if not capture_json:
        return None
    lines = [line for line in (completed.stdout or "").splitlines() if line.strip()]
    for line in reversed(lines):
        try:
            return json.loads(line)
        except Exception:
            pass
    raise RuntimeError(f"{label} did not emit a final JSON object")


def show_async(template, **kwargs) -> None:
    if not template:
        return
    cmd = [str(value).format(**kwargs) for value in template]
    try:
        subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
    except Exception as exc:
        debug_write(f"viewer failed: {exc}")


def show_image_path(path: Path, cfg: dict) -> None:
    path = Path(path)
    if not path.is_file():
        return
    template = cfg.get("show_overlay_command") or ["xdg-open", "{overlay}"]
    try:
        show_async(template, overlay=str(path), rgb=str(path))
    except Exception as exc:
        print(f"    ⚠ 图片打开失败：{path} ({exc})")


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_planning_funnel(cycle_root: Path, funnel: dict) -> Path:
    path = cycle_root / "planning_funnel.json"
    write_json(path, funnel)
    return path


def _paeth(a: int, b: int, c: int) -> int:
    p = a + b - c
    pa = abs(p - a)
    pb = abs(p - b)
    pc = abs(p - c)
    if pa <= pb and pa <= pc:
        return a
    if pb <= pc:
        return b
    return c


def _read_png_rgb(path: Path) -> np.ndarray:
    data = Path(path).read_bytes()
    if data[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError(f"not a PNG: {path}")
    offset = 8
    width = height = color_type = bit_depth = None
    idat = bytearray()
    while offset < len(data):
        length = struct.unpack(">I", data[offset:offset + 4])[0]
        chunk_type = data[offset + 4:offset + 8]
        chunk = data[offset + 8:offset + 8 + length]
        offset += 12 + length
        if chunk_type == b"IHDR":
            width, height, bit_depth, color_type, _comp, _filter, _interlace = struct.unpack(">IIBBBBB", chunk)
        elif chunk_type == b"IDAT":
            idat.extend(chunk)
        elif chunk_type == b"IEND":
            break
    if bit_depth != 8 or color_type not in (2, 6):
        raise ValueError(f"unsupported PNG format bit_depth={bit_depth} color_type={color_type}: {path}")
    channels = 3 if color_type == 2 else 4
    stride = int(width) * channels
    raw = zlib.decompress(bytes(idat))
    rows = np.zeros((int(height), stride), dtype=np.uint8)
    src = 0
    for y in range(int(height)):
        filter_type = raw[src]
        src += 1
        row = np.frombuffer(raw[src:src + stride], dtype=np.uint8).copy()
        src += stride
        prev = rows[y - 1] if y else np.zeros(stride, dtype=np.uint8)
        recon = row
        bpp = channels
        for x in range(stride):
            left = int(recon[x - bpp]) if x >= bpp else 0
            up = int(prev[x])
            up_left = int(prev[x - bpp]) if x >= bpp else 0
            if filter_type == 1:
                recon[x] = (int(recon[x]) + left) & 0xFF
            elif filter_type == 2:
                recon[x] = (int(recon[x]) + up) & 0xFF
            elif filter_type == 3:
                recon[x] = (int(recon[x]) + ((left + up) // 2)) & 0xFF
            elif filter_type == 4:
                recon[x] = (int(recon[x]) + _paeth(left, up, up_left)) & 0xFF
            elif filter_type != 0:
                raise ValueError(f"unsupported PNG filter {filter_type}: {path}")
        rows[y] = recon
    image = rows.reshape((int(height), int(width), channels))
    return image[:, :, :3].copy()


def _write_png_rgb(path: Path, image: np.ndarray) -> None:
    image = np.asarray(image, dtype=np.uint8)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"expected RGB uint8 image, got {image.shape}")
    height, width, _ = image.shape
    raw = bytearray()
    for y in range(height):
        raw.append(0)
        raw.extend(image[y].tobytes())

    def chunk(kind: bytes, payload: bytes) -> bytes:
        return (
            struct.pack(">I", len(payload))
            + kind
            + payload
            + struct.pack(">I", zlib.crc32(kind + payload) & 0xFFFFFFFF)
        )

    payload = (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(bytes(raw), level=6))
        + chunk(b"IEND", b"")
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def _blend_pixels(image: np.ndarray, mask: np.ndarray, color: tuple[int, int, int], alpha: float) -> None:
    if not np.any(mask):
        return
    src = np.asarray(color, dtype=np.float32)
    image[mask] = np.clip((1.0 - alpha) * image[mask].astype(np.float32) + alpha * src, 0, 255).astype(np.uint8)


def _draw_circle(image: np.ndarray, x: float, y: float, radius: int, color: tuple[int, int, int], alpha: float) -> None:
    h, w, _ = image.shape
    x0 = max(0, int(math.floor(x - radius)))
    x1 = min(w - 1, int(math.ceil(x + radius)))
    y0 = max(0, int(math.floor(y - radius)))
    y1 = min(h - 1, int(math.ceil(y + radius)))
    if x0 > x1 or y0 > y1:
        return
    yy, xx = np.ogrid[y0:y1 + 1, x0:x1 + 1]
    mask = (xx - float(x)) ** 2 + (yy - float(y)) ** 2 <= float(radius * radius)
    region = image[y0:y1 + 1, x0:x1 + 1]
    _blend_pixels(region, mask, color, alpha)


_DIGITS_3X5 = {
    "0": ("111", "101", "101", "101", "111"),
    "1": ("010", "110", "010", "010", "111"),
    "2": ("111", "001", "111", "100", "111"),
    "3": ("111", "001", "111", "001", "111"),
    "4": ("101", "101", "111", "001", "001"),
    "5": ("111", "100", "111", "001", "111"),
    "6": ("111", "100", "111", "101", "111"),
    "7": ("111", "001", "010", "010", "010"),
    "8": ("111", "101", "111", "101", "111"),
    "9": ("111", "101", "111", "001", "111"),
}


def _draw_small_digits(image: np.ndarray, x: int, y: int, text: str, color: tuple[int, int, int]) -> None:
    h, w, _ = image.shape
    cx = int(x)
    for char in str(text):
        bitmap = _DIGITS_3X5.get(char)
        if bitmap is None:
            cx += 4
            continue
        for row_index, row in enumerate(bitmap):
            for col_index, value in enumerate(row):
                if value != "1":
                    continue
                px = cx + col_index
                py = int(y) + row_index
                if 0 <= px < w and 0 <= py < h:
                    image[py, px] = color
        cx += 4


def generate_leap_candidate_overlay(
    *,
    rgb_path: Path,
    mask_path: Path,
    prediction: Path,
    intrinsics_path: Path,
    T_world_camera_path: Path,
    output_path: Path,
    query: str,
) -> Path | None:
    """Draw DGN2 LEAP root candidate positions without changing candidate data."""
    try:
        from PIL import Image, ImageDraw, ImageFont
    except Exception as exc:
        Image = ImageDraw = ImageFont = None
        debug_write(f"LEAP overlay using stdlib PNG fallback: PIL unavailable: {exc}")

    try:
        mask = np.load(mask_path).astype(bool)
        K = np.asarray(np.load(intrinsics_path), dtype=np.float64)
        T_world_camera = np.asarray(np.load(T_world_camera_path), dtype=np.float64)
        T_camera_world = np.linalg.inv(T_world_camera)
        with np.load(prediction, allow_pickle=False) as z:
            roots_world = np.asarray(z["translation_world"], dtype=np.float64)
            order = np.asarray(z["target_score_descending_candidate_index"], dtype=np.int64)
            scores = np.asarray(z["score"], dtype=np.float64)
        if order.size == 0:
            return None
        title = (
            f"LEAP grasp root region | total target candidates={len(order)} | "
            f"top score={float(scores[int(order[0])]):.6f} | query={query}"
        )

        if Image is not None:
            rgb = Image.open(rgb_path).convert("RGBA")
            width, height = rgb.size
        else:
            rgb_np = _read_png_rgb(rgb_path)
            height, width, _ = rgb_np.shape
        roots = roots_world[order]
        hom = np.concatenate([roots, np.ones((len(roots), 1), dtype=np.float64)], axis=1)
        cam = (T_camera_world @ hom.T).T[:, :3]
        z = cam[:, 2]
        valid = z > 1e-6
        u = K[0, 0] * cam[:, 0] / np.maximum(z, 1e-9) + K[0, 2]
        v = K[1, 1] * cam[:, 1] / np.maximum(z, 1e-9) + K[1, 2]
        valid &= (u >= 0) & (u < width) & (v >= 0) & (v < height)

        if Image is None:
            image = rgb_np.copy()
            if mask.shape == (height, width):
                _blend_pixels(image, mask, (255, 0, 0), 0.35)
            # Dark title strip and visual legend. Full title is saved beside the PNG.
            image[:24, :, :] = (0.35 * image[:24, :, :]).astype(np.uint8)
            for idx in np.where(valid)[0]:
                _draw_circle(image, float(u[idx]), float(v[idx]), 2, (0, 220, 255), 0.35)
            for idx in np.where(valid[: min(100, len(order))])[0]:
                _draw_circle(image, float(u[idx]), float(v[idx]), 3, (255, 160, 0), 0.70)
            for idx in np.where(valid[: min(20, len(order))])[0]:
                _draw_circle(image, float(u[idx]), float(v[idx]), 5, (255, 255, 0), 0.90)
                _draw_small_digits(image, int(float(u[idx]) + 6), int(float(v[idx]) - 6), str(int(idx)), (255, 255, 255))
            _write_png_rgb(output_path, image)
            write_json(
                output_path.with_suffix(".metadata.json"),
                {
                    "title": title,
                    "total_target_candidates": int(len(order)),
                    "top_score": float(scores[int(order[0])]),
                    "query": query,
                    "projected_candidate_count": int(np.count_nonzero(valid)),
                    "note": "PNG generated with stdlib fallback; title metadata is stored here because PIL is unavailable.",
                },
            )
            return output_path

        overlay = Image.new("RGBA", rgb.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        if mask.shape == (height, width):
            mask_img = Image.fromarray((mask.astype(np.uint8) * 120), mode="L")
            mask_rgba = Image.new("RGBA", rgb.size, (255, 0, 0, 0))
            mask_rgba.putalpha(mask_img)
            overlay = Image.alpha_composite(overlay, mask_rgba)
            draw = ImageDraw.Draw(overlay)

        # All target candidates: small translucent cyan points.
        for idx in np.where(valid)[0]:
            x = float(u[idx])
            y = float(v[idx])
            draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill=(0, 220, 255, 90))

        # Top100: orange.
        for idx in np.where(valid[: min(100, len(order))])[0]:
            x = float(u[idx])
            y = float(v[idx])
            draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill=(255, 160, 0, 170))

        # Top20: larger yellow points with target_rank.
        font = ImageFont.load_default()
        for idx in np.where(valid[: min(20, len(order))])[0]:
            x = float(u[idx])
            y = float(v[idx])
            draw.ellipse((x - 5, y - 5, x + 5, y + 5), fill=(255, 255, 0, 230))
            draw.text((x + 6, y - 6), str(int(idx)), fill=(255, 255, 255, 255), font=font)

        composed = Image.alpha_composite(rgb, overlay)
        draw = ImageDraw.Draw(composed)
        draw.rectangle((0, 0, width, 24), fill=(0, 0, 0, 180))
        draw.text((8, 6), title, fill=(255, 255, 255, 255), font=font)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        composed.convert("RGB").save(output_path)
        return output_path
    except Exception as exc:
        debug_write(f"LEAP overlay failed: {type(exc).__name__}: {exc}")
        return None


def rfs_funnel_stats(rfs_runtime) -> dict:
    payload = rfs_runtime.to_jsonable()
    filter_path = payload.get("filter_json")
    if filter_path and Path(filter_path).is_file():
        try:
            data = load_json(Path(filter_path))
            candidate_count = int(data.get("candidate_count", payload.get("pass_count", 0) + payload.get("reject_count", 0)))
            pass_count = int(data.get("pass_count", payload.get("pass_count", 0)))
            reject_target = int(data.get("reject_target_reach_count", 0))
            reject_trajectory = int(data.get("reject_trajectory_space_count", 0))
            payload.update({
                "candidate_count": candidate_count,
                "target_reach_pass_count": int(candidate_count - reject_target),
                "trajectory_space_pass_count": int(pass_count),
                "pass_count": pass_count,
                "reject_count": int(candidate_count - pass_count),
                "reject_target_reach_count": reject_target,
                "reject_trajectory_space_count": reject_trajectory,
            })
        except Exception as exc:
            payload["stats_parse_error"] = f"{type(exc).__name__}: {exc}"
    report_path = payload.get("report_json")
    if report_path and Path(report_path).is_file():
        try:
            report = load_json(Path(report_path))
            endpoints = report.get("candidate_endpoints", {})
            trajectory = report.get("trajectory_space", {})
            payload.update({
                "grasp_coarse_ik_count": endpoints.get("grasp_direct_coarse_ik"),
                "pregrasp_coarse_ik_count": endpoints.get("pregrasp_direct_coarse_ik"),
                "support_pose_count": trajectory.get("support_pose_count"),
                "support_ik_reachable_count": trajectory.get("support_pose_ik_reachable_count"),
                "support_states_admitted_count": trajectory.get("support_pose_admitted_count", trajectory.get("support_pose_with_collision_free_ik_count")),
                "trajectory_branch_pass_count": trajectory.get("successful_branch_count"),
                "trajectory_branch_count": trajectory.get("anchor_count"),
                "observed_esdf_bypassed": trajectory.get("diagnostic_observed_esdf_bypassed"),
            })
        except Exception as exc:
            payload["report_parse_error"] = f"{type(exc).__name__}: {exc}"
    return payload


def open_rfs_diagnostics(rfs_runtime, cfg: dict) -> list[str]:
    output_dir = rfs_runtime.output_dir
    if not output_dir:
        return []
    root = Path(output_dir)
    opened = []
    for name in ("target_reach_region_overlay.png", "candidate_filter_overlay.png"):
        path = root / name
        if path.is_file():
            show_image_path(path, cfg)
            opened.append(str(path))
        else:
            debug_write(f"RFS diagnostic image missing: {path}")
    return opened


def route_failure_stage(route: dict) -> str:
    if route.get("status") == "PASS":
        return "PASS"
    return str(route.get("failed_stage") or "FLEXIBLE_ROUTE")


def print_route_diagnostics(item: dict, route: dict) -> None:
    print(f"\n    Candidate rank={item['target_rank']} candidate={item['candidate_index']}")
    summaries = route.get("stage_summaries") or []
    by_stage = {str(row.get("stage")): row for row in summaries if isinstance(row, dict)}
    pre = by_stage.get("pregrasp", {})
    pick = by_stage.get("pick_path", {})
    print(
        "      PREGRASP "
        f"targets={pre.get('target_count', 0)} raw={pre.get('raw_success_target_count', 0)} "
        f"reachable={pre.get('reachable_target_count', 0)} "
        f"accepted={pre.get('accepted_solution_count', 0)} nodes={pre.get('node_count', 0)}"
    )
    print(
        "      PRE/COVER "
        f"pairs={pick.get('pair_candidates', 0)} tested={pick.get('pairs_tested', 0)} "
        f"home_pre_fail={pick.get('home_pregrasp_fail', 0)} "
        f"home_pre_self={pick.get('home_pregrasp_self_collision_fail', 0)} "
        f"home_pre_esdf={pick.get('home_pregrasp_esdf_fail', 0)} "
        f"home_pre_pass={pick.get('home_pregrasp_pass', 0)} "
        f"pre_cover_self={pick.get('pregrasp_cover_self_collision_fail', 0)} "
        f"pre_cover_esdf={pick.get('pregrasp_cover_esdf_fail', 0)} "
        f"pre_cover_pass={pick.get('pregrasp_cover_pass', 0)} "
        f"pre_cover_fail={pick.get('pregrasp_cover_fail', 0)} "
        f"status={pick.get('status', 'NA')}"
    )
    for stage in ("lift", "transfer", "place", "retreat"):
        row = by_stage.get(stage, {})
        beam = by_stage.get(f"{stage}_beam", {})
        print(
            f"      {stage.upper():<8} "
            f"endpoint={row.get('target_count', 0)} "
            f"reachable={row.get('reachable_target_count', 0)} "
            f"nodes={row.get('node_count', 0)} "
            f"parents={beam.get('parent_route_count', 0)} "
            f"pairs={beam.get('possible_parent_node_pairs', 0)} "
            f"beam={beam.get('retained_beam_count', 0)}"
        )
    print(f"      FULL ROUTE = {route.get('status')} | reason={route.get('reason', '')}")


def print_funnel_summary(funnel: dict) -> None:
    batches = funnel.get("retarget_batches", [])
    first = batches[0] if batches else {}
    failures = Counter(funnel.get("flexible_route", {}).get("failure_stage_counts", {}))
    if not failures and first.get("failure_stage_counts"):
        failures.update(first["failure_stage_counts"])
    rows = {
        "FINALIZE": int(sum(int(b.get("finalize_reject", 0)) for b in batches)),
        "EXACT_COVER": int(funnel.get("exact_cover", {}).get("reject", 0)),
    }
    for key in ("PREGRASP", "HOME_TO_PRE", "PRE_TO_COVER", "LIFT", "TRANSFER", "PLACE", "RETREAT"):
        rows[key] = int(failures.get(key, 0))
    max_stage = "NA"
    if rows:
        max_stage = max(rows.items(), key=lambda item: item[1])[0]

    print("\n================ CANDIDATE FUNNEL ================")
    dgn = funnel.get("dgn2", {})
    print("[DGN2]")
    print(f"target candidates = {dgn.get('target_candidates', 0)}")
    rfs = funnel.get("rfs_v2", {})
    print("\n[RFS V2 - 粗可达性/粗路径排序]")
    print(
        f"target reach = {rfs.get('target_reach_pass_count', 'NA')}/"
        f"{rfs.get('candidate_count', dgn.get('target_candidates', 'NA'))}"
    )
    print(
        f"trajectory reach = {rfs.get('trajectory_space_pass_count', 'NA')}/"
        f"{rfs.get('candidate_count', dgn.get('target_candidates', 'NA'))}"
    )
    print(f"PASS = {rfs.get('pass_count', 'NA')}")
    print(f"REJECT/rescue = {rfs.get('reject_count', 'NA')}")
    if rfs.get("mode") == "priority_then_rescue":
        print("RFS REJECT is rescue tier; NOT a hard deletion")

    print("\n===== FIRST BATCH SURVIVAL SUMMARY =====")
    print(f"Input               {first.get('input_candidates', 0)}")
    print(f"Finalize PASS       {first.get('finalize_pass', 0)}")
    print(f"Exact COVER PASS    {first.get('exact_cover_pass', 0)}")
    print(f"Full Route PASS     {first.get('full_route_pass', 0)}")
    print("\nFailure breakdown:")
    for key in ("FINALIZE", "EXACT_COVER", "PREGRASP", "HOME_TO_PRE", "PRE_TO_COVER", "LIFT", "TRANSFER", "PLACE", "RETREAT"):
        print(f"{key:<20} {rows.get(key, 0)}")
    print(f"\n最大淘汰阶段 = {max_stage}")
    flex = funnel.get("flexible_route", {})
    hp = flex.get("home_pregrasp_path", {})
    if hp:
        print("\nHOME->PREGRASP path:")
        print(f"tested pairs          {hp.get('tested_pairs', 0)}")
        print(f"self-collision fail   {hp.get('self_collision_failures', 0)}")
        print(f"ESDF fail             {hp.get('esdf_failures', 0)}")
        print(f"PASS                  {hp.get('pass_count', 0)}")
        if hp.get("observed_esdf_bypassed"):
            print("HOME->PRE observed ESDF is DISABLED for this diagnostic run.")
        if hp.get("self_collision_bypassed"):
            print("HOME->PRE self collision is DISABLED for this diagnostic run.")
    pc = flex.get("pregrasp_cover_path", {})
    if pc:
        print("\nPRE->COVER path:")
        print(f"tested pairs          {pc.get('tested_pairs', 0)}")
        print(f"self-collision fail   {pc.get('self_collision_failures', 0)}")
        print(f"ESDF fail             {pc.get('esdf_failures', 0)}")
        print(f"PASS                  {pc.get('pass_count', 0)}")
        if pc.get("observed_esdf_bypassed"):
            print("PRE->COVER observed ESDF is DISABLED for this diagnostic run.")
        if pc.get("self_collision_bypassed"):
            print("PRE->COVER self collision is DISABLED for this diagnostic run.")
    print_flexible_stage_survival(funnel)
    print("==================================================")


def accumulate_home_pre_stats(funnel: dict, route: dict) -> None:
    target = funnel.setdefault("flexible_route", {}).setdefault(
        "home_pregrasp_path",
        {
            "tested_pairs": 0,
            "self_collision_failures": 0,
            "esdf_failures": 0,
            "pass_count": 0,
            "observed_esdf_bypassed": False,
            "self_collision_bypassed": False,
        },
    )
    for row in route.get("stage_summaries") or []:
        if not isinstance(row, dict) or row.get("stage") != "pick_path":
            continue
        target["tested_pairs"] += int(row.get("home_pregrasp_tested", 0))
        target["self_collision_failures"] += int(row.get("home_pregrasp_self_collision_fail", 0))
        target["esdf_failures"] += int(row.get("home_pregrasp_esdf_fail", 0))
        target["pass_count"] += int(row.get("home_pregrasp_pass", 0))
        target["observed_esdf_bypassed"] = bool(
            target.get("observed_esdf_bypassed", False)
            or row.get("home_pregrasp_esdf_bypassed", False)
        )
        target["self_collision_bypassed"] = bool(
            target.get("self_collision_bypassed", False)
            or row.get("home_pregrasp_self_collision_bypassed", False)
        )


def accumulate_pre_cover_stats(funnel: dict, route: dict) -> None:
    target = funnel.setdefault("flexible_route", {}).setdefault(
        "pregrasp_cover_path",
        {
            "tested_pairs": 0,
            "self_collision_failures": 0,
            "esdf_failures": 0,
            "pass_count": 0,
            "observed_esdf_bypassed": False,
            "self_collision_bypassed": False,
        },
    )
    for row in route.get("stage_summaries") or []:
        if not isinstance(row, dict) or row.get("stage") != "pick_path":
            continue
        target["tested_pairs"] += int(row.get("pregrasp_cover_tested", 0))
        target["self_collision_failures"] += int(row.get("pregrasp_cover_self_collision_fail", 0))
        target["esdf_failures"] += int(row.get("pregrasp_cover_esdf_fail", 0))
        target["pass_count"] += int(row.get("pregrasp_cover_pass", 0))
        target["observed_esdf_bypassed"] = bool(
            target.get("observed_esdf_bypassed", False)
            or row.get("pregrasp_cover_esdf_bypassed", False)
        )
        target["self_collision_bypassed"] = bool(
            target.get("self_collision_bypassed", False)
            or row.get("pregrasp_cover_self_collision_bypassed", False)
        )


def update_flexible_stage_aggregates(funnel: dict, route: dict) -> None:
    flex = funnel.setdefault("flexible_route", {})
    aggregate = flex.setdefault("stage_survival", {})
    status = str(route.get("status"))
    failed_stage = route_failure_stage(route) if status != "PASS" else None
    summaries = route.get("stage_summaries") or []
    by_stage = {str(row.get("stage")): row for row in summaries if isinstance(row, dict)}
    stage_order = ("lift", "transfer", "place", "retreat")
    blocked = False
    for stage in stage_order:
        row = by_stage.get(stage)
        beam = by_stage.get(f"{stage}_beam")
        bucket = aggregate.setdefault(
            stage,
            {
                "attempted_candidate_count": 0,
                "not_attempted_blocked_count": 0,
                "endpoint_target_count": 0,
                "raw_success_target_count": 0,
                "reachable_target_count": 0,
                "accepted_solution_count": 0,
                "candidate_node_count": 0,
                "parent_route_count": 0,
                "possible_parent_node_pairs": 0,
                "retained_beam_count": 0,
                "stage_pass_candidate_count": 0,
                "stage_fail_candidate_count": 0,
                "complete_route_candidate_count": 0,
                "full_route_pass_count": 0,
            },
        )
        if blocked or row is None:
            bucket["not_attempted_blocked_count"] += 1
            blocked = True
            continue
        bucket["attempted_candidate_count"] += 1
        bucket["endpoint_target_count"] += int(row.get("target_count", 0))
        bucket["raw_success_target_count"] += int(row.get("raw_success_target_count", 0))
        bucket["reachable_target_count"] += int(row.get("reachable_target_count", 0))
        bucket["accepted_solution_count"] += int(row.get("accepted_solution_count", row.get("solution_count", 0)))
        bucket["candidate_node_count"] += int(row.get("node_count", 0))
        if beam is not None:
            bucket["parent_route_count"] += int(beam.get("parent_route_count", 0))
            bucket["possible_parent_node_pairs"] += int(beam.get("possible_parent_node_pairs", 0))
            bucket["retained_beam_count"] += int(beam.get("retained_beam_count", 0))
            bucket["complete_route_candidate_count"] += int(beam.get("complete_route_candidate_count", 0))
        stage_failed = failed_stage == stage.upper()
        if stage_failed:
            bucket["stage_fail_candidate_count"] += 1
            blocked = True
        elif row is not None and beam is not None and int(beam.get("retained_beam_count", 0)) > 0:
            bucket["stage_pass_candidate_count"] += 1
    if status == "PASS":
        aggregate.setdefault("retreat", {})["full_route_pass_count"] = int(
            aggregate.setdefault("retreat", {}).get("full_route_pass_count", 0)
        ) + 1


def print_flexible_stage_survival(funnel: dict) -> None:
    flex = funnel.get("flexible_route", {})
    stage_stats = flex.get("stage_survival", {})
    pc = flex.get("pregrasp_cover_path", {})
    route_reports = flex.get("candidate_reports", [])
    attempted = len(route_reports)
    pre_cover_pass_candidates = sum(
        1
        for report in route_reports
        for row in report.get("stage_summaries", [])
        if isinstance(row, dict) and row.get("stage") == "pick_path" and row.get("status") == "PASS"
    )
    print("\n===== FLEXIBLE ROUTE STAGE SURVIVAL =====")
    print("\nPRE->COVER")
    print(f"attempted candidates     {attempted}")
    print(f"tested pairs             {pc.get('tested_pairs', 0)}")
    print(f"PASS                     {pre_cover_pass_candidates}")
    print(f"FAIL                     {max(0, attempted - pre_cover_pass_candidates)}")
    for stage in ("lift", "transfer", "place", "retreat"):
        row = stage_stats.get(stage, {})
        label = stage.upper()
        print(f"\n{label}")
        if int(row.get("attempted_candidate_count", 0)) == 0 and int(row.get("not_attempted_blocked_count", 0)) > 0:
            print("NOT ATTEMPTED - BLOCKED BY UPSTREAM STAGE")
            continue
        print(f"attempted candidates     {row.get('attempted_candidate_count', 0)}")
        print(
            "endpoint reachable       "
            f"{row.get('reachable_target_count', 0)} / {row.get('endpoint_target_count', 0)}"
        )
        print(f"raw success targets      {row.get('raw_success_target_count', 0)}")
        print(f"accepted solutions       {row.get('accepted_solution_count', 0)}")
        print(f"IK nodes                 {row.get('candidate_node_count', 0)}")
        print(f"parent routes            {row.get('parent_route_count', 0)}")
        print(f"possible parent-node     {row.get('possible_parent_node_pairs', 0)}")
        print(f"beam retained            {row.get('retained_beam_count', 0)}")
        if stage == "retreat":
            print(f"complete routes          {row.get('complete_route_candidate_count', 0)}")
        print(f"candidate PASS           {row.get('stage_pass_candidate_count', 0)}")
        print(f"candidate FAIL           {row.get('stage_fail_candidate_count', 0)}")
    print("\nFULL ROUTE")
    print(f"PASS candidates          {flex.get('full_route_pass_count', 0)}")
    print(f"FAIL candidates          {flex.get('full_route_fail_count', 0)}")


def print_exact_cover_subfunnel(summary: dict) -> None:
    total = int(summary.get("input_candidates", 0))
    raw_targets = int(summary.get("raw_curobo_reachable_targets", 0))
    strict_targets = int(summary.get("strict_ik_targets", 0))
    post_collision_targets = int(summary.get("post_collision_targets", 0))
    final_targets = int(summary.get("final_exact_cover_pass_targets", 0))
    cover_esdf_bypassed = bool(summary.get("cover_esdf_bypassed", False))
    bypass_reason = str(summary.get("cover_collision_bypass_reason") or "")
    raw_solutions = int(summary.get("raw_success_solution_count", 0))
    strict_solutions = int(summary.get("strict_ik_accepted_solution_count", 0))
    collision_rejected = int(summary.get("collision_rejected_solution_count", 0))
    feasible_solutions = int(summary.get("feasible_solution_count", 0))
    print("\n[Exact COVER SUB-FUNNEL]")
    print(f"Input candidates              {total}")
    print(f"Raw cuRobo reachable          {raw_targets} / {total}")
    print(f"Strict 5mm/5deg/3deg IK       {strict_targets} / {total}")
    if cover_esdf_bypassed:
        print("COVER ESDF collision          BYPASSED")
    else:
        print(f"After COVER ESDF collision    {post_collision_targets} / {strict_targets}")
    print(f"Final Exact COVER PASS        {final_targets} / {total}")
    print("")
    print("Solutions:")
    print(f"raw success                   {raw_solutions}")
    print(f"strict accepted               {strict_solutions}")
    if cover_esdf_bypassed:
        if bypass_reason == "routeB_policy":
            print("collision rejected            0 (Route B policy)")
        else:
            print("collision rejected            0 (diagnostic bypass)")
    else:
        print(f"collision rejected            {collision_rejected}")
    print(f"feasible                      {feasible_solutions}")
    if cover_esdf_bypassed:
        if bypass_reason == "routeB_policy":
            print("Route B policy:")
            print("Exact COVER collision filtering is disabled; strict IK only.")
        else:
            print("DIAGNOSTIC ONLY:")
            print("Exact COVER ESDF collision filtering is disabled.")
    elif strict_solutions > 0 and feasible_solutions == 0:
        print("Exact COVER is being eliminated by collision filtering, not IK reachability.")
    elif strict_solutions == 0:
        print("Exact COVER is being eliminated by strict IK acceptance before collision filtering.")


def prompt_scene(project_root: Path, supplied: str | None) -> Path:
    if supplied:
        folder = Path(supplied).expanduser()
    else:
        print("\n请输入场景文件夹地址（文件夹内需直接包含 scene_manifest.json）")
        print("例如：/home/lin/Projects/DexGraspNet2_Wuji2/02_training_dataset/.../scenes/scene_0000")
        folder = Path(input("Scene folder > ").strip()).expanduser()
    folder = (project_root / folder).resolve() if not folder.is_absolute() else folder.resolve()
    manifest = folder / "scene_manifest.json"
    if not folder.is_dir() or not manifest.is_file():
        raise FileNotFoundError(f"场景目录必须包含 scene_manifest.json: {folder}")
    print(f"✓ 场景：{folder}")
    return folder


def load_robot_state(path: Path) -> tuple[np.ndarray, dict]:
    state = load_json(path)
    return (
        np.asarray(state["right_arm_q_current_rad"], dtype=np.float64),
        {str(key): float(value) for key, value in state["joint_positions_by_name"].items()},
    )


def world_from_base(project_root: Path) -> np.ndarray:
    layout = load_json(project_root / "08_dual_arm_scene_layout/config/manual_layout_calibrated.json")
    return np.asarray(
        layout["transforms"]["dual_arm_mount"]["Gf_local_to_world_row_major"],
        dtype=np.float64,
    ).T


def candidate_order(prediction: Path) -> tuple[list[dict], int]:
    with np.load(prediction, allow_pickle=False) as z:
        order = np.asarray(z["target_score_descending_candidate_index"], dtype=np.int64)
        score = np.asarray(z["score"], dtype=np.float64)
        graspness = np.asarray(z["graspness"], dtype=np.float64)
        log_prob = np.asarray(z["log_prob"], dtype=np.float64)
        total = int(len(score))
    rows = []
    for rank, index in enumerate(order):
        idx = int(index)
        rows.append({
            "target_rank": int(rank),
            "candidate_index": idx,
            "score": float(score[idx]),
            "graspness": float(graspness[idx]),
            "log_prob": float(log_prob[idx]),
        })
    return rows, total


def legacy_coarse_prefilter(
    *,
    client,
    project_root: Path,
    prediction: Path,
    q_current: np.ndarray,
    cfg: dict,
) -> tuple[list[dict], list[int], dict]:
    """Optional compatibility gate; default config bypasses it completely."""
    settings = cfg["coarse_ik_prefilter"]
    candidates, grasp_targets, pregrasp_targets, total = load_targets(
        project_root,
        prediction,
        float(settings.get("legacy_pregrasp_offset_m", 0.10)),
    )
    survivors = list(range(len(candidates)))
    report = {
        "enabled": True,
        "total_proposals": total,
        "target_candidates": len(candidates),
        "grasp_enabled": bool(settings.get("grasp_enabled", False)),
        "pregrasp_enabled": bool(settings.get("pregrasp_enabled", False)),
    }
    if bool(settings.get("grasp_enabled", False)):
        result = client.solve_ik(grasp_targets, q_current, select_chain=False)
        counts = [int(value) for value in result["accepted_per_target"]]
        survivors = [index for index in survivors if counts[index] > 0]
        report["grasp_survivors"] = len(survivors)
    if bool(settings.get("pregrasp_enabled", False)):
        if survivors:
            result = client.solve_ik(pregrasp_targets[survivors], q_current, select_chain=False)
            counts = [int(value) for value in result["accepted_per_target"]]
            survivors = [survivors[local] for local, count in enumerate(counts) if count > 0]
        else:
            survivors = []
        report["pregrasp_survivors"] = len(survivors)
    return candidates, survivors, report


def init_registry(path: Path) -> None:
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "schema_version": 2,
        "purpose": "session-local nominal-size placement centres",
        "placements": [],
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def commit_placement(path: Path, *, cycle: int, execution: dict, selected: dict) -> None:
    registry = load_json(path)
    centre = np.asarray(execution["final_object_position_world_m"][:2], dtype=np.float64)
    registry.setdefault("placements", []).append({
        "cycle": int(cycle),
        "candidate_index": int(selected["candidate_index"]),
        "target_rank": int(selected["target_rank"]),
        "task_type": selected.get("task_type"),
        "color": selected.get("color"),
        "zone_id": selected.get("zone_id"),
        "instance_id": selected.get("instance_id"),
        "target_segmentation_id": int(execution["target_segmentation_id"]),
        "center_world_xy_m": centre.tolist(),
        "actual_final_object_position_world_m": execution["final_object_position_world_m"],
        "committed_local": datetime.now().isoformat(timespec="seconds"),
    })
    path.write_text(json.dumps(registry, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _print_route_summary(report: dict) -> None:
    for row in report.get("stage_summaries", []):
        stage = str(row.get("stage", "")).upper()
        if not stage:
            continue
        if "target_count" in row:
            print(
                f"    {stage:<10} 目标={row.get('target_count')} | "
                f"可达目标={row.get('reachable_target_count', row.get('solution_count', '—'))} | "
                f"IK节点={row.get('node_count', row.get('beam_count', '—'))}"
            )


def choose_motion_route_interactively(value: str | None) -> str:
    if value in {"legacy", "curobo"}:
        return value
    if not sys.stdin.isatty():
        return "legacy"
    while True:
        print("\n==================================================")
        print(" 请选择机械臂路径实现")
        print("==================================================")
        print(" [1] Route A - Legacy")
        print("     关键点 / flexible q7 + quintic interpolation\n")
        print(" [2] Route B - cuRobo")
        print("     true 7DOF MotionPlanner + dense trajectory")
        print("==================================================")
        choice = input("请输入 1 或 2: ").strip()
        if choice == "1":
            return "legacy"
        if choice == "2":
            return "curobo"
        print("请输入 1 或 2。")


def choose_task_interactively(value: str | None) -> str:
    if value in {"semantic-grasp", "color-sort"}:
        return value
    if not sys.stdin.isatty():
        return "semantic-grasp"
    while True:
        print("\n==================================================")
        print(" 请选择任务功能")
        print("==================================================")
        print(" [1] 指定物体名称抓取")
        print("     GroundingDINO + SAM\n")
        print(" [2] 红/蓝颜色分类抓取")
        print("     HSV + 多轮自动抓放")
        print("==================================================")
        choice = input("请输入 1 或 2: ").strip()
        if choice == "1":
            return "semantic-grasp"
        if choice == "2":
            return "color-sort"
        print("请输入 1 或 2。")


def choose_color_seed_interactively(value: int | None, task: str) -> int:
    if task != "color-sort":
        return 42 if value is None else int(value)
    if value is not None:
        return int(value)
    if not sys.stdin.isatty():
        return 42
    raw = input("红/蓝随机染色 seed [默认 42]: ").strip()
    if raw == "":
        return 42
    return int(raw)


def canonical_sort_color(text: str) -> str | None:
    aliases = {
        "red": "red",
        "红": "red",
        "红色": "red",
        "blue": "blue",
        "蓝": "blue",
        "蓝色": "blue",
    }
    return aliases.get(text.strip().lower())


def choose_sort_color_interactively(value: str | None, task: str) -> str | None:
    if task != "color-sort":
        return None
    if value is not None:
        color = canonical_sort_color(value)
        if color is None:
            raise ValueError("--target-color must be red or blue")
        return color
    if not sys.stdin.isatty():
        raise RuntimeError("color-sort requires --target-color red|blue in non-TTY mode")
    while True:
        raw = input("请输入要抓取的颜色（red / blue；红 / 蓝）: ").strip()
        color = canonical_sort_color(raw)
        if color is not None:
            return color
        print("请输入 red / blue，或 红 / 蓝。")


def write_color_attempt(
    *,
    color_root: Path,
    status: str,
    requested_color: str,
    instance_id: str | None,
    detail: str,
) -> Path:
    """Record one non-terminal color-query attempt for later diagnosis.

    This is deliberately only bookkeeping: it never uses the runtime material
    assignment to decide what object to grasp.
    """
    path = color_root / "color_query_result.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": str(status),
                "target_color": str(requested_color),
                "instance_id": None if instance_id is None else str(instance_id),
                "detail": str(detail),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def match_grounded_color_proposals(
    *,
    grounded_sam_result: dict,
    selected_mask_path: Path,
    hsv_instances: list[dict],
) -> list[dict]:
    """Match every legal SAM residual against current RGB HSV instances.

    The backend still selects one highest-score proposal for semantic-grasp.
    Color-sort consumes the complete already-computed legal proposal archive so
    a failed color instance does not hide lower-score proposals for other
    instances of the same requested color.
    """
    detections = {
        int(row["index"]): row
        for row in grounded_sam_result.get("detections", [])
    }
    proposal_masks: list[tuple[int, np.ndarray, str]] = []
    archive_value = grounded_sam_result.get("legal_proposal_masks")
    archive_path = Path(str(archive_value)).resolve() if archive_value else None
    if archive_path is not None and archive_path.is_file():
        with np.load(archive_path, allow_pickle=False) as archive:
            indices = np.asarray(archive["proposal_indices"], dtype=np.int64)
            masks = np.asarray(archive["masks"], dtype=bool)
        if masks.ndim != 3 or len(indices) != len(masks):
            raise RuntimeError(
                f"invalid legal GroundedSAM proposal archive: {archive_path}"
            )
        proposal_masks.extend(
            (int(index), mask, f"{archive_path}#{row}")
            for row, (index, mask) in enumerate(zip(indices, masks))
        )
    else:
        selected_index = grounded_sam_result.get("selected_detection")
        proposal_masks.append(
            (
                -1 if selected_index is None else int(selected_index),
                np.load(Path(selected_mask_path)).astype(bool),
                str(Path(selected_mask_path).resolve()),
            )
        )

    rows: list[dict] = []
    for proposal_index, sam_mask, mask_source in proposal_masks:
        detection = detections.get(proposal_index, {})
        for instance in hsv_instances:
            hsv_mask = np.load(Path(str(instance["mask_path"]))).astype(bool)
            if hsv_mask.shape != sam_mask.shape:
                raise RuntimeError(
                    "GroundedSAM/HSV mask shape mismatch: "
                    f"{sam_mask.shape} vs {hsv_mask.shape}"
                )
            overlap_px = int(np.count_nonzero(sam_mask & hsv_mask))
            rows.append(
                {
                    "proposal_index": int(proposal_index),
                    "proposal_mask_source": mask_source,
                    "dino_score": float(detection.get("score", 0.0)),
                    "instance_id": str(instance["instance_id"]),
                    "overlap_px": overlap_px,
                    "sam_overlap_fraction": float(
                        overlap_px / max(1, int(sam_mask.sum()))
                    ),
                    "hsv_overlap_fraction": float(
                        overlap_px / max(1, int(hsv_mask.sum()))
                    ),
                    "instance": instance,
                    "proposal_mask": sam_mask,
                }
            )
    rows.sort(
        key=lambda row: (
            -int(row["overlap_px"]),
            -float(row["hsv_overlap_fraction"]),
            -float(row["dino_score"]),
            int(row["proposal_index"]),
            str(row["instance_id"]),
        )
    )
    return rows


def is_recoverable_color_planning_failure(message: str) -> bool:
    """True only for an exhausted target/candidate planning funnel.

    The robot is frozen at HOME while planning, so these failures are safe to
    recover by taking a fresh capture and selecting another color instance.
    Worker/protocol/asset failures remain fatal and visible.
    """
    text = str(message)
    markers = (
        "Route B exhausted all front-half goals",
        "Route B found no complete back-half endpoint chain",
        "Route B full planning did not produce a PASS report",
        "NO_PREGRASP_GOAL_WITH_VALID_CUROBO_TRAJECTORY",
        "no back-half chain passed true MotionPlanner",
        "no complete feasible route",
    )
    return any(marker in text for marker in markers)


def is_recoverable_color_target_generation_failure(message: str) -> bool:
    """Classify only a target-local DGN2 empty result as recoverable.

    A missing checkpoint, CUDA failure, corrupt input, or subprocess protocol
    error remains fatal.  The one accepted marker means the official sampler
    completed all configured rounds but generated no seed on this segmented
    instance, while the robot is still at HOME.
    """
    return "Official sampler produced no seed on the segmented target" in str(
        message
    )


def canonical_query(text: str) -> str:
    aliases = {
        "铅笔": "pencil",
        "瓶子": "bottle",
        "杯子": "cup",
        "狗": "dog",
        "马克杯": "mug",
        "遥控器": "remote",
        "手电筒": "flashlight",
        "橡皮": "eraser",
        "罐子": "canister",
    }
    stripped = text.strip()
    return aliases.get(stripped, stripped)


def print_motion_route_banner(route: str) -> None:
    if route == "curobo":
        print("\n========================================")
        print(" MOTION ROUTE B — CUROBO")
        print("========================================")
        print("Task targets      : shared A task semantics（复用A路径任务目标语义）")
        print("Arm planner       : true 7DOF cuRobo（右臂7自由度真实路径规划）")
        print("Dense execution   : cuRobo time trajectory（按cuRobo时间轨迹执行）")
        print("Environment coll. : ON（环境/ESDF碰撞开启）")
        print("Self collision    : OFF（机器人自碰撞关闭）")
        print("Persistent Isaac  : ON（Isaac常驻会话开启）")
        print("Isaac PhysX       : ON（Isaac物理碰撞开启）")
        print("========================================")
        return
    print("\n========================================")
    print(" MOTION ROUTE A — LEGACY")
    print("========================================")
    print("Task targets      : legacy/A production")
    print("Arm path          : keypoint/flexible q7")
    print("Arm interpolation : quintic")
    print("Persistent Isaac  : ON")
    print("Isaac PhysX       : ON")
    print("========================================")


def main() -> int:
    global VERBOSE, DEBUG_LOG
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--scene-folder")
    parser.add_argument("--task", choices=["semantic-grasp", "color-sort"], default=None)
    parser.add_argument("--query", default=None)
    parser.add_argument("--color-seed", type=int, default=None)
    parser.add_argument("--target-color", default=None,
                        help="color-sort target: red/blue (also accepts 红/蓝 interactively)")
    parser.add_argument("--planning-only", action="store_true")
    parser.add_argument("--sim-execute", action="store_true")
    parser.add_argument("--motion-route", choices=["legacy", "curobo"], default=None)
    parser.add_argument("--no-planner-collision-check", action="store_true")
    parser.add_argument(
        "--diagnostic-ignore-static-gate", action="store_true",
        help="兼容旧命令；V2 persistent执行器不再使用旧static gate。",
    )
    parser.add_argument(
        "--diagnostic-full-first-batch",
        action="store_true",
        help="诊断模式：完整评估第一个retarget batch并输出漏斗统计；不执行Isaac动作。",
    )
    parser.add_argument(
        "--diagnostic-disable-rfs-esdf",
        action="store_true",
        help="DIAGNOSTIC ONLY: disable observed RGB-D/ESDF environment rejection inside RFS V2 support-state screening.",
    )
    parser.add_argument(
        "--diagnostic-disable-cover-esdf",
        action="store_true",
        help="DIAGNOSTIC ONLY: disable observed RGB-D/ESDF collision filtering for Exact COVER.",
    )
    parser.add_argument(
        "--diagnostic-disable-home-pre-esdf",
        action="store_true",
        help="DIAGNOSTIC ONLY: disable observed RGB-D/ROI ESDF collision filtering for HOME->PREGRASP only; self-collision remains enabled.",
    )
    parser.add_argument(
        "--diagnostic-disable-home-pre-self-collision",
        action="store_true",
        help="DIAGNOSTIC ONLY: disable self-collision filtering for HOME->PREGRASP only.",
    )
    parser.add_argument(
        "--diagnostic-disable-pre-cover-esdf",
        action="store_true",
        help="DIAGNOSTIC ONLY: disable observed RGB-D/ROI ESDF collision filtering for PREGRASP->COVER only.",
    )
    parser.add_argument(
        "--diagnostic-disable-pre-cover-self-collision",
        action="store_true",
        help="DIAGNOSTIC ONLY: disable self-collision filtering for PREGRASP->COVER only.",
    )
    parser.add_argument(
        "--experimental-bypass-planner-collision",
        action="store_true",
        help=(
            "EXPERIMENTAL ONLY: with --sim-execute, bypass selected planner collision "
            "gates that are known to over-reject while keeping Isaac/PhysX execution enabled."
        ),
    )
    parser.add_argument("--isaac-headless", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    VERBOSE = bool(args.verbose)
    args.task = choose_task_interactively(args.task)
    args.color_seed = choose_color_seed_interactively(args.color_seed, args.task)
    args.motion_route = choose_motion_route_interactively(args.motion_route)
    args.target_color = choose_sort_color_interactively(args.target_color, args.task)
    if args.task == "color-sort":
        print("\n==================================================")
        print(" COLOR SORT TASK — 红/蓝颜色分类抓取")
        print("==================================================")
        print(f"HSV perception   : ON（使用RGB->HSV识别红/蓝实例）")
        print(f"Color seed       : {int(args.color_seed)}")
        print(f"Target color     : {str(args.target_color).upper()}（用户指定）")
        print("GroundingDINO/SAM : ON（以颜色文本匹配物体，不使用颜色GT选择）")
        print("==================================================")
    else:
        print("\n==================================================")
        print(" PERCEPTION ROBOT EXCLUSION CONTRACT")
        print("==================================================")
        print("Robot mask source       : RobotSegmenter（当前cycle唯一robot_mask）")
        print("RGB robot removal       : ON（生成rgb_no_robot；原rgb保留）")
        print("Depth robot removal     : ON（生成filtered_depth）")
        print("DINO robot hard gate    : ON（box主要为机器人直接拒绝）")
        print("SAM robot hard gate     : ON（mask主要为机器人直接拒绝）")
        print("SourceZone gate         : ON（使用3D刚体SourceZone合同）")
        print("Stale mask check        : ON（robot_mask与RGB必须来自同一capture）")
        print("Robot selectable target : NO（机器人绝不能成为最终目标）")
        print("==================================================")
    print_motion_route_banner(args.motion_route)
    routeb_mode = args.motion_route == "curobo"
    if args.planning_only and args.sim_execute:
        raise ValueError("--planning-only and --sim-execute are mutually exclusive")
    if args.diagnostic_full_first_batch and args.sim_execute:
        raise ValueError("--diagnostic-full-first-batch is planning-only and cannot be combined with --sim-execute")
    if (
        args.diagnostic_disable_rfs_esdf
        or args.diagnostic_disable_cover_esdf
        or args.diagnostic_disable_home_pre_esdf
        or args.diagnostic_disable_home_pre_self_collision
        or args.diagnostic_disable_pre_cover_esdf
        or args.diagnostic_disable_pre_cover_self_collision
    ) and not args.diagnostic_full_first_batch:
        raise RuntimeError(
            "--diagnostic-disable-rfs-esdf, --diagnostic-disable-cover-esdf, "
            "--diagnostic-disable-home-pre-esdf, and "
            "--diagnostic-disable-home-pre-self-collision, "
            "--diagnostic-disable-pre-cover-esdf, and "
            "--diagnostic-disable-pre-cover-self-collision "
            "are allowed only with --diagnostic-full-first-batch"
        )
    if args.experimental_bypass_planner_collision and not args.sim_execute:
        raise RuntimeError("--experimental-bypass-planner-collision requires --sim-execute")
    rfs_esdf_bypassed = bool(
        args.diagnostic_disable_rfs_esdf or args.experimental_bypass_planner_collision
    )
    cover_esdf_bypassed = bool(
        args.diagnostic_disable_cover_esdf or args.experimental_bypass_planner_collision
    )
    home_pre_esdf_bypassed = bool(
        args.diagnostic_disable_home_pre_esdf or args.experimental_bypass_planner_collision
    )
    home_pre_self_bypassed = bool(
        args.diagnostic_disable_home_pre_self_collision
        or args.experimental_bypass_planner_collision
    )
    pre_cover_esdf_bypassed = bool(
        args.diagnostic_disable_pre_cover_esdf or args.experimental_bypass_planner_collision
    )
    pre_cover_self_bypassed = bool(
        args.diagnostic_disable_pre_cover_self_collision
        or args.experimental_bypass_planner_collision
    )
    if args.experimental_bypass_planner_collision:
        print("\n==================================================")
        print("EXPERIMENTAL COLLISION BYPASS EXECUTION")
        print("WARNING:")
        print("planner collision gates are disabled.")
        print("This mode is for baseline feasibility test only.")
        print("Isaac physical execution ENABLED.")
        print("==================================================")
    if routeb_mode:
        print("\n========================================")
        print(" Route B COLLISION POLICY")
        print("========================================")
        print("LEAP reach collision        : OFF（LEAP粗可达不做ESDF/自碰撞/路径碰撞）")
        print("Exact COVER observed ESDF   : OFF（COVER只做严格IK，不做观测ESDF过滤）")
        print("Exact COVER collision audit : OFF（COVER不做旧collision_filter_ik审计）")
        print("PREGRASP endpoint collision : OFF（PREGRASP端点只做relaxed IK）")
        print("legacy HOME->PRE path gate  : NOT USED（不使用旧HOME到PRE路径门）")
        print("legacy PRE->COVER path gate : NOT USED（不使用旧PRE到COVER路径门）")
        print("RouteB env collision        : ON（Route B MotionPlanner环境/ESDF碰撞开启）")
        print("RouteB self collision       : OFF（Route B MotionPlanner机器人自碰撞关闭）")
        print(f"Isaac execution             : {'ON（会执行仿真）' if args.sim_execute else 'OFF（planning-only，不执行仿真）'}")
        print("========================================")
    if (
        args.diagnostic_disable_rfs_esdf
        or args.diagnostic_disable_cover_esdf
        or args.diagnostic_disable_home_pre_esdf
        or args.diagnostic_disable_home_pre_self_collision
        or args.diagnostic_disable_pre_cover_esdf
        or args.diagnostic_disable_pre_cover_self_collision
    ):
        print("\n==================================================")
        print("DIAGNOSTIC PLANNER COLLISION BYPASS")
        print(
            "RFS observed ESDF            "
            f"{'DISABLED' if rfs_esdf_bypassed else 'ENABLED'}"
        )
        print(
            "Exact COVER ESDF             "
            f"{'DISABLED' if cover_esdf_bypassed else 'ENABLED'}"
        )
        print(
            "HOME->PRE observed ESDF      "
            f"{'DISABLED' if home_pre_esdf_bypassed else 'ENABLED'}"
        )
        print(
            "HOME->PRE self collision     "
            f"{'DISABLED' if home_pre_self_bypassed else 'ENABLED'}"
        )
        print(
            "PRE->COVER observed ESDF     "
            f"{'DISABLED' if pre_cover_esdf_bypassed else 'ENABLED'}"
        )
        print(
            "PRE->COVER self collision    "
            f"{'DISABLED' if pre_cover_self_bypassed else 'ENABLED'}"
        )
        print("")
        print("LIFT/TRANSFER/PLACE/RETREAT:")
        print("planner collision gate       NONE (existing architecture)")
        print("")
        print("Isaac physical execution     DISABLED")
        print("==================================================")
    if args.diagnostic_ignore_static_gate:
        print("⚠ --diagnostic-ignore-static-gate 在V2中仅为旧命令兼容参数，不再参与筛选。")

    root = args.project_root.expanduser().resolve()
    cfg = load_json(args.config)
    if routeb_mode:
        rb_full_cfg = cfg.get("routeB_full_pipeline", {})
        transfer_attachment = bool(rb_full_cfg.get("transfer_attachment", True))
        print("\n========================================")
        print(" Route B ACTIVE FALSE/OFF OPTIONS")
        print("========================================")
        print("LEAP reach collision        : OFF（LEAP粗可达不做ESDF/自碰撞/路径碰撞）")
        print("Exact COVER observed ESDF   : OFF（COVER只做严格IK，不做观测ESDF过滤）")
        print("Exact COVER collision audit : OFF（COVER不做旧collision_filter_ik审计）")
        print("PREGRASP endpoint collision : OFF（PREGRASP端点只做relaxed IK）")
        print("legacy HOME->PRE path gate  : NOT USED（不使用旧HOME到PRE路径门）")
        print("legacy PRE->COVER path gate : NOT USED（不使用旧PRE到COVER路径门）")
        print("RouteB self collision       : OFF（cuRobo robot self-collision关闭）")
        print(
            "RouteB transfer attachment  : "
            f"{'ON（LIFT到TRANSFER挂载目标proxy）' if transfer_attachment else 'OFF（LIFT到TRANSFER不挂载目标proxy）'}"
        )
        print("RouteB env collision        : ON（机械臂/手仍检查环境ESDF）")
        print(f"Isaac execution             : {'ON（会执行仿真）' if args.sim_execute else 'OFF（planning-only，不执行仿真）'}")
        print("========================================")
    scene_folder = prompt_scene(root, args.scene_folder)
    scene_manifest = scene_folder / "scene_manifest.json"

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_root = resolve(root, cfg["session_root"]) / stamp
    session_root.mkdir(parents=True, exist_ok=False)
    DEBUG_LOG = session_root / "debug.log"
    registry = session_root / "placement_registry.json"
    init_registry(registry)
    (session_root / "session.json").write_text(json.dumps({
        "schema_version": 2,
        "created_local": stamp,
        "task": str(args.task),
        "color_seed": int(args.color_seed) if args.task == "color-sort" else None,
        "target_color": str(args.target_color) if args.task == "color-sort" else None,
        "architecture": cfg.get("architecture"),
        "source_scene_folder": str(scene_folder),
        "sim_execute": bool(args.sim_execute),
        "planner_collision_checks_disabled": bool(args.no_planner_collision_check),
        "experimental_collision_bypass": bool(args.experimental_bypass_planner_collision),
        "routeB_transfer_attachment": bool(
            cfg.get("routeB_full_pipeline", {}).get("transfer_attachment", True)
        ) if routeb_mode else None,
        "diagnostic_full_first_batch": bool(args.diagnostic_full_first_batch),
        "diagnostic_collision_bypass": {
            "rfs_observed_esdf": bool(rfs_esdf_bypassed),
            "exact_cover_observed_esdf": bool(cover_esdf_bypassed),
            "home_pre_observed_esdf": bool(home_pre_esdf_bypassed),
            "home_pre_self_collision": bool(home_pre_self_bypassed),
            "pre_cover_observed_esdf": bool(pre_cover_esdf_bypassed),
            "pre_cover_self_collision": bool(pre_cover_self_bypassed),
        },
        "coarse_ik_prefilter": cfg["coarse_ik_prefilter"],
        "flexible_ik": cfg["flexible_ik"],
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    persistent_config = resolve(root, cfg["persistent_isaac_config"])
    network_py = Path(cfg["network_python"])
    retarget_py = Path(cfg["retarget_python"])
    planner_py = Path(cfg["planner_python"])
    for path in (persistent_config, network_py, retarget_py, planner_py):
        if not path.is_file():
            raise FileNotFoundError(path)

    worker_cfg = WorkerConfig(
        startup_timeout_s=float(cfg.get("worker_startup_timeout_s", 180.0)),
        request_timeout_s=float(cfg.get("worker_request_timeout_s", 600.0)),
    )
    T_world_base = world_from_base(root)
    T_base_from_world = np.linalg.inv(T_world_base)
    color_assignment_path = session_root / "color_assignment.json" if args.task == "color-sort" else None
    color_zone_specs = (
        build_color_sort_zone_specs(root=root, session_root=session_root)
        if args.task == "color-sort"
        else {}
    )

    print("\n============================================================")
    print("  Wuji2 语义灵巧抓取闭环 V2")
    print("  ✓ Isaac Sim 持续会话：一次启动，全程保留物理场景")
    print("  ✓ cuRobo 按轮启动，规划完成后释放 GPU")
    print("  ✓ COVER 精确 IK；其余阶段采用 6D 可行域批量 IK")
    print("  ✓ 执行前不再重复 IK / FK")
    print("============================================================")

    try:
        with PersistentIsaacClient(
            project_root=root,
            scene_manifest=scene_manifest,
            runtime_config=persistent_config,
            startup_timeout_s=float(cfg.get("isaac_startup_timeout_s", 300.0)),
            request_timeout_s=float(cfg.get("isaac_request_timeout_s", 300.0)),
            headless=bool(args.isaac_headless),
            verbose=VERBOSE,
            log_callback=debug_write,
            task=str(args.task),
            color_seed=int(args.color_seed),
            color_assignment=color_assignment_path,
        ) as isaac:
            print("✓ Isaac 持续场景已连接")

            cycle = 0
            failed_targets_current_scene: set[str] = set()
            while True:
                cycle += 1
                cycle_started = time.perf_counter()
                cycle_root = session_root / f"cycle_{cycle:03d}"
                capture_root = cycle_root / "capture"
                scratch_root = cycle_root / "scratch/final_planning"
                cycle_root.mkdir(parents=True, exist_ok=False)
                scratch_root.mkdir(parents=True, exist_ok=True)

                print(f"\n================ 第 {cycle:03d} 轮 ================")
                capture = isaac.capture(capture_root)
                rgb = Path(capture["rgb"])
                settled = Path(capture["settled_scene_manifest"])
                robot_state_path = Path(capture["robot_state"])
                show_async(cfg.get("show_rgb_command"), rgb=str(rgb))
                print(
                    f"[1] ✓ RGB-D 拍照完成 | HOME静置={float(capture['hold_s']):.1f}s "
                    f"| 有效深度={100.0*float(capture['valid_depth_fraction']):.1f}%"
                )
                print("[1.5] RobotSegmenter 统一生成 RGB/深度机器人排除工件 ...")
                filtered_depth_path = ensure_robot_segmented_depth(
                    project_root=root,
                    capture_dir=capture_root,
                    settings=cfg.get("routeB_front_half", {}).get("robot_segmenter", {}),
                )
                robot_mask_path = filtered_depth_path.parent / "robot_mask.npy"
                robot_report_path = filtered_depth_path.parent / "robot_segmentation_report.json"
                rgb_no_robot_path = filtered_depth_path.parent / "rgb_no_robot.png"
                if not all(path.is_file() for path in (robot_mask_path, robot_report_path, rgb_no_robot_path)):
                    raise RuntimeError("RobotSegmenter did not produce current-capture RGB/depth robot-exclusion artifacts")
                print(f"      ✓ mask={robot_mask_path.name} | rgb_no_robot={rgb_no_robot_path.name} | filtered_depth={filtered_depth_path.name}")

                stop_words = {str(value).lower() for value in cfg.get("stop_words", [])}
                target_source = ""
                target_selection: dict[str, object] = {}
                if args.task == "semantic-grasp":
                    if args.query:
                        query_original = str(args.query)
                    else:
                        print("\n你要抓什么东西？（例如 dog / pencil；输入“抓取完成”结束）")
                        print("规划过程中 Ctrl+C = 取消当前目标并重新选择；输入“抓取完成” = 结束会话")
                        query_original = input("Target > ").strip()
                    if query_original.lower() in stop_words:
                        final_snapshot = session_root / "final_scene_manifest.json"
                        isaac.snapshot(final_snapshot)
                        print(f"\n✓ 抓取会话完成，最终场景已保存：{final_snapshot}")
                        return 0
                    if not query_original:
                        print("⚠ 输入为空，本轮不执行规划；场景保持不变。")
                        continue
                    query = canonical_query(query_original)
                    if query != query_original:
                        print(f"    中文目标映射：{query_original} -> {query}")
                    target_slug = safe_slug(query)

                    # 2) GroundingDINO receives the same-cycle RGB with its
                    # RobotSegmenter pixels neutralized.  Final selection is gated by
                    # the same mask and 3D SourceZone membership.
                    # current robot mask and 3D SourceZone membership.
                    gs_root = capture_root / "grounded_sam" / target_slug
                    backend = cfg.get("grounded_sam_backend")
                    if not backend:
                        raise RuntimeError("grounded_sam_backend is not configured")
                    command = [str(x).format(project_root=root, rgb=rgb_no_robot_path, text=query, output=gs_root) for x in backend]
                    started = time.perf_counter()
                    command.extend([
                        "--robot-mask", robot_mask_path,
                        "--settled-scene-manifest", settled,
                        "--capture-dir", capture_root,
                    ])
                    print("[2] GroundingDINO(rgb_no_robot) + SAM + robot/SourceZone safety gate ...")
                    run("GroundingDINO(text + rgb_no_robot) -> SAM", command, cwd=root)
                    gs_result = load_json(gs_root / "result.json")
                    if gs_result.get("status") == "NO_LEGAL_TARGET":
                        print("    ⚠ 没有合法桌面目标：机器人候选已拒绝，不会执行抓取。")
                        if args.query:
                            return 2
                        continue
                    gs_check = run("validate Grounded-SAM output", [
                        network_py, SCRIPTS / "validate_grounded_sam_output.py",
                        "--rgb", rgb, "--output-root", gs_root, "--query", query,
                    ], cwd=root, capture_json=True)
                    overlay = Path(gs_check["overlay"])
                    show_async(cfg.get("show_overlay_command"), overlay=str(overlay))
                    print(
                        f"    ✓ 识别完成 | prompt={query} | score={gs_result.get('grounding_score', gs_result.get('score', 'NA'))} "
                        f"| mask={gs_result.get('mask_pixels', gs_result.get('mask_area_px', 'NA'))} "
                        f"| {time.perf_counter()-started:.1f}s"
                    )
                    target_mask_path = gs_root / "mask.npy"
                    target_source = "grounded_sam"
                    target_selection = {
                        "task_type": "semantic-grasp",
                        "target_source": target_source,
                        "target_mask_path": str(target_mask_path),
                        "query_original": query_original,
                        "query_canonical": query,
                        "color": None,
                        "instance_id": None,
                        "placement_zone_override": None,
                    }
                else:
                    requested_color = str(args.target_color)
                    color_root = cycle_root / "color_sort"
                    print(f"[2] HSV 当前RGB颜色审计（目标={requested_color.upper()}；不使用颜色GT选择）...")
                    color_cmd = [
                        network_py,
                        root / "08_dual_arm_scene_layout/isaaclab_control/closed_loop/color_sort/segmentation.py",
                        "--capture-root", capture_root,
                        "--output-root", color_root,
                        "--config", root / "08_dual_arm_scene_layout/isaaclab_control/closed_loop/config/color_sort.json",
                        "--layout", root / "08_dual_arm_scene_layout/config/manual_layout_calibrated.json",
                        "--assignment", color_assignment_path,
                        "--preferred-color", requested_color,
                    ]
                    for instance_id in sorted(failed_targets_current_scene):
                        color_cmd.extend(["--exclude-instance-id", instance_id])
                    run("HSV color audit", color_cmd, cwd=root, capture_json=False)
                    detection_report = load_json(color_root / "detection_report.json")
                    show_image_path(Path(detection_report["overlay"]), cfg)
                    selected_color_target = detection_report.get("selected")
                    red_source = int(detection_report["red"]["inside_source_zone"])
                    blue_source = int(detection_report["blue"]["inside_source_zone"])
                    print("\n[COLOR SORT DETECTION]")
                    print(f"    RED  source={red_source} visible={detection_report['red']['visible_components']}")
                    print(f"    BLUE source={blue_source} visible={detection_report['blue']['visible_components']}")
                    if selected_color_target is None:
                        final_snapshot = session_root / "final_scene_manifest.json"
                        isaac.snapshot(final_snapshot)
                        summary_path = session_root / "color_sort_summary.json"
                        status = (
                            "PARTIAL_COMPLETE"
                            if failed_targets_current_scene
                            else "COLOR_COMPLETE"
                        )
                        summary_path.write_text(json.dumps({
                            "schema_version": 1,
                            "status": status,
                            "target_color": requested_color,
                            "cycle": cycle,
                            "failed_instance_ids_current_scene": sorted(failed_targets_current_scene),
                            "red_source_remaining": red_source,
                            "blue_source_remaining": blue_source,
                            "final_snapshot": str(final_snapshot),
                        }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
                        print(f"✓ {requested_color.upper()} 颜色任务结束：{status} | summary={summary_path}")
                        return 0

                    # GroundingDINO is the color-query matcher.  HSV remains an
                    # independent current-RGB audit and supplies a stable instance
                    # id for failure recovery; neither reads sort_color ground truth.
                    query_original = requested_color
                    query = f"{requested_color} object"
                    target_slug = safe_slug(f"{requested_color}_object")
                    gs_root = capture_root / "grounded_sam" / target_slug
                    backend = cfg.get("grounded_sam_backend")
                    if not backend:
                        raise RuntimeError("grounded_sam_backend is not configured")
                    command = [
                        str(x).format(
                            project_root=root,
                            rgb=rgb_no_robot_path,
                            text=query,
                            output=gs_root,
                        )
                        for x in backend
                    ]
                    command.extend([
                        "--robot-mask", robot_mask_path,
                        "--settled-scene-manifest", settled,
                        "--capture-dir", capture_root,
                    ])
                    started = time.perf_counter()
                    print(
                        f"[2.5] GroundingDINO(rgb_no_robot, prompt={query!r}) + SAM + safety gate ..."
                    )
                    run("GroundingDINO color prompt -> SAM", command, cwd=root)
                    gs_result = load_json(gs_root / "result.json")
                    if gs_result.get("status") == "NO_LEGAL_TARGET":
                        failed_instance_id = str(selected_color_target["instance_id"])
                        failed_targets_current_scene.add(failed_instance_id)
                        summary_path = write_color_attempt(
                            color_root=color_root,
                            status="NO_LEGAL_COLOR_TARGET",
                            requested_color=requested_color,
                            instance_id=failed_instance_id,
                            detail=(
                                f"GroundingDINO/SAM prompt={query!r} produced no legal target; "
                                f"report={gs_root / 'result.json'}"
                            ),
                        )
                        print(
                            f"⚠ GroundingDINO/SAM没有找到合法{requested_color.upper()}桌面物体；"
                            f"本scene跳过 HSV 实例 {failed_instance_id}，机械臂保持 HOME，重新拍照尝试其他实例。"
                        )
                        print(f"    attempt={summary_path}")
                        continue
                    gs_check = run("validate Grounded-SAM color output", [
                        network_py, SCRIPTS / "validate_grounded_sam_output.py",
                        "--rgb", rgb, "--output-root", gs_root, "--query", query,
                    ], cwd=root, capture_json=True)
                    overlay = Path(gs_check["overlay"])
                    show_async(cfg.get("show_overlay_command"), overlay=str(overlay))
                    grounded_sam_mask_path = gs_root / "mask.npy"
                    hsv_instances = [
                        row for row in detection_report[requested_color]["instances"]
                        if str(row["instance_id"]) not in failed_targets_current_scene
                    ]
                    match_rows = match_grounded_color_proposals(
                        grounded_sam_result=gs_result,
                        selected_mask_path=grounded_sam_mask_path,
                        hsv_instances=hsv_instances,
                    )
                    color_match = match_rows[0] if match_rows else None
                    match_path = color_root / "grounding_color_match.json"
                    match_path.write_text(json.dumps({
                        "schema_version": 1,
                        "target_color": requested_color,
                        "query_groundingdino": query,
                        "grounded_sam_mask": str(grounded_sam_mask_path),
                        "legal_proposal_masks": gs_result.get("legal_proposal_masks"),
                        "matches": [
                            {
                                key: value
                                for key, value in row.items()
                                if key != "proposal_mask"
                            }
                            for row in match_rows
                        ],
                    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
                    if color_match is None or int(color_match["overlap_px"]) <= 0:
                        failed_instance_id = str(selected_color_target["instance_id"])
                        failed_targets_current_scene.add(failed_instance_id)
                        write_color_attempt(
                            color_root=color_root,
                            status="NO_HSV_COLOR_MATCH",
                            requested_color=requested_color,
                            instance_id=failed_instance_id,
                            detail=(
                                f"DINO/SAM mask={grounded_sam_mask_path} has no overlap with a current "
                                f"{requested_color} HSV instance; match={match_path}"
                            ),
                        )
                        print(
                            f"⚠ DINO/SAM目标没有与当前RGB的{requested_color.upper()} HSV实例匹配；"
                            f"本scene跳过 HSV 实例 {failed_instance_id}，重新拍照尝试其他实例。"
                        )
                        continue
                    matched_instance = color_match["instance"]
                    matched_hsv_mask = np.load(
                        Path(str(matched_instance["mask_path"]))
                    ).astype(bool)
                    sam_mask = np.asarray(color_match["proposal_mask"], dtype=bool)
                    color_match_metadata = {
                        key: value
                        for key, value in color_match.items()
                        if key != "proposal_mask"
                    }
                    # GroundingDINO establishes the user-requested color prompt;
                    # the same RGB HSV instance makes its SAM region a *single*
                    # physical target when SAM grouped several same-color objects.
                    # This uses neither `sort_color` nor the color-assignment JSON.
                    selected_target_mask = sam_mask & matched_hsv_mask
                    target_mask_path = gs_root / "color_matched_instance_mask.npy"
                    np.save(target_mask_path, selected_target_mask)
                    target_source = "grounded_sam_color_prompt_hsv_instance_intersection"
                    target_selection = {
                        "task_type": "color-sort",
                        "target_source": target_source,
                        "target_mask_path": str(target_mask_path),
                        "query_original": query_original,
                        "query_canonical": query,
                        "color": requested_color,
                        "instance_id": str(matched_instance["instance_id"]),
                        "bbox": matched_instance["bbox_xyxy"],
                        "centroid_uv": matched_instance["centroid_uv"],
                        "centroid_world": matched_instance["centroid_world_m"],
                        "placement_zone_override": f"{requested_color}_zone",
                        "metadata": {
                            "grounded_sam_result": str(gs_root / "result.json"),
                            "grounded_sam_mask": str(grounded_sam_mask_path),
                            "selected_mask_pixels": int(selected_target_mask.sum()),
                            "hsv_color_match": color_match_metadata,
                            "hsv_audit": str(color_root / "detection_report.json"),
                        },
                    }
                    (color_root / "selected_target.json").write_text(
                        json.dumps(target_selection, ensure_ascii=False, indent=2) + "\n",
                        encoding="utf-8",
                    )
                    print(
                        f"    selected={matched_instance['instance_id']} | color={requested_color.upper()} | "
                        f"DINO proposal={color_match['proposal_index']} "
                        f"score={color_match['dino_score']} | "
                        f"HSV overlap={int(color_match['overlap_px'])} px | "
                        f"{time.perf_counter()-started:.1f}s"
                    )

                # 3) RGB-D -> official 40k input
                dgn_root = capture_root / "dgn2" / target_slug
                dgn_depth_path = filtered_depth_path
                print("[3] DGN2/ESDF 使用当前cycle同一份 RobotSegmenter filtered_depth ...")
                print("[3] 构建 DGN2 40k 场景点云 ...")
                dgn_command = [
                    network_py, root / "08_dual_arm_scene_layout/scripts/08_build_target_network_input.py",
                    "--target", target_slug,
                    "--target-segmentation-id", str(int(cfg["dgn2_target_membership_id"])),
                    "--capture-root", capture_root,
                    "--mask", target_mask_path,
                    "--depth-path", dgn_depth_path,
                ]
                run("RGB-D -> full-scene 40k + target membership", dgn_command, cwd=root)
                net_meta = load_json(dgn_root / "network_input.json")
                print(f"    ✓ 40k输入完成 | target_points={net_meta.get('sampled_target_point_count', 'NA')}")

                # 4) DGN2
                print("[4] DGN2 生成抓取候选 ...")
                started = time.perf_counter()
                try:
                    run("Official DGN2 LEAP inference", [
                        network_py, root / "08_dual_arm_scene_layout/scripts/09_predict_official_leap_target.py",
                        "--target", target_slug,
                        "--rounds", str(int(cfg["dgn2_rounds"])),
                        "--input-root", dgn_root,
                    ], cwd=root)
                except RuntimeError as exc:
                    if (
                        args.task != "color-sort"
                        or not is_recoverable_color_target_generation_failure(
                            str(exc)
                        )
                    ):
                        raise
                    failed_instance_id = str(target_selection["instance_id"])
                    failed_targets_current_scene.add(failed_instance_id)
                    attempt_path = write_color_attempt(
                        color_root=color_root,
                        status="DGN2_NO_TARGET_SEED",
                        requested_color=requested_color,
                        instance_id=failed_instance_id,
                        detail=(
                            "Official DGN2 completed all sampler rounds but produced "
                            "no seed on this segmented instance. Robot remained at HOME."
                        ),
                    )
                    print(
                        "[COLOR SORT] 当前颜色实例没有生成DGN2抓取候选。"
                    )
                    print(f"    skipped instance : {failed_instance_id}")
                    print("    robot state      : HOME（尚未执行机械臂动作）")
                    print("    next action      : fresh capture -> 尝试下一同色实例")
                    print(f"    attempt          : {attempt_path}")
                    continue
                prediction = dgn_root / "official_leap_1024_target_ranked.npz"
                candidates_plain, total_proposals = candidate_order(prediction)
                print(
                    f"    ✓ proposals={total_proposals} | 目标候选={len(candidates_plain)} "
                    f"| {time.perf_counter()-started:.1f}s"
                )
                funnel = {
                    "schema_version": 1,
                    "query": query,
                    "motion_route": str(args.motion_route),
                    "experimental_collision_bypass": bool(args.experimental_bypass_planner_collision),
                    "diagnostic_full_first_batch": bool(args.diagnostic_full_first_batch),
                    "diagnostic_collision_bypass": {
                        "rfs_observed_esdf": bool(rfs_esdf_bypassed),
                        "exact_cover_observed_esdf": bool(cover_esdf_bypassed),
                        "home_pre_observed_esdf": bool(home_pre_esdf_bypassed),
                        "home_pre_self_collision": bool(home_pre_self_bypassed),
                        "pre_cover_observed_esdf": bool(pre_cover_esdf_bypassed),
                        "pre_cover_self_collision": bool(pre_cover_self_bypassed),
                    },
                    "dgn2": {
                        "total_proposals": int(total_proposals),
                        "target_candidates": int(len(candidates_plain)),
                    },
                    "rfs_v2": {},
                    "routeB_front_half": {},
                    "retarget_batches": [],
                    "exact_cover": {"tested": 0, "pass": 0, "reject": 0, "pass_rate": 0.0},
                    "flexible_route": {
                        "candidate_reports": [],
                        "full_route_pass_count": 0,
                        "full_route_fail_count": 0,
                        "failure_stage_counts": {},
                    },
                }
                leap_overlay = generate_leap_candidate_overlay(
                    rgb_path=rgb,
                    mask_path=target_mask_path,
                    prediction=prediction,
                    intrinsics_path=capture_root / "intrinsics.npy",
                    T_world_camera_path=capture_root / "T_world_camera.npy",
                    output_path=dgn_root / "leap_grasp_candidate_region_overlay.png",
                    query=query,
                )
                if leap_overlay is not None:
                    funnel["dgn2"]["leap_grasp_candidate_region_overlay"] = str(leap_overlay)
                    print(f"    ✓ LEAP候选区域图：{leap_overlay}")
                    show_image_path(leap_overlay, cfg)
                else:
                    print("    ⚠ LEAP候选区域图生成失败；pipeline继续，详见debug.log")
                write_planning_funnel(cycle_root, funnel)
                if args.motion_route == "legacy":
                    rfs_runtime = run_candidate_rfs_v2(
                        project_root=root,
                        cycle_root=cycle_root,
                        query=target_slug,
                        candidates=candidates_plain,
                        settings=cfg.get("candidate_rfs_v2", {}),
                        diagnostic_disable_observed_esdf=bool(rfs_esdf_bypassed),
                    )
                else:
                    rfs_runtime = run_leap_reach_prefilter_runtime(
                        project_root=root,
                        cycle_root=cycle_root,
                        query=target_slug,
                        candidates=candidates_plain,
                        settings=cfg.get("leap_reach_prefilter_routeB", {}),
                    )
                rfs_priority_indices = list(rfs_runtime.ordered_indices)
                rfs_stats = rfs_funnel_stats(rfs_runtime)
                rfs_opened = open_rfs_diagnostics(rfs_runtime, cfg)
                rfs_stats["opened_diagnostic_images"] = rfs_opened
                if args.motion_route == "legacy":
                    funnel["rfs_v2"] = rfs_stats
                    print("\n[RFS V2 - 粗可达性/粗路径排序]")
                    print(
                        f"    GRASP coarse IK={rfs_stats.get('grasp_coarse_ik_count', 'NA')}/"
                        f"{rfs_stats.get('candidate_count', len(candidates_plain))} | "
                        f"PREGRASP coarse IK={rfs_stats.get('pregrasp_coarse_ik_count', 'NA')}/"
                        f"{rfs_stats.get('candidate_count', len(candidates_plain))}"
                    )
                    print(
                        f"    mode={rfs_stats.get('mode')} status={rfs_stats.get('status')} | "
                        f"target reach={rfs_stats.get('target_reach_pass_count', 'NA')}/"
                        f"{rfs_stats.get('candidate_count', len(candidates_plain))}"
                    )
                    print(
                        f"    trajectory reach={rfs_stats.get('trajectory_space_pass_count', 'NA')}/"
                        f"{rfs_stats.get('candidate_count', len(candidates_plain))} | "
                        f"PASS={rfs_stats.get('pass_count')} | REJECT/rescue={rfs_stats.get('reject_count')}"
                    )
                    print(
                        f"    support IK reachable={rfs_stats.get('support_ik_reachable_count', 'NA')}/"
                        f"{rfs_stats.get('support_pose_count', 'NA')} | "
                        f"support admitted={rfs_stats.get('support_states_admitted_count', 'NA')}/"
                        f"{rfs_stats.get('support_pose_count', 'NA')} | "
                        f"branches PASS={rfs_stats.get('trajectory_branch_pass_count', 'NA')}/"
                        f"{rfs_stats.get('trajectory_branch_count', 'NA')}"
                    )
                    if rfs_stats.get("mode") == "priority_then_rescue":
                        print("    RFS REJECT is rescue tier; NOT a hard deletion")
                else:
                    funnel["routeB_front_half"]["leap_reach"] = rfs_stats
                    print("\n[Route B LEAP reach prior - reach-region only（LEAP粗可达排序）]")
                    print(
                        f"    mode={rfs_stats.get('mode')} status={rfs_stats.get('status')} | "
                        f"PASS={rfs_stats.get('pass_count')}/{rfs_stats.get('candidate_count', len(candidates_plain))} | "
                        f"direct={rfs_stats.get('direct_count', 'NA')} near={rfs_stats.get('near_region_count', 'NA')} | "
                        f"REJECT/rescue={rfs_stats.get('reject_count')}"
                    )
                    print("    说明：只做LEAP抓取根位姿的粗可达排序；不做ESDF碰撞、不做路径检查。")
                    if rfs_stats.get("mode") == "priority_then_rescue":
                        print("    说明：REJECT只是救援队列，不是永久删除；PASS候选失败后仍可回退尝试。")
                write_planning_funnel(cycle_root, funnel)

                # simulation-only binding after semantic selection
                sim_binding = cycle_root / "sim_target.json"
                bind = run("simulation-only mask -> rigid-body binding", [
                    network_py, SCRIPTS / "resolve_sim_target.py",
                    "--capture-root", capture_root,
                    "--mask", target_mask_path,
                    "--settled-manifest", settled,
                    "--output", sim_binding,
                ], cwd=root, capture_json=True)
                sim_target_id = int(bind["segmentation_id"])
                q_current, measured = load_robot_state(robot_state_path)

                try:
                    home_gate_cfg = cfg.get("home_pregrasp_collision_gate", {})
                    home_gate_enabled = bool(home_gate_cfg.get("enabled", True))
                    # Route B front-half deliberately does not use legacy ROI ESDF gates.
                    # Its only environment collision gate is the true 7DOF MotionPlanner
                    # built from RobotSegmenter-filtered depth after cuRobo IK is released.
                    need_observed_map = (
                        not routeb_mode
                        and (home_gate_enabled or not args.no_planner_collision_check)
                    )
                    if need_observed_map:
                        roi_depth_path, roi_depth_meta = prepare_roi_depth_for_esdf(capture_root)
                        map_report = {
                            "status": "PER_BATCH",
                            "depth_path": str(roi_depth_path),
                            "roi_metadata": str(roi_depth_meta),
                            "home_pregrasp_collision_gate": home_gate_enabled,
                            "full_planner_collision_check": not bool(args.no_planner_collision_check),
                            "batch_maps": [],
                        }
                        print(
                            "[5] ✓ ROI ESDF输入已准备 | "
                            f"ROI={list(WORKSPACE_ROI_XYXY)} | depth shape/K/T保持不变"
                        )
                    else:
                        skip_reason = (
                            "Route B uses RobotSegmenter-filtered ESDF in the true 7DOF backend"
                            if routeb_mode
                            else "all planner collision checks disabled"
                        )
                        map_report = {
                            "status": "SKIPPED",
                            "reason": skip_reason,
                        }
                        if routeb_mode:
                            print("[5] ✓ Route B跳过旧ROI ESDF门（旧路径碰撞门关闭）；真实7DOF MotionPlanner的环境ESDF仍开启")
                        else:
                            print("[5] ✓ 规划器碰撞检查：全部关闭（Isaac/PhysX仍开启）")

                    # Optional legacy approximate prefilter; OFF by default.
                    coarse_cfg = cfg["coarse_ik_prefilter"]
                    if bool(coarse_cfg.get("grasp_enabled")) or bool(coarse_cfg.get("pregrasp_enabled")):
                        raise RuntimeError(
                            "legacy coarse_ik_prefilter requires a cycle-wide cuRobo worker and is disabled "
                            "for the final per-batch worker architecture"
                        )
                    else:
                        candidates = candidates_plain
                        survivor_indices = list(range(len(candidates)))
                        coarse_report = {
                            "enabled": False,
                            "grasp_enabled": False,
                            "pregrasp_enabled": False,
                            "survivors": len(survivor_indices),
                        }
                        print(
                            f"[6] ✓ 旧粗 GRASP/PREGRASP IK：关闭 | {len(candidates)} 个目标候选进入真实 Wuji2 retarget + Exact COVER"
                        )

                    allowed_survivors = {int(index) for index in survivor_indices}
                    survivor_indices = [
                        int(index) for index in rfs_priority_indices
                        if int(index) in allowed_survivors
                    ]
                    if args.motion_route == "legacy":
                        coarse_report["candidate_rfs_v2"] = rfs_runtime.to_jsonable()
                    else:
                        coarse_report["routeB_leap_reach_prefilter"] = rfs_runtime.to_jsonable()
                    print(
                        f"[{'RFS V2' if args.motion_route == 'legacy' else 'Route B LEAP reach'}] production order applied | "
                        f"ordered survivors={len(survivor_indices)} | "
                        f"status={rfs_runtime.status}"
                    )
                    if routeb_mode:
                        print(
                            "    说明：排序后继续走 Wuji2重定向 -> Exact COVER -> PREGRASP goal pool -> Route B 7DOF路径规划。"
                        )

                    max_to_test = int(cfg.get("max_candidates_to_test", 0))
                    if max_to_test > 0:
                        survivor_indices = survivor_indices[:max_to_test]
                    retarget_chunk_size = int(cfg.get("retarget_chunk_size", 64))
                    total_batches = math.ceil(len(survivor_indices) / retarget_chunk_size)
                    selected = None
                    routeb_goal_pool_path = None
                    routeb_goal_pool_report = None
                    routeb_dense_report = None
                    tested_cover = 0
                    retargeted = 0
                    exact_cover_pass_total = 0
                    full_route_pass_count = 0
                    full_route_fail_count = 0
                    failure_stage_counts: Counter[str] = Counter()

                    print(
                        "[7] Wuji2 重定向 + 精确 COVER + "
                        + ("Route B PREGRASP goal pool" if args.motion_route == "curobo" else "Flexible IK 搜索")
                    )
                    for chunk_index, start in enumerate(range(0, len(survivor_indices), retarget_chunk_size), start=1):
                        local_indices = survivor_indices[start:start + retarget_chunk_size]
                        chunk_items = []
                        for local_index in local_indices:
                            item = candidates[local_index]
                            rank = int(item["target_rank"])
                            idx = int(item["candidate_index"])
                            case_id = f"{cfg.get('candidate_case_prefix','closedloop')}_r{rank:04d}_cand{idx:04d}"
                            case_root = scratch_root / f"rank_{rank:04d}" / case_id
                            chunk_items.append({
                                "local_target_index": int(local_index),
                                "target_rank": rank,
                                "candidate_index": idx,
                                "official_score": float(item.get("score", item.get("official_score", float('nan')))),
                                "case_id": case_id,
                                "case_root": str(case_root),
                            })
                        if not chunk_items:
                            continue
                        chunk_dir = scratch_root / f"batch_{chunk_index:03d}"
                        chunk_dir.mkdir(parents=True, exist_ok=True)
                        items_json = chunk_dir / "items.json"
                        items_json.write_text(json.dumps(chunk_items, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
                        batch_started = time.perf_counter()
                        batch_funnel = {
                            "batch_index": int(chunk_index),
                            "total_batches": int(total_batches),
                            "rank_range": [
                                int(chunk_items[0]["target_rank"]),
                                int(chunk_items[-1]["target_rank"]),
                            ],
                            "input_candidates": int(len(chunk_items)),
                            "retargeted": 0,
                            "finalize_pass": 0,
                            "finalize_reject": 0,
                            "exact_cover_tested": 0,
                            "exact_cover_pass": 0,
                            "exact_cover_reject": 0,
                            "full_route_pass": 0,
                            "full_route_fail": 0,
                            "failure_stage_counts": {},
                        }
                        print(
                            f"\n[Batch {chunk_index:02d}/{total_batches:02d}] "
                            f"ranks {batch_funnel['rank_range'][0]}..{batch_funnel['rank_range'][1]} | "
                            f"input={len(chunk_items)}"
                        )
                        build_report = run("batch build candidate cases", [
                            network_py, SCRIPTS / "batch_build_candidate_cases.py",
                            "--project-root", root,
                            "--prediction", prediction,
                            "--network-input", dgn_root / "network_input.npz",
                            "--capture-root", capture_root,
                            "--settled-manifest", settled,
                            "--sim-target-segmentation-id", str(sim_target_id),
                            "--items-json", items_json,
                            "--output", chunk_dir / "batch_build_report.json",
                        ], cwd=root, capture_json=True)
                        retarget_report = run("batch LEAP->Wuji2 retarget", [
                            retarget_py, SCRIPTS / "batch_retarget_cases.py",
                            "--items-json", items_json,
                            "--output", chunk_dir / "batch_retarget_report.json",
                        ], cwd=root, capture_json=True)
                        finalize_report = run("batch finalize Wuji2 + arm targets", [
                            network_py, SCRIPTS / "batch_finalize_candidate_cases.py",
                            "--items-json", items_json,
                            "--output", chunk_dir / "batch_finalize_report.json",
                        ], cwd=root, capture_json=True)
                        retargeted += len(chunk_items)
                        batch_funnel["retargeted"] = int(len(chunk_items))
                        batch_funnel["build_time_s"] = float((build_report or {}).get("wall_time_s", 0.0))
                        batch_funnel["retarget_time_s"] = float((retarget_report or {}).get("wall_time_s", 0.0))
                        batch_funnel["finalize_time_s"] = float((finalize_report or {}).get("wall_time_s", 0.0))
                        finalize_results = json.loads(
                            (chunk_dir / "batch_finalize_report.json").read_text(encoding="utf-8")
                        ).get("results", [])
                        failures_jsonl = chunk_dir / "flexible_route_failures.jsonl"
                        item_by_case = {
                            str(Path(item["case_root"]).resolve()): item
                            for item in chunk_items
                        }
                        with failures_jsonl.open("a", encoding="utf-8") as stream:
                            for row in finalize_results:
                                if row.get("status") == "PASS":
                                    continue
                                failure_stage_counts["FINALIZE"] += 1
                                item = item_by_case.get(str(Path(row.get("case_root", "")).resolve()))
                                stream.write(json.dumps({
                                    "target_rank": None if item is None else int(item["target_rank"]),
                                    "candidate_index": None if item is None else int(item["candidate_index"]),
                                    "failure_stage": "FINALIZE",
                                    "failure_reason": row.get("reason", row.get("error", "finalize failed")),
                                    "stage_summaries": row,
                                }, ensure_ascii=False) + "\n")
                        finalized_case_roots = {
                            str(Path(row["case_root"]).resolve())
                            for row in finalize_results
                            if row.get("status") == "PASS"
                        }
                        finalized_items = [
                            item for item in chunk_items
                            if str(Path(item["case_root"]).resolve()) in finalized_case_roots
                        ]
                        finalize_reject_count = int(finalize_report.get("reject_count", len(chunk_items) - len(finalized_items)))
                        batch_funnel["finalize_pass"] = int(len(finalized_items))
                        batch_funnel["finalize_reject"] = int(finalize_reject_count)
                        if not finalized_items:
                            batch_funnel["wall_s"] = float(time.perf_counter() - batch_started)
                            batch_funnel["failure_stage_counts"] = dict(Counter({"FINALIZE": finalize_reject_count}))
                            funnel["retarget_batches"].append(batch_funnel)
                            funnel["flexible_route"]["failure_stage_counts"] = dict(failure_stage_counts)
                            write_planning_funnel(cycle_root, funnel)
                            print(
                                f"    Batch {chunk_index:02d}/{total_batches:02d} ✓ 重定向={len(chunk_items)} | "
                                f"finalize PASS=0 REJECT={finalize_reject_count} | "
                                f"{time.perf_counter()-batch_started:.1f}s"
                            )
                            if args.diagnostic_full_first_batch:
                                print("    [DIAGNOSTIC] first batch complete: no finalized candidates.")
                                break
                            continue

                        print(
                            f"    [cuRobo Batch {chunk_index:02d}/{total_batches:02d}] start | "
                            f"GPU {gpu_memory_snapshot()}"
                        )
                        with CuroboWorkerClient(
                            root,
                            worker_config=worker_cfg,
                            seeds=int(cfg.get("gpu_ik_seeds", 48)),
                            batch_size=int(cfg.get("gpu_ik_batch_size", 512)),
                        ) as curobo:
                            if need_observed_map:
                                map_started = time.perf_counter()
                                batch_map = curobo.build_map(
                                    roi_depth_path,
                                    capture_root / "intrinsics.npy",
                                    capture_root / "T_world_camera.npy",
                                    target_mask_path,
                                )
                                batch_map.update({
                                    "batch_index": int(chunk_index),
                                    "workspace_roi_xyxy": list(WORKSPACE_ROI_XYXY),
                                    "home_pregrasp_collision_gate": home_gate_enabled,
                                    "full_planner_collision_check": not bool(args.no_planner_collision_check),
                                    "build_wall_s": time.perf_counter() - map_started,
                                })
                                map_report["batch_maps"].append(batch_map)
                                print(
                                    f"      map ROI build {batch_map['build_wall_s']:.2f}s | "
                                    f"GPU {gpu_memory_snapshot()}"
                                )
                            else:
                                print(f"      map skipped | GPU {gpu_memory_snapshot()}")

                            cover_rows = screen_exact_cover_batch(
                                client=curobo,
                                case_roots=[Path(item["case_root"]) for item in finalized_items],
                                q_current=q_current,
                                measured=measured,
                                T_base_from_world=T_base_from_world,
                                T_world_base=T_world_base,
                                no_planner_collision_check=bool(
                                    args.no_planner_collision_check or routeb_mode
                                ),
                                block_unknown=bool(cfg.get("block_unknown_space", False)),
                                solutions_per_candidate=int(cfg["flexible_ik"]["selection"]["cover_solutions_per_candidate"]),
                                diagnostic_disable_cover_esdf=bool(cover_esdf_bypassed),
                                cover_collision_bypass_reason=(
                                    "routeB_policy" if routeb_mode else None
                                ),
                            )
                            passed_cover = [row for row in cover_rows if row["pass"]]
                            exact_subfunnel = summarize_exact_cover_subfunnel(cover_rows)
                            print_exact_cover_subfunnel(exact_subfunnel)
                            tested_cover += len(cover_rows)
                            exact_cover_pass_total += len(passed_cover)
                            batch_funnel["exact_cover_tested"] = int(len(cover_rows))
                            batch_funnel["exact_cover_pass"] = int(len(passed_cover))
                            batch_funnel["exact_cover_reject"] = int(len(cover_rows) - len(passed_cover))
                            batch_funnel["exact_cover_subfunnel"] = exact_subfunnel
                            cover_by_case = {str(Path(row["case_root"]).resolve()): row for row in cover_rows}
                            for item in finalized_items:
                                row = cover_by_case.get(str(Path(item["case_root"]).resolve()))
                                if row is not None and row.get("pass"):
                                    continue
                                failure_stage_counts["EXACT_COVER"] += 1
                                with failures_jsonl.open("a", encoding="utf-8") as stream:
                                    stream.write(json.dumps({
                                        "target_rank": int(item["target_rank"]),
                                        "candidate_index": int(item["candidate_index"]),
                                        "failure_stage": "EXACT_COVER",
                                        "failure_reason": None if row is None else row.get("reason", "Exact COVER failed"),
                                        "stage_summaries": None if row is None else row,
                                    }, ensure_ascii=False) + "\n")
                            print(
                                f"    Batch {chunk_index:02d}/{total_batches:02d} ✓ 重定向={len(chunk_items)} | "
                                f"finalize PASS={len(finalized_items)} REJECT={finalize_reject_count} | "
                                f"精确COVER可达={len(passed_cover)} | {time.perf_counter()-batch_started:.1f}s"
                            )

                            if args.motion_route == "curobo":
                                if not passed_cover:
                                    batch_funnel["wall_s"] = float(time.perf_counter() - batch_started)
                                    batch_funnel["failure_stage_counts"] = dict(failure_stage_counts)
                                    funnel["retarget_batches"].append(batch_funnel)
                                    funnel["exact_cover"] = {
                                        "tested": int(tested_cover),
                                        "pass": int(exact_cover_pass_total),
                                        "reject": int(tested_cover - exact_cover_pass_total),
                                        "pass_rate": float(exact_cover_pass_total / tested_cover) if tested_cover else 0.0,
                                        "latest_batch_subfunnel": exact_subfunnel,
                                    }
                                    write_planning_funnel(cycle_root, funnel)
                                    print("    Route B: exact COVER PASS=0，进入下一批。")
                                    continue

                                rb_cfg = cfg.get("routeB_front_half", {})
                                pool = build_front_half_goal_pool(
                                    client=curobo,
                                    project_root=root,
                                    passed_cover_rows=passed_cover,
                                    q_current=q_current,
                                    config=cfg,
                                    max_candidate_cases=int(rb_cfg.get("max_candidate_cases", 32)),
                                    goals_per_case=int(rb_cfg.get("goals_per_case", 8)),
                                    max_total_goals=int(rb_cfg.get("max_total_goals", 128)),
                                )
                                batch_funnel["routeB_goal_pool"] = {
                                    "goal_count": int(pool.goal_count),
                                    "case_count": int(pool.case_count),
                                    "case_summaries": pool.case_summaries,
                                }
                                funnel["routeB_front_half"]["goal_pool"] = batch_funnel["routeB_goal_pool"]
                                if pool.goal_count == 0:
                                    batch_funnel["wall_s"] = float(time.perf_counter() - batch_started)
                                    batch_funnel["failure_stage_counts"] = dict(failure_stage_counts)
                                    funnel["retarget_batches"].append(batch_funnel)
                                    write_planning_funnel(cycle_root, funnel)
                                    print("    Route B: PREGRASP endpoint goal pool=0，进入下一批。")
                                    continue
                                routeb_goal_pool_path = cycle_root / "routeB_front_half/routeB_front_half_goal_pool.npz"
                                routeb_goal_pool_path, routeb_goal_pool_report = save_front_half_goal_pool(
                                    routeb_goal_pool_path,
                                    pool,
                                )
                                print(
                                    f"    ✓ Route B PREGRASP goal pool | "
                                    f"cases={pool.case_count} goals={pool.goal_count} | {routeb_goal_pool_path}"
                                )
                                batch_funnel["wall_s"] = float(time.perf_counter() - batch_started)
                                batch_funnel["failure_stage_counts"] = dict(failure_stage_counts)
                                funnel["retarget_batches"].append(batch_funnel)
                                funnel["exact_cover"] = {
                                    "tested": int(tested_cover),
                                    "pass": int(exact_cover_pass_total),
                                    "reject": int(tested_cover - exact_cover_pass_total),
                                    "pass_rate": float(exact_cover_pass_total / tested_cover) if tested_cover else 0.0,
                                    "latest_batch_subfunnel": exact_subfunnel,
                                }
                                funnel["routeB_front_half"]["goal_pool_npz"] = str(routeb_goal_pool_path)
                                funnel["routeB_front_half"]["goal_pool_json"] = str(routeb_goal_pool_report)
                                write_planning_funnel(cycle_root, funnel)
                                # Leave the CuroboWorkerClient context before
                                # RobotSegmenter and true 7DOF dense planning.
                                break

                            # Preserve official DGN2 order inside the batch.
                            by_case = {str(Path(item["case_root"]).resolve()): item for item in finalized_items}
                            for cover_row in passed_cover:
                                item = by_case[cover_row["case_root"]]
                                route_started = time.perf_counter()
                                route = plan_flexible_route(
                                    client=curobo,
                                    project_root=root,
                                    case_root=Path(item["case_root"]),
                                    cover_solutions=cover_row["cover_solutions"],
                                    q_current=q_current,
                                    measured=measured,
                                    placement_registry=registry,
                                    config=cfg,
                                    no_planner_collision_check=bool(args.no_planner_collision_check),
                                    block_unknown=bool(cfg.get("block_unknown_space", False)),
                                    diagnostic_disable_home_pre_esdf=bool(home_pre_esdf_bypassed),
                                    diagnostic_disable_home_pre_self_collision=bool(home_pre_self_bypassed),
                                    diagnostic_disable_pre_cover_esdf=bool(pre_cover_esdf_bypassed),
                                    diagnostic_disable_pre_cover_self_collision=bool(pre_cover_self_bypassed),
                                )
                                route["diagnostic_wall_s"] = float(time.perf_counter() - route_started)
                                accumulate_home_pre_stats(funnel, route)
                                accumulate_pre_cover_stats(funnel, route)
                                update_flexible_stage_aggregates(funnel, route)
                                route_report = {
                                    "target_rank": int(item["target_rank"]),
                                    "candidate_index": int(item["candidate_index"]),
                                    "official_score": float(item["official_score"]),
                                    "status": str(route.get("status")),
                                    "failed_stage": None if route.get("status") == "PASS" else route_failure_stage(route),
                                    "reason": route.get("reason"),
                                    "wall_s": route["diagnostic_wall_s"],
                                    "stage_summaries": route.get("stage_summaries", []),
                                }
                                funnel["flexible_route"]["candidate_reports"].append(route_report)
                                if route.get("status") == "PASS":
                                    full_route_pass_count += 1
                                    batch_funnel["full_route_pass"] += 1
                                    selected = {
                                        "target_rank": int(item["target_rank"]),
                                        "candidate_index": int(item["candidate_index"]),
                                        "official_score": float(item["official_score"]),
                                        "case_root": str(Path(item["case_root"]).resolve()),
                                        "route": route,
                                    }
                                    print(
                                        f"    ✓ Flexible Route PASS | rank={selected['target_rank']} "
                                        f"candidate={selected['candidate_index']} | {time.perf_counter()-route_started:.2f}s"
                                    )
                                    print_route_diagnostics(item, route)
                                    _print_route_summary(route)
                                    if not args.diagnostic_full_first_batch:
                                        break
                                    continue
                                full_route_fail_count += 1
                                batch_funnel["full_route_fail"] += 1
                                stage = route_failure_stage(route)
                                failure_stage_counts[stage] += 1
                                with failures_jsonl.open("a", encoding="utf-8") as stream:
                                    stream.write(json.dumps({
                                        "target_rank": int(item["target_rank"]),
                                        "candidate_index": int(item["candidate_index"]),
                                        "failure_stage": route.get("failed_stage", "FLEXIBLE_ROUTE"),
                                        "failure_reason": route.get("reason"),
                                        "stage_summaries": route.get("stage_summaries"),
                                    }, ensure_ascii=False) + "\n")
                                if VERBOSE:
                                    print(
                                        f"    ✗ route rank={item['target_rank']} cand={item['candidate_index']}: "
                                        f"{route.get('reason')}"
                                    )
                            batch_funnel["wall_s"] = float(time.perf_counter() - batch_started)
                            batch_funnel["failure_stage_counts"] = dict(failure_stage_counts)
                            funnel["retarget_batches"].append(batch_funnel)
                            funnel["exact_cover"] = {
                                "tested": int(tested_cover),
                                "pass": int(exact_cover_pass_total),
                                "reject": int(tested_cover - exact_cover_pass_total),
                                "pass_rate": float(exact_cover_pass_total / tested_cover) if tested_cover else 0.0,
                                "latest_batch_subfunnel": exact_subfunnel,
                            }
                            funnel["flexible_route"]["full_route_pass_count"] = int(full_route_pass_count)
                            funnel["flexible_route"]["full_route_fail_count"] = int(full_route_fail_count)
                            funnel["flexible_route"]["failure_stage_counts"] = dict(failure_stage_counts)
                            write_planning_funnel(cycle_root, funnel)
                        print(
                            f"    [cuRobo Batch {chunk_index:02d}/{total_batches:02d}] closed | "
                            f"GPU {gpu_memory_snapshot()}"
                        )
                        if args.diagnostic_full_first_batch:
                            print("    [DIAGNOSTIC] first batch fully evaluated; stopping before Isaac execution.")
                            break
                        if args.motion_route == "curobo" and routeb_goal_pool_path is not None:
                            break
                        if selected is not None:
                            break

                    if args.motion_route == "curobo" and routeb_goal_pool_path is not None:
                        print("[Route B] cuRobo IK worker released before RobotSegmenter/dense backend.")
                        ensure_robot_segmented_depth(
                            project_root=root,
                            capture_dir=capture_root,
                            settings=cfg.get("routeB_front_half", {}).get("robot_segmenter", {}),
                        )
                        excluded_routeb_cases: set[str] = set()
                        routeb_attempt = 0
                        routeb_full_report = None
                        backhalf_pool_report = None
                        pool = None
                        while True:
                            routeb_attempt += 1
                            current_goal_pool = routeb_goal_pool_path
                            if excluded_routeb_cases:
                                current_goal_pool = (
                                    cycle_root
                                    / "routeB_front_half"
                                    / f"routeB_front_half_goal_pool_attempt_{routeb_attempt:02d}.npz"
                                )
                                remaining_goals = write_npz_goal_pool_subset(
                                    routeb_goal_pool_path,
                                    current_goal_pool,
                                    excluded_case_roots=excluded_routeb_cases,
                                )
                                if remaining_goals <= 0:
                                    raise RuntimeError(
                                        "Route B exhausted all front-half goals before full plan PASS"
                                    )
                                print(
                                    f"[Route B] retry front-half with remaining goals={remaining_goals} "
                                    f"after excluding {len(excluded_routeb_cases)} case(s)"
                                )

                            # Route B must not spend true MotionPlanner time on a grasp
                            # whose shared A/B back-half endpoints cannot form a complete
                            # LIFT->TRANSFER->PLACE->RETREAT chain.  This preflight is
                            # endpoint IK only and reuses the Route A task samplers.
                            backhalf_root = cycle_root / "routeB_full"
                            preflight_goal_pool = None
                            preflight_case_root = None
                            preflight_candidate = None
                            preflight_rank = None
                            preflight_pool = None
                            checked_preflight_cases: set[str] = set()
                            with np.load(current_goal_pool, allow_pickle=False) as z:
                                preflight_case_roots = np.asarray(z["case_root"]).astype(str)
                                preflight_candidates = np.asarray(z["candidate_index"], dtype=np.int64)
                                preflight_q_covers = np.asarray(z["q_cover_rad"], dtype=np.float64)
                            with CuroboWorkerClient(
                                root,
                                worker_config=worker_cfg,
                                seeds=int(cfg.get("gpu_ik_seeds", 48)),
                                batch_size=int(cfg.get("gpu_ik_batch_size", 512)),
                            ) as curobo:
                                for local_goal_index, case_root_str in enumerate(preflight_case_roots):
                                    candidate_case_root = Path(case_root_str).resolve()
                                    case_key = str(candidate_case_root)
                                    if case_key in checked_preflight_cases:
                                        continue
                                    checked_preflight_cases.add(case_key)
                                    rank_match = re.search(r"rank_(\d+)", case_key)
                                    rank_value = (
                                        int(rank_match.group(1))
                                        if rank_match
                                        else -1
                                    )
                                    candidate_value = int(preflight_candidates[local_goal_index])
                                    print(
                                        "[Route B][ENDPOINT PREFLIGHT] "
                                        f"rank={rank_value} cand={candidate_value}"
                                    )
                                    pool = build_backhalf_chain_pool(
                                        client=curobo,
                                        project_root=root,
                                        case_root=candidate_case_root,
                                        q_cover_rad=np.asarray(
                                            preflight_q_covers[local_goal_index],
                                            dtype=np.float64,
                                        ),
                                        measured=measured,
                                        placement_registry=registry,
                                        config=cfg,
                                        chain_limit=int(cfg["routeB_full_pipeline"]["backhalf_chain_limit"]),
                                        placement_zone_override=(
                                            color_zone_specs.get(
                                                str(target_selection.get("placement_zone_override", ""))
                                            )
                                            if args.task == "color-sort"
                                            else None
                                        ),
                                    )
                                    if int(pool.chain_count) <= 0:
                                        empty_report = (
                                            backhalf_root
                                            / f"backhalf_chain_pool_preflight_rank_{rank_value:04d}.empty.json"
                                        )
                                        empty_report.parent.mkdir(parents=True, exist_ok=True)
                                        empty_report.write_text(
                                            json.dumps(
                                                {
                                                    "schema_version": 1,
                                                    "stage": "ROUTEB_BACKHALF_ENDPOINT_PREFLIGHT",
                                                    "chain_count": 0,
                                                    "selected_case_root": str(candidate_case_root),
                                                    "selected_candidate": int(candidate_value),
                                                    "target_rank": int(rank_value),
                                                    "summaries": pool.summaries,
                                                },
                                                ensure_ascii=False,
                                                indent=2,
                                            )
                                            + "\n",
                                            encoding="utf-8",
                                        )
                                        for summary in pool.summaries:
                                            print(
                                                "    "
                                                f"{summary.get('stage', 'unknown').upper():<8} "
                                                f"targets={summary.get('target_count', 'NA')} "
                                                f"raw={summary.get('raw_success_target_count', 'NA')} "
                                                f"reachable={summary.get('reachable_target_count', 'NA')} "
                                                f"nodes={summary.get('node_count', 'NA')}"
                                            )
                                        continue
                                    preflight_goal_pool = (
                                        cycle_root
                                        / "routeB_front_half"
                                        / f"routeB_front_half_goal_pool_endpoint_pass_attempt_{routeb_attempt:02d}.npz"
                                    )
                                    write_npz_goal_pool_indices(
                                        current_goal_pool,
                                        preflight_goal_pool,
                                        indices=[int(local_goal_index)],
                                    )
                                    preflight_case_root = candidate_case_root
                                    preflight_candidate = candidate_value
                                    preflight_rank = rank_value
                                    preflight_pool = pool
                                    print(
                                        "[Route B][ENDPOINT PREFLIGHT] PASS | "
                                        f"rank={rank_value} cand={candidate_value} "
                                        f"chains={pool.chain_count}"
                                    )
                                    break
                            if preflight_goal_pool is None or preflight_case_root is None:
                                raise RuntimeError(
                                    "Route B found no complete back-half endpoint chain "
                                    "in the current front-half goal pool"
                                )
                            current_goal_pool = preflight_goal_pool

                            routeb_dense_report = run_routeB_dense_backend(
                                project_root=root,
                                capture_dir=capture_root,
                                goal_pool=current_goal_pool,
                                output_dir=cycle_root / "routeB_front_half/planning",
                                settings=cfg.get("routeB_front_half", {}).get("dense_backend", {}),
                            )
                            if not bool(routeb_dense_report.get("success")):
                                excluded_routeb_cases.add(str(preflight_case_root))
                                print(
                                    "[Route B] front-half failed after endpoint preflight; "
                                    f"exclude rank={preflight_rank} cand={preflight_candidate} and retry"
                                )
                                print(
                                    "    reason="
                                    f"{routeb_dense_report.get('reason', 'unknown')} | "
                                    f"report={cycle_root / 'routeB_front_half/planning/routeB_front_half_report.json'}"
                                )
                                continue
                            required_routeb_outputs = [
                                cycle_root / "routeB_front_half/planning/routeB_front_half_report.json",
                                cycle_root / "routeB_front_half/planning/routeB_front_half_plan.npz",
                                cycle_root / "routeB_front_half/planning/trajectory_right_arm.npz",
                            ]
                            missing_routeb = [str(path) for path in required_routeb_outputs if not path.is_file()]
                            if missing_routeb:
                                raise RuntimeError("Route B front-half missing outputs: " + ", ".join(missing_routeb))
                            selected_rb = routeb_dense_report["selected"]
                            post_rb = routeb_dense_report["postcheck"]
                            traj_rb = routeb_dense_report["trajectory"]
                            routeb_dense_report["goal_pool_npz"] = str(current_goal_pool)
                            routeb_dense_report["goal_pool_json"] = str(routeb_goal_pool_report)
                            routeb_dense_report["front_half_attempt"] = int(routeb_attempt)
                            funnel["routeB_front_half"]["dense_backend"] = routeb_dense_report
                            write_planning_funnel(cycle_root, funnel)
                            planning_result = {
                                "schema_version": 2,
                                "status": "PASS",
                                "architecture": "routeB_front_half_v1",
                                "motion_route": "curobo",
                                "query": query,
                                "total_proposals": total_proposals,
                                "target_candidates": len(candidates),
                                "retargeted_candidate_count": retargeted,
                                "exact_cover_tested": tested_cover,
                                "coarse_prefilter": coarse_report,
                                "map": map_report,
                                "routeB_front_half": routeb_dense_report,
                                "planning_funnel": str(cycle_root / "planning_funnel.json"),
                                "planning_wall_s": time.perf_counter() - cycle_started,
                            }
                            planning_path = cycle_root / "planning_result.json"
                            planning_path.write_text(json.dumps(planning_result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
                            funnel["planning_result"] = str(planning_path)
                            funnel["planning_wall_s"] = float(planning_result["planning_wall_s"])
                            write_planning_funnel(cycle_root, funnel)
                            print("\n========================================")
                            print(" Route B FRONT HALF PASS")
                            print("========================================")
                            leap_stats = funnel.get("routeB_front_half", {}).get("leap_reach", {})
                            print(
                                f"LEAP reach prior : {leap_stats.get('pass_count', 'NA')}/"
                                f"{leap_stats.get('candidate_count', len(candidates))}"
                            )
                            print(f"candidate        : {selected_rb.get('candidate_index')}")
                            print(f"case             : {selected_rb.get('case_root')}")
                            print(f"PREGRASP goals   : {len(routeb_dense_report.get('trials', []))}/{(load_json(Path(routeb_goal_pool_report)).get('goal_count') if routeb_goal_pool_report else 'NA')}")
                            print(f"trajectory       : {traj_rb.get('shape')} | points={traj_rb.get('point_count')}")
                            print(f"duration         : {traj_rb.get('duration_s')}")
                            print(f"min clearance    : {post_rb.get('min_environment_clearance_m')}")
                            print(f"scene_collision  : {post_rb.get('scene_collision_max')}")
                            print(f"cspace           : {post_rb.get('cspace_max')}")
                            selected_case_root = Path(selected_rb["case_root"]).resolve()
                            selected_rank_match = re.search(
                                r"rank_(\d+)",
                                str(selected_case_root),
                            )
                            selected_rank = (
                                int(selected_rank_match.group(1))
                                if selected_rank_match
                                else int(selected_rb.get("target_rank", -1))
                            )

                            print("[Route B] reuse endpoint-preflight back-half chain pool")
                            backhalf_root = cycle_root / "routeB_full"
                            backhalf_pool_path = backhalf_root / "backhalf_chain_pool.npz"
                            if preflight_pool is None or int(preflight_pool.chain_count) <= 0:
                                raise RuntimeError(
                                    "Route B endpoint preflight did not preserve a valid back-half chain pool"
                                )
                            pool = preflight_pool
                            backhalf_pool_path, backhalf_pool_report = save_backhalf_chain_pool(
                                backhalf_pool_path,
                                pool,
                            )
                            print(
                                f"[Route B] back-half endpoint pool PASS | chains={pool.chain_count} | "
                                f"{backhalf_pool_path}"
                            )

                            try:
                                routeb_full_report = run_full_motion_backend(
                                    project_root=root,
                                    capture_dir=capture_root,
                                    query=target_slug,
                                    case_root=selected_case_root,
                                    front_half_dir=cycle_root / "routeB_front_half/planning",
                                    backhalf_pool=backhalf_pool_path,
                                    output_dir=cycle_root / "routeB_full/planning",
                                    settings=cfg["routeB_full_pipeline"],
                                    target_mask_path=target_mask_path,
                                )
                            except RuntimeError as exc:
                                message = str(exc)
                                first_line = message.splitlines()[0] if message.splitlines() else message
                                if len(first_line) > 240:
                                    first_line = first_line[:237] + "..."
                                motion_planner_no_path = (
                                    "MotionPlanner success=false" in message
                                    or "PREGRASP->COVER failed" in message
                                    or "COVER->LIFT failed" in message
                                    or "LIFT->TRANSFER failed" in message
                                    or "TRANSFER->PLACE failed" in message
                                    or "PLACE->RETREAT failed" in message
                                    or "RETREAT->HOME failed" in message
                                )
                                if not motion_planner_no_path:
                                    raise
                                excluded_routeb_cases.add(str(selected_case_root))
                                print(
                                    "[Route B] full MotionPlanner failed for this candidate; "
                                    f"exclude rank={selected_rank} cand={selected_rb['candidate_index']} and retry"
                                )
                                print(f"    reason: {first_line}")
                                continue
                            if not bool(routeb_full_report.get("success")):
                                excluded_routeb_cases.add(str(selected_case_root))
                                summary = routeb_full_report.get("trial_summary") or {}
                                failed_stage = (
                                    routeb_full_report.get("failure_stage")
                                    or summary.get("dominant_failed_stage")
                                    or "UNKNOWN"
                                )
                                counts = summary.get("failed_stage_counts") or {}
                                count_text = ", ".join(
                                    f"{stage}={count}" for stage, count in counts.items()
                                ) or "no stage counts"
                                report_path = (
                                    cycle_root
                                    / "routeB_full/planning/routeB_full_plan_report.json"
                                )
                                print(
                                    "[Route B] full MotionPlanner failed for this candidate; "
                                    f"exclude rank={selected_rank} cand={selected_rb['candidate_index']} and retry"
                                )
                                print(
                                    f"    failed_stage={failed_stage} | {count_text} | report={report_path}"
                                )
                                continue
                            required_full_outputs = [
                                cycle_root / "routeB_full/planning/routeB_execution_manifest.json",
                                cycle_root / "routeB_full/planning/routeB_full_plan_report.json",
                                cycle_root / "routeB_full/planning/traj_pregrasp_to_cover.npz",
                                cycle_root / "routeB_full/planning/traj_cover_to_lift.npz",
                                cycle_root / "routeB_full/planning/traj_lift_to_transfer.npz",
                                cycle_root / "routeB_full/planning/traj_transfer_to_place.npz",
                                cycle_root / "routeB_full/planning/traj_place_to_retreat.npz",
                                cycle_root / "routeB_full/planning/traj_retreat_to_home.npz",
                            ]
                            missing_full = [
                                str(path)
                                for path in required_full_outputs
                                if not path.is_file()
                            ]
                            if missing_full:
                                raise RuntimeError("Route B full plan missing outputs: " + ", ".join(missing_full))
                            break

                        if routeb_full_report is None or pool is None or backhalf_pool_report is None:
                            raise RuntimeError("Route B full planning did not produce a PASS report")

                        funnel["routeB_full_pipeline"] = {
                            "backhalf_pool_npz": str(backhalf_pool_path),
                            "backhalf_pool_json": str(backhalf_pool_report),
                            "full_plan": routeb_full_report,
                        }
                        write_planning_funnel(cycle_root, funnel)

                        planning_result = {
                            "schema_version": 2,
                            "status": "PASS",
                            "architecture": "routeB_full_pipeline_v1",
                            "motion_route": "curobo",
                            "query": query,
                            "total_proposals": total_proposals,
                            "target_candidates": len(candidates),
                            "retargeted_candidate_count": retargeted,
                            "exact_cover_tested": tested_cover,
                            "selected": {
                                "target_rank": selected_rank,
                                "candidate_index": int(selected_rb["candidate_index"]),
                                "case_root": str(selected_case_root),
                            },
                            "coarse_prefilter": coarse_report,
                            "map": map_report,
                            "routeB_front_half": routeb_dense_report,
                            "routeB_backhalf_pool": {
                                "npz": str(backhalf_pool_path),
                                "json": str(backhalf_pool_report),
                                "chain_count": int(pool.chain_count),
                            },
                            "routeB_full_pipeline": routeb_full_report,
                            "planning_funnel": str(cycle_root / "planning_funnel.json"),
                            "planning_wall_s": time.perf_counter() - cycle_started,
                        }
                        planning_path = cycle_root / "planning_result.json"
                        planning_path.write_text(
                            json.dumps(planning_result, ensure_ascii=False, indent=2) + "\n",
                            encoding="utf-8",
                        )
                        funnel["planning_result"] = str(planning_path)
                        funnel["planning_wall_s"] = float(planning_result["planning_wall_s"])
                        write_planning_funnel(cycle_root, funnel)

                        if args.planning_only or not args.sim_execute:
                            print("✓ Route B full planning-only 完成；未执行物理抓取。")
                            return 0

                        print("[8] Route B 同一 Isaac 场景执行 dense cuRobo arm trajectories")
                        execution_root = cycle_root / "execution_routeB"
                        execution = isaac.execute_routeB(
                            manifest_path=routeb_full_report["execution_manifest"],
                            output_dir=execution_root,
                            target_segmentation_id=sim_target_id,
                        )
                        if str(execution.get("status")) != "PASS":
                            failure_stage = str(execution.get("failure_stage") or "UNKNOWN")
                            failure_reason = (
                                execution.get("failure_reason")
                                or execution.get("empty_grasp_reason")
                                or ""
                            )
                            verify_lift = execution.get("verify_lift_mm")
                            max_lift = execution.get("max_object_lift_mm")
                            green = execution.get("final_object_center_inside_green_zone")
                            print("\n✗ Route B 物理执行结果：FAIL")
                            print(f"    failure_stage={failure_stage}")
                            if failure_reason:
                                print(f"    reason={failure_reason}")
                            if verify_lift is not None:
                                print(f"    verified_lift={float(verify_lift):.2f} mm")
                            if max_lift is not None:
                                print(f"    max_lift={float(max_lift):.2f} mm")
                            if green is not None:
                                print(f"    final_green_zone={green}")
                            print(f"    report={execution.get('report')}")
                            safe_home = (
                                failure_stage in {"EMPTY_GRASP", "FINAL_GREEN_ZONE"}
                                or str(execution.get("recovery_status") or "") == "HOME"
                            )
                            if safe_home:
                                if args.task == "color-sort" and target_selection.get("instance_id"):
                                    failed_instance_id = str(target_selection["instance_id"])
                                    failed_targets_current_scene.add(failed_instance_id)
                                    write_color_attempt(
                                        color_root=cycle_root / "color_sort",
                                        status="EXECUTION_FAILED_INSTANCE_SKIPPED",
                                        requested_color=str(target_selection.get("color") or args.target_color),
                                        instance_id=failed_instance_id,
                                        detail=(
                                            f"stage={failure_stage}; recovery="
                                            f"{execution.get('recovery_status') or 'route completed HOME'}; "
                                            f"report={execution.get('report')}"
                                        ),
                                    )
                                    print(
                                        f"    color-sort: HOME已确认，跳过实例 {failed_instance_id}；"
                                        "fresh capture 后尝试另一个同色实例。"
                                    )
                                else:
                                    failed_targets_current_scene.clear()
                                print("    说明：本轮不提交placement；机械臂已回HOME，继续下一轮拍照。")
                                continue
                            print("    说明：执行阶段异常，无法确认已安全回HOME；停止会话。")
                            return 3
                        commit_placement(
                            registry,
                            cycle=cycle,
                            execution=execution,
                            selected={
                                "candidate_index": int(selected_rb["candidate_index"]),
                                "target_rank": selected_rank,
                                "task_type": str(args.task),
                                "color": target_selection.get("color"),
                                "zone_id": target_selection.get("placement_zone_override"),
                                "instance_id": target_selection.get("instance_id"),
                            },
                        )
                        print(
                            f"✓ Route B 物理执行 PASS | 抬升={float(execution['max_object_lift_mm']):.1f}mm "
                            f"| 放置中心在绿色区域={execution['final_object_center_inside_green_zone']}"
                        )
                        print("✓ 机械臂已回 HOME；下一轮拍照前仅静置 1.0s，场景不会重新加载。")
                        print(f"✓ 本轮总耗时={time.perf_counter()-cycle_started:.1f}s")
                        failed_targets_current_scene.clear()
                        continue

                    planning_result = {
                        "schema_version": 2,
                        "status": (
                            "DIAGNOSTIC_FIRST_BATCH_COMPLETE"
                            if args.diagnostic_full_first_batch
                            else "PASS"
                            if selected is not None
                            else "FAIL"
                        ),
                        "architecture": cfg.get("architecture"),
                        "query": query,
                        "total_proposals": total_proposals,
                        "target_candidates": len(candidates),
                        "retargeted_candidate_count": retargeted,
                        "exact_cover_tested": tested_cover,
                        "coarse_prefilter": coarse_report,
                        "map": map_report,
                        "selected": selected,
                        "planning_funnel": str(cycle_root / "planning_funnel.json"),
                        "experimental_collision_bypass": bool(args.experimental_bypass_planner_collision),
                        "diagnostic_full_first_batch": bool(args.diagnostic_full_first_batch),
                        "planning_wall_s": time.perf_counter() - cycle_started,
                    }
                    planning_path = cycle_root / "planning_result.json"
                    planning_path.write_text(json.dumps(planning_result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
                    funnel["planning_result"] = str(planning_path)
                    funnel["planning_wall_s"] = float(planning_result["planning_wall_s"])
                    funnel_path = write_planning_funnel(cycle_root, funnel)
                    print_funnel_summary(funnel)
                    print(f"planning_funnel.json = {funnel_path}")
                except KeyboardInterrupt:
                    planning_result = {
                        "schema_version": 2,
                        "status": "CANCELLED",
                        "architecture": cfg.get("architecture"),
                        "query": query,
                        "planning_wall_s": time.perf_counter() - cycle_started,
                    }
                    planning_path = cycle_root / "planning_result.json"
                    planning_path.write_text(json.dumps(planning_result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
                    print(f"[CANCEL] 已取消当前目标：{query}")
                    print("✓ cuRobo 已释放")
                    print("✓ Isaac 会话继续保留，机械臂仍在 HOME")
                    continue
                except RuntimeError as exc:
                    # During planning the worker deliberately keeps physics frozen
                    # and the arm at the captured HOME state.  For color-sort an
                    # exhausted candidate funnel is therefore not a reason to end
                    # the whole requested-color session: skip this visual instance,
                    # take a fresh capture, and let DINO/SAM choose another one.
                    if (
                        args.task != "color-sort"
                        or not target_selection.get("instance_id")
                        or not is_recoverable_color_planning_failure(str(exc))
                    ):
                        raise
                    failed_instance_id = str(target_selection["instance_id"])
                    failed_targets_current_scene.add(failed_instance_id)
                    color_root = cycle_root / "color_sort"
                    attempt_path = write_color_attempt(
                        color_root=color_root,
                        status="PLANNING_FAILED_INSTANCE_SKIPPED",
                        requested_color=str(target_selection.get("color") or args.target_color),
                        instance_id=failed_instance_id,
                        detail=str(exc).splitlines()[0][:500],
                    )
                    print("\n[COLOR SORT] 当前实例的所有候选/位姿均未生成完整路径。")
                    print(f"    skipped instance : {failed_instance_id}")
                    print("    robot state      : HOME（规划期间未执行机械臂动作）")
                    print("    next action      : fresh capture -> DINO/SAM -> 下一颜色实例")
                    print(f"    attempt          : {attempt_path}")
                    continue

                if args.diagnostic_full_first_batch:
                    print("✓ diagnostic-full-first-batch 完成；未执行 Isaac 物理动作。")
                    return 0

                if selected is None:
                    print(f"\n✗ 本轮未找到完整可行路线；场景保持原样，可重新描述目标或继续尝试。")
                    print(f"  详细日志：{DEBUG_LOG}")
                    if args.task == "color-sort" and target_selection.get("instance_id"):
                        failed_targets_current_scene.add(str(target_selection["instance_id"]))
                        print(
                            f"  color-sort: 本scene暂时跳过失败实例 "
                            f"{target_selection['instance_id']}"
                        )
                    # Persistent Isaac is still paused at the captured state.
                    # The next cycle will only perform the configured 1 s HOME
                    # hold + fresh RGB-D capture; it will NOT reload the scene.
                    continue

                print("\n---------------- 规划结果 ----------------")
                print(f"✓ target rank : {selected['target_rank']}")
                print(f"✓ candidate   : {selected['candidate_index']}")
                print(f"✓ route plan  : {selected['route']['output_npz']}")
                print(f"✓ planning    : {planning_result['planning_wall_s']:.1f}s")
                print("------------------------------------------")

                if args.planning_only or not args.sim_execute:
                    print("✓ Planning-only 完成；未执行物理抓取。")
                    return 0

                print("[8] 同一 Isaac 场景直接执行（不重复加载，不二次 IK）")
                execution_root = cycle_root / "execution"
                execution = isaac.execute(
                    case_root=selected["case_root"],
                    plan_npz=selected["route"]["output_npz"],
                    output_dir=execution_root,
                    target_segmentation_id=sim_target_id,
                )
                execution_status = str(execution.get("status", ""))
                if execution_status == "RECOVERED_FAIL":
                    print("✗ 物理执行失败，但 runtime 已完成恢复；不提交 placement，进入下一轮。")
                    print(f"  failure_stage   : {execution.get('failure_stage')}")
                    print(f"  failure_type    : {execution.get('failure_type')}")
                    print(f"  failure_reason  : {execution.get('failure_reason')}")
                    print(f"  recovery_status : {execution.get('recovery_status')}")
                    print(f"  report          : {execution.get('report')}")
                    continue
                if execution_status != "PASS":
                    print(f"✗ 物理执行失败：{execution.get('report')}")
                    return 3
                commit_placement(registry, cycle=cycle, execution=execution, selected=selected)
                print(
                    f"✓ 物理执行 PASS | 抬升={float(execution['max_object_lift_mm']):.1f}mm "
                    f"| 放置中心在绿色区域={execution['final_object_center_inside_green_zone']}"
                )
                print("✓ 机械臂已回 HOME；下一轮拍照前仅静置 1.0s，场景不会重新加载。")
                print(f"✓ 本轮总耗时={time.perf_counter()-cycle_started:.1f}s")
                failed_targets_current_scene.clear()

    except KeyboardInterrupt:
        print("\n[STOP] 用户中断")
        return 130
    except Exception as exc:
        debug_write(traceback_text := f"{type(exc).__name__}: {exc}")
        print(f"\n✗ ERROR: {traceback_text}")
        print(f"详细日志：{DEBUG_LOG}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
