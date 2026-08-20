#!/usr/bin/env python3
"""Closed-loop adapter for local GroundingDINO + SAM with robot hard gates.

GroundingDINO receives the current RobotSegmenter-derived ``rgb_no_robot``;
raw depth/intrinsics/pose are used only for target validation and point-cloud
output.  DINO proposals are advisory until box-level and SAM-level robot gates
plus the SourceZone 3D gate accept one proposal.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


LOCAL_GROUNDED_SAM_ROOT = Path("/home/lin/Projects/分类抓取开源项/03_检测加分割_GroundedSAM")
LOCAL_BACKEND = LOCAL_GROUNDED_SAM_ROOT / "scripts/grounded_sam_to_pointcloud.py"
DINO_RESULT_JSON = "dino_result.json"
WORKSPACE_ROI_XYXY = (170, 0, 970, 700)


DINO_SUBPROCESS_CODE = r"""
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image

backend_path = Path(sys.argv[1])
rgb_path = Path(sys.argv[2])
query = sys.argv[3]
output_path = Path(sys.argv[4])
device_request = sys.argv[5]
roi_x1, roi_y1, roi_x2, roi_y2 = [int(value) for value in sys.argv[6:10]]

spec = importlib.util.spec_from_file_location("local_grounded_sam_backend", backend_path)
backend = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(backend)

device = backend.select_device(device_request)
prompt = backend.normalize_query(query)
print(f"[DINO] query: {query!r} -> prompt: {prompt!r}", flush=True)
print(f"[DINO] loading GroundingDINO on {device}", flush=True)
rgb = np.asarray(Image.open(rgb_path).convert("RGB"))
height, width = rgb.shape[:2]
if not (0 <= roi_x1 < roi_x2 <= width and 0 <= roi_y1 < roi_y2 <= height):
    raise ValueError(f"Invalid DINO ROI {(roi_x1, roi_y1, roi_x2, roi_y2)} for image {(width, height)}")
rgb_roi = rgb[roi_y1:roi_y2, roi_x1:roi_x2].copy()
output_path.parent.mkdir(parents=True, exist_ok=True)
Image.fromarray(rgb_roi).save(output_path.parent / "dino_roi_input.png")
dino, load_report = backend.load_dino(backend.DEFAULT_DINO_CONFIG, backend.DEFAULT_DINO_WEIGHT, device)
boxes, scores, phrases = backend.detect(
    dino,
    rgb_roi,
    prompt,
    box_threshold=0.25,
    text_threshold=0.20,
    device=device,
)
# backend.detect already returns CPU numpy arrays; force a detached CPU-only
# serialization boundary before this CUDA process exits.
boxes = np.asarray(boxes, dtype=np.float32)
boxes[:, [0, 2]] += float(roi_x1)
boxes[:, [1, 3]] += float(roi_y1)
scores = np.asarray(scores, dtype=np.float32)
roi_metadata = {
    "schema_version": 1,
    "roi_xyxy_pixels": [roi_x1, roi_y1, roi_x2, roi_y2],
    "roi_width_height": [roi_x2 - roi_x1, roi_y2 - roi_y1],
    "full_image_width_height": [width, height],
    "dino_input": "caller-provided RGB (normally RobotSegmenter rgb_no_robot)[0:700,170:970]",
    "bbox_contract": "DINO runs on ROI; boxes_xyxy_pixels are converted back to full-image coordinates before SAM",
}
(output_path.parent / "dino_roi_metadata.json").write_text(json.dumps(roi_metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
output = {
    "schema_version": 1,
    "query_original": query,
    "query_groundingdino": prompt,
    "device": device,
    "thresholds": {"box": 0.25, "text": 0.20},
    "workspace_roi": roi_metadata,
    "boxes_xyxy_pixels": boxes.tolist(),
    "scores": scores.tolist(),
    "phrases": [str(value) for value in phrases],
    "selected_detection": None,
    "model_load": load_report,
}
output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(f"[DINO] proposals={len(boxes)}; final selection waits for SAM robot/SourceZone gates", flush=True)
"""


SAM_SUBPROCESS_CODE = r"""
import importlib.util
import contextlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
from PIL import Image
import torch

backend_path = Path(sys.argv[1])
rgb_path = Path(sys.argv[2])
depth_path = Path(sys.argv[3])
intrinsics_path = Path(sys.argv[4])
world_from_camera_path = Path(sys.argv[5])
dino_result_path = Path(sys.argv[6])
output_dir = Path(sys.argv[7])
device_request = sys.argv[8]
robot_mask_path = Path(sys.argv[9])
settled_manifest_path = Path(sys.argv[10])
safety_path = Path(sys.argv[11])
capture_dir = Path(sys.argv[12]).resolve()

spec = importlib.util.spec_from_file_location("local_grounded_sam_backend", backend_path)
backend = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(backend)
safe_spec = importlib.util.spec_from_file_location("perception_target_safety", safety_path)
safety = importlib.util.module_from_spec(safe_spec)
assert safe_spec.loader is not None
safe_spec.loader.exec_module(safety)

device = backend.select_device(device_request)
dino = json.loads(dino_result_path.read_text(encoding="utf-8"))
rgb = np.asarray(Image.open(rgb_path).convert("RGB"))
depth = np.load(depth_path).astype(np.float32)
intrinsic = np.load(intrinsics_path).astype(np.float64)
backend.validate_rgbd(rgb, depth, intrinsic)

boxes = np.asarray(dino["boxes_xyxy_pixels"], dtype=np.float32)
scores = np.asarray(dino["scores"], dtype=np.float32)
phrases = [str(value) for value in dino["phrases"]]
robot_mask = np.load(robot_mask_path).astype(bool)
if robot_mask.shape != depth.shape:
    raise ValueError(f"robot_mask/depth shape mismatch: {robot_mask.shape} vs {depth.shape}")
robot_report_path = robot_mask_path.parent / "robot_segmentation_report.json"
if not robot_report_path.is_file():
    raise FileNotFoundError(f"robot segmentation report missing: {robot_report_path}")
robot_report = json.loads(robot_report_path.read_text(encoding="utf-8"))
safety.assert_current_capture_robot_mask(
    robot_report_capture_dir=robot_report["capture_dir"],
    capture_dir=capture_dir,
)
settled = json.loads(settled_manifest_path.read_text(encoding="utf-8"))
valid_depth, inside_source_zone = safety.source_zone_membership(
    depth_m=depth,
    intrinsics=intrinsic,
    T_world_camera=np.load(world_from_camera_path).astype(np.float64),
    T_world_source_zone=np.asarray(settled["world_from_source_zone"], dtype=np.float64),
    source_zone_size_xy_m=np.asarray(settled["table"]["paper_size_m"], dtype=np.float64),
)

def _vram_snapshot():
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
    lines = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    return "; ".join(f"gpu{index}: used/free MiB={line}" for index, line in enumerate(lines)) or "unavailable"


def _log_vram(label):
    print(f"[VRAM] {label}: {_vram_snapshot()}", flush=True)


precision = "autocast_fp16" if device == "cuda" else "fp32"
print(f"[SAM] precision={precision}", flush=True)
print(f"[SAM] loading SAM ViT-B on {device}", flush=True)
_log_vram("before SAM load")
model = backend.sam_model_registry["vit_b"](checkpoint=str(backend.DEFAULT_SAM_WEIGHT)).to(device)
model.eval()
predictor = backend.SamPredictor(model)
_log_vram("after SAM load")

def _autocast_context():
    if device == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return contextlib.nullcontext()

with torch.inference_mode():
    _log_vram("before set_image")
    with _autocast_context():
        predictor.set_image(rgb)
    _log_vram("after set_image")
    detections = []
    residual_masks = {}
    raw_masks = {}
    for index, (box_xyxy, score, phrase) in enumerate(zip(boxes, scores, phrases)):
        box_row = safety.evaluate_dino_box(xyxy=box_xyxy, robot_mask=robot_mask)
        if not box_row["dino_box_legal"]:
            detections.append({
                "index": int(index),
                "idx": int(index),
                "phrase": phrase,
                "score": float(score),
                "box_xyxy_pixels": [float(value) for value in box_xyxy],
                "sam_predicted_iou": None,
                "robot_overlap_px": None,
                "robot_overlap_fraction": None,
                "valid_depth_fraction": None,
                "source_zone_overlap_fraction": None,
                "residual_mask_px": None,
                "legal": False,
                "reject_reason": box_row["dino_box_reject_reason"],
                **box_row,
            })
            continue
        with _autocast_context():
            masks, quality, _ = predictor.predict(
                box=box_xyxy.astype(np.float32),
                multimask_output=False,
                return_logits=False,
            )
        raw_mask = masks[0].astype(bool)
        safety_row, residual = safety.evaluate_sam_proposal(
            sam_mask=raw_mask,
            robot_mask=robot_mask,
            valid_depth=valid_depth,
            inside_source_zone=inside_source_zone,
        )
        raw_masks[index] = raw_mask
        residual_masks[index] = residual
        detections.append({
            "index": int(index),
            "idx": int(index),
            "phrase": phrase,
            "score": float(score),
            "box_xyxy_pixels": [float(value) for value in box_xyxy],
            "sam_predicted_iou": float(quality[0]),
            **box_row,
            **safety_row,
        })
    legal_proposal_indices = [
        int(row["index"])
        for row in detections
        if bool(row.get("legal")) and int(row["index"]) in residual_masks
    ]
    legal_proposal_masks = (
        np.stack(
            [residual_masks[index] for index in legal_proposal_indices],
            axis=0,
        ).astype(bool)
        if legal_proposal_indices
        else np.zeros((0, *depth.shape), dtype=bool)
    )
    legal_proposal_masks_path = output_dir / "legal_proposal_masks.npz"
    np.savez_compressed(
        legal_proposal_masks_path,
        proposal_indices=np.asarray(legal_proposal_indices, dtype=np.int64),
        masks=legal_proposal_masks,
    )
    archive_row_by_proposal = {
        proposal_index: archive_index
        for archive_index, proposal_index in enumerate(legal_proposal_indices)
    }
    for detection in detections:
        proposal_index = int(detection["index"])
        detection["legal_proposal_mask_archive_index"] = (
            int(archive_row_by_proposal[proposal_index])
            if proposal_index in archive_row_by_proposal
            else None
        )
    _log_vram("after all proposal masks")

selected = safety.select_legal_proposal(detections)
for detection in detections:
    detection["selected"] = int(detection["index"]) == selected
if selected is None:
    dino["selected_detection"] = None
    dino["proposal_audit"] = detections
    dino["selected_reason"] = "NO_LEGAL_TARGET"
    dino_result_path.write_text(json.dumps(dino, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    Image.fromarray(rgb).save(output_dir / "overlay.png")
    report = {
        "schema_version": 1,
        "status": "NO_LEGAL_TARGET",
        "input": {"rgb": str(rgb_path.resolve()), "capture_dir": str(capture_dir)},
        "query_original": dino["query_original"],
        "query_groundingdino": dino["query_groundingdino"],
        "detections": detections,
        "selected_detection": None,
        "selected_reason": "NO_LEGAL_TARGET",
        "legal_proposal_masks": str(legal_proposal_masks_path.resolve()),
        "robot_mask": str(robot_mask_path.resolve()),
        "robot_mask_capture_dir": str(capture_dir),
        "robot_exclusion": "ON",
        "source_zone_gate": "ON",
        "stale_mask_check": "PASS",
    }
    (output_dir / "result.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("[SAM] NO_LEGAL_TARGET: every proposal was rejected by robot/SourceZone/depth gates", flush=True)
    raise SystemExit(0)

raw_mask = raw_masks[selected]
mask = residual_masks[selected]
sam_quality = float(detections[selected]["sam_predicted_iou"])
points_camera, rows, cols, valid_mask = backend.backproject(mask, depth, intrinsic, max_depth=3.0)
colors = rgb[rows, cols]
if len(points_camera) == 0:
    raise RuntimeError("NO_LEGAL_TARGET: selected non-robot SAM residual contains no valid depth")

np.save(output_dir / "sam_raw_mask.npy", raw_mask)
np.save(output_dir / "mask.npy", mask)
Image.fromarray((mask * 255).astype(np.uint8)).save(output_dir / "mask.png")
np.save(output_dir / "target_points_camera.npy", points_camera)
backend.write_ply(output_dir / "target_points_camera.ply", points_camera, colors)

world_from_camera = np.load(world_from_camera_path).astype(np.float64)
points_world = backend.transform_points(points_camera, world_from_camera)
np.save(output_dir / "target_points_world.npy", points_world)
backend.write_ply(output_dir / "target_points_world.ply", points_world, colors)

overlay = backend.draw_overlay(rgb, boxes, scores, phrases, selected, mask)
Image.fromarray(overlay).save(output_dir / "overlay.png")
dino["selected_detection"] = int(selected)
dino["proposal_audit"] = detections
dino["selected_reason"] = "highest_score_legal_after_robot_and_sourcezone_gates"
dino_result_path.write_text(json.dumps(dino, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
report = {
    "schema_version": 1,
    "status": "PASS",
    "input": {
        "rgb": str(rgb_path.resolve()),
        "capture_dir": str(capture_dir),
        "depth_m": str(depth_path.resolve()),
        "intrinsics": str(intrinsics_path.resolve()),
        "T_world_camera": str(world_from_camera_path.resolve()),
        "resolution_hw": list(depth.shape),
    },
    "query_original": dino["query_original"],
    "query_groundingdino": dino["query_groundingdino"],
    "device": device,
    "thresholds": dino["thresholds"],
    "detections": detections,
    "selected_detection": selected,
    "selected_reason": dino["selected_reason"],
    "legal_proposal_masks": str(legal_proposal_masks_path.resolve()),
    "robot_mask": str(robot_mask_path.resolve()),
    "robot_mask_capture_dir": str(capture_dir),
    "robot_exclusion": "ON",
    "source_zone_gate": "ON",
    "stale_mask_check": "PASS",
    "sam_predicted_iou": float(sam_quality),
    "mask_pixels": int(mask.sum()),
    "valid_depth_mask_pixels": int(valid_mask.sum()),
    "target_point_count": int(len(points_camera)),
    "coordinate_contract": {
        "camera": "OpenCV: +x image-right, +y image-down, +z camera-forward",
        "world": "points_world = R_world_camera @ points_camera + t_world_camera",
    },
    "model_load": dino["model_load"],
}
(output_dir / "result.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(f"[SAM] mask={int(mask.sum())} px, valid target cloud={len(points_camera)} points", flush=True)
"""


def _require(path: Path, label: str) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"{label}: {path}")
    return path


def _vram_snapshot() -> str:
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
    values = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    if not values:
        return "unavailable (empty nvidia-smi output)"
    return "; ".join(f"gpu{index}: used/free MiB={line}" for index, line in enumerate(values))


def _log_vram(label: str) -> None:
    print(f"[VRAM] {label}: {_vram_snapshot()}", flush=True)


def _normalize_result(output_root: Path, query: str) -> None:
    result_path = output_root / "result.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if result.get("selected_detection") is None:
        result.update(
            {
                "query": query,
                "backend": {
                    "adapter": "closed_loop.scripts.grounded_sam_backend",
                    "source_project": str(LOCAL_GROUNDED_SAM_ROOT),
                    "detector": "GroundingDINO Swin-T OGC",
                    "segmenter": "Segment Anything ViT-B",
                    "semantic_inputs": ["rgb_no_robot", "text"],
                },
            }
        )
        result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return
    selected_index = int(result["selected_detection"])
    selected = result["detections"][selected_index]
    result.update(
        {
            "query": query,
            "grounding_score": float(selected["score"]),
            "box_xyxy": [float(value) for value in selected["box_xyxy_pixels"]],
            "backend": {
                "adapter": "closed_loop.scripts.grounded_sam_backend",
                "source_project": str(LOCAL_GROUNDED_SAM_ROOT),
                "detector": "GroundingDINO Swin-T OGC",
                "segmenter": "Segment Anything ViT-B",
                "semantic_inputs": ["rgb_no_robot", "text"],
            },
        }
    )
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def run_backend(
    image_path: Path,
    text_query: str,
    output_root: Path,
    *,
    capture_dir: Path,
    robot_mask_path: Path,
    settled_manifest_path: Path,
) -> None:
    image_path = _require(image_path.resolve(), "RGB image")
    capture_root = Path(capture_dir).resolve()
    depth = _require(capture_root / "depth_m.npy", "aligned depth")
    intrinsics = _require(capture_root / "intrinsics.npy", "camera intrinsics")
    world_from_camera = _require(capture_root / "T_world_camera.npy", "camera pose")
    robot_mask_path = _require(robot_mask_path.resolve(), "current-capture robot mask")
    settled_manifest_path = _require(settled_manifest_path.resolve(), "settled scene manifest")
    _require(LOCAL_BACKEND, "local Grounded-SAM backend")

    output_root.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.setdefault("MPLCONFIGDIR", str(output_root / "matplotlib_cache"))
    env.setdefault("PYTHONUNBUFFERED", "1")

    dino_result = output_root / DINO_RESULT_JSON
    dino_command = [
        sys.executable,
        "-c",
        DINO_SUBPROCESS_CODE,
        str(LOCAL_BACKEND),
        str(image_path),
        text_query,
        str(dino_result),
        "auto",
        *[str(value) for value in WORKSPACE_ROI_XYXY],
    ]
    _log_vram("before DINO")
    completed = subprocess.run(dino_command, cwd=LOCAL_GROUNDED_SAM_ROOT, env=env, text=True)
    _log_vram("DINO finished")
    if completed.returncode != 0:
        raise RuntimeError(f"GroundingDINO subprocess failed with exit code {completed.returncode}")
    _require(dino_result, DINO_RESULT_JSON)
    _log_vram("after DINO process exit")

    sam_command = [
        sys.executable,
        "-c",
        SAM_SUBPROCESS_CODE,
        str(LOCAL_BACKEND),
        str(image_path),
        str(depth),
        str(intrinsics),
        str(world_from_camera),
        str(dino_result),
        str(output_root),
        "auto",
        str(robot_mask_path),
        str(settled_manifest_path),
        str(Path(__file__).with_name("perception_target_safety.py").resolve()),
        str(capture_root),
    ]
    _log_vram("before SAM")
    completed = subprocess.run(sam_command, cwd=LOCAL_GROUNDED_SAM_ROOT, env=env, text=True)
    _log_vram("SAM finished")
    if completed.returncode != 0:
        raise RuntimeError(f"SAM subprocess failed with exit code {completed.returncode}")
    _log_vram("after SAM process exit")

    for name in ("overlay.png", "result.json"):
        _require(output_root / name, name)
    _normalize_result(output_root, text_query)
    result = json.loads((output_root / "result.json").read_text(encoding="utf-8"))
    if result.get("status") == "NO_LEGAL_TARGET":
        return
    _require(output_root / "mask.npy", "mask.npy")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--text", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--capture-dir", type=Path, required=True,
                        help="Raw RGB-D capture directory; may differ from --image parent for rgb_no_robot.")
    parser.add_argument("--robot-mask", type=Path, required=True)
    parser.add_argument("--settled-scene-manifest", type=Path, required=True)
    args = parser.parse_args()
    query = args.text.strip()
    if not query:
        raise ValueError("--text must not be empty")
    run_backend(
        args.image,
        query,
        args.output.resolve(),
        capture_dir=args.capture_dir,
        robot_mask_path=args.robot_mask,
        settled_manifest_path=args.settled_scene_manifest,
    )


if __name__ == "__main__":
    main()
