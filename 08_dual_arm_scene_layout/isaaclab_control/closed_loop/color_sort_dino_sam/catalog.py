from __future__ import annotations

from dataclasses import dataclass, asdict
import json
from pathlib import Path
from typing import Any

import numpy as np

try:
    from ..target_contract import PerceptionTarget, write_perception_target
    from ..planning.perception_target_geometry import build_perception_target_geometry
except (ImportError, ValueError):
    import sys
    HERE = Path(__file__).resolve()
    CLOSED_LOOP = HERE.parents[1]
    if str(CLOSED_LOOP) not in sys.path:
        sys.path.insert(0, str(CLOSED_LOOP))
    from target_contract import PerceptionTarget, write_perception_target
    from planning.perception_target_geometry import build_perception_target_geometry


@dataclass(frozen=True)
class TrustedColorObject:
    target_label: int
    target_id: str
    proposal_index: int
    dino_score: float
    target_grasp_mask_path: str
    target_seed_mask_path: str
    target_removal_mask_path: str
    target_geometry_path: str
    perception_target_json: str
    source_valid_depth_pixels: int
    source_valid_depth_fraction: float

    def to_jsonable(self) -> dict[str, Any]:
        return asdict(self)


def _dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    mask = np.asarray(mask, dtype=bool)
    radius = int(radius)
    if radius <= 0:
        return mask.copy()
    try:
        import cv2
        k = np.ones((2 * radius + 1, 2 * radius + 1), dtype=np.uint8)
        return cv2.dilate(mask.astype(np.uint8), k, iterations=1).astype(bool)
    except Exception:
        out = mask.copy()
        h, w = mask.shape
        ys, xs = np.nonzero(mask)
        for y, x in zip(ys.tolist(), xs.tolist()):
            out[
                max(0, y-radius):min(h, y+radius+1),
                max(0, x-radius):min(w, x+radius+1),
            ] = True
        return out


def _mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = int(np.count_nonzero(a & b))
    union = int(np.count_nonzero(a | b))
    return float(inter / max(1, union))


def build_trusted_color_catalog(
    *,
    capture_root: Path,
    color_root: Path,
    grounded_sam_result_path: Path,
    source_zone_depth_mask_path: Path,
    requested_color: str,
    duplicate_iou_threshold: float = 0.85,
    minimum_source_valid_depth_pixels: int = 100,
    minimum_source_fraction: float = 0.80,
    removal_expand_px: int = 2,
) -> dict[str, Any]:
    """Build ALL trusted requested-color objects from DINO+SAM only.

    There is no HSV and no perception-level "best object" selection.
    """
    capture_root = Path(capture_root).resolve()
    color_root = Path(color_root).resolve()
    color_root.mkdir(parents=True, exist_ok=True)
    requested_color = str(requested_color).lower()
    if requested_color not in {"red", "blue"}:
        raise ValueError(requested_color)

    result = json.loads(
        Path(grounded_sam_result_path).resolve().read_text(encoding="utf-8")
    )
    archive_path = Path(str(result.get("legal_proposal_masks", ""))).resolve()
    if not archive_path.is_file():
        raise FileNotFoundError(
            f"GroundedSAM legal proposal archive missing: {archive_path}"
        )
    with np.load(archive_path, allow_pickle=False) as z:
        proposal_indices = np.asarray(z["proposal_indices"], dtype=np.int64)
        masks = np.asarray(z["masks"], dtype=bool)

    detections = {
        int(row["index"]): dict(row)
        for row in result.get("detections", [])
        if "index" in row
    }

    source_depth = np.load(
        Path(source_zone_depth_mask_path).resolve()
    ).astype(bool)
    filtered_depth = np.load(
        capture_root / "planning/filtered_depth.npy"
    ).astype(np.float32)
    robot_mask = np.load(
        capture_root / "planning/robot_mask.npy"
    ).astype(bool)
    valid_depth = np.isfinite(filtered_depth) & (filtered_depth > 0.0)

    if masks.ndim != 3 or len(masks) != len(proposal_indices):
        raise RuntimeError("invalid legal_proposal_masks.npz")
    if source_depth.shape != filtered_depth.shape:
        raise ValueError("source-zone/depth shape mismatch")

    candidates: list[dict[str, Any]] = []
    for proposal_index, sam_mask in zip(proposal_indices.tolist(), masks):
        proposal_index = int(proposal_index)
        sam = np.asarray(sam_mask, dtype=bool)
        det = detections.get(proposal_index, {})
        if not bool(det.get("legal", True)):
            continue

        valid_sam = sam & valid_depth
        source_seed = sam & source_depth & valid_depth
        sam_valid_px = int(np.count_nonzero(valid_sam))
        source_px = int(np.count_nonzero(source_seed))
        source_fraction = float(source_px / max(1, sam_valid_px))

        reject: list[str] = []
        if source_px < int(minimum_source_valid_depth_pixels):
            reject.append("SOURCE_VALID_DEPTH_PIXELS")
        if source_fraction < float(minimum_source_fraction):
            reject.append("SOURCE_FRACTION")

        candidates.append({
            "proposal_index": proposal_index,
            "dino_score": float(det.get("score", 0.0)),
            "sam_mask": sam,
            "source_seed_mask": source_seed,
            "source_valid_depth_pixels": source_px,
            "source_valid_depth_fraction": source_fraction,
            "reject_reasons": reject,
        })

    # Only near-identical proposals are deduplicated; this is not object ranking.
    accepted: list[dict[str, Any]] = []
    duplicates: list[dict[str, Any]] = []
    for row in sorted(
        (r for r in candidates if not r["reject_reasons"]),
        key=lambda r: (-float(r["dino_score"]), int(r["proposal_index"])),
    ):
        best_iou = 0.0
        duplicate_of = None
        for kept in accepted:
            iou = _mask_iou(row["sam_mask"], kept["sam_mask"])
            if iou > best_iou:
                best_iou = iou
                duplicate_of = kept
        if duplicate_of is not None and best_iou >= float(duplicate_iou_threshold):
            duplicates.append({
                "proposal_index": int(row["proposal_index"]),
                "duplicate_of_proposal_index": int(duplicate_of["proposal_index"]),
                "iou": float(best_iou),
            })
            continue
        accepted.append(row)

    # Capture-local labels are spatial bookkeeping, not confidence ordering.
    def spatial_key(row: dict[str, Any]) -> tuple:
        ys, xs = np.nonzero(row["source_seed_mask"])
        return (
            float(xs.mean()) if len(xs) else float("inf"),
            float(ys.mean()) if len(ys) else float("inf"),
            int(row["proposal_index"]),
        )
    accepted.sort(key=spatial_key)

    targets_dir = color_root / "targets"
    targets_dir.mkdir(parents=True, exist_ok=True)
    label_image = np.zeros(source_depth.shape, dtype=np.int32)
    score_image = np.full(source_depth.shape, -np.inf, dtype=np.float32)
    objects: list[TrustedColorObject] = []

    for label, row in enumerate(accepted, start=1):
        target_id = f"{requested_color}_sam_{label:03d}"
        target_dir = targets_dir / target_id
        target_dir.mkdir(parents=True, exist_ok=True)

        sam = np.asarray(row["sam_mask"], dtype=bool)
        seed_mask = np.asarray(row["source_seed_mask"], dtype=bool)
        removal = _dilate(sam, int(removal_expand_px)) & ~robot_mask

        grasp_path = target_dir / "target_grasp_mask.npy"
        seed_path = target_dir / "target_seed_mask.npy"
        removal_path = target_dir / "target_removal_mask.npy"
        geometry_path = target_dir / "target_geometry.json"
        target_json = target_dir / "target.json"

        np.save(grasp_path, sam)
        np.save(seed_path, seed_mask)
        np.save(removal_path, removal)

        build_perception_target_geometry(
            capture_root=capture_root,
            target_mask_path=grasp_path,
            output_path=geometry_path,
            depth_path=capture_root / "planning/filtered_depth.npy",
            minimum_valid_pixels=50,
        )

        target = PerceptionTarget(
            capture_id=capture_root.parent.name,
            target_id=target_id,
            task_type="color-sort",
            query_canonical=f"{requested_color} object",
            mask_path=str(grasp_path),
            color=requested_color,
            placement_zone_override=f"{requested_color}_zone",
            source="dino_sam_multi_sourcezone",
            metrics={
                "target_label": int(label),
                "proposal_index": int(row["proposal_index"]),
                "dino_score": float(row["dino_score"]),
                "source_valid_depth_pixels": int(row["source_valid_depth_pixels"]),
                "source_valid_depth_fraction": float(row["source_valid_depth_fraction"]),
                "target_seed_mask_path": str(seed_path),
                "target_removal_mask_path": str(removal_path),
                "target_geometry_path": str(geometry_path),
                "perception_contract": "DINO+SAM only; no HSV",
            },
        ).validate()
        write_perception_target(target_json, target)

        # Overlap conflict only: higher DINO score owns the shared target pixel.
        update = seed_mask & (float(row["dino_score"]) > score_image)
        label_image[update] = int(label)
        score_image[update] = float(row["dino_score"])

        objects.append(TrustedColorObject(
            target_label=int(label),
            target_id=target_id,
            proposal_index=int(row["proposal_index"]),
            dino_score=float(row["dino_score"]),
            target_grasp_mask_path=str(grasp_path),
            target_seed_mask_path=str(seed_path),
            target_removal_mask_path=str(removal_path),
            target_geometry_path=str(geometry_path),
            perception_target_json=str(target_json),
            source_valid_depth_pixels=int(row["source_valid_depth_pixels"]),
            source_valid_depth_fraction=float(row["source_valid_depth_fraction"]),
        ))

    label_path = color_root / "target_label_image.npy"
    np.save(label_path, label_image)

    raw_count = len(result.get("detections", []))
    payload = {
        "schema_version": 1,
        "status": (
            "PASS" if objects else
            ("COLOR_COMPLETE_DINO_EMPTY" if raw_count == 0
             else "NO_TRUSTED_COLOR_OBJECT")
        ),
        "perception_mode": "DINO_SAM_MULTI_NO_HSV",
        "requested_color": requested_color,
        "prompt": f"{requested_color} object",
        "raw_dino_proposal_count": int(raw_count),
        "legal_sam_proposal_count": int(len(proposal_indices)),
        "trusted_object_count": int(len(objects)),
        "duplicate_iou_threshold": float(duplicate_iou_threshold),
        "minimum_source_valid_depth_pixels": int(minimum_source_valid_depth_pixels),
        "minimum_source_fraction": float(minimum_source_fraction),
        "removal_expand_px": int(removal_expand_px),
        "target_label_image": str(label_path),
        "objects": [obj.to_jsonable() for obj in objects],
        "duplicates_rejected": duplicates,
        "rejected_proposals": [
            {
                "proposal_index": int(r["proposal_index"]),
                "dino_score": float(r["dino_score"]),
                "reject_reasons": list(r["reject_reasons"]),
                "source_valid_depth_pixels": int(r["source_valid_depth_pixels"]),
                "source_valid_depth_fraction": float(r["source_valid_depth_fraction"]),
            }
            for r in candidates if r["reject_reasons"]
        ],
        "no_hsv": True,
        "no_best_object_selection": True,
        "simulator_identity_used": False,
    }
    catalog_path = color_root / "trusted_color_objects.json"
    catalog_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    payload["catalog"] = str(catalog_path)
    return payload
