from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

try:
    from ..target_contract import PerceptionTarget, write_perception_target
    from ..perception.target_removal_mask import write_target_removal_artifacts
except (ImportError, ValueError):
    # Standalone/offline replay fallback.
    import sys
    HERE = Path(__file__).resolve()
    CLOSED_LOOP = HERE.parents[1]
    if str(CLOSED_LOOP) not in sys.path:
        sys.path.insert(0, str(CLOSED_LOOP))
    from target_contract import PerceptionTarget, write_perception_target
    from perception.target_removal_mask import write_target_removal_artifacts


def _load_legal_sam_masks(
    grounded_sam_result: dict[str, Any],
    fallback_selected_mask: Path,
) -> list[tuple[int, np.ndarray, str]]:
    value = grounded_sam_result.get("legal_proposal_masks")
    if value:
        archive_path = Path(str(value)).expanduser().resolve()
        if archive_path.is_file():
            with np.load(archive_path, allow_pickle=False) as z:
                indices = np.asarray(z["proposal_indices"], dtype=np.int64)
                masks = np.asarray(z["masks"], dtype=bool)

            if masks.ndim != 3 or len(indices) != len(masks):
                raise RuntimeError(
                    f"invalid GroundedSAM legal proposal archive: {archive_path}"
                )

            return [
                (int(index), np.asarray(mask, dtype=bool), f"{archive_path}#{row}")
                for row, (index, mask) in enumerate(zip(indices, masks))
            ]

    selected_index = grounded_sam_result.get("selected_detection")
    fallback_selected_mask = Path(fallback_selected_mask).resolve()
    if not fallback_selected_mask.is_file():
        raise FileNotFoundError(fallback_selected_mask)
    return [
        (
            -1 if selected_index is None else int(selected_index),
            np.load(fallback_selected_mask).astype(bool),
            str(fallback_selected_mask),
        )
    ]


def _quality_sort_key(row: dict[str, Any]) -> tuple:
    """
    Deliberately NOT a weighted black-box score.

    Ranking is lexicographic:
      1) worst-direction overlap consistency,
      2) Dice,
      3) visible target depth support,
      4) DINO confidence,
      5) absolute overlap.
    """
    mutual = min(
        float(row["sam_overlap_fraction"]),
        float(row["hsv_overlap_fraction"]),
    )
    return (
        -mutual,
        -float(row["dice"]),
        -float(row["target_valid_depth_fraction"]),
        -float(row["dino_score"]),
        -int(row["overlap_px"]),
        int(row["proposal_index"]),
        str(row["instance_id"]),
    )


def build_color_target_pool(
    *,
    capture_root: Path,
    color_root: Path,
    detection_report: dict[str, Any],
    grounded_sam_result: dict[str, Any],
    selected_sam_mask_path: Path,
    requested_color: str,
    gate_cfg: dict[str, Any],
) -> dict[str, Any]:
    """
    Build all legal capture-local targets for one requested color.

    IMPORTANT CONTRACT:
    - DINO/SAM only confirms that an HSV component is a semantically legal
      requested-color object.
    - The DGN2/grasp target geometry is the complete matched DINO/SAM mask.
    - The ESDF target-removal geometry is a separate conservative mask built
      from matched SAM, selected HSV instance neighbourhood, and depth
      consistency.
    - No simulator object identity is read or stored.
    """

    capture_root = Path(capture_root).resolve()
    color_root = Path(color_root).resolve()
    color_root.mkdir(parents=True, exist_ok=True)

    requested_color = str(requested_color).lower()
    if requested_color not in {"red", "blue"}:
        raise ValueError(requested_color)

    required_gate_keys = {
        "min_overlap_px",
        "min_hsv_overlap_fraction",
        "min_sam_overlap_fraction",
        "min_dice",
        "min_intersection_valid_depth_px",
        "min_target_valid_depth_px",
    }
    missing = sorted(required_gate_keys - set(gate_cfg))
    if missing:
        raise RuntimeError(
            f"color_sort.matching config missing required keys: {missing}"
        )

    filtered_depth_path = capture_root / "planning/filtered_depth.npy"
    if not filtered_depth_path.is_file():
        raise FileNotFoundError(filtered_depth_path)

    depth = np.load(filtered_depth_path).astype(np.float32)
    valid_depth = np.isfinite(depth) & (depth > 0.0)

    instances = [
        row
        for row in detection_report[requested_color]["instances"]
        if bool(row.get("inside_source_zone", False))
    ]

    detections = {
        int(row["index"]): row
        for row in grounded_sam_result.get("detections", [])
        if "index" in row
    }

    proposals = _load_legal_sam_masks(
        grounded_sam_result,
        selected_sam_mask_path,
    )

    rows: list[dict[str, Any]] = []
    proposal_by_index: dict[int, np.ndarray] = {}

    for proposal_index, sam_mask, sam_source in proposals:
        proposal_by_index[int(proposal_index)] = sam_mask
        if sam_mask.shape != depth.shape:
            raise RuntimeError(
                f"SAM/depth shape mismatch: {sam_mask.shape} vs {depth.shape}"
            )

        sam_px = int(np.count_nonzero(sam_mask))
        detection = detections.get(int(proposal_index), {})
        dino_score = float(detection.get("score", 0.0))

        for instance in instances:
            hsv_mask_path = Path(str(instance["mask_path"])).resolve()
            hsv_mask = np.load(hsv_mask_path).astype(bool)

            if hsv_mask.shape != sam_mask.shape:
                raise RuntimeError(
                    f"HSV/SAM shape mismatch: {hsv_mask.shape} vs {sam_mask.shape}"
                )

            hsv_px = int(np.count_nonzero(hsv_mask))
            intersection = sam_mask & hsv_mask
            overlap_px = int(np.count_nonzero(intersection))

            sam_fraction = float(overlap_px / max(1, sam_px))
            hsv_fraction = float(overlap_px / max(1, hsv_px))
            dice = float(2.0 * overlap_px / max(1, sam_px + hsv_px))

            intersection_valid_depth_px = int(
                np.count_nonzero(intersection & valid_depth)
            )
            target_valid_depth_px = int(
                np.count_nonzero(hsv_mask & valid_depth)
            )
            target_valid_depth_fraction = float(
                target_valid_depth_px / max(1, hsv_px)
            )

            reject_reasons: list[str] = []

            if overlap_px < int(gate_cfg["min_overlap_px"]):
                reject_reasons.append("OVERLAP_PX")
            if hsv_fraction < float(gate_cfg["min_hsv_overlap_fraction"]):
                reject_reasons.append("HSV_OVERLAP_FRACTION")
            if sam_fraction < float(gate_cfg["min_sam_overlap_fraction"]):
                reject_reasons.append("SAM_OVERLAP_FRACTION")
            if dice < float(gate_cfg["min_dice"]):
                reject_reasons.append("DICE")
            if intersection_valid_depth_px < int(
                gate_cfg["min_intersection_valid_depth_px"]
            ):
                reject_reasons.append("INTERSECTION_VALID_DEPTH")
            if target_valid_depth_px < int(
                gate_cfg["min_target_valid_depth_px"]
            ):
                reject_reasons.append("TARGET_VALID_DEPTH")

            rows.append(
                {
                    "proposal_index": int(proposal_index),
                    "proposal_mask_source": sam_source,
                    "dino_score": dino_score,
                    "instance_id": str(instance["instance_id"]),
                    "hsv_mask_path": str(hsv_mask_path),
                    "sam_pixels": sam_px,
                    "hsv_pixels": hsv_px,
                    "overlap_px": overlap_px,
                    "sam_overlap_fraction": sam_fraction,
                    "hsv_overlap_fraction": hsv_fraction,
                    "dice": dice,
                    "intersection_valid_depth_px":
                        intersection_valid_depth_px,
                    "target_valid_depth_px":
                        target_valid_depth_px,
                    "target_valid_depth_fraction":
                        target_valid_depth_fraction,
                    "inside_source_zone": True,
                    "accepted": not reject_reasons,
                    "reject_reasons": reject_reasons,
                }
            )

    rows.sort(key=_quality_sort_key)

    # One target per current-capture HSV connected component.
    best_by_instance: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not bool(row["accepted"]):
            continue
        instance_id = str(row["instance_id"])
        if instance_id not in best_by_instance:
            best_by_instance[instance_id] = row

    chosen = sorted(best_by_instance.values(), key=_quality_sort_key)

    targets_dir = color_root / "targets"
    targets_dir.mkdir(parents=True, exist_ok=True)

    targets: list[PerceptionTarget] = []

    for rank, row in enumerate(chosen):
        instance_id = str(row["instance_id"])
        target_id = f"{requested_color}_target_{rank:03d}_{instance_id}"

        target_dir = targets_dir / target_id
        target_dir.mkdir(parents=True, exist_ok=True)

        proposal = proposal_by_index[int(row["proposal_index"])]
        hsv_mask_path = Path(row["hsv_mask_path"]).resolve()
        hsv_mask = np.load(hsv_mask_path).astype(bool)

        hsv_instance_mask_path = target_dir / "hsv_instance_mask.npy"
        np.save(hsv_instance_mask_path, hsv_mask)

        removal_audit = write_target_removal_artifacts(
            output_dir=target_dir,
            sam_mask=proposal,
            hsv_mask=hsv_mask,
            depth_m=depth,
            sam_source=str(row["proposal_mask_source"]),
            hsv_source=hsv_mask_path,
            depth_source=filtered_depth_path,
        )

        target_mask_path = target_dir / "target_grasp_mask.npy"

        # Backward-compatibility alias for older tools that still open
        # target_mask.npy.  Its meaning is now the grasp/DGN2 mask, not the
        # HSV color-selection mask.
        np.save(target_dir / "target_mask.npy", proposal.astype(bool))

        intersection_path = target_dir / "sam_hsv_intersection.npy"
        np.save(intersection_path, proposal & hsv_mask)

        metrics = {
            key: value
            for key, value in row.items()
            if key != "hsv_mask_path"
        }
        metrics["sam_hsv_intersection_path"] = str(intersection_path)
        metrics["hsv_instance_mask_path"] = str(hsv_instance_mask_path)
        metrics["target_grasp_mask_path"] = str(Path(removal_audit["target_grasp_mask"]))
        metrics["target_removal_mask_path"] = str(Path(removal_audit["target_removal_mask"]))
        metrics["target_removal_audit_path"] = str(Path(removal_audit["target_removal_audit"]))
        metrics["mask_contract"] = {
            "target_grasp_mask": "matched DINO/SAM mask for DGN2 target points",
            "target_removal_mask": "matched SAM AND selected HSV neighbourhood AND depth consistency for ESDF target removal",
            "hsv_instance_mask": "color selection only; not used as ESDF removal mask",
        }

        target = PerceptionTarget(
            capture_id=capture_root.parent.name,
            target_id=target_id,
            task_type="color-sort",
            query_canonical=f"{requested_color} object",
            mask_path=str(target_mask_path),
            color=requested_color,
            placement_zone_override=f"{requested_color}_zone",
            source="dino_sam_confirmed_hsv_instance",
            metrics=metrics,
        ).validate()

        write_perception_target(target_dir / "target.json", target)
        targets.append(target)

    matrix_path = color_root / "dino_sam_hsv_match_matrix.json"
    matrix_path.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "requested_color": requested_color,
                "gate": gate_cfg,
                "matches": rows,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    pool_path = color_root / "target_pool.json"
    pool_payload = {
        "schema_version": 2,
        "requested_color": requested_color,
        "capture_root": str(capture_root),
        "hsv_instance_count": int(len(instances)),
        "legal_sam_proposal_count": int(len(proposals)),
        "target_count": int(len(targets)),
        "targets": [target.to_jsonable() for target in targets],
        "match_matrix": str(matrix_path),
    }
    pool_path.write_text(
        json.dumps(pool_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return pool_payload
