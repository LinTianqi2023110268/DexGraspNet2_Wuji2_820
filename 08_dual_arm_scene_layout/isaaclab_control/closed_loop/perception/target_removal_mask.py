from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def _dilate(mask: np.ndarray, radius: int = 5) -> np.ndarray:
    """Small binary dilation helper.

    OpenCV is used when available; the fallback is intentionally simple and
    dependency-free for offline replay environments.
    """
    mask = np.asarray(mask, dtype=bool)
    radius = int(radius)
    if radius <= 0:
        return mask.copy()

    try:
        import cv2  # type: ignore

        kernel = np.ones((2 * radius + 1, 2 * radius + 1), dtype=np.uint8)
        return cv2.dilate(mask.astype(np.uint8), kernel, iterations=1).astype(bool)
    except Exception:
        out = mask.copy()
        ys, xs = np.nonzero(mask)
        h, w = mask.shape
        for y, x in zip(ys, xs):
            y0 = max(0, int(y) - radius)
            y1 = min(h, int(y) + radius + 1)
            x0 = max(0, int(x) - radius)
            x1 = min(w, int(x) + radius + 1)
            out[y0:y1, x0:x1] = True
        return out


def build_target_removal_mask(
    sam_mask: np.ndarray,
    hsv_mask: np.ndarray,
    depth_m: np.ndarray,
    *,
    sam_expand_radius: int = 0,
    hsv_expand_radius: int = 12,
    depth_threshold_m: float = 0.03,
    depth_percentile_low: float = 1.0,
    depth_percentile_high: float = 99.0,
    depth_percentile_padding_m: float = 0.01,
) -> np.ndarray:
    """Build ESDF-only target removal mask for color-sort.

    Contract:
    - matched SAM mask supplies complete target geometry support;
    - selected HSV instance neighbourhood prevents deleting other same-color
      objects or unrelated SAM pixels;
    - depth consistency prevents deleting table/background pixels.
    """
    sam = np.asarray(sam_mask, dtype=bool)
    hsv = np.asarray(hsv_mask, dtype=bool)
    depth = np.asarray(depth_m, dtype=np.float32)

    if sam.shape != hsv.shape or sam.shape != depth.shape:
        raise ValueError(
            f"shape mismatch: sam={sam.shape} hsv={hsv.shape} depth={depth.shape}"
        )

    valid = np.isfinite(depth) & (depth > 0.0)
    hsv_valid = hsv & valid
    if not np.any(hsv_valid):
        return np.zeros_like(sam, dtype=bool)

    seed = sam & hsv & valid
    if np.count_nonzero(seed) >= 8:
        seed_depth = depth[seed]
    else:
        seed_depth = depth[hsv_valid]

    median_depth = float(np.median(seed_depth))
    legacy_gate = valid & (np.abs(depth - median_depth) < float(depth_threshold_m))

    lo, hi = np.percentile(
        seed_depth,
        [float(depth_percentile_low), float(depth_percentile_high)],
    )
    lo = float(lo) - float(depth_percentile_padding_m)
    hi = float(hi) + float(depth_percentile_padding_m)
    adaptive_gate = valid & (depth >= lo) & (depth <= hi)

    # Use the adaptive range when the selected SAM/HSV overlap has a real
    # depth span.  This fixes tilted/deep objects where median +/- 3cm removes
    # most of the object itself.  The mask is still constrained by matched SAM
    # and the selected HSV neighbourhood, so it does not become a same-color or
    # table deletion rule.
    depth_consistent = adaptive_gate

    return (
        _dilate(sam, sam_expand_radius)
        & _dilate(hsv, hsv_expand_radius)
        & depth_consistent
    )


def audit_target_removal(
    *,
    sam_mask: np.ndarray,
    hsv_mask: np.ndarray,
    target_grasp_mask: np.ndarray,
    target_removal_mask: np.ndarray,
    depth_m: np.ndarray | None = None,
    parameters: dict[str, Any] | None = None,
    sources: dict[str, str] | None = None,
) -> dict[str, Any]:
    sam = np.asarray(sam_mask, dtype=bool)
    hsv = np.asarray(hsv_mask, dtype=bool)
    grasp = np.asarray(target_grasp_mask, dtype=bool)
    removal = np.asarray(target_removal_mask, dtype=bool)

    payload: dict[str, Any] = {
        "schema_version": 1,
        "contract": "color-sort separates target_grasp_mask from ESDF target_removal_mask",
        "sam_pixels": int(np.count_nonzero(sam)),
        "hsv_pixels": int(np.count_nonzero(hsv)),
        "grasp_mask_pixels": int(np.count_nonzero(grasp)),
        "removal_pixels": int(np.count_nonzero(removal)),
        "sam_leftover_pixels": int(np.count_nonzero(sam & ~removal)),
        "sam_leftover_fraction": float(
            np.count_nonzero(sam & ~removal) / max(1, np.count_nonzero(sam))
        ),
        "hsv_leftover_pixels": int(np.count_nonzero(hsv & ~removal)),
        "hsv_leftover_fraction": float(
            np.count_nonzero(hsv & ~removal) / max(1, np.count_nonzero(hsv))
        ),
        "removal_over_sam_fraction": float(
            np.count_nonzero(removal & sam) / max(1, np.count_nonzero(sam))
        ),
        "removal_over_hsv_fraction": float(
            np.count_nonzero(removal & hsv) / max(1, np.count_nonzero(hsv))
        ),
        "sam_hsv_overlap_px": int(np.count_nonzero(sam & hsv)),
        "removal_outside_sam_px": int(np.count_nonzero(removal & ~sam)),
        "removal_outside_hsv_px": int(np.count_nonzero(removal & ~hsv)),
    }
    if depth_m is not None:
        depth = np.asarray(depth_m, dtype=np.float32)
        valid = np.isfinite(depth) & (depth > 0.0)
        payload["removal_valid_depth_pixels"] = int(np.count_nonzero(removal & valid))
    if parameters is not None:
        payload["parameters"] = dict(parameters)
    if sources is not None:
        payload["sources"] = dict(sources)
    return payload


def write_target_removal_artifacts(
    *,
    output_dir: Path,
    sam_mask: np.ndarray,
    hsv_mask: np.ndarray,
    depth_m: np.ndarray,
    sam_source: Path | str | None = None,
    hsv_source: Path | str | None = None,
    depth_source: Path | str | None = None,
    sam_expand_radius: int = 0,
    hsv_expand_radius: int = 12,
    depth_threshold_m: float = 0.03,
    depth_percentile_low: float = 1.0,
    depth_percentile_high: float = 99.0,
    depth_percentile_padding_m: float = 0.01,
) -> dict[str, Any]:
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    target_grasp_mask = np.asarray(sam_mask, dtype=bool)
    target_removal_mask = build_target_removal_mask(
        target_grasp_mask,
        hsv_mask,
        depth_m,
        sam_expand_radius=sam_expand_radius,
        hsv_expand_radius=hsv_expand_radius,
        depth_threshold_m=depth_threshold_m,
        depth_percentile_low=depth_percentile_low,
        depth_percentile_high=depth_percentile_high,
        depth_percentile_padding_m=depth_percentile_padding_m,
    )

    grasp_path = output_dir / "target_grasp_mask.npy"
    removal_path = output_dir / "target_removal_mask.npy"
    audit_path = output_dir / "target_removal_audit.json"
    np.save(grasp_path, target_grasp_mask)
    np.save(removal_path, target_removal_mask)
    try:
        from PIL import Image

        Image.fromarray((target_grasp_mask.astype(np.uint8) * 255)).save(
            output_dir / "target_grasp_mask.png"
        )
        Image.fromarray((target_removal_mask.astype(np.uint8) * 255)).save(
            output_dir / "target_removal_mask.png"
        )
    except Exception:
        pass

    parameters = {
        "sam_expand_radius": int(sam_expand_radius),
        "hsv_expand_radius": int(hsv_expand_radius),
        "depth_threshold_m": float(depth_threshold_m),
        "depth_gate": "adaptive_percentile_from_sam_hsv_overlap",
        "depth_percentile_low": float(depth_percentile_low),
        "depth_percentile_high": float(depth_percentile_high),
        "depth_percentile_padding_m": float(depth_percentile_padding_m),
    }
    sources = {
        "sam_mask": "" if sam_source is None else str(Path(sam_source).resolve()),
        "hsv_mask": "" if hsv_source is None else str(Path(hsv_source).resolve()),
        "depth_m": "" if depth_source is None else str(Path(depth_source).resolve()),
    }
    audit = audit_target_removal(
        sam_mask=target_grasp_mask,
        hsv_mask=hsv_mask,
        target_grasp_mask=target_grasp_mask,
        target_removal_mask=target_removal_mask,
        depth_m=depth_m,
        parameters=parameters,
        sources=sources,
    )
    audit["target_grasp_mask"] = str(grasp_path)
    audit["target_removal_mask"] = str(removal_path)
    audit_path.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    audit["target_removal_audit"] = str(audit_path)
    return audit
