from __future__ import annotations

import json
from pathlib import Path
from typing import Any
import numpy as np


def enrich_candidate_rows(
    *,
    rows: list[dict[str, Any]],
    prediction_path: Path,
    catalog_path: Path,
) -> list[dict[str, Any]]:
    """Attach capture-local object metadata to globally-ranked DGN2 candidates."""
    catalog = json.loads(Path(catalog_path).resolve().read_text(encoding="utf-8"))
    by_label = {
        int(row["target_label"]): dict(row)
        for row in catalog.get("objects", [])
    }
    if not by_label:
        raise RuntimeError("trusted color catalog is empty")

    with np.load(Path(prediction_path).resolve(), allow_pickle=False) as z:
        labels = np.asarray(z["seed_target_label"], dtype=np.int64)

    out = []
    for row in rows:
        idx = int(row["candidate_index"])
        label = int(labels[idx])
        target = by_label.get(label)
        if target is None:
            raise RuntimeError(
                f"candidate {idx} references unknown target_label={label}"
            )
        enriched = dict(row)
        enriched.update({
            "target_label": label,
            "target_id": str(target["target_id"]),
            "target_grasp_mask_path": str(target["target_grasp_mask_path"]),
            "target_seed_mask_path": str(target["target_seed_mask_path"]),
            "target_removal_mask_path": str(target["target_removal_mask_path"]),
            "target_geometry_path": str(target["target_geometry_path"]),
            "perception_target_json": str(target["perception_target_json"]),
        })
        out.append(enriched)
    return out


def candidate_metadata_by_index(
    rows: list[dict[str, Any]],
) -> dict[int, dict[str, Any]]:
    return {int(row["candidate_index"]): dict(row) for row in rows}
