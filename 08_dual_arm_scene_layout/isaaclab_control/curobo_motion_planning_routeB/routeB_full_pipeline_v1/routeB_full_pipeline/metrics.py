from __future__ import annotations

from typing import Any
import numpy as np


def to_numpy(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    try:
        return np.asarray(value)
    except Exception:
        return None


def constraint_items(collection: Any) -> list[dict[str, Any]]:
    if collection is None:
        return []
    names = list(getattr(collection, "names", []) or [])
    values = list(getattr(collection, "values", []) or [])
    out = []
    for name, value in zip(names, values):
        arr = to_numpy(value)
        if arr is None:
            out.append({"name": str(name), "available": False})
            continue
        arr = np.asarray(arr, dtype=np.float64)
        flat = arr.reshape(-1)
        out.append(
            {
                "name": str(name),
                "available": True,
                "shape": list(arr.shape),
                "max_value": float(np.max(flat)) if flat.size else None,
                "sum_value": float(np.sum(flat)) if flat.size else None,
                "positive_count": int(np.count_nonzero(flat > 0.0)),
            }
        )
    return out


def raw_constraint_summary(planner: Any, raw_result: Any) -> dict[str, Any]:
    solution = getattr(raw_result, "solution", None)
    if solution is None:
        raise RuntimeError("raw_result.solution unavailable")
    action = solution.reshape(-1, solution.shape[-2], solution.shape[-1])
    expected = int(
        getattr(planner.trajopt_solver.config, "num_seeds", action.shape[0])
    )
    if action.shape[0] == 1 and expected > 1:
        action = action.repeat(expected, 1, 1)
    metrics = planner.trajopt_solver.metrics_rollout.compute_metrics_from_action(
        action
    )
    cc = getattr(metrics, "costs_and_constraints", None)
    constraints = constraint_items(getattr(cc, "constraints", None))
    hybrid = constraint_items(getattr(cc, "hybrid_costs_constraints", None))
    out = {
        "scene_collision_max": None,
        "scene_collision_positive_count": None,
        "cspace_max": None,
        "cspace_positive_count": None,
        "failed_constraints": [],
        "constraints": constraints,
        "hybrid_constraints": hybrid,
    }
    for row in constraints + hybrid:
        name = row.get("name")
        if int(row.get("positive_count") or 0) > 0:
            out["failed_constraints"].append(name)
        if name == "scene_collision":
            out["scene_collision_max"] = row.get("max_value")
            out["scene_collision_positive_count"] = row.get(
                "positive_count"
            )
        elif name == "cspace":
            out["cspace_max"] = row.get("max_value")
            out["cspace_positive_count"] = row.get("positive_count")
    out["failed_constraints"] = sorted(set(out["failed_constraints"]))
    return out
