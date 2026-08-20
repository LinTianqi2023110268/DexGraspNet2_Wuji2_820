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


def constraint_items(
    collection: Any,
    *,
    sphere_link_names: list[str] | None = None,
) -> list[dict[str, Any]]:
    if collection is None:
        return []
    names = list(getattr(collection, "names", []) or [])
    values = list(getattr(collection, "values", []) or [])
    out: list[dict[str, Any]] = []
    for name, value in zip(names, values):
        arr = to_numpy(value)
        if arr is None:
            out.append({"name": str(name), "available": False})
            continue
        arr = np.asarray(arr, dtype=np.float64)
        flat = arr.reshape(-1)
        positive = flat > 0.0
        worst_flat = int(np.nanargmax(flat)) if flat.size else None
        worst_unravel = (
            list(np.unravel_index(worst_flat, arr.shape))
            if worst_flat is not None else None
        )
        row = {
            "name": str(name),
            "available": True,
            "shape": list(arr.shape),
            "max_value": float(np.nanmax(flat)) if flat.size else None,
            "min_value": float(np.nanmin(flat)) if flat.size else None,
            "sum_value": float(np.nansum(flat)) if flat.size else None,
            "positive_count": int(np.count_nonzero(positive)),
            "worst_timestep": worst_unravel,
            "worst_value": (
                None if worst_flat is None else float(flat[worst_flat])
            ),
        }
        if (
            str(name) == "scene_collision"
            and worst_unravel
            and sphere_link_names
        ):
            sphere_i = int(worst_unravel[-1])
            row["worst_sphere_index"] = sphere_i
            row["worst_link_name"] = (
                sphere_link_names[sphere_i]
                if sphere_i < len(sphere_link_names)
                else None
            )
        out.append(row)
    return out


def metrics_audit(
    metrics: Any,
    *,
    sphere_link_names: list[str] | None = None,
) -> dict[str, Any]:
    if metrics is None:
        return {
            "present": False,
            "feasible": None,
            "constraints": [],
            "failed_constraint_names": [],
        }
    cc = getattr(metrics, "costs_and_constraints", None)
    constraints: list[dict[str, Any]] = []
    hybrid: list[dict[str, Any]] = []
    feasible = None
    sum_constraint = None
    if cc is not None:
        constraints = constraint_items(
            getattr(cc, "constraints", None),
            sphere_link_names=sphere_link_names,
        )
        hybrid = constraint_items(
            getattr(cc, "hybrid_costs_constraints", None),
            sphere_link_names=sphere_link_names,
        )
        try:
            f = to_numpy(
                cc.get_feasible(
                    include_all_hybrid=False,
                    sum_horizon=True,
                )
            )
            feasible = (
                None if f is None
                else bool(np.asarray(f, dtype=bool).all())
            )
        except Exception as exc:
            feasible = f"{type(exc).__name__}: {exc}"
        try:
            s = to_numpy(
                cc.get_sum_constraint(
                    include_all_hybrid=False,
                    sum_horizon=True,
                )
            )
            if s is not None:
                s = np.asarray(s)
                sum_constraint = {
                    "shape": list(s.shape),
                    "max": float(s.max()),
                    "min": float(s.min()),
                    "values": s.reshape(-1).tolist(),
                }
        except Exception as exc:
            sum_constraint = {
                "error": f"{type(exc).__name__}: {exc}"
            }
    failed = [
        row["name"]
        for row in constraints + hybrid
        if int(row.get("positive_count") or 0) > 0
    ]
    return {
        "present": True,
        "type": f"{type(metrics).__module__}.{type(metrics).__name__}",
        "feasible": feasible,
        "sum_constraint": sum_constraint,
        "constraints": constraints,
        "hybrid_costs_constraints": hybrid,
        "failed_constraint_names": sorted(set(failed)),
    }


def raw_constraint_summary(
    planner: Any,
    raw_result: Any,
    *,
    sphere_link_names: list[str] | None = None,
) -> dict[str, Any]:
    solution = getattr(raw_result, "solution", None)
    if solution is None:
        raise RuntimeError("raw_result.solution is unavailable")
    action = solution.reshape(
        -1,
        solution.shape[-2],
        solution.shape[-1],
    )
    expected = int(
        getattr(
            planner.trajopt_solver.config,
            "num_seeds",
            action.shape[0],
        )
    )
    if action.shape[0] == 1 and expected > 1:
        action = action.repeat(expected, 1, 1)
    metrics = (
        planner.trajopt_solver.metrics_rollout
        .compute_metrics_from_action(action)
    )
    audit = metrics_audit(
        metrics,
        sphere_link_names=sphere_link_names,
    )

    summary: dict[str, Any] = {
        "raw_metrics": audit,
        "scene_collision_max": None,
        "scene_collision_positive_count": None,
        "cspace_max": None,
        "cspace_positive_count": None,
        "failed_constraints": list(
            audit.get("failed_constraint_names", [])
        ),
    }
    for row in audit.get("constraints", []) + audit.get(
        "hybrid_costs_constraints", []
    ):
        name = row.get("name")
        if name == "scene_collision":
            summary["scene_collision_max"] = row.get("max_value")
            summary["scene_collision_positive_count"] = row.get(
                "positive_count"
            )
        elif name == "cspace":
            summary["cspace_max"] = row.get("max_value")
            summary["cspace_positive_count"] = row.get(
                "positive_count"
            )
    return summary
