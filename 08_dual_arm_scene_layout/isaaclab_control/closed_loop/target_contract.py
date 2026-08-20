from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any


FORBIDDEN_PLANNING_KEYS = {
    "segmentation_id",
    "target_segmentation_id",
    "object_code",
    "target_object_code",
    "object_pool_index",
    "simulation_usd",
    "collision_aabb",
    "collision_aabb_world_min_m",
    "collision_aabb_world_max_m",
    "settled_pose_layout_world",
    "pose_world_object",
}


def _assert_no_sim_identity(value: Any, path: str = "target") -> None:
    """Fail closed if simulator-only target identity leaks into planning metadata."""
    if isinstance(value, dict):
        for key, child in value.items():
            if str(key) in FORBIDDEN_PLANNING_KEYS:
                raise RuntimeError(
                    f"SIM_IDENTITY_LEAK_IN_PLANNING_TARGET: {path}.{key}"
                )
            _assert_no_sim_identity(child, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _assert_no_sim_identity(child, f"{path}[{index}]")


@dataclass(frozen=True)
class PerceptionTarget:
    """
    Capture-local target selected entirely from current perception.

    This object is allowed to enter DGN2 / retarget / Route A / Route B.
    Simulator object identity is deliberately absent.
    """

    capture_id: str
    target_id: str
    task_type: str
    query_canonical: str
    mask_path: str

    color: str | None = None
    placement_zone_override: str | None = None

    source: str = "perception"
    metrics: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> "PerceptionTarget":
        if self.task_type not in {"semantic-grasp", "color-sort"}:
            raise ValueError(f"invalid task_type={self.task_type!r}")

        mask = Path(self.mask_path).expanduser().resolve()
        if not mask.is_file():
            raise FileNotFoundError(mask)

        payload = asdict(self)
        _assert_no_sim_identity(payload)
        return self

    def to_jsonable(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)


def write_perception_target(path: Path, target: PerceptionTarget) -> Path:
    target.validate()
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(target.to_jsonable(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return path
