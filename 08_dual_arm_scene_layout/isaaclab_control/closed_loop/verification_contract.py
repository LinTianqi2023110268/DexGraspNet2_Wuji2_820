from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class SimulationVerificationBinding:
    """
    Simulator identity bound to an already-selected perception target.

    This object is ALLOWED to contain simulator-only identity because its scope
    is execution verification only.

    It must never be consumed by:
      - target selection,
      - 40k/DGN2 input construction,
      - LEAP/Wuji2 retargeting,
      - Route A planning,
      - Route B planning,
      - attachment geometry construction,
      - PLACE endpoint generation.

    It MAY be consumed by:
      - PersistentIsaacClient.execute(...),
      - PersistentIsaacClient.execute_routeB(...),
      - contact sensors,
      - lift verification,
      - final placement verification,
      - experiment/audit reports.
    """

    capture_id: str
    perception_target_id: str
    target_segmentation_id: int

    binding_report: str
    binding_method: str

    object_code: str | None = None
    mask_centroid_world_m: list[float] | None = None
    nearest_origin_distance_m: float | None = None
    second_nearest_origin_distance_m: float | None = None

    scope: str = "EXECUTION_VERIFICATION_ONLY"
    created_after_full_route_pass: bool = True

    def validate(self) -> "SimulationVerificationBinding":
        if self.scope != "EXECUTION_VERIFICATION_ONLY":
            raise RuntimeError(
                f"invalid verification scope: {self.scope!r}"
            )
        if not self.created_after_full_route_pass:
            raise RuntimeError(
                "SIM_VERIFICATION_BINDING_CREATED_BEFORE_FULL_ROUTE_PASS"
            )
        if int(self.target_segmentation_id) < 0:
            raise ValueError(
                f"invalid target_segmentation_id={self.target_segmentation_id}"
            )
        report = Path(self.binding_report).expanduser().resolve()
        if not report.is_file():
            raise FileNotFoundError(report)
        return self

    def to_jsonable(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)


def make_verification_binding(
    *,
    capture_id: str,
    perception_target_id: str,
    full_route_status: str,
    sim_target_result: dict[str, Any],
    binding_report: Path,
    output_path: Path,
) -> SimulationVerificationBinding:
    """
    Convert the existing resolve_sim_target.py result into a strongly-scoped
    verification binding.

    The caller MUST invoke this only after Route A/B has produced FULL ROUTE PASS.
    """
    if str(full_route_status).upper() != "PASS":
        raise RuntimeError(
            "SIM_VERIFICATION_BINDING_REQUIRES_FULL_ROUTE_PASS: "
            f"got {full_route_status!r}"
        )

    required = {
        "segmentation_id",
        "sim_binding_method",
    }
    missing = sorted(required - set(sim_target_result))
    if missing:
        raise RuntimeError(
            f"resolve_sim_target result missing fields: {missing}"
        )

    binding = SimulationVerificationBinding(
        capture_id=str(capture_id),
        perception_target_id=str(perception_target_id),
        target_segmentation_id=int(sim_target_result["segmentation_id"]),
        binding_report=str(Path(binding_report).expanduser().resolve()),
        binding_method=str(sim_target_result["sim_binding_method"]),
        object_code=(
            None
            if sim_target_result.get("object_code") is None
            else str(sim_target_result["object_code"])
        ),
        mask_centroid_world_m=(
            None
            if sim_target_result.get("mask_centroid_world_m") is None
            else [
                float(x)
                for x in sim_target_result["mask_centroid_world_m"]
            ]
        ),
        nearest_origin_distance_m=(
            None
            if sim_target_result.get("nearest_origin_distance_m") is None
            else float(sim_target_result["nearest_origin_distance_m"])
        ),
        second_nearest_origin_distance_m=(
            None
            if sim_target_result.get(
                "second_nearest_origin_distance_m"
            ) is None
            else float(
                sim_target_result["second_nearest_origin_distance_m"]
            )
        ),
    ).validate()

    output_path = Path(output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(binding.to_jsonable(), ensure_ascii=False, indent=2)
        + "\n",
        encoding="utf-8",
    )
    return binding
