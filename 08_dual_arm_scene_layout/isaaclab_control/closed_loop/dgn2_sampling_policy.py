"""
DGN2 dual sampling policy for the closed-loop orchestrator.

Two independent lines are preserved:

- scene_postfilter:
  existing 09_predict_official_leap_target.py
  official cate=False -> scene-wide seeds -> target post-filter.

- target_cate:
  new 09_predict_official_leap_target_cate.py
  official cate=True -> full-scene features -> seeds only inside the single
  non-zero perception-target segmentation category.

No official_core code or perception-mask semantics are changed.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
from typing import Any

import numpy as np


SCENE_POSTFILTER = "scene_postfilter"
TARGET_CATE = "target_cate"
VALID_MODES = {SCENE_POSTFILTER, TARGET_CATE}
ENV_OVERRIDE = "WUJI2_DGN2_SAMPLING_MODE"


@dataclass(frozen=True)
class DGN2SamplingPlan:
    mode: str
    task_type: str
    script: Path
    command: tuple[str, ...]
    network_input: Path
    input_audit: dict[str, Any]

    def to_jsonable(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "mode": self.mode,
            "task_type": self.task_type,
            "script": str(self.script),
            "command": list(self.command),
            "network_input": str(self.network_input),
            "input_audit": self.input_audit,
            "legacy_scene_postfilter_preserved": True,
        }


def resolve_sampling_mode(cfg: dict, task_type: str) -> str:
    block = dict(cfg.get("dgn2_sampling", {}))
    mode = str(
        block.get("mode_by_task", {}).get(
            str(task_type),
            block.get("default_mode", SCENE_POSTFILTER),
        )
    ).strip()

    if bool(block.get("allow_env_override", True)):
        override = os.environ.get(ENV_OVERRIDE, "").strip()
        if override:
            mode = override

    if mode not in VALID_MODES:
        raise RuntimeError(
            f"invalid DGN2 sampling mode {mode!r}; "
            f"expected one of {sorted(VALID_MODES)}"
        )
    return mode


def audit_network_input(
    network_input: Path,
    *,
    expected_target_id: int,
    mode: str,
) -> dict[str, Any]:
    network_input = Path(network_input).resolve()
    if not network_input.is_file():
        raise FileNotFoundError(network_input)

    with np.load(network_input, allow_pickle=False) as z:
        required = {"pc", "seg", "target_segmentation_id"}
        missing = sorted(required - set(z.files))
        if missing:
            raise RuntimeError(
                f"{network_input} missing required arrays: {missing}"
            )
        pc = np.asarray(z["pc"])
        seg = np.asarray(z["seg"], dtype=np.int64)
        target_id = int(np.asarray(z["target_segmentation_id"]).item())

    if pc.shape != (1, 40000, 3):
        raise RuntimeError(f"invalid DGN2 pc shape: {pc.shape}")
    if seg.shape != (1, 40000):
        raise RuntimeError(f"invalid DGN2 seg shape: {seg.shape}")
    if target_id != int(expected_target_id):
        raise RuntimeError(
            "DGN2 target membership id mismatch: "
            f"network={target_id}, expected={expected_target_id}"
        )

    target_count = int(np.count_nonzero(seg == target_id))
    background_count = int(np.count_nonzero(seg == 0))
    nonzero_ids = sorted(int(v) for v in np.unique(seg) if int(v) != 0)

    if target_count < 100:
        raise RuntimeError(
            f"target membership too small for DGN2: {target_count}/40000"
        )
    if mode == TARGET_CATE and nonzero_ids != [target_id]:
        raise RuntimeError(
            "target_cate requires one and only one non-zero seg id: "
            f"target={target_id}, nonzero_ids={nonzero_ids}"
        )
    if mode == TARGET_CATE and background_count <= 0:
        raise RuntimeError(
            "target_cate must preserve full-scene context; "
            "no seg==0 points found"
        )

    return {
        "full_scene_point_count": int(pc.shape[1]),
        "target_segmentation_id": target_id,
        "target_point_count": target_count,
        "target_fraction": float(target_count / pc.shape[1]),
        "background_point_count": background_count,
        "nonzero_segmentation_ids": nonzero_ids,
        "full_scene_context_preserved": background_count > 0,
        "target_only_seed_domain": mode == TARGET_CATE,
    }


def build_sampling_plan(
    *,
    root: Path,
    cfg: dict,
    task_type: str,
    network_python: str | Path,
    dgn_root: Path,
    target_slug: str,
) -> DGN2SamplingPlan:
    root = Path(root).resolve()
    dgn_root = Path(dgn_root).resolve()
    mode = resolve_sampling_mode(cfg, task_type)

    scripts = {
        SCENE_POSTFILTER: (
            root
            / "08_dual_arm_scene_layout/scripts/"
            "09_predict_official_leap_target.py"
        ),
        TARGET_CATE: (
            root
            / "08_dual_arm_scene_layout/scripts/"
            "09_predict_official_leap_target_cate.py"
        ),
    }
    script = scripts[mode]
    if not script.is_file():
        raise FileNotFoundError(script)

    network_input = dgn_root / "network_input.npz"
    audit = audit_network_input(
        network_input,
        expected_target_id=int(cfg["dgn2_target_membership_id"]),
        mode=mode,
    )

    command = (
        str(network_python),
        str(script),
        "--target",
        str(target_slug),
        "--rounds",
        str(int(cfg["dgn2_rounds"])),
        "--input-root",
        str(dgn_root),
    )

    return DGN2SamplingPlan(
        mode=mode,
        task_type=str(task_type),
        script=script,
        command=command,
        network_input=network_input,
        input_audit=audit,
    )


def write_sampling_plan(
    plan: DGN2SamplingPlan,
    output_path: Path,
) -> Path:
    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(plan.to_jsonable(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return output_path
