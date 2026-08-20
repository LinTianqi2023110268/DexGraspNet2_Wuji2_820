from __future__ import annotations

"""Thin production runtime hooks for Route B front-half integration.

This module intentionally has no cuRobo/torch imports so orchestrator.py can
import it in the IsaacLab environment. GPU backends run in `curobo_v2` via
subprocess.
"""

from dataclasses import dataclass
import json
from pathlib import Path
import subprocess
import time
from typing import Any, Sequence

from .reach_contract import ReachOrdering, order_candidates_from_filter


@dataclass
class ReachPrefilterRuntimeResult:
    status: str
    ordering: ReachOrdering
    wall_time_s: float
    output_dir: str | None
    filter_json: str | None
    report_json: str | None
    fallback_reason: str | None = None

    @property
    def ordered_indices(self) -> list[int]:
        return list(self.ordering.ordered_indices)

    def to_jsonable(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "mode": self.ordering.mode,
            "pass_count": self.ordering.pass_count,
            "reject_count": self.ordering.reject_count,
            "direct_count": len(self.ordering.direct_indices),
            "near_region_count": len(
                self.ordering.near_region_indices
            ),
            "wall_time_s": self.wall_time_s,
            "output_dir": self.output_dir,
            "filter_json": self.filter_json,
            "report_json": self.report_json,
            "fallback_reason": self.fallback_reason,
        }


def _resolve(project_root: Path, value: str | Path) -> Path:
    p = Path(value).expanduser()
    return (
        p.resolve()
        if p.is_absolute()
        else (project_root / p).resolve()
    )


def _run_streaming(
    cmd: Sequence[str],
    *,
    cwd: Path,
) -> tuple[int, str]:
    proc = subprocess.Popen(
        [str(x) for x in cmd],
        cwd=str(cwd),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
    )
    lines: list[str] = []
    assert proc.stdout is not None
    for line in proc.stdout:
        stripped = line.lstrip()
        noisy_error = (
            stripped.startswith("Traceback ")
            or stripped.startswith("File ")
            or stripped.startswith("ERROR conda.cli")
        )
        if not (
            stripped.startswith("{")
            or stripped.startswith("[{")
            or noisy_error
        ):
            print(line, end="", flush=True)
        lines.append(line)
    return proc.wait(), "".join(lines)


def _identity_all_order(candidates: Sequence[dict[str, Any]]) -> ReachOrdering:
    indices = list(range(len(candidates)))
    return ReachOrdering(
        ordered_indices=indices,
        pass_indices=indices,
        reject_indices=[],
        direct_indices=[],
        near_region_indices=indices,
        mode="fallback_original_order",
    )


def run_leap_reach_prefilter_runtime(
    *,
    project_root: Path,
    cycle_root: Path,
    query: str,
    candidates: list[dict[str, Any]],
    settings: dict[str, Any],
) -> ReachPrefilterRuntimeResult:
    project_root = Path(project_root).expanduser().resolve()
    cycle_root = Path(cycle_root).expanduser().resolve()

    enabled = bool(settings.get("enabled", True))
    mode = str(settings.get("mode", "priority_then_rescue"))
    fallback_on_error = bool(settings.get("fallback_on_error", True))
    if not enabled:
        return ReachPrefilterRuntimeResult(
            status="DISABLED",
            ordering=_identity_all_order(candidates),
            wall_time_s=0.0,
            output_dir=None,
            filter_json=None,
            report_json=None,
        )

    conda_exe = Path(
        settings.get("conda_exe", "/home/lin/miniconda3/bin/conda")
    ).expanduser().resolve()
    conda_env = str(settings.get("conda_env", "curobo_v2"))
    backend = _resolve(
        project_root,
        settings.get(
            "script",
            "08_dual_arm_scene_layout/isaaclab_control/"
            "curobo_motion_planning_routeB/routeB_front_half_v1/"
            "routeB_front_half/leap_reach_backend.py",
        ),
    )
    bridge = _resolve(
        project_root,
        settings.get(
            "bridge_npz",
            "08_dual_arm_scene_layout/isaaclab_control/closed_loop/"
            "rfs_prototype/calibration_production/"
            "bridge_calibration_bottle512.npz",
        ),
    )
    output_dir = cycle_root / str(
        settings.get(
            "output_subdir",
            "routeB_front_half/leap_reach",
        )
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    input_json = output_dir / "production_candidates.json"
    input_json.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "query": query,
                "candidates": candidates,
            },
            ensure_ascii=False,
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )

    required = (conda_exe, backend, bridge)
    missing = [str(x) for x in required if not x.is_file()]
    if missing:
        reason = "missing LEAP reach runtime files: " + ", ".join(missing)
        if fallback_on_error:
            return ReachPrefilterRuntimeResult(
                status="FALLBACK",
                ordering=_identity_all_order(candidates),
                wall_time_s=0.0,
                output_dir=str(output_dir),
                filter_json=None,
                report_json=None,
                fallback_reason=reason,
            )
        raise FileNotFoundError(reason)

    cmd = [
        str(conda_exe),
        "run",
        "--no-capture-output",
        "-n",
        conda_env,
        "python",
        str(backend),
        "--project-root",
        str(project_root),
        "--cycle-root",
        str(cycle_root),
        "--query",
        str(query),
        "--bridge-npz",
        str(bridge),
        "--input-candidates-json",
        str(input_json),
        "--output-dir",
        str(output_dir),
        "--endpoint-ik-seeds",
        str(int(settings.get("endpoint_ik_seeds", 24))),
        "--endpoint-ik-batch-size",
        str(int(settings.get("endpoint_ik_batch_size", 512))),
        "--coarse-joint-margin-deg",
        str(float(settings.get("coarse_joint_margin_deg", 0.0))),
        "--extra-position-inflation-m",
        str(float(settings.get("extra_position_inflation_m", 0.0))),
        "--extra-orientation-inflation-deg",
        str(float(settings.get("extra_orientation_inflation_deg", 0.0))),
    ]

    print(
        "[LEAP REACH] reach-region-only prefilter | "
        f"candidates={len(candidates)} | mode={mode}",
        flush=True,
    )
    started = time.perf_counter()
    code, output = _run_streaming(cmd, cwd=project_root)
    wall = time.perf_counter() - started

    filter_path = output_dir / "leap_target_reach_filter.json"
    report_path = output_dir / "leap_target_reach_report.json"

    def fallback(reason: str) -> ReachPrefilterRuntimeResult:
        if not fallback_on_error:
            raise RuntimeError(reason)
        print(f"[LEAP REACH FALLBACK] {reason}", flush=True)
        return ReachPrefilterRuntimeResult(
            status="FALLBACK",
            ordering=_identity_all_order(candidates),
            wall_time_s=wall,
            output_dir=str(output_dir),
            filter_json=(
                str(filter_path) if filter_path.exists() else None
            ),
            report_json=(
                str(report_path) if report_path.exists() else None
            ),
            fallback_reason=reason,
        )

    if code != 0:
        useful = [
            line.strip()
            for line in output.splitlines()
            if line.strip()
            and not line.lstrip().startswith("File ")
            and not line.lstrip().startswith("Traceback ")
            and not line.lstrip().startswith("ERROR conda.cli")
        ]
        tail = useful[-1] if useful else "see debug log"
        if len(tail) > 220:
            tail = tail[:217] + "..."
        return fallback(f"backend exited {code}; {tail}")
    if not filter_path.is_file():
        return fallback(f"missing reach filter: {filter_path}")

    try:
        payload = json.loads(
            filter_path.read_text(encoding="utf-8")
        )
        rows = payload.get("rows")
        if not isinstance(rows, list):
            raise RuntimeError("filter rows missing")
        ordering = order_candidates_from_filter(
            candidates,
            rows,
            mode=mode,
        )
    except Exception as exc:
        return fallback(
            f"filter validation failed: {type(exc).__name__}: {exc}"
        )

    print(
        "[LEAP REACH] "
        f"PASS={ordering.pass_count}/{len(candidates)} "
        f"(direct={len(ordering.direct_indices)}, "
        f"near={len(ordering.near_region_indices)}) | "
        f"reject={ordering.reject_count} | wall={wall:.2f}s",
        flush=True,
    )
    return ReachPrefilterRuntimeResult(
        status="PASS",
        ordering=ordering,
        wall_time_s=wall,
        output_dir=str(output_dir),
        filter_json=str(filter_path),
        report_json=(
            str(report_path) if report_path.exists() else None
        ),
    )


def ensure_robot_segmented_depth(
    *,
    project_root: Path,
    capture_dir: Path,
    settings: dict[str, Any],
) -> Path:
    project_root = Path(project_root).expanduser().resolve()
    capture_dir = Path(capture_dir).expanduser().resolve()
    output = capture_dir / "planning/filtered_depth.npy"
    report = capture_dir / "planning/robot_segmentation_report.json"
    robot_mask = capture_dir / "planning/robot_mask.npy"
    rgb_no_robot = capture_dir / "planning/rgb_no_robot.png"

    if all(path.is_file() for path in (output, report, robot_mask, rgb_no_robot)) and bool(
        settings.get("reuse_existing", True)
    ):
        print(f"[Route B] reuse current-cycle RobotSegmenter artifacts: {output.parent}", flush=True)
        return output

    conda_exe = Path(
        settings.get("conda_exe", "/home/lin/miniconda3/bin/conda")
    ).expanduser().resolve()
    env = str(settings.get("conda_env", "curobo_v2"))
    script = _resolve(
        project_root,
        settings.get(
            "script",
            "08_dual_arm_scene_layout/isaaclab_control/perception/"
            "robot_segmentation/run_robot_segmenter_capture.py",
        ),
    )
    cmd = [
        str(conda_exe),
        "run",
        "--no-capture-output",
        "-n",
        env,
        "python",
        str(script),
        "--capture-dir",
        str(capture_dir),
    ]
    print("[Route B] RobotSegmenter -> filtered_depth.npy", flush=True)
    code, output_text = _run_streaming(cmd, cwd=project_root)
    if code != 0:
        raise RuntimeError(
            "RobotSegmenter failed: "
            + " | ".join(output_text.splitlines()[-10:])
        )
    if not output.is_file():
        raise RuntimeError(
            f"RobotSegmenter returned success but output missing: {output}"
        )
    return output


def run_routeB_dense_backend(
    *,
    project_root: Path,
    capture_dir: Path,
    goal_pool: Path,
    output_dir: Path,
    settings: dict[str, Any],
) -> dict[str, Any]:
    project_root = Path(project_root).expanduser().resolve()
    capture_dir = Path(capture_dir).expanduser().resolve()
    goal_pool = Path(goal_pool).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    conda_exe = Path(
        settings.get("conda_exe", "/home/lin/miniconda3/bin/conda")
    ).expanduser().resolve()
    env = str(settings.get("conda_env", "curobo_v2"))
    backend = _resolve(
        project_root,
        settings.get(
            "dense_backend_script",
            "08_dual_arm_scene_layout/isaaclab_control/"
            "curobo_motion_planning_routeB/routeB_front_half_v1/"
            "routeB_front_half/routeB_dense_backend.py",
        ),
    )
    cmd = [
        str(conda_exe),
        "run",
        "--no-capture-output",
        "-n",
        env,
        "python",
        str(backend),
        "--project-root",
        str(project_root),
        "--capture-dir",
        str(capture_dir),
        "--goal-pool",
        str(goal_pool),
        "--output-dir",
        str(output_dir),
        "--max-attempts",
        str(int(settings.get("max_attempts", 2))),
        "--enable-graph-attempt",
        str(int(settings.get("enable_graph_attempt", 1000000))),
        "--num-ik-seeds",
        str(int(settings.get("num_ik_seeds", 32))),
        "--num-trajopt-seeds",
        str(int(settings.get("num_trajopt_seeds", 4))),
        "--interpolation-dt-s",
        str(float(settings.get("interpolation_dt_s", 0.025))),
        "--warmup-iterations",
        str(int(settings.get("warmup_iterations", 1))),
        "--max-goals-to-try",
        str(int(settings.get("max_goals_to_try", 128))),
    ]
    if bool(settings.get("use_cuda_graph", False)):
        cmd.append("--use-cuda-graph")

    print("[Route B] launch true 7DOF current->PREGRASP backend", flush=True)
    code, output = _run_streaming(cmd, cwd=project_root)
    report_path = output_dir / "routeB_front_half_report.json"
    if not report_path.is_file():
        raise RuntimeError(
            "Route B dense backend did not write report; tail="
            + " | ".join(output.splitlines()[-10:])
        )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    # A no-path result is expected at candidate level.  The backend always
    # writes a structured failure report in that case (currently exit code 3),
    # and the orchestrator owns the policy to exclude that candidate and retry
    # the remaining goal pool.  Reserve exceptions for protocol/backend
    # failures: no report, or a nonzero exit paired with a claimed PASS.
    if not bool(report.get("success")):
        return report
    if code != 0:
        raise RuntimeError(
            f"Route B front-half planning failed (exit={code}): "
            f"{report.get('reason', 'unknown')}"
        )
    return report
