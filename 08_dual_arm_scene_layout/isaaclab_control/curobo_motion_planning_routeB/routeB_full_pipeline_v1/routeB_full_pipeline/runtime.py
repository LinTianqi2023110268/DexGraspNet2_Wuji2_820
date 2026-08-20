from __future__ import annotations

import json
from pathlib import Path
import subprocess
from typing import Any


def _stream(cmd: list[str], *, cwd: Path) -> tuple[int, str]:
    """Show stage telemetry, but keep backend tracebacks out of the live UI.

    The complete backend output is still returned to the caller, which records
    it in the per-cycle debug log.  A candidate retry must not fill the
    terminal with a repeated Python traceback or a serialized list of every
    rejected chain.
    """
    proc = subprocess.Popen(
        [str(x) for x in cmd],
        cwd=str(cwd),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
    )
    lines = []
    assert proc.stdout is not None
    traceback_lines = 0
    suppress_traceback = False
    for line in proc.stdout:
        stripped = line.lstrip()
        is_traceback = (
            stripped.startswith("Traceback (most recent call last):")
            or stripped.startswith('File "')
            or stripped.startswith("raise ")
            or stripped.startswith("RuntimeError:")
            or stripped.startswith("ERROR conda.cli.main_run:")
        )
        if is_traceback:
            suppress_traceback = True
            traceback_lines += 1
        elif suppress_traceback and (not stripped or stripped.startswith("During handling")):
            traceback_lines += 1
        elif suppress_traceback:
            suppress_traceback = False
        if not (
            stripped.startswith("{")
            or stripped.startswith("[{")
            or is_traceback
            or suppress_traceback
        ):
            print(line, end="", flush=True)
        lines.append(line)
    code = proc.wait()
    if code and traceback_lines:
        print(
            f"[Route B][backend] 规划后端退出 code={code}；详细 traceback 将保存到 routeB_full_backend.log。",
            flush=True,
        )
    return code, "".join(lines)


def _failure_summary(output: str) -> str:
    """Produce one bounded human-readable reason from backend output."""
    lines = [line.strip() for line in output.splitlines() if line.strip()]
    for line in reversed(lines):
        if line.startswith("RuntimeError:"):
            text = line[len("RuntimeError:"):].strip()
            break
    else:
        text = lines[-1] if lines else "backend emitted no diagnostic"
    if "no back-half chain passed true MotionPlanner" in text:
        text = "所有已测试 back-half endpoint chain 均未通过 MotionPlanner（详情见本轮 debug log）"
    return text[:240] + ("..." if len(text) > 240 else "")


def run_full_motion_backend(
    *,
    project_root: Path,
    capture_dir: Path,
    query: str,
    case_root: Path,
    front_half_dir: Path,
    backhalf_pool: Path,
    output_dir: Path,
    settings: dict[str, Any],
    target_mask_path: Path | None = None,
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    script_value = settings.get(
        "script",
        "08_dual_arm_scene_layout/isaaclab_control/"
        "curobo_motion_planning_routeB/routeB_full_pipeline_v1/"
        "routeB_full_pipeline/full_motion_backend.py",
    )
    script = Path(script_value).expanduser()
    if not script.is_absolute():
        script = (root / script).resolve()
    cmd = [
        str(settings.get("conda_exe", "/home/lin/miniconda3/bin/conda")),
        "run",
        "--no-capture-output",
        "-n",
        str(settings.get("conda_env", "curobo_v2")),
        "python",
        str(script),
        "--project-root",
        str(root),
        "--capture-dir",
        str(Path(capture_dir).resolve()),
        "--query",
        str(query),
        "--case-root",
        str(Path(case_root).resolve()),
        "--front-half-dir",
        str(Path(front_half_dir).resolve()),
        "--backhalf-pool",
        str(Path(backhalf_pool).resolve()),
        "--output-dir",
        str(output_dir),
        "--attachment-padding-m",
        str(float(settings.get("attachment_padding_m", 0.005))),
        "--attachment-min-dim-m",
        str(float(settings.get("attachment_min_dim_m", 0.02))),
        "--attachment-sphere-slots",
        str(int(settings.get("attachment_sphere_slots", 48))),
        "--attachment-sphere-count",
        str(int(settings.get("attachment_sphere_count", 32))),
        "--max-chain-trials",
        str(int(settings.get("max_chain_trials", 16))),
    ]
    if target_mask_path is not None:
        cmd.extend(["--target-mask-path", str(Path(target_mask_path).resolve())])
    if not bool(settings.get("transfer_attachment", True)):
        cmd.append("--disable-transfer-attachment")
    report_path = output_dir / "routeB_full_plan_report.json"
    # Candidate retries share one planning directory.  Never allow a backend
    # crash to make the caller consume the previous candidate's report.
    report_path.unlink(missing_ok=True)
    code, output = _stream(cmd, cwd=root)
    (output_dir / "routeB_full_backend.log").write_text(output, encoding="utf-8")
    if not report_path.is_file():
        raise RuntimeError(
            "full Route B backend produced no report: " + _failure_summary(output)
        )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if code != 0:
        raise RuntimeError(
            f"full Route B planning failed exit={code}: "
            f"{str(report.get('reason', 'unknown'))[:240]}"
        )
    return report


def load_full_plan_report(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))
