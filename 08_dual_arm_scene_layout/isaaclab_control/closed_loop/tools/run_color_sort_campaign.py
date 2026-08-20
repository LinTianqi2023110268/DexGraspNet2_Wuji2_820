#!/usr/bin/env python3
"""Run a timestamped, resumable color-sort validation campaign.

This is an experiment organizer, not another grasp implementation.  Every case
invokes the production ``run_closed_loop.sh`` entry with explicit task, color,
route, scene, and output configuration.  All generated sessions, logs, and
machine-readable summaries remain under one campaign root.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime
import json
from pathlib import Path
import re
import subprocess
import sys
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[4]
CONTROL_ROOT = PROJECT_ROOT / "08_dual_arm_scene_layout/isaaclab_control"
DEFAULT_CONFIG = CONTROL_ROOT / "closed_loop/config/closed_loop.json"
DEFAULT_CAMPAIGN_PARENT = CONTROL_ROOT / "outputs/color_sort_campaigns"
DEFAULT_SCENE_PARENT = (
    PROJECT_ROOT
    / "02_training_dataset/data/scene_datasets/"
    "wuji2_train60_100seminal_256view_force_adjusted_legacy_v1/scenes"
)
DEFAULT_CASES = (
    (DEFAULT_SCENE_PARENT / "scene_0000", "red"),
    (DEFAULT_SCENE_PARENT / "scene_0020", "blue"),
    (DEFAULT_SCENE_PARENT / "scene_0065", "red"),
)


@dataclass(frozen=True)
class CampaignCase:
    index: int
    scene_folder: str
    scene_name: str
    target_color: str
    case_id: str


def local_now() -> datetime:
    return datetime.now().astimezone()


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_case(text: str) -> tuple[Path, str]:
    try:
        scene_text, color = text.rsplit(":", 1)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "case must be /absolute/scene_folder:red or :blue"
        ) from exc
    color = color.strip().lower()
    if color not in {"red", "blue"}:
        raise argparse.ArgumentTypeError(f"unsupported target color: {color!r}")
    return Path(scene_text).expanduser().resolve(), color


def validate_cases(raw_cases: Iterable[tuple[Path, str]]) -> list[CampaignCase]:
    cases: list[CampaignCase] = []
    for index, (scene, color) in enumerate(raw_cases, start=1):
        scene = scene.resolve()
        if not (scene / "scene_manifest.json").is_file():
            raise FileNotFoundError(f"scene_manifest.json missing: {scene}")
        case_id = f"case_{index:02d}_{scene.name}_{color}"
        cases.append(
            CampaignCase(
                index=index,
                scene_folder=str(scene),
                scene_name=scene.name,
                target_color=color,
                case_id=case_id,
            )
        )
    if len(cases) != 3:
        raise ValueError(f"three-scene campaign requires exactly 3 cases, got {len(cases)}")
    return cases


def build_effective_config(base: dict[str, Any], *, sessions_root: Path) -> dict[str, Any]:
    cfg = json.loads(json.dumps(base))
    cfg["session_root"] = str(sessions_root.resolve())
    # Automated/headless validation must not launch asynchronous desktop viewers.
    cfg["show_rgb_command"] = []
    cfg["show_overlay_command"] = []
    return cfg


def latest_session(sessions_root: Path) -> Path | None:
    if not sessions_root.is_dir():
        return None
    sessions = sorted(path for path in sessions_root.iterdir() if path.is_dir())
    return sessions[-1] if sessions else None


def tail_lines(path: Path, count: int = 40) -> list[str]:
    if not path.is_file():
        return []
    return path.read_text(encoding="utf-8", errors="replace").splitlines()[-count:]


def summarize_terminal(path: Path) -> dict[str, Any]:
    """Extract a compact failure/execution index without replacing raw logs."""

    lines = (
        path.read_text(encoding="utf-8", errors="replace").splitlines()
        if path.is_file()
        else []
    )
    selected_instances: list[str] = []
    skipped_instances: list[str] = []
    failure_stage_counts: dict[str, int] = {}

    def add_failure(stage: str) -> None:
        failure_stage_counts[stage] = failure_stage_counts.get(stage, 0) + 1

    for line in lines:
        selected = re.search(r"selected=(red|blue)_\d+", line)
        if selected:
            selected_instances.append(selected.group(0).split("=", 1)[1])
        skipped = re.search(r"skipped instance\s*:\s*((?:red|blue)_\d+)", line)
        if skipped:
            skipped_instances.append(skipped.group(1))
        dominant = re.search(r"dominant_stage=([A-Z0-9_]+)", line)
        if dominant:
            add_failure(dominant.group(1))
        if "当前颜色实例没有生成DGN2抓取候选" in line:
            add_failure("DGN2_NO_TARGET_SEED")
        if "NO_PREGRASP_GOAL_WITH_VALID_CUROBO_TRAJECTORY" in line:
            add_failure("CURRENT_TO_PREGRASP")
        if re.search(r"\bPLACE\s+targets=\d+\s+raw=0", line):
            add_failure("PLACE_ENDPOINT_RAW_IK_ZERO")
        if "DINO/SAM目标没有与当前RGB" in line:
            add_failure("DINO_SAM_NO_INSTANCE_MATCH")

    return {
        "selected_instances": list(dict.fromkeys(selected_instances)),
        "skipped_instances": list(dict.fromkeys(skipped_instances)),
        "failure_stage_counts": failure_stage_counts,
        "full_motion_plan_pass_count": sum(
            "Route B FULL MOTION PLAN PASS" in line for line in lines
        ),
        "execution_pass_count": sum(
            "Route B 物理执行成功" in line or "execution      : PASS" in line
            for line in lines
        ),
    }


def summarize_case(
    *,
    case: CampaignCase,
    case_root: Path,
    command: list[str],
    return_code: int,
    started_at: datetime,
    ended_at: datetime,
    terminal_log: Path,
) -> dict[str, Any]:
    sessions_root = case_root / "sessions"
    session = latest_session(sessions_root)
    color_summary = None
    debug_log = None
    if session is not None:
        summary_path = session / "color_sort_summary.json"
        if summary_path.is_file():
            color_summary = load_json(summary_path)
        debug_log = session / "debug.log"
    if color_summary is not None:
        status = str(color_summary.get("status", "UNKNOWN"))
    elif return_code == 0:
        status = "PROCESS_EXIT_0_WITHOUT_COLOR_SUMMARY"
    else:
        status = "PROCESS_FAILED"
    return {
        "schema_version": 1,
        "case": asdict(case),
        "status": status,
        "return_code": int(return_code),
        "started_at": started_at.isoformat(timespec="seconds"),
        "ended_at": ended_at.isoformat(timespec="seconds"),
        "duration_s": float((ended_at - started_at).total_seconds()),
        "command": command,
        "terminal_log": str(terminal_log.resolve()),
        "terminal_summary": summarize_terminal(terminal_log),
        "case_root": str(case_root),
        "session_root": str(session) if session is not None else None,
        "color_sort_summary": color_summary,
        "debug_tail": tail_lines(debug_log, count=12) if debug_log is not None else [],
    }


def write_campaign_report(campaign_root: Path, summary: dict[str, Any]) -> Path:
    """Write the single human-readable campaign index beside the JSON summary."""

    rows = [
        "# Three-scene color-sort campaign",
        "",
        f"- Campaign: `{summary.get('campaign_id', campaign_root.name)}`",
        f"- Updated: `{summary.get('updated_at', 'unknown')}`",
        f"- Status: **{summary.get('status', 'UNKNOWN')}**",
        f"- Canonical root: `{campaign_root}`",
        "",
        "| Case | Scene | Color | Status | Exit | Selected | Main blockers | Session |",
        "|---|---|---|---|---:|---|---|---|",
    ]
    for result in summary.get("results", []):
        case = result.get("case", {})
        terminal = result.get("terminal_summary", {})
        failures = terminal.get("failure_stage_counts", {})
        blockers = ", ".join(
            f"{name}×{count}" for name, count in sorted(failures.items())
        ) or "none"
        selected = ", ".join(terminal.get("selected_instances", [])) or "none"
        session = result.get("session_root") or "none"
        rows.append(
            "| {case_id} | {scene} | {color} | {status} | {exit_code} | "
            "{selected} | {blockers} | `{session}` |".format(
                case_id=case.get("case_id", "unknown"),
                scene=case.get("scene_name", "unknown"),
                color=str(case.get("target_color", "unknown")).upper(),
                status=result.get("status", "UNKNOWN"),
                exit_code=result.get("return_code", "?"),
                selected=selected,
                blockers=blockers,
                session=session,
            )
        )
    rows.extend(
        [
            "",
            "Raw terminal logs and all cycle artifacts remain under each case directory. ",
            "This file is the canonical human-readable index; `campaign_summary.json` is the machine-readable index.",
            "",
        ]
    )
    path = campaign_root / "campaign_report.md"
    path.write_text("\n".join(rows), encoding="utf-8")
    return path


def write_campaign_summary(
    campaign_root: Path,
    *,
    results: list[dict[str, Any]],
    total: int,
) -> dict[str, Any]:
    summary = {
        "schema_version": 1,
        "campaign_id": campaign_root.name,
        "updated_at": local_now().isoformat(timespec="seconds"),
        "status": campaign_status(results, total),
        "completed_cases": len(results),
        "total_cases": total,
        "results": results,
    }
    atomic_write_json(campaign_root / "campaign_summary.json", summary)
    write_campaign_report(campaign_root, summary)
    return summary


def terminal_log_for_previous(case_root: Path, previous: dict[str, Any]) -> Path:
    recorded = str(previous.get("terminal_log") or "").strip()
    if recorded:
        path = Path(recorded)
        if path.is_file():
            return path
    timestamped = sorted(case_root.glob("terminal_*.log"))
    if timestamped:
        return timestamped[-1]
    return case_root / "terminal.log"


def print_banner(title: str) -> None:
    line = "=" * 72
    print(f"\n{line}\n {title}\n{line}", flush=True)


def run_case(
    *,
    case: CampaignCase,
    campaign_root: Path,
    base_config: dict[str, Any],
    seed: int,
    motion_route: str,
    planning_only: bool,
    gui: bool,
) -> dict[str, Any]:
    case_root = campaign_root / "cases" / case.case_id
    case_root.mkdir(parents=True, exist_ok=True)
    config_path = case_root / "effective_closed_loop.json"
    atomic_write_json(
        config_path,
        build_effective_config(base_config, sessions_root=case_root / "sessions"),
    )
    command = [
        str(PROJECT_ROOT / "run_closed_loop.sh"),
        "--config",
        str(config_path),
        "--scene-folder",
        case.scene_folder,
        "--task",
        "color-sort",
        "--target-color",
        case.target_color,
        "--motion-route",
        motion_route,
        "--color-seed",
        str(seed),
    ]
    command.append("--planning-only" if planning_only else "--sim-execute")
    if not gui:
        command.append("--isaac-headless")

    started_at = local_now()
    terminal_path = case_root / f"terminal_{started_at.strftime('%Y%m%d_%H%M%S')}.log"
    print_banner(
        f"COLOR CAMPAIGN {case.index}/3 — {case.scene_name} — {case.target_color.upper()}"
    )
    print(f"case output : {case_root}", flush=True)
    print(f"started     : {started_at.isoformat(timespec='seconds')}", flush=True)
    with terminal_path.open("w", encoding="utf-8") as log:
        log.write("COMMAND: " + " ".join(command) + "\n")
        log.flush()
        process = subprocess.Popen(
            command,
            cwd=PROJECT_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            log.write(line)
            log.flush()
        return_code = process.wait()
    ended_at = local_now()
    result = summarize_case(
        case=case,
        case_root=case_root,
        command=command,
        return_code=return_code,
        started_at=started_at,
        ended_at=ended_at,
        terminal_log=terminal_path,
    )
    atomic_write_json(case_root / "result.json", result)
    print_banner(
        f"CASE RESULT — {case.scene_name} — {result['status']} — exit={return_code}"
    )
    return result


def campaign_status(results: list[dict[str, Any]], total: int) -> str:
    if any(int(row["return_code"]) != 0 for row in results):
        return "FAILED_NEEDS_DIAGNOSIS"
    if len(results) < total:
        return "IN_PROGRESS"
    statuses = {str(row["status"]) for row in results}
    if statuses <= {"COLOR_COMPLETE"}:
        return "PASS"
    if "PARTIAL_COMPLETE" in statuses:
        return "PARTIAL_COMPLETE"
    return "COMPLETE_WITH_NONSTANDARD_STATUS"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--case",
        action="append",
        type=parse_case,
        help="exactly three /absolute/scene_folder:red|blue entries",
    )
    parser.add_argument("--campaign-root", type=Path, default=None)
    parser.add_argument("--output-parent", type=Path, default=DEFAULT_CAMPAIGN_PARENT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--color-seed", type=int, default=42)
    parser.add_argument("--motion-route", choices=("legacy", "curobo"), default="curobo")
    parser.add_argument("--planning-only", action="store_true")
    parser.add_argument("--gui", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--rerun-case",
        action="append",
        default=[],
        help="with --resume, rerun this case_id even when its prior return code is 0",
    )
    args = parser.parse_args()

    cases = validate_cases(args.case or DEFAULT_CASES)
    if args.campaign_root is not None:
        campaign_root = args.campaign_root.expanduser().resolve()
    else:
        stamp = local_now().strftime("%Y%m%d_%H%M%S")
        campaign_root = args.output_parent.expanduser().resolve() / stamp
    if campaign_root.exists() and not args.resume:
        raise FileExistsError(f"campaign root already exists; use --resume: {campaign_root}")
    campaign_root.mkdir(parents=True, exist_ok=True)

    manifest_path = campaign_root / "campaign_manifest.json"
    manifest = {
        "schema_version": 1,
        "campaign_id": campaign_root.name,
        "created_at": local_now().isoformat(timespec="seconds"),
        "project_root": str(PROJECT_ROOT),
        "base_config": str(args.config.resolve()),
        "motion_route": args.motion_route,
        "color_seed": int(args.color_seed),
        "execution": "planning-only" if args.planning_only else "sim-execute",
        "isaac_headless": not bool(args.gui),
        "cases": [asdict(case) for case in cases],
    }
    if not manifest_path.exists():
        atomic_write_json(manifest_path, manifest)
    elif load_json(manifest_path)["cases"] != manifest["cases"]:
        raise RuntimeError("resume case list does not match existing campaign manifest")

    print_banner("THREE-SCENE COLOR-SORT CAMPAIGN")
    print(f"campaign : {campaign_root}")
    for case in cases:
        print(f"  {case.index}. {case.scene_name:<12} {case.target_color.upper():<4} {case.scene_folder}")
    if args.dry_run:
        atomic_write_json(
            campaign_root / "campaign_summary.json",
            {"schema_version": 1, "status": "DRY_RUN", "results": []},
        )
        print("DRY RUN: no Isaac, DGN2, cuRobo, or grasp process started.")
        return 0

    base_config = load_json(args.config.resolve())
    rerun_case_ids = {str(value) for value in args.rerun_case}
    known_case_ids = {case.case_id for case in cases}
    unknown_reruns = rerun_case_ids - known_case_ids
    if unknown_reruns:
        raise ValueError(f"unknown --rerun-case values: {sorted(unknown_reruns)}")
    results: list[dict[str, Any]] = []
    for case in cases:
        case_result_path = campaign_root / "cases" / case.case_id / "result.json"
        if args.resume and case_result_path.is_file():
            previous = load_json(case_result_path)
            previous_log = terminal_log_for_previous(case_result_path.parent, previous)
            previous["terminal_log"] = str(previous_log.resolve())
            previous["terminal_summary"] = summarize_terminal(previous_log)
            atomic_write_json(case_result_path, previous)
            if (
                int(previous.get("return_code", 1)) == 0
                and case.case_id not in rerun_case_ids
            ):
                results.append(previous)
                print(f"[CAMPAIGN] resume skip PASS: {case.case_id}")
                continue
        result = run_case(
            case=case,
            campaign_root=campaign_root,
            base_config=base_config,
            seed=args.color_seed,
            motion_route=args.motion_route,
            planning_only=args.planning_only,
            gui=args.gui,
        )
        results.append(result)
        write_campaign_summary(campaign_root, results=results, total=len(cases))
        if int(result["return_code"]) != 0:
            print("[CAMPAIGN] stopped after failure for diagnosis; rerun with --resume after repair.")
            return int(result["return_code"]) or 1

    final_status = campaign_status(results, len(cases))
    write_campaign_summary(campaign_root, results=results, total=len(cases))
    print_banner(f"THREE-SCENE CAMPAIGN FINAL — {final_status}")
    print(f"summary : {campaign_root / 'campaign_summary.json'}")
    return 0 if final_status in {"PASS", "PARTIAL_COMPLETE"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
