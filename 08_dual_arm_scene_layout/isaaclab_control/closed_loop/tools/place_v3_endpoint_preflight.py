#!/usr/bin/env python3
from __future__ import annotations

"""Offline PLACE V3 endpoint preflight.

This tool reuses the production Route B back-half endpoint builder but stops
before dense MotionPlanner and before Isaac execution.  It is intended to
validate the color-sort PLACE V3 contract:

    ordered near->far slots -> release endpoint IK -> back-half endpoint chain

No simulator target identity is used by the V3 PLACE endpoint generation.
"""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _stage_summary(summaries: list[dict[str, Any]], stage: str) -> dict[str, Any]:
    for row in summaries:
        if row.get("stage") == stage:
            return row
    return {}


def _rank_from_case_root(path: str) -> int | None:
    match = re.search(r"rank_(\d+)", path)
    return int(match.group(1)) if match else None


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--cycle-root", type=Path, required=True)
    parser.add_argument("--goal-pool", type=Path, required=True)
    parser.add_argument("--placement-registry", type=Path, required=True)
    parser.add_argument("--color-zones", type=Path, required=True)
    parser.add_argument("--zone-id", choices=("red_zone", "blue_zone"), required=True)
    parser.add_argument("--scan-limit", type=int, default=8)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    root = args.project_root.expanduser().resolve()
    control = root / "08_dual_arm_scene_layout/isaaclab_control"
    sys.path.insert(0, str(control))
    sys.path.insert(0, str(control / "closed_loop"))
    sys.path.insert(0, str(control / "curobo_motion_planning_routeB/routeB_full_pipeline_v1"))

    from core.bridge import CuroboWorkerClient  # noqa: WPS433
    from core.config import WorkerConfig  # noqa: WPS433
    from routeB_full_pipeline.backhalf_pool import build_backhalf_chain_pool  # noqa: WPS433

    cfg = _load_json(control / "closed_loop/config/closed_loop.json")
    robot_state = _load_json(args.cycle_root / "capture/robot_state.json")
    measured = {
        str(key): float(value)
        for key, value in robot_state["joint_positions_by_name"].items()
    }
    zones = _load_json(args.color_zones)["zones"]
    placement_zone_override = zones[args.zone_id]

    with np.load(args.goal_pool, allow_pickle=False) as z:
        case_roots = np.asarray(z["case_root"]).astype(str)
        candidates = np.asarray(z["candidate_index"], dtype=np.int64)
        q_covers = np.asarray(z["q_cover_rad"], dtype=np.float64)

    worker_cfg = WorkerConfig(
        startup_timeout_s=float(cfg.get("worker_startup_timeout_s", 180.0)),
        request_timeout_s=float(cfg.get("worker_request_timeout_s", 600.0)),
    )

    rows: list[dict[str, Any]] = []
    first_pass: dict[str, Any] | None = None
    seen: set[str] = set()
    scan_limit = max(1, int(args.scan_limit))

    print("")
    print("============================================================")
    print(" PLACE V3 ENDPOINT PREFLIGHT")
    print("============================================================")
    print(f"zone          : {args.zone_id}")
    print(f"goal_pool     : {args.goal_pool}")
    print(f"scan_limit    : {scan_limit}")
    print("dense planner : NOT RUN")
    print("Isaac execute : NOT RUN")
    print("============================================================")

    with CuroboWorkerClient(
        root,
        worker_config=worker_cfg,
        seeds=int(cfg.get("gpu_ik_seeds", 48)),
        batch_size=int(cfg.get("gpu_ik_batch_size", 512)),
    ) as client:
        for case_root_str, candidate, q_cover in zip(case_roots, candidates, q_covers):
            resolved = str(Path(case_root_str).resolve())
            if resolved in seen:
                continue
            seen.add(resolved)
            if len(rows) >= scan_limit:
                break
            rank = _rank_from_case_root(resolved)
            print(f"[PLACE V3] rank={rank} cand={int(candidate)}", flush=True)
            pool = build_backhalf_chain_pool(
                client=client,
                project_root=root,
                case_root=Path(case_root_str),
                q_cover_rad=np.asarray(q_cover, dtype=np.float64),
                measured=measured,
                placement_registry=args.placement_registry,
                config=cfg,
                chain_limit=int(cfg["routeB_full_pipeline"]["backhalf_chain_limit"]),
                placement_zone_override=placement_zone_override,
            )
            stage = {row.get("stage"): row for row in pool.summaries}
            place = _stage_summary(pool.summaries, "place")
            v3 = place.get("place_v3", {})
            accepted_slots = sorted(
                {
                    str(solution.get("slot_id"))
                    for solution in place.get("accepted_solutions", [])
                    if solution.get("slot_id") is not None
                }
            )
            row = {
                "rank": rank,
                "candidate_index": int(candidate),
                "case_root": resolved,
                "chain_count": int(pool.chain_count),
                "lift": stage.get("lift", {}),
                "transfer": stage.get("transfer", {}),
                "place": place,
                "retreat_summaries": [
                    item for item in pool.summaries if item.get("stage") == "retreat"
                ],
                "place_v3": {
                    "slot_count": int(v3.get("slot_count", 0)),
                    "ordered_slots": v3.get("ordered_slots", []),
                    "release_pose_count": int(v3.get("release_pose_count", 0)),
                    "release_pose_count_by_slot": v3.get("release_pose_count_by_slot", {}),
                    "accepted_slots": accepted_slots,
                },
            }
            rows.append(row)
            print(
                "    "
                f"LIFT raw={stage.get('lift', {}).get('raw_success_target_count', 0)} "
                f"TRANSFER raw={stage.get('transfer', {}).get('raw_success_target_count', 0)} "
                f"PLACE raw={place.get('raw_success_target_count', 0)} "
                f"accepted={place.get('reachable_target_count', 0)} "
                f"chains={pool.chain_count}",
                flush=True,
            )
            if pool.chain_count > 0 and first_pass is None:
                first_pass = row

    result = {
        "schema_version": 1,
        "status": "PASS" if first_pass is not None else "FAIL",
        "reason": None if first_pass is not None else "NO_PLACE_V3_ENDPOINT_CHAIN_PASS",
        "project_root": str(root),
        "cycle_root": str(args.cycle_root.resolve()),
        "goal_pool": str(args.goal_pool.resolve()),
        "placement_registry": str(args.placement_registry.resolve()),
        "color_zones": str(args.color_zones.resolve()),
        "zone_id": args.zone_id,
        "placement_zone_override": placement_zone_override,
        "first_pass": first_pass,
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(_jsonable(result), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print("============================================================")
    print(f"RESULT        : {result['status']}")
    if first_pass is not None:
        print(
            "first pass    : "
            f"rank={first_pass['rank']} cand={first_pass['candidate_index']} "
            f"chains={first_pass['chain_count']}"
        )
    print(f"report        : {args.output}")
    print("============================================================")
    return 0 if first_pass is not None else 2


if __name__ == "__main__":
    raise SystemExit(main())
