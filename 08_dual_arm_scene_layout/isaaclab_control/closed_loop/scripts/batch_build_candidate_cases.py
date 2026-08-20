#!/usr/bin/env python3
"""Batch wrapper for existing build_candidate_case.py.

This is glue only: it keeps one Python interpreter alive for a chunk and
executes the existing CLI implementation once per candidate with sys.argv.
"""
from __future__ import annotations

import argparse
import json
import runpy
import sys
import time
from pathlib import Path


HERE = Path(__file__).resolve().parent
BUILD = HERE / "build_candidate_case.py"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--prediction", type=Path, required=True)
    parser.add_argument("--network-input", type=Path, required=True)
    parser.add_argument("--capture-root", type=Path, required=True)
    parser.add_argument("--settled-manifest", type=Path, required=True)
    parser.add_argument("--sim-target-segmentation-id", type=int, default=None)
    parser.add_argument("--target-geometry", type=Path, required=True)
    parser.add_argument("--items-json", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    items = json.loads(args.items_json.read_text(encoding="utf-8"))
    started = time.perf_counter()
    results = []
    for item in items:
        case_id = str(item["case_id"])
        case_root = Path(item["case_root"]).expanduser().resolve()
        candidate_index = int(item["candidate_index"])
        sys.argv = [
            str(BUILD),
            "--case-id", case_id,
            "--case-root", str(case_root),
            "--candidate-index", str(candidate_index),
            "--prediction", str(args.prediction),
            "--network-input", str(args.network_input),
            "--capture-root", str(args.capture_root),
            "--settled-manifest", str(args.settled_manifest),
            "--target-geometry", str(args.target_geometry),
            "--replace",
        ]
        runpy.run_path(str(BUILD), run_name="__main__")
        results.append({
            "case_id": case_id,
            "case_root": str(case_root),
            "candidate_index": candidate_index,
            "status": "PASS",
        })
    out = {
        "status": "PASS",
        "candidate_count": len(items),
        "wall_time_s": time.perf_counter() - started,
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": out["status"],
        "candidate_count": out["candidate_count"],
        "wall_time_s": out["wall_time_s"],
        "output": str(args.output),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
