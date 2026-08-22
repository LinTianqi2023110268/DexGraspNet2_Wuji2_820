#!/usr/bin/env python3
"""Multi-object wrapper around existing build_candidate_case.py."""
from __future__ import annotations
import argparse, json, runpy, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
BUILD = HERE / "build_candidate_case.py"

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--prediction", type=Path, required=True)
    p.add_argument("--network-input", type=Path, required=True)
    p.add_argument("--capture-root", type=Path, required=True)
    p.add_argument("--settled-manifest", type=Path, required=True)
    p.add_argument("--items-json", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()

    items = json.loads(a.items_json.read_text(encoding="utf-8"))
    started = time.perf_counter()
    results = []
    for item in items:
        required = {
            "case_id","case_root","candidate_index","target_label",
            "target_geometry_path",
        }
        missing = sorted(required - set(item))
        if missing:
            raise RuntimeError(f"multi candidate item missing fields: {missing}")

        case_id = str(item["case_id"])
        case_root = Path(item["case_root"]).resolve()
        idx = int(item["candidate_index"])
        target_geometry = Path(item["target_geometry_path"]).resolve()

        sys.argv = [
            str(BUILD),
            "--case-id", case_id,
            "--case-root", str(case_root),
            "--candidate-index", str(idx),
            "--prediction", str(a.prediction),
            "--network-input", str(a.network_input),
            "--capture-root", str(a.capture_root),
            "--settled-manifest", str(a.settled_manifest),
            "--target-geometry", str(target_geometry),
            "--replace",
        ]
        runpy.run_path(str(BUILD), run_name="__main__")

        binding = {
            "schema_version": 1,
            "candidate_index": idx,
            "target_label": int(item["target_label"]),
            "target_id": str(item.get("target_id","")),
            "target_geometry_path": str(target_geometry),
            "target_grasp_mask_path": str(item.get("target_grasp_mask_path","")),
            "target_removal_mask_path": str(item.get("target_removal_mask_path","")),
            "perception_target_json": str(item.get("perception_target_json","")),
            "simulator_identity_used": False,
        }
        (case_root/"01_input/color_target_binding.json").write_text(
            json.dumps(binding, ensure_ascii=False, indent=2)+"\n",
            encoding="utf-8",
        )
        results.append({
            "case_id": case_id,
            "case_root": str(case_root),
            "candidate_index": idx,
            "target_label": int(item["target_label"]),
            "status": "PASS",
        })

    payload = {
        "status":"PASS",
        "candidate_count":len(items),
        "wall_time_s":time.perf_counter()-started,
        "results":results,
    }
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2)+"\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "status":"PASS",
        "candidate_count":len(items),
        "wall_time_s":payload["wall_time_s"],
    }, ensure_ascii=False))

if __name__ == "__main__":
    main()
