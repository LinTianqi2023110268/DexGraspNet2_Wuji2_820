#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import re
import sys

FORBIDDEN = {
    "target_segmentation_id",
    "target_object_code",
    "object_pool_index",
    "simulation_usd",
    "object_physics_audit",
    "collision_aabb_world",
    "pose_world_object",
    "surface_points",
}

ALLOW_HINTS = (
    "resolve_sim_target.py",
    "persistent_isaac/",
    "scene_migration",
    "audit_",
    "verification",
)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    args = p.parse_args()
    root = args.root.resolve()

    scan_roots = [
        root / "08_dual_arm_scene_layout/isaaclab_control/closed_loop",
        root / "08_dual_arm_scene_layout/isaaclab_control/curobo_motion_planning_routeB",
        root / "06_leap_to_wuji2_final_pipeline",
    ]

    bad = []
    for scan_root in scan_roots:
        if not scan_root.exists():
            continue
        for path in scan_root.rglob("*.py"):
            rel = path.relative_to(root).as_posix()
            text = path.read_text(encoding="utf-8", errors="replace")
            for token in sorted(FORBIDDEN):
                for m in re.finditer(re.escape(token), text):
                    line = text.count("\n", 0, m.start()) + 1
                    allowed = any(hint in rel for hint in ALLOW_HINTS)
                    bad.append((rel, line, token, allowed))

    print("==================================================")
    print(" PERCEPTION-ONLY PLANNING STATIC AUDIT")
    print("==================================================")
    hard = 0
    for rel, line, token, allowed in bad:
        tag = "ALLOW/REVIEW" if allowed else "PLANNING-REVIEW"
        if not allowed:
            hard += 1
        print(f"{tag:15s} {rel}:{line}  {token}")

    print("--------------------------------------------------")
    print(f"review hits : {len(bad)}")
    print(f"planning-review hits : {hard}")
    print("NOTE: this is a review aid, not a proof by itself.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
