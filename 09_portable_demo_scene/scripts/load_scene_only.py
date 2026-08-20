#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'08_dual_arm_scene_layout/isaaclab_control/closed_loop'))
from persistent_isaac import PersistentIsaacClient  # noqa: E402

def main() -> int:
    ap=argparse.ArgumentParser(description='Load the portable demo scene in Persistent Isaac, then shut down. No grasp/perception/planning is run.')
    ap.add_argument('--headless', action='store_true')
    ap.add_argument('--hold-s', type=float, default=3.0)
    args=ap.parse_args()
    scene = ROOT/'09_portable_demo_scene/scene/portable_scene_manifest.json'
    runtime = ROOT/'09_portable_demo_scene/config/persistent_closed_loop.json'
    print('PORTABLE DEMO LOAD-ONLY')
    print('scene:', scene)
    print('headless:', args.headless)
    client = PersistentIsaacClient(
        project_root=ROOT,
        scene_manifest=scene,
        runtime_config=runtime,
        headless=args.headless,
        verbose=True,
        task='semantic-grasp',
        scene_migration_audit=True,
    )
    try:
        print('Persistent Isaac scene loaded. Holding %.1fs; no robot motion.' % args.hold_s, flush=True)
        # A ping proves the worker loaded the scene and protocol is alive.
        client.request({'op':'ping'}, timeout=30.0)
        time.sleep(max(0.0, args.hold_s))
        print('PORTABLE DEMO LOAD-ONLY PASS')
    finally:
        client.close()
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
