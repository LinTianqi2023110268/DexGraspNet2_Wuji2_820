# Project status

Updated: 2026-08-20.

## Production entry

`./run_closed_loop.sh` is the only user-facing closed-loop entry.  It dispatches
two independent task choices (`semantic-grasp`, `color-sort`) and two motion
implementations (`legacy`, `curobo`).

The current experiment focus is semantic-grasp + Route B, followed by
color-sort + Route B. Route A remains available and must not be replaced by the
Route B planner.

## Verified contracts

- Scene migration regression guard: PASS.  SourceZone visual scale is excluded
  from object migration; asset fallback uses dataset lineage and object
  identity; pre-physics pose and pairwise geometry audits are retained.
- Perception target safety: PASS.  RobotDepthCleaner is the single robot-mask
  source for RGB, HSV, and planning depth. DINO and SAM robot gates are hard
  gates; SourceZone and stale-capture checks fail closed.
- Route B static contract: true right-arm 7DOF, environment collision ON, self
  collision OFF in IK/TrajOpt/graph.
- Task/route CLI dispatch and Persistent Isaac lifecycle preflight: PASS.

## Evidence boundaries

- Formal current perception smoke:
  `08_dual_arm_scene_layout/isaaclab_control/outputs/pre_experiment_prefight/20260820_100016_semantic_curobo_perception_smoke/`.
- Corrected scene migration audits:
  `08_dual_arm_scene_layout/isaaclab_control/outputs/scene_migration_audits/`.
- Last retained seven-segment Route B planner PASS:
  `.../outputs/closed_loop_sessions/20260820_073606/cycle_001/`.
  It is retained as planner regression evidence, not as a capture or trajectory
  to reuse in a new scene.
- Historical physical baselines remain indexed under `verified/`; their older
  A/B/C research-route names are provenance labels, not the current CLI motion
  route menu.

## Not yet claimed

- A fresh, post-migration-fix single-target Route B full physical execution has
  not been promoted to a new frozen baseline in this cleanup.
- Multi-round color-sort physical completion has not been promoted to a frozen
  baseline.
- Diagnostic collision bypass results are not real-robot safety evidence.

Current commands and behavior are documented in `README.md` and
`08_dual_arm_scene_layout/isaaclab_control/MAINLINE.md`.
