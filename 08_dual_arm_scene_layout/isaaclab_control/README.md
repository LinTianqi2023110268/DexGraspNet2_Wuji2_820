# Isaac Lab closed-loop control

This directory owns the current dual-arm closed loop. The user entry remains at
the repository root:

```bash
./run_closed_loop.sh --sim-execute
```

Do not launch a second Isaac/Kit process while a persistent worker is active.

## Mainline ownership

- `closed_loop/`: task/route dispatch, perception, planning orchestration,
  persistent Isaac protocol, and recovery.
- `curobo_motion_planning_routeB/`: true right-arm 7DOF Route B planning and
  dense-trajectory contracts.
- `perception/robot_segmentation/`: the authoritative RobotDepthCleaner adapter.
- `runtime/`: retained Route A executor/configuration used by the closed loop.
- `core/`: shared math, cuRobo worker protocol, IK, RGB-D/ESDF, tests, and worklog.
- `diagnostics/`: reproducible control/physics audits, not user entry points.
- `tools/`: scene/capture/target preparation utilities.
- `evidence/`: compact older frozen physical evidence.
- `outputs/`: generated sessions and audits; never a source-code dependency.

## Current behavior

The task and motion route are independent:

- `semantic-grasp` or `color-sort`;
- `legacy` (Route A) or `curobo` (Route B).

Both tasks share fresh capture, RobotSegmenter, target selection, DGN2,
LEAP→Wuji2, endpoint semantics, hand actions, placement helpers, and execution
verification. Route B replaces only the right-arm path generator/executor with
the cuRobo dense trajectory path.

## Critical contracts

- Single Persistent Isaac worker per session.
- Scene migration is rigid; SourceZone display scale is not a pose transform.
- `robot_mask` is the only robot mask for RGB, HSV, and planning depth.
- `selected_target_mask` is separate and removed only for target-contact stages.
- Route B has exactly seven active arm joints, environment collision ON, and
  self collision OFF in IK/TrajOpt/graph.
- Route A `execute_segment()` numerical behavior is retained.
- No official Wuji2 URDF/USD edits and no acceptance-threshold relaxation.

See `MAINLINE.md` for the compact runnable architecture and
`core/worklog/PRE_EXPERIMENT_BASELINE.md` for the latest preflight evidence.
