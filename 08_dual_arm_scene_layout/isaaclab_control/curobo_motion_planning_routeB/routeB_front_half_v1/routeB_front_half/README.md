# Route B front-half v1

This package connects the already-validated Route B pieces into one **planning-only
front half** while leaving Route A unchanged.

## Final Route B front-half semantics

```text
Persistent Isaac capture
  -> GroundingDINO + SAM
  -> DGN2 LEAP candidates
  -> LEAP TARGET REACH REGION ONLY
       * one batched coarse GRASP IK
       * calibrated bridge inflation
       * NO PREGRASP coarse IK
       * NO ESDF collision
       * NO self collision
       * NO rough trajectory/corridor
  -> priority order (recommended: priority_then_rescue)
  -> LEAP -> Wuji2 retarget
  -> exact Wuji2 COVER IK (existing strict gate)
  -> relaxed Wuji2 PREGRASP endpoint IK (existing legal 6D region)
  -> build ordered q_PREGRASP/q_COVER goal pool
       * NO old HOME->PREGRASP interpolation gate
       * NO old PREGRASP->COVER rough path gate
  -> RobotSegmenter
  -> filtered_depth -> Mapper/ESDF
  -> true 7DOF locked-joint cuRobo MotionPlanner
       current -> PREGRASP
  -> trajectory_right_arm.npz
  -> STOP (no Isaac execution in v1 front-half)
```

## Why the LEAP reach filter remains

It is a cheap prior.  A region populated by many fast coarse-IK-reachable LEAP
roots is a good place to spend expensive retarget + exact Wuji2 IK effort.

It is **not** a safety gate.  Route B's actual safety comes later from the
post-retarget true 7DOF MotionPlanner against the RobotSegmenter-cleaned ESDF.

`priority_then_rescue` is therefore the recommended first integration mode.

## Files

- `leap_reach_backend.py`
  - replaces the full candidate-centric RFS V2 for Route B only.
  - implements only the Target Reach Region.
- `reach_contract.py`
  - SE(3) inflated-union membership and identity-safe ordering.
- `pregrasp_pool.py`
  - reuses existing exact COVER and relaxed PREGRASP IK machinery.
  - builds an ordered goal pool without old path checks.
- `routeB_dense_backend.py`
  - builds one locked 7DOF planner and tries ordered PREGRASP goals.
  - reuses the validated RouteB adapter collision policy and VoxelData shape fix.
- `runtime.py`
  - subprocess hooks safe to import from the IsaacLab orchestrator.

## Expected production outputs

Per cycle:

```text
routeB_front_half/
  leap_reach/
    production_candidates.json
    leap_target_reach_filter.json
    leap_target_reach_report.json
    leap_target_reach_map.npz

  routeB_front_half_goal_pool.npz
  routeB_front_half_goal_pool.json

  planning/
    routeB_front_half_report.json
    routeB_front_half_plan.npz
    trajectory_right_arm.npz
```

RobotSegmenter remains under:

```text
capture/planning/
  filtered_depth.npy
  robot_segmentation_report.json
```

## Important invariants

- Route A is untouched and remains the default.
- `--motion-route curobo` selects this front-half.
- exact COVER stays the existing strict post-retarget gate.
- Route B environment collision stays ON.
- Route B self collision stays OFF per the already-validated policy.
- no 35DOF free plan is sliced to pretend it was 7DOF.
- no extra quintic interpolation is added.
- v1 front-half does not execute Isaac.
