# Motion Route B — cuRobo

Route B is the production alternative to the retained legacy Route A. It uses
the same task endpoints and hand actions, but generates the right-arm path with
the true cuRobo V2 7DOF `MotionPlanner`.

## Fixed contract

- active joints exactly `arm_r_joint_1` through `arm_r_joint_7`;
- environment/ESDF collision ON for real free motion;
- self collision OFF in endpoint IK, TrajOpt, and graph rollouts;
- dense `q_rad` + `time_s` trajectory output;
- no 35DOF planning followed by slicing;
- no Route A quintic replacement for Route B arm trajectories;
- no hard-coded endpoint q7.

## Seven segments

1. `CURRENT_TO_PREGRASP`
2. `PREGRASP_TO_COVER`
3. `COVER_TO_LIFT`
4. `LIFT_TO_TRANSFER`
5. `TRANSFER_TO_PLACE`
6. `PLACE_TO_RETREAT`
7. `RETREAT_TO_HOME`

The Isaac executor may only piecewise-linearly sample adjacent dense cuRobo
samples at physics time. Hand-only actions retain the shared Route A semantics.

## Package layout

- `routeB_adapter.py`: shared MotionPlanner adapter and collision-policy wiring.
- `routeB_right_arm_only_core_v1/`: active-joint and trajectory contract.
- `routeB_front_half_v1/`: reach prior, endpoint goal pool, CURRENT→PREGRASP.
- `routeB_full_pipeline_v1/`: back-half pool and all seven dense trajectories.
- `trajectory_visualizer/`: canonical visualization package.
- `test_*audit.py` and package tests: long-term collision/planner regression
  diagnostics.

Run cuRobo modules only in `curobo_v2`. The normal user entry is still
`./run_closed_loop.sh --task ... --motion-route curobo`; standalone backends are
for regression and audit, not a second application architecture.
