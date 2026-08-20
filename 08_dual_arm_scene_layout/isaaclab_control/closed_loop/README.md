# Closed-loop mainline

`orchestrator.py` is the production coordinator behind `./run_closed_loop.sh`.
It keeps one Isaac scene alive across capture, planning, execution, recovery,
and repeated task cycles.

## Task dispatch

- `semantic-grasp`: user text -> GroundingDINO/SAM target selection.
- `color-sort`: requested red/blue -> GroundingDINO/SAM color match -> current
  RGB HSV instance -> repeat until that requested color is exhausted or all
  current-scene instances have failed.

Color-sort does not read simulation `sort_color` to select a target. The seeded
runtime assignment is visual/audit data only.

## Motion dispatch

- `legacy`: established q7 waypoint and quintic Route A executor.
- `curobo`: Route B seven-segment true 7DOF MotionPlanner trajectories and
  `execute_routeB()`.

The two routes share task endpoints and Wuji2 hand actions; they do not share an
arm trajectory implementation.

## Per-cycle data contract

```text
capture/{rgb.png, depth_m.npy, intrinsics.npy, T_world_camera.npy, robot_state.json}
  -> capture/planning/robot_mask.npy
  -> capture/planning/rgb_no_robot.png
  -> capture/planning/filtered_depth.npy
```

The RobotSegmenter report must point to the current capture. A stale mask is a
hard error. DINO/SAM proposals must pass robot overlap, valid-depth, and rigid
SourceZone gates before becoming a target.

Planning starts from `filtered_depth`. Only the selected current target mask is
removed for intentional contact; other objects and same-color instances remain
obstacles.

## Recovery

Planning failure never sends an arm command. In color-sort it marks the current
instance failed while the arm remains HOME and tries another current-scene
instance. Physical execution errors may continue only after the recovery path
has returned and verified HOME; then the next attempt begins with a fresh
capture.

## Configuration

- `config/closed_loop.json`: environment paths, candidate funnel, endpoint and
  route settings.
- `config/color_sort.json`: palette, HSV, morphology, and instance settings.
- `../runtime/config/persistent_closed_loop.json`: Isaac execution timing and
  controller settings.

Do not use Route A collision-bypass flags as Route B options. Route B maintains
its independent environment/self-collision contract.
