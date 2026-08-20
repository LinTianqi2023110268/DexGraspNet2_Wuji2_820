# Current architecture

This document describes the current closed-loop architecture.  The historical
research routes indexed in `verified/` use older A/B/C labels; those labels must
not be confused with the current CLI motion routes.

## Independent choice axes

```text
TASK                         MOTION ROUTE
semantic-grasp               legacy (Route A)
color-sort                   curobo (Route B)
```

The task decides how the current target is selected and where it is placed.
The motion route decides only how the right-arm q7 path between shared task
endpoints is generated and executed.

## Capture and perception

Every cycle starts with one coherent Persistent Isaac capture:

```text
rgb.png + depth_m.npy + intrinsics + T_world_camera + robot_state
  -> RobotDepthCleaner.remove_robot()
       -> robot_mask.npy
       -> rgb_no_robot.png
       -> filtered_depth.npy
```

`robot_mask.npy` is the only robot-pixel definition for the cycle. Semantic
recognition, color recognition, and planning depth all consume it. The report
capture path must match the current capture or the pipeline raises
`STALE_ROBOT_MASK`.

### Semantic target selection

```text
rgb_no_robot
  -> GroundingDINO proposals
  -> robot box overlap hard gate
  -> SAM
  -> robot mask overlap hard gate / residual
  -> valid raw depth
  -> world projection
  -> rigid SourceZone membership
  -> legal TargetSelection
```

GroundingDINO may still propose robot-shaped false positives; a proposal is not
a legal target.  The selected mask must survive every hard gate.

### Color target selection

```text
requested red/blue
  -> GroundingDINO + SAM on rgb_no_robot
  -> the same semantic safety gates
  -> RGB-to-HSV color masks
  -> mask & ~robot_mask before morphology
  -> connected SourceZone instances
  -> intersect legal DINO/SAM mask with one current instance
  -> legal TargetSelection
```

Simulation color assignment is audit ground truth only; target selection never
reads `sort_color` from object records.

## Shared grasp downstream

Both target sources feed the same downstream pipeline:

```text
TargetSelection
  -> current-cycle target mask + RobotSegmenter-filtered depth
  -> 40k DGN2 input
  -> DexGraspNet2 proposals
  -> LEAP reach ordering
  -> LEAP-to-Wuji2 retarget + exact COVER
  -> PREGRASP/LIFT/TRANSFER/PLACE/RETREAT endpoint funnel
  -> selected motion route
```

The target mask is removed only for intentional target contact. Other objects,
including other objects of the same color, remain environmental obstacles.

## Motion Route A — legacy

Route A retains the established endpoint/keypoint implementation and quintic
arm executor:

```text
shared task semantics
  -> flexible/keypoint q7 waypoints
  -> legacy quintic arm interpolation
  -> PersistentIsaacClient.execute()
  -> PersistentScene.execute()/execute_segment()
```

## Motion Route B — cuRobo

Route B changes only the arm path implementation:

```text
shared task endpoints
  -> true 7DOF cuRobo MotionPlanner
  -> dense time-parameterized q7 trajectory
  -> PersistentIsaacClient.execute_routeB()
  -> PersistentScene.execute_routeB()
```

The seven required arm segments are CURRENT→PREGRASP, PREGRASP→COVER,
COVER→LIFT, LIFT→TRANSFER, TRANSFER→PLACE, PLACE→RETREAT, and RETREAT→HOME.
Only adjacent dense samples may be linearly resampled at the Isaac physics
step; Route B arm motion is not replanned or replaced by a quintic path.

Route B contract:

- joints exactly `arm_r_joint_1` through `arm_r_joint_7`;
- environment/ESDF collision ON;
- self collision OFF in endpoint IK, TrajOpt, and graph rollouts;
- no 35DOF solve-and-slice and no hard-coded PLACE q7.

## Scene migration

Training scene coordinates are mapped into the calibrated SourceZone with a
rigid transform only:

```text
T_world_object = T_world_source_zone_rigid @ T_scene_centered_object
```

SourceZone cube display scale never enters this matrix.  The persistent worker
resolves the simulation USD from the scene dataset lineage and object identity,
uses `SetTranslateOnly()` so orientation is preserved, and writes pre-physics
pose/pairwise assertions before settling. SourceZone is a visual/coordinate
region; the table collision is the physical support.

## Process boundary

- orchestrator and Persistent Isaac worker: `isaaclab22_sim50`;
- GroundingDINO/SAM: `groundedsam`;
- DGN2 network: `graspnet2.0`;
- cuRobo IK/RobotSegmenter/MotionPlanner: `curobo_v2`;
- hand retarget: project-local `wuji_retargeting`.

There is one Persistent Isaac worker per session. GPU planning workers are
launched in their own environment and released when their stage finishes.
