# Route B full pipeline v1

This package completes Route B end-to-end while preserving Route A.

## Motion stages

```text
CAPTURE
  -> GroundingDINO + SAM
  -> DGN2
  -> LEAP Target Reach Region only
  -> LEAP->Wuji2 retarget
  -> exact COVER IK (collision gate OFF)
  -> relaxed PREGRASP endpoint IK (collision gate OFF)
  -> RobotSegmenter + ESDF
  -> current -> PREGRASP              true 7DOF MotionPlanner
  -> PREGRASP -> COVER                true 7DOF MotionPlanner, target removed from ESDF
  -> COVER_HAND                       hand only
  -> GRASP                            hand only
  -> SQUEEZE                          existing 41-point Wuji2 hand path
  -> COVER -> LIFT                    true 7DOF MotionPlanner, target removed from ESDF
  -> LIFT -> TRANSFER                 true 7DOF MotionPlanner + attached target proxy
  -> TRANSFER -> PLACE                true 7DOF MotionPlanner, target removed from ESDF
  -> RELEASE                          hand only
  -> PLACE -> RETREAT                 true 7DOF MotionPlanner + placed target proxy
  -> RETREAT -> HOME                  true 7DOF MotionPlanner + placed target proxy
```

## Attachment design

The target proxy comes from `capture/object_physics_audit.json`, which is produced
by Persistent Isaac from the actual target collision geometry. Its world AABB is
transformed to `arm_base_link`, padded, and represented as a cuboid.

For carry planning, the copied cuRobo robot config gets a fixed extra link
`routeB_attached_object` under `arm_r_link_tf`, plus collision sphere slots. This
uses cuRobo's official `extra_links` + `extra_collision_spheres` contract. The
official `AttachmentManager.fit_spheres()` + `update()` path fills those slots at
the measured COVER object pose. No URDF/USD file is modified.

## Scene policy

- `current -> PREGRASP`: already validated RobotSegmenter-cleaned ESDF with target present.
- `PREGRASP -> COVER` and carry: use RobotSegmenter-cleaned depth with the GroundedSAM target mask removed. Other objects and table remain obstacles.
- COVER->LIFT: target is removed from ESDF and not attached yet, so intentional target/table support contact does not make the start state infeasible.
- LIFT->TRANSFER: target is represented by attached spheres in true free space.
- TRANSFER->PLACE: target is again removed from ESDF/attachment for the intentional placement-contact descent; robot/hand still avoid all non-target scene geometry.
- Post-release: target is represented by a predicted placed cuboid in addition to the non-target ESDF.
- Environment collision ON; self collision OFF.

## Execution policy

Persistent Isaac receives a new additive `execute_routeB` protocol operation.
Route A's existing `execute()` and quintic arm segments are not changed.

Route B arm trajectories are not regenerated with quintic interpolation. At the
Isaac physics rate, the executor performs only piecewise-linear interpolation in
time between adjacent cuRobo dense samples. Hand-only actions keep the existing
quintic behavior; SQUEEZE keeps the existing 41-point hand path.

## Expected commands after integration

Full planning only:

```bash
cd ~/Projects/DexGraspNet2_Wuji2
./run_closed_loop.sh --motion-route curobo --planning-only
```

Full Isaac execution:

```bash
cd ~/Projects/DexGraspNet2_Wuji2
./run_closed_loop.sh --motion-route curobo --sim-execute
```

Legacy Route A remains unchanged.

## Route A semantics deliberately reused

Route B does **not** invent new task endpoints. It reuses Route A's existing:

- exact COVER target and strict endpoint definition;
- Wuji2 PREGRASP/COVER/GRASP hand waypoints;
- 41-point SQUEEZE path;
- LIFT sampling policy;
- TRANSFER sampling policy;
- PLACE/free green-zone center generation and lower placement point selection;
- RETREAT sampling and HOME target;
- endpoint refinement thresholds for COVER/GRASP;
- object-lift threshold and final green-zone verification.

The change is only the **arm motion between those task states**: Route A's quintic joint interpolation is replaced by true 7DOF cuRobo dense trajectories, with self collision OFF and environment collision ON on the Route B planning segments.
