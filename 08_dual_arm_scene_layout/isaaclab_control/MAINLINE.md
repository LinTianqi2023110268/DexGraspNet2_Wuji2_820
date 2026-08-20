# Formal experiment mainline

## Task and motion route

- `semantic-grasp`: text query -> GroundingDINO/SAM.
- `color-sort`: user chooses `red` or `blue` -> GroundingDINO/SAM matches that
  color in current RGB -> HSV instance intersection makes one physical target.
  Runtime red/blue assignment is visual-only; it is never used to select a
  target.
- `legacy`: Route A keypoint/flexible q7 with quintic arm execution.
- `curobo`: Route B true right-arm 7DOF dense cuRobo trajectory.

## Shared perception contract

Each capture preserves raw `rgb.png` and `depth_m.npy`, then runs the single
authoritative producer `RobotDepthCleaner.remove_robot()`:

`robot_mask.npy` -> `rgb_no_robot.png` for DINO/SAM, `~robot_mask` for HSV,
and `filtered_depth.npy` for planning/ESDF.

Semantic candidates follow:

`rgb_no_robot -> DINO proposal -> robot box gate -> SAM -> robot mask gate -> valid depth -> rigid SourceZone gate -> selected target`.

`robot_mask` always means robot pixels. `selected_target_mask` is separate and
is removed only from `filtered_depth` for intentional target-contact stages.

Color candidates follow:

`rgb_no_robot -> GroundingDINO("red object" | "blue object") -> SAM safety
gates -> intersect best matching current-RGB HSV instance -> one target mask`.
After each successful placement, capture and matching repeat until no requested
color remains in SourceZone.  An exhausted candidate funnel skips that instance
while the arm remains HOME and tries another current capture; execution errors
continue only after HOME recovery is confirmed.

## Route B contract

- active joints: `arm_r_joint_1` through `arm_r_joint_7` only;
- environment/ESDF collision: ON;
- self collision: OFF in IK, TrajOpt, and graph rollouts;
- no 35DOF slicing and no quintic replacement for Route B arm paths.

## Formal commands

```bash
./run_closed_loop.sh --task semantic-grasp --motion-route curobo --sim-execute
./run_closed_loop.sh --task semantic-grasp --motion-route legacy --sim-execute --no-planner-collision-check
./run_closed_loop.sh --task color-sort --target-color red --motion-route curobo --color-seed 42 --sim-execute
```

## Standardized three-scene color campaign

`closed_loop/tools/run_color_sort_campaign.py` invokes only the production
entry.  It creates one automatic timestamp root containing a manifest, summary,
and three case directories with their effective config, terminal log, result,
and session. It stops on a real process error for diagnosis and supports
`--resume` without rerunning completed cases. `campaign_report.md` is the
canonical compact human index; `campaign_summary.json` is the machine-readable
index. Use `--rerun-case <case_id>` only when a code repair invalidates one
previous case result.
