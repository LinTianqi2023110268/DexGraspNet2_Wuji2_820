# Pre-experiment baseline — 2026-08-20

- Scene migration: PASS. Production worker retains `SetTranslateOnly()` and
  lineage/object-identity USD resolution; SourceZone visual scale is not part
  of object mapping.
- Perception safety: PASS. The only robot-mask source is
  `RobotDepthCleaner.remove_robot()`.
- DINO image input: `capture/planning/rgb_no_robot.png`.
- Planning image input: `capture/planning/filtered_depth.npy`.
- Semantic target safety: DINO robot hard gate, SAM robot hard gate, valid
  depth and rigid SourceZone gate, plus fail-closed `STALE_ROBOT_MASK`.
- Color-sort: user enters `red`/`blue`; GroundingDINO/SAM receives
  `rgb_no_robot.png`, then its legal mask is intersected with a current-RGB HSV
  instance. HSV masks apply `& ~robot_mask` before morphology. Runtime color
  assignment is visual-only and never selects the object.
- Route B: true 7DOF; environment collision ON; self collision OFF.
- Latest GPU perception preflight: PASS.
  `outputs/pre_experiment_prefight/20260820_100016_semantic_curobo_perception_smoke/capture/`
- Route B full-plan baseline retained:
  `outputs/closed_loop_sessions/20260820_073606/cycle_001/`

Formal first experiment:

```bash
./run_closed_loop.sh --task semantic-grasp --motion-route curobo --sim-execute
```

Color experiment (explicit, non-TTY-safe):

```bash
./run_closed_loop.sh --task color-sort --target-color red --motion-route curobo --color-seed 42 --sim-execute
```
