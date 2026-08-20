# Authoritative robot segmentation

`RobotDepthCleaner.remove_robot()` is the production robot-mask source for each
closed-loop capture. It is not a planning-only adapter.

## Input contract

The capture directory contains coherent artifacts from one cycle:

- `rgb.png` and `depth_m.npy`;
- `intrinsics.npy` and `T_world_camera.npy`;
- `robot_state.json`.

The adapter projects the configured cuRobo robot geometry with the current
robot state and produces one authoritative mask.

## Outputs

Under `<capture>/planning/`:

- `robot_mask.npy`, `robot_mask.png`, `robot_mask_overlay.png`;
- `rgb_no_robot.png` (raw RGB remains unchanged);
- `filtered_depth.npy`, `filtered_depth_preview.png`;
- `robot_segmentation_report.json`.

The same `robot_mask.npy` is consumed by:

1. semantic GroundingDINO/SAM hard gates;
2. color HSV exclusion before morphology;
3. depth/point-cloud/ESDF planning.

The report's `capture_dir` must equal the current capture path. Mismatches raise
`STALE_ROBOT_MASK`; a previous-cycle mask is never reused.

## Standalone capture check

Run only in `curobo_v2`:

```bash
conda run -n curobo_v2 python \
  08_dual_arm_scene_layout/isaaclab_control/perception/robot_segmentation/run_robot_segmenter_capture.py \
  --capture-dir /absolute/path/to/current/capture
```

Coordinate contract:

- `T_world_camera` maps camera coordinates into calibrated layout world;
- layout configuration provides `T_world_base`;
- `T_base_camera = inv(T_world_base) @ T_world_camera` is passed to cuRobo.

`selected_target_mask` is not a robot mask. It is handled downstream and may be
removed only for the current intentional-contact planning stage.
