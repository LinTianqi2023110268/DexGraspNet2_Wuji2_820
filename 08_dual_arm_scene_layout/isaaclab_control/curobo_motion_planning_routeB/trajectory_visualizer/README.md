# Route B trajectory visualizer v1

Interactive **planning-space** visualization for the validated Route B
`current -> PREGRASP` right-arm-only trajectory.

This viewer intentionally does **not** start Isaac Sim and does not execute the robot.

## What it shows

- RobotSegmenter-cleaned environment point cloud in `arm_base_link`.
- cuRobo collision-sphere representation of the robot.
- Moving right-arm spheres vs locked/static spheres.
- 41-point right-arm trajectory.
- Right-arm flange/tool trajectory.
- Current frame, time and q7 values.
- Optional per-frame minimum ESDF clearance and worst sphere.
- Start/goal ghost poses.

Controls:

- Slider: jump to any frame.
- Play/Pause button.
- Space: play/pause.
- Left/Right: single frame.
- Home/End: first/last frame.

## Why collision spheres, not URDF meshes?

For this stage the point is to verify the **same geometry the planner used**.
The collision-sphere model is therefore more useful than a pretty visual mesh.
Sphere marker sizes are schematic screen-space sizes; centers/radii in the
bundle are the true cuRobo collision model.

## Inputs

The generic viewer reads one file:

`visualization_right_arm_bundle.npz`

The local project bridge generates it from already-validated Route B objects.

Required arrays:

- `scene_points_base` `[P,3]`
- `sphere_centers_base` `[N,S,3]`
- `sphere_radii_m` `[S]`
- `sphere_link_names` `[S]`
- `sphere_active_mask` `[S]`
- `ee_positions_base` `[N,3]`
- `time_s` `[N]`
- `q_rad` `[N,7]`
- `joint_names` `[7]`

Optional:

- `frame_min_clearance_m` `[N]`
- `frame_worst_sphere_index` `[N]`
- `scene_colors_rgb` `[P,3]`

## Run

```bash
conda activate curobo_v2

python -m trajectory_visualizer.viewer \
  /path/to/visualization_right_arm_bundle.npz
```

No video export is part of v1. The first goal is simply to inspect the actual
41-point planner trajectory interactively.
