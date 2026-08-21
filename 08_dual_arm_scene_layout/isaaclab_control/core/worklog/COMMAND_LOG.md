# Route-C V2 command log

## 2026-08-20 15:xx +08:00 — PLACE V3 near-first endpoint preflight integration

- Purpose: apply `/home/lin/Projects/_wuji2_refactor_packages/wuji2_color_place_near_first_v3_20260820/` PLACE V3 semantics and run endpoint-only preflight without Isaac execution or dense MotionPlanner.
- Conda environment: `curobo_v2` for endpoint IK preflight.
- Working directory: `/home/lin/Projects/DexGraspNet2_Wuji2`
- Commands:

  ```bash
  git status --short
  git diff --check
  sed -n '1,760p' /home/lin/Projects/_wuji2_refactor_packages/wuji2_color_place_near_first_v3_20260820/{README.md,PLACE_V3_ARCHITECTURE.md,SIMULATION_VERIFICATION_POLICY.md,CODEX_TASK.md}
  python -m json.tool /home/lin/Projects/_wuji2_refactor_packages/wuji2_color_place_near_first_v3_20260820/MANIFEST.json
  python -m py_compile 08_dual_arm_scene_layout/isaaclab_control/closed_loop/planning/ordered_place_policy.py 08_dual_arm_scene_layout/isaaclab_control/closed_loop/planning/simplified_route_search.py 08_dual_arm_scene_layout/isaaclab_control/core/bridge/curobo_worker.py 08_dual_arm_scene_layout/isaaclab_control/curobo_motion_planning_routeB/routeB_full_pipeline_v1/routeB_full_pipeline/backhalf_pool.py 08_dual_arm_scene_layout/isaaclab_control/closed_loop/tools/place_v3_endpoint_preflight.py
  /home/lin/miniconda3/bin/conda run --no-capture-output -n curobo_v2 python 08_dual_arm_scene_layout/isaaclab_control/closed_loop/tools/place_v3_endpoint_preflight.py --project-root /home/lin/Projects/DexGraspNet2_Wuji2 --cycle-root /home/lin/Projects/DexGraspNet2_Wuji2/08_dual_arm_scene_layout/isaaclab_control/outputs/closed_loop_sessions/20260820_142510/cycle_001 --goal-pool /home/lin/Projects/DexGraspNet2_Wuji2/08_dual_arm_scene_layout/isaaclab_control/outputs/closed_loop_sessions/20260820_142510/cycle_001/routeB_front_half/routeB_front_half_goal_pool.npz --placement-registry /home/lin/Projects/DexGraspNet2_Wuji2/08_dual_arm_scene_layout/isaaclab_control/outputs/closed_loop_sessions/20260820_142510/placement_registry.json --color-zones /home/lin/Projects/DexGraspNet2_Wuji2/08_dual_arm_scene_layout/isaaclab_control/outputs/closed_loop_sessions/20260820_142510/color_sort_zones.json --zone-id red_zone --scan-limit 8 --output /home/lin/Projects/DexGraspNet2_Wuji2/08_dual_arm_scene_layout/isaaclab_control/outputs/closed_loop_sessions/20260820_142510/cycle_001/routeB_full/place_v3_endpoint_preflight_red.json
  /home/lin/miniconda3/bin/conda run --no-capture-output -n curobo_v2 python 08_dual_arm_scene_layout/isaaclab_control/closed_loop/tools/place_v3_endpoint_preflight.py --project-root /home/lin/Projects/DexGraspNet2_Wuji2 --cycle-root /home/lin/Projects/DexGraspNet2_Wuji2/08_dual_arm_scene_layout/isaaclab_control/outputs/closed_loop_sessions/20260820_140119/cycle_001 --goal-pool /home/lin/Projects/DexGraspNet2_Wuji2/08_dual_arm_scene_layout/isaaclab_control/outputs/closed_loop_sessions/20260820_140119/cycle_001/routeB_front_half/routeB_front_half_goal_pool.npz --placement-registry /home/lin/Projects/DexGraspNet2_Wuji2/08_dual_arm_scene_layout/isaaclab_control/outputs/closed_loop_sessions/20260820_140119/placement_registry.json --color-zones /home/lin/Projects/DexGraspNet2_Wuji2/08_dual_arm_scene_layout/isaaclab_control/outputs/closed_loop_sessions/20260820_140119/color_sort_zones.json --zone-id blue_zone --scan-limit 32 --output /home/lin/Projects/DexGraspNet2_Wuji2/08_dual_arm_scene_layout/isaaclab_control/outputs/closed_loop_sessions/20260820_140119/cycle_001/routeB_full/place_v3_endpoint_preflight_blue_scan32.json
  ```

- Key output: red-zone V3 endpoint preflight PASS on `20260820_142510/cycle_001`, first pass `rank=0 cand=1536`, slot count `4`, release poses `108`, PLACE raw `25`, relaxed accepted targets `18`, chains `32`; accepted slots were the near column lanes `column_00_lane_00` and `column_00_lane_01`.

## 2026-08-20 16:xx +08:00 — color-sort target removal mask split

- Purpose: fix the color-sort perception→ESDF interface so the HSV selected
  instance no longer serves as both color-selection mask and full target-removal
  mask.
- Read package:
  `/tmp/wuji2_target_removal_mask_v1_20260820/TARGET_REMOVAL_POLICY.md`,
  `CODEX_TASK.md`, and
  `/tmp/wuji2_target_removal_mask_v1_20260820/closed_loop/perception/target_removal_mask.py`.
- Focused inspection commands:

  ```bash
  grep -n "target_mask_path\\|build_perception_target_geometry\\|filtered_depth_no_target\\|selected_target_mask\\|target_removal\\|target_grasp_mask" 08_dual_arm_scene_layout/isaaclab_control/closed_loop/orchestrator.py
  find 08_dual_arm_scene_layout/isaaclab_control/outputs/closed_loop_sessions/20260820_142510/cycle_001 -maxdepth 6 -type f \\( -name '*mask*.npy' -o -name '*mask*.png' -o -name '*target*.json' -o -name '*result*.json' -o -name '*audit*.json' \\)
  grep -RIn "target_mask_path\\|filtered_depth_no_target\\|target_removal\\|selected_target_mask\\|no_target" 08_dual_arm_scene_layout/isaaclab_control/curobo_motion_planning_routeB/routeB_full_pipeline_v1 08_dual_arm_scene_layout/isaaclab_control/closed_loop/planning 08_dual_arm_scene_layout/isaaclab_control/core
  ```

- Validation commands:

  ```bash
  /home/lin/miniconda3/envs/curobo_v2/bin/python -m py_compile 08_dual_arm_scene_layout/isaaclab_control/closed_loop/perception/target_removal_mask.py 08_dual_arm_scene_layout/isaaclab_control/closed_loop/color_sort/target_pool.py 08_dual_arm_scene_layout/isaaclab_control/closed_loop/orchestrator.py 08_dual_arm_scene_layout/isaaclab_control/closed_loop/tools/replay_target_removal_mask.py 08_dual_arm_scene_layout/isaaclab_control/curobo_motion_planning_routeB/routeB_full_pipeline_v1/routeB_full_pipeline/runtime.py 08_dual_arm_scene_layout/isaaclab_control/curobo_motion_planning_routeB/routeB_full_pipeline_v1/routeB_full_pipeline/full_motion_backend.py
  /home/lin/miniconda3/envs/curobo_v2/bin/python 08_dual_arm_scene_layout/isaaclab_control/closed_loop/tools/replay_target_removal_mask.py --cycle-root 08_dual_arm_scene_layout/isaaclab_control/outputs/closed_loop_sessions/20260820_142510/cycle_001 --target-id red_target_000_red_003
  git diff --check
  ```

- Replay result: SAM `17039` px, HSV `10425` px, old HSV-removal leftover
  `6779` px (`39.785%`); new removal mask `12835` px, SAM leftover `4804` px
  (`28.194%`).
- ESDF audit wrote
  `outputs/closed_loop_sessions/20260820_142510/cycle_001/capture/planning/target_removal_esdf_input_audit.json`.
- Isaac, dense MotionPlanner, and full grasp execution were not started.

## 2026-08-20 20:xx +08:00 — target-removal depth gate correction

- User visual check on `20260820_195131/cycle_001` showed yellow
  `target_grasp_mask` covered the full object while purple
  `target_removal_mask` covered only a small part.
- Generated breakdown under
  `outputs/closed_loop_sessions/20260820_195131/cycle_001/capture/planning/target_removal_debug_breakdown/`.
  Root cause: HSV neighbourhood lost only `303` SAM pixels (`1.93%`), but the
  fixed `median_depth +/- 0.03m` gate lost `9487` SAM pixels (`60.51%`).
- Changed `target_removal_mask` depth consistency from fixed median +/- 3cm to
  adaptive `[P1, P99] +/- 1cm` computed from `SAM & HSV & valid_depth`, while
  keeping the final mask constrained to matched SAM and selected HSV
  neighbourhood.  Default SAM expansion is now `0`, so target removal does not
  delete pixels outside the matched SAM mask.
- Replayed `20260820_195131/cycle_001/red_target_000_red_000`: new removal
  `15369` px, SAM leftover `309` px (`1.971%`).  Breakdown v2 shows depth gate
  now loses only `6` SAM pixels; the remaining `309` are HSV-neighbourhood edge
  misses.
- Replayed `20260820_142510/cycle_001/red_target_000_red_003`: new removal
  `16543` px, SAM leftover `496` px (`2.911%`), improved from old HSV leftover
  `6779` px (`39.785%`).
- Future production/replay writes binary PNG previews for
  `target_grasp_mask.png` and `target_removal_mask.png`; current session
  overlays were regenerated for visual inspection.

## 2026-08-21 01:xx +08:00 — target-removal supplement expansion v2

- Read `/tmp/wuji2_target_removal_expand_v2_20260820/CODEX_TASK.md` and
  `perception/target_removal_expand_v2.py`.
- Scope: only `target_removal_mask`; no DGN2, Route A, Route B, cuRobo,
  PLACE V3, IK, Isaac, or MotionPlanner execution.
- Preserved the existing core formula:
  `core = SAM ∩ HSV_neighbourhood ∩ adaptive_depth_gate`.
- Added v2 supplement:
  `supplement = SAM ∩ dilate(core, radius=8px)`, final
  `target_removal_mask = core | supplement`.
- Offline replay command:
  `/home/lin/miniconda3/envs/curobo_v2/bin/python 08_dual_arm_scene_layout/isaaclab_control/closed_loop/tools/replay_target_removal_mask.py --cycle-root 08_dual_arm_scene_layout/isaaclab_control/outputs/closed_loop_sessions/20260820_195131/cycle_001 --target-id red_target_000_red_000`.
- Result for `20260820_195131/cycle_001/red_target_000_red_000`: old core
  removal `15369` px, old leftover `309` px (`1.971%`); supplement `305` px;
  new removal `15674` px, new leftover `4` px (`0.026%`).
- ESDF audit replay wrote
  `capture/planning/filtered_depth_no_target_replay_target_removal_v2.npy`
  with `target_pixels_removed=15674`.
- Regenerated visual checks:
  `target_removal_mask_overlay.png` and
  `target_removal_leftover_overlay.png`.
- Follow-up visual inspection found a few remaining SAM edge pixels.  Increased
  supplement dilation from `8px` to `12px` while keeping
  `supplement = SAM ∩ dilate(core)` bounded by the matched SAM mask.
- Replay result after 12px expansion on
  `20260820_195131/cycle_001/red_target_000_red_000`: old core removal
  `15369` px, supplement `309` px, new removal `15678` px, new leftover `0` px.

## 2026-08-21 — DGN2 target-cate dual-line wiring

- Read `/tmp/wuji2_dgn2_target_cate_dual_v1_20260821/CODEX_TASK.md`,
  `INTEGRATION_SPEC.md`, and `closed_loop_config_snippet.json`.
- Copied overlay source files:
  `08_dual_arm_scene_layout/scripts/09_predict_official_leap_target_cate.py`,
  `08_dual_arm_scene_layout/isaaclab_control/closed_loop/dgn2_sampling_policy.py`,
  and
  `08_dual_arm_scene_layout/isaaclab_control/closed_loop/tools/audit_dgn2_target_membership.py`.
- Preserved legacy script:
  `git diff -- 08_dual_arm_scene_layout/scripts/09_predict_official_leap_target.py`
  produced no output.
- Changed only orchestrator DGN2 inference command to use
  `build_sampling_plan(...)` / `write_sampling_plan(...)`.
- Added top-level `dgn2_sampling` config:
  `semantic-grasp=scene_postfilter`, `color-sort=target_cate`, with
  `WUJI2_DGN2_SAMPLING_MODE` override enabled.
- Offline validation:
  copied
  `outputs/closed_loop_sessions/20260821_114952/cycle_001/capture/dgn2/red_object`
  to `/tmp/wuji2_dgn2_target_cate_replay_20260821_114952_red_object`.
- Membership audit PASS: full scene `40000`, target points `2948`,
  background `37052`, nonzero ids `[1]`.
- New target-cate sampler command in `graspnet2.0`:
  `/home/lin/miniconda3/envs/graspnet2.0/bin/python 08_dual_arm_scene_layout/scripts/09_predict_official_leap_target_cate.py --target red_object --rounds 8 --input-root /tmp/wuji2_dgn2_target_cate_replay_20260821_114952_red_object`.
- Result: proposal_count `8192`, target_proposal_count `8192`,
  non_target_seed_count `0`, selected_score `-10.863729476928711`.
- Validation: py_compile PASS; `git diff --check` PASS.  Isaac, MotionPlanner,
  route planning, and git push were not run.
- Key output: blue-zone V3 endpoint preflight still FAIL on `20260820_140119/cycle_001` after scanning all 17 unique front-half cases; each PLACE had raw `0`. Residual audit for representative `rank=16 cand=3525`: best position error `0.22438162565231323 m`, P50 `0.3483843207359314 m`; best orientation error `9.0158 deg`; best inner joint margin `-8.6631 deg`; failure type `multiple_fail=108/108`.
- Conclusion: V3 wiring fixes old red PLACE endpoint generation and proves at least one PLACE V3 endpoint PASS. The old blue session is no longer failing because of precise near-table pose semantics alone; it is a reachability/joint-margin issue for that target/color-zone configuration and must not be masked by threshold loosening.

## 2026-08-16 15:15 +08:00 — Initial read-only Git baseline

- Purpose: verify the required working directory and preserve the user's existing work.
- Conda environment: shell/base context; no project environment activated.
- Working directory: `/home/lin/Projects/DexGraspNet2_Wuji2`
- Command:

  ```bash
  pwd
  git status --short
  git diff --stat
  git diff
  ```

- Exit code: `0`
- Key output: correct project root; `core/`, three diagnostic scripts, `outputs/ik_failure_diagnosis/`, `MANIFEST.txt`, and `SELF_TEST_REPORT.txt` are untracked; tracked diff is empty.
- Conclusion: preserve all existing untracked paths and avoid destructive cleanup.

## 2026-08-16 15:16 +08:00 — Read integration and skill instructions

- Purpose: read all user-designated fact sources plus the two applicable local skill files.
- Conda environment: shell/base context.
- Working directory: `/home/lin/Projects/DexGraspNet2_Wuji2`
- Command:

  ```bash
  cat .agents/skills/isaaclab22-manipulator-control/SKILL.md
  cat .agents/skills/isaacsim5-manipulator-control/SKILL.md
  cat 08_dual_arm_scene_layout/isaaclab_control/core/CODEX_INSTRUCTION.md
  cat 08_dual_arm_scene_layout/isaaclab_control/core/INTEGRATION_GUIDE.md
  cat 08_dual_arm_scene_layout/isaaclab_control/core/README.md
  ```

- Exit code: `0`
- Key output: confirmed two-conda worker architecture, fixed IK thresholds, observed-scene Mapper/TSDF/ESDF contract, continuity selection, and static-stability gate.
- Conclusion: integrate supplied code with minimal compatibility changes; do not redesign or alter vendor assets.

## 2026-08-16 15:17 +08:00 — Skill reference and file inventory check

- Purpose: load skill-linked references and inventory relevant source files.
- Conda environment: shell/base context.
- Working directory: `/home/lin/Projects/DexGraspNet2_Wuji2`
- Command:

  ```bash
  cat .agents/skills/isaaclab22-manipulator-control/reference.md
  cat .agents/skills/isaaclab22-manipulator-control/evaluations.md
  cat .agents/skills/isaacsim5-manipulator-control/reference.md
  cat .agents/skills/isaacsim5-manipulator-control/evaluations.md
  rg --files 08_dual_arm_scene_layout/isaaclab_control/core
  ```

- Exit code: `127`
- Key output: all four referenced files are absent; `rg` is not installed.
- Conclusion: use the fully read skill bodies and fall back to `find`/`grep` for source audit.

## 2026-08-16 15:17 +08:00 — Fallback source inventory

- Purpose: inventory core, runtime, tools, and diagnostics using available utilities.
- Conda environment: shell/base context.
- Working directory: `/home/lin/Projects/DexGraspNet2_Wuji2`
- Command:

  ```bash
  find 08_dual_arm_scene_layout/isaaclab_control/core -type f -print | sort
  find 08_dual_arm_scene_layout/isaaclab_control/runtime 08_dual_arm_scene_layout/isaaclab_control/tools 08_dual_arm_scene_layout/isaaclab_control/diagnostics -type f -print | sort
  ```

- Exit code: `0`
- Key output: supplied core includes IK, worker bridge, RGB-D mapper, ESDF collision, robot spheres, tests, and a generated robot YAML; legacy runtime/tool paths remain present.
- Conclusion: proceed to content-level writer/consumer and production import audit.

## 2026-08-16 15:20 +08:00 — A. cuRobo environment probe

- Purpose: verify the frozen `curobo_v2` environment and required current APIs.
- Conda environment: `curobo_v2`.
- Working directory: `/home/lin/Projects/DexGraspNet2_Wuji2`
- Command:

  ```bash
  conda run -n curobo_v2 python 08_dual_arm_scene_layout/isaaclab_control/core/tools/probe_environment.py --project-root /home/lin/Projects/DexGraspNet2_Wuji2
  ```

- Exit code: `0`
- Key output: Python `3.11.15`; PyTorch `2.13.0+cu130`; cuRobo `0.8.0.post1.dev42`; Mapper, InverseKinematics, Kinematics, and RobotBuilder imports all OK; `cuda_available=False`.
- Conclusion: API imports pass, but this process currently has no usable CUDA device. GPU-dependent acceptance cannot be claimed unless a later run sees CUDA.

## 2026-08-16 15:21 +08:00 — B. Core pure tests, initial attempt

- Purpose: run supplied NumPy-only core tests with the required package path.
- Conda environment: base interpreter context.
- Working directory: `/home/lin/Projects/DexGraspNet2_Wuji2`
- Command:

  ```bash
  PYTHONPATH="$PWD/08_dual_arm_scene_layout/isaaclab_control" python -m unittest discover -s 08_dual_arm_scene_layout/isaaclab_control/core/tests -v
  ```

- Exit code: `1`
- Key output: all five test modules fail during collection with `ModuleNotFoundError: No module named 'numpy'`; interpreter is base Python 3.14.
- Conclusion: environment-only failure. Do not install into base; rerun unchanged tests with the existing `curobo_v2` interpreter.

## 2026-08-16 15:22 +08:00 — B. Core pure tests, correct environment

- Purpose: validate branch selection, RGB-D geometry, ESDF query, unknown-space semantics, and worker serialization.
- Conda environment: `curobo_v2`.
- Working directory: `/home/lin/Projects/DexGraspNet2_Wuji2`
- Command:

  ```bash
  PYTHONPATH="$PWD/08_dual_arm_scene_layout/isaaclab_control" conda run -n curobo_v2 python -m unittest discover -s 08_dual_arm_scene_layout/isaaclab_control/core/tests -v
  ```

- Exit code: `0`
- Key output: 11 tests run; all passed.
- Conclusion: supplied pure logic passes unchanged in the correct existing environment.

## 2026-08-16 15:24 +08:00 — C. Robot collision model build attempt

- Purpose: generate or validate the cuRobo collision-sphere robot model without touching vendor assets.
- Conda environment: `curobo_v2`.
- Working directory: `/home/lin/Projects/DexGraspNet2_Wuji2`
- Command:

  ```bash
  conda run -n curobo_v2 python 08_dual_arm_scene_layout/isaaclab_control/core/tools/build_robot_collision_model.py --project-root /home/lin/Projects/DexGraspNet2_Wuji2 --compute-metrics
  ```

- Exit code: `1`
- Key output: builder refused to overwrite the existing generated YAML and requested `--force` only after review.
- Conclusion: safe expected refusal. Existing YAML and alias adapter are present; inspect and load-test them before any rebuild.

## 2026-08-16 15:25 +08:00 — C. Existing collision model review

- Purpose: verify the generated model's provenance and fixed robot contract.
- Conda environment: shell/base context; read-only inspection.
- Working directory: `/home/lin/Projects/DexGraspNet2_Wuji2`
- Command:

  ```bash
  sha256sum 08_dual_arm_scene_layout/isaaclab_control/core/generated/dual_arm_right_wuji2_curobo.yml
  grep -nE 'urdf|base_link|arm_r_joint|collision_spheres|asset_alias' 08_dual_arm_scene_layout/isaaclab_control/core/generated/dual_arm_right_wuji2_curobo.yml
  ls -la 08_dual_arm_scene_layout/isaaclab_control/core/generated/asset_aliases
  ```

- Exit code: `0`
- Key output: SHA-256 `f743b132fe15c8e66b6f31c6e7aad5b161d041e31c51dd12b77fe1275c313f3c`; base `arm_base_link`; right-arm order J1..J7; tool frames include `arm_r_link_tf`; YAML points to the official combined URDF; both package aliases are symlinks into the vendor description tree.
- Conclusion: contract and non-invasive asset adapter are present; GPU load validation remains pending.

## 2026-08-16 15:28 +08:00 — D. Persistent worker smoke, restricted execution channel

- Purpose: ping the persistent `curobo_v2` worker from `isaaclab22_sim50`.
- Conda environment: client `isaaclab22_sim50`; worker `curobo_v2`.
- Working directory: `/home/lin/Projects/DexGraspNet2_Wuji2`
- Command:

  ```bash
  conda run -n isaaclab22_sim50 python 08_dual_arm_scene_layout/isaaclab_control/core/tools/smoke_test_worker.py --project-root /home/lin/Projects/DexGraspNet2_Wuji2
  ```

- Exit code: `1`
- Key output: client timed out after 120 s; delayed worker stderr reports `IKSolverUnavailable: CUDA is not visible to PyTorch in curobo_v2`.
- Conclusion: failure is tied to the restricted execution channel's GPU visibility, not proof that the machine lacks a GPU. Retest outside the sandbox.

## 2026-08-16 15:31 +08:00 — GPU visibility diagnosis

- Purpose: distinguish repository/environment errors from execution-channel device isolation.
- Conda environment: `curobo_v2` plus host driver utility.
- Working directory: `/home/lin/Projects/DexGraspNet2_Wuji2`
- Commands:

  ```bash
  nvidia-smi -L
  conda run -n curobo_v2 python -c "import torch; ..."
  ```

- Exit codes: restricted channel `1`; approved host channel `0`.
- Key output: restricted channel cannot communicate with the driver and reports zero devices; host channel reports `NVIDIA GeForce RTX 4070 Laptop GPU`, PyTorch `2.13.0+cu130`, CUDA `13.0`, `cuda_available=True`, device count `1`.
- Conclusion: the GPU is healthy and available on the host. All GPU-dependent acceptance commands must run through the host execution channel.

## 2026-08-16 15:33 +08:00 — A. Formal host-channel cuRobo probe

- Purpose: record the authoritative environment probe with real GPU access.
- Conda environment: `curobo_v2`.
- Working directory: `/home/lin/Projects/DexGraspNet2_Wuji2`
- Command:

  ```bash
  conda run -n curobo_v2 python 08_dual_arm_scene_layout/isaaclab_control/core/tools/probe_environment.py --project-root /home/lin/Projects/DexGraspNet2_Wuji2
  ```

- Exit code: `0`
- Key output: Python `3.11.15`; PyTorch `2.13.0+cu130`; CUDA true; cuRobo `0.8.0.post1.dev42`; GPU `NVIDIA GeForce RTX 4070 Laptop GPU`; Mapper, IK, Kinematics, and RobotBuilder imports OK.
- Conclusion: A. environment probe PASS on the authoritative host execution channel.

## 2026-08-16 15:34 +08:00 — D. Persistent worker host-channel smoke

- Purpose: validate isolated cross-conda IPC from Isaac Lab to persistent cuRobo.
- Conda environment: client `isaaclab22_sim50`; worker `curobo_v2`.
- Working directory: `/home/lin/Projects/DexGraspNet2_Wuji2`
- Command:

  ```bash
  conda run -n isaaclab22_sim50 python 08_dual_arm_scene_layout/isaaclab_control/core/tools/smoke_test_worker.py --project-root /home/lin/Projects/DexGraspNet2_Wuji2
  ```

- Exit code: `0`
- Key output: pong from cuRobo `0.8.0.post1.dev42` with right-arm joints `arm_r_joint_1` through `arm_r_joint_7` in exact order.
- Conclusion: D. persistent worker smoke PASS; environment isolation and IPC architecture are valid.

## 2026-08-16 15:39 +08:00 — E. candidate3800 exact GPU IK, initial client attempt

- Purpose: solve the five known exact flange targets through the persistent worker.
- Conda environment: bare `isaaclab22_sim50` client before AppLauncher; worker not reached.
- Working directory: `/home/lin/Projects/DexGraspNet2_Wuji2`
- Command: `conda run -n isaaclab22_sim50 python -c '<load case NPZ and call CuroboWorkerClient.solve_ik>'`
- Exit code: `1`
- Key output: `ModuleNotFoundError: No module named 'numpy'` while the bare client tried to read the NPZ.
- Conclusion: this is not an IK/IPC failure. The real Isaac runtime obtains NumPy after AppLauncher; for standalone exact-case regression, use the existing `curobo_v2` NumPy without changing environments.

## 2026-08-16 15:41 +08:00 — E. candidate3800 exact five-stage GPU IK

- Purpose: verify current cuRobo V2 solver API, returned solution shapes, fixed thresholds, and continuity selection on a known case.
- Conda environment: `curobo_v2` client and persistent `curobo_v2` worker.
- Working directory: `/home/lin/Projects/DexGraspNet2_Wuji2`
- Command: `conda run -n curobo_v2 python -c '<load candidate3800 exact base-frame targets and call CuroboWorkerClient.solve_ik>'`
- Exit code: `0`
- Key output: accepted solutions per PREGRASP/COVER/GRASP/SQUEEZE/LIFT = `32/34/34/34/31` out of 48; raw success = `33/37/37/37/31`; every stage selected continuously; solve time `1.989 s`; selected position/orientation errors are far below fixed limits and minimum selected inner margin is `0.4959 rad` (~28.41 deg).
- Conclusion: E. known candidate3800 GPU IK PASS. Multi-solution preservation inside the solver and q-reference chaining work with installed cuRobo V2.

## 2026-08-16 15:42 +08:00 — C. Existing robot collision model GPU load

- Purpose: validate that the reviewed YAML and asset aliases load in installed cuRobo and produce collision geometry.
- Conda environment: `curobo_v2` client and persistent worker.
- Working directory: `/home/lin/Projects/DexGraspNet2_Wuji2`
- Command: `conda run -n curobo_v2 python -c '<call CuroboWorkerClient.robot_spheres at the reference posture>'`
- Exit code: `0`
- Key output: 35 active joints and 233 collision spheres returned.
- Conclusion: C. robot collision model build/load PASS using the existing generated YAML; no overwrite or vendor change is needed.

## 2026-08-16 15:45 +08:00 — F/G/H. Top-20 five-stage coarse GPU regression

- Purpose: run fixed-threshold pre-cleanup regression covering candidate3800, candidate34, and score-ranked Top-20.
- Conda environment: `curobo_v2`.
- Working directory: `/home/lin/Projects/DexGraspNet2_Wuji2`
- Command:

  ```bash
  conda run -n curobo_v2 python 08_dual_arm_scene_layout/isaaclab_control/diagnostics/scripts/GPT_cuRoboV2_GPU全场景IK对比.py --reference-case 06_leap_to_wuji2_final_pipeline/01_cases/live_dynamic_scene0000_dog_candidate3800 --target dog --capture-target 08_dual_arm_scene_layout/captures/live_dynamic_scene0000/dgn2/dog --scope valid --stages pregrasp,cover,grasp,squeeze,lift --order score --limit 20 --seeds 48 --batch-size 64 --output-root 08_dual_arm_scene_layout/isaaclab_control/core/worklog/raw/top20_preintegration
  ```

- Exit code: `0`
- Key output: 8/20 candidates PASS all five stages; candidate3800 PASS (worst-stage margin 32.49 deg); candidate34 PASS (worst-stage margin 28.41 deg); Top-10 4/10; throughput 1812.55 candidates/s; no CPU/GPU classification mismatches among these 20.
- Raw output: `core/worklog/raw/top20_preintegration/curobo_gpu_scene_scan.json`, `.csv`, and `cpu_vs_curobo_comparison.json`.
- Conclusion: F candidate3800 PASS; G candidate34 PASS; H Top-20 8/20 PASS. This is coarse pure-IK evidence only, not exact collision/path/physics reachability.

## 2026-08-16 15:48 +08:00 — RGB-D Mapper smoke, initial attempt

- Purpose: build separate non-target and GroundedSAM target TSDF/ESDF layers from the formal capture.
- Conda environment: `curobo_v2`.
- Working directory: `/home/lin/Projects/DexGraspNet2_Wuji2`
- Command: `conda run -n curobo_v2 python core/tools/smoke_test_rgbd_map.py ...depth_m.npy ...intrinsics.npy ...T_world_camera.npy ...grounded_sam/dog/mask.npy`
- Exit code: `1`
- Key output: installed cuRobo 0.8.x camera integrator dereferenced `rgb_images.shape`; supplied depth-only `CameraObservation.rgb_image` was `None`.
- Conclusion: concrete installed-API mismatch; official local tests use uint8 `BxHxWx3` RGB. Add only a zero-valued placeholder plus resolution metadata.

## 2026-08-16 15:50 +08:00 — RGB-D Mapper smoke after minimal compatibility fix

- Purpose: verify native Mapper -> TSDF -> `compute_esdf()` for scene and target layers.
- Conda environment: `curobo_v2`.
- Working directory: `/home/lin/Projects/DexGraspNet2_Wuji2`
- Command: same formal capture smoke command as above.
- Exit code: `0`
- Key output: PASS; depth 720x1280; 407,537 valid pixels; fitted extent `[1.0604, 0.7002, 0.4318] m`; scene and target ESDF both shape `[54,36,22]`, voxel size `0.02 m`.
- Conclusion: native observed-scene and target-layer TSDF/ESDF construction works with installed cuRobo 0.8.x.

## 2026-08-16 15:53 +08:00 — ESDF semantic and robot-sphere diagnostics

- Purpose: verify observed-free/surface/occluded semantics and robot-state collision queries.
- Conda environment: `curobo_v2` persistent worker.
- Working directory: `/home/lin/Projects/DexGraspNet2_Wuji2`
- Command: worker `build_map`, diagnostic `query_spheres`, then `check_robot_state` with `T_world_base`.
- Exit code: `0`
- Key output: farther front sphere is `OBSERVED_FREE`; surface sphere is near-surface and collides when its radius exceeds ESDF distance; behind-surface sphere is `OCCLUDED_UNKNOWN`; reference robot has 0 blocking collisions and 233/233 unknown spheres because the current camera view does not observe the robot.
- Conclusion: collision and unknown are separate. Observed-safe must not be reported as guaranteed collision-free.

## 2026-08-16 15:58 +08:00 — Collision-aware candidate3800 multi-solution IK

- Purpose: hard-filter every IK-accepted solution with phase-aware scene/target ESDF before continuity selection.
- Conda environment: `curobo_v2` persistent worker.
- Working directory: `/home/lin/Projects/DexGraspNet2_Wuji2`
- Command: worker `build_map` followed by collision-aware `solve_ik` for five exact candidate3800 targets, 48 seeds.
- Exit code: `0`
- Key output: IK accepted `32/34/34/34/31`; collision-feasible `32/34/34/34/31`; selected chain observed-collision PASS at every stage; unknown exposure true at every stage (167–233 of 233 robot spheres unknown).
- Conclusion: IK and observed-scene collision gates PASS, while single-view unknown exposure remains explicitly unresolved and cannot be called guaranteed safe.

## 2026-08-16 16:02 +08:00 — J. Isaac Lab dry-run compatibility iterations

- Purpose: launch the formal runtime in `isaaclab22_sim50`, settle only, read q_current, and execute Route-C V2 planning without action.
- Conda environment: Isaac runtime `isaaclab22_sim50`; worker `curobo_v2`.
- Working directory: `/home/lin/Projects/DexGraspNet2_Wuji2`
- Command: `bash runtime/launchers/run_full_pick_place_25s_dog_candidate3800.sh --headless --dry-run`
- Initial outcomes: missing local Isaac Lab source path; then missing lightweight Isaac Lab dependencies; then missing archived-case `leap_selected_rank0.npz`; then worker inherited Kit `PYTHONPATH` and polluted base conda plugins.
- Fixes: launcher-level local Isaac Lab 2.2 source path; minimal fixed-environment dependencies only (`setuptools<81`, `numpy<2`, `flatdict`, `gymnasium==0.29.0`, `prettytable`, `toml`, `hidapi`, `trimesh`, `h5py`); use `case.json` candidate metadata rather than legacy collision-filter output; sanitize worker subprocess Python/Isaac/LD paths and disable base conda plugins.
- Conclusion: each failure was isolated; no CUDA/PyTorch/Isaac Sim/Isaac Lab version upgrade or environment rebuild occurred.

## 2026-08-16 16:08 +08:00 — J. Route-C V2 Isaac Lab dry-run

- Purpose: validate the full production planning call chain after real PhysX settle, without entering any motion state.
- Conda environment: Isaac runtime `isaaclab22_sim50`; worker `curobo_v2`.
- Working directory: `/home/lin/Projects/DexGraspNet2_Wuji2`
- Command: same formal `--headless --dry-run` launcher command.
- Exit/result: planning completed and report was written; Kit shutdown then hung and required interrupt after completion.
- Key output: fixed-base 35-DOF audit PASS; q_current deg `[49.058,-69.498,-0.798,39.175,33.535,1.191,25.523]`; PREGRASP through RETREAT all IK threshold PASS and observed-collision PASS; all stages unknown exposure true; report at `outputs/full_pick_place_25s_dog_candidate3800/route_c_v2_planning.json`.
- Conclusion: J. runtime dry-run planning PASS, with a separate headless Kit shutdown-hang limitation and explicit unknown exposure.

## 2026-08-16 16:12 +08:00 — K prerequisite: current ft04 static stability gate

- Purpose: enforce the 10-second gravity-on, IK-off, no-motion stability gate before any action smoke.
- Conda environment: `isaaclab22_sim50`.
- Working directory: `/home/lin/Projects/DexGraspNet2_Wuji2`
- Command: `bash diagnostics/launchers/run_initial_stability.sh --config core/worklog/static_gate_ft04.json --headless`
- Exit/result: formal report `FAIL`; Kit shutdown hung after report and was interrupted.
- Key output: right arm FAIL (max target error `1.465 deg`, max speed `0.0877 rad/s`); Wuji2 FAIL (max target error `0.321 deg`, max speed `0.0398 rad/s`); flange PASS; wrist PASS. Report: `core/worklog/raw/static_gate_ft04_current/report.json`.
- Conclusion: K physical/action smoke is locked and was not run. No threshold, gain, effort, asset, or drive tuning was changed.

## 2026-08-16 16:14 +08:00 — Legacy production-path archive

- Purpose: remove old SciPy/Pinocchio and complete-mesh collision implementations from active runtime/tools while preserving provenance.
- Conda environment: shell/base context.
- Working directory: `/home/lin/Projects/DexGraspNet2_Wuji2`
- Command: move `runtime_rebase_ik.py` plus tools `04`–`09` and `12`–`14` to `history/legacy_route_c_cpu_mesh/`.
- Exit code: `0`
- Key output: files moved intact; historical outputs untouched.
- Conclusion: active production runtime no longer imports/calls SciPy `least_squares`, old coarse reachability, or complete-mesh path collision.

## 2026-08-16 15:50 +08:00 — Final local regression and audit

- Purpose: re-run the post-integration regression, core tests, production-path scan, syntax compilation, and final Git checks without staging or publishing.
- Conda environment: `curobo_v2` for core/GPU work; shell for read-only Git audit.
- Working directory: `/home/lin/Projects/DexGraspNet2_Wuji2`
- Commands:

  ```bash
  conda run -n curobo_v2 python -m unittest discover -s 08_dual_arm_scene_layout/isaaclab_control/core/tests -t 08_dual_arm_scene_layout/isaaclab_control
  conda run -n curobo_v2 python 08_dual_arm_scene_layout/isaaclab_control/diagnostics/scripts/GPT_cuRoboV2_GPU全场景IK对比.py --reference-case 06_leap_to_wuji2_final_pipeline/01_cases/live_dynamic_scene0000_dog_candidate3800 --target dog --capture-target 08_dual_arm_scene_layout/captures/live_dynamic_scene0000/dgn2/dog --scope valid --stages pregrasp,cover,grasp,squeeze,lift --order score --limit 20 --seeds 48 --batch-size 64 --output-root 08_dual_arm_scene_layout/isaaclab_control/core/worklog/raw/top20_postintegration
  git status --short
  git diff --check
  git diff --stat
  git diff
  ```

- Exit code: `0` for unit tests, GPU regression, diff check, and compilation audit.
- Key output: 15/15 tests PASS; post-integration Top-20 remains 8/20 PASS with candidate3800 and candidate34 PASS; no active runtime/core legacy IK/collision import; tracked diff stat is 23 files, 177 insertions, 2051 deletions (untracked new core/archive files are not included by Git until staging, which was intentionally not done).
- Conclusion: local audit is clean. No vendor, DGN2, URDF, USD, acceptance-threshold, Git index, commit, remote, or GitHub mutation occurred.

## 2026-08-16 15:50 +08:00 — Isaac runtime environment version audit

- Purpose: record the minimal fixed-environment dependencies used by the real Isaac Lab runtime.
- Conda environment: `isaaclab22_sim50`.
- Working directory: `/home/lin/Projects/DexGraspNet2_Wuji2`
- Command: `conda run -n isaaclab22_sim50 python -c '<print Python and dependency versions>'`
- Exit code: `0`
- Key output: Python `3.11.15`; NumPy `1.26.4`; Gymnasium `0.29.0`; flatdict `4.0.1`; prettytable `3.3.0`; trimesh `5.0.0`; h5py `3.16.0`.
- Conclusion: only lightweight compatibility dependencies were added; CUDA, driver, PyTorch, Isaac Sim, and Isaac Lab were not upgraded or rebuilt.

## 2026-08-16 18:47 +08:00 — Closed-loop V1 read-only kickoff audit

- Purpose: confirm worktree/root, re-read durable rules and closed-loop instructions, inspect the closed-loop patch and local Grounded-SAM environment before editing.
- Conda environment: shell/base for read-only repository audit; `groundedsam` for vision environment audit.
- Working directory: `/home/lin/Projects/DexGraspNet2_Wuji2`
- Commands:

  ```bash
  pwd
  git status --short
  git diff --stat
  sed -n '1,260p' .agents/skills/isaaclab22-manipulator-control/SKILL.md
  sed -n '1,220p' AGENTS.md
  sed -n '1,260p' 08_dual_arm_scene_layout/isaaclab_control/closed_loop/CODEX_INSTRUCTION.md
  find 08_dual_arm_scene_layout/isaaclab_control/closed_loop -type f | sort
  conda env list
  conda run -n groundedsam python scripts/check_environment.py
  ```

- Exit code: `0` for root/status/diff, closed-loop reads, env list, and Grounded-SAM audit; `reference.md`/`evaluations.md` referenced by the skill are absent.
- Key output: project root correct; `closed_loop/` and `run_closed_loop.sh` are untracked patch files; tracked diff stat is empty; `groundedsam` exists with valid GroundingDINO/SAM weights and local third-party sources.
- Conclusion: proceed with local integration only; physical execution remains locked because the previous ft04 static gate is FAIL.

## 2026-08-16 18:50 +08:00 — Restricted-channel GPU visibility check

- Purpose: distinguish real GPU availability from this execution channel's CUDA visibility while auditing cuRobo and Grounded-SAM.
- Conda environment: `groundedsam`, `curobo_v2`.
- Working directory: `/home/lin/Projects/DexGraspNet2_Wuji2`
- Commands:

  ```bash
  conda run -n groundedsam python scripts/check_environment.py
  conda run -n curobo_v2 python -c '<load cuRobo robot YAML on cuda:0>'
  ```

- Exit code: Grounded-SAM audit `0`; cuRobo CUDA load `1` in this restricted channel.
- Key output: Grounded-SAM packages and weights are valid, but `torch.cuda.is_available()` is false in this channel; cuRobo CUDA load reports `RuntimeError: No CUDA GPUs are available`.
- Conclusion: do not treat the restricted-channel CUDA failure as a hardware failure. GPU validation must use the host execution channel; CPU/static tests can proceed locally.

## 2026-08-16 18:54 +08:00 — Closed-loop interface implementation and pure tests

- Purpose: connect missing closed-loop interfaces without enabling physical motion.
- Conda environment: shell/base for edits; `curobo_v2` for pure Python compile/unit tests; `groundedsam` for vision smoke.
- Working directory: `/home/lin/Projects/DexGraspNet2_Wuji2`
- Commands:

  ```bash
  find 08_dual_arm_scene_layout/isaaclab_control/closed_loop 08_dual_arm_scene_layout/isaaclab_control/core 08_dual_arm_scene_layout/isaaclab_control/runtime/scripts 08_dual_arm_scene_layout/isaaclab_control/tools -name '*.py' -print0 | xargs -0 python -m py_compile
  python -m unittest discover -s 08_dual_arm_scene_layout/isaaclab_control/closed_loop/tests -t /home/lin/Projects/DexGraspNet2_Wuji2
  python -m unittest discover -s 08_dual_arm_scene_layout/isaaclab_control/core/tests -t 08_dual_arm_scene_layout/isaaclab_control
  python -m json.tool 08_dual_arm_scene_layout/isaaclab_control/closed_loop/config/closed_loop.json
  python -m json.tool 08_dual_arm_scene_layout/isaaclab_control/runtime/config/full_pick_place_closed_loop_template.json
  conda run --no-capture-output -n groundedsam python closed_loop/scripts/grounded_sam_backend.py --image <GroundedSAM smoke dog RGB> --text dog --output /tmp/dgn2_closed_loop_groundedsam_smoke
  ```

- Exit code: `0` for py_compile, core tests, closed-loop tests after adding `tests/__init__.py`, JSON parsing, and Grounded-SAM smoke.
- Key output: closed-loop tests `4/4 PASS`; core tests `15/15 PASS`; GroundingDINO `dog` score `0.917552`, SAM mask `76198` pixels, required `mask.npy`, `overlay.png`, and normalized `result.json` written.
- Conclusion: live Grounded-SAM adapter, measured capture state output, generic runtime launcher/template, self-collision field wiring, continuous-path field wiring, and fail-closed runtime gate are ready for GPU/Isaac planning validation.

## 2026-08-16 19:37 +08:00 — GPU batch screening regression and gate semantics

- Purpose: stop slow per-candidate scanning, restore a known-good GPU IK regression first, then implement and validate the grouped batched worker primitive for production candidate screening.
- Conda environment: `isaaclab22_sim50` parent process; cuRobo worker subprocess in `curobo_v2`.
- Working directory: `/home/lin/Projects/DexGraspNet2_Wuji2`
- Commands:

  ```bash
  /home/lin/miniconda3/envs/isaaclab22_sim50/bin/python \
    08_dual_arm_scene_layout/isaaclab_control/closed_loop/scripts/regress_known_good_5stage_ik.py \
    --project-root /home/lin/Projects/DexGraspNet2_Wuji2 \
    --output 08_dual_arm_scene_layout/isaaclab_control/core/worklog/raw/known_good_5stage_ik_regression.json

  /home/lin/miniconda3/envs/isaaclab22_sim50/bin/python \
    08_dual_arm_scene_layout/isaaclab_control/closed_loop/scripts/regress_known_good_5stage_ik.py \
    --project-root /home/lin/Projects/DexGraspNet2_Wuji2 \
    --output 08_dual_arm_scene_layout/isaaclab_control/core/worklog/raw/known_good_5stage_ik_regression_after_batch_patch.json

  /home/lin/miniconda3/envs/isaaclab22_sim50/bin/python - <<'PY'
  # grouped known-good IK regression; raw report:
  # core/worklog/raw/known_good_grouped_ik_regression.json
  PY

  /home/lin/miniconda3/envs/isaaclab22_sim50/bin/python \
    08_dual_arm_scene_layout/isaaclab_control/closed_loop/scripts/validate_collision_gate_semantics.py \
    --project-root /home/lin/Projects/DexGraspNet2_Wuji2 \
    --output 08_dual_arm_scene_layout/isaaclab_control/core/worklog/raw/collision_gate_semantics.json

  python3 -m py_compile \
    08_dual_arm_scene_layout/isaaclab_control/core/bridge/worker_client.py \
    08_dual_arm_scene_layout/isaaclab_control/core/bridge/curobo_worker.py \
    08_dual_arm_scene_layout/isaaclab_control/closed_loop/scripts/*.py \
    08_dual_arm_scene_layout/isaaclab_control/closed_loop/orchestrator.py \
    06_leap_to_wuji2_final_pipeline/02_scripts/case_paths.py \
    06_leap_to_wuji2_final_pipeline/02_scripts/05_build_isaacsim_validation.py

  /home/lin/miniconda3/envs/isaaclab22_sim50/bin/python -m unittest \
    08_dual_arm_scene_layout.isaaclab_control.closed_loop.tests.test_closed_loop_logic
  ```

- Exit code: `0` for known-good regression before/after patch, grouped IK regression, collision semantics validation, py_compile, and closed-loop unit tests. One base `python3 -m unittest` attempt failed because base Python has no NumPy; it was rerun in `isaaclab22_sim50` and passed.
- Key output: known-good candidate3800 first five stages old raw `[32,36,36,36,31]`, current raw `[33,37,37,37,31]`; grouped request `group_count=1`, `pose_count=5`, raw `[33,37,37,37,31]`; self/path semantics PASS with safe self `true`, folded self `false`, safe path `true`, colliding path `false`; closed-loop logic tests `4/4 PASS`.
- Conclusion: current worker contract is not the cause of the earlier raw-success concern. The slow path was architectural: one candidate process/worker/map per candidate. The new grouped worker primitive and chunk gate are ready for same-frame planning validation, but no physical action was run.

## 2026-08-16 19:51 +08:00 — Top-16 lazy batch pick validation

- Purpose: fix the pick-stage path contract, guarantee one worker/map per closed-loop screening call, lazy-materialize only the current 16-candidate chunk, and validate on one real capture + mask without full-route or physical motion.
- Conda environment: parent `isaaclab22_sim50`; cuRobo worker subprocess in `curobo_v2`; DGN2 case builder in `graspnet2.0`; retarget scripts in `wuji_retargeting`.
- Working directory: `/home/lin/Projects/DexGraspNet2_Wuji2`
- Commands:

  ```bash
  python3 -m py_compile \
    08_dual_arm_scene_layout/isaaclab_control/core/bridge/curobo_worker.py \
    08_dual_arm_scene_layout/isaaclab_control/core/bridge/worker_client.py \
    08_dual_arm_scene_layout/isaaclab_control/closed_loop/scripts/screen_pick_batches.py \
    08_dual_arm_scene_layout/isaaclab_control/closed_loop/scripts/batch_pick_candidate_gate.py \
    08_dual_arm_scene_layout/isaaclab_control/closed_loop/orchestrator.py

  /home/lin/miniconda3/envs/isaaclab22_sim50/bin/python \
    08_dual_arm_scene_layout/isaaclab_control/closed_loop/scripts/screen_pick_batches.py \
    --project-root /home/lin/Projects/DexGraspNet2_Wuji2 \
    --prediction 08_dual_arm_scene_layout/isaaclab_control/outputs/closed_loop_sessions/20260816_190311/cycle_001/capture/dgn2/dog/official_leap_1024_target_ranked.npz \
    --network-input 08_dual_arm_scene_layout/isaaclab_control/outputs/closed_loop_sessions/20260816_190311/cycle_001/capture/dgn2/dog/network_input.npz \
    --capture-root 08_dual_arm_scene_layout/isaaclab_control/outputs/closed_loop_sessions/20260816_190311/cycle_001/capture \
    --settled-manifest 08_dual_arm_scene_layout/isaaclab_control/outputs/closed_loop_sessions/20260816_190311/cycle_001/capture/settled_scene_manifest.json \
    --robot-state 08_dual_arm_scene_layout/isaaclab_control/outputs/closed_loop_sessions/20260816_190311/cycle_001/capture/robot_state.json \
    --mask 08_dual_arm_scene_layout/isaaclab_control/outputs/closed_loop_sessions/20260816_190311/cycle_001/capture/grounded_sam/dog/mask.npy \
    --sim-target-segmentation-id 3 \
    --scratch-root 08_dual_arm_scene_layout/isaaclab_control/outputs/closed_loop_sessions/20260816_190311/cycle_001/scratch/top16_batch_validation \
    --output 08_dual_arm_scene_layout/isaaclab_control/core/worklog/raw/top16_batch_pick_validation.json \
    --network-python /home/lin/miniconda3/envs/graspnet2.0/bin/python \
    --retarget-python /home/lin/Projects/DexGraspNet2_Wuji2/01_environment/conda/wuji_retargeting/bin/python \
    --planner-python /home/lin/miniconda3/envs/isaaclab22_sim50/bin/python \
    --candidate-case-prefix top16batch \
    --limit 16 \
    --chunk-size 16

  /home/lin/miniconda3/envs/isaaclab22_sim50/bin/python -m unittest \
    08_dual_arm_scene_layout.isaaclab_control.closed_loop.tests.test_closed_loop_logic

  git diff --check
  git status --short
  git diff --stat
  ```

- Exit code: `0` for py_compile, Top-16 validation command, closed-loop unit tests, and `git diff --check`.
- Key output: Top-16 validation status `FAIL` because no candidate passed every pick gate; worker_start_count `1`; map_build_count `1`; chunk_size `16`; tested/materialized candidates `16`; one grouped solve contained `80` poses; mean wall time per tested candidate `4.137 s`; rank14/candidate1422 had raw IK `[36,37,37,37,34]` and IK accepted `[33,35,35,35,33]` but collision-filtered accepted `[0,0,0,0,0]`.
- Conclusion: the three requested architecture corrections are in place. Top-16 same-frame pick screening did not find a feasible candidate, and the run stopped before full-route/placement/physical execution as requested.

## 2026-08-16 20:08 +08:00 — Candidate1422 collision diagnosis and all-candidate GPU prefilter

- Purpose: diagnose why candidate1422 collision filtering kills every IK solution, then benchmark all-candidate GPU prefilter without retarget/full-route/physical motion.
- Conda environment: parent `isaaclab22_sim50`; cuRobo worker subprocess in `curobo_v2`.
- Working directory: `/home/lin/Projects/DexGraspNet2_Wuji2`
- Commands:

  ```bash
  /home/lin/miniconda3/envs/isaaclab22_sim50/bin/python \
    08_dual_arm_scene_layout/isaaclab_control/closed_loop/scripts/diagnose_candidate_collision.py \
    --project-root /home/lin/Projects/DexGraspNet2_Wuji2 \
    --case-root 08_dual_arm_scene_layout/isaaclab_control/outputs/closed_loop_sessions/20260816_190311/cycle_001/scratch/top16_batch_validation/chunk_000/top16batch_r014_cand1422 \
    --capture-root 08_dual_arm_scene_layout/isaaclab_control/outputs/closed_loop_sessions/20260816_190311/cycle_001/capture \
    --robot-state 08_dual_arm_scene_layout/isaaclab_control/outputs/closed_loop_sessions/20260816_190311/cycle_001/capture/robot_state.json \
    --mask 08_dual_arm_scene_layout/isaaclab_control/outputs/closed_loop_sessions/20260816_190311/cycle_001/capture/grounded_sam/dog/mask.npy \
    --output 08_dual_arm_scene_layout/isaaclab_control/core/worklog/raw/candidate1422_collision_diagnosis.json \
    --top-k 6

  /home/lin/miniconda3/envs/isaaclab22_sim50/bin/python - <<'PY'
  # measured baseline and candidate1422 hand-state self-collision spot checks
  PY

  /home/lin/miniconda3/envs/isaaclab22_sim50/bin/python \
    08_dual_arm_scene_layout/isaaclab_control/closed_loop/scripts/all_candidate_gpu_prefilter.py \
    --project-root /home/lin/Projects/DexGraspNet2_Wuji2 \
    --prediction 08_dual_arm_scene_layout/isaaclab_control/outputs/closed_loop_sessions/20260816_190311/cycle_001/capture/dgn2/dog/official_leap_1024_target_ranked.npz \
    --capture-root 08_dual_arm_scene_layout/isaaclab_control/outputs/closed_loop_sessions/20260816_190311/cycle_001/capture \
    --robot-state 08_dual_arm_scene_layout/isaaclab_control/outputs/closed_loop_sessions/20260816_190311/cycle_001/capture/robot_state.json \
    --mask 08_dual_arm_scene_layout/isaaclab_control/outputs/closed_loop_sessions/20260816_190311/cycle_001/capture/grounded_sam/dog/mask.npy \
    --output 08_dual_arm_scene_layout/isaaclab_control/core/worklog/raw/all_candidate_gpu_prefilter.json \
    --gpu-batch-size 512

  /home/lin/miniconda3/envs/isaaclab22_sim50/bin/python -m unittest \
    08_dual_arm_scene_layout.isaaclab_control.closed_loop.tests.test_closed_loop_logic

  git diff --check
  git status --short
  git diff --stat
  ```

- Exit code: `0` for collision diagnosis, baseline self-collision spot checks, all-candidate prefilter, unit tests, and `git diff --check`.
- Key output: candidate1422 stage counts: pregrasp `raw=36 threshold=33 self=33 scene=0 target=0 survivors=0`; cover `37/35 self=35`; grasp `37/35 self=35 target=35 multiple=35`; squeeze `37/35 self=35 target=35 multiple=35`; lift `34/33 self=33`. Measured baseline alone reports one self-collision pair with max penetration `0.000251 m`, indicating systematic sphere/self-collision configuration sensitivity. All-candidate prefilter: total proposals `8192`, target proposals `7454`, batch size `512`, batch count `15`, raw IK reachable `4435`, threshold accepted `4347`, scene collision pass `4269`, self collision pass `0`, coarse survivors `0`, IK time `2.387 s`, total wall `18.242 s`, `408.6 candidates/s`, peak VRAM `1608 MiB`.
- Conclusion: all-candidate GPU IK prefilter is fast and viable, but current self-collision sphere semantics are over-rejecting at baseline; do not treat zero coarse survivors as proven physical impossibility until self-collision model semantics are corrected or calibrated under explicit review.

## 2026-08-16 20:22 +08:00 — Self-collision report-only policy patch

- Purpose: switch closed-loop planning-only candidate feasibility to `SELF_COLLISION_POLICY=REPORT_ONLY_UNRESOLVED` while preserving self-collision computation and diagnostics.
- Conda environment: shell/base for syntax and Git checks; attempted `isaaclab22_sim50` parent with `curobo_v2` worker for GPU rerun.
- Working directory: `/home/lin/Projects/DexGraspNet2_Wuji2`
- Commands:

  ```bash
  python3 -m py_compile \
    08_dual_arm_scene_layout/isaaclab_control/core/bridge/curobo_worker.py \
    08_dual_arm_scene_layout/isaaclab_control/core/bridge/worker_client.py \
    08_dual_arm_scene_layout/isaaclab_control/closed_loop/scripts/all_candidate_gpu_prefilter.py \
    08_dual_arm_scene_layout/isaaclab_control/closed_loop/scripts/screen_pick_batches.py \
    08_dual_arm_scene_layout/isaaclab_control/closed_loop/scripts/batch_pick_candidate_gate.py \
    08_dual_arm_scene_layout/isaaclab_control/closed_loop/scripts/route_candidate_gate.py

  /home/lin/miniconda3/envs/isaaclab22_sim50/bin/python \
    08_dual_arm_scene_layout/isaaclab_control/closed_loop/scripts/all_candidate_gpu_prefilter.py \
    --project-root /home/lin/Projects/DexGraspNet2_Wuji2 \
    --prediction 08_dual_arm_scene_layout/isaaclab_control/outputs/closed_loop_sessions/20260816_190311/cycle_001/capture/dgn2/dog/official_leap_1024_target_ranked.npz \
    --capture-root 08_dual_arm_scene_layout/isaaclab_control/outputs/closed_loop_sessions/20260816_190311/cycle_001/capture \
    --robot-state 08_dual_arm_scene_layout/isaaclab_control/outputs/closed_loop_sessions/20260816_190311/cycle_001/capture/robot_state.json \
    --mask 08_dual_arm_scene_layout/isaaclab_control/outputs/closed_loop_sessions/20260816_190311/cycle_001/capture/grounded_sam/dog/mask.npy \
    --output 08_dual_arm_scene_layout/isaaclab_control/core/worklog/raw/all_candidate_gpu_prefilter_self_report_only.json \
    --gpu-batch-size 512 \
    --pregrasp-offset-m 0.10

  git diff --check
  git diff --stat
  ```

- Exit code: `0` for py_compile and `git diff --check`. GPU rerun in the non-escalated channel failed because `CUDA is not visible to PyTorch in curobo_v2`; the escalated rerun request was rejected by execution policy.
- Key output: policy code is patched and syntax-valid. The previous completed all-candidate run already establishes `survivors_without_self_collision = scene_collision_pass = 4269` for GRASP coarse filtering. The new PREGRASP/approach-path benchmark code is present but not executed successfully due to GPU channel access.
- Conclusion: report-only policy is implemented locally; PREGRASP/approach-path benchmark requires a GPU-visible execution channel before reporting measured counts.

## 2026-08-16 20:58 +08:00 — One-command planning-only orchestrator integration attempt

- Purpose: integrate existing closed-loop modules into `./run_closed_loop.sh --planning-only` and attempt a real scene_0000 / dog one-command run.
- Conda environment: launcher now uses `isaaclab22_sim50`; cuRobo worker subprocess is configured for `curobo_v2`.
- Working directory: `/home/lin/Projects/DexGraspNet2_Wuji2`
- Commands:

  ```bash
  pwd
  git status --short
  git diff --stat -- 08_dual_arm_scene_layout/isaaclab_control/closed_loop \
    08_dual_arm_scene_layout/isaaclab_control/core/bridge \
    08_dual_arm_scene_layout/isaaclab_control/core/worklog run_closed_loop.sh

  /home/lin/miniconda3/envs/isaaclab22_sim50/bin/python -m py_compile \
    08_dual_arm_scene_layout/isaaclab_control/closed_loop/orchestrator.py \
    08_dual_arm_scene_layout/isaaclab_control/closed_loop/scripts/all_candidate_gpu_prefilter.py \
    08_dual_arm_scene_layout/isaaclab_control/closed_loop/scripts/screen_pick_batches.py \
    08_dual_arm_scene_layout/isaaclab_control/closed_loop/scripts/build_cartesian_route.py

  ./run_closed_loop.sh --planning-only
  # stdin:
  # /home/lin/Projects/DexGraspNet2_Wuji2/02_training_dataset/data/scene_datasets/wuji2_test60_10upright_10view_v1/scenes/scene_0000
  # dog

  nvidia-smi

  /home/lin/miniconda3/envs/curobo_v2/bin/python - <<'PY'
  import torch
  print(torch.cuda.is_available())
  print(torch.cuda.device_count())
  PY

  git diff --check
  ```

- Exit code: `0` for syntax checks and `git diff --check`; the first `./run_closed_loop.sh --planning-only` was manually interrupted after Isaac capture hung with GPU initialization failures in this Codex execution channel. Escalated rerun was rejected by execution policy.
- Key output: Isaac/Kit reported `NVML_ERROR_DRIVER_NOT_LOADED`, `No device could be created`, and `no CUDA-capable device is detected`; in the same Codex channel, `nvidia-smi` failed and `curobo_v2` reported `torch.cuda.is_available() == False`.
- Conclusion: the orchestrator integration is syntax-valid, but the actual one-command validation cannot be completed from this restricted command channel because it cannot see the host NVIDIA driver/GPU. This does not contradict the user's terminal `nvidia-smi`; it is a channel visibility/permission blocker.

## 2026-08-16 21:xx +08:00 — Scratch case path contract and strict coarse prefilter funnel

- Purpose: fix final-planning scratch case path contract and enforce cheap-to-expensive all-candidate coarse prefilter ordering.
- Conda environment: syntax checks in `isaaclab22_sim50`; no GPU/Isaac rerun in the restricted Codex channel.
- Working directory: `/home/lin/Projects/DexGraspNet2_Wuji2`
- Commands:

  ```bash
  grep -n "case_root =\\|coarse_prefilter\\|coarse_approach\\|survivor_indices" \
    08_dual_arm_scene_layout/isaaclab_control/closed_loop/orchestrator.py \
    08_dual_arm_scene_layout/isaaclab_control/closed_loop/scripts/all_candidate_gpu_prefilter.py \
    08_dual_arm_scene_layout/isaaclab_control/core/bridge/curobo_worker.py

  /home/lin/miniconda3/envs/isaaclab22_sim50/bin/python -m py_compile \
    08_dual_arm_scene_layout/isaaclab_control/closed_loop/orchestrator.py \
    08_dual_arm_scene_layout/isaaclab_control/closed_loop/scripts/all_candidate_gpu_prefilter.py

  git diff --check
  ```

- Exit code: `0` for py_compile and `git diff --check`.
- Key output: final planning case root is now `scratch/final_planning/rank_{rank:04d}/{case_id}`, so `Path(case_root).name == case_id`. The strict prefilter function now forwards only previous-stage survivors through GRASP IK/threshold/scene, PREGRASP IK/threshold/scene, q_current→PREGRASP path, and PREGRASP→GRASP path.
- Conclusion: path contract is fixed without changing `build_candidate_case.py`; coarse prefilter ordering no longer runs PREGRASP/path checks for candidates that failed earlier hard gates.

## 2026-08-17 -- Simulation diagnostic execution wiring

- Purpose: restore existing closed-loop execution wiring and add explicit Isaac Sim diagnostic execution flags.
- Conda environment: syntax checks in `isaaclab22_sim50`; no full Isaac/GPU run in the restricted Codex channel.
- Working directory: `/home/lin/Projects/DexGraspNet2_Wuji2`
- Commands:

  ```bash
  /home/lin/miniconda3/envs/isaaclab22_sim50/bin/python -m py_compile \
    08_dual_arm_scene_layout/isaaclab_control/closed_loop/orchestrator.py \
    08_dual_arm_scene_layout/isaaclab_control/runtime/scripts/10_run_full_pick_place.py

  git diff --check

  grep -n "sim-execute\\|no-planner-collision-check\\|diagnostic-ignore-static-gate\\|build_next_scene_manifest" \
    08_dual_arm_scene_layout/isaaclab_control/closed_loop/orchestrator.py \
    08_dual_arm_scene_layout/isaaclab_control/runtime/scripts/10_run_full_pick_place.py
  ```

- Exit code: `0` for py_compile and `git diff --check`.
- Key output: `orchestrator.py` now supports `--sim-execute --no-planner-collision-check --diagnostic-ignore-static-gate`, creates a session-local placement registry, calls the existing runtime launcher, calls `build_next_scene_manifest.py`, updates `current_scene_manifest`, and continues the existing `while True` loop.
- Conclusion: no second state machine or executor was added. Planner collision checks can be skipped without disabling Isaac/PhysX collisions; static gate remains recorded false with an explicit diagnostic override flag.

## 2026-08-17 -- Batch retarget + grouped exact IK wiring

- Purpose: replace per-survivor retarget subprocess churn with score-ordered chunk retargeting and one grouped exact IK call per chunk.
- Conda environment: build/finalize wrappers in `graspnet2.0`; LEAP->Wuji2 wrapper in `wuji_retargeting`; syntax checks in `isaaclab22_sim50`.
- Working directory: `/home/lin/Projects/DexGraspNet2_Wuji2`
- Commands:

  ```bash
  /home/lin/miniconda3/envs/graspnet2.0/bin/python \
    08_dual_arm_scene_layout/isaaclab_control/closed_loop/scripts/batch_build_candidate_cases.py \
    --project-root /home/lin/Projects/DexGraspNet2_Wuji2 \
    --prediction 08_dual_arm_scene_layout/isaaclab_control/outputs/closed_loop_sessions/20260816_190311/cycle_001/capture/dgn2/dog/official_leap_1024_target_ranked.npz \
    --network-input 08_dual_arm_scene_layout/isaaclab_control/outputs/closed_loop_sessions/20260816_190311/cycle_001/capture/dgn2/dog/network_input.npz \
    --capture-root 08_dual_arm_scene_layout/isaaclab_control/outputs/closed_loop_sessions/20260816_190311/cycle_001/capture \
    --settled-manifest 08_dual_arm_scene_layout/isaaclab_control/outputs/closed_loop_sessions/20260816_190311/cycle_001/capture/settled_scene_manifest.json \
    --sim-target-segmentation-id 3 \
    --items-json /tmp/batch_retarget_regression_seg3/items.json \
    --output /tmp/batch_retarget_regression_seg3/build_report.json

  /home/lin/Projects/DexGraspNet2_Wuji2/01_environment/conda/wuji_retargeting/bin/python \
    08_dual_arm_scene_layout/isaaclab_control/closed_loop/scripts/batch_retarget_cases.py \
    --items-json /tmp/batch_retarget_regression_seg3/items.json \
    --output /tmp/batch_retarget_regression_seg3/retarget_report.json

  /home/lin/miniconda3/envs/graspnet2.0/bin/python \
    08_dual_arm_scene_layout/isaaclab_control/closed_loop/scripts/batch_finalize_candidate_cases.py \
    --items-json /tmp/batch_retarget_regression_seg3/items.json \
    --output /tmp/batch_retarget_regression_seg3/finalize_report.json

  /home/lin/miniconda3/envs/isaaclab22_sim50/bin/python -m py_compile \
    08_dual_arm_scene_layout/isaaclab_control/closed_loop/orchestrator.py \
    08_dual_arm_scene_layout/isaaclab_control/closed_loop/scripts/batch_build_candidate_cases.py \
    08_dual_arm_scene_layout/isaaclab_control/closed_loop/scripts/batch_retarget_cases.py \
    08_dual_arm_scene_layout/isaaclab_control/closed_loop/scripts/batch_finalize_candidate_cases.py

  git diff --check
  ```

- Exit code: `0`.
- Key output: candidate330 regression against the old top16 scratch case has max numeric abs diff `0.0` for `grasp_official.npz`, `root_alignment.npz`, `squeeze_official.npz`, `final_waypoints.npz`, and `arm_flange_targets.npz`; only `retarget_source_npz` path metadata differs. One-candidate wrapper timing: build `1.005 s`, retarget `0.347 s`, finalize `1.019 s`.
- Conclusion: batch wrappers preserve retarget/flange numerical outputs. Orchestrator now processes coarse survivors by score-ordered chunks (`retarget_chunk_size=32`) and sends each chunk's `N*5` exact pick poses to one `solve_ik_groups` call.

## 2026-08-17 -- Retarget chunk size 64 and bash runtime launcher

- Purpose: increase batch retarget chunk size from 32 to 64 and avoid executable-bit dependency for the runtime launcher.
- Conda environment: syntax check in `isaaclab22_sim50`; no full Isaac/GPU run in the restricted Codex channel.
- Working directory: `/home/lin/Projects/DexGraspNet2_Wuji2`
- Commands:

  ```bash
  /home/lin/miniconda3/envs/isaaclab22_sim50/bin/python -m py_compile \
    08_dual_arm_scene_layout/isaaclab_control/closed_loop/orchestrator.py

  git diff --check

  grep -n '"retarget_chunk_size"\\|cfg.get("retarget_chunk_size"\\|sim_cmd = \\[\\|"bash"\\|runtime_launcher' \
    08_dual_arm_scene_layout/isaaclab_control/closed_loop/config/closed_loop.json \
    08_dual_arm_scene_layout/isaaclab_control/closed_loop/orchestrator.py
  ```

- Exit code: `0`.
- Key output: config has `"retarget_chunk_size": 64`; orchestrator fallback is `cfg.get("retarget_chunk_size", 64)`; runtime launcher command is `["bash", runtime_launcher, ...]`.
- Conclusion: each batch now targets 64 score-ordered survivors, and the runtime launcher no longer requires executable permission.

## 2026-08-17 -- Runtime shutdown watchdog for closed-loop continuation

- Purpose: allow the orchestrator to continue to `build_next_scene_manifest.py` after runtime has written `report.json` and `physical_replay_30fps.npz`, even if Isaac/Kit hangs during shutdown.
- Conda environment: syntax check in `isaaclab22_sim50`; no full Isaac/GPU run in the restricted Codex channel.
- Working directory: `/home/lin/Projects/DexGraspNet2_Wuji2`
- Commands:

  ```bash
  /home/lin/miniconda3/envs/isaaclab22_sim50/bin/python -m py_compile \
    08_dual_arm_scene_layout/isaaclab_control/closed_loop/orchestrator.py

  git diff --check

  grep -n "run_runtime_until_report\\|runtime_exit_grace_s\\|RUNTIME WATCHDOG" \
    08_dual_arm_scene_layout/isaaclab_control/closed_loop/orchestrator.py \
    08_dual_arm_scene_layout/isaaclab_control/closed_loop/config/closed_loop.json
  ```

- Exit code: `0`.
- Key output: `run_runtime_until_report()` streams runtime output, detects a completed `report.json` plus replay file, waits `runtime_exit_grace_s=20`, then terminates a lingering runtime process group and continues.
- Conclusion: the observed stop after `[FULL PIPELINE PASS]` is handled as an Isaac/Kit shutdown tail, not a state-machine failure.

## 2026-08-17 -- 25s runtime timing template audit

- Purpose: compare the validated `full_pick_place_25s_dog_candidate3800.json` timing/controller fields against the current closed-loop generated `runtime_config.json`.
- Conda environment: syntax check in `isaaclab22_sim50`; comparison with local Python JSON reader.
- Working directory: `/home/lin/Projects/DexGraspNet2_Wuji2`
- Commands:

  ```bash
  find 08_dual_arm_scene_layout/isaaclab_control/outputs/closed_loop_sessions \
    -path '*/execution/runtime_config.json' -printf '%T@ %p\n' | sort -n | tail -n 5

  python3 - <<'PY'
  # Compare physics_dt_s, render_interval, initial_hold_s, telemetry_hz,
  # replay_record_fps, action_duration_limit_s, durations_s,
  # endpoint_refinement, and right_arm_force_natural_frequency_groups.
  PY

  /home/lin/miniconda3/envs/isaaclab22_sim50/bin/python -m py_compile \
    08_dual_arm_scene_layout/isaaclab_control/closed_loop/orchestrator.py \
    08_dual_arm_scene_layout/isaaclab_control/runtime/scripts/10_run_full_pick_place.py

  git diff --check
  ```

- Exit code: `0`.
- Key output: all requested timing/controller fields in the latest closed-loop runtime config match the 25s candidate3800 config exactly.
- Conclusion: no code/config change was needed for 25s timing reuse. The observed long wall time is Isaac/renderer/simulation wall-clock slowdown; simulated action time remains within the 25s template (`23.58 s / 25.00 s`).

## 2026-08-17 -- Nonblocking runtime watchdog fix

- Purpose: fix the orchestrator still hanging after `[FULL PIPELINE PASS]` because the watchdog loop blocked on `stdout.readline()` and could not check `report.json`.
- Conda environment: syntax check in `isaaclab22_sim50`.
- Working directory: `/home/lin/Projects/DexGraspNet2_Wuji2`
- Commands:

  ```bash
  /home/lin/miniconda3/envs/isaaclab22_sim50/bin/python -m py_compile \
    08_dual_arm_scene_layout/isaaclab_control/closed_loop/orchestrator.py

  git diff --check
  ```

- Exit code: `0`.
- Key output: `run_runtime_until_report()` now uses `selectors.select(timeout=0.2)` before reading stdout, so it continues polling `report.json`/replay even when Isaac/Kit has stopped printing but has not exited.
- Conclusion: after report/replay are ready, the watchdog can now terminate lingering Kit shutdown and continue to `build_next_scene_manifest.py` / cycle 002.

## 2026-08-17 -- Rank 0..684 failure histogram and concise logging

- Purpose: analyze completed session `20260817_102537` without recomputation and simplify default terminal output.
- Conda environment: JSON/CSV parsing and syntax checks in `isaaclab22_sim50`.
- Working directory: `/home/lin/Projects/DexGraspNet2_Wuji2`
- Commands:

  ```bash
  /home/lin/miniconda3/envs/isaaclab22_sim50/bin/python - <<'PY'
  # Parse cycle_001/planning_result.json and batch reports only.
  # Write rank_0_684_failure_histogram.json/csv.
  PY

  /home/lin/miniconda3/envs/isaaclab22_sim50/bin/python -m py_compile \
    08_dual_arm_scene_layout/isaaclab_control/closed_loop/orchestrator.py

  git diff --check
  ```

- Exit code: `0`.
- Key output: rank0..684 counts: `EXACT_PICK_IK_FAIL=210`, `FULL_ROUTE_IK_FAIL=30`, `NOT_EVALUATED_OR_MISSING=445`; counts sum to `685`. First exact 5-stage pass is rank `14` candidate `1422`. Rank `622` candidate `6718` and rank `640` candidate `2597` both passed pick exact IK but failed full-route IK. Rank `685` candidate `5989` is the first full-route PASS in score order.
- Conclusion: histogram artifacts are written under `core/worklog/raw/`. Default terminal output is now concise; detailed subprocess stdout/stderr is written to `<session>/debug.log`, and `--verbose` restores detailed output.

## 2026-08-18 -- DualArmMount y=0.16 layout and virtual camera recalibration

- Purpose: apply final workspace-scan layout decision by moving only `/World/Layout/DualArmMount` from `[0, 0.42, 0.8]` to `[0, 0.16, 0.8]`, preserving rotation/scale and recalibrating the virtual D435i camera to SourceZone.
- Conda environment: `isaaclab22_sim50` with Isaac USD Python extension paths; no Isaac Sim app, DGN2, cuRobo, retarget, or closed-loop execution.
- Working directory: `/home/lin/Projects/DexGraspNet2_Wuji2`
- Commands:

  ```bash
  git status --short
  mkdir -p _backup_before_mount_y016_20260818
  cp 08_dual_arm_scene_layout/scenes/manual_layout_calibrated.usda _backup_before_mount_y016_20260818/
  cp 08_dual_arm_scene_layout/scenes/manual_layout_calibrated_mass_fixed.usda _backup_before_mount_y016_20260818/
  cp 08_dual_arm_scene_layout/config/manual_layout_calibrated.json _backup_before_mount_y016_20260818/

  PYTHONPATH=/home/lin/isaacsim/extscache/omni.usd.libs-1.0.1+8131b85d.lx64.r.cp311 \
  LD_LIBRARY_PATH=/home/lin/isaacsim/extscache/omni.usd.libs-1.0.1+8131b85d.lx64.r.cp311/bin:/home/lin/miniconda3/envs/isaaclab22_sim50/lib \
  conda run -n isaaclab22_sim50 python /tmp/update_mount_y016_camera.py

  python -m py_compile \
    08_dual_arm_scene_layout/scripts/05_create_virtual_depth_camera_frustum.py \
    08_dual_arm_scene_layout/scripts/06_preview_virtual_depth_camera.py \
    08_dual_arm_scene_layout/scripts/07_capture_single_rgbd.py

  git diff --check
  ```

- Exit code: `0`.
- Key output: mount `[0.0, 0.16, 0.8]`; rotation `[0,0,-90]`; scale `[1,1,1]`; d435i/camera `[3.7e-09, 0.08499997, 0.96000004]`; target `[-0.42382277, -0.15291664, 0.46]`; HFOV `81.6881 deg`; VFOV `51.8666 deg`; focal `12.11945 mm`; coverage `PASS`.
- Conclusion: both calibrated USD stages, layout JSON, markers/distances, and virtual camera metadata are synchronized. No production closed-loop, robot asset, URDF/USD vendor, table/source/placement, DGN2, retarget, cuRobo, or control logic was changed.

## 2026-08-18 -- Static layout validation attempt from Codex channel

- Purpose: run final static layout acceptance for new DualArmMount `[0,0.16,0.8]`: real sensor preview and HOME stability only.
- Conda environment: intended `isaaclab22_sim50`.
- Working directory: `/home/lin/Projects/DexGraspNet2_Wuji2`
- Commands:

  ```bash
  pgrep -a -f '/kit/kit|isaac-sim\.sh|00_check_initial_stability\.py|persistent_isaac/worker.py' || true
  nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader,nounits || true

  08_dual_arm_scene_layout/isaaclab_control/diagnostics/launchers/run_initial_stability.sh \
    --config 08_dual_arm_scene_layout/isaaclab_control/diagnostics/config/initial_stability_grouped_pd_round1_mass_fixed.json \
    --headless
  ```

- Exit code: not run to completion.
- Key output: restricted Codex channel cannot communicate with NVIDIA driver; escalation to host GPU/Isaac execution was rejected by execution policy.
- Conclusion: true Isaac/PhysX HOME stability and real rendered occlusion cannot be certified from this channel. Safe static USD/layout checks remain PASS; final physical/static validation must be run in the user's GPU-visible terminal.

## 2026-08-19 -- Simplified planning plumbing: ROI, per-batch cuRobo, concise failure logs

- Purpose: implement non-core plumbing for the simplified mechanical-arm planning direction without changing IK thresholds, retarget math, DGN2, HOME, camera, robot layout, or RFS core algorithm.
- Conda environment: `isaaclab22_sim50` for CPU unit tests; no Isaac app, DGN2, cuRobo GPU, retarget, RFS backend, or full closed-loop execution.
- Working directory: `/home/lin/Projects/DexGraspNet2_Wuji2`
- Commands:

  ```bash
  python -m py_compile \
    08_dual_arm_scene_layout/isaaclab_control/closed_loop/orchestrator.py \
    08_dual_arm_scene_layout/isaaclab_control/closed_loop/scripts/grounded_sam_backend.py \
    08_dual_arm_scene_layout/isaaclab_control/closed_loop/persistent_isaac/worker.py \
    08_dual_arm_scene_layout/isaaclab_control/closed_loop/planning/candidate_rfs_v2_runtime.py

  PYTHONPATH=08_dual_arm_scene_layout/isaaclab_control/closed_loop \
  python -m unittest -v \
    08_dual_arm_scene_layout/isaaclab_control/closed_loop/tests/test_flexible_planning.py \
    08_dual_arm_scene_layout/isaaclab_control/closed_loop/tests/test_closed_loop_logic.py

  conda run -n isaaclab22_sim50 bash -lc 'PYTHONPATH=08_dual_arm_scene_layout/isaaclab_control/closed_loop python -m unittest -v 08_dual_arm_scene_layout/isaaclab_control/closed_loop/tests/test_flexible_planning.py 08_dual_arm_scene_layout/isaaclab_control/closed_loop/tests/test_closed_loop_logic.py'

  git diff --check
  ```

- Exit code: `0` for py_compile, `0` for `isaaclab22_sim50` unittest, `0` for `git diff --check`. The base-Python unittest attempt failed only because base Python 3.14 has no NumPy.
- Key output: DINO now runs on fixed ROI `[170,0,970,700]` and writes full-image bbox coordinates; ESDF map input uses `depth_m_workspace_roi.npy` with full 720x1280 shape and unchanged K/T; persistent capture hides SourceZone/PlacementZone/Frustum/Markers only while capturing; cuRobo worker lifecycle moved to candidate batch scope; each batch writes `flexible_route_failures.jsonl`; RFS 0-PASS is reported as `NO_TARGET_REACH` or `NO_TRAJECTORY_SPACE`, not fallback.
- Conclusion: non-core plumbing is ready for user/GPU-side validation. Stage-specific IK tolerances, simplified joint-space route semantics, and RFS support-pose algorithm replacement are intentionally left for the next ChatGPT-provided core patch.

## 2026-08-19 -- PlacementZone 5 cm X-edge gap static layout update

- Purpose: move only the authored PlacementZone X coordinate so the SourceZone/PlacementZone nearest X-edge gap is exactly `0.05 m`.
- Conda environment: base for static checks; `isaaclab22_sim50` for CPU-only unittest.
- Working directory: `/home/lin/Projects/DexGraspNet2_Wuji2`
- Commands:

  ```bash
  pwd
  git status --short
  grep -RIn --exclude='*.log' --exclude-dir='.git' --exclude-dir='__pycache__' --exclude-dir='outputs' -E '0\.40069047181642775|0\.400690|PlacementZone|placement_zone|PLACEMENT_ZONE|PLACEMENT_CENTER' .
  python 08_dual_arm_scene_layout/scripts/check_placement_zone_gap.py
  python -m py_compile 08_dual_arm_scene_layout/scripts/01_create_manual_layout.py 08_dual_arm_scene_layout/scripts/check_placement_zone_gap.py
  conda run -n isaaclab22_sim50 python -m unittest -v 08_dual_arm_scene_layout.isaaclab_control.closed_loop.tests.test_flexible_planning 08_dual_arm_scene_layout.isaaclab_control.closed_loop.tests.test_closed_loop_logic
  git diff --check
  git grep -n --untracked -E '0\.40069047181642775|0\.400690' -- . ':(exclude)08_dual_arm_scene_layout/isaaclab_control/outputs/**' ':(exclude)**/*.log' ':(exclude)**/__pycache__/**' ':(exclude)DexGraspNet2_Wuji2_flexible_persistent_update/**' ':(exclude)_backup_before_mount_y016_20260818/**'
  ```

- Exit code: `0` for invariant script, py_compile, `isaaclab22_sim50` unittest, and `git diff --check`; final old-coordinate `git grep` returned `1` with no matches.
- Key output: formal gap `0.049999999999999989 m`; draft gap `0.049999999999999989 m`; old formal PlacementZone X no longer appears in active source/config/scene outside excluded historical/overlay/backup paths.
- Conclusion: formal JSON and both production USD stages are synchronized at PlacementZone center X `0.27617723418526796`; planner/runtime consumers remain dynamic readers.

## 2026-08-19 -- Experimental planner collision bypass execution flag

- Purpose: add `--experimental-bypass-planner-collision` for real Isaac simulation execution while bypassing the planner collision gates already isolated by diagnostics.
- Conda environment: base for py_compile/diff; `isaaclab22_sim50` for CPU-only unittest.
- Working directory: `/home/lin/Projects/DexGraspNet2_Wuji2`
- Commands:

  ```bash
  python -m py_compile 08_dual_arm_scene_layout/isaaclab_control/closed_loop/orchestrator.py
  conda run -n isaaclab22_sim50 python -m unittest -v 08_dual_arm_scene_layout.isaaclab_control.closed_loop.tests.test_flexible_planning 08_dual_arm_scene_layout.isaaclab_control.closed_loop.tests.test_closed_loop_logic
  git diff --check
  ```

- Exit code: `0` for all commands.
- Key output: existing `--diagnostic-disable-*` flags remain restricted to `--diagnostic-full-first-batch`; new experimental flag requires `--sim-execute` and sets effective bypass for RFS observed ESDF, Exact COVER observed ESDF, HOME->PRE observed ESDF/self-collision, and PRE->COVER observed ESDF/self-collision.
- Conclusion: experimental execution entry is wired without changing route parameters, IK tolerances, sampling, scene layout, camera, DGN2, retarget, Wuji2 control, or Isaac execution code.

## 2026-08-19 -- Standalone cuRobo RobotSegmenter capture adapter

- Purpose: adapt cuRobo V2 `RobotSegmenter` as a standalone capture-depth cleaner that writes robot-filtered planning depth without changing baseline planning/execution.
- Conda environment: `curobo_v2` for RobotSegmenter import/run; base for py_compile and diff check.
- Working directory: `/home/lin/Projects/DexGraspNet2_Wuji2`
- Commands:

  ```bash
  conda run -n curobo_v2 python -c "from curobo.perception import RobotSegmenter; from curobo.types import CameraObservation, JointState, Pose; print('RobotSegmenter import PASS')"
  python -m py_compile 08_dual_arm_scene_layout/isaaclab_control/perception/robot_segmentation/curobo_robot_segmenter.py 08_dual_arm_scene_layout/isaaclab_control/perception/robot_segmentation/run_robot_segmenter_capture.py
  conda run -n curobo_v2 python 08_dual_arm_scene_layout/isaaclab_control/perception/robot_segmentation/run_robot_segmenter_capture.py --help
  conda run -n curobo_v2 python 08_dual_arm_scene_layout/isaaclab_control/perception/robot_segmentation/run_robot_segmenter_capture.py --capture-dir 08_dual_arm_scene_layout/isaaclab_control/outputs/closed_loop_sessions/20260819_174407/cycle_001/capture
  git diff --check
  ```

- Exit code: import/py_compile/help/run/diff-check all `0` after adapter dtype fix.
- Key output: output dir `.../cycle_001/capture/planning`; active joints `35`; depth shape `[720,1280]`; robot mask pixels `225800`; robot mask fraction of valid depth `0.44399612240764663`.
- Conclusion: standalone RobotSegmenter adapter works on an existing capture and writes `robot_mask.npy/png`, `filtered_depth.npy`, `filtered_depth_preview.png`, and `robot_segmentation_report.json`; it is not connected to the baseline planner.

## 2026-08-19 22:10 +08:00 -- Route B current-to-PREGRASP MotionPlanner adapter

- Purpose: add an independent Route B backend for cuRobo V2 MotionPlanner phase 1 while preserving Route A.
- Conda environment: `curobo_v2` for cuRobo imports and standalone smoke test.
- Working directory: `/home/lin/Projects/DexGraspNet2_Wuji2`
- Commands:

  ```bash
  conda run -n curobo_v2 python -c "from curobo.motion_planner import MotionPlanner, MotionPlannerCfg; from curobo.perception import Mapper, MapperCfg; print('MotionPlanner/Mapper import PASS')"
  conda run -n curobo_v2 python -m py_compile 08_dual_arm_scene_layout/isaaclab_control/curobo_motion_planning_routeB/routeB_adapter.py 08_dual_arm_scene_layout/isaaclab_control/curobo_motion_planning_routeB/test_current_to_pregrasp.py 08_dual_arm_scene_layout/isaaclab_control/curobo_motion_planning_routeB/tests/test_import.py
  PYTHONPATH=08_dual_arm_scene_layout/isaaclab_control conda run -n curobo_v2 python -m unittest -v 08_dual_arm_scene_layout/isaaclab_control/curobo_motion_planning_routeB/tests/test_import.py
  conda run -n curobo_v2 python 08_dual_arm_scene_layout/isaaclab_control/curobo_motion_planning_routeB/test_current_to_pregrasp.py --capture-dir 08_dual_arm_scene_layout/isaaclab_control/outputs/closed_loop_sessions/20260819_174407/cycle_001/capture --route-plan 08_dual_arm_scene_layout/isaaclab_control/outputs/closed_loop_sessions/20260819_174407/cycle_001/scratch/final_planning/rank_0001/closedloop_r0001_cand0676/07_arm_execution/flexible_route_plan.npz
  git diff --check
  ```

- Exit code: import/py_compile/unittest/diff-check `0`; standalone MotionPlanner smoke exited `2`.
- Key output: Route B built RobotSegmenter-filtered ESDF and invoked `MotionPlanner.plan_cspace`; cuRobo returned `Start or End state in collision`; report/trajectory placeholders written to `.../cycle_001/capture/curobo_test_result/`.
- Conclusion: Route B phase-1 import/path/API wiring is in place. The first blocker is a real MotionPlanner endpoint collision verdict against the filtered ESDF, not a missing import or Isaac dependency.

## 2026-08-19 22:32 +08:00 -- Route B endpoint collision audit

- Purpose: isolate why Route B `current -> PREGRASP` reports `Start or End state in collision`.
- Conda environment: `curobo_v2`.
- Working directory: `/home/lin/Projects/DexGraspNet2_Wuji2`
- Commands:

  ```bash
  conda run -n curobo_v2 python -m py_compile 08_dual_arm_scene_layout/isaaclab_control/curobo_motion_planning_routeB/test_collision_audit.py
  conda run -n curobo_v2 python 08_dual_arm_scene_layout/isaaclab_control/curobo_motion_planning_routeB/test_collision_audit.py --capture-dir 08_dual_arm_scene_layout/isaaclab_control/outputs/closed_loop_sessions/20260819_174407/cycle_001/capture --route-plan 08_dual_arm_scene_layout/isaaclab_control/outputs/closed_loop_sessions/20260819_174407/cycle_001/scratch/final_planning/rank_0001/closedloop_r0001_cand0676/07_arm_execution/flexible_route_plan.npz
  conda run -n curobo_v2 python -m py_compile 08_dual_arm_scene_layout/isaaclab_control/curobo_motion_planning_routeB/routeB_adapter.py 08_dual_arm_scene_layout/isaaclab_control/curobo_motion_planning_routeB/test_current_to_pregrasp.py 08_dual_arm_scene_layout/isaaclab_control/curobo_motion_planning_routeB/test_collision_audit.py 08_dual_arm_scene_layout/isaaclab_control/curobo_motion_planning_routeB/tests/test_import.py
  PYTHONPATH=08_dual_arm_scene_layout/isaaclab_control conda run -n curobo_v2 python -m unittest -v 08_dual_arm_scene_layout/isaaclab_control/curobo_motion_planning_routeB/tests/test_import.py
  git diff --check
  ```

- Exit code: all commands `0`.
- Key output: `empty_scene=false`, `esdf_scene=false`, `self_collision_disabled=true`, `right_arm_only=true`; q_current/q_pregrasp environment ESDF collision both false with positive clearance, but self collision pair count is `1` with max penetration `0.000251334 m`, pair linear index `2576`.
- Conclusion: the observed endpoint blocker is dominated by cuRobo self-collision false positive / model semantics, not by filtered ESDF scene collision or right-arm-only environment collision.

## 2026-08-19 22:55 +08:00 -- Route B formal collision policy wiring

- Purpose: enforce Route B policy `environment_collision=true`, `self_collision=false` at cuRobo MotionPlanner config/runtime level.
- Conda environment: `curobo_v2`.
- Working directory: `/home/lin/Projects/DexGraspNet2_Wuji2`
- Commands:

  ```bash
  conda run -n curobo_v2 python -m py_compile 08_dual_arm_scene_layout/isaaclab_control/curobo_motion_planning_routeB/routeB_adapter.py 08_dual_arm_scene_layout/isaaclab_control/curobo_motion_planning_routeB/test_current_to_pregrasp.py 08_dual_arm_scene_layout/isaaclab_control/curobo_motion_planning_routeB/test_collision_audit.py 08_dual_arm_scene_layout/isaaclab_control/curobo_motion_planning_routeB/tests/test_import.py
  conda run -n curobo_v2 python 08_dual_arm_scene_layout/isaaclab_control/curobo_motion_planning_routeB/test_current_to_pregrasp.py --capture-dir 08_dual_arm_scene_layout/isaaclab_control/outputs/closed_loop_sessions/20260819_174407/cycle_001/capture --route-plan 08_dual_arm_scene_layout/isaaclab_control/outputs/closed_loop_sessions/20260819_174407/cycle_001/scratch/final_planning/rank_0001/closedloop_r0001_cand0676/07_arm_execution/flexible_route_plan.npz
  PYTHONPATH=08_dual_arm_scene_layout/isaaclab_control conda run -n curobo_v2 python -m unittest -v 08_dual_arm_scene_layout/isaaclab_control/curobo_motion_planning_routeB/tests/test_import.py
  git diff --check
  ```

- Exit code: py_compile/unittest/diff-check `0`; standalone Route B planning command exited `2` because MotionPlanner success remained false.
- Key output: old `Start or End state in collision` message no longer appeared. Report says `collision_policy={'environment_collision': true, 'self_collision': false}`, `all_self_collision_rollouts_disabled=true`, `environment_scene_collision_cfg_present=true`, `graph.enabled_in_this_run=false`, start matches q_current true, end matches q_pregrasp true, trajectory shape `(0, 35)`.
- Conclusion: Route B self-collision is disabled across IK, TrajOpt, and Graph rollout configs while environment ESDF remains enabled. The new blocker is TrajOpt failure to produce a successful C-space trajectory, not the previous self-collision start/end rejection.

## 2026-08-19 23:18 +08:00 -- Route B MotionPlanner failure mode audit

- Purpose: classify Route B `plan_cspace()` failure after endpoint collision and self-collision were ruled out.
- Conda environment: `curobo_v2`.
- Working directory: `/home/lin/Projects/DexGraspNet2_Wuji2`
- Commands:

  ```bash
  conda run -n curobo_v2 python -m py_compile 08_dual_arm_scene_layout/isaaclab_control/curobo_motion_planning_routeB/test_motion_planner_failure_audit.py
  conda run -n curobo_v2 python 08_dual_arm_scene_layout/isaaclab_control/curobo_motion_planning_routeB/test_motion_planner_failure_audit.py --capture-dir 08_dual_arm_scene_layout/isaaclab_control/outputs/closed_loop_sessions/20260819_174407/cycle_001/capture --route-plan 08_dual_arm_scene_layout/isaaclab_control/outputs/closed_loop_sessions/20260819_174407/cycle_001/scratch/final_planning/rank_0001/closedloop_r0001_cand0676/07_arm_execution/flexible_route_plan.npz
  PYTHONPATH=08_dual_arm_scene_layout/isaaclab_control conda run -n curobo_v2 python -m unittest -v 08_dual_arm_scene_layout/isaaclab_control/curobo_motion_planning_routeB/tests/test_import.py
  git diff --check
  ```

- Exit code: all diagnostic/static commands `0`.
- Key output: `graph_instance_present=true`; configured `enable_graph_attempt=1000000`, `max_attempts=2`, so both plan attempts were ordinary TrajOpt and graph fallback did not occur. Direct graph `find_path()` returned `success=false`, `debug_info=Start or End state in collision`. Linear q_current->q_pregrasp ESDF sweep with 151 samples was collision-free, min clearance `0.075222 m`. TrajOpt success count `0`; endpoint errors were near zero (`position_error=3.88e-08`, `rotation_error=2.73e-08`), but `seed_cost=1e16`.
- Conclusion: current blocker is D/E: ordinary TrajOpt failed while graph fallback was not used in configured Route B; direct graph is present but fails its own start/end feasibility path. The linear ESDF path is free, so the failure is not explained by environment collision along the simple joint interpolation.

## 2026-08-19 23:43 +08:00 -- Route B TrajOpt feasibility constraint audit

- Purpose: locate the concrete cuRobo feasibility constraint behind Route B `TrajOptSolverResult.success=false`.
- Conda environment: `curobo_v2`.
- Working directory: `/home/lin/Projects/DexGraspNet2_Wuji2`
- Commands:

  ```bash
  conda run -n curobo_v2 python -m py_compile 08_dual_arm_scene_layout/isaaclab_control/curobo_motion_planning_routeB/test_trajopt_feasibility_audit.py
  conda run -n curobo_v2 python 08_dual_arm_scene_layout/isaaclab_control/curobo_motion_planning_routeB/test_trajopt_feasibility_audit.py --capture-dir 08_dual_arm_scene_layout/isaaclab_control/outputs/closed_loop_sessions/20260819_174407/cycle_001/capture --route-plan 08_dual_arm_scene_layout/isaaclab_control/outputs/closed_loop_sessions/20260819_174407/cycle_001/scratch/final_planning/rank_0001/closedloop_r0001_cand0676/07_arm_execution/flexible_route_plan.npz
  PYTHONPATH=08_dual_arm_scene_layout/isaaclab_control conda run -n curobo_v2 python -m unittest -v 08_dual_arm_scene_layout/isaaclab_control/curobo_motion_planning_routeB/tests/test_import.py
  git diff --check
  ```

- Exit code: all final commands `0`.
- Key output: audit report written to `.../cycle_001/capture/curobo_test_result/trajopt_feasibility_audit.json`. Recomputed raw and interpolated metrics both report feasible `false`; failed constraints are `scene_collision` and `cspace`. `scene_collision` dominates with max `169.164154`, worst timestep `[0,76,112]`, sphere `112`, link `arm_r_link_7`. `cspace` is tiny numerical positive max `5.80343e-08`. Project-side ESDF post-check of the returned trajectory reports no collision, min clearance `0.078266 m`; acceleration/jerk are under configured limits.
- Conclusion: cuRobo TrajOpt success=false is caused by cuRobo's own `scene_collision` constraint plus a negligible `cspace` bound residual, while the project ESDF post-check does not reproduce an actual collision on the returned trajectory.

## 2026-08-20 00:16 +08:00 -- Route B cuRobo scene_collision semantics one-to-one audit

- Purpose: compare the same returned TrajOpt trajectory sphere samples between project `query_spheres` and cuRobo `SceneCollisionCost`/raw checker.
- Conda environment: `curobo_v2`.
- Working directory: `/home/lin/Projects/DexGraspNet2_Wuji2`
- Commands:

  ```bash
  conda run -n curobo_v2 python -m py_compile 08_dual_arm_scene_layout/isaaclab_control/curobo_motion_planning_routeB/test_scene_collision_semantics_audit.py
  conda run -n curobo_v2 python 08_dual_arm_scene_layout/isaaclab_control/curobo_motion_planning_routeB/test_scene_collision_semantics_audit.py --capture-dir 08_dual_arm_scene_layout/isaaclab_control/outputs/closed_loop_sessions/20260819_174407/cycle_001/capture --route-plan 08_dual_arm_scene_layout/isaaclab_control/outputs/closed_loop_sessions/20260819_174407/cycle_001/scratch/final_planning/rank_0001/closedloop_r0001_cand0676/07_arm_execution/flexible_route_plan.npz
  PYTHONPATH=08_dual_arm_scene_layout/isaaclab_control conda run -n curobo_v2 python -m unittest -v 08_dual_arm_scene_layout/isaaclab_control/curobo_motion_planning_routeB/tests/test_import.py
  git diff --check
  ```

- Exit code: all final commands `0`.
- Key output: `scene_collision_semantics_audit.json` written under `capture/curobo_test_result/`. Worst sample `t=76`, sphere `112`, link `arm_r_link_7`: project signed distance `0.143382 m`, project clearance `0.112177 m`, cuRobo rollout/raw cost `169.164154`, unit-weight penetration `0.0338328 m`, inferred cuRobo signed distance `-0.0026277 m`. First positive sample `t=61`, sphere `184`, link `r_wrist`: project clearance `0.096531 m`, cuRobo cost `0.368715`, inferred signed distance `0.0151378 m`. Project-min-clearance sample `t=64`, sphere `185`, link `r_wrist`: project clearance `0.078266 m`, cuRobo cost `4.716897`, inferred signed distance `0.0254433 m`.
- Conclusion: cuRobo raw scene collision checker and rollout constraint agree exactly; the mismatch is between cuRobo's SceneCollision voxel query and the project `query_spheres` interpretation of the VoxelGrid. Activation distance is `0`, weight is `5000`, and the formula is `constraint = weight * max(radius - signed_distance, 0)`.

## 2026-08-20 00:38 +08:00 -- Route B ESDF filtered-depth ground-truth audit

- Purpose: use original `filtered_depth` surface points as a third-party truth source to decide whether project ESDF query or cuRobo VoxelData query is correct.
- Conda environment: `curobo_v2`.
- Working directory: `/home/lin/Projects/DexGraspNet2_Wuji2`
- Commands:

  ```bash
  conda run -n curobo_v2 python -m py_compile 08_dual_arm_scene_layout/isaaclab_control/curobo_motion_planning_routeB/test_esdf_ground_truth_audit.py
  conda run -n curobo_v2 python 08_dual_arm_scene_layout/isaaclab_control/curobo_motion_planning_routeB/test_esdf_ground_truth_audit.py --capture-dir 08_dual_arm_scene_layout/isaaclab_control/outputs/closed_loop_sessions/20260819_174407/cycle_001/capture
  PYTHONPATH=08_dual_arm_scene_layout/isaaclab_control conda run -n curobo_v2 python -m unittest -v 08_dual_arm_scene_layout/isaaclab_control/curobo_motion_planning_routeB/tests/test_import.py
  git diff --check
  ```

- Exit code: all final commands `0`.
- Key output: `esdf_ground_truth_audit.json` written under `capture/curobo_test_result/`. Project query uses `grid_sample` order `zyx`, `align_corners=True`, tensor shape `[nx,ny,nz]`. On 500 filtered-depth surface points, project median/p90 abs SDF are `0.002678/0.003372 m`; cuRobo VoxelData semantic query median/p90 abs SDF are `0.168801/0.278765 m`. Voxel-center direct-read median error: project `1.49e-08 m`, cuRobo semantic query `0.071903 m`.
- Conclusion: verdict `CUROBO_SCENE_REPRESENTATION_WRONG`. Root cause is `VoxelData.params` dimension precision: params dims are `[36.0, 55.0000038, 26.9999981]`; cuRobo Warp kernel casts with `wp.int32()`, producing `[36,55,26]`, while the actual feature tensor shape is `[36,55,27]`. This corrupts X-slowest/Z-fastest flat indexing.

## 2026-08-20 01:06 +08:00 -- Route B VoxelData discrete dimension contract fix

- Purpose: fix Route B cuRobo `VoxelData.params[...,0:3]` to use authoritative `scene_grid.feature_tensor.shape`.
- Conda environment: `curobo_v2`.
- Working directory: `/home/lin/Projects/DexGraspNet2_Wuji2`
- Commands:

  ```bash
  conda run -n curobo_v2 python -m py_compile 08_dual_arm_scene_layout/isaaclab_control/curobo_motion_planning_routeB/routeB_adapter.py 08_dual_arm_scene_layout/isaaclab_control/curobo_motion_planning_routeB/test_current_to_pregrasp.py 08_dual_arm_scene_layout/isaaclab_control/curobo_motion_planning_routeB/test_esdf_ground_truth_audit.py
  conda run -n curobo_v2 python 08_dual_arm_scene_layout/isaaclab_control/curobo_motion_planning_routeB/test_esdf_ground_truth_audit.py --capture-dir 08_dual_arm_scene_layout/isaaclab_control/outputs/closed_loop_sessions/20260819_174407/cycle_001/capture
  conda run -n curobo_v2 python 08_dual_arm_scene_layout/isaaclab_control/curobo_motion_planning_routeB/test_scene_collision_semantics_audit.py --capture-dir 08_dual_arm_scene_layout/isaaclab_control/outputs/closed_loop_sessions/20260819_174407/cycle_001/capture --route-plan 08_dual_arm_scene_layout/isaaclab_control/outputs/closed_loop_sessions/20260819_174407/cycle_001/scratch/final_planning/rank_0001/closedloop_r0001_cand0676/07_arm_execution/flexible_route_plan.npz
  conda run -n curobo_v2 python 08_dual_arm_scene_layout/isaaclab_control/curobo_motion_planning_routeB/test_current_to_pregrasp.py --capture-dir 08_dual_arm_scene_layout/isaaclab_control/outputs/closed_loop_sessions/20260819_174407/cycle_001/capture --route-plan 08_dual_arm_scene_layout/isaaclab_control/outputs/closed_loop_sessions/20260819_174407/cycle_001/scratch/final_planning/rank_0001/closedloop_r0001_cand0676/07_arm_execution/flexible_route_plan.npz
  conda run -n curobo_v2 python 08_dual_arm_scene_layout/isaaclab_control/curobo_motion_planning_routeB/test_trajopt_feasibility_audit.py --capture-dir 08_dual_arm_scene_layout/isaaclab_control/outputs/closed_loop_sessions/20260819_174407/cycle_001/capture --route-plan 08_dual_arm_scene_layout/isaaclab_control/outputs/closed_loop_sessions/20260819_174407/cycle_001/scratch/final_planning/rank_0001/closedloop_r0001_cand0676/07_arm_execution/flexible_route_plan.npz
  PYTHONPATH=08_dual_arm_scene_layout/isaaclab_control conda run -n curobo_v2 python -m unittest -v 08_dual_arm_scene_layout/isaaclab_control/curobo_motion_planning_routeB/tests/test_import.py
  git diff --check
  ```

- Exit code: py_compile/ground-truth audit/scene-collision audit/feasibility audit/unittest/diff-check `0`; `test_current_to_pregrasp.py` still exits `2` because only `cspace` remains.
- Key output: `VoxelData.params` changed from `[[[36.0, 55.0000038, 26.9999981, 0.02]]]` to `[[[36.0, 55.0, 27.0, 0.02]]]` before warmup/solve. Feature shape `[36,55,27]`; feature count `53460`; dims/inv_pose/features unchanged. Ground-truth cuRobo surface median/p90 abs SDF improved from `0.168801/0.278765 m` to `0.002678/0.003372 m`; voxel-center median error from `0.071903 m` to `0`. Scene-collision constraint positive count is now `0`; timestep76 sphere112 project/curobo SDF differ by `1.68e-08 m`.
- Conclusion: Route B scene_collision false positive is fixed. `plan_cspace` still returns success=false due only to `cspace` residual max `5.80343e-08` and tiny left-arm bound residual; scene_collision is no longer a blocker.

## 2026-08-20 01:42 +08:00 -- Route B cspace numerical-bound sanitization

- Purpose: fix the final Route B `cspace` feasibility residual without changing joint limits, cspace thresholds, TrajOpt, graph, ESDF, collision spheres, or Route A.
- Conda environment: `curobo_v2`.
- Working directory: `/home/lin/Projects/DexGraspNet2_Wuji2`
- Commands:

  ```bash
  python -m py_compile 08_dual_arm_scene_layout/isaaclab_control/curobo_motion_planning_routeB/routeB_adapter.py 08_dual_arm_scene_layout/isaaclab_control/curobo_motion_planning_routeB/test_trajopt_feasibility_audit.py 08_dual_arm_scene_layout/isaaclab_control/curobo_motion_planning_routeB/test_scene_collision_semantics_audit.py 08_dual_arm_scene_layout/isaaclab_control/curobo_motion_planning_routeB/test_current_to_pregrasp.py
  conda run -n curobo_v2 python 08_dual_arm_scene_layout/isaaclab_control/curobo_motion_planning_routeB/test_trajopt_feasibility_audit.py --capture-dir 08_dual_arm_scene_layout/isaaclab_control/outputs/closed_loop_sessions/20260819_174407/cycle_001/capture --route-plan 08_dual_arm_scene_layout/isaaclab_control/outputs/closed_loop_sessions/20260819_174407/cycle_001/scratch/final_planning/rank_0001/closedloop_r0001_cand0676/07_arm_execution/flexible_route_plan.npz
  conda run -n curobo_v2 python 08_dual_arm_scene_layout/isaaclab_control/curobo_motion_planning_routeB/test_current_to_pregrasp.py --capture-dir 08_dual_arm_scene_layout/isaaclab_control/outputs/closed_loop_sessions/20260819_174407/cycle_001/capture --route-plan 08_dual_arm_scene_layout/isaaclab_control/outputs/closed_loop_sessions/20260819_174407/cycle_001/scratch/final_planning/rank_0001/closedloop_r0001_cand0676/07_arm_execution/flexible_route_plan.npz
  git diff --check
  ```

- Exit code: all final commands `0`.
- Key output: Route B now sanitizes only tiny pre-planning numeric joint-bound residuals with tolerance `1e-5 rad` and interior margin `1e-6 rad`, using MotionPlanner rollout position bounds. Corrections were limited to static left-arm DOF: `arm_l_joint_2` `-7.394e-08 -> +1.0e-06` and `arm_l_joint_4` `+4.818e-06 -> -1.0e-06`, applied to both `q_current_planning` and `q_pregrasp_planning`. Max original violation `4.818e-06 rad`; no large violation.
- Conclusion: feasibility audit now reports raw/interpolated `scene_collision=0`, `cspace=0`, no failed constraints, and min environment clearance `0.075290 m`. `test_current_to_pregrasp.py` returns `success=True`, `trajectory_point_count=41`, planning time `1.863 s`; returned trajectory postcheck reports environment collision false, min clearance `0.075290 m`, joint-limit violations `0`.

## 2026-08-20 02:18 +08:00 -- Route B true right-arm-only current-to-PREGRASP integration

- Purpose: wire ChatGPT-provided `right_arm_only_core` into local Route B without rewriting its algorithm, and prove `current -> PREGRASP` is a true 7DOF cuRobo MotionPlanner solve.
- Conda environment: `curobo_v2`.
- Working directory: `/home/lin/Projects/DexGraspNet2_Wuji2`
- Commands:

  ```bash
  conda run -n curobo_v2 python -m py_compile 08_dual_arm_scene_layout/isaaclab_control/curobo_motion_planning_routeB/routeB_adapter.py 08_dual_arm_scene_layout/isaaclab_control/curobo_motion_planning_routeB/test_current_to_pregrasp_right_arm_only.py 08_dual_arm_scene_layout/isaaclab_control/curobo_motion_planning_routeB/routeB_right_arm_only_core_v1/right_arm_only_core/*.py
  PYTHONPATH=08_dual_arm_scene_layout/isaaclab_control/curobo_motion_planning_routeB/routeB_right_arm_only_core_v1 conda run -n curobo_v2 python -m unittest -v right_arm_only_core.test_contract
  conda run -n curobo_v2 python 08_dual_arm_scene_layout/isaaclab_control/curobo_motion_planning_routeB/test_current_to_pregrasp_right_arm_only.py --capture-dir 08_dual_arm_scene_layout/isaaclab_control/outputs/closed_loop_sessions/20260819_174407/cycle_001/capture --route-plan 08_dual_arm_scene_layout/isaaclab_control/outputs/closed_loop_sessions/20260819_174407/cycle_001/scratch/final_planning/rank_0001/closedloop_r0001_cand0676/07_arm_execution/flexible_route_plan.npz
  PYTHONPATH=08_dual_arm_scene_layout/isaaclab_control conda run -n curobo_v2 python -m unittest -v 08_dual_arm_scene_layout/isaaclab_control/curobo_motion_planning_routeB/tests/test_import.py
  git diff --check
  ```

- Exit code: all final commands `0`.
- Key output: `trajectory_right_arm.npz` and `report_right_arm.json` generated under `capture/curobo_test_result/`. `MotionPlanner success=True`, `planner.action_dim=7`, active joints exactly `arm_r_joint_1..7`, locked joint count `28`, `raw_result.solution.shape=[1,1,16,7]`, trajectory shape `[41,7]`, dt `0.02500000037 s`, duration `1.0000000149 s`.
- Conclusion: true 7DOF Route B `current -> PREGRASP` planning is working. Postcheck reports environment collision false, min clearance `0.075289 m`, `scene_collision=0`, `cspace=0`, joint-limit PASS, velocity/acceleration/jerk finite PASS. Route A was not modified.

## 2026-08-20 02:45 +08:00 -- Route B right-arm trajectory visualization bundle

- Purpose: wire GPT-provided `trajectory_visualizer` to the validated Route B right-arm-only trajectory without modifying viewer/bundle core or planning logic.
- Conda environment: `curobo_v2`.
- Working directory: `/home/lin/Projects/DexGraspNet2_Wuji2`
- Commands:

  ```bash
  conda run -n curobo_v2 python -m py_compile 08_dual_arm_scene_layout/isaaclab_control/curobo_motion_planning_routeB/build_visualization_right_arm_bundle.py 08_dual_arm_scene_layout/isaaclab_control/curobo_motion_planning_routeB/trajectory_visualizer/*.py
  PYTHONPATH=08_dual_arm_scene_layout/isaaclab_control/curobo_motion_planning_routeB conda run -n curobo_v2 python -m unittest -v trajectory_visualizer.test_bundle
  conda run -n curobo_v2 python 08_dual_arm_scene_layout/isaaclab_control/curobo_motion_planning_routeB/build_visualization_right_arm_bundle.py --capture-dir 08_dual_arm_scene_layout/isaaclab_control/outputs/closed_loop_sessions/20260819_174407/cycle_001/capture
  timeout 30s bash -lc 'PYTHONPATH=08_dual_arm_scene_layout/isaaclab_control/curobo_motion_planning_routeB conda run -n curobo_v2 python -m trajectory_visualizer.viewer 08_dual_arm_scene_layout/isaaclab_control/outputs/closed_loop_sessions/20260819_174407/cycle_001/capture/curobo_test_result/visualization_right_arm_bundle.npz'
  git diff --check
  ```

- Exit code: py_compile/bundle unittest/bundle build/diff-check `0`; viewer command exited `124` due to intentional 30s timeout after entering matplotlib blocking window with no traceback.
- Key output: generated `visualization_right_arm_bundle.npz` and `visualization_right_arm_summary.json`. Bundle has `scene_points=30000`, `sphere_count=233`, `moving=95`, `static=138`, `ee_frame=arm_r_link_tf`, `frames=41`, global min clearance `0.075289 m` at frame `24`, sphere `114`, link `arm_r_link_6`.
- Conclusion: visualization bundle validates and viewer loads it without error. Manual GUI interactions such as slider/play-pause require user-side confirmation in the opened matplotlib window.

## 2026-08-20 04:15 +08:00 -- Route B full planning/execution integration and PLACE=0 audit

- Purpose: diagnose `candidate=789` `PLACE raw IK=0/945`, separate Route A/Route B command-line paths, and run Route B full planning plus one complete Isaac execution attempt.
- Conda environments: `isaaclab22_sim50` for persistent Isaac/orchestrator; `curobo_v2` for Route B MotionPlanner backend subprocesses.
- Working directory: `/home/lin/Projects/DexGraspNet2_Wuji2`
- Key commands:

  ```bash
  python -m py_compile 08_dual_arm_scene_layout/isaaclab_control/closed_loop/orchestrator.py 08_dual_arm_scene_layout/isaaclab_control/closed_loop/persistent_isaac/client.py 08_dual_arm_scene_layout/isaaclab_control/closed_loop/persistent_isaac/worker.py 08_dual_arm_scene_layout/isaaclab_control/closed_loop/planning/flexible_route_search.py 08_dual_arm_scene_layout/isaaclab_control/curobo_motion_planning_routeB/routeB_full_pipeline_v1/routeB_full_pipeline/*.py
  PYTHONPATH=08_dual_arm_scene_layout/isaaclab_control/curobo_motion_planning_routeB/routeB_full_pipeline_v1:08_dual_arm_scene_layout/isaaclab_control/curobo_motion_planning_routeB/routeB_front_half_v1:08_dual_arm_scene_layout/isaaclab_control/closed_loop conda run --no-capture-output -n curobo_v2 python -m unittest -v routeB_full_pipeline.test_attachment_proxy routeB_full_pipeline.test_robot_config routeB_front_half.test_reach_contract routeB_front_half.test_goal_pool_io 08_dual_arm_scene_layout/isaaclab_control/closed_loop/tests/test_flexible_planning.py 08_dual_arm_scene_layout/isaaclab_control/closed_loop/tests/test_closed_loop_logic.py
  ./run_closed_loop.sh --motion-route curobo --planning-only --scene-folder /home/lin/Projects/DexGraspNet2_Wuji2/02_training_dataset/data/scene_datasets/wuji2_train60_100seminal_256view_force_adjusted_legacy_v1/scenes/scene_0072
  ./run_closed_loop.sh --motion-route curobo --sim-execute --scene-folder /home/lin/Projects/DexGraspNet2_Wuji2/02_training_dataset/data/scene_datasets/wuji2_train60_100seminal_256view_force_adjusted_legacy_v1/scenes/scene_0072
  git diff --check
  ```

- Key output:
  - `candidate=789` PLACE failure is not a Route B-specific bug: Route B and Route A use the same PLACE target contract and both get `PLACE raw IK=0/945` for that candidate. Other candidates in the same goal pool do have PLACE/RETREAT endpoint chains, so the shared placement-zone contract is not globally broken.
  - Config/production PlacementZone is synchronized: center `[0.27617723418526796, -0.1446419350251421, 0.46]`, size `[0.800000011920929, 0.30000001192092896, 0.0010000000474974513]`.
  - Route B now endpoint-preflights LIFT/TRANSFER/PLACE/RETREAT before spending true MotionPlanner time, skips candidates without a complete endpoint chain, and reuses the preflight chain pool for the selected candidate.
  - Route B full planning PASS on session `20260820_040547/cycle_001`, selected `candidate=676`, chain `0`; all seven dense 7DOF segments PASS with environment collision ON and self collision OFF.
  - Route B full Isaac execution completed the full route through `RETREAT_TO_HOME`. Task status is `FAIL` because the simulated grasp was empty: `verify_lift=0.3519 mm`, max lift `0.7849 mm`, final green zone `false`. This no longer aborts the motion route; it is recorded as `EMPTY_GRASP`.
- Exit code: final static checks/unit tests `0`; Route B planning-only `0`; Route B sim-execute returned `3` because task outcome was FAIL after full route completion.
- Cleanup: stale persistent Isaac workers were checked and stopped with SIGINT after planning/execution runs to release GPU memory.


## 2026-08-17 16:18:00 +0800 - final worktree cleanup
- Purpose: clean generated outputs/history in `/home/lin/Projects/DexGraspNet2_Wuji2`, preserve vendor submodule gitlinks, retain compact candidate5989 evidence.
- Conda env: base / no GPU workloads.
- Working directory: /home/lin/Projects/DexGraspNet2_Wuji2
- Command: guarded Python cleanup script on explicit cleanup paths; no DINO/SAM/DGN2/retarget/cuRobo/Isaac/training.
- Exit code: 0
- Key output: copied compact evidence, removed generated outputs/captures/archive/scratch, moved layout calibration to config, pruned intermediate checkpoints, updated docs and .gitignore.
- Conclusion: cleanup edits prepared for user review; no git add/commit/push performed.

## 2026-08-20 Route B full-planning stdout cleanup
- Checked no active Isaac/full_motion/orchestrator process after user stopped the run.
- Latest session inspected: `outputs/closed_loop_sessions/20260820_063033/cycle_001`.
- Finding: front-half candidate 7938 PASS; back-half endpoint pool exists; 16 tried chains generated `COVER_TO_LIFT` trajectory then failed at `LIFT_TO_TRANSFER` before any `traj_lift_to_transfer.npz`.
- Code-only logging/control-flow fix: Route B full backend now writes `routeB_full_plan_report.json(success=false)` for expected no-path cases and prints compact failure summary; orchestrator excludes the candidate using report stage counts instead of printing traceback tails.
- Validation: `python -m py_compile` on orchestrator/full backend/runtime PASS; `git diff --check` PASS.

## 2026-08-20 Route B attachment audit
- Confirmed and terminated stale Isaac worker PID 61449 via SIGINT; GPU compute apps were empty afterwards.
- Ran offline Route B full backend for session `20260820_063033/cycle_001`, candidate `7938`: attachment=True fails `LIFT_TO_TRANSFER` for 16/16 chains.
- Added diagnostic-only `--diagnostic-disable-transfer-attachment`; with attachment disabled only for `LIFT_TO_TRANSFER`, all seven Route B dense arm segments PASS offline without Isaac.
- Added and ran `test_attachment_audit.py`; at LIFT endpoint with attachment=True, the attached object spheres already collide with the no-target ESDF: 8/48 spheres colliding, min clearance `-0.036814 m`; with padding=0 still 6/48 colliding, min clearance `-0.031881 m`.
- Outputs: `routeB_full/attachment_audit/attachment_audit_chain00.json`, `.npz`, `.png`; padding0 outputs in `routeB_full/attachment_audit_padding0/`.
- Validation: py_compile PASS; git diff --check PASS.

## 2026-08-20 Route B transfer attachment default OFF
- Set `closed_loop/config/closed_loop.json` `routeB_full_pipeline.transfer_attachment=false`.
- Runtime now passes `--disable-transfer-attachment` to Route B full backend when this config is false.
- Startup Route B policy print now lists active OFF/FALSE items with Chinese explanations, including `RouteB transfer attachment: OFF（LIFT到TRANSFER不挂载目标proxy）`.
- Offline verification on session `20260820_063033/cycle_001`, candidate `7938`: full Route B backend PASSed 7/7 segments with `transfer_attachment=false`, environment collision ON, self collision OFF, and no Isaac launched.
- Validation: JSON parse PASS; py_compile PASS; git diff --check PASS.

## 2026-08-20 Route B concise stdout and recoverable execution failure
- Removed long raw JSON stdout from Route B front-half dense backend and full motion backend; reports remain written to JSON files.
- Route B execution FAIL in orchestrator now prints concrete stage/reason/lift/green-zone/report.
- EMPTY_GRASP and FINAL_GREEN_ZONE are treated as completed physical attempts: no placement commit, continue to next capture/query instead of returning process error.
- Latest execution report inspected: session `20260820_073606/cycle_001` failed `EMPTY_GRASP`, `verify_lift_mm≈0.00006`, `max_object_lift_mm≈6.88`, threshold `30.0`; COVER/GRASP refinement errors were not the blocker.
- Validation: py_compile PASS; git diff --check PASS.

## 2026-08-20 07:54 +0800 - Runtime task-object friction set to 0.3
- Purpose: user requested lowering runtime table/object contact material friction from 1.0 to 0.3 for subsequent Isaac attempts.
- Scope: changed only `closed_loop/persistent_isaac/worker.py` runtime `RigidBodyMaterialCfg`; did not modify URDF/USD, robot dynamics, joint limits, IK, Route A/B planning, or hand q.
- Commands:

  ```bash
  python -m py_compile 08_dual_arm_scene_layout/isaaclab_control/closed_loop/persistent_isaac/worker.py
  git diff --check
  ```

- Exit code: 0.
- Key output: static and dynamic friction now both `0.3`; restitution remains `0.0`.

## 2026-08-20 08:00 +0800 - Scene-folder Trimesh view commands added
- Purpose: add a direct offline Trimesh scene-folder viewer and put a copy-paste view command after each scene path in `TRAIN_SCENE_INPUT_PATHS_AND_OBJECTS.txt`.
- Scope: added `trimesh/view_scene_folder.py`; updated the scene address document only. No Isaac, DGN2, cuRobo, retarget, or physical simulation was run.
- Commands:

  ```bash
  python -m py_compile trimesh/view_scene_folder.py
  git diff --check
  conda run -n graspnet2.0 python trimesh/view_scene_folder.py --scene-folder /home/lin/Projects/DexGraspNet2_Wuji2/02_training_dataset/data/scene_datasets/wuji2_train60_100seminal_256view_v1/scenes/scene_0000 --export /tmp/scene_0000_view_test.glb
  ```

- Exit code: 0.
- Key output: `scene_0000` exported successfully to `/tmp/scene_0000_view_test.glb`; each scene entry now has a `View command:` line using `trimesh/view_scene_folder.py --scene-folder ... --show`.

## 2026-08-20 - task/route CLI foundation + color-sort HSV plumbing
- Purpose: begin two-axis architecture: task=`semantic-grasp|color-sort`, motion_route=`legacy|curobo`; keep Route A intact and reuse current Route B planner.
- Commands:

  ```bash
  git status --short
  git diff --check
  /home/lin/miniconda3/envs/graspnet2.0/bin/python - <<'PY'
  for m in ['numpy','cv2','PIL']:
      __import__(m)
      print(m, 'OK')
  PY
  python -m py_compile 08_dual_arm_scene_layout/isaaclab_control/closed_loop/orchestrator.py 08_dual_arm_scene_layout/isaaclab_control/closed_loop/persistent_isaac/client.py 08_dual_arm_scene_layout/isaaclab_control/closed_loop/persistent_isaac/worker.py 08_dual_arm_scene_layout/isaaclab_control/closed_loop/color_sort/segmentation.py 08_dual_arm_scene_layout/isaaclab_control/curobo_motion_planning_routeB/routeB_full_pipeline_v1/routeB_full_pipeline/backhalf_pool.py
  /home/lin/miniconda3/envs/graspnet2.0/bin/python 08_dual_arm_scene_layout/isaaclab_control/closed_loop/color_sort/segmentation.py --help
  /home/lin/miniconda3/envs/graspnet2.0/bin/python 08_dual_arm_scene_layout/isaaclab_control/closed_loop/tests/test_closed_loop_logic.py
  /home/lin/miniconda3/envs/graspnet2.0/bin/python 08_dual_arm_scene_layout/isaaclab_control/closed_loop/tests/test_flexible_planning.py
  PYTHONPATH=08_dual_arm_scene_layout/isaaclab_control/curobo_motion_planning_routeB/routeB_front_half_v1 /home/lin/miniconda3/envs/graspnet2.0/bin/python 08_dual_arm_scene_layout/isaaclab_control/curobo_motion_planning_routeB/routeB_front_half_v1/routeB_front_half/test_goal_pool_io.py
  PYTHONPATH=08_dual_arm_scene_layout/isaaclab_control/curobo_motion_planning_routeB/routeB_full_pipeline_v1 /home/lin/miniconda3/envs/graspnet2.0/bin/python 08_dual_arm_scene_layout/isaaclab_control/curobo_motion_planning_routeB/routeB_full_pipeline_v1/routeB_full_pipeline/test_attachment_proxy.py
  PYTHONPATH=08_dual_arm_scene_layout/isaaclab_control/curobo_motion_planning_routeB/routeB_full_pipeline_v1 /home/lin/miniconda3/envs/graspnet2.0/bin/python 08_dual_arm_scene_layout/isaaclab_control/curobo_motion_planning_routeB/routeB_full_pipeline_v1/routeB_full_pipeline/test_robot_config.py
  ./run_closed_loop.sh --help
  git diff --check
  ```

- Exit code: all listed static checks and lightweight tests passed.
- Key output:
  - Added `--task semantic-grasp|color-sort`, `--query`, and `--color-seed` CLI plumbing.
  - TTY startup now asks task first, then motion route; non-TTY remains semantic-grasp + legacy unless explicitly set.
  - Semantic-grasp maps common Chinese aliases (铅笔/瓶子/杯子/狗 etc.) to English prompts before GroundingDINO.
  - Added runtime color assignment for color-sort in Persistent Isaac worker; it binds red/blue PreviewSurface materials to runtime visual prims only, without changing mesh/collision/mass/friction/URDF/USD.
  - Added `closed_loop/color_sort/segmentation.py` and `closed_loop/config/color_sort.json`; HSV segmentation runs in `graspnet2.0` via `network_python`, produces per-instance masks, overlay, `detection_report.json`, and `selected_target.json`.
  - Added dynamic red_zone/blue_zone split from the current calibrated PlacementZone and Route B endpoint preflight uses the selected color zone by passing a layout override into the existing A placement helper.
  - Added current-scene failed HSV instance exclusion to avoid selecting the same failed component repeatedly.
  - RouteB front/full subprocess runtime no longer prints huge raw JSON lines to stdout; artifacts remain written to JSON.
- No Isaac app was launched in this phase.

## 2026-08-20 - color-sort RouteB DGN2 slug/mask path fix
- Purpose: fix first real color-sort planning-only run where selected HSV instance `red_001` produced DGN2 under `capture/dgn2/red_001`, but RouteB LEAP reach backend looked under `capture/dgn2/red` because it received color name as query.
- Commands:

  ```bash
  pgrep -af '[p]ersistent_isaac/worker.py|[i]saaclab.sh|[S]imulationApp|[i]saac-sim|[k]it/kit' || true
  python -m py_compile 08_dual_arm_scene_layout/isaaclab_control/closed_loop/orchestrator.py 08_dual_arm_scene_layout/isaaclab_control/curobo_motion_planning_routeB/routeB_full_pipeline_v1/routeB_full_pipeline/full_motion_backend.py 08_dual_arm_scene_layout/isaaclab_control/curobo_motion_planning_routeB/routeB_full_pipeline_v1/routeB_full_pipeline/runtime.py 08_dual_arm_scene_layout/isaaclab_control/curobo_motion_planning_routeB/routeB_full_pipeline_v1/routeB_full_pipeline/attachment_proxy.py
  /home/lin/miniconda3/envs/graspnet2.0/bin/python 08_dual_arm_scene_layout/isaaclab_control/closed_loop/tests/test_closed_loop_logic.py
  /home/lin/miniconda3/envs/graspnet2.0/bin/python 08_dual_arm_scene_layout/isaaclab_control/closed_loop/tests/test_flexible_planning.py
  git diff --check
  ```

- Exit code: 0.
- Key output:
  - No persistent Isaac/Kit process was running after the failed user planning-only attempt.
  - Orchestrator now passes `target_slug` (e.g. `red_001`) to RFS/RouteB LEAP reach runtime so it locates the correct DGN2 prediction folder.
  - RouteB full backend now accepts `--target-mask-path` and uses the selected HSV instance mask for target ESDF removal and attachment-proxy fallback. Semantic GroundedSAM path remains backward compatible.
  - LEAP reach fallback stdout is compacted to avoid dumping traceback tails into the terminal.

## 2026-08-20 - Source/Placement zone and scene alignment audit
- Purpose: respond to observed Isaac object/table penetration and Trimesh-vs-Isaac scene mismatch.
- Commands:

  ```bash
  /home/lin/miniconda3/envs/graspnet2.0/bin/python - <<'PY'
  import json, numpy as np
  from pathlib import Path
  layout=json.loads(Path('08_dual_arm_scene_layout/config/manual_layout_calibrated.json').read_text())
  print(layout['geometry'].get('zone_collision'))
  print(layout['transforms']['table']['position_world_m'], layout['geometry']['table_size_m'])
  print(layout['transforms']['source_zone']['position_world_m'], layout['geometry']['source_zone_size_m'])
  print(layout['transforms']['placement_zone']['position_world_m'], layout['geometry']['placement_zone_size_m'])
  PY
  /home/lin/miniconda3/envs/graspnet2.0/bin/python trimesh/view_scene_folder.py --scene-folder 02_training_dataset/data/scene_datasets/wuji2_train60_100seminal_256view_force_adjusted_legacy_v1/scenes/scene_0065 --calibrated-layout --export /tmp/scene_0065_calibrated.glb
  python -m py_compile 08_dual_arm_scene_layout/isaaclab_control/closed_loop/persistent_isaac/worker.py trimesh/view_scene_folder.py
  git diff --check
  ```

- Exit code: 0.
- Key output:
  - `manual_layout_calibrated.json` has `zone_collision=false`; SourceZone/PlacementZone are visual/planning regions, not physical support surfaces.
  - Isaac worker embeds scene_manifest object poses via `world_from_source @ T_world_centered_object`; the old Trimesh viewer displayed only the local training tabletop frame, so it did not match calibrated Isaac world layout.
  - Added `trimesh/view_scene_folder.py --calibrated-layout` to show the scene embedded into calibrated SourceZone and draw blue SourceZone / green PlacementZone markers.
  - Fixed object physics audit so invalid USD/PhysX AABB sentinel values no longer produce PASS; they now report WARN when CollisionAPI prims exist but no valid AABB can be computed.

## 2026-08-20 — scene_0065 migration-contract audit

- Purpose: prove the training-scene -> SourceZone -> Persistent USD bridge before any physics step, then compare the training default object-object policy against the persistent filtered policy.
- Commands:

  ```bash
  python -m py_compile \
    08_dual_arm_scene_layout/isaaclab_control/closed_loop/persistent_isaac/worker.py \
    08_dual_arm_scene_layout/isaaclab_control/closed_loop/persistent_isaac/client.py \
    08_dual_arm_scene_layout/isaaclab_control/closed_loop/tools/audit_scene_migration_contract.py
  /home/lin/miniconda3/envs/graspnet2.0/bin/python \
    08_dual_arm_scene_layout/isaaclab_control/closed_loop/tools/audit_scene_migration_contract.py \
    --scene-manifest 02_training_dataset/data/scene_datasets/wuji2_train60_100seminal_256view_force_adjusted_legacy_v1/scenes/scene_0065/scene_manifest.json \
    --capture-dir 08_dual_arm_scene_layout/isaaclab_control/outputs/scene_migration_audits/20260820_scene0065_pose_fix_headless/capture
  git diff --check
  ```

- Exit code: 0 for the corrected headless migration captures and static checks.
- Key result: old worker loaded a wrong test-split editable USD fallback and, independently, `set_reference_transform()` lost the manifest rotation because `Gf.Matrix4d.SetTranslate()` resets the matrix.  The corrected worker resolves the explicit training lineage asset and uses `SetTranslateOnly()` after setting rotation.
- Corrected zero-step result: max expected-vs-rigid position error `0.0126 mm`; max rotation error `0.0303 deg`; max pairwise geometry error `0.0118 mm`; all collision AABBs valid and within `1.1 mm` of table contact.
- Settle P1 (training object-object collision ON) and P2 (persistent object-object filtering OFF): both had maximum 1.0 s root drift `0.0017 mm` for scene_0065.  Therefore the previous 18--85 mm errors were migration-frame failures, not normal settling or collision-policy behavior.
- Artifacts: `outputs/scene_migration_audits/20260820_scene0065_pose_fix_headless/capture/` and `outputs/scene_migration_audits/20260820_scene0065_training_collision_on/capture/`.

## 2026-08-20 — unified RobotSegmenter mask consumers

- Scope was perception-consumer wiring only; scene migration, asset lineage, DGN2/LEAP/Wuji2, IK, and Route B collision/planner contracts were not changed.
- One current-capture RobotSegmenter run now writes immutable-input derivatives under `capture/planning/`: `robot_mask.npy/png`, `rgb_no_robot.png`, `robot_mask_overlay.png`, and `filtered_depth.npy`. Raw `capture/rgb.png` and `capture/depth_m.npy` remain untouched.
- Semantic flow now uses `rgb_no_robot` for GroundingDINO, a RobotSegmenter box hard gate before SAM, a second SAM-mask hard gate, then valid-depth + rigid SourceZone acceptance. `NO_LEGAL_TARGET` is structured and never selects the robot.
- HSV flow now intersects red/blue masks with `~robot_mask` before morphology/components and records `robot_mask_capture_dir` plus `stale_mask_check`.
- Commands: `PYTHONPATH=08_dual_arm_scene_layout/isaaclab_control/closed_loop/scripts /home/lin/miniconda3/envs/graspnet2.0/bin/python -m unittest 08_dual_arm_scene_layout/isaaclab_control/closed_loop/scripts/test_semantic_target_rejects_robot.py -v`; current-capture HSV replay; `git diff --check`.
- Results: seven safety regressions PASS. HSV replay on `20260820_091931/cycle_001` PASS (`red source=0`, `blue source=2`, selected `blue_001`) with current-capture robot-mask validation. Its old selected bottle DINO box had 0.6308 robot-box overlap and is now rejected pre-SAM. The sandbox cannot expose CUDA for a new RobotSegmenter/GroundedSAM replay; no Isaac was launched and no raw capture was modified.

## 2026-08-20 — pre-experiment pipeline preflight

- No planner, grasp, physics, scene-migration, or threshold parameters changed in this review.
- Static checks: `py_compile` for the perception/orchestrator/persistent-Isaac/Route-B-runtime files; seven semantic safety regressions PASS; stale capture mismatch raises `STALE_ROBOT_MASK`; HSV replay on `20260820_091931/cycle_001` has zero robot pixels in both final red/blue masks; all four task/route dispatch pairs preserve their explicit values; `git diff --check` PASS.
- Historical replay evidence: old bottle DINO boxes 0, 1, and 3 have RobotSegmenter overlap 0.5645, 0.6308, and 0.8489 respectively and now evaluate to `REJECT_ROBOT_OVERLAP`.
- Real GPU smoke (headless, no arm motion): started one Persistent Isaac worker with scene_0065, migration pre-physics guard PASS (`pairwise=0.0118 mm`, visual scale unused), captured, then cleanly shut down.  cuRobo RobotSegmenter produced 225801 robot pixels and all five expected artifacts. GroundingDINO/SAM ran on `capture/planning/rgb_no_robot.png` for query `pencil`: two legal SourceZone proposals, selected index 0, 5286 valid target points, `stale_mask_check=PASS`. No IK, MotionPlanner, GRASP, or execution ran.
- Smoke artifacts: `outputs/pre_experiment_prefight/20260820_100016_semantic_curobo_perception_smoke/capture/`. Final host check: no Persistent Isaac/Kit process; GPU 16 MiB used / 7791 MiB free.

## 2026-08-20 — semantic target robot-exclusion regression repair

- Facts-first audit found exactly one robot-mask source: `perception/robot_segmentation/curobo_robot_segmenter.py:RobotDepthCleaner.remove_robot`, producing `capture/planning/robot_mask.npy`.  It was previously consumed only by Route B planning depth/ESDF, not semantic final target selection.
- First regression difference: `closed_loop/scripts/grounded_sam_backend.py` called `backend.choose_detection(boxes, scores)` before SAM and never evaluated robot overlap or SourceZone membership.  The historical implementation did not contain a separate semantic robot-exclusion gate; the missing behavior was a wiring gap, not a threshold regression.
- Current screenshot replay: session `20260820_091607`, query `cup`, prior selected DINO idx 0 had SAM robot overlap `0.999581` and SourceZone overlap `0.0`; it is now `REJECT_ROBOT_OVERLAP`.  idx 1 (`0.781461` robot overlap) is also rejected; legal idx 2 is selected after gates.
- Minimal repair: reuse the existing current-capture RobotSegmenter mask for semantic proposals; SAM evaluates every DINO box, hard-rejects dominant robot masks, removes non-dominant robot pixels, then requires valid residual geometry in the rigid SourceZone.  `STALE_ROBOT_MASK` fails if robot report capture_dir differs from RGB capture.
- Validation: six pure regression cases PASS (`robot only`, `real target vs robot`, `partial occlusion`, `outside SourceZone`, `stale mask`, shape mismatch); offline DINO+SAM replay PASS; `py_compile` and `git diff --check` PASS.  No Isaac was started for replay.

## 2026-08-20 — pre-experiment mainline cleanup

- Purpose: remove superseded generated output and caches without changing frozen production behavior.
- Commands: exact validated cleanup lists only (`xargs -r -d '\n' rm -rf < /tmp/cleanup_sessions_exact.txt`; `xargs -r -d '\n' rm -rf < /tmp/isaaclab_cache_inventory.txt`), followed by `py_compile`, semantic target safety unittest, HSV replay, scene migration read-only audit, Route B static tests, CLI dispatch smoke, and `git diff --check`.
- Results: deleted exactly 42 superseded `closed_loop_sessions` directories and one superseded migration audit; retained `20260820_063033`, `20260820_073606`, `20260820_091607`, `20260820_091931`, both formal migration audits, and the latest preflight smoke.  Moved two unreferenced legacy files to `/home/lin/DexGraspNet2_Wuji2_cleanup_archive_20260820/`; production references: none.
- Validation: `py_compile` PASS; semantic target safety 7/7 PASS; HSV replay PASS with zero robot pixels in final red/blue masks; migration audit PASS (`pairwise=0.011837 mm`, final drift `0.001711 mm`); Route B import/reach/config static tests PASS; four CLI task/route combinations PASS.  No Isaac, planner, or grasp execution was launched.

## 2026-08-20 — scene_0020 pre-physics guard false positive

- User formal experiment stopped before CAPTURE/perception at `SCENE_MIGRATION_GEOMETRY_FAIL`; artifact: `outputs/closed_loop_sessions/20260820_101901/cycle_001/capture/scene_migration_spawn_audit.json`.
- Evidence: all scene_0020 manifest expected poses exactly matched the USD `actual_rigid_body_world_pose` (`0.000000 mm / 0.000000 deg`, including Pencil). The only failed value was the post-`SimulationContext.reset()` runtime cache for Pencil: `0.024183 mm / 0.103546 deg`.
- Root cause: `_pre_physics_migration_audit()` incorrectly used `RigidObject.data.root_pose_w` as the zero-step migration truth. That runtime float representation is useful for physical-state diagnosis but is not the authored USD rigid-root frame being audited.
- Minimal repair: the live guard and offline migration audit now use `actual_rigid_body_world_pose` for pose and pairwise geometry acceptance. Runtime pose deltas remain serialized as `runtime_pose_diagnostic_error_*`; settling drift remains referenced to the runtime initial state.
- Validation: `py_compile` PASS and an offline assertion on the failed artifact confirms `PREPHYSICS_USD_GUARD_PASS` while preserving the runtime `0.103546 deg` diagnostic. No Route B, grasp, collision, scene mapping, or physics parameter changed.

## 2026-08-20 — Route B candidate retry exception-boundary repair

- User session `20260820_102515/cycle_002`: candidate `2898` correctly failed all 16 full-route chains at `LIFT_TO_TRANSFER` and was excluded. Candidate `1884` then had a structured front-half no-path (`NO_PREGRASP_GOAL_WITH_VALID_CUROBO_TRAJECTORY`, exit 3, report present), but the process stopped instead of trying the remaining goals.
- Root cause: `routeB_front_half/runtime.py:run_routeB_dense_backend()` raised on every nonzero backend exit even when the backend had written an expected candidate-level `success=false` report. Additionally, orchestrator checked success-only trajectory artifacts before its existing failure/retry branch.
- Repair: structured `success=false` front-half reports now return to orchestrator; orchestrator handles that report first, logs candidate/rank/reason/report, excludes the case, then retries. Missing trajectory artifacts remain fatal only for a claimed-success report.
- Added `routeB_front_half/test_runtime_candidate_failure.py` regression. Validation: py_compile PASS; new retry regression PASS; existing reach-contract 2/2 PASS; `git diff --check` PASS. No Isaac, trajectory planning, or grasp run during this repair.

## 2026-08-20 — color-query GroundingDINO mainline

- User requirement: color-sort is a task parallel to semantic-grasp. The user enters `red`/`blue` (also `红`/`蓝` interactively); the system must visually find and move every current SourceZone object of that requested color, without choosing targets from the seeded runtime color assignment.
- Command shape: `./run_closed_loop.sh --task color-sort --target-color red --motion-route curobo --color-seed 42 --sim-execute`. In interactive mode the color prompt appears after task/seed/route selection.
- Selection contract: raw RGB -> RobotSegmenter `rgb_no_robot` -> GroundingDINO prompt `"red object"` or `"blue object"` -> existing SAM robot/SourceZone safety gates -> intersection with the best matching *current RGB* HSV instance. The final DGN2/ESDF target mask is this one-instance intersection, not the broad SAM mask and not a `sort_color` GT lookup.
- Color HSV remains used only for current-image color instance validation, RobotSegmenter exclusion before morphology, and failure-instance bookkeeping. `color_assignment.json` remains runtime material/audit metadata only.
- Continuity: candidate/endpoint/full Route B retries still explore alternate grasp/endpoint poses first. If that funnel is exhausted for a color instance, the robot has not moved (planning is frozen at HOME), the instance is recorded and skipped, then the next cycle takes a fresh capture and tries another same-color instance. If Route B execution returns `HOME` after a failure, that instance is likewise skipped and the session continues. Route-B executor recovery reverses only already executed cuRobo dense samples (no arm IK, quintic replacement, or replanning) and verifies HOME.
- Validation commands: `isaaclab22_sim50 python -m py_compile` for orchestrator/color segmentation/Route-B executor; semantic safety unittest `7/7 PASS`; HSV preferred-color + robot-mask replay PASS; GroundingDINO/SAM offline `red object` replay on retained scene_0020 color capture PASS. The visual result had red overlap only (best `red_003`, 7784 pixels) and zero blue overlap. `git diff --check` PASS.

## 2026-08-20 — repository cleanup and standardized color campaign

- Read and classified tracked source/docs plus untracked root packages before
  deletion. Protected `02_training_dataset/TRAIN_SCENE_INPUT_PATHS_AND_OBJECTS.txt`,
  all dataset/model assets, official URDF/USD, production paths, tests, and audit tools.
- Moved six unreferenced legacy/duplicate trees to
  `/home/lin/DexGraspNet2_Wuji2_cleanup_archive_20260820/repository_cleanup/`.
- Removed only explicit generated targets: root diagnostic logs, Trimesh/inference
  output, sessions `20260820_101901` and `20260820_102515`, non-selected Route B
  scratch, superseded color planning scratch, and Python/pytest caches.
- Rewrote root/current-mainline READMEs and architecture documentation. No
  planner, perception, physics, grasp, threshold, or official asset behavior changed.
- Cleanup regression commands: `bash -n`; focused `py_compile`; semantic safety
  unittest (7/7); core unittest (17/17); closed-loop unittest (9/9); Route B
  unit/static tests (10 PASS); HSV replay with zero robot overlap; read-only scene
  migration audit; four-way CLI dispatch smoke; `git diff --check`.
- Added `closed_loop/tools/run_color_sort_campaign.py`. Dry-run PASS for a single
  timestamp root containing `scene_0000:RED`, `scene_0020:BLUE`, and
  `scene_0065:RED`. The tool invokes the unchanged production entry, streams and
  records terminal output, stops on a real error, and resumes passed cases.

## 2026-08-20 — perception-only color refactor v2 verification boundary

- Read v2 package at
  `/home/lin/Projects/_wuji2_refactor_packages/wuji2_perception_only_color_refactor_verification_v2_20260820/`.
- Installed supplied modules:
  `closed_loop/target_contract.py`,
  `closed_loop/verification_contract.py`,
  `closed_loop/color_sort/target_pool.py`,
  `closed_loop/planning/perception_target_geometry.py`,
  and `closed_loop/tools/audit_perception_only_planning.py`.
- Appended `sample_place_from_perception_anchor()` to
  `closed_loop/planning/flexible_pose_sampling.py`; existing place sampler was
  preserved.
- Changed color CLI timing so unspecified `--target-color` is asked after the
  first fresh RGB capture, not before Persistent Isaac starts.
- Changed color-sort selection from HSV-selected-single-instance to
  GroundedSAM legal proposals x HSV component target pool.  The final DGN2 mask
  is the full matched HSV instance; `SAM & HSV` is saved only for audit.
- Changed HSV SourceZone membership to use the current capture
  `settled_scene_manifest.json -> world_from_source_zone` rigid transform, not
  a layout world-axis AABB.
- Moved simulator target binding in orchestrator to after Route A/B full route
  PASS and before `execute()` / `execute_routeB()`.  The execution target id is
  wrapped by `SimulationVerificationBinding`.
- Refactored production candidate case construction to accept
  `--target-geometry` and to build a perception proxy from current mask/RGB-D,
  instead of resolving exact target segmentation id / object mesh before
  planning.
- Changed Route B attachment proxy production path to use current
  perception mask + depth only; exact Isaac collision AABB is no longer used for
  planning proxy generation.
- Updated Route B dense executor so planning manifest proxy id does not need to
  match the post-plan execution verification id.
- Replay command summary: rebuilt v2 target pools for
  `20260820_124401/cycle_001` and `cycle_002`.  Cycle 001 healthy blue match
  PASS.  Cycle 002 old 59 px pair rejected by overlap/fraction/Dice/depth gates;
  legal `blue_000` remains in the same capture target pool.
- Validation commands run: py_compile focused production files PASS; semantic
  robot safety 7/7 PASS; closed-loop logic 6/6 PASS; flexible planning 5/5
  PASS; Route B front-half reach/runtime 3/3 PASS; Route B full runtime,
  attachment proxy, robot config 4/4 PASS; right-arm contract 2/2 PASS;
  `check_v2_wiring.py` PASS; `git diff --check` PASS.

## 2026-08-20 — supervised three-scene color campaign and error repairs

- Campaign command: `/home/lin/miniconda3/bin/python 08_dual_arm_scene_layout/isaaclab_control/closed_loop/tools/run_color_sort_campaign.py`; canonical timestamp root:
  `outputs/color_sort_campaigns/20260820_111709/`.
- Cases: `scene_0000:RED`, `scene_0020:BLUE`, `scene_0065:RED`; seed 42,
  headless Persistent Isaac, production color-sort + Route B sim-execute entry.
- Error repair 1: a candidate-level `PREGRASP_TO_COVER` no-path formerly raised
  before writing a current full-plan report, allowing a stale prior candidate
  report to be read. The full backend now writes structured `success=false`, and
  runtime unlinks the previous report before each candidate. Candidate failure
  returns to the existing retry loop; worker/protocol failures remain fatal.
- Error repair 2: target-local DGN2 `Official sampler produced no seed` formerly
  ended the process. Color-sort now classifies only that exact exhausted-sampler
  condition as recoverable, keeps/validates HOME, records the failed instance,
  performs a fresh capture, and tries another instance. CUDA, checkpoint, input,
  or protocol errors remain fatal.
- Error repair 3: GroundedSAM already evaluated multiple legal DINO proposals,
  but color-sort persisted and matched only the highest proposal. It now saves
  all legal residual proposal masks in `legal_proposal_masks.npz` and matches
  each current unfailed HSV instance against all legal proposals. Semantic-grasp
  selection behavior is unchanged and runtime color assignment remains audit-only.
- Error repair 4: the campaign resume path now always rewrites a complete
  three-case summary, uses timestamped terminal logs, emits compact
  `campaign_report.md`, and indexes failure stages without dumping long backend
  JSON to the human report. `--rerun-case` permits one invalidated case to be
  repeated under the same campaign root.
- Final case status: all three processes exited normally with
  `PARTIAL_COMPLETE`; no arm execution was started because no case produced all
  seven Route B paths. `scene_0000` blockers were predominantly
  `COVER_TO_LIFT`; `scene_0020` had `PLACE targets=385 raw IK=0` for the selected
  blue instance plus one DINO/SAM-to-HSV no-match; `scene_0065` had
  `CURRENT_TO_PREGRASP` no-path, `PLACE raw IK=0`, and two small-instance DGN2
  no-seed results. These are preserved as truthful blockers; no placement zone,
  IK tolerance, collision threshold, physics, grasp, or planner parameter was
  changed to force PASS.
- Removed four superseded campaign sessions and two obsolete un-timestamped
  logs after repaired reruns; retained exactly one canonical latest session per
  case. The removed generated data totalled about 312 MiB.
- Final regression commands/results: Isaac-env focused `py_compile` PASS;
  semantic safety 7/7 PASS; closed-loop 11/11 PASS; core 17/17 PASS; Route B
  10/10 PASS in `curobo_v2`; GroundedSAM embedded subprocess compile PASS; HSV
  replay stale-mask PASS with red/blue robot overlap both zero; scene migration
  audit PASS; `git diff --check` PASS.
