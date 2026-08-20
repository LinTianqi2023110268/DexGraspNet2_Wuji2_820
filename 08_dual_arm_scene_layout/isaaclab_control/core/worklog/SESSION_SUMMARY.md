# Route-C V2 session summary

## Current phase

Closed-loop one-command integration in progress; physical execution remains locked.

## Completed

- Confirmed the project root is `/home/lin/Projects/DexGraspNet2_Wuji2`.
- Captured the initial Git status and diff; tracked diff is empty.
- Read `CODEX_INSTRUCTION.md`, `INTEGRATION_GUIDE.md`, and `README.md` completely.
- Read both available Isaac Lab / Isaac Sim skill instruction files completely.
- Added the minimal repository-level `AGENTS.md` requested for durable rules.
- Completed the `curobo_v2` probe; all required cuRobo V2 imports succeed.
- Formal host-channel environment probe sees the RTX 4070 and CUDA successfully.
- Core pure tests pass: 15/15 after adding runtime-math, collision-payload, and worker-environment regressions.
- Persistent worker ping passes from `isaaclab22_sim50` to `curobo_v2` with exact right-arm joint order.
- candidate3800 exact five-stage GPU IK passes with 31–34 accepted solutions per target and continuous chained selection.
- Existing generated robot YAML loads on GPU and returns 233 spheres for all 35 active joints.
- Pre-integration Top-20 five-stage coarse GPU regression passes 8/20; candidate3800 and candidate34 both pass.
- Formal RGB-D + GroundedSAM mask builds separate scene/target ESDF layers after one minimal cuRobo 0.8.x RGB-placeholder compatibility fix.
- Worker now supports per-solution phase-aware observed-collision hard filtering before q-reference continuity selection and preserves the audited solution sets.
- candidate3800 collision-aware exact solve passes observed collision at all five stages, but every selected state has single-view unknown exposure.
- Formal Isaac Lab dry-run passes the complete q_current -> worker -> GPU IK -> ESDF planning chain for nine stages.
- Archived the former runtime SciPy IK and old reachability/complete-mesh collision tool cohort under history.
- Post-integration Top-20 regression is unchanged at 8/20 PASS; candidate3800 and candidate34 remain PASS.
- Final `git diff --check` and Python compile/import audit pass; no vendor, DGN2, URDF, or USD file was modified.
- Began the closed-loop integration pass after explicit user approval.
- Re-read `AGENTS.md`, the closed-loop task instruction, and the Isaac Lab control skill body.
- Audited the copied `closed_loop/` patch and confirmed the main missing interfaces: live Grounded-SAM backend, capture-time measured `robot_state.json`, cuRobo self-collision, and continuous path gate.
- Found the local isolated Grounded-SAM project at `/home/lin/Projects/分类抓取开源项/03_检测加分割_GroundedSAM`; its environment and DINO/SAM weights pass the bundled audit.
- Stopped the slow per-candidate full pipeline after user correction; confirmed the architecture issue in source: per candidate retarget/materialization, per candidate `CuroboWorkerClient`, and per candidate ESDF rebuild.
- Added known-good candidate3800 five-stage regression against the current worker; PREGRASP/COVER/GRASP/SQUEEZE/LIFT raw success is restored and positive.
- Added grouped batched IK worker/client support so one request can solve multiple independent candidate waypoint groups while preserving q_current-based branch continuity per group.
- Added `batch_pick_candidate_gate.py` as the chunked pick-stage screening primitive: one worker, one map build, DGN2 score order, `chunk_size * 5` poses per grouped GPU request.
- Added explicit worker self-collision and joint-path regression hooks and validated positive/negative semantics.
- Added scratch case-root support for reviewed LEAP→Wuji2 scripts via `DGN2_CASE_ROOT`, so rejected candidates do not need permanent `01_cases/active` directories.
- Fixed pick-stage path screening so it checks only `q_current→PREGRASP→COVER→GRASP→SQUEEZE→LIFT`; it no longer appends `LIFT→HOME` during pick screening.
- Added cycle-scoped `screen_pick_batches.py`: one cuRobo worker, one map build, lazy chunk materialization, and one grouped solve per chunk.
- Replaced the closed-loop orchestrator's old per-candidate gate loop with one call to the cycle-scoped lazy batch screener.
- Ran same-frame Top-16 dog validation on `closed_loop_sessions/20260816_190311/cycle_001`.
- Diagnosed candidate1422 collision filtering in detail; every threshold-accepted pick-stage IK solution is rejected by self-collision, and GRASP/SQUEEZE are additionally rejected by target ESDF.
- Implemented and benchmarked all-candidate GPU coarse prefilter using DGN2 GRASP root poses only, without LEAP→Wuji2 retargeting or scratch cases.
- Switched closed-loop planning-only feasibility to `SELF_COLLISION_POLICY=REPORT_ONLY_UNRESOLVED`; self-collision is still computed/reported but no longer a hard planning rejection.
- Added low-cost approximate PREGRASP/approach-path coarse prefilter code, but its GPU benchmark could not be completed in this turn because the available execution channel could not expose CUDA and escalation was rejected.
- Integrated the one-command planning-only orchestrator so `./run_closed_loop.sh --planning-only` uses existing modules for capture, GroundedSAM, 40k RGB-D input, DGN2, all-candidate GPU coarse prefilter, survivor-only LEAP→Wuji2 retargeting, exact pick gate, placement allocation, full-route gate, and HOME planning.
- The integrated orchestrator keeps one `CuroboWorkerClient` and one Mapper/TSDF/ESDF build per closed-loop cycle and no longer calls the CLI scripts that would restart worker/map inside the cycle.
- Removed the obsolete physical-execution branch from the planning-only orchestrator; `execution_enabled` remains false and no physical motion is launched.
- Fixed final-planning scratch case path construction so `case_root.name == case_id` for `build_candidate_case.py`.
- Reordered the all-candidate coarse prefilter into a strict cheap-to-expensive funnel: GRASP IK/threshold/scene, PREGRASP IK/threshold/scene only for GRASP scene survivors, then q_current→PREGRASP and PREGRASP→GRASP continuous observed-scene path only for previous-stage survivors.
- Restored the existing multi-cycle execution wiring for explicit Isaac Sim diagnostic execution: selected full-route candidate -> existing runtime launcher -> existing full pick/place/HOME runtime -> existing replay -> existing next-scene-manifest builder -> continue the original `while True` loop.
- Added diagnostic flags `--sim-execute --no-planner-collision-check --diagnostic-ignore-static-gate`; planner collision checks are skipped without disabling Isaac/PhysX collision/contact.
- Added a session-local placement policy/registry under each closed-loop session so diagnostic cycles do not consume or depend on the global placement registry.
- Replaced per-survivor retarget/exact-IK with score-ordered batch retarget chunks and grouped exact IK. Each chunk uses one build-case process, one persistent retarget process for 01/02/03, one finalize/flange process, then one `solve_ik_groups` request for `N*5` exact pick poses.
- Retarget chunk size is now 64. Runtime launcher is invoked as `bash run_full_pick_place_closed_loop.sh ...`, so executable permission on the launcher file is no longer required.

## Current conclusion

- `core/` and several diagnostic/result paths are currently untracked user work and must be preserved.
- The `isaaclab22-manipulator-control` skill's referenced `reference.md` and `evaluations.md` files are absent; work continues from the complete skill body and repository source.
- `rg` is unavailable, so repository audits use `find` and `grep`.
- Installed versions are API-compatible. CUDA is unavailable only in the restricted execution channel; the host channel sees the RTX 4070 correctly.
- Production runtime no longer imports SciPy/Pinocchio IK or consumes the old path-collision report as a gate.
- `closed_loop/config/closed_loop.json` currently has `execution_enabled=false`; this must remain false until a fresh 10 s static gate passes.
- The current cuRobo worker is not raw-success broken on the known-good case: old raw `[32,36,36,36,31]`, current raw `[33,37,37,37,31]`.
- The new batched primitive is implemented and validated at the IK contract level. Same-frame observed-ESDF chunk validation still requires matching capture + mask + scratch candidate cases; mismatched archived candidate/capture data must not be used to claim observed collision PASS.
- Same-frame Top-16 validation used one worker and one map build as required. It found no pick-stage feasible candidate in the first 16 official-score candidates; rank14/candidate1422 had positive raw IK but all IK-accepted solutions were rejected by collision filtering.
- candidate1422 and the measured baseline both expose a tiny recurring self-collision penetration (`0.000251 m`), so current self-collision sphere semantics are likely over-rejecting and must not be treated as final physical truth.
- All-candidate GPU prefilter is fast: `7454` target proposals, batch size `512`, `15` batches, `4435` raw reachable, `4347` threshold accepted, `4269` scene-ESDF pass, but `0` self-collision pass and therefore `0` coarse survivors under the current self-collision gate.
- Under report-only self-collision policy, the already measured GRASP coarse survivors without self-collision are `4269`. PREGRASP/approach survivors are not yet measured because the GPU rerun was blocked by execution-channel CUDA visibility.
- One-command code is integrated and syntax-valid, but real end-to-end validation from this Codex command channel is blocked: Isaac/Kit and `curobo_v2` cannot see the NVIDIA driver/GPU here, while the user's own terminal shows the RTX 4070 via `nvidia-smi`.
- The first user-side one-command run reached GroundedSAM/DGN2/coarse prefilter and found `2748` coarse survivors before hitting the scratch path contract bug at candidate330; that path bug is fixed locally.
- Simulation diagnostic execution wiring is in place but intentionally not run from the restricted Codex channel.
- Batch retarget regression for candidate330 is numerically identical to the old scratch output; GPU exact IK timing/classification must be measured in the user's GPU-visible terminal.
- User-side batch result showed exact IK is no longer the bottleneck (`160` poses solved in `0.027 s` GPU time); retarget/finalize dominate.

## Current blockers

Physical/action smoke is blocked by the unchanged ft04 static stability thresholds: right arm and Wuji2 both FAIL. GPU-dependent commands require a GPU-visible host execution channel; the current Codex execution channel reports `nvidia-smi` failure and `torch.cuda.is_available() == False` in `curobo_v2`. Link-scoped intentional finger/target contact remains a limitation unless implemented later. Self-collision model semantics remain unresolved and are now report-only for planning.

## Next step

Run `./run_closed_loop.sh --planning-only` from a GPU-visible terminal using `scene_0000` and target text `dog`. The expected flow is capture → DINO/SAM → 40k → DGN2 → all-candidate prefilter → survivor retarget → exact pick gate → placement/full-route/HOME planning. Do not add, commit, push, upload, or run physical motion while the static gate is FAIL.

## Modified files

- New/updated core production implementation: `bridge/curobo_worker.py`, `bridge/worker_client.py`, `perception_collision/rgbd_mapper.py`, `runtime_math.py`, tests, generated robot YAML/asset aliases, READMEs, and worklogs.
- Runtime replacement: `runtime/scripts/10_run_full_pick_place.py`, runtime launchers, visible launcher, and runtime README.
- Environment-entry fixes: five active diagnostics launchers now activate `isaaclab22_sim50` and expose local Isaac Lab sources.
- Documentation: `isaaclab_control/README.md`, `tools/README.md`, `history/INDEX.md`, root `AGENTS.md`, and archive README.
- Archived intact: `runtime_rebase_ik.py` and tools `04`-`09`, `12`-`14` under `history/legacy_route_c_cpu_mesh/`.
- Generated evidence: post/pre-integration Top-20 reports, current ft04 report, and `route_c_v2_planning.json`.
- Preserved pre-existing untracked diagnostics/results/manifest files; they were not edited or deleted.

## Test results

- A. Environment probe: PASS on host channel (CUDA true, RTX 4070 Laptop GPU); restricted-channel false result retained only as execution-channel diagnosis.
- B. Core unit tests: PASS (15/15) in `curobo_v2`; the base-interpreter attempt failed only because base has no NumPy.
- C. Robot model: PASS; existing YAML loads on GPU (35 joints, 233 spheres).
- D. Persistent worker: PASS from `isaaclab22_sim50`.
- E. candidate3800 exact GPU IK: PASS; accepted counts `32/34/34/34/31`, solve time `1.989 s`.
- F. candidate3800 coarse five-stage: PASS.
- G. candidate34 coarse five-stage: PASS.
- H. Top-20 coarse five-stage: 8/20 PASS (Top-10 4/10); limitation: pure coarse IK only.
- I. Post-integration Top-20 regression: unchanged at 8/20 PASS; candidate3800 and candidate34 remain PASS.
- RGB-D Mapper: PASS; 407,537 valid pixels, separate scene/target ESDF grids.
- J. Isaac Lab dry-run: planning PASS after real settle; q_current read and nine continuous stages selected; headless Kit shutdown hang remains.
- K. Physical smoke: NOT RUN because the current 10-second static gate FAILS (right arm FAIL, Wuji2 FAIL, flange/wrist PASS).
- Final checks: `git diff --check` PASS; related Python compilation PASS; production-path grep found no active legacy IK/collision imports.
- Known-good 5-stage worker regression: PASS; current raw `[33,37,37,37,31]`, current accepted `[32,34,34,34,31]`, worker starts `1`, map builds `0`.
- Grouped batched IK regression: PASS; `group_count=1`, `pose_count=5`, raw `[33,37,37,37,31]`.
- Self-collision semantics: PASS; safe state `self_collision_pass=true`, folded all-zero right-arm state `self_collision_pass=false`.
- Continuous-path semantics: PASS; safe same-state path `path_pass=true`, deliberate path into folded collision state `path_pass=false`.
- Closed-loop logic tests: PASS `4/4` in `isaaclab22_sim50`.
- Top-16 same-frame batch pick validation: command PASS / screening result FAIL; worker starts `1`, map builds `1`, chunk size `16`, materialized candidates `16`, grouped solve poses `80`, selected candidate `null`.
- candidate1422 collision diagnosis: command PASS; per-stage raw/threshold/self/scene/target/survivors are `pregrasp 36/33/33/0/0/0`, `cover 37/35/35/0/0/0`, `grasp 37/35/35/0/35/0`, `squeeze 37/35/35/0/35/0`, `lift 34/33/33/0/0/0`.
- All-candidate GPU prefilter: command PASS; total proposals `8192`, target proposals `7454`, GPU batch size `512`, batch count `15`, raw reachable `4435`, threshold accepted `4347`, scene pass `4269`, self pass `0`, coarse survivors `0`, total wall `18.242 s`, peak VRAM `1608 MiB`.
- Self report-only patch: py_compile PASS; `git diff --check` PASS. GPU rerun for PREGRASP/approach report-only benchmark was blocked because this channel reported CUDA unavailable and escalation was rejected.
- One-command planning-only integration: `orchestrator.py` py_compile PASS; `git diff --check` PASS. Real run reached Isaac capture but was interrupted after this command channel reported NVML/CUDA/Vulkan GPU initialization failures; escalated rerun was rejected by execution policy.
- Scratch path + strict prefilter patch: `orchestrator.py` and `all_candidate_gpu_prefilter.py` py_compile PASS; `git diff --check` PASS. No GPU/Isaac rerun was attempted in the restricted channel.
- Simulation diagnostic wiring: `orchestrator.py` and `runtime/scripts/10_run_full_pick_place.py` py_compile PASS; `git diff --check` PASS.
- Batch retarget + grouped exact IK wiring: py_compile PASS; `git diff --check` PASS. Candidate330 retarget/flange numerical regression PASS with max abs diff `0.0`; only path metadata differs.
- Retarget chunk 64 + bash launcher patch: py_compile PASS; `git diff --check` PASS.

## Current environment versions

- Python `3.11.15`
- PyTorch `2.13.0+cu130`
- cuRobo `0.8.0.post1.dev42`
- CUDA available: `True` on the host execution channel (`False` only in the restricted sandbox channel)
- Isaac runtime conda: Python `3.11.15`, NumPy `1.26.4`, Gymnasium `0.29.0`; Isaac Sim `5.0`, local Isaac Lab `2.2.x` source tree


## 2026-08-17 16:18:00 +0800 - Final worktree cleanup
- Current phase: final local worktree cleanup after closed-loop diagnostic success.
- Completed: preserved compact candidate5989 evidence; removed generated outputs/scratch/history payloads; moved calibration out of outputs; pruned intermediate checkpoints; updated README/PROJECT_STATUS/.gitignore.
- Current conclusion: runtime outputs are ignored/regenerated; compact evidence is under `08_dual_arm_scene_layout/isaaclab_control/evidence/`.
- Current blockers: user review required before git add/commit/push.
- Modified files: docs, .gitignore, worklog, calibration path references, verified indices/readmes.
- Test results: pending static validation commands.
- Environment: no expensive GPU/Isaac/network/training workloads run.

## 2026-08-18 21:40:00 +0800 - DualArmMount y=0.16 final layout calibration
- Current phase: final mechanical arm/table relative position calibration.
- Completed: moved only `/World/Layout/DualArmMount` to `[0, 0.16, 0.8]` in both calibrated stages; preserved `RotateXYZ=[0,0,-90]` and `Scale=[1,1,1]`; regenerated `/World/Sensors/TopD435iVirtual/Camera` and `Frustum` from current `arm_base_link_d435i_2` anchor to SourceZone center; updated `config/manual_layout_calibrated.json`; refreshed RobotRoot distance markers.
- Current conclusion: production persistent Isaac still loads `manual_layout_calibrated_mass_fixed.usda`; orchestrator/cuRobo still read `manual_layout_calibrated.json`; Stage and JSON are synchronized.
- Current blockers: exact PhysX penetration was not run from this static edit pass; USD AABB robot/table overlap is not conclusive and should be checked visually/with Isaac static validation before long runs.
- Modified files: `scenes/manual_layout_calibrated.usda`, `scenes/manual_layout_calibrated_mass_fixed.usda`, `config/manual_layout_calibrated.json`, plus generated output metadata under `08_dual_arm_scene_layout/outputs/`.
- Test results: USD open and JSON sync PASS; camera coverage PASS; `05/06/07` py_compile PASS; `git diff --check` PASS.
- Environment: `isaaclab22_sim50` Python with Isaac USD extension paths; no Isaac app, DGN2, RFS, cuRobo planning, retarget, grasp execution, or full closed-loop was run.

## 2026-08-19 - Simplified planning plumbing pass
- Current phase: prepare production code for simplified endpoint/joint-space planning without touching core IK/RFS mathematics.
- Completed: GroundingDINO fixed workspace ROI crop; ESDF ROI depth invalidation outside `[170,0,970,700]`; persistent capture debug-prim hide/restore; per-64-candidate cuRobo worker lifecycle; per-batch GPU memory print hooks; `flexible_route_failures.jsonl`; RFS normal zero-pass status separation.
- Current conclusion: Exact COVER core remains unchanged; `CuroboGpuIK` still computes accepted with global config thresholds and exposes raw success/residual/margin for every returned seed. Stage-specific tolerance support still needs the planned core patch.
- Current blockers: `flexible_route_search.py` still contains existing broad task-space route sampling and final observed-map path check semantics; RFS core still contains 5120 support-pose/layer-graph algorithm. Both are intentionally not rewritten here.
- Modified files: `closed_loop/orchestrator.py`, `closed_loop/scripts/grounded_sam_backend.py`, `closed_loop/persistent_isaac/worker.py`, `closed_loop/planning/candidate_rfs_v2_runtime.py`, worklogs.
- Test results: py_compile PASS; `isaaclab22_sim50` closed-loop CPU unit tests PASS 9/9; `git diff --check` PASS.
- Environment: no full closed-loop, Isaac app, DGN2, RFS backend, cuRobo GPU worker, retarget, or physical execution was run.

## 2026-08-19 - PlacementZone 5 cm X-edge gap layout update
- Current phase: static layout parameter update only.
- Completed: moved only formal `PlacementZone` X from `0.40069047181642775` to `0.27617723418526796` in `manual_layout_calibrated.json`, `manual_layout_calibrated.usda`, and `manual_layout_calibrated_mass_fixed.usda`; synchronized PlacementZoneCenter marker and SourceToPlacement measurement; changed draft generator to derive the placement center from `ZONE_EDGE_GAP_M=0.05`; synchronized `manual_layout_draft.usda` to X `0.30`.
- Current conclusion: SourceZone right edge and PlacementZone left edge gap is `0.049999999999999989 m`, satisfying the required `<1e-9` invariant. SourceZone, Table, DualArmMount, Camera, zone sizes, robot joint pose, and planning tuning were not changed.
- Current blockers: none for this static layout update.
- Modified files: `08_dual_arm_scene_layout/config/manual_layout_calibrated.json`, `08_dual_arm_scene_layout/scenes/manual_layout_calibrated.usda`, `08_dual_arm_scene_layout/scenes/manual_layout_calibrated_mass_fixed.usda`, `08_dual_arm_scene_layout/scenes/manual_layout_draft.usda`, `08_dual_arm_scene_layout/scripts/01_create_manual_layout.py`, `08_dual_arm_scene_layout/scripts/check_placement_zone_gap.py`, worklog files.
- Test results: invariant script PASS; modified Python py_compile PASS; `isaaclab22_sim50` closed-loop CPU unit tests PASS 9/9; `git diff --check` PASS; old formal X grep over active paths returned no matches.
- Environment: no Isaac Sim, DGN2, RFS, cuRobo, retarget, full closed-loop, or physical simulation was run.

## 2026-08-19 - Experimental planner collision bypass execution flag
- Current phase: diagnostic/experimental execution plumbing only.
- Completed: added `--experimental-bypass-planner-collision` to `closed_loop/orchestrator.py`; it is allowed only with `--sim-execute` and does not change existing `--diagnostic-disable-*` semantics.
- Current conclusion: in experimental execution mode, RFS observed ESDF, Exact COVER observed ESDF, HOME->PRE observed ESDF/self-collision, and PRE->COVER observed ESDF/self-collision are bypassed through the existing per-gate plumbing while Isaac/PhysX execution remains enabled.
- Current blockers: not runtime-validated in this channel by request.
- Modified files: `closed_loop/orchestrator.py`, worklog files.
- Test results: orchestrator py_compile PASS; `isaaclab22_sim50` closed-loop CPU unit tests PASS 9/9; `git diff --check` PASS.
- Environment: no Isaac Sim, DGN2, RFS, cuRobo, retarget, full closed-loop, or physical simulation was run.

## 2026-08-19 - Standalone cuRobo RobotSegmenter capture adapter
- Current phase: local engineering adapter only; no baseline planner integration.
- Completed: added `isaaclab_control/perception/robot_segmentation/` with a standalone RobotDepthCleaner and CLI for existing capture folders.
- Current conclusion: `RobotSegmenter` import works in `curobo_v2`; the adapter maps `T_world_camera` to `T_base_camera = inv(T_world_base) @ T_world_camera`, adds the `[1,H,W]` batch depth shape, maps all 35 measured robot joints by name, and writes planning outputs under `capture/planning/`.
- Current blockers: none for standalone use; planner integration intentionally not performed.
- Modified files: new `perception/robot_segmentation` package and worklogs.
- Test results: py_compile PASS; CLI help PASS; standalone RobotSegmenter run PASS on `20260819_174407/cycle_001/capture`; `git diff --check` PASS.
- Environment: `curobo_v2`; no Isaac app, DGN2, retarget, baseline planner, or execution was run.

## 2026-08-19 - Route B current-to-PREGRASP MotionPlanner adapter

- Current phase: Route B phase-1 standalone backend integration.
- Completed: added `isaaclab_control/curobo_motion_planning_routeB/` with config, adapter, README, import test, and `test_current_to_pregrasp.py`; adapter builds ESDF from `capture/planning/filtered_depth.npy`, converts camera pose into `arm_base_link` frame, creates `MotionPlanner`, and calls `plan_cspace(q_current, q_pregrasp)`.
- Current conclusion: Route A remains untouched and is still the default via `use_legacy_keypoint_route: true`; Route B import/API wiring reaches cuRobo MotionPlanner.
- Current blocker: standalone test on `20260819_174407/cycle_001` returns `Start or End state in collision` from MotionPlanner with the filtered ESDF; no IK/retarget/Route A logic was changed to mask this.
- Modified files: new `08_dual_arm_scene_layout/isaaclab_control/curobo_motion_planning_routeB/` package plus worklogs.
- Test results: cuRobo MotionPlanner/Mapper import PASS; Route B py_compile PASS; Route B import unittest PASS; `git diff --check` PASS; standalone Route B smoke FAIL at MotionPlanner endpoint collision.
- Environment: `curobo_v2`; no Isaac app, DGN2, retarget, full closed-loop, or physical execution was run.

## 2026-08-19 - Route B formal collision policy wiring

- Current phase: enforce official Route B collision policy.
- Completed: Route B config now records `collision.environment_collision=true` and `collision.self_collision=false`; adapter explicitly disables self-collision on IK metrics/optimizer rollouts, TrajOpt metrics/optimizer rollouts, and Graph metrics rollout before constructing `MotionPlanner`; graph seed is disabled by default for phase-1 C-space TrajOpt planning to avoid the separate PRM start/end feasibility path.
- Current conclusion: environment ESDF collision remains enabled (`scene_collision_cfg_present=true`); self-collision is confirmed disabled across all Route B rollout configs (`all_self_collision_rollouts_disabled=true`); Route A remains untouched/default.
- Current blocker: same `current -> PREGRASP` command no longer emits `Start or End state in collision`, but MotionPlanner still returns `success=false` with zero trajectory points. This is now a TrajOpt planning failure under ESDF ON / self OFF, not the prior self-collision pair 2576 endpoint rejection.
- Modified files: `curobo_motion_planning_routeB/routeB_adapter.py`, `config_example.yaml`, `test_current_to_pregrasp.py`, `test_collision_audit.py`, and worklogs.
- Test results: Route B py_compile PASS; import unittest PASS; `git diff --check` PASS; standalone MotionPlanner run FAIL with new status `success=false`, `trajectory_points=0`, no start/end collision message.
- Environment: `curobo_v2`; no Isaac app, DGN2, retarget, full closed-loop, or physical execution was run.

## 2026-08-19 - Route B TrajOpt feasibility constraint audit

- Current phase: diagnose Route B `current -> PREGRASP` TrajOpt success=false without changing planning parameters.
- Completed: added standalone `test_trajopt_feasibility_audit.py`; recomputed cuRobo raw/interpolated metrics from the returned top trajectory; independently checked project ESDF clearance, joint position bounds, acceleration, and jerk.
- Current conclusion: raw and interpolated cuRobo metrics are both infeasible because `scene_collision` is strongly positive and `cspace` is tiny positive. Worst cuRobo scene-collision entry is timestep `[0,76,112]`, sphere `112`, link `arm_r_link_7`, value `169.164154`. Project ESDF post-check over the same returned trajectory reports no environment collision and min clearance `0.078266 m`; acceleration and jerk are within YAML limits.
- Current blocker: mismatch between cuRobo TrajOpt `scene_collision` metric feasibility and the project-side ESDF sphere query for the same trajectory.
- Modified files: new `curobo_motion_planning_routeB/test_trajopt_feasibility_audit.py`, worklogs.
- Test results: py_compile PASS; Route B import unittest PASS; `git diff --check` PASS.
- Environment: `curobo_v2`; no Isaac app, DGN2, retarget, full closed-loop, or physical execution was run.

## 2026-08-20 - Route B cuRobo scene_collision semantics audit

- Current phase: one-to-one diagnose cuRobo `scene_collision` vs project ESDF query for the same returned TrajOpt trajectory samples.
- Completed: added standalone `test_scene_collision_semantics_audit.py`; compared samples `t=76/sphere112`, `t=61/sphere184`, and `t=64/sphere185`; recorded cuRobo cost config, activation rule, VoxelData metadata, project ESDF query, cuRobo raw checker cost, and rollout constraint.
- Current conclusion: cuRobo raw checker and rollout constraint agree exactly, so the mismatch is not TrajOpt postprocessing. For `t=76/sphere112`, project query reports signed distance `0.143382 m` and clearance `0.112177 m`, while cuRobo raw checker implies signed distance `-0.002628 m` and penetration `0.033833 m`. Activation distance is `0`, weight is `5000`, and `constraint = weight * max(radius - signed_distance, 0)`.
- Current blocker: cuRobo SceneCollision/VoxelData query semantics or scene representation does not match the project `query_spheres` interpretation of the same `VoxelGrid`.
- Modified files: new `curobo_motion_planning_routeB/test_scene_collision_semantics_audit.py`, worklogs.
- Test results: py_compile PASS; Route B import unittest PASS; `git diff --check` PASS.
- Environment: `curobo_v2`; no Isaac app, DGN2, retarget, full closed-loop, or physical execution was run.

## 2026-08-20 - Route B ESDF filtered-depth ground-truth audit

- Current phase: decide project ESDF query vs cuRobo VoxelData query using original `filtered_depth` surface points.
- Completed: added standalone `test_esdf_ground_truth_audit.py`; sampled 500 valid filtered-depth surface points in `arm_base_link`; compared project `query_esdf_distance`, a CPU reproduction of cuRobo VoxelData sampling semantics, cuRobo radius-0 raw collision probe, and 20 direct voxel-center reads.
- Current conclusion: project query is correct for the current VoxelGrid. It uses `grid_sample` order `zyx`, `align_corners=True`, `[nx,ny,nz]`, and gives surface median/p90 abs SDF `0.002678/0.003372 m`; voxel-center direct-read median error is `1.49e-08 m`. cuRobo VoxelData semantic query gives surface median/p90 abs SDF `0.168801/0.278765 m` and voxel-center median error `0.071903 m`.
- Current blocker: `CUROBO_SCENE_REPRESENTATION_WRONG`. `VoxelData.params` stores dims as floats `[36.0, 55.0000038, 26.9999981]`; cuRobo Warp kernel truncates with `wp.int32()` to `[36,55,26]`, while feature tensor shape is `[36,55,27]`, corrupting Z-fastest indexing.
- Modified files: new `curobo_motion_planning_routeB/test_esdf_ground_truth_audit.py`, worklogs.
- Test results: py_compile PASS; Route B import unittest PASS; `git diff --check` PASS.
- Environment: `curobo_v2`; no Isaac app, DGN2, retarget, full closed-loop, or physical execution was run.

## 2026-08-20 - Route B VoxelData dimension contract fix

- Current phase: fix Route B cuRobo VoxelData discrete voxel count contract after ground-truth audit.
- Completed: `RouteBMotionPlannerAdapter` now normalizes shared cuRobo `VoxelData.params[...,0:3]` from authoritative `scene_grid.feature_tensor.shape` after `MotionPlanner(cfg)` and before `warmup/plan_cspace`; report plumbing records `voxel_shape_contract`.
- Current conclusion: scene-collision false positive is fixed. Params changed from `[[[36.0,55.0000038,26.9999981,0.02]]]` to `[[[36.0,55.0,27.0,0.02]]]`; feature shape `[36,55,27]`, feature count `53460`; dims/inv_pose/features unchanged. Ground-truth cuRobo surface median/p90 abs SDF improved to `0.002678/0.003372 m`, matching project query; voxel center median error is `0`.
- Current blocker: `plan_cspace` still returns success=false, but feasibility audit now reports only `cspace` constraint, max `5.80343e-08`; environment collision is false and min clearance is `0.075243 m`.
- Modified files: `curobo_motion_planning_routeB/routeB_adapter.py`, `test_current_to_pregrasp.py`, `test_scene_collision_semantics_audit.py`, worklogs.
- Test results: py_compile PASS; ESDF ground-truth audit PASS; scene_collision semantics audit PASS; Route B import unittest PASS; `git diff --check` PASS; current-to-PREGRASP standalone still FAIL due only to cspace residual.
- Environment: `curobo_v2`; no Isaac app, DGN2, retarget, full closed-loop, or physical execution was run.

## 2026-08-20 - Route B cspace numerical-bound residual fix

- Current phase: finish Route B phase-1 `current -> PREGRASP` standalone MotionPlanner feasibility.
- Completed: added Route B-only pre-planning joint-state numerical-bound sanitization in `RouteBMotionPlannerAdapter`; raw `robot_state.json` and raw route plan are not modified. The sanitizer uses the MotionPlanner rollout's actual position bounds, fixes only violations within `1e-5 rad`, moves corrected values `1e-6 rad` inside bounds, and raises on larger violations.
- Current conclusion: the previous `cspace` residual was caused by tiny static left-arm bound residuals, not by right-arm motion or environment collision. Corrected DOF are `arm_l_joint_2` and `arm_l_joint_4` for both `q_current_planning` and `q_pregrasp_planning`; max original violation `4.818e-06 rad`.
- Current blocker: none for standalone Route B `current -> PREGRASP`; formal closed-loop Route B wiring is still a future step.
- Modified files: `curobo_motion_planning_routeB/routeB_adapter.py`, `test_trajopt_feasibility_audit.py`, `test_scene_collision_semantics_audit.py`, `test_current_to_pregrasp.py`, and worklogs.
- Test results: py_compile PASS; feasibility audit PASS with `scene_collision=0`, `cspace=0`, no failed constraints, min clearance `0.075290 m`; standalone `test_current_to_pregrasp.py` PASS with `success=True`, `trajectory_point_count=41`, planning time `1.863 s`; returned trajectory postcheck PASS with environment collision false and joint-limit violations `0`; `git diff --check` PASS.
- Environment: `curobo_v2`; no Isaac app, DGN2, retarget, full closed-loop, or physical execution was run.

## 2026-08-20 - Route B true right-arm-only current-to-PREGRASP

- Current phase: integrate ChatGPT-provided `right_arm_only_core` into local Route B as a standalone test path.
- Completed: added `test_current_to_pregrasp_right_arm_only.py`; reused existing Route B `MotionPlannerCfg`/SceneCfg parameters, collision policy `environment=true/self=false`, VoxelData feature-shape fix, and postchecks. The 35DOF adapter is used only to build the scene and sanitized full q contract; it does not generate a 35DOF trajectory.
- Current conclusion: true right-arm-only cuRobo MotionPlanner succeeds for `current -> PREGRASP`. Planner action_dim is `7`; active joints are exactly `arm_r_joint_1..7`; locked joint count is `28`; raw optimizer solution shape is `[1,1,16,7]`; dense trajectory is `[41,7]`.
- Current blocker: none for standalone right-arm-only phase-1 test. Production closed-loop Route B wiring remains a future step.
- Modified files: `curobo_motion_planning_routeB/routeB_adapter.py`, new `test_current_to_pregrasp_right_arm_only.py`, and worklogs.
- Test results: right-arm-only core contract unittest PASS; Route B import unittest PASS; py_compile PASS; `git diff --check` PASS; generated `trajectory_right_arm.npz` and `report_right_arm.json`; postcheck PASS with environment collision false, min clearance `0.075289 m`, `scene_collision=0`, `cspace=0`, joint limits PASS, velocity/acceleration/jerk finite PASS.
- Environment: `curobo_v2`; no Isaac app, DGN2, retarget, full closed-loop, or physical execution was run.

## 2026-08-20 - Route B right-arm trajectory visualization

- Current phase: connect GPT-provided `trajectory_visualizer` to the validated Route B right-arm-only path.
- Completed: added `build_visualization_right_arm_bundle.py`; reused RobotSegmenter-filtered depth, Route B arm_base_link scene contract, true 7DOF locked planner FK/collision spheres, verified ESDF sphere query, and `trajectory_right_arm.npz`.
- Current conclusion: visualization bundle generated successfully with 30000 scene points, 233 collision spheres, 95 moving spheres, 138 static spheres, 41 frames, EE frame `arm_r_link_tf`, and global min clearance `0.075289 m`.
- Current blocker: none for bundle generation. GUI interaction confirmation is manual; viewer loaded the bundle and remained in matplotlib blocking window until the intentional 30s timeout.
- Modified files: new `curobo_motion_planning_routeB/build_visualization_right_arm_bundle.py`, worklogs.
- Test results: py_compile PASS; `trajectory_visualizer.test_bundle` PASS; bundle build PASS; viewer launch no traceback before timeout; `git diff --check` PASS.
- Environment: `curobo_v2`; no Isaac app, DGN2, retarget, full closed-loop, video export, or physical execution was run.

## 2026-08-20 - Route B full pipeline integration

- Current phase: Route B closed-loop integration with shared Route A task endpoints and true 7DOF cuRobo dense arm trajectories.
- Completed:
  - Added front-end route selection: explicit `--motion-route legacy|curobo`; if omitted in a TTY, the program prompts before starting heavy processes. Non-TTY defaults to legacy for compatibility.
  - Route A remains the legacy/flexible q7 + quintic executor path through `PersistentIsaacClient.execute()`.
  - Route B now uses LEAP reach-region-only prefilter, strict Exact COVER with observed ESDF collision gate OFF, relaxed PREGRASP endpoint collision OFF, and true Route B MotionPlanner with environment collision ON / self collision OFF.
  - Route B full backend produces all seven arm segments: `CURRENT_TO_PREGRASP`, `PREGRASP_TO_COVER`, `COVER_TO_LIFT`, `LIFT_TO_TRANSFER`, `TRANSFER_TO_PLACE`, `PLACE_TO_RETREAT`, `RETREAT_TO_HOME`.
  - Route B execution is additive through `execute_routeB`; Route A `execute()` / `execute_segment()` are not replaced.
- PLACE=0 conclusion:
  - `candidate=789` has `PLACE raw IK=0/945` in both Route A and Route B for the same case/layout/q_cover/placement registry. The root cause is candidate-specific endpoint infeasibility for the shared PLACE task, not a stale PlacementZone or Route B frame bug.
  - Other candidates in the same goal pool have complete endpoint chains, e.g. `candidate=676`, so Route B now runs cheap LIFT/TRANSFER/PLACE/RETREAT endpoint-chain preflight before expensive dense MotionPlanner.
- Latest successful planning:
  - Session: `20260820_040547/cycle_001`
  - Query: `pencil`
  - Selected candidate: `676`
  - Full planning status: PASS
  - All seven Route B segments pass with `scene_collision_max=0`, `cspace_max=0`, and active joints exactly `arm_r_joint_1..7`.
- Latest physical execution:
  - Same session completed the dense route through `RETREAT_TO_HOME`.
  - Task outcome: FAIL due to `EMPTY_GRASP`, not due to route interruption.
  - `verify_lift=0.3519 mm`, max lift `0.7849 mm`, final green zone `false`.
  - COVER refinement: `4.62 mm / 1.13 deg`; GRASP refinement: `4.12 mm / 1.06 deg`.
  - Execution report: `08_dual_arm_scene_layout/isaaclab_control/outputs/closed_loop_sessions/20260820_040547/cycle_001/execution_routeB/report_routeB.json`.
- Current blocker:
  - Route B motion planning and full motion execution wiring are functional.
  - Simulated physical grasp did not lift the object. Per current decision, EMPTY_GRASP is a physical grasp-quality result and does not block route-completion smoke execution.
- Validation:
  - py_compile PASS.
  - Route B/closed-loop unit tests PASS.
  - `git diff --check` PASS.
  - Persistent Isaac background workers were stopped with SIGINT after runs.

## 2026-08-20 - Route B cup run stopped during full planning

- Session inspected: `20260820_063033/cycle_001`; user interrupted during Route B planning for query `cup`.
- No Isaac/full_motion/orchestrator background process remained after inspection.
- Front-half selected `candidate=7938` and PASSed true 7DOF `CURRENT_TO_PREGRASP` with trajectory `[61,7]`, environment collision false, `scene_collision=0`, `cspace=0`.
- Back-half endpoint chain pool existed (`chain_count=32`), so this was not a PLACE endpoint-IK-zero case.
- Actual blocking gate observed from artifacts: true Route B full MotionPlanner reached `COVER_TO_LIFT` for 16 chains, generated `traj_cover_to_lift.npz`, then failed before producing `traj_lift_to_transfer.npz`; terminal tail identifies `LIFT_TO_TRANSFER` as failed stage.
- Logging fix: expected no-path full MotionPlanner failures now write `routeB_full_plan_report.json(success=false)` with `trial_summary` and concise stdout; orchestrator excludes that candidate with stage counts instead of printing traceback tails.

## 2026-08-20 - Route B attachment root cause

- For `cup`, session `20260820_063033/cycle_001`, selected candidate `7938`, the full Route B blocker is not endpoint IK and not ordinary right-arm path reachability.
- Offline control: disabling the carried-object attachment only on `LIFT_TO_TRANSFER` makes all seven Route B dense arm segments PASS with environment collision ON and self collision OFF.
- Attachment audit at LIFT shows the attached target proxy is already invalid against the current no-target ESDF before motion: 8/48 attached spheres collide; min clearance `-36.8 mm`. Setting attachment padding to `0` still collides: 6/48 spheres; min clearance `-31.9 mm`.
- Proxy source is mask/depth fallback due invalid Isaac collision AABB; proxy dims are about `[0.157, 0.173, 0.148] m`.
- Current root cause: `LIFT_TO_TRANSFER` fails because the attached cup proxy/sphere representation overlaps the observed ESDF at the LIFT start state. This is an attachment proxy/scene contract issue, not a q7 endpoint or Route A issue.

## 2026-08-20 - Route B transfer attachment OFF for flow-through

- Decision applied: Route B `LIFT_TO_TRANSFER` no longer attaches the target proxy by default (`routeB_full_pipeline.transfer_attachment=false`).
- This keeps Route B environment collision ON and self collision OFF; it only disables the carried-object proxy during the transfer segment.
- Rationale: attachment audit showed the cup proxy already collides with the no-target ESDF at LIFT start, while the same q7 path without attachment produces a complete 7-segment Route B plan.
- Runtime policy banner now prints the active false/off options in English plus Chinese explanation to avoid confusing this flow-through setting with production carried-object collision safety.

## 2026-08-20 - Route B stdout and execution failure handling

- Raw Route B report JSON is no longer printed to terminal by front-half/full backends; terminal now shows concise summaries and report paths.
- Route B physical FAIL now reports concrete failure stage and metrics.
- For completed task-result failures (`EMPTY_GRASP`, `FINAL_GREEN_ZONE`), orchestrator keeps the persistent Isaac session alive, does not commit placement, and continues to the next capture/query cycle.
- Latest observed physical failure was `EMPTY_GRASP`, not motion-generation failure: verified lift was approximately zero and max lift was only `6.88 mm`; COVER/GRASP refinement errors were small.
- Separate user observation remains: hand appears to push object before contact. This should be diagnosed next from `trace_routeB.csv` and per-stage object pose/contact telemetry before changing grasp parameters.

## 2026-08-20 - task/route architecture and color-sort foundation

- Current phase: two-axis CLI/task architecture and first color-sort perception/planning plumbing.
- Completed:
  - Added task selection dimension: `semantic-grasp` and `color-sort`.
  - Preserved motion-route dimension: `legacy` and `curobo`.
  - Added interactive TTY task menu before heavy Isaac/cuRobo/DGN2 processes; non-TTY defaults stay legacy semantic-grasp for compatibility.
  - Semantic-grasp remains GroundingDINO+SAM based and does not activate color-sort dye/HSV flow.
  - Color-sort now assigns balanced seeded red/blue colors once per Persistent Isaac session and saves `color_assignment.json`.
  - Runtime dye is visual-material-only; it does not modify official meshes, URDF/USD, collision, mass, scale, or grasp data.
  - Added HSV red/blue connected-component instance segmentation, SourceZone 3D filtering, selected-instance target metadata, and failed-instance exclusion for the current scene.
  - Color-sort destination zones are derived dynamically by splitting the calibrated PlacementZone into `red_zone` and `blue_zone`; Route B endpoint preflight reuses the existing A placement sampler with a layout override for the selected color zone.
  - RouteB stdout is more compact: large raw JSON report lines are suppressed from terminal but kept in files.
- Validated:
  - `py_compile` PASS for modified Python files.
  - closed-loop logic/flexible planning lightweight tests PASS in `graspnet2.0`.
  - RouteB goal-pool/attachment/robot-config tests PASS with module PYTHONPATH.
  - Synthetic HSV smoke PASS: red and blue instances are detected inside SourceZone; excluding `red_000` selects `blue_000`.
  - `git diff --check` PASS.
- Not yet run in this phase:
  - Real Isaac color material smoke.
  - Real semantic-grasp + cuRobo regression after this task/route refactor.
  - Real color-sort 2-object repeated execution.
  - Full multi-object color-sort attempt.

## 2026-08-20 - color-sort first-run path bug fixed

- User run: `--task color-sort --motion-route curobo --color-seed 42 --planning-only`, scene `scene_0065`.
- COLOR SORT material/perception reached real capture successfully: RED source=2, BLUE source=2, selected HSV instance `red_001`, DGN2 produced `capture/dgn2/red_001/official_leap_1024_target_ranked.npz` with 4187 target candidates.
- Failure root cause: RouteB LEAP reach runtime received `query=red` and looked for `capture/dgn2/red/official_leap_1024_target_ranked.npz`; color-sort DGN2 uses per-instance slug `red_001`.
- Fix: RouteB/RFS runtimes now use `target_slug` for file layout, while the semantic/color query remains metadata. RouteB full backend also accepts the explicit selected target mask path, so color-sort does not pretend HSV masks are GroundedSAM masks.
- No Isaac process remained after the failed attempt.

## 2026-08-20 - Zone semantics and Trimesh/Isaac alignment

- SourceZone and PlacementZone are confirmed visual/planning regions only: `zone_collision=false`. They should not be expected to physically support objects; only the Table collision should support objects.
- The apparent Trimesh-vs-Isaac mismatch is due to frame mode: scene folders are authored in a local tabletop/source-area frame, while Persistent Isaac places them with `T_world_object = T_world_SourceZone @ T_source_object` in the calibrated layout.
- Added calibrated Trimesh viewing mode to display the same scene embedded in the Isaac layout, including blue/green region markers.
- Existing `object_physics_audit.json` from session `20260820_081936` contains invalid AABB sentinel values for objects, so its PASS status was not trustworthy. The audit code now treats those as invalid/WARN instead of PASS.

## 2026-08-20 - scene_0065 migration contract: root cause fixed

- Root cause was pre-physics, not Route B, DGN2, IK, ESDF, or ordinary PhysX settling.
- Two defects were found in Persistent Isaac spawn:
  1. when a manifest lacked `simulation_usd`, the worker used the same numeric object index in the unrelated test split; for example, training scene_0065 Clock index 27 could load a test-split asset at index 27;
  2. `set_reference_transform()` called `Gf.Matrix4d.SetRotate()` followed by `SetTranslate()`.  In Gf, `SetTranslate()` resets the transform to translation-only, so every authored object orientation was discarded before the first physics step.
- The worker now resolves the validated source dataset lineage by object code + pool index, records source-mesh and editable-USD hashes, and fails closed if identity cannot be proven.  It writes the full 6D pose using `SetTranslateOnly()` so orientation is retained.
- New zero-step artifact records reference root, rigid root, `T_reference_root_rigid_body`, transform ops/scales, visual/collision AABBs, expected pose error, SourceZone membership, and pairwise-distance preservation.  Physics does not start if this audit fails.
- `scene_0065` corrected pre-physics result: PASS; SourceZone 0.500 x 0.300 m matches training tabletop; no SourceZone scale enters the mapping; max pairwise distance error 0.011837 mm; all objects retain the manifest 6D pose within 0.013 mm / 0.031 deg.
- New settle traces at 1 step / 0.05 / 0.10 / 0.25 / 0.50 / 1.00 s show maximum 1 s drift 0.0017 mm for both P1 (training default object-object collision ON) and P2 (persistent task-object filtering OFF).  The previous Clock/Microscope/Pencil/Ipod 18--85 mm movement was caused by the pre-physics pose contract bug.
- Persistent production keeps its existing collision filtering and the user-requested 0.3 friction; the audit-only `training_default` policy is available only with `--scene-migration-audit` for evidence collection.

## 2026-08-20 - semantic target robot exclusion restored

- The cuRobo RobotSegmenter mask is the sole robot-mask authority.  It now serves its original planning-depth consumer and, separately, semantic final-target rejection; no second robot segmentation method was introduced.
- The semantic GroundingDINO adapter previously selected the highest-score box before SAM.  It now performs SAM on each proposal and selects only the highest-score legal proposal after mask-level robot exclusion and 3D SourceZone validation.
- Current `cup` replay from session `20260820_091607`: old selected idx 0 was `99.958%` robot mask and outside SourceZone; it is rejected.  idx 1 (`78.146%` robot) is rejected.  idx 2 is selected as the highest-score legal residual.
- The adapter asserts RobotSegmenter report `capture_dir == RGB capture_dir`, so cross-cycle masks fail loudly with `STALE_ROBOT_MASK`.

## 2026-08-20 - unified RobotSegmenter RGB/depth perception contract

- `RobotDepthCleaner.remove_robot()` remains the only robot-pixel authority. Its current-cycle output is now consumed by all three paths: semantic RGB (`rgb_no_robot`, DINO/SAM gates), color HSV (`~robot_mask` before morphology), and depth/point-cloud planning (`filtered_depth`).
- Raw capture assets remain immutable. New derived artifacts are `capture/planning/rgb_no_robot.png` (neutral `[127,127,127]` robot fill) and `robot_mask_overlay.png`, alongside existing mask/depth artifacts.
- Semantic rejection is two-stage and hard: DINO box overlap first, SAM mask overlap second. The threshold is 0.5 only for a robot-dominant proposal; a smaller SAM overlap is removed and the residual must still have valid depth inside rigid SourceZone. Cross-cycle report-path mismatch raises `STALE_ROBOT_MASK`.
- `20260820_091931/cycle_001` evidence: previous selected bottle box had 0.6308 robot pixel fraction; it is now pre-SAM `REJECT_ROBOT_OVERLAP`. Seven pure safety regressions and an HSV replay passed. Fresh GPU replay is pending a normal user runtime because this agent sandbox reports no CUDA device.

## 2026-08-20 - pre-experiment preflight PASS

- Fresh headless GPU perception smoke completed without planning or arm motion: scene_0065 migration guard PASS, RGB-D capture, current-cycle RobotSegmenter, `rgb_no_robot`, `filtered_depth`, GroundingDINO/SAM `pencil` selection, then clean Persistent Isaac shutdown.
- Smoke result: RobotSegmenter mask `225801` pixels; GroundingDINO actual input was `capture/planning/rgb_no_robot.png`; two proposal masks passed robot and SourceZone gates; selected index 0 yielded 5286 valid points; stale-mask check PASS.  No robot proposal became final target.
- Static preflight confirms: SourceZone migration fix remains (`SetTranslateOnly`, lineage asset resolver, no visual-scale mapping); selected-target removal operates only on `filtered_depth`; Route B history has true action dimension 7, environment collision ON, all IK/TrajOpt/Graph self-collision rollout weights disabled; task/route dispatch all four combinations pass; HSV masks exclude RobotSegmenter pixels before morphology; no stale Isaac process remains.

## 2026-08-20 — mainline cleanup complete

- Frozen mainline behavior was not modified. Added only `MAINLINE.md`, `PRE_EXPERIMENT_BASELINE.md`, and `CLEANUP_20260820.md` to document the production contract and cleanup record.
- Generated output is reduced to four retained session baselines, two migration audits, and the latest preflight smoke. Superseded captures/plans from known bad migration, stale perception, and intermediate retries were removed.
- The only source files moved outside the repository were unreferenced historical templates/backups; they are recoverable at `/home/lin/DexGraspNet2_Wuji2_cleanup_archive_20260820/`.
- Core no-motion regressions all PASS. The repository is ready for a user-initiated formal experiment; cleanup intentionally did not run Isaac, DGN2, Route B planning, or grasp execution.

## 2026-08-20 — scene_0020 migration guard correction

- The first post-cleanup experiment was correctly stopped by the pre-physics guard, but the guard had selected the wrong evidence source. Scene_0020's USD reference/rigid roots exactly match the expected SourceZone-embedded manifest poses; no frame, scale, asset-lineage, or physics contract regression occurred.
- Pencil's `RigidObject.data.root_pose_w` differed by `0.024183 mm / 0.103546 deg` immediately after `SimulationContext.reset()`. This is retained as a runtime diagnostic, not accepted as a failure of the pre-physics USD migration contract.
- The guard and formal offline audit now consistently validate `actual_rigid_body_world_pose`. This is a measurement-source correction only; no object transform, SourceZone mapping, physics setting, perception, DGN2, Route B, or grasp contract changed.

## 2026-08-20 — Route B retry continuity repair

- Candidate-level MotionPlanner no-path is now nonfatal. A failed `CURRENT_TO_PREGRASP` backend report follows the same retry policy as a failed full Route B report: exclude that candidate, retain the persistent Isaac scene, and test the next endpoint-feasible candidate.
- The session only stops after the available Route B goal pool is genuinely exhausted or a real protocol/backend failure occurs; it no longer stops at the first structured front-half no-path report.

## 2026-08-20 — color-query task semantics

- `color-sort` is now explicitly parallel to `semantic-grasp`, not an HSV-only automatic-red task. The user selects one requested color (`red`/`blue`, or Chinese aliases); the persistent session repeatedly captures, identifies, plans for, and places every available object of that requested color.
- GroundingDINO/SAM is now the required color-text matcher using prompt `"<color> object"` on `rgb_no_robot.png`. Existing RobotSegmenter/SAM/SourceZone gates remain active. HSV never reads `sort_color` or `color_assignment.json` to choose a target; it validates the current rendered color and converts a broad same-color SAM region into a single current-image instance mask.
- A retained live color capture (`20260820_104319/cycle_001`) replayed `red object` successfully: GroundingDINO/SAM status PASS, best visual match `red_003` (7784 pixels; 49.33% of the broad SAM region; 99.31% of that red component); every blue instance had zero overlap.
- Route B candidate retry remains first-line alternate-pose recovery. An exhausted planning funnel is non-terminal for color-sort: planning is already frozen at HOME, so the failed instance is skipped and a fresh capture selects another same-color instance. Route B execution exceptions now open the hand and reverse only already-issued cuRobo dense trajectories before accepting continuation; it records `RECOVERED_FAIL` only after HOME tolerance passes.

## 2026-08-20 — repository mainline cleanup

- Production behavior is frozen and unchanged. Five superseded top-level
  integration/backup packages and one byte-identical Route B visualizer package
  were archived outside the repository after production-reference searches
  returned none.
- Generated closed-loop output was reduced from roughly 381 MiB to roughly
  96 MiB while retaining the selected candidate for both Route B baselines,
  perception false-positive fixtures, HSV evidence, preflight smoke, and formal
  scene-migration audits. Regenerable Trimesh output removed another ~463 MiB.
- Current documentation now consistently uses TASK (`semantic-grasp`,
  `color-sort`) and MOTION ROUTE (`legacy`, `curobo`) rather than the obsolete
  user-facing Route C description. The authoritative environment names and
  unified RobotSegmenter contract are documented.
- Cleanup regressions PASS after correcting two test-command mistakes (a shell
  script was initially passed to `py_compile`; one import smoke initially lacked
  the control-root `PYTHONPATH`). These were command errors, not code failures.
- A standardized, resumable three-scene color campaign runner now owns a single
  auto-timestamp output root. It does not implement perception or planning; it
  invokes `run_closed_loop.sh` explicitly and stops after a genuine process
  error so the first failure can be diagnosed before continuing.

## 2026-08-20 — three-scene campaign completed without process aborts

- Canonical result: `outputs/color_sort_campaigns/20260820_111709/` with
  `campaign_report.md` (human index) and `campaign_summary.json` (machine index).
  All case logs and sessions are under this one timestamped root; superseded
  pre-repair attempts were deleted after their root causes were recorded.
- The campaign is `PARTIAL_COMPLETE`, not a grasp PASS. All three production
  processes exited 0 and all planning-time failures kept the robot HOME, but no
  case reached a seven-segment full Route B plan or Isaac arm execution.
- `scene_0000:RED`: `red_000` passed several complete endpoint funnels; dense
  planning was dominated by `COVER_TO_LIFT` failure, with additional
  `CURRENT_TO_PREGRASP` and one `PREGRASP_TO_COVER` no-path.
- `scene_0020:BLUE`: `blue_000` had reachable LIFT/TRANSFER endpoint nodes, but
  every attempted blue-zone PLACE endpoint set reported `targets=385 raw=0`.
  A remaining HSV blue instance had no legal GroundedSAM proposal overlap.
- `scene_0065:RED`: all four current red instances were independently attempted
  after the all-proposal wiring repair. `red_000` exhausted
  CURRENT_TO_PREGRASP; `red_001` reached LIFT/TRANSFER endpoint sets but PLACE
  was raw IK zero; `red_002` and `red_003` were too small for the official DGN2
  sampler to produce a seed.
- Three runtime-continuity defects were fixed with regressions: stale full-plan
  report reuse, target-local DGN2 no-seed process abort, and single-DINO-proposal
  color matching. None of these repairs relaxed task, collision, IK, scene,
  physics, grasp, or Route B contracts.
- Post-campaign checks: no Isaac/Kit process and no GPU compute process remained;
  semantic safety 7/7, closed-loop 11/11, core 17/17, Route B 10/10, HSV replay,
  GroundedSAM embedded compile, scene migration audit, and `git diff --check`
  all PASS.
