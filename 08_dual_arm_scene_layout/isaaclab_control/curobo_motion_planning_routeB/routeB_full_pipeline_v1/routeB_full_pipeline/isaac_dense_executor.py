from __future__ import annotations

"""Additive Persistent-Isaac executor for Route B.

The existing Route A `execute()` method remains untouched.  Route B arm paths
are cuRobo dense trajectories and are not replaced by quintic interpolation.
At the Isaac physics rate we perform only piecewise-linear time interpolation
between adjacent cuRobo samples.
"""

import csv
import json
import math
from pathlib import Path
from typing import Any, Callable

import numpy as np


def _load_dense(path: Path) -> dict[str, np.ndarray]:
    with np.load(Path(path), allow_pickle=False) as z:
        data = {key: np.asarray(z[key]) for key in z.files}
    q = np.asarray(data["q_rad"], dtype=np.float64)
    t = np.asarray(data["time_s"], dtype=np.float64).reshape(-1)
    names = [str(x) for x in data["joint_names"].tolist()]
    expected = [f"arm_r_joint_{i}" for i in range(1, 8)]
    if names != expected:
        raise RuntimeError(f"Route B dense joint order changed: {names}")
    if q.ndim != 2 or q.shape[1] != 7 or q.shape[0] <= 1:
        raise RuntimeError(f"invalid dense q shape {q.shape}: {path}")
    if t.shape != (len(q),) or np.any(np.diff(t) < -1e-12):
        raise RuntimeError(f"invalid dense time array: {path}")
    if float(t[-1]) <= 0.0:
        raise RuntimeError(f"zero-duration dense trajectory: {path}")
    return {**data, "q_rad": q, "time_s": t}


def execute_routeB_manifest(
    scene: Any,
    *,
    project_root: Path,
    manifest_path: Path,
    output_dir: Path,
    target_segmentation_id: int,
    quintic: Callable[[float], float],
) -> dict[str, Any]:
    import torch

    project_root = Path(project_root).resolve()
    manifest_path = Path(manifest_path).resolve()
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    case_root = Path(manifest["case_root"]).resolve()
    candidate_index = int(manifest["candidate_index"])
    if int(target_segmentation_id) not in scene.objects_by_seg:
        raise KeyError(
            f"target segmentation id {target_segmentation_id} absent from persistent scene"
        )
    target = scene.objects_by_seg[int(target_segmentation_id)]

    with np.load(Path(manifest["hand_waypoints"]), allow_pickle=False) as z:
        hand_names = [str(x) for x in z["finger_joint_names"].tolist()]
        hand_stage_names = [str(x) for x in z["waypoint_names"].tolist()]
        hand_q = np.asarray(z["waypoint_joint_positions"][0], dtype=np.float64)
        squeeze_dense = np.asarray(z["squeeze_dense_q20_path"], dtype=np.float64)
    hand_index = {name: i for i, name in enumerate(hand_stage_names)}
    hand_ids, matched = scene.robot.find_joints(hand_names, preserve_order=True)
    if matched != hand_names or len(hand_ids) != 20:
        raise RuntimeError("Wuji2 20-DOF mapping changed")

    # Reuse Route A's exact post-retarget flange targets for endpoint refinement.
    arm_target_path = case_root / "07_arm_execution/arm_flange_targets.npz"
    with np.load(arm_target_path, allow_pickle=False) as z:
        arm_target_names = [str(x) for x in z["waypoint_names"].tolist()]
        arm_target_world = np.asarray(z["world_from_right_flange"], dtype=np.float64)
    arm_target_index = {name: i for i, name in enumerate(arm_target_names)}
    for required_target in ("cover", "grasp"):
        if required_target not in arm_target_index:
            raise RuntimeError(f"missing exact {required_target} flange target in {arm_target_path}")

    segments = {row["state"]: row for row in manifest["segments"]}
    required = {
        "CURRENT_TO_PREGRASP",
        "PREGRASP_TO_COVER",
        "COVER_TO_LIFT",
        "LIFT_TO_TRANSFER",
        "TRANSFER_TO_PLACE",
        "PLACE_TO_RETREAT",
        "RETREAT_TO_HOME",
    }
    if required.difference(segments):
        raise RuntimeError(
            f"Route B manifest missing segments: {sorted(required.difference(segments))}"
        )

    durations = scene.config["durations_s"]
    render = not bool(getattr(__import__("__main__"), "ARGS").headless)
    trace: list[dict[str, Any]] = []
    sim_time = 0.0
    initial_object_position = target.data.root_pos_w[0].clone()
    max_object_lift_mm = 0.0
    scene.current_desired_arm = scene.robot.data.joint_pos[:, scene.arm_ids].clone()
    failure_stage = None
    # Route B recovery never invents a new arm path.  It only reverses the
    # already-issued cuRobo dense samples, then verifies that the initial HOME
    # state was reached.  This is deliberately separate from hand-only quintic
    # actions and does not call IK or a planner during execution.
    arm_history: list[tuple[str, dict[str, np.ndarray]]] = []
    active_arm: dict[str, Any] | None = None

    def target_lift_mm() -> float:
        return 1000.0 * float(
            target.data.root_pos_w[0, 2] - initial_object_position[2]
        )

    def step_trace(state: str, progress: float) -> None:
        nonlocal sim_time, max_object_lift_mm
        scene.step(render=render)
        sim_time += scene.dt
        lift = target_lift_mm()
        max_object_lift_mm = max(max_object_lift_mm, lift)
        trace.append(
            {
                "time_s": float(sim_time),
                "state": state,
                "progress": float(progress),
                "object_lift_mm": float(lift),
                "max_arm_qdot_rad_s": float(
                    torch.max(
                        torch.abs(scene.robot.data.joint_vel[:, scene.arm_ids])
                    )
                ),
                "max_arm_joint_goal_error_deg": float(
                    torch.max(
                        torch.abs(
                            torch.rad2deg(
                                scene.current_desired_arm
                                - scene.robot.data.joint_pos[:, scene.arm_ids]
                            )
                        )
                    )
                ),
            }
        )

    def hold_state(state: str, seconds: float) -> None:
        count = max(1, round(float(seconds) / scene.dt))
        for i in range(count):
            step_trace(state, (i + 1) / count)

    def hand_segment(state: str, goal_q20: np.ndarray, duration_s: float) -> None:
        start = scene.command[:, hand_ids].clone()
        goal = torch.as_tensor(
            goal_q20, device=scene.robot.device, dtype=scene.command.dtype
        ).reshape(1, 20)
        count = max(2, round(float(duration_s) / scene.dt))
        for i in range(count + 1):
            alpha = quintic(i / count)
            scene.command[:, hand_ids] = start + alpha * (goal - start)
            step_trace(state, i / count)

    def _dense_command(data: dict[str, np.ndarray], t: float) -> torch.Tensor:
        q = data["q_rad"]
        ts = data["time_s"]
        j = int(np.searchsorted(ts, t, side="right") - 1)
        j = max(0, min(j, len(ts) - 2))
        denom = max(float(ts[j + 1] - ts[j]), 1e-12)
        alpha = float(np.clip((t - ts[j]) / denom, 0.0, 1.0))
        q_cmd = (1.0 - alpha) * q[j] + alpha * q[j + 1]
        return torch.as_tensor(
            q_cmd, device=scene.robot.device, dtype=scene.command.dtype
        ).reshape(1, 7)

    def dense_arm(state: str, path: Path) -> None:
        nonlocal active_arm
        data = _load_dense(path)
        ts = data["time_s"]
        total = float(ts[-1])
        count = max(1, round(total / scene.dt))
        print(f"[Route B][ARM] {state}: {len(data['q_rad'])} cuRobo points, {total:.3f}s", flush=True)
        last_bucket = -1
        active_arm = {"state": state, "data": data, "time_s": 0.0}
        for k in range(count + 1):
            t = min(total, k * scene.dt)
            desired = _dense_command(data, t)
            scene.command[:, scene.arm_ids] = desired
            scene.current_desired_arm = desired
            active_arm["time_s"] = float(t)
            progress = t / total
            step_trace(state, progress)
            bucket = min(10, int(progress * 10.0 + 1e-9))
            if bucket != last_bucket:
                last_bucket = bucket
                max_err = float(torch.max(torch.abs(torch.rad2deg(
                    scene.current_desired_arm - scene.robot.data.joint_pos[:, scene.arm_ids]
                ))))
                print(f"  {state:<22} {100*progress:5.1f}% | arm max err={max_err:6.2f} deg | object lift={target_lift_mm():7.1f} mm", flush=True)
        arm_history.append((state, data))
        active_arm = None

    def reverse_dense_arm(
        state: str,
        data: dict[str, np.ndarray],
        *,
        start_time_s: float | None = None,
    ) -> None:
        """Replay a previously traversed dense trajectory backwards."""
        total = float(data["time_s"][-1])
        start = total if start_time_s is None else float(np.clip(start_time_s, 0.0, total))
        count = max(1, round(start / scene.dt))
        print(f"[Route B][RECOVERY] reverse {state}: {start:.3f}s cuRobo samples", flush=True)
        for k in range(count + 1):
            t = max(0.0, start - k * scene.dt)
            desired = _dense_command(data, t)
            scene.command[:, scene.arm_ids] = desired
            scene.current_desired_arm = desired
            step_trace(f"RECOVERY_{state}", 1.0 - (t / max(start, 1.0e-12)))

    def recover_home_after_routeb_failure() -> tuple[str, str | None]:
        """Open the hand and reverse only executed cuRobo arm segments to HOME."""
        nonlocal active_arm
        try:
            hand_segment(
                "RECOVERY_OPEN",
                hand_q[hand_index["pregrasp"]],
                float(durations.get("release", 1.0)),
            )
            if active_arm is not None:
                reverse_dense_arm(
                    str(active_arm["state"]),
                    active_arm["data"],
                    start_time_s=float(active_arm["time_s"]),
                )
                active_arm = None
            for prior_state, prior_data in reversed(arm_history):
                reverse_dense_arm(prior_state, prior_data)
            actual = scene.robot.data.joint_pos[:, scene.arm_ids]
            home = torch.as_tensor(
                scene.home_q, device=actual.device, dtype=actual.dtype
            ).reshape(1, 7)
            err_deg = float(torch.max(torch.abs(torch.rad2deg(actual - home))))
            tolerance_deg = float(scene.config.get("recovery_home_joint_tolerance_deg", 8.0))
            if err_deg > tolerance_deg:
                raise RuntimeError(
                    f"HOME recovery joint error {err_deg:.2f}deg > {tolerance_deg:.2f}deg"
                )
            print(f"[Route B][RECOVERY] HOME PASS | max joint error={err_deg:.2f}deg", flush=True)
            return "HOME", None
        except Exception as recovery_exc:
            return "FAILED", f"{type(recovery_exc).__name__}: {recovery_exc}"

    def _quat_from_matrix(R: np.ndarray) -> torch.Tensor:
        # Local equivalent of Route A matrix_to_quaternion_wxyz for refinement only.
        R = np.asarray(R, dtype=np.float64)
        tr = float(np.trace(R))
        if tr > 0.0:
            sc = math.sqrt(tr + 1.0) * 2.0
            q = [0.25*sc, (R[2,1]-R[1,2])/sc, (R[0,2]-R[2,0])/sc, (R[1,0]-R[0,1])/sc]
        else:
            ii = int(np.argmax(np.diag(R)))
            if ii == 0:
                sc = math.sqrt(max(1.0+R[0,0]-R[1,1]-R[2,2],1e-16))*2.0
                q = [(R[2,1]-R[1,2])/sc,0.25*sc,(R[0,1]+R[1,0])/sc,(R[0,2]+R[2,0])/sc]
            elif ii == 1:
                sc = math.sqrt(max(1.0+R[1,1]-R[0,0]-R[2,2],1e-16))*2.0
                q = [(R[0,2]-R[2,0])/sc,(R[0,1]+R[1,0])/sc,0.25*sc,(R[1,2]+R[2,1])/sc]
            else:
                sc = math.sqrt(max(1.0+R[2,2]-R[0,0]-R[1,1],1e-16))*2.0
                q = [(R[1,0]-R[0,1])/sc,(R[0,2]+R[2,0])/sc,(R[1,2]+R[2,1])/sc,0.25*sc]
        out = torch.as_tensor(q, device=scene.robot.device, dtype=scene.command.dtype)
        return out / torch.linalg.vector_norm(out)

    def exact_pose_errors(target_matrix: np.ndarray) -> tuple[float, float]:
        body = scene.robot.data.body_pose_w[0, scene.flange_id]
        target_position = torch.as_tensor(target_matrix[:3,3], device=body.device, dtype=body.dtype)
        target_quat = _quat_from_matrix(target_matrix[:3,:3]).to(device=body.device, dtype=body.dtype)
        pos_mm = 1000.0 * float(torch.linalg.vector_norm(body[:3]-target_position))
        dot = torch.abs(torch.sum(body[3:7] * target_quat)).clamp(0.0,1.0)
        rot_deg = float(torch.rad2deg(2.0*torch.acos(dot)))
        return pos_mm, rot_deg

    def refine_exact_stage(stage_name: str, desired_q7: np.ndarray) -> tuple[bool, dict[str, float]]:
        # Copy Route A endpoint-refinement semantics without touching Route A code.
        configured = set(scene.config.get("endpoint_refinement_stages", ["cover", "grasp"]))
        if stage_name not in configured:
            return True, {"position_error_mm": 0.0, "orientation_error_deg": 0.0}
        target_matrix = arm_target_world[arm_target_index[stage_name]]
        desired = torch.as_tensor(desired_q7, device=scene.robot.device, dtype=scene.command.dtype).reshape(1,7)
        settings = scene.config["endpoint_refinement"]
        bias = scene.command[:, scene.arm_ids].clone() - desired
        gain = float(settings["integral_gain_per_s"])
        max_bias = math.radians(float(settings["max_command_bias_deg"]))
        max_steps = max(1, round(float(settings["max_duration_s"]) / scene.dt))
        stable_required = max(1, round(float(settings["stable_duration_s"]) / scene.dt))
        stable = 0
        lower = scene.robot.data.soft_joint_pos_limits[:, scene.arm_ids, 0]
        upper = scene.robot.data.soft_joint_pos_limits[:, scene.arm_ids, 1]
        pos_limit = float(scene.config["stage_tolerances"]["contact_position_mm"])
        rot_limit = float(scene.config["stage_tolerances"]["contact_orientation_deg"])
        print(f"[Route B][REFINE] {stage_name.upper()} target <= {pos_limit:.1f}mm/{rot_limit:.1f}deg", flush=True)
        for i in range(max_steps):
            actual = scene.robot.data.joint_pos[:, scene.arm_ids]
            bias = torch.clamp(bias + gain * (desired-actual) * scene.dt, -max_bias, max_bias)
            scene.command[:, scene.arm_ids] = torch.maximum(torch.minimum(desired+bias, upper), lower)
            scene.current_desired_arm = desired
            step_trace(stage_name.upper()+"_REFINE", (i+1)/max_steps)
            pos_err, rot_err = exact_pose_errors(target_matrix)
            if pos_err <= pos_limit and rot_err <= rot_limit:
                stable += 1
                if stable >= stable_required:
                    print(f"[Route B][REFINE] PASS {stage_name.upper()} {pos_err:.2f}mm/{rot_err:.2f}deg", flush=True)
                    return True, {"position_error_mm": pos_err, "orientation_error_deg": rot_err}
            else:
                stable = 0
        pos_err, rot_err = exact_pose_errors(target_matrix)
        print(f"[Route B][REFINE] FAIL {stage_name.upper()} {pos_err:.2f}mm/{rot_err:.2f}deg", flush=True)
        return False, {"position_error_mm": pos_err, "orientation_error_deg": rot_err}

    scene.play()
    status = "FAIL"
    verified_lift_mm = None
    object_lift_pass = False
    empty_grasp_reason = None
    final_in_green = False
    try:
        failure_stage = "FORM_PREGRASP"
        hand_segment(
            "FORM_PREGRASP",
            hand_q[hand_index["pregrasp"]],
            float(durations["form_pregrasp"]),
        )

        failure_stage = "CURRENT_TO_PREGRASP"
        dense_arm(
            "CURRENT_TO_PREGRASP",
            Path(segments["CURRENT_TO_PREGRASP"]["trajectory"]),
        )

        # Planner used open/pregrasp hand while approaching the intended contact.
        failure_stage = "PREGRASP_TO_COVER"
        dense_arm(
            "PREGRASP_TO_COVER",
            Path(segments["PREGRASP_TO_COVER"]["trajectory"]),
        )
        hold_state("COVER_SETTLE", float(durations.get("cover_hold", 0.15)))

        failure_stage = "COVER_HAND"
        print("[Route B][HAND] COVER", flush=True)
        hand_segment(
            "COVER_HAND",
            hand_q[hand_index["cover"]],
            float(durations["cover"]),
        )
        cover_ok, cover_refine = refine_exact_stage("cover", np.asarray(manifest["selected_chain"]["q_cover_rad"], dtype=np.float64) if "q_cover_rad" in manifest.get("selected_chain", {}) else _load_dense(Path(segments["PREGRASP_TO_COVER"]["trajectory"]))["q_rad"][-1])
        if not cover_ok:
            raise RuntimeError(f"COVER exact endpoint failed: {cover_refine}")

        failure_stage = "GRASP"
        print("[Route B][HAND] GRASP", flush=True)
        hand_segment(
            "GRASP",
            hand_q[hand_index["grasp"]],
            float(durations["grasp"]),
        )
        grasp_q = _load_dense(Path(segments["PREGRASP_TO_COVER"]["trajectory"]))["q_rad"][-1]
        grasp_ok, grasp_refine = refine_exact_stage("grasp", grasp_q)
        if not grasp_ok:
            raise RuntimeError(f"GRASP exact endpoint failed: {grasp_refine}")

        failure_stage = "SQUEEZE"
        print("[Route B][HAND] SQUEEZE 41-point", flush=True)
        squeeze_duration = float(durations["squeeze"])
        steps_each = max(
            1,
            round(
                squeeze_duration
                / max(1, len(squeeze_dense) - 1)
                / scene.dt
            ),
        )
        for path_index in range(1, len(squeeze_dense)):
            start = scene.command[:, hand_ids].clone()
            goal = torch.as_tensor(
                squeeze_dense[path_index],
                device=scene.robot.device,
                dtype=scene.command.dtype,
            ).reshape(1, 20)
            for local in range(steps_each):
                alpha = quintic((local + 1) / steps_each)
                scene.command[:, hand_ids] = start + alpha * (goal - start)
                step_trace(
                    "SQUEEZE",
                    (
                        path_index
                        - 1
                        + (local + 1) / steps_each
                    )
                    / max(1, len(squeeze_dense) - 1),
                )

        failure_stage = "COVER_TO_LIFT"
        dense_arm(
            "COVER_TO_LIFT",
            Path(segments["COVER_TO_LIFT"]["trajectory"]),
        )
        hold_state("LIFT_HOLD", float(durations.get("lift_hold", 0.4)))
        verified_lift_mm = target_lift_mm()
        lift_threshold_mm = float(scene.config.get("object_lift_pass_mm", 30.0))
        object_lift_pass = bool(verified_lift_mm >= lift_threshold_mm)
        if not object_lift_pass:
            empty_grasp_reason = (
                f"EMPTY_GRASP: lift={verified_lift_mm:.1f}mm < {lift_threshold_mm:.1f}mm"
            )
            print(
                "[Route B][VERIFY] EMPTY_GRASP "
                f"lift={verified_lift_mm:.1f}mm < {lift_threshold_mm:.1f}mm; "
                "continuing route-completion smoke execution",
                flush=True,
            )
        else:
            print(f"[Route B][VERIFY] PASS lift={verified_lift_mm:.1f}mm >= {lift_threshold_mm:.1f}mm", flush=True)

        failure_stage = "LIFT_TO_TRANSFER"
        dense_arm(
            "LIFT_TO_TRANSFER",
            Path(segments["LIFT_TO_TRANSFER"]["trajectory"]),
        )
        failure_stage = "TRANSFER_TO_PLACE"
        dense_arm(
            "TRANSFER_TO_PLACE",
            Path(segments["TRANSFER_TO_PLACE"]["trajectory"]),
        )
        hold_state("PLACE_HOLD", float(durations.get("place_hold", 0.4)))

        failure_stage = "RELEASE"
        print("[Route B][HAND] RELEASE", flush=True)
        hand_segment(
            "RELEASE",
            hand_q[hand_index["pregrasp"]],
            float(durations["release"]),
        )
        hold_state(
            "RELEASE_HOLD", float(durations.get("release_hold", 0.4))
        )

        failure_stage = "PLACE_TO_RETREAT"
        dense_arm(
            "PLACE_TO_RETREAT",
            Path(segments["PLACE_TO_RETREAT"]["trajectory"]),
        )
        failure_stage = "RETREAT_TO_HOME"
        dense_arm(
            "RETREAT_TO_HOME",
            Path(segments["RETREAT_TO_HOME"]["trajectory"]),
        )

        final_position = (
            target.data.root_pos_w[0].detach().cpu().numpy().astype(np.float64)
        )
        layout = json.loads(
            (
                project_root
                / "08_dual_arm_scene_layout/config/manual_layout_calibrated.json"
            ).read_text(encoding="utf-8")
        )
        center = np.asarray(
            layout["transforms"]["placement_zone"]["position_world_m"],
            dtype=np.float64,
        )
        size = np.asarray(
            layout["geometry"]["placement_zone_size_m"], dtype=np.float64
        )
        lower = center[:2] - 0.5 * size[:2]
        upper = center[:2] + 0.5 * size[:2]
        edge = float(scene.config.get("placement_center_edge_margin_m", 0.01))
        final_in_green = bool(
            np.all(final_position[:2] >= lower + edge)
            and np.all(final_position[:2] <= upper - edge)
        )
        status = "PASS" if (object_lift_pass and final_in_green) else "FAIL"
        print(f"[Route B][VERIFY] final green zone = {final_in_green}", flush=True)
        if status == "PASS":
            failure_stage = None
        elif not object_lift_pass:
            failure_stage = "EMPTY_GRASP"
        else:
            failure_stage = "FINAL_GREEN_ZONE"
    except Exception as exc:
        final_position = (
            target.data.root_pos_w[0].detach().cpu().numpy().astype(np.float64)
        )
        failure_reason = f"{type(exc).__name__}: {exc}"
        recovery_status, recovery_error = recover_home_after_routeb_failure()
        if recovery_status == "HOME":
            status = "RECOVERED_FAIL"
        else:
            status = "FAIL"
            failure_reason += f" | recovery failed: {recovery_error}"
    else:
        failure_reason = None
    finally:
        scene.pause()

    trace_path = output / "trace_routeB.csv"
    with trace_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                "time_s",
                "state",
                "progress",
                "object_lift_mm",
                "max_arm_qdot_rad_s",
                "max_arm_joint_goal_error_deg",
            ],
        )
        writer.writeheader()
        writer.writerows(trace)

    report = {
        "schema_version": 1,
        "route": "RouteB",
        "status": status,
        "candidate_index": candidate_index,
        "target_segmentation_id": int(target_segmentation_id),
        "failure_stage": failure_stage,
        "failure_reason": failure_reason,
        "recovery_status": locals().get("recovery_status"),
        "max_object_lift_mm": float(max_object_lift_mm),
        "verify_lift_mm": (
            None if verified_lift_mm is None else float(verified_lift_mm)
        ),
        "object_lift_pass": bool(object_lift_pass),
        "empty_grasp_reason": empty_grasp_reason,
        "final_object_position_world_m": final_position.tolist(),
        "final_object_center_inside_green_zone": bool(final_in_green),
        "action_simulation_time_s": float(sim_time),
        "arm_dense_execution": "piecewise-linear in cuRobo time; no quintic arm regeneration",
        "cover_refinement": locals().get("cover_refine"),
        "grasp_refinement": locals().get("grasp_refine"),
        "trace_csv": str(trace_path),
    }
    report_path = output / "report_routeB.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    scene.execute_count += 1
    scene.write_snapshot(output / "scene_after_execution.json")
    print("========================================", flush=True)
    print(f" Route B FULL EXECUTION {status}", flush=True)
    print("========================================", flush=True)
    print(f"candidate       : {candidate_index}", flush=True)
    print(f"failure stage   : {failure_stage}", flush=True)
    print(f"verified lift   : {verified_lift_mm} mm", flush=True)
    print(f"max lift        : {max_object_lift_mm:.1f} mm", flush=True)
    print(f"green zone      : {final_in_green}", flush=True)
    print(f"report          : {report_path}", flush=True)
    print("========================================", flush=True)
    return {**report, "report": str(report_path)}
