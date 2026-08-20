"""Client for one persistent Isaac Lab/Isaac Sim subprocess.

The worker owns AppLauncher/SimulationContext for the whole interactive session.
The orchestrator only exchanges line-delimited JSON commands, mirroring the
project's already-proven persistent cuRobo worker pattern.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import queue
import signal
import subprocess
import threading
import time


_PROTOCOL = "__ISAAC_SESSION__"


def _jsonable(value):
    try:
        import numpy as np
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, (np.integer, np.floating, np.bool_)):
            return value.item()
    except Exception:
        pass
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


class PersistentIsaacClient:
    """Start Isaac once; reuse the same physical world for every cycle."""

    def __init__(
        self,
        *,
        project_root: Path | str,
        scene_manifest: Path | str,
        runtime_config: Path | str,
        startup_timeout_s: float = 240.0,
        request_timeout_s: float = 180.0,
        headless: bool = False,
        verbose: bool = False,
        log_callback=None,
        task: str = "semantic-grasp",
        color_seed: int = 42,
        color_assignment: Path | str | None = None,
        scene_migration_audit: bool = False,
        task_object_collision_policy: str = "persistent_filtered",
    ) -> None:
        self.project_root = Path(project_root).expanduser().resolve()
        self.scene_manifest = Path(scene_manifest).expanduser().resolve()
        self.runtime_config = Path(runtime_config).expanduser().resolve()
        self.request_timeout_s = float(request_timeout_s)
        self.verbose = bool(verbose)
        self.log_callback = log_callback
        launcher = (
            self.project_root
            / "08_dual_arm_scene_layout/isaaclab_control/closed_loop/launchers/run_persistent_isaac_worker.sh"
        )
        for path in (launcher, self.scene_manifest, self.runtime_config):
            if not path.is_file():
                raise FileNotFoundError(path)
        cmd = [
            "bash", str(launcher),
            "--project-root", str(self.project_root),
            "--scene-manifest", str(self.scene_manifest),
            "--config", str(self.runtime_config),
            "--task", str(task),
            "--color-seed", str(int(color_seed)),
            "--stdio",
        ]
        if color_assignment is not None:
            cmd.extend(["--color-assignment", str(Path(color_assignment).expanduser().resolve())])
        if scene_migration_audit:
            cmd.append("--scene-migration-audit")
            cmd.extend(["--task-object-collision-policy", str(task_object_collision_policy)])
        elif task_object_collision_policy != "persistent_filtered":
            raise ValueError("task_object_collision_policy requires scene_migration_audit=True")
        if headless:
            cmd.append("--headless")
        env = dict(os.environ)
        env["PYTHONUNBUFFERED"] = "1"
        self.proc = subprocess.Popen(
            cmd,
            cwd=self.project_root,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            start_new_session=True,
            env=env,
        )
        self._responses: queue.Queue[dict] = queue.Queue()
        self._stdout_thread = threading.Thread(target=self._stdout_loop, daemon=True)
        self._stdout_thread.start()
        self.request({"op": "ping"}, timeout=float(startup_timeout_s))

    def _log(self, text: str) -> None:
        if self.log_callback is not None:
            try:
                self.log_callback(text)
            except Exception:
                pass
        if self.verbose:
            print(f"[isaac] {text}", flush=True)

    def _stdout_loop(self) -> None:
        assert self.proc.stdout is not None
        for raw in self.proc.stdout:
            line = raw.rstrip("\n")
            if line.startswith(_PROTOCOL):
                try:
                    self._responses.put(json.loads(line[len(_PROTOCOL):]))
                except Exception as exc:
                    self._responses.put({"ok": False, "error": f"Isaac protocol decode failed: {exc}"})
            elif line:
                self._log(line)

    def request(self, payload: dict, timeout: float | None = None) -> dict:
        proc = getattr(self, "proc", None)
        if proc is None:
            raise RuntimeError("persistent Isaac worker is closed")
        if proc.poll() is not None:
            raise RuntimeError(f"persistent Isaac worker exited with code {proc.returncode}")
        assert proc.stdin is not None
        proc.stdin.write(json.dumps(_jsonable(payload), separators=(",", ":"), ensure_ascii=False) + "\n")
        proc.stdin.flush()
        wait_s = self.request_timeout_s if timeout is None else float(timeout)
        deadline = time.monotonic() + wait_s
        response = None
        while response is None:
            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                raise TimeoutError(f"persistent Isaac request timed out after {wait_s:.1f}s")
            if proc.poll() is not None:
                raise RuntimeError(f"persistent Isaac worker exited with code {proc.returncode}")
            try:
                response = self._responses.get(timeout=min(0.5, remaining))
            except queue.Empty:
                continue
        if not bool(response.get("ok", False)):
            raise RuntimeError(response.get("error", "unknown persistent Isaac error"))
        return response

    def capture(self, output_dir: Path | str, *, hold_s: float | None = None) -> dict:
        payload = {"op": "capture", "output_dir": str(Path(output_dir).expanduser().resolve())}
        if hold_s is not None:
            payload["hold_s"] = float(hold_s)
        return self.request(payload)

    def execute(
        self,
        *,
        case_root: Path | str,
        plan_npz: Path | str,
        output_dir: Path | str,
        target_segmentation_id: int,
    ) -> dict:
        return self.request({
            "op": "execute",
            "case_root": str(Path(case_root).expanduser().resolve()),
            "plan_npz": str(Path(plan_npz).expanduser().resolve()),
            "output_dir": str(Path(output_dir).expanduser().resolve()),
            "target_segmentation_id": int(target_segmentation_id),
        }, timeout=max(self.request_timeout_s, 300.0))

    def execute_routeB(
        self,
        *,
        manifest_path: Path | str,
        output_dir: Path | str,
        target_segmentation_id: int,
    ) -> dict:
        return self.request({
            "op": "execute_routeB",
            "manifest_path": str(Path(manifest_path).expanduser().resolve()),
            "output_dir": str(Path(output_dir).expanduser().resolve()),
            "target_segmentation_id": int(target_segmentation_id),
        }, timeout=max(self.request_timeout_s, 600.0))

    def snapshot(self, output_path: Path | str) -> dict:
        return self.request({"op": "snapshot", "output": str(Path(output_path).expanduser().resolve())})

    def close(self) -> None:
        proc = getattr(self, "proc", None)
        if proc is None:
            return
        if proc.poll() is None:
            try:
                self.request({"op": "shutdown"}, timeout=15.0)
            except Exception:
                pass
            try:
                # The launcher owns an Isaac/Kit child in the same dedicated
                # session.  A protocol-level shutdown can return before Kit has
                # released CUDA, so wait first and then signal the *group*, not
                # merely the launcher parent.
                proc.wait(timeout=10.0)
            except Exception:
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGINT)
                    proc.wait(timeout=5.0)
                except Exception:
                    try:
                        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                        proc.wait(timeout=5.0)
                    except Exception:
                        pass
        self.proc = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
