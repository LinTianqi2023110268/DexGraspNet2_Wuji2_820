#!/usr/bin/env python3
"""Generate robot-filtered planning depth from an existing RGB-D capture.

This module is intentionally standalone.  It does not change the current
baseline planner, DGN2, LEAP->Wuji2 retargeting, or Isaac execution path.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
import torch

from curobo.perception import RobotSegmenter
from curobo.types import CameraObservation, JointState, Pose


PROJECT_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_ROBOT_FILE = (
    PROJECT_ROOT
    / "08_dual_arm_scene_layout/isaaclab_control/core/generated/dual_arm_right_wuji2_curobo.yml"
)
DEFAULT_LAYOUT_JSON = PROJECT_ROOT / "08_dual_arm_scene_layout/config/manual_layout_calibrated.json"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).resolve().read_text(encoding="utf-8"))


def world_from_base(layout_json: Path = DEFAULT_LAYOUT_JSON) -> np.ndarray:
    layout = load_json(layout_json)
    # Stored as OpenUSD row-vector Gf.Matrix4d.  Project math uses column-vector
    # T_A_B convention, so transpose before use.
    return np.asarray(
        layout["transforms"]["dual_arm_mount"]["Gf_local_to_world_row_major"],
        dtype=np.float64,
    ).T


def depth_preview(depth: np.ndarray, *, mask: np.ndarray | None = None) -> Image.Image:
    valid = np.isfinite(depth) & (depth > 0.0)
    if not np.any(valid):
        base = np.zeros(depth.shape, dtype=np.uint8)
    else:
        lo, hi = np.percentile(depth[valid], [2.0, 98.0])
        if hi <= lo:
            hi = lo + 1.0e-6
        base = np.clip((depth - lo) / (hi - lo), 0.0, 1.0)
        base = (255.0 * base).astype(np.uint8)
        base[~valid] = 0
    rgb = np.repeat(base[..., None], 3, axis=2)
    if mask is not None:
        m = np.asarray(mask, dtype=bool)
        rgb[m, 0] = 255
        rgb[m, 1] = (0.35 * rgb[m, 1]).astype(np.uint8)
        rgb[m, 2] = (0.35 * rgb[m, 2]).astype(np.uint8)
    return Image.fromarray(rgb)


def rgb_without_robot_preview(rgb: np.ndarray, robot_mask: np.ndarray) -> tuple[Image.Image, Image.Image]:
    """Create derived RGB artifacts without altering the raw capture.

    A neutral fill deliberately avoids inventing texture with inpainting.  The
    overlay is an audit image only; neither image changes the authoritative
    depth/robot mask contract.
    """
    image = np.asarray(rgb, dtype=np.uint8)
    mask = np.asarray(robot_mask, dtype=bool)
    if image.ndim != 3 or image.shape[2] != 3 or image.shape[:2] != mask.shape:
        raise ValueError(f"RGB/robot mask shape mismatch: {image.shape} vs {mask.shape}")
    no_robot = image.copy()
    no_robot[mask] = np.asarray([127, 127, 127], dtype=np.uint8)
    overlay = image.astype(np.float32)
    overlay[mask] = 0.45 * overlay[mask] + 0.55 * np.asarray([255, 40, 40], dtype=np.float32)
    return Image.fromarray(no_robot), Image.fromarray(np.clip(overlay, 0, 255).astype(np.uint8))


def pose_from_matrix(matrix: np.ndarray, *, device: torch.device) -> Pose:
    tensor = torch.as_tensor(matrix, dtype=torch.float32, device=device)
    return Pose.from_matrix(tensor.reshape(1, 4, 4))


def camera_observation_from_capture(
    *,
    depth_m: np.ndarray,
    intrinsics: np.ndarray,
    T_world_camera: np.ndarray,
    T_world_base: np.ndarray,
    device: torch.device,
) -> CameraObservation:
    if depth_m.ndim != 2:
        raise ValueError(f"depth_m.npy must be [H,W], got {depth_m.shape}")
    if intrinsics.shape != (3, 3):
        raise ValueError(f"intrinsics.npy must be [3,3], got {intrinsics.shape}")
    if T_world_camera.shape != (4, 4):
        raise ValueError(f"T_world_camera.npy must be [4,4], got {T_world_camera.shape}")
    if T_world_base.shape != (4, 4):
        raise ValueError(f"T_world_base must be [4,4], got {T_world_base.shape}")

    T_base_camera = np.linalg.inv(T_world_base) @ T_world_camera
    depth = torch.as_tensor(depth_m, dtype=torch.float32, device=device).unsqueeze(0)
    K = torch.as_tensor(intrinsics, dtype=torch.float32, device=device).unsqueeze(0)
    return CameraObservation(
        name="persistent_capture",
        depth_image=depth,
        intrinsics=K,
        pose=pose_from_matrix(T_base_camera, device=device),
        depth_to_meter=1.0,
        resolution=[int(depth_m.shape[1]), int(depth_m.shape[0])],
    )


class RobotDepthCleaner:
    """Thin wrapper around cuRobo V2 ``RobotSegmenter``."""

    def __init__(
        self,
        *,
        robot_file: Path = DEFAULT_ROBOT_FILE,
        distance_threshold_m: float = 0.05,
        collision_sphere_buffer_m: float | None = None,
        use_cuda_graph: bool = False,
        device: str = "cuda:0",
    ) -> None:
        self.robot_file = Path(robot_file).expanduser().resolve()
        if not self.robot_file.is_file():
            raise FileNotFoundError(self.robot_file)
        self.device = torch.device(device)
        self.segmenter = RobotSegmenter.from_robot_file(
            robot_file=str(self.robot_file),
            collision_sphere_buffer=collision_sphere_buffer_m,
            distance_threshold=float(distance_threshold_m),
            use_cuda_graph=bool(use_cuda_graph),
        )
        # cuRobo 0.8.x RobotSegmenter defaults ops_dtype to bfloat16, while the
        # active tensor checker path for collision spheres requires float32.
        # Keep this as an adapter compatibility fix; do not patch cuRobo itself.
        self.segmenter._ops_dtype = torch.float32
        self.active_joint_names = [str(name) for name in self.segmenter.kinematics.joint_names]

    def joint_state_from_robot_state(self, robot_state: dict[str, Any]) -> JointState:
        measured = {
            str(name): float(value)
            for name, value in robot_state["joint_positions_by_name"].items()
        }
        missing = [name for name in self.active_joint_names if name not in measured]
        if missing:
            raise KeyError(
                "robot_state.json does not contain all cuRobo active joints: "
                + ", ".join(missing)
            )
        q = torch.as_tensor(
            [measured[name] for name in self.active_joint_names],
            dtype=torch.float32,
            device=self.device,
        ).reshape(1, -1)
        return JointState.from_position(q, joint_names=self.active_joint_names)

    def remove_robot(
        self,
        *,
        depth_m: np.ndarray,
        intrinsics: np.ndarray,
        T_world_camera: np.ndarray,
        T_world_base: np.ndarray,
        robot_state: dict[str, Any],
    ) -> dict[str, np.ndarray]:
        camera_obs = camera_observation_from_capture(
            depth_m=depth_m,
            intrinsics=intrinsics,
            T_world_camera=T_world_camera,
            T_world_base=T_world_base,
            device=self.device,
        )
        joint_state = self.joint_state_from_robot_state(robot_state)
        mask, filtered_depth = self.segmenter.get_robot_mask(camera_obs, joint_state)
        return {
            "robot_mask": mask.squeeze(0).detach().cpu().numpy().astype(bool),
            "filtered_depth": filtered_depth.squeeze(0).detach().cpu().numpy().astype(np.float32),
        }


def run_capture_robot_segmentation(
    *,
    capture_dir: Path,
    robot_file: Path = DEFAULT_ROBOT_FILE,
    layout_json: Path = DEFAULT_LAYOUT_JSON,
    output_dir: Path | None = None,
    distance_threshold_m: float = 0.05,
    collision_sphere_buffer_m: float | None = None,
    use_cuda_graph: bool = False,
    device: str = "cuda:0",
) -> dict[str, Any]:
    capture_dir = Path(capture_dir).expanduser().resolve()
    output_dir = capture_dir / "planning" if output_dir is None else Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    required = {
        "rgb": capture_dir / "rgb.png",
        "depth_m": capture_dir / "depth_m.npy",
        "intrinsics": capture_dir / "intrinsics.npy",
        "T_world_camera": capture_dir / "T_world_camera.npy",
        "robot_state": capture_dir / "robot_state.json",
    }
    missing = [str(path) for path in required.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing capture inputs: " + ", ".join(missing))

    depth_m = np.load(required["depth_m"])
    intrinsics = np.load(required["intrinsics"])
    T_world_camera = np.load(required["T_world_camera"])
    robot_state = load_json(required["robot_state"])
    T_world_base = world_from_base(layout_json)

    cleaner = RobotDepthCleaner(
        robot_file=robot_file,
        distance_threshold_m=distance_threshold_m,
        collision_sphere_buffer_m=collision_sphere_buffer_m,
        use_cuda_graph=use_cuda_graph,
        device=device,
    )
    result = cleaner.remove_robot(
        depth_m=depth_m,
        intrinsics=intrinsics,
        T_world_camera=T_world_camera,
        T_world_base=T_world_base,
        robot_state=robot_state,
    )

    robot_mask = result["robot_mask"]
    filtered_depth = result["filtered_depth"]
    rgb = np.asarray(Image.open(required["rgb"]).convert("RGB"), dtype=np.uint8)
    rgb_no_robot, robot_overlay = rgb_without_robot_preview(rgb, robot_mask)
    np.save(output_dir / "robot_mask.npy", robot_mask)
    np.save(output_dir / "filtered_depth.npy", filtered_depth)
    Image.fromarray((255 * robot_mask.astype(np.uint8))).save(output_dir / "robot_mask.png")
    rgb_no_robot.save(output_dir / "rgb_no_robot.png")
    robot_overlay.save(output_dir / "robot_mask_overlay.png")
    depth_preview(filtered_depth, mask=robot_mask).save(output_dir / "filtered_depth_preview.png")

    valid_input = np.isfinite(depth_m) & (depth_m > 0.0)
    valid_filtered = np.isfinite(filtered_depth) & (filtered_depth > 0.0)
    report = {
        "schema_version": 1,
        "status": "PASS",
        "backend": "curobo.perception.RobotSegmenter",
        "capture_dir": str(capture_dir),
        "robot_file": str(Path(robot_file).expanduser().resolve()),
        "layout_json": str(Path(layout_json).expanduser().resolve()),
        "output_dir": str(output_dir),
        "distance_threshold_m": float(distance_threshold_m),
        "collision_sphere_buffer_m": collision_sphere_buffer_m,
        "use_cuda_graph": bool(use_cuda_graph),
        "device": str(device),
        "base_link": str(cleaner.segmenter.base_link),
        "active_joint_count": int(len(cleaner.active_joint_names)),
        "active_joint_names": cleaner.active_joint_names,
        "robot_state_joint_count": int(robot_state.get("joint_count", len(robot_state.get("joint_positions_by_name", {})))),
        "depth_shape_hw": [int(depth_m.shape[0]), int(depth_m.shape[1])],
        "rgb_shape_hwc": [int(value) for value in rgb.shape],
        "intrinsics_shape": list(intrinsics.shape),
        "T_world_camera_shape": list(T_world_camera.shape),
        "T_base_camera": (np.linalg.inv(T_world_base) @ T_world_camera).tolist(),
        "input_valid_depth_pixels": int(np.count_nonzero(valid_input)),
        "robot_mask_pixels": int(np.count_nonzero(robot_mask)),
        "robot_mask_fraction_of_valid_depth": (
            float(np.count_nonzero(robot_mask & valid_input) / max(1, np.count_nonzero(valid_input)))
        ),
        "filtered_valid_depth_pixels": int(np.count_nonzero(valid_filtered)),
        "outputs": {
            "robot_mask_npy": str(output_dir / "robot_mask.npy"),
            "robot_mask_png": str(output_dir / "robot_mask.png"),
            "rgb_no_robot_png": str(output_dir / "rgb_no_robot.png"),
            "robot_mask_overlay_png": str(output_dir / "robot_mask_overlay.png"),
            "filtered_depth_npy": str(output_dir / "filtered_depth.npy"),
            "filtered_depth_preview_png": str(output_dir / "filtered_depth_preview.png"),
            "report": str(output_dir / "robot_segmentation_report.json"),
        },
        "coordinate_contract": {
            "T_world_camera_input": "T_world_camera maps camera coordinates into layout world",
            "T_world_base": "manual_layout_calibrated.transforms.dual_arm_mount",
            "segmenter_pose": "T_base_camera = inv(T_world_base) @ T_world_camera",
            "depth": "meters, shape [H,W]; adapter adds batch dimension [1,H,W]",
            "rgb": "raw capture/rgb.png is immutable; rgb_no_robot.png fills RobotSegmenter pixels with neutral [127,127,127]",
        },
    }
    (output_dir / "robot_segmentation_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-dir", type=Path, required=True)
    parser.add_argument("--robot-file", type=Path, default=DEFAULT_ROBOT_FILE)
    parser.add_argument("--layout-json", type=Path, default=DEFAULT_LAYOUT_JSON)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--distance-threshold-m", type=float, default=0.05)
    parser.add_argument("--collision-sphere-buffer-m", type=float)
    parser.add_argument("--use-cuda-graph", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    report = run_capture_robot_segmentation(
        capture_dir=args.capture_dir,
        robot_file=args.robot_file,
        layout_json=args.layout_json,
        output_dir=args.output_dir,
        distance_threshold_m=args.distance_threshold_m,
        collision_sphere_buffer_m=args.collision_sphere_buffer_m,
        use_cuda_graph=args.use_cuda_graph,
        device=args.device,
    )
    print(json.dumps({
        "status": report["status"],
        "output_dir": report["output_dir"],
        "robot_mask_pixels": report["robot_mask_pixels"],
        "robot_mask_fraction_of_valid_depth": report["robot_mask_fraction_of_valid_depth"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
