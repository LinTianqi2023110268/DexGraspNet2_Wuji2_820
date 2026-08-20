from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[4]
CLOSED_LOOP = PROJECT_ROOT / "08_dual_arm_scene_layout/isaaclab_control/closed_loop"
RUNTIME_SCRIPTS = PROJECT_ROOT / "08_dual_arm_scene_layout/isaaclab_control/runtime/scripts"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class ClosedLoopLogicTests(unittest.TestCase):
    def test_color_matching_uses_all_legal_grounded_sam_proposals(self):
        orchestrator = load_module(
            "closed_loop_orchestrator_color_proposals",
            CLOSED_LOOP / "orchestrator.py",
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            archive_path = root / "legal_proposal_masks.npz"
            proposal_a = np.zeros((4, 5), dtype=bool)
            proposal_a[0:2, 0:2] = True
            proposal_b = np.zeros((4, 5), dtype=bool)
            proposal_b[2:4, 3:5] = True
            np.savez_compressed(
                archive_path,
                proposal_indices=np.asarray([3, 7], dtype=np.int64),
                masks=np.stack([proposal_a, proposal_b]),
            )
            instance_a_path = root / "red_000.npy"
            instance_b_path = root / "red_001.npy"
            np.save(instance_a_path, proposal_a)
            np.save(instance_b_path, proposal_b)
            result = {
                "legal_proposal_masks": str(archive_path),
                "selected_detection": 3,
                "detections": [
                    {"index": 3, "score": 0.9},
                    {"index": 7, "score": 0.7},
                ],
            }
            rows = orchestrator.match_grounded_color_proposals(
                grounded_sam_result=result,
                selected_mask_path=instance_a_path,
                hsv_instances=[
                    {"instance_id": "red_001", "mask_path": str(instance_b_path)}
                ],
            )
            self.assertEqual(rows[0]["proposal_index"], 7)
            self.assertEqual(rows[0]["instance_id"], "red_001")
            self.assertEqual(rows[0]["overlap_px"], 4)

    def test_only_target_local_dgn2_empty_result_is_recoverable(self):
        orchestrator = load_module(
            "closed_loop_orchestrator_dgn2_recovery",
            CLOSED_LOOP / "orchestrator.py",
        )
        self.assertTrue(
            orchestrator.is_recoverable_color_target_generation_failure(
                "RuntimeError: Official sampler produced no seed on the segmented target"
            )
        )
        for fatal in (
            "CUDA is not visible",
            "checkpoint missing",
            "seed/input mismatch",
            "Official DGN2 LEAP inference failed without diagnostic",
        ):
            self.assertFalse(
                orchestrator.is_recoverable_color_target_generation_failure(fatal)
            )

    def test_scene_folder_parsing_requires_manifest(self):
        orchestrator = load_module("closed_loop_orchestrator", CLOSED_LOOP / "orchestrator.py")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            scene = root / "scene_0000"
            scene.mkdir()
            (scene / "scene_manifest.json").write_text("{}", encoding="utf-8")
            self.assertEqual(orchestrator.prompt_scene(root, str(scene)), scene.resolve())
            with self.assertRaises(FileNotFoundError):
                orchestrator.prompt_scene(root, str(root / "missing"))

    def test_candidate_order_uses_descending_official_score(self):
        with tempfile.TemporaryDirectory() as tmp:
            prediction = Path(tmp) / "prediction.npz"
            np.savez_compressed(
                prediction,
                target_score_descending_candidate_index=np.asarray([2, 0, 1], dtype=np.int64),
                score=np.asarray([3.0, 1.0, 5.0], dtype=np.float32),
                graspness=np.asarray([0.2, 0.1, 0.4], dtype=np.float32),
                log_prob=np.asarray([2.0, 0.5, 3.0], dtype=np.float32),
            )
            completed = subprocess.run(
                [
                    sys.executable,
                    str(CLOSED_LOOP / "scripts/list_candidate_order.py"),
                    "--prediction",
                    str(prediction),
                    "--limit",
                    "2",
                ],
                check=True,
                text=True,
                stdout=subprocess.PIPE,
            )
            rows = json.loads(completed.stdout)
            self.assertEqual([row["candidate_index"] for row in rows], [2, 0])
            self.assertEqual([row["target_rank"] for row in rows], [0, 1])

    def test_next_scene_manifest_persists_all_final_object_poses(self):
        build_next = load_module(
            "build_next_scene_manifest",
            CLOSED_LOOP / "scripts/build_next_scene_manifest.py",
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            settled = root / "settled.json"
            replay = root / "replay.npz"
            output = root / "next.json"
            manifest = {
                "schema_version": 2,
                "world_from_source_zone": np.eye(4).tolist(),
                "objects": [
                    {"segmentation_id": 1, "pose_world_object": np.eye(4).tolist()},
                    {"segmentation_id": 2, "pose_world_object": np.eye(4).tolist()},
                ],
            }
            settled.write_text(json.dumps(manifest), encoding="utf-8")
            poses = np.asarray(
                [[[0.1, 0.2, 0.3, 1.0, 0.0, 0.0, 0.0],
                  [0.4, 0.5, 0.6, 1.0, 0.0, 0.0, 0.0]]],
                dtype=np.float32,
            )
            metadata = {"objects": [{"segmentation_id": 1}, {"segmentation_id": 2}]}
            np.savez_compressed(
                replay,
                object_pose_world_wxyz=poses,
                metadata_json=np.asarray(json.dumps(metadata)),
            )
            old_argv = sys.argv
            try:
                sys.argv = [
                    "build_next_scene_manifest.py",
                    "--settled-manifest",
                    str(settled),
                    "--physical-replay",
                    str(replay),
                    "--output",
                    str(output),
                ]
                build_next.main()
            finally:
                sys.argv = old_argv
            saved = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(saved["schema_version"], 3)
            self.assertEqual(len(saved["objects"]), 2)
            self.assertAlmostEqual(saved["objects"][1]["settled_pose_layout_world"][0][3], 0.4)

    def test_placement_second_cycle_slot_differs_after_commit(self):
        allocator = load_module("placement_allocator", RUNTIME_SCRIPTS / "placement_allocator.py")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            registry = root / "placement_registry.json"
            layout = {
                "transforms": {
                    "placement_zone": {"position_world_m": [0.0, 0.0, 0.0]},
                    "table": {"position_world_m": [0.0, 0.0, 0.0]},
                },
                "geometry": {
                    "placement_zone_size_m": [0.30, 0.20, 0.02],
                    "table_size_m": [0.60, 0.40, 0.02],
                },
            }
            policy = {
                "edge_margin_m": 0.0,
                "inter_object_clearance_m": 0.02,
                "grid_step_xy_m": [0.05, 0.05],
                "footprint_xy_mode": "oriented_aabb",
                "preferred_world_y_m": 0.0,
                "occupancy_registry": str(registry),
                "release_hand_height_above_grasp_m": 0.01,
                "transfer_clearance_m": 0.18,
                "retreat_clearance_m": 0.12,
            }
            surface = np.asarray(
                [[-0.01, -0.01, 0.0], [0.01, 0.01, 0.02]],
                dtype=np.float64,
            )
            pose = np.eye(4)
            first = allocator.allocate_placement(
                project_root=root,
                layout=layout,
                policy=policy,
                surface_points_object=surface,
                world_from_object_initial=pose,
            )
            allocator.commit_placement(
                registry,
                {
                    "placement_id": "first",
                    "footprint_world_xy_min_m": first["footprint_world_xy_min_m"],
                    "footprint_world_xy_max_m": first["footprint_world_xy_max_m"],
                },
            )
            second = allocator.allocate_placement(
                project_root=root,
                layout=layout,
                policy=policy,
                surface_points_object=surface,
                world_from_object_initial=pose,
            )
            self.assertNotEqual(
                first["object_root_place_world_m"][:2],
                second["object_root_place_world_m"][:2],
            )


if __name__ == "__main__":
    unittest.main()
