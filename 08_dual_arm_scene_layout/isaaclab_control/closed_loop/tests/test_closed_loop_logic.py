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
    def test_routeb_goal_pool_merge_keeps_all_candidates(self):
        orchestrator = load_module(
            "closed_loop_orchestrator_routeb_pool_merge",
            CLOSED_LOOP / "orchestrator.py",
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sources = []
            for batch, candidates in enumerate(([8, 3], [11]), start=1):
                source = root / f"pool_{batch}.npz"
                count = len(candidates)
                np.savez_compressed(
                    source,
                    arm_joint_names=np.asarray(["joint_1", "joint_2"]),
                    q_current_rad=np.asarray([0.1, 0.2], dtype=np.float32),
                    candidate_index=np.asarray(candidates, dtype=np.int64),
                    candidate_order=np.arange(count, dtype=np.int64),
                    case_root=np.asarray([f"case_{value}" for value in candidates]),
                )
                sources.append(source)
            output, report = orchestrator.concatenate_routeb_goal_pools(
                sources,
                root / "merged.npz",
            )
            with np.load(output, allow_pickle=False) as merged:
                self.assertEqual(merged["candidate_index"].tolist(), [8, 3, 11])
                self.assertEqual(merged["candidate_order"].tolist(), [0, 1, 2])
                np.testing.assert_allclose(merged["q_current_rad"], [0.1, 0.2])
            self.assertEqual(json.loads(report.read_text())["candidate_count"], 3)

    def test_color_candidates_bind_seed_labels_to_object_metadata(self):
        binding = load_module(
            "color_sort_candidate_binding",
            CLOSED_LOOP / "color_sort_dino_sam/candidate_binding.py",
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            prediction = root / "prediction.npz"
            np.savez_compressed(
                prediction,
                seed_target_label=np.asarray([2, 1], dtype=np.int64),
            )
            catalog = root / "catalog.json"
            catalog.write_text(json.dumps({"objects": [
                {
                    "target_label": label,
                    "target_id": f"red_sam_{label:03d}",
                    "target_grasp_mask_path": f"grasp_{label}.npy",
                    "target_seed_mask_path": f"seed_{label}.npy",
                    "target_removal_mask_path": f"removal_{label}.npy",
                    "target_geometry_path": f"geometry_{label}.json",
                    "perception_target_json": f"target_{label}.json",
                }
                for label in (1, 2)
            ]}), encoding="utf-8")
            rows = binding.enrich_candidate_rows(
                rows=[{"candidate_index": 0}, {"candidate_index": 1}],
                prediction_path=prediction,
                catalog_path=catalog,
            )
            self.assertEqual([row["target_label"] for row in rows], [2, 1])
            self.assertEqual(rows[0]["target_id"], "red_sam_002")
            self.assertEqual(rows[1]["target_geometry_path"], "geometry_1.json")

    def test_only_exhausted_capture_planning_is_recoverable(self):
        orchestrator = load_module(
            "closed_loop_orchestrator_capture_recovery",
            CLOSED_LOOP / "orchestrator.py",
        )
        self.assertTrue(
            orchestrator.is_recoverable_color_planning_failure(
                "Route B exhausted all front-half goals before full plan PASS"
            )
        )
        for fatal in (
            "CUDA is not visible",
            "checkpoint missing",
            "seed/input mismatch",
            "Official DGN2 LEAP inference failed without diagnostic",
        ):
            self.assertFalse(
                orchestrator.is_recoverable_color_planning_failure(fatal)
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
