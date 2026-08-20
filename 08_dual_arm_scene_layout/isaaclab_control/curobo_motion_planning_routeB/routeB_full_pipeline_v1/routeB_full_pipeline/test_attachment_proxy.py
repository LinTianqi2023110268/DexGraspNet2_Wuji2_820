import json
import tempfile
import unittest
from pathlib import Path
import numpy as np

from routeB_full_pipeline.attachment_proxy import build_target_proxy_from_capture


class AttachmentProxyTest(unittest.TestCase):
    def test_aabb_proxy(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            layout = root / "08_dual_arm_scene_layout/config"
            layout.mkdir(parents=True)
            (layout / "manual_layout_calibrated.json").write_text(
                json.dumps(
                    {
                        "transforms": {
                            "dual_arm_mount": {
                                "Gf_local_to_world_row_major": np.eye(4).T.tolist()
                            }
                        }
                    }
                ),
                encoding="utf-8",
            )
            capture = root / "capture"
            capture.mkdir()
            (capture / "object_physics_audit.json").write_text(
                json.dumps(
                    {
                        "objects": [
                            {
                                "segmentation_id": 3,
                                "collision_aabb_world_min_m": [0, 0, 0],
                                "collision_aabb_world_max_m": [0.1, 0.2, 0.3],
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            p = build_target_proxy_from_capture(
                project_root=root,
                capture_dir=capture,
                target_segmentation_id=3,
                padding_m=0.0,
                minimum_dim_m=0.0,
            )
            self.assertTrue(np.allclose(p.dims_base_m, [0.1, 0.2, 0.3]))
            self.assertTrue(np.allclose(p.center_base_m, [0.05, 0.1, 0.15]))


if __name__ == "__main__":
    unittest.main()
