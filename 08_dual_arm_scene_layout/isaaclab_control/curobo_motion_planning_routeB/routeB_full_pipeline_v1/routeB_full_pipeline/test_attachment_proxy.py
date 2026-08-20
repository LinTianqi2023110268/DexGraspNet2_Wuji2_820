import json
import tempfile
import unittest
from pathlib import Path
import numpy as np

from routeB_full_pipeline.attachment_proxy import build_target_proxy_from_capture


class AttachmentProxyTest(unittest.TestCase):
    def test_mask_depth_proxy(self):
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
            depth = np.ones((4, 4), dtype=np.float32)
            depth[2:, 2:] = 1.2
            np.save(capture / "depth_m.npy", depth)
            np.save(capture / "intrinsics.npy", np.eye(3, dtype=np.float64))
            np.save(capture / "T_world_camera.npy", np.eye(4, dtype=np.float64))
            mask = np.zeros((4, 4), dtype=bool)
            mask[1:4, 1:4] = True
            mask_path = capture / "target_mask.npy"
            np.save(mask_path, mask)
            p = build_target_proxy_from_capture(
                project_root=root,
                capture_dir=capture,
                target_segmentation_id=3,
                target_mask_path=mask_path,
                padding_m=0.0,
                minimum_dim_m=0.0,
            )
            self.assertTrue(p.source.startswith("perception_mask_depth:"))
            self.assertTrue(np.all(p.dims_base_m >= 0.0))
            self.assertEqual(int(p.target_segmentation_id), 3)


if __name__ == "__main__":
    unittest.main()
