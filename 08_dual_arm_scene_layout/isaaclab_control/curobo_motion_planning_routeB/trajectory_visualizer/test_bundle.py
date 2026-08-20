import tempfile
import unittest
from pathlib import Path
import numpy as np

from trajectory_visualizer.bundle import (
    RIGHT_ARM_JOINTS,
    VisualizationBundle,
    load_bundle,
    save_bundle,
)


class BundleTest(unittest.TestCase):
    def test_roundtrip(self):
        n, s, p = 41, 12, 100
        b = VisualizationBundle(
            scene_points_base=np.zeros((p, 3)),
            sphere_centers_base=np.zeros((n, s, 3)),
            sphere_radii_m=np.full(s, 0.02),
            sphere_link_names=np.array([f"link_{i}" for i in range(s)]),
            sphere_active_mask=np.array([i < 5 for i in range(s)]),
            ee_positions_base=np.zeros((n, 3)),
            time_s=np.arange(n) * 0.025,
            q_rad=np.zeros((n, 7)),
            joint_names=np.array(RIGHT_ARM_JOINTS),
            frame_min_clearance_m=np.full(n, 0.05),
            frame_worst_sphere_index=np.zeros(n, dtype=int),
        )
        with tempfile.TemporaryDirectory() as d:
            pth = save_bundle(Path(d) / "bundle.npz", b)
            r = load_bundle(pth)
            self.assertEqual(r.sphere_centers_base.shape, (41, 12, 3))
            self.assertEqual(r.q_rad.shape, (41, 7))


if __name__ == "__main__":
    unittest.main()
