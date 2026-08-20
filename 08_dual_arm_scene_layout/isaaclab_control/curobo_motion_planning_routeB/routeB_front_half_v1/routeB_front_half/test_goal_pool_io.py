import tempfile
import unittest
from pathlib import Path
import numpy as np

from routeB_front_half.pregrasp_pool import (
    FrontHalfGoal,
    FrontHalfGoalPool,
    load_front_half_goal_pool,
    save_front_half_goal_pool,
)


class GoalPoolIOTest(unittest.TestCase):
    def test_roundtrip(self):
        g = FrontHalfGoal(
            case_root="/tmp/case",
            candidate_index=123,
            official_score=0.9,
            candidate_order=0,
            q_pregrasp_rad=np.zeros(7),
            q_cover_rad=np.ones(7) * 0.1,
            pregrasp_pose_world=np.eye(4),
            pair_score=1.2,
            pregrasp_target_index=3,
            pregrasp_solution_index=2,
            cover_solution_index=1,
            pregrasp_inner_limit_margin_rad=0.2,
            cover_inner_limit_margin_rad=0.3,
        )
        pool = FrontHalfGoalPool(
            goals=[g],
            case_summaries=[{"status": "PASS"}],
            q_current_rad=np.zeros(7),
        )
        with tempfile.TemporaryDirectory() as d:
            path, report = save_front_half_goal_pool(
                Path(d) / "pool.npz", pool
            )
            data = load_front_half_goal_pool(path)
            self.assertEqual(data["q_pregrasp_rad"].shape, (1, 7))
            self.assertTrue(report.is_file())


if __name__ == "__main__":
    unittest.main()
