import math
import unittest
import numpy as np

from routeB_front_half.reach_contract import (
    PASS_DIRECT,
    PASS_NEAR_REGION,
    REJECT_OUTSIDE_REACH_REGION,
    order_candidates_from_filter,
    pose_region_membership,
)


class ReachContractTest(unittest.TestCase):
    def test_pose_region_membership(self):
        ref = np.repeat(np.eye(4)[None], 2, axis=0)
        ref[1, 0, 3] = 1.0

        q = np.repeat(np.eye(4)[None], 3, axis=0)
        q[0, 0, 3] = 0.02
        q[1, 0, 3] = 1.04
        q[2, 0, 3] = 2.0

        keep, pos, rot, idx = pose_region_membership(
            q, ref, 0.05, math.radians(5.0)
        )
        self.assertEqual(keep.tolist(), [True, True, False])
        self.assertLess(pos[0], 0.05)
        self.assertEqual(idx[0], 0)

    def test_priority_then_rescue(self):
        prod = [
            {"target_rank": 0, "candidate_index": 10},
            {"target_rank": 1, "candidate_index": 20},
            {"target_rank": 2, "candidate_index": 30},
        ]
        filt = [
            {"target_rank": 0, "candidate_index": 10, "status": REJECT_OUTSIDE_REACH_REGION},
            {"target_rank": 1, "candidate_index": 20, "status": PASS_NEAR_REGION},
            {"target_rank": 2, "candidate_index": 30, "status": PASS_DIRECT},
        ]
        x = order_candidates_from_filter(prod, filt, mode="priority_then_rescue")
        self.assertEqual(x.ordered_indices, [1, 2, 0])
        self.assertEqual(x.pass_indices, [1, 2])
        self.assertEqual(x.reject_indices, [0])


if __name__ == "__main__":
    unittest.main()
