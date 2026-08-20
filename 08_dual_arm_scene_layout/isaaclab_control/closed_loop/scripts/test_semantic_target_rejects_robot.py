"""Regression tests for the final semantic target safety gate."""
from __future__ import annotations

import unittest

import numpy as np

from perception_target_safety import (
    assert_current_capture_robot_mask,
    evaluate_dino_box,
    evaluate_sam_proposal,
    select_legal_proposal,
)


class SemanticTargetSafetyTest(unittest.TestCase):
    def setUp(self) -> None:
        self.valid = np.ones((8, 8), dtype=bool)
        self.source = np.ones((8, 8), dtype=bool)
        self.robot = np.zeros((8, 8), dtype=bool)

    def test_real_target_beats_robot_proposal(self) -> None:
        robot_only = np.zeros((8, 8), dtype=bool); robot_only[:5, :5] = True
        self.robot[:5, :5] = True
        cup = np.zeros((8, 8), dtype=bool); cup[5:, 5:] = True
        reject, _ = evaluate_sam_proposal(sam_mask=robot_only, robot_mask=self.robot, valid_depth=self.valid, inside_source_zone=self.source)
        accept, _ = evaluate_sam_proposal(sam_mask=cup, robot_mask=self.robot, valid_depth=self.valid, inside_source_zone=self.source)
        self.assertEqual(reject["reject_reason"], "REJECT_ROBOT_OVERLAP")
        self.assertTrue(accept["legal"])
        self.assertEqual(select_legal_proposal([{"idx": 0, "score": 0.9, **reject}, {"idx": 1, "score": 0.2, **accept}]), 1)

    def test_only_robot_means_no_selection(self) -> None:
        mask = np.ones((8, 8), dtype=bool); self.robot[:] = True
        row, _ = evaluate_sam_proposal(sam_mask=mask, robot_mask=self.robot, valid_depth=self.valid, inside_source_zone=self.source)
        self.assertEqual(row["reject_reason"], "REJECT_ROBOT_OVERLAP")
        self.assertIsNone(select_legal_proposal([{"idx": 0, "score": 0.9, **row}]))

    def test_dino_box_dominantly_robot_is_rejected_before_sam(self) -> None:
        self.robot[:6, :6] = True
        row = evaluate_dino_box(xyxy=np.asarray([0.0, 0.0, 6.0, 6.0]), robot_mask=self.robot)
        self.assertFalse(row["dino_box_legal"])
        self.assertEqual(row["dino_box_reject_reason"], "REJECT_ROBOT_OVERLAP")

    def test_robot_occluded_target_retains_nonrobot_residual(self) -> None:
        mask = np.ones((8, 8), dtype=bool); self.robot[:, :3] = True
        row, residual = evaluate_sam_proposal(sam_mask=mask, robot_mask=self.robot, valid_depth=self.valid, inside_source_zone=self.source)
        self.assertTrue(row["legal"])
        self.assertEqual(int(residual.sum()), 40)

    def test_outside_source_zone_rejected(self) -> None:
        self.source[:] = False
        row, _ = evaluate_sam_proposal(sam_mask=np.ones((8, 8), dtype=bool), robot_mask=self.robot, valid_depth=self.valid, inside_source_zone=self.source)
        self.assertEqual(row["reject_reason"], "REJECT_OUTSIDE_SOURCE_ZONE")

    def test_stale_robot_mask_fails_loudly(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "STALE_ROBOT_MASK"):
            assert_current_capture_robot_mask(
                robot_report_capture_dir="/tmp/cycle_001/capture",
                capture_dir="/tmp/cycle_002/capture",
            )

    def test_shape_mismatch_fails_loudly(self) -> None:
        with self.assertRaises(ValueError):
            evaluate_sam_proposal(sam_mask=np.ones((8, 8), dtype=bool), robot_mask=np.ones((7, 8), dtype=bool), valid_depth=self.valid, inside_source_zone=self.source)


if __name__ == "__main__":
    unittest.main()
