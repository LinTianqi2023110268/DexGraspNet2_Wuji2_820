import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from routeB_front_half import runtime


class RouteBDenseRuntimeTest(unittest.TestCase):
    def test_structured_no_path_is_returned_for_orchestrator_retry(self):
        """Candidate no-path is not a worker/protocol failure."""
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            expected = {
                "success": False,
                "reason": "NO_PREGRASP_GOAL_WITH_VALID_CUROBO_TRAJECTORY",
            }
            (output_dir / "routeB_front_half_report.json").write_text(
                json.dumps(expected), encoding="utf-8"
            )
            with patch.object(runtime, "_run_streaming", return_value=(3, "backend no path")):
                actual = runtime.run_routeB_dense_backend(
                    project_root=Path("/tmp"),
                    capture_dir=Path("/tmp/capture"),
                    goal_pool=Path("/tmp/goals.npz"),
                    output_dir=output_dir,
                    settings={},
                )
            self.assertEqual(actual, expected)


if __name__ == "__main__":
    unittest.main()
