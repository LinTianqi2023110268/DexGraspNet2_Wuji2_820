import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from routeB_full_pipeline import runtime


class RouteBFullRuntimeCandidateFailureTest(unittest.TestCase):
    def _run(self, output_dir: Path):
        return runtime.run_full_motion_backend(
            project_root=Path("/tmp/project"),
            capture_dir=Path("/tmp/capture"),
            query="red_000",
            case_root=Path("/tmp/case"),
            front_half_dir=Path("/tmp/front"),
            backhalf_pool=Path("/tmp/backhalf.npz"),
            output_dir=output_dir,
            settings={},
        )

    def test_backend_crash_cannot_reuse_previous_candidate_report(self):
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            report_path = output_dir / "routeB_full_plan_report.json"
            report_path.write_text(
                json.dumps({"success": False, "selected_candidate": 6539}),
                encoding="utf-8",
            )
            backend_output = (
                "Traceback (most recent call last):\n"
                "RuntimeError: PREGRASP->COVER failed for candidate 4368\n"
            )
            with patch.object(runtime, "_stream", return_value=(1, backend_output)):
                with self.assertRaisesRegex(
                    RuntimeError,
                    "produced no report: PREGRASP->COVER failed for candidate 4368",
                ):
                    self._run(output_dir)
            self.assertFalse(report_path.exists())

    def test_structured_candidate_failure_is_returned_for_retry(self):
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            expected = {
                "success": False,
                "selected_candidate": 4368,
                "failure_stage": "PREGRASP_TO_COVER",
            }

            def write_current_report(_cmd, *, cwd):
                self.assertEqual(cwd, Path("/tmp/project"))
                (output_dir / "routeB_full_plan_report.json").write_text(
                    json.dumps(expected), encoding="utf-8"
                )
                return 0, "[Route B][FULL] FAIL candidate=4368\n"

            with patch.object(runtime, "_stream", side_effect=write_current_report):
                actual = self._run(output_dir)
            self.assertEqual(actual, expected)


if __name__ == "__main__":
    unittest.main()
