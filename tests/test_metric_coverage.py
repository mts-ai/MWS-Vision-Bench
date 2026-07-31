import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from src.evaluation.get_score_ru import VISION_CATEGORIES, get_metrics


class MetricCoverageTests(unittest.TestCase):
    @staticmethod
    def _write_rows(tmp, rows):
        path = Path(tmp) / "scores.json"
        path.write_text(json.dumps(rows), encoding="utf-8")
        return path

    def test_complete_vision_run_is_leaderboard_comparable(self):
        rows = [
            {"type": "text grounding ru", "score": 0.1},
            {"type": "reasoning VQA ru", "score": 0.2},
            {"type": "full-page OCR ru", "score": 0.3},
            {"type": "document parsing ru", "score": 0.4},
            {"type": "key information extraction ru", "score": 0.5},
        ]
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write_rows(tmp, rows)
            _, detailed = get_metrics(str(path))

        self.assertTrue(detailed["overall"]["coverage_complete"])
        self.assertEqual(
            detailed["overall"]["present_categories"],
            list(VISION_CATEGORIES),
        )
        self.assertEqual(detailed["overall"]["missing_categories"], [])
        self.assertAlmostEqual(detailed["overall"]["average"], 0.3)

    def test_partial_run_is_explicitly_marked(self):
        rows = [{"type": "reasoning VQA ru", "score": 0.8}]
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write_rows(tmp, rows)
            _, detailed = get_metrics(str(path))
            completed = subprocess.run(
                [
                    sys.executable,
                    "src/evaluation/get_score_ru.py",
                    "--input_path",
                    str(path),
                ],
                check=True,
                capture_output=True,
                text=True,
            )

        self.assertFalse(detailed["overall"]["coverage_complete"])
        self.assertEqual(
            detailed["overall"]["present_categories"],
            ["reasoning_vqa"],
        )
        self.assertIn("Russian Partial Overall Score: 0.800", completed.stdout)
        self.assertIn("not leaderboard-comparable", completed.stdout)

    def test_unknown_task_types_are_reported_with_counts(self):
        rows = [
            {"type": "reasoning VQA ru", "score": 0.8},
            {"type": "future task", "score": 1.0},
            {"type": "future task", "score": 0.0},
        ]
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write_rows(tmp, rows)
            _, detailed = get_metrics(str(path))

        self.assertEqual(
            detailed["unknown_task_types"],
            {"future task": 2},
        )
        self.assertEqual(detailed["overall"]["count"], 1)


if __name__ == "__main__":
    unittest.main()
