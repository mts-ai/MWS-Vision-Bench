import json
import subprocess
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path

from src.evaluation.bootstrap import (
    bootstrap_category_means,
    bootstrap_stratified_score,
)
from src.evaluation.get_score_ru import get_metrics


class BootstrapConfidenceTests(unittest.TestCase):
    @staticmethod
    def _write_rows(tmp, rows):
        path = Path(tmp) / "scores.json"
        path.write_text(json.dumps(rows), encoding="utf-8")
        return path

    def test_constant_scores_produce_zero_width_intervals(self):
        result = bootstrap_category_means(
            {"a": [0.25, 0.25], "b": [0.75, 0.75]},
            samples=100,
            seed=7,
        )

        self.assertEqual(
            result["categories"]["a"],
            {"low": 0.25, "high": 0.25},
        )
        self.assertEqual(
            result["overall"],
            {"low": 0.5, "high": 0.5},
        )

    def test_bootstrap_is_deterministic_for_a_seed(self):
        options = {
            "category_scores": {"a": [0.0, 0.5, 1.0]},
            "samples": 200,
            "seed": 123,
        }
        first = bootstrap_category_means(**options)
        second = bootstrap_category_means(**options)

        self.assertEqual(first, second)

    def test_stratified_bootstrap_preserves_class_counts(self):
        items = [
            {"class": "a", "score": 0.0},
            {"class": "a", "score": 1.0},
            {"class": "b", "score": 0.5},
        ]
        observed_counts = []

        def score_fn(replicate):
            observed_counts.append(Counter(row["class"] for row in replicate))
            return sum(row["score"] for row in replicate) / len(replicate)

        interval = bootstrap_stratified_score(
            items,
            strata_key="class",
            score_fn=score_fn,
            samples=50,
            seed=3,
        )

        self.assertTrue(observed_counts)
        self.assertTrue(
            all(counts == {"a": 2, "b": 1} for counts in observed_counts)
        )
        self.assertLessEqual(interval["low"], interval["high"])

    def test_get_metrics_adds_category_and_overall_intervals(self):
        task_types = [
            "text grounding ru",
            "reasoning VQA ru",
            "full-page OCR ru",
            "document parsing ru",
            "key information extraction ru",
        ]
        rows = [
            {"type": task_type, "score": score}
            for task_type in task_types
            for score in (0.0, 1.0)
        ]
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write_rows(tmp, rows)
            metrics, detailed = get_metrics(
                str(path),
                bootstrap_samples=200,
                bootstrap_seed=11,
            )

        bootstrap = detailed["bootstrap"]
        self.assertEqual(bootstrap["samples"], 200)
        self.assertEqual(bootstrap["seed"], 11)
        self.assertEqual(set(bootstrap["metrics"]), set(metrics))
        self.assertIn("overall", bootstrap)
        self.assertTrue(detailed["overall"]["coverage_complete"])

    def test_antifraud_interval_is_stratified_and_reported(self):
        rows = [
            {
                "type": "antifraud ru",
                "dataset_name": label,
                "correct": True,
                "reason_score": 1.0 if label == "edited" else 0.0,
            }
            for label in ("ai_gen", "edited", "original")
        ]
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write_rows(tmp, rows)
            metrics, detailed = get_metrics(
                str(path),
                bootstrap_samples=50,
            )

        metric_name = "antifraud (document_verification)"
        self.assertEqual(metrics[metric_name], 1.0)
        self.assertEqual(
            detailed["bootstrap"]["antifraud"],
            {"low": 1.0, "high": 1.0},
        )

    def test_cli_prints_confidence_interval(self):
        rows = [
            {"type": "reasoning VQA ru", "score": 0.5},
            {"type": "reasoning VQA ru", "score": 0.5},
        ]
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write_rows(tmp, rows)
            completed = subprocess.run(
                [
                    sys.executable,
                    "src/evaluation/get_score_ru.py",
                    "--input_path",
                    str(path),
                    "--bootstrap-samples",
                    "50",
                ],
                check=True,
                capture_output=True,
                text=True,
            )

        self.assertIn("[95% CI: 0.500-0.500]", completed.stdout)

    def test_invalid_bootstrap_options_are_rejected(self):
        with self.assertRaises(ValueError):
            bootstrap_category_means(
                {"a": [1.0]},
                samples=10,
                confidence_level=1.0,
            )

        with tempfile.TemporaryDirectory() as tmp:
            path = self._write_rows(tmp, [])
            with self.assertRaises(ValueError):
                get_metrics(
                    str(path),
                    bootstrap_samples=-1,
                )


if __name__ == "__main__":
    unittest.main()
