import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.evaluation.eval_parallel import (
    _parse_antifraud_predict,
    process_single_item,
)
from src.evaluation.get_score_ru import (
    _compute_antifraud_details,
    _compute_antifraud_score,
    get_metrics,
    get_summary_score,
)


class AntifraudParserTests(unittest.TestCase):
    def test_parses_plain_json(self):
        parsed = _parse_antifraud_predict(
            '{"label": "edited", "arguments": "изменено ФИО"}'
        )
        self.assertEqual(parsed["predicted_label"], "edited")
        self.assertEqual(parsed["arguments"], "изменено ФИО")

    def test_parses_fenced_json_after_thinking(self):
        parsed = _parse_antifraud_predict(
            '<think>internal reasoning</think>\n'
            '```json\n{"label":"AI_GEN","arguments":"артефакты"}\n```'
        )
        self.assertEqual(parsed["predicted_label"], "ai_gen")
        self.assertEqual(parsed["arguments"], "артефакты")

    def test_parses_json_embedded_in_text(self):
        parsed = _parse_antifraud_predict(
            'Ответ: {"label": "original", "arguments": ""}.'
        )
        self.assertEqual(parsed["predicted_label"], "original")

    def test_rejects_invalid_label_and_non_json(self):
        self.assertIsNone(
            _parse_antifraud_predict(
                '{"label": "fake", "arguments": "x"}'
            )["predicted_label"]
        )
        self.assertIsNone(
            _parse_antifraud_predict("edited")["predicted_label"]
        )


class AntifraudItemEvaluationTests(unittest.TestCase):
    @patch(
        "src.evaluation.eval_parallel.vqa_evaluation",
        return_value=0.75,
    )
    def test_edited_item_keeps_classification_and_reason_scores(self, _mock_vqa):
        item = {
            "id": "1",
            "type": "antifraud ru",
            "dataset_name": "edited",
            "answers": ["ФИО"],
            "predict": (
                '{"label": "edited", "arguments": "изменено ФИО"}'
            ),
        }
        result = process_single_item(item)
        self.assertTrue(result["correct"])
        self.assertEqual(result["score"], 1.0)
        self.assertEqual(result["reason_score"], 0.75)

    def test_transport_error_has_complete_antifraud_fields(self):
        result = process_single_item(
            {
                "id": "2",
                "type": "antifraud ru",
                "dataset_name": "original",
                "answers": [],
                "predict": "ERROR in getting response",
            }
        )
        self.assertEqual(result["score"], 0.0)
        self.assertFalse(result["correct"])
        self.assertIsNone(result["predicted_label"])
        self.assertEqual(result["arguments"], "")
        self.assertEqual(result["reason_score"], 0.0)


class AntifraudMetricTests(unittest.TestCase):
    @staticmethod
    def _item(label, correct, reason=0.0):
        return {
            "type": "antifraud ru",
            "dataset_name": label,
            "correct": correct,
            "reason_score": reason,
            "score": 1.0 if correct else 0.0,
        }

    def test_perfect_score_is_one(self):
        items = [
            self._item("ai_gen", True),
            self._item("edited", True, 1.0),
            self._item("original", True),
        ]
        self.assertAlmostEqual(_compute_antifraud_score(items), 1.0)

    def test_single_class_predictor_is_chance_level(self):
        items = [
            self._item("ai_gen", False),
            self._item("edited", False, 0.0),
            self._item("original", True),
        ]
        self.assertAlmostEqual(_compute_antifraud_score(items), 0.0)

    def test_missing_classes_do_not_inflate_sample_score(self):
        details = _compute_antifraud_details(
            [self._item("original", True)]
        )
        self.assertAlmostEqual(details["balanced_accuracy"], 1 / 3)
        self.assertAlmostEqual(details["score"], 0.0)
        self.assertEqual(
            details["count_by_class"],
            {"ai_gen": 0, "edited": 0, "original": 1},
        )

    def test_reason_score_is_clamped(self):
        items = [
            self._item("ai_gen", True),
            self._item("edited", True, 4.0),
            self._item("original", True),
        ]
        self.assertAlmostEqual(_compute_antifraud_score(items), 1.0)

    def test_antifraud_does_not_change_five_category_overall(self):
        rows = [
            {"type": "full-page OCR ru", "score": 0.9},
            {"type": "text grounding ru", "score": 0.1},
            {"type": "key information extraction ru", "score": 0.8},
            {"type": "document parsing ru", "score": 0.7},
            {"type": "reasoning VQA ru", "score": 1.0},
            self._item("ai_gen", True),
            self._item("edited", True, 1.0),
            self._item("original", True),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "mixed.json"
            path.write_text(
                json.dumps(rows, ensure_ascii=False),
                encoding="utf-8",
            )
            metrics, detailed = get_metrics(str(path))

        self.assertAlmostEqual(detailed["overall"]["average"], 0.7)
        self.assertEqual(detailed["overall"]["count"], 5)
        self.assertEqual(detailed["total_count"], 8)
        self.assertFalse(detailed["overall"]["includes_antifraud"])
        self.assertAlmostEqual(
            metrics["antifraud (document_verification)"],
            1.0,
        )
        self.assertAlmostEqual(
            get_summary_score(metrics, detailed, "vision"),
            0.7,
        )
        self.assertAlmostEqual(
            get_summary_score(metrics, detailed, "antifraud"),
            1.0,
        )

    def test_antifraud_only_cli_does_not_print_fake_overall(self):
        rows = [
            self._item("ai_gen", True),
            self._item("edited", True, 1.0),
            self._item("original", True),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "antifraud.json"
            path.write_text(
                json.dumps(rows, ensure_ascii=False),
                encoding="utf-8",
            )
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

        self.assertIn("Anti-fraud Score: 1.000", completed.stdout)
        self.assertNotIn("Russian Overall Score", completed.stdout)


if __name__ == "__main__":
    unittest.main()
