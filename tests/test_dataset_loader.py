import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from src.utils.dataset_loader import (
    HF_DATASET_REPOSITORIES,
    _convert_hf_to_list,
    _load_hf_datasets,
    get_dataset_info,
    load_benchmark_datasets,
)


class DatasetLoaderTests(unittest.TestCase):
    def test_repositories_are_explicit_for_both_families(self):
        self.assertEqual(
            HF_DATASET_REPOSITORIES["vision"],
            (
                "MTSAIR/MWS-Vision-Bench",
                "MTSAIR/MWS-Vision-Bench-Test",
            ),
        )
        self.assertEqual(
            HF_DATASET_REPOSITORIES["antifraud"],
            (
                "MTSAIR/MWS-Antifraud-Bench",
                "MTSAIR/MWS-Antifraud-Bench-Test",
            ),
        )

    def test_invalid_dataset_family_fails_before_download(self):
        with self.assertRaisesRegex(ValueError, "Unsupported dataset_family"):
            _load_hf_datasets(
                hf_token=None,
                hf_revision=None,
                hf_test_revision=None,
                cache_dir=None,
                sample=1,
                silent=True,
                dataset_family="unknown",
            )

    def test_validation_and_test_can_be_pinned_independently(self):
        calls = []

        def fake_load_dataset(repo, **kwargs):
            calls.append((repo, kwargs))
            return [
                {
                    "id": "1",
                    "type": "antifraud ru",
                    "dataset_name": "original",
                    "question": "q",
                    "answers": [],
                    "image_path": "",
                }
            ]

        fake_module = types.ModuleType("datasets")
        fake_module.load_dataset = fake_load_dataset
        with patch.dict(sys.modules, {"datasets": fake_module}):
            datasets, names = _load_hf_datasets(
                hf_token="token",
                hf_revision="validation-sha",
                hf_test_revision="test-sha",
                cache_dir=None,
                sample=1,
                silent=False,
                dataset_family="antifraud",
            )

        self.assertEqual(names, ["validation", "test"])
        self.assertEqual([len(dataset) for dataset in datasets], [1, 1])
        self.assertEqual(
            [call[0] for call in calls],
            [
                "MTSAIR/MWS-Antifraud-Bench",
                "MTSAIR/MWS-Antifraud-Bench-Test",
            ],
        )
        self.assertEqual(calls[0][1]["revision"], "validation-sha")
        self.assertEqual(calls[1][1]["revision"], "test-sha")

    def test_image_cache_is_isolated_by_family_and_split(self):
        item = {
            "id": "1",
            "type": "antifraud ru",
            "dataset_name": "original",
            "question": "q",
            "answers": [],
            "image_path": "",
        }
        vision = _convert_hf_to_list(
            [item],
            split_name="validation",
            cache_namespace="vision",
        )
        antifraud = _convert_hf_to_list(
            [item],
            split_name="validation",
            cache_namespace="antifraud",
        )
        self.assertEqual(len(vision), 1)
        self.assertEqual(len(antifraud), 1)

        root = Path(tempfile.gettempdir()) / "mws_vision_bench_cache"
        self.assertTrue((root / "vision" / "validation").is_dir())
        self.assertTrue((root / "antifraud" / "validation").is_dir())

    def test_local_sample_mode_does_not_need_huggingface(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp) / "images"
            base.mkdir()
            data_path = Path(tmp) / "data.json"
            data_path.write_text(
                json.dumps([{"id": "1"}, {"id": "2"}]),
                encoding="utf-8",
            )
            datasets, names = load_benchmark_datasets(
                data_paths=[str(data_path)],
                base_path=str(base),
                sample=1,
                dataset_family="antifraud",
            )
        self.assertEqual(names, ["part1"])
        self.assertEqual(datasets, [[{"id": "1"}]])

    def test_get_dataset_info_uses_selected_test_repo(self):
        calls = []

        def fake_load_dataset(repo, **kwargs):
            calls.append((repo, kwargs))
            return []

        fake_module = types.ModuleType("datasets")
        fake_module.load_dataset = fake_load_dataset
        with patch.dict(sys.modules, {"datasets": fake_module}):
            info = get_dataset_info(
                hf_token="token",
                dataset_family="antifraud",
            )

        self.assertTrue(info["test_available"])
        self.assertEqual(
            calls[0][0],
            "MTSAIR/MWS-Antifraud-Bench-Test",
        )


if __name__ == "__main__":
    unittest.main()
