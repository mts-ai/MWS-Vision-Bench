import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from src.evaluation.metrics import page_ocr_metric


class _FakeResponse:
    def __init__(self, chunks):
        self._chunks = chunks

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False

    def raise_for_status(self):
        return None

    def iter_content(self, chunk_size):
        del chunk_size
        return iter(self._chunks)


class WordNetBootstrapTests(unittest.TestCase):
    def test_verified_archive_is_installed_in_nltk_data(self):
        payload = b"pinned-wordnet-archive"

        with tempfile.TemporaryDirectory() as temporary:
            data_root = Path(temporary)
            with (
                mock.patch.object(
                    page_ocr_metric,
                    "_wordnet_available",
                    side_effect=(False, True),
                ),
                mock.patch.object(
                    page_ocr_metric,
                    "_wordnet_data_root",
                    return_value=data_root,
                ),
                mock.patch.object(
                    page_ocr_metric,
                    "_WORDNET_SHA256",
                    hashlib.sha256(payload).hexdigest(),
                ),
                mock.patch.object(
                    page_ocr_metric.requests,
                    "get",
                    return_value=_FakeResponse((payload[:8], payload[8:])),
                ),
            ):
                page_ocr_metric.ensure_wordnet()

            self.assertEqual(
                (data_root / "corpora" / "wordnet.zip").read_bytes(),
                payload,
            )

    def test_unverified_archive_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            data_root = Path(temporary)
            with (
                mock.patch.object(
                    page_ocr_metric,
                    "_wordnet_available",
                    return_value=False,
                ),
                mock.patch.object(
                    page_ocr_metric,
                    "_wordnet_data_root",
                    return_value=data_root,
                ),
                mock.patch.object(
                    page_ocr_metric.requests,
                    "get",
                    return_value=_FakeResponse((b"corrupt",)),
                ),self.assertRaisesRegex(RuntimeError, "SHA-256")
            ):
                page_ocr_metric.ensure_wordnet()

            self.assertFalse(
                (data_root / "corpora" / "wordnet.zip").exists()
            )


if __name__ == "__main__":
    unittest.main()
