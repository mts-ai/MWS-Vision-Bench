import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import requests

from src.inference.api_inference import OpenAIInference
from src.inference.inference_unified import create_inference_handler


class InferenceImportTests(unittest.TestCase):
    def test_openai_handler_does_not_import_optional_provider_sdks(self):
        handler = create_inference_handler("anthropic/claude-fable-5")
        self.assertIsInstance(handler, OpenAIInference)


class OpenAIInferenceTests(unittest.TestCase):
    def test_initialization_does_not_print_api_key(self):
        handler = OpenAIInference()
        handler.args = SimpleNamespace(api_key="secret-value")
        stdout = io.StringIO()
        with redirect_stdout(stdout):
            handler.initialize_client()
        self.assertNotIn("secret-value", stdout.getvalue())

    def test_jpeg_payload_uses_matching_media_type(self):
        handler = OpenAIInference()
        with tempfile.TemporaryDirectory() as tmp:
            image_path = Path(tmp) / "document.jpg"
            image_path.write_bytes(b"\xff\xd8\xff\xd9")
            handler.args = SimpleNamespace(
                model_name="anthropic/claude-fable-5",
                api_url="https://example.invalid",
                base_path=None,
            )
            handler.processed_ids = set()
            response = Mock(status_code=200)
            response.iter_lines.return_value = [
                b'data: {"choices":[{"delta":{"content":"ok"}}]}',
                b"data: [DONE]",
            ]
            with patch.object(
                handler,
                "get_response",
                return_value=response,
            ) as request:
                result = handler.process_item(
                    {
                        "id": "1",
                        "image_path": str(image_path),
                        "question": "question",
                    }
                )

        payload = request.call_args.args[0]
        image_url = payload["messages"][0]["content"][1]["image_url"]["url"]
        self.assertTrue(image_url.startswith("data:image/jpeg;base64,"))
        self.assertEqual(result["predict"], "ok")
        self.assertTrue(request.call_args.kwargs["use_streaming"])

    def test_streaming_heartbeats_cannot_extend_total_timeout(self):
        handler = OpenAIInference()
        response = Mock()
        response.iter_lines.return_value = [b": keep-alive"]
        with patch(
            "src.inference.api_inference.time.monotonic",
            return_value=1201.0,
        ):
            with self.assertRaisesRegex(
                requests.exceptions.Timeout,
                "total response timeout",
            ):
                handler.extract_answer(
                    response,
                    use_streaming=True,
                    deadline=1200.0,
                )


if __name__ == "__main__":
    unittest.main()
