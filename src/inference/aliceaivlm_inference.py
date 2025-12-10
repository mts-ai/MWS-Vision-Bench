"""
MWSVisionBench - Russian OCR benchmark for multimodal LLMs

This file: Yandex Alice AI VLM API Inference

Copyright (c) 2025 MWS AI
Licensed under MIT License
"""

# Standard library imports
import json
import sys
from pathlib import Path

# Third-party imports
import requests
import urllib3
# Disable SSL warnings for insecure requests to yangdex.ru
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Local application imports
# Add project root to Python path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.inference.api_inference import OpenAIInference


class AliceAiVlmInference(OpenAIInference):
    def __init__(self):
        super().__init__('aliceaivlm')
        self.client = None
        self.api_key = None
        self.streaming_supported = False

    def get_default_max_workers(self) -> int:
        return 1
    
    def get_response(self, payload, use_streaming=False):
        messages = payload['messages']
        assert use_streaming is False, "streaming is not supported"
        return requests.post(
            self.args.api_url,
            params={"cbird": self.api_key},
            files={
                "gaw_dialog_request": json.dumps(
                    {"messages": messages},
                    ensure_ascii=False,
                )
            },
            verify=False
        )
    
    def extract_answer(self, resp, use_streaming=False):
        assert use_streaming is False, "streaming is not supported"
        vlm_response_data = resp.json()["vlm"][0]
        answer = vlm_response_data["Response"]
        return answer
        