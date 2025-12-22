"""
MWSVisionBench - Russian OCR benchmark for multimodal LLMs

This file: OpenAI-compatible inference implementation using the unified base class.
Supports OpenAI API, vLLM endpoints, and other OpenAI-compatible APIs.

Copyright (c) 2024 MWS AI
Licensed under MIT License
"""

# Standard library imports
import base64
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

# Third-party imports
import requests

# Local application imports
# Add project root to Python path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.inference.inference_base import InferenceBase


class OpenAIInference(InferenceBase):
    """OpenAI-compatible inference implementation"""
    
    def __init__(self):
        super().__init__('openai')
        self.api_key = None
        self.headers = None
        self.streaming_supported = None  # Will be determined on first request
    
    def get_default_model(self) -> str:
        return "gpt-4o-mini"
    
    def get_default_api_url(self) -> str:
        return "https://api.openai.com/v1/chat/completions"
    
    def get_default_max_workers(self) -> int:
        return 5  # OpenAI hight tiers and private vLLM servers can easily handle higher parallelism - up to 30
    
    def initialize_client(self):
        """Initialize OpenAI API client"""
        # Load API key
        if self.args.api_key is not None:
            self.api_key = self.args.api_key
        else:
            self.api_key = os.getenv("OPENAI_API_KEY")
        
        print("Loaded OPENAI_API_KEY =", self.api_key)
        
        # Set up headers
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
    
    def get_response(self, payload, use_streaming=False):
        """Get answer from OpenAI API"""
        return requests.post(
            self.args.api_url,
            headers=self.headers,
            json=payload,
            timeout=(10, 1200),  # (connect, read) - 10s connect, 20 min read
            stream=use_streaming  # Enable streaming response in requests
        )

    def extract_answer(self, resp, use_streaming=False):
        """Extract answer from OpenAI API response"""
        if use_streaming:
            # Process Server-Sent Events (SSE) stream
            answer = ""
            for line in resp.iter_lines():
                if line:
                    line_str = line.decode('utf-8')
                    # SSE format: "data: <json>"
                    if line_str.startswith('data: '):
                        data_str = line_str[6:]  # Remove "data: " prefix
                        if data_str == '[DONE]':
                            # Stream finished
                            break
                        try:
                            chunk = json.loads(data_str)
                            # Extract content delta from chunk
                            if 'choices' in chunk and len(chunk['choices']) > 0:
                                delta = chunk['choices'][0].get('delta', {})
                                if 'content' in delta:
                                    answer += delta['content']
                        except json.JSONDecodeError:
                            # Skip malformed chunks
                            continue
        else:
            # Process regular response
            answer = resp.json()["choices"][0]["message"]["content"].strip()
        return answer

    def process_item(self, item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Process a single item using OpenAI-compatible API"""
        if item["id"] in self.processed_ids:
            return None
        
        # Handle both absolute paths (from HF cache) and relative paths
        if os.path.isabs(item["image_path"]):
            image_path = item["image_path"]
        else:
            image_path = os.path.join(self.args.base_path or "", item["image_path"])
        question = item["question"]
        
        try:
            with open(image_path, "rb") as f:
                img_bytes = f.read()
            img_b64 = base64.b64encode(img_bytes).decode('utf-8')
        except Exception as e:
            logging.error(f"[ERROR] Failed to load image {image_path}: {e}")
            return None
        
        # Get supported parameters
        params = self.get_supported_params()
        
        # Determine if this is GPT-5 (has different parameter handling)
        is_gpt5 = "gpt-5" in self.args.model_name.lower()
        
        # Form messages
        messages = []
        
        # Add system message if SYSTEM_PROMPT is set
        system_prompt = os.getenv('SYSTEM_PROMPT')
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        # Add user message
        messages.append({
            "role": "user", 
            "content": [
                {"type": "text", "text": question},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}}
            ]
        })
        
        # Build base payload
        payload = {
            "model": self.args.model_name,
            "messages": messages
        }
        
        # Add supported parameters
        if is_gpt5:
            # GPT-5 only supports max_completion_tokens
            if 'max_tokens' in params:
                payload["max_completion_tokens"] = params['max_tokens']
        else:
            # Regular models support all parameters
            if 'temperature' in params:
                payload["temperature"] = params['temperature']
            if 'top_p' in params:
                payload["top_p"] = params['top_p']
            if 'presence_penalty' in params:
                payload["presence_penalty"] = params['presence_penalty']
            if 'frequency_penalty' in params:
                payload["frequency_penalty"] = params['frequency_penalty']
            if 'max_tokens' in params:
                payload["max_tokens"] = params['max_tokens']
        
        # Try streaming first (if not explicitly disabled), fallback to non-streaming if not supported
        # Streaming advantages:
        # - Connection stays active with continuous data flow
        # - Server doesn't think client disconnected during long processing
        # - Ideal for long OCR tasks that take minutes to complete
        
        # Determine if we should try streaming
        use_streaming = self.streaming_supported is not False  # Try if unknown or True
        
        max_retries = 5
        for attempt in range(max_retries):
            try:
                # Set streaming mode in payload
                current_payload = payload.copy()
                if use_streaming:
                    current_payload["stream"] = True
                
                resp = self.get_response(current_payload)
                if resp.status_code == 200:
                    if use_streaming:
                        # Mark streaming as supported (first successful use)
                        if self.streaming_supported is None:
                            self.streaming_supported = True
                            logging.info("✓ Streaming mode enabled and working")
                        
                        answer = self.extract_answer(resp, use_streaming=True)
                        if answer:
                            item_result = item.copy()
                            item_result["predict"] = answer.strip()
                            return item_result
                    else:
                        answer = self.extract_answer(resp, use_streaming=False)
                        item_result = item.copy()
                        item_result["predict"] = answer
                        return item_result
                elif resp.status_code == 400 and use_streaming:
                    # Check if error is due to unsupported streaming
                    try:
                        error_json = resp.json()
                        error_msg = str(error_json.get('error', {}))
                        if 'stream' in error_msg.lower():
                            # Streaming not supported, switch to non-streaming permanently
                            self.streaming_supported = False
                            use_streaming = False
                            logging.warning("⚠ Streaming not supported by API, switching to non-streaming mode")
                            continue  # Retry with non-streaming
                    except:
                        pass
                    
                    logging.warning(
                        f"[WARN] Request failed (attempt {attempt + 1}/{max_retries}) "
                        f"with status {resp.status_code}, image path: {image_path}"
                    )
                else:
                    logging.warning(
                        f"[WARN] Request failed (attempt {attempt + 1}/{max_retries}) "
                        f"with status {resp.status_code}, image path: {image_path}"
                    )
            except requests.exceptions.Timeout as e:
                logging.error(
                    f"[ERROR] Timeout (attempt {attempt + 1}/{max_retries}): {e}, "
                    f"image path: {image_path}"
                )
            except requests.exceptions.ConnectionError as e:
                logging.error(
                    f"[ERROR] Connection error (attempt {attempt + 1}/{max_retries}): {e}, "
                    f"image path: {image_path}"
                )
            except Exception as e:
                logging.error(
                    f"[ERROR] Exception during request (attempt {attempt + 1}/{max_retries}): {e}, "
                    f"image path: {image_path}, question: {question}"
                )
            
            # Exponential backoff between retries
            if attempt < max_retries - 1:
                wait_time = min(30 * (2 ** attempt), 300)  # 30, 60, 120, 240, 300 seconds max
                logging.info(f"Waiting {wait_time}s before retry...")
                time.sleep(wait_time)
        
        # All retries exhausted
        item_result = item.copy()
        item_result["predict"] = "ERROR in getting response"
        return item_result


def main():
    """Main entry point"""
    inference = OpenAIInference()
    inference.run()


if __name__ == "__main__":
    main()
