"""
MWSVisionBench - Russian OCR benchmark for multimodal LLMs

This file: GigaChat inference implementation using the unified base class

Copyright (c) 2024 MWS AI
Licensed under MIT License
"""

# Standard library imports
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

# Third-party imports
try:
    from gigachat import GigaChat
except ImportError:
    raise SystemExit("Requires 'gigachat' package. Install: pip install gigachat")

# Local application imports
# Add project root to Python path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.inference.inference_base import InferenceBase


class GigaChatInference(InferenceBase):
    """GigaChat-specific inference implementation"""
    
    def __init__(self):
        super().__init__('gigachat')
        self.client = None
        self.api_key = None
        self.streaming_supported = None  # Will be determined on first request
    
    def get_default_model(self) -> str:
        return "GigaChat-Max"
    
    def get_default_api_url(self) -> Optional[str]:
        return None  # GigaChat doesn't use URL parameter
    
    def get_default_max_workers(self) -> int:
        return 1  # GigaChat works better with fewer workers
    
    def initialize_client(self):
        """Initialize GigaChat client"""
        # Load API key
        if self.args.api_key is not None:
            self.api_key = self.args.api_key
        else:
            self.api_key = os.getenv("GIGACHAT_KEY")
        
        if not self.api_key:
            raise ValueError("GIGACHAT_KEY not provided. Set it via --api_key or environment variable.")
        
        print("Loaded GIGACHAT_KEY =", self.api_key[:10] + "..." if self.api_key else "None")
        
        # Initialize client
        self.client = GigaChat(
            model=self.args.model_name,
            credentials=self.api_key,
            scope="GIGACHAT_API_PERS",
            verify_ssl_certs=False
        )
    
    def process_item(self, item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Process a single item using GigaChat API"""
        if item["id"] in self.processed_ids:
            return None
        
        # Handle both absolute paths (from HF cache) and relative paths
        if os.path.isabs(item["image_path"]):
            image_path = item["image_path"]
        else:
            image_path = os.path.join(self.args.base_path or "", item["image_path"])
        question = item["question"]
        
        try:
            # Upload image
            with open(image_path, "rb") as f:
                file = self.client.upload_file(f)
        except Exception as e:
            logging.error(f"[ERROR] Failed to load image {image_path}: {e}")
            return None
        
        # Get supported parameters
        params = self.get_supported_params()
        
        # Prepare messages with system prompt if available
        messages = []
        
        # Add system message if SYSTEM_PROMPT is set
        system_prompt = os.getenv('SYSTEM_PROMPT')
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        # Add user message with image
        messages.append({
            "role": "user",
            "content": question,
            "attachments": [file.id_]
        })
        
        # Build payload with only supported parameters
        payload = {"messages": messages}
        
        # Add supported parameters to payload
        if 'temperature' in params:
            payload["temperature"] = params['temperature']
        if 'top_p' in params:
            payload["top_p"] = params['top_p']
        if 'max_tokens' in params:
            payload["max_tokens"] = params['max_tokens']
        if 'repetition_penalty' in params:
            payload["repetition_penalty"] = params['repetition_penalty']
        
        # GigaChat API supports: temperature, top_p, max_tokens, repetition_penalty, stream
        # https://developers.sber.ru/docs/ru/gigachat/api/grpc/grpc-methods
        
        # Try streaming first (if not explicitly disabled), fallback to non-streaming if not supported
        # Streaming advantages:
        # - Connection stays active with continuous data flow
        # - Server doesn't think client disconnected during long processing
        # - Ideal for long OCR tasks that take minutes to complete
        
        # Determine if we should try streaming
        use_streaming = self.streaming_supported is not False  # Try if unknown or True
        
        max_retries = 2
        for attempt in range(max_retries):
            try:
                # Set streaming mode in payload
                current_payload = payload.copy()
                if use_streaming:
                    current_payload["stream"] = True
                
                if use_streaming:
                    # Mark streaming as supported (first successful use)
                    if self.streaming_supported is None:
                        self.streaming_supported = True
                        logging.info("✓ Streaming mode enabled and working for GigaChat")
                    
                    # Process stream
                    answer = ""
                    try:
                        for chunk in self.client.stream(current_payload):
                            if chunk.choices and len(chunk.choices) > 0:
                                delta = chunk.choices[0].delta
                                if hasattr(delta, 'content') and delta.content:
                                    answer += delta.content
                        
                        if answer:
                            item_result = item.copy()
                            item_result["predict"] = answer.strip()
                            return item_result
                    except Exception as stream_error:
                        # If streaming fails, try to detect if it's unsupported
                        error_msg = str(stream_error).lower()
                        if 'stream' in error_msg or 'not supported' in error_msg:
                            # Streaming not supported, switch to non-streaming permanently
                            self.streaming_supported = False
                            use_streaming = False
                            logging.warning("⚠ Streaming not supported by GigaChat, switching to non-streaming mode")
                            continue  # Retry with non-streaming
                        else:
                            # Other error, re-raise
                            raise
                else:
                    # Non-streaming mode
                    response = self.client.chat(current_payload)
                    answer = response.choices[0].message.content.strip()
                    item_result = item.copy()
                    item_result["predict"] = answer
                    return item_result
            except Exception as e:
                logging.error(f"[ERROR] Exception during request (attempt {attempt + 1}/{max_retries}): {e}, image path:{image_path}, question: {question}")
            
            # Wait before retry
            if attempt < max_retries - 1:
                time.sleep(10)
        
        item_result = item.copy()
        item_result["predict"] = "ERROR in getting response"
        return item_result


def main():
    """Main entry point"""
    inference = GigaChatInference()
    inference.run()


if __name__ == "__main__":
    main()
