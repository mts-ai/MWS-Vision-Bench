"""
MWSVisionBench - Russian OCR benchmark for multimodal LLMs

This file: OpenAI Responses API inference implementation using the unified base class.
Specialized for GPT-5 models with reasoning capabilities.

Copyright (c) 2024 MWS AI
Licensed under MIT License
"""

# Standard library imports
import base64
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

# Third-party imports
try:
    from openai import OpenAI
except ImportError:
    raise SystemExit("Requires 'openai' package (>=1.99). Install: pip install -U openai")

# Local application imports
# Add project root to Python path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.inference.inference_base import InferenceBase


class ResponsesInference(InferenceBase):
    """OpenAI Responses API inference implementation"""
    
    def __init__(self):
        super().__init__('responses')
        self.client = None
        self.api_key = None
    
    def get_default_model(self) -> str:
        return "gpt-5"
    
    def get_default_api_url(self) -> Optional[str]:
        return None  # Using OpenAI client, not direct requests
    
    def get_default_max_workers(self) -> int:
        return 5  # Responses API has rate limits - but for tier 5 feel free to increase up to 30
    
    def initialize_client(self):
        """Initialize OpenAI Responses API client"""
        # Load API key
        if self.args.api_key is not None:
            self.api_key = self.args.api_key
        else:
            self.api_key = os.getenv("OPENAI_API_KEY")
        
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY not provided. Set it via --api_key or environment variable.")
        
        print("Loaded OPENAI_API_KEY =", self.api_key[:10] + "..." if self.api_key else "None")
        
        # Initialize client with official endpoint
        self.client = OpenAI(api_key=self.api_key, base_url="https://api.openai.com/v1")
        
        # Available models check
        available_models = {
            "gpt-5": "GPT-5 (most powerful model)",
            "gpt-5-mini": "GPT-5 Mini (balance of speed and quality)",
            "gpt-5-nano": "GPT-5 Nano (faster but lower quality)"
        }
        
        if self.args.model_name not in available_models:
            logging.warning(f"Model {self.args.model_name} may not be available for Responses API")
            logging.info(f"Available models: {list(available_models.keys())}")
    
    def process_item(self, item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Process a single item using OpenAI Responses API"""
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
        
        # Prepare messages
        input_messages = []
        
        # Add system message if SYSTEM_PROMPT is set
        system_prompt = os.getenv('SYSTEM_PROMPT')
        if system_prompt:
            input_messages.append({
                "role": "system",
                "content": [
                    {"type": "input_text", "text": system_prompt}
                ]
            })
        
        # Add user message with image (Responses API format!)
        input_messages.append({
            "role": "user",
            "content": [
                {"type": "input_text", "text": question},
                {"type": "input_image", "image_url": f"data:image/png;base64,{img_b64}"}
            ]
        })
        
        # Get environment parameters
        verbosity = os.getenv('RESPONSES_VERBOSITY', 'low')  # low, medium, high
        reasoning_effort = os.getenv('RESPONSES_REASONING_EFFORT', 'minimal')  # minimal, medium, high
        max_tokens = int(os.getenv('MAX_TOKENS', '8192'))
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                # Use Responses API (not chat completions!)
                resp = self.client.responses.create(
                    model=self.args.model_name,
                    input=input_messages,
                    text={"verbosity": verbosity},
                    reasoning={"effort": reasoning_effort},
                    max_output_tokens=max_tokens
                )
                
                # Extract text from Responses API
                output_text = ""
                try:
                    for item_output in resp.output:
                        if hasattr(item_output, "content") and item_output.content:
                            for c in item_output.content:
                                if hasattr(c, "text") and c.text:
                                    output_text += c.text
                except Exception:
                    # fallback for different client versions
                    pass
                
                output_text = (output_text or "").strip()
                
                if output_text:
                    item_result = item.copy()
                    item_result["predict"] = output_text
                    return item_result
                else:
                    logging.warning(f"[WARN] Empty output from Responses API (attempt {attempt + 1})")
                    
            except TypeError as e:
                # If parameters not supported, try without them
                logging.warning(f"Responses API doesn't support generation parameters: {e}")
                try:
                    resp = self.client.responses.create(
                        model=self.args.model_name,
                        input=input_messages,
                        text={"verbosity": verbosity},
                        reasoning={"effort": reasoning_effort}
                    )
                    # Same text extraction logic...
                    output_text = ""
                    try:
                        for item_output in resp.output:
                            if hasattr(item_output, "content") and item_output.content:
                                for c in item_output.content:
                                    if hasattr(c, "text") and c.text:
                                        output_text += c.text
                    except Exception:
                        pass
                    
                    if output_text:
                        item_result = item.copy()
                        item_result["predict"] = output_text.strip()
                        return item_result
                        
                except Exception as e2:
                    logging.error(f"[ERROR] Exception during request (attempt {attempt + 1}): {e2}, image path:{image_path}, question: {question}")
                    
            except Exception as e:
                logging.error(f"[ERROR] Exception during request (attempt {attempt + 1}): {e}, image path:{image_path}, question: {question}")
                
            if attempt < max_retries - 1:
                time.sleep(60)  # Longer delay for Responses API
        
        item_result = item.copy()
        item_result["predict"] = "ERROR in getting response"
        return item_result


def main():
    """Main entry point"""
    inference = ResponsesInference()
    inference.run()


if __name__ == "__main__":
    main()
