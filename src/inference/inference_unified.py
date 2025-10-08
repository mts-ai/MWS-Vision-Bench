"""
MWSVisionBench - Russian OCR benchmark for multimodal LLMs

This file: Unified inference script that automatically selects the appropriate API implementation.
Single entry point for all model types.

Copyright (c) 2024 MWS AI
Licensed under MIT License
"""

# Standard library imports
import argparse
import os
import sys
from pathlib import Path

# Local application imports
# Add project root to Python path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.inference.api_inference import OpenAIInference
from src.inference.gigachat_inference import GigaChatInference
from src.inference.inference_base import InferenceBase


def create_inference_handler(model_name: str) -> InferenceBase:
    """Factory function to create appropriate inference handler based on model name"""
    model_lower = model_name.lower()
    
    # GPT-5 (exact match) goes to Responses API
    if model_name == 'gpt-5':
        # Import here to avoid dependency issues if not needed
        from src.inference.responses_inference import ResponsesInference
        return ResponsesInference()
    
    # GPT-5 family (mini, nano) goes to regular OpenAI Completions API
    elif model_name.startswith('gpt-5-'):
        return OpenAIInference()
    
    # GigaChat
    elif 'gigachat' in model_lower:
        return GigaChatInference()
    
    # Default: OpenAI-compatible (vLLM, OpenAI, etc.)
    else:
        return OpenAIInference()


def main():
    """Main entry point - parse model name and delegate to appropriate handler"""
    # Quick parse to get model name
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', required=False, default="gpt-4o-mini")
    temp_args, _ = parser.parse_known_args()
    
    # Create appropriate inference handler
    inference = create_inference_handler(temp_args.model_name)
    
    # Run inference (handler will parse all args properly)
    inference.run()


if __name__ == "__main__":
    main()
