"""
MWSVisionBench - Russian OCR benchmark for multimodal LLMs

This package: Inference implementations for various multimodal language model APIs

Copyright (c) 2024 MWS AI
Licensed under MIT License
"""

from .inference_base import InferenceBase
from .api_inference import OpenAIInference
from .gigachat_inference import GigaChatInference
from .responses_inference import ResponsesInference
from .inference_unified import create_inference_handler

__all__ = [
    'InferenceBase',
    'OpenAIInference', 
    'GigaChatInference',
    'ResponsesInference',
    'create_inference_handler'
]
