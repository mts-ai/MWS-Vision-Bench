"""
MWSVisionBench - Russian OCR benchmark for multimodal LLMs

This package: Core source modules for MWSVisionBench

Copyright (c) 2024 MWS AI
Licensed under MIT License
"""

# Main package modules
from . import inference
from . import evaluation
from . import utils

__all__ = [
    'inference',
    'evaluation', 
    'utils'
]

__version__ = '1.0.0'