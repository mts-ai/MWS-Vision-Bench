"""
MWSVisionBench - Russian OCR benchmark for multimodal LLMs

This package: Evaluation modules for calculating metrics and processing results

Copyright (c) 2024 MWS AI
Licensed under MIT License
"""

from .get_score_ru import get_metrics

__all__ = [
    'get_metrics'
]