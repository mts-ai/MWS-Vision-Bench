"""
MWSVisionBench - Russian document benchmark for multimodal LLMs

This package: Inference implementations for various multimodal language model APIs

Copyright (c) 2024 MWS AI
Licensed under MIT License
"""

from importlib import import_module

__all__ = [
    "InferenceBase",
    "OpenAIInference",
    "GigaChatInference",
    "ResponsesInference",
    "create_inference_handler",
]

_IMPORTS = {
    "InferenceBase": (".inference_base", "InferenceBase"),
    "OpenAIInference": (".api_inference", "OpenAIInference"),
    "GigaChatInference": (".gigachat_inference", "GigaChatInference"),
    "ResponsesInference": (".responses_inference", "ResponsesInference"),
    "create_inference_handler": (
        ".inference_unified",
        "create_inference_handler",
    ),
}


def __getattr__(name):
    """Avoid importing optional provider SDKs until their backend is used."""
    try:
        module_name, attribute = _IMPORTS[name]
    except KeyError as exc:
        raise AttributeError(
            f"module {__name__!r} has no attribute {name!r}"
        ) from exc
    value = getattr(import_module(module_name, __name__), attribute)
    globals()[name] = value
    return value
