"""
MWSVisionBench - Russian document benchmark for multimodal LLMs

This package: Core source modules for MWSVisionBench

Copyright (c) 2024 MWS AI
Licensed under MIT License
"""

from importlib import import_module

__all__ = [
    "inference",
    "evaluation",
    "utils",
]

__version__ = "1.0.0"


def __getattr__(name):
    """Load package namespaces only when callers actually request them."""
    if name in __all__:
        module = import_module(f"{__name__}.{name}")
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
