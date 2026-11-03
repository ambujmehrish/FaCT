"""FaCT. The model classes load on first use, so light tools that import a
subpackage (fact.ops.queue, the runner on a login node) never import torch."""

__all__ = ["FaCTConfig", "tiny_config", "Batch", "FaCT", "TokenizerOutput"]
__version__ = "0.1.0"

_LAZY = {"FaCTConfig": "config", "tiny_config": "config",
         "Batch": "model", "FaCT": "model", "TokenizerOutput": "model"}


def __getattr__(name):
    if name in _LAZY:
        import importlib
        return getattr(importlib.import_module(f".{_LAZY[name]}", __name__), name)
    raise AttributeError(f"module 'fact' has no attribute {name!r}")
