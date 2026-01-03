# This is a registry to call models by name
from typing import Dict, Any, Callable, Tuple
import importlib
import torch.nn as nn

# Global name → constructor registry
_MODEL_REGISTRY: Dict[str, Callable[..., nn.Module]] = {}

def register_model(name: str):
    """Decorator to register a model class/constructor under a name."""
    def _wrap(cls_or_fn):
        _MODEL_REGISTRY[name] = cls_or_fn
        return cls_or_fn
    return _wrap

def create_model(model_cfg: Dict[str, Any], input_shape: Tuple[int, ...], output_shape: Tuple[int, ...]) -> nn.Module:
    """
    Factory: expects config like {"name": "mlp", "hparams": {...}}.
    Returns an instantiated nn.Module.
    """
    name = model_cfg.name.lower()
    hparams = model_cfg.hparams
    if name not in _MODEL_REGISTRY:
        # Lazy-import common modules to avoid circular imports
        # (only import once; registry will be populated via decorators)
        try:
            importlib.import_module("models.mlp")
            importlib.import_module("models.simplecnn")
            importlib.import_module("models.lstm")
            importlib.import_module("models.transformer")
        except Exception:
            pass
    if name not in _MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{name}'. Known: {list(_MODEL_REGISTRY.keys())}")
    return _MODEL_REGISTRY[name](hparams, input_shape, output_shape)
