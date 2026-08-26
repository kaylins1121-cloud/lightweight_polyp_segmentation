import os
import importlib

MODEL_REGISTRY = {}


def register_model(name):
    def decorator(cls):
        if name in MODEL_REGISTRY:
            raise ValueError(f"Model '{name}' is already registered")
        MODEL_REGISTRY[name] = cls
        return cls
    return decorator


def build_model(name, **kwargs):
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Model '{name}' not found. Available models: {list(MODEL_REGISTRY.keys())}")
    return MODEL_REGISTRY[name](**kwargs)


def list_models():
    return sorted(MODEL_REGISTRY.keys())


_current_dir = os.path.dirname(__file__)
for _file in os.listdir(_current_dir):
    if _file.endswith(".py") and _file != "__init__.py":
        _module_name = _file[:-3]
        importlib.import_module(f"{__name__}.{_module_name}")
