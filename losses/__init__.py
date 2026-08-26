import os
import importlib

LOSS_REGISTRY = {}


def register_loss(name):
    def decorator(cls):
        if name in LOSS_REGISTRY:
            raise ValueError(f"Loss '{name}' is already registered")
        LOSS_REGISTRY[name] = cls
        return cls
    return decorator


def build_loss(name, **kwargs):
    if name not in LOSS_REGISTRY:
        raise ValueError(f"Loss '{name}' not found. Available losses: {list(LOSS_REGISTRY.keys())}")
    return LOSS_REGISTRY[name](**kwargs)


def list_losses():
    return sorted(LOSS_REGISTRY.keys())


_current_dir = os.path.dirname(__file__)
for _file in os.listdir(_current_dir):
    if _file.endswith(".py") and _file != "__init__.py":
        _module_name = _file[:-3]
        importlib.import_module(f"{__name__}.{_module_name}")
