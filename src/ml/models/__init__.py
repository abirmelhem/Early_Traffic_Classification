"""Lazy model dispatch: requesting one model never imports another model backend."""
from importlib import import_module

MODELS = {"logistic_regression": "logistic_regression", "random_forest": "random_forest", "xgboost": "xgboost_model"}


def model_module(name):
    if name not in MODELS:
        raise ValueError(f"Unknown model: {name}")
    return import_module(f"{__name__}.{MODELS[name]}")
