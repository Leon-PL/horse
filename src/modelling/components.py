"""Explicit component dispatch shared by evaluation and candidate fitting."""
from src.modelling.performance import PerformanceModel, PerformanceSpec, TRAINING_COLUMNS
from src.modelling.spec import ModelSpec
from src.modelling.win import WinModel


def spec_from_dict(value):
    if value.get("kind", "win") == "performance":
        return PerformanceSpec.from_dict(value)
    if value.get("kind", "win") != "win":
        raise ValueError("Unknown modelling component kind")
    return ModelSpec.from_dict({k: v for k, v in value.items() if k != "kind"})


def component_model(spec):
    if isinstance(spec, PerformanceSpec):
        return PerformanceModel(spec)
    if isinstance(spec, ModelSpec):
        return WinModel(spec)
    raise ValueError("Unsupported component recipe")


def training_columns(specs):
    return TRAINING_COLUMNS if any(isinstance(s, PerformanceSpec) for s in specs) else ()
