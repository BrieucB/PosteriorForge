"""Explicit opt-in loader for the byte-identical NPE-PFN baseline.

Built with PriorLabs-TabPFN. Vendored code is under Prior Labs License 1.1.
The scientific package and CPU CI do not import these heavy dependencies.
"""
from importlib import import_module
from importlib.machinery import ModuleSpec
from types import ModuleType
from pathlib import Path
import sys


def load_baseline():
    """Return the frozen upstream constructor without rewriting its imports."""
    directory = Path(__file__).resolve().parent
    existing = sys.modules.get("npe_pfn")
    if existing is not None:
        location = Path(existing.__file__).resolve()
        if directory not in location.parents:
            raise RuntimeError("A different npe_pfn is already imported; use a clean process")
    # The upstream __init__ eagerly imports TSNPE, an unused training path.
    # SBI 0.23.0's released wheel/sdist omit samplers.score used by that path.
    # Create only the package namespace, then execute the byte-identical core.
    # No inference, support, or rejection implementation is changed.
    if existing is None:
        package = ModuleType("npe_pfn")
        package.__path__ = [str(directory / "npe_pfn")]
        package.__package__ = "npe_pfn"
        package.__file__ = str(directory / "npe_pfn" / "__init__.py")
        package.__spec__ = ModuleSpec("npe_pfn", loader=None, is_package=True)
        package.__spec__.submodule_search_locations = package.__path__
        sys.modules["npe_pfn"] = package
    return import_module("npe_pfn.npe_pfn").TabPFN_Based_NPE_PFN


def frozen_regressor_kwargs(model_path, *, device="cuda"):
    """Explicit FP32 configuration matching pinned 2.0.9 ensemble defaults.

    Raw observations enter TabPFN; its upstream default feature and target
    preprocessors remain enabled. No caller-provided label may enter queries.
    """
    import torch

    return {
        "n_estimators": 8,
        "categorical_features_indices": None,
        "softmax_temperature": 0.9,
        "average_before_softmax": False,
        "model_path": str(Path(model_path).resolve()),
        "device": device,
        "ignore_pretraining_limits": False,
        "inference_precision": torch.float32,
        "fit_mode": "fit_preprocessors",
        "memory_saving_mode": "auto",
        "random_state": 0,
        "n_jobs": 1,
        "inference_config": None,
    }
