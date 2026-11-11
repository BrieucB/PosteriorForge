"""Explicit, source-guarded TabPFN2.0.9 finite-CDF arithmetic compatibility.

Built with PriorLabs-TabPFN. Importing this module requires only the standard
library. No SDK file, default inference entrypoint, sampler or tail family is
changed. Activation replaces one regressor-module binding for the context only.
"""
from __future__ import annotations

import ast
import __future__
from contextlib import contextmanager
import hashlib
from importlib import import_module, metadata
import inspect
from pathlib import Path
from threading import Lock
from types import FunctionType, MappingProxyType

VARIANT = "cdf_arithmetic_v1"
ADAPTER_ID = "tabpfn-2.0.9-finite-cdf-parentheses-v1"
ADAPTER_VERSION = 1
PINNED_TABPFN_VERSION = "2.0.9"
PINNED_SDK_SHA256 = MappingProxyType({
    "tabpfn/utils.py": "a0521d68cd2880e041a49ac5acee42a2b8d1297e246e3f7a9c477e49f5e97fc1",
    "tabpfn/regressor.py": "bbb7d6dec265b18a30a54ff28719c957bc05d4164f693f76d998380132d2c46e",
})
_ACTIVE = Lock()


class CompatibilityError(RuntimeError):
    """The explicitly requested compatibility contract could not be verified."""


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def adapter_source_sha256():
    """Return this adapter's source hash without importing Torch or TabPFN."""
    return _sha256(__file__)


def _load_sdk():
    # No tensor operation, random draw or seed operation is performed here.
    return (metadata.version("tabpfn"), import_module("tabpfn.utils"),
            import_module("tabpfn.regressor"))


def _sdk_hashes(utils, regressor):
    return {"tabpfn/utils.py": _sha256(utils.__file__),
            "tabpfn/regressor.py": _sha256(regressor.__file__)}


def _guard_sdk(version, utils, regressor):
    if version != PINNED_TABPFN_VERSION:
        raise CompatibilityError("CDF adapter requires exactly TabPFN2.0.9")
    hashes = _sdk_hashes(utils, regressor)
    if hashes != dict(PINNED_SDK_SHA256):
        raise CompatibilityError("CDF adapter SDK source hash mismatch")
    translator = utils.translate_probs_across_borders
    helper = utils._cdf
    mapper = utils._map_to_bucket_ix
    if (regressor.translate_probs_across_borders is not translator
            or translator.__globals__.get("_cdf") is not helper
            or helper.__globals__.get("_map_to_bucket_ix") is not mapper):
        raise CompatibilityError("CDF adapter found conflicting SDK aliases")
    for function, expected_name in ((translator, "translate_probs_across_borders"),
                                     (helper, "_cdf"), (mapper, "_map_to_bucket_ix")):
        if (not isinstance(function, FunctionType)
                or function.__module__ != "tabpfn.utils"
                or function.__name__ != expected_name
                or Path(inspect.getfile(function)).resolve() != Path(utils.__file__).resolve()):
            raise CompatibilityError("CDF adapter SDK function identity mismatch")
    return hashes, translator, helper, mapper


def _corrected_helper(original):
    """Compile the exact guarded function, changing one expression only."""
    tree = ast.parse(inspect.getsource(original))
    expected = ast.parse(
        "ys - borders[y_buckets] / bucket_widths[y_buckets]", mode="eval").body
    matches = [node for node in ast.walk(tree)
               if isinstance(node, ast.Assign)
               and any(isinstance(target, ast.Name) and target.id == "share_of_bucket_left"
                       for target in node.targets)
               and ast.dump(node.value, include_attributes=False)
               == ast.dump(expected, include_attributes=False)]
    if len(matches) != 1:
        raise CompatibilityError("CDF adapter interpolation expression mismatch")
    matches[0].value = ast.parse(
        "(ys - borders[y_buckets]) / bucket_widths[y_buckets]", mode="eval").body
    ast.fix_missing_locations(tree)
    ast.increment_lineno(tree, original.__code__.co_firstlineno - 1)
    namespace = dict(original.__globals__)
    compiled = compile(tree, original.__code__.co_filename, "exec",
                       flags=original.__code__.co_flags & __future__.annotations.compiler_flag,
                       dont_inherit=True)
    exec(compiled, namespace)
    return namespace[original.__name__]


@contextmanager
def finite_cdf_compatibility():
    """Activate the CDF-only adapter explicitly; yield mutable provenance.

    Serialize the yielded dictionary after exiting: restoration and final SDK
    hashes are recorded on exit. Nested/concurrent or conflicting activation
    fails closed. The original alias is restored even when the body raises.
    """
    if not _ACTIVE.acquire(blocking=False):
        raise CompatibilityError("CDF adapter is already active")
    try:
        version, utils, regressor = _load_sdk()
        hashes, original, helper, mapper = _guard_sdk(version, utils, regressor)
        fixed_helper = _corrected_helper(helper)
        descriptor = {
            "schema": "posteriorforge.compat.finite_cdf.v1",
            "variant": VARIANT, "adapter_id": ADAPTER_ID,
            "adapter_version": ADAPTER_VERSION,
            "adapter_sha256": adapter_source_sha256(),
            "tabpfn_version": version, "sdk_source_sha256": hashes,
            "activation": "explicit", "scope": "finite_ensemble_cdf_parentheses_only",
            "correction": "finite_cdf_parentheses_only",
            "source_guard_passed": True, "sdk_files_modified": False,
            "tail_family_changed": False, "execution_optimized": False,
            "original_alias_restored": False, "patched_call_count": 0,
            "translator_code_unchanged": True,
        }

        def counted_helper(*args, **kwargs):
            value = fixed_helper(*args, **kwargs)
            descriptor["patched_call_count"] += 1
            return value

        namespace = dict(original.__globals__)
        namespace["_cdf"] = counted_helper
        replacement = FunctionType(original.__code__, namespace, original.__name__,
                                   original.__defaults__, original.__closure__)
        replacement.__kwdefaults__ = original.__kwdefaults__
        replacement.__annotations__ = original.__annotations__.copy()
        replacement.__module__ = original.__module__
        replacement.__qualname__ = original.__qualname__
        if replacement.__code__ is not original.__code__:
            raise CompatibilityError("CDF adapter translator code changed")
        regressor.translate_probs_across_borders = replacement
        try:
            yield descriptor
        finally:
            conflicting_alias = regressor.translate_probs_across_borders is not replacement
            regressor.translate_probs_across_borders = original
            descriptor["original_alias_restored"] = (
                regressor.translate_probs_across_borders is original)
            descriptor["sdk_source_sha256_after"] = _sdk_hashes(utils, regressor)
            descriptor["sdk_sources_unchanged"] = descriptor["sdk_source_sha256_after"] == hashes
            descriptor["sdk_aliases_unchanged"] = (
                utils.translate_probs_across_borders is original
                and utils._cdf is helper and utils._map_to_bucket_ix is mapper)
            if conflicting_alias or not descriptor["sdk_aliases_unchanged"]:
                raise CompatibilityError("CDF adapter detected a conflicting alias change")
            if not descriptor["sdk_sources_unchanged"]:
                raise CompatibilityError("CDF adapter SDK sources changed during activation")
    finally:
        _ACTIVE.release()
