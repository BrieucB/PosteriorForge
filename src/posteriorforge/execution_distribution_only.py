"""Private, source-guarded distribution-only TabPFN prediction.

Built with PriorLabs-TabPFN. The installed SDK remains untouched. This opt-in
API preserves its prediction AST through the final CPU logits, including the
sklearn configuration decorator, and returns the actual criterion and logits.
The caller owns any corrected-CDF/capture context and model RNG isolation.
No old executor or runner imports this module.
"""
from __future__ import annotations

import __future__
import ast
import copy
from functools import lru_cache
import hashlib
from importlib import import_module, metadata
import inspect
import marshal
from pathlib import Path
from types import CodeType, FunctionType

PINNED_SOURCES = {
    "regressor.py": "bbb7d6dec265b18a30a54ff28719c957bc05d4164f693f76d998380132d2c46e",
    "utils.py": "a0521d68cd2880e041a49ac5acee42a2b8d1297e246e3f7a9c477e49f5e97fc1",
    "preprocessing.py": "fba9df553f3f7d190972a531c5627f09b88e8b8f537d298df7083f7dd972843d",
}
PREDICT_AST_SHA256 = "84a768596f7514279ebae5f19ae6bcc2f8da150cf34ca8aed86cea8f82a588e7"
PREFIX_AST_SHA256 = "801c08bc3e0d44732efd8a5b40241acfdd4143284e9455506d92f42f6a501a90"
SUFFIX_AST_SHA256 = "bc54807721c7f9876dfafd415ff1b7eaaba2e8103315d68e1167cec59c2ae232"
_MISSING = object()


class DistributionOnlyError(RuntimeError):
    pass


def _require(condition, message):
    if not condition:
        raise DistributionOnlyError(message)


def _sha(value):
    return hashlib.sha256(value).hexdigest()


def _ast_bytes(nodes):
    return ast.dump(ast.Module(body=nodes, type_ignores=[]),
                    include_attributes=False).encode()


def _implementation_code(module_code):
    owner = next(c for c in module_code.co_consts
                 if isinstance(c, CodeType) and c.co_name == "TabPFNRegressor")
    return [c for c in owner.co_consts
            if isinstance(c, CodeType) and c.co_name == "predict"][-1]


def _compile_clone(source, filename):
    """Compile without executing the SDK module, preserving its symbol table."""
    tree = ast.parse(source)
    owner = next(n for n in tree.body
                 if isinstance(n, ast.ClassDef) and n.name == "TabPFNRegressor")
    function = [n for n in owner.body
                if isinstance(n, ast.FunctionDef) and n.name == "predict"][-1]
    _require(_sha(ast.dump(function, include_attributes=False).encode()) ==
             PREDICT_AST_SHA256, "pinned predict AST differs")
    boundary = ast.dump(ast.parse("logits = logits.cpu()").body[0],
                        include_attributes=False)
    matches = [i for i, statement in enumerate(function.body)
               if ast.dump(statement, include_attributes=False) == boundary]
    _require(len(matches) == 1, "ambiguous final-logit boundary")
    cut = matches[0] + 1
    _require(cut == 17 and len(function.body[cut:]) == 3,
             "unexpected prediction prefix/suffix structure")
    prefix = _ast_bytes(function.body[:cut])
    _require(_sha(prefix) == PREFIX_AST_SHA256 and
             _sha(_ast_bytes(function.body[cut:])) == SUFFIX_AST_SHA256,
             "prediction prefix or summary suffix differs")
    flags = __future__.annotations.compiler_flag
    trusted_code = _implementation_code(compile(
        tree, filename, "exec", flags=flags, dont_inherit=True))
    clone_tree = copy.deepcopy(tree)
    clone_owner = next(n for n in clone_tree.body
                       if isinstance(n, ast.ClassDef) and n.name == "TabPFNRegressor")
    clone = [n for n in clone_owner.body
             if isinstance(n, ast.FunctionDef) and n.name == "predict"][-1]
    tail = ast.parse(
        "return {'criterion': self.renormalized_criterion_, 'logits': logits}"
    ).body[0]
    ast.copy_location(tail, clone.body[cut])
    clone.body = clone.body[:cut] + [tail]
    _require(_ast_bytes(clone.body[:cut]) == prefix,
             "retained prediction AST changed")
    ast.fix_missing_locations(clone_tree)
    private_code = _implementation_code(compile(
        clone_tree, filename, "exec", flags=flags, dont_inherit=True))
    return trusted_code, private_code


class _Guard:
    def __init__(self):
        _require(metadata.version("tabpfn") == "2.0.9", "exact TabPFN 2.0.9 required")
        self.module = import_module("tabpfn.regressor")
        self.cls = self.module.TabPFNRegressor
        self.wrapper = self.cls.predict
        self.original = inspect.unwrap(self.wrapper)
        self.root = Path(self.module.__file__).parent
        self._verify_sources()
        self.trusted_code, self.private_code = _compile_clone(
            (self.root / "regressor.py").read_text(), self.original.__code__.co_filename)
        _require(self.original.__code__ == self.trusted_code,
                 "loaded SDK implementation differs from pinned source")
        _require(self.original.__globals__ is vars(self.module) and
                 not self.original.__code__.co_freevars,
                 "unexpected SDK implementation globals or closure")
        _require(self.wrapper is not self.original and
                 self.wrapper.__code__.co_freevars == ("func", "self"),
                 "original sklearn configuration decorator required")
        cells = dict(zip(self.wrapper.__code__.co_freevars,
                         self.wrapper.__closure__))
        _require(cells["func"].cell_contents is self.original,
                 "unexpected decorator target")
        context = cells["self"].cell_contents
        expected_context = self.module.config_context(transform_output="default")
        _require(type(context) is type(expected_context) and
                 context.func is expected_context.func and
                 context.args == expected_context.args and
                 context.kwds == expected_context.kwds,
                 "original sklearn default-output context required")
        expected_context.gen.close()
        self.context = context
        self.context_type = type(context)
        self.context_values = dict(vars(context))
        self.context_values["kwds"] = dict(context.kwds)
        self.wrapper_code = self.wrapper.__code__
        self._verify()

    def _verify_sources(self):
        hashes = {name: _sha((self.root / name).read_bytes())
                  for name in PINNED_SOURCES}
        _require(hashes == PINNED_SOURCES, "SDK source hash mismatch")
        return hashes

    def _verify(self):
        self._verify_sources()
        _require(self.module.TabPFNRegressor is self.cls and
                 vars(self.cls).get("predict") is self.wrapper and
                 self.wrapper.__code__ is self.wrapper_code and
                 getattr(self.wrapper, "__wrapped__", None) is self.original and
                 self.original.__code__ == self.trusted_code and
                 self.original.__globals__ is vars(self.module),
                 "SDK class, method alias, code or globals changed")
        cells = dict(zip(self.wrapper.__code__.co_freevars, self.wrapper.__closure__))
        _require(cells["func"].cell_contents is self.original and
                 type(cells["self"].cell_contents) is self.context_type and
                 vars(cells["self"].cell_contents) == self.context_values,
                 "SDK decorator context changed")

    def snapshot(self, regressor):
        self._verify()
        return (tuple(sorted((name, id(value))
                             for name, value in vars(self.module).items())),
                vars(regressor).get("predict", _MISSING))

    def verify_snapshot(self, regressor, before):
        self._verify()
        globals_before, instance_predict = before
        _require(tuple(sorted((name, id(value))
                              for name, value in vars(self.module).items())) ==
                 globals_before and
                 vars(regressor).get("predict", _MISSING) is instance_predict,
                 "SDK global or instance prediction alias changed during call")

    def call(self, regressor, query):
        _require(type(regressor) is self.cls,
                 "exact SDK regressor instance required")
        torch = import_module("torch")
        _require(regressor.use_autocast_ is False and
                 regressor.forced_inference_dtype_ == torch.float32,
                 "distribution-only API is restricted to the frozen FP32 path")
        before = self.snapshot(regressor)
        # A fresh namespace observes the caller's current corrected translator,
        # target-border observer and component-mass capture on every call.
        namespace = dict(self.original.__globals__)
        private = FunctionType(self.private_code, namespace,
                               self.original.__name__, self.original.__defaults__)
        private.__kwdefaults__ = dict(self.original.__kwdefaults__ or {})
        private.__annotations__ = dict(self.original.__annotations__)
        private = self.context(private)
        try:
            result = private(regressor, query, output_type="full", quantiles=[])
            _require(set(result) == {"criterion", "logits"} and
                     result["criterion"] is regressor.renormalized_criterion_,
                     "actual criterion/logits result required")
            _require(torch.is_tensor(result["logits"]) and
                     result["logits"].dtype == torch.float32 and
                     result["logits"].device.type == "cpu",
                     "original CPU FP32 distribution logits required")
            return result
        finally:
            self.verify_snapshot(regressor, before)

    def identity(self):
        self._verify()
        return {
            "schema": "posteriorforge.distribution_only_predict_identity.v1",
            "attribution": "Built with PriorLabs-TabPFN",
            "sdk_version": "2.0.9",
            "sdk_sources": dict(PINNED_SOURCES),
            "sdk_predict_ast_sha256": PREDICT_AST_SHA256,
            "retained_prefix_ast_sha256": PREFIX_AST_SHA256,
            "removed_summary_suffix_ast_sha256": SUFFIX_AST_SHA256,
            "original_implementation_code_sha256": _sha(marshal.dumps(self.original.__code__)),
            "private_implementation_code_sha256": _sha(marshal.dumps(self.private_code)),
            "retained_prefix_statements": 17,
            "removed_suffix_statements": 3,
            "sklearn_configuration_decorator_preserved": True,
            "globals_rebound_per_call": True,
            "return_keys": ["criterion", "logits"],
            "fake_summary_values_returned": False,
            "existing_engines_or_runner_changed": False,
            "qualified_for_production": False,
        }


@lru_cache(maxsize=1)
def _guard():
    return _Guard()


def predict_distribution(regressor, query):
    """Return only the actual criterion and logits, with original FP32 math.

    This does not bind or replace any SDK method. A caller-owned CDF or capture
    alias is copied into the private function's globals afresh for this call.
    Model preparation, neural execution and caller RNG isolation are unchanged.
    """
    return _guard().call(regressor, query)


def execution_metadata():
    """Verify and describe the private implementation; no model/CUDA calls."""
    return _guard().identity()
