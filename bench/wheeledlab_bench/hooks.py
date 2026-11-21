"""Instrument Isaac Lab's manager-based RL env and rsl_rl's PPO by class, as they are imported.

Nothing in a training script changes. The classes are found by name in whichever module of
`isaaclab*` or `rsl_rl` defines them, so the same hooks serve Isaac Lab 2.x and 3.x. A target
that is never found is named in the run's summary (`unpatched`), not skipped silently.
"""
from __future__ import annotations

import functools
import importlib.abc
import importlib.machinery
import sys

# class name -> {method: stage}; the env's step also feeds the learning curve, and PPO.update
# ends the iteration
TARGETS = {
    "ManagerBasedRLEnv": {"step": "env_step", "_reset_idx": "resets"},
    "ActionManager": {"process_action": "actions", "apply_action": "actions"},
    "SimulationContext": {"step": "physics", "render": "render"},
    "InteractiveScene": {"write_data_to_sim": "scene", "update": "scene"},
    "TerminationManager": {"compute": "terminations"},
    "RewardManager": {"compute": "rewards"},
    "CommandManager": {"compute": "commands"},
    "EventManager": {"apply": "events"},
    "ObservationManager": {"compute": "observations"},
    "PPO": {"act": "inference", "process_env_step": "storage", "compute_returns": "returns",
            "update": "update"},
}
PREFIXES = ("isaaclab", "rsl_rl")

_bench = None
patched: dict[str, str] = {}       # "Class.method" -> module


def _wrap(cls, method: str, stage: str):
    fn = cls.__dict__[method]
    bench = _bench

    if cls.__name__ == "ManagerBasedRLEnv" and method == "step":
        @functools.wraps(fn)
        def step(self, *a, **k):
            bench.begin()
            bench.timer.start(stage)
            try:
                out = fn(self, *a, **k)
            finally:
                bench.timer.stop(stage)
            _obs, rew, terminated, truncated = out[:4]
            bench.on_env_step(rew, terminated | truncated)
            return out
        return step

    if cls.__name__ == "PPO" and method == "update":
        @functools.wraps(fn)
        def update(self, *a, **k):
            bench.timer.start(stage)
            try:
                out = fn(self, *a, **k)
            finally:
                bench.timer.stop(stage)
            bench.end_iteration()
            return out
        return update

    @functools.wraps(fn)
    def timed(self, *a, **k):
        bench.timer.start(stage)
        try:
            return fn(self, *a, **k)
        finally:
            bench.timer.stop(stage)
    return timed


def _patch_module(mod) -> None:
    for cname, methods in TARGETS.items():
        cls = mod.__dict__.get(cname)
        if not isinstance(cls, type) or cls.__module__ != mod.__name__:
            continue
        for method, stage in methods.items():
            key = f"{cname}.{method}"
            if key in patched or method not in cls.__dict__:
                continue
            setattr(cls, method, _wrap(cls, method, stage))
            patched[key] = mod.__name__


class _Loader(importlib.abc.Loader):
    def __init__(self, inner):
        self.inner = inner

    def create_module(self, spec):
        return self.inner.create_module(spec)

    def exec_module(self, module):
        self.inner.exec_module(module)
        _patch_module(module)

    def __getattr__(self, name):        # e.g. get_resource_reader, is_package
        return getattr(self.inner, name)


class _Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path, target=None):
        if not name.startswith(PREFIXES):
            return None
        for finder in sys.meta_path:
            if finder is self or not hasattr(finder, "find_spec"):
                continue
            spec = finder.find_spec(name, path, target)
            if spec is not None:
                if spec.loader is not None and hasattr(spec.loader, "exec_module"):
                    spec.loader = _Loader(spec.loader)
                return spec
        return None


def install(bench) -> None:
    """Hook `bench` into the classes above: those imported already, and those imported later."""
    global _bench
    _bench = bench
    for name, mod in list(sys.modules.items()):
        if name.startswith(PREFIXES) and mod is not None:
            _patch_module(mod)
    sys.meta_path.insert(0, _Finder())


def unpatched() -> list[str]:
    return [f"{c}.{m}" for c, ms in TARGETS.items() for m in ms if f"{c}.{m}" not in patched]
