"""Instrument Isaac Lab's RL envs and rsl_rl's PPO by class, as they are imported.

Nothing in a training script changes. The classes are found by name in whichever module of
`isaaclab*` or `rsl_rl` defines them, so the same hooks serve Isaac Lab 2.x and 3.x. A target
that is never found is named in the run's summary (`unpatched`), not skipped silently.

A direct env (`DirectRLEnv`) does its work in methods its own subclass defines (a task's
`_apply_action`, `_get_observations`, ...), so those are hooked on the env's own class when it is
built: whichever class in its MRO defines each one, once per class. Their stages are the
manager-based env's names, so both kinds write one schema: `actions` (`_pre_physics_step`,
`_apply_action`), `observations`, `rewards`, `terminations` (`_get_dones`), `resets`
(`_reset_idx`); `physics` and `scene` come from the same SimulationContext and InteractiveScene
hooks.
"""
from __future__ import annotations

import functools
import importlib.abc
import importlib.machinery
import sys

# class name -> {method: stage}; the env's step also feeds the learning curve, PPO.update
# ends the iteration, and the runner's learn() returning ends the run (stage None: not timed)
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
    "DirectRLEnv": {"step": "env_step", "__init__": None},
    "OnPolicyRunner": {"learn": None},
}
PREFIXES = ("isaaclab", "rsl_rl")
ENV_STEPS = ("ManagerBasedRLEnv", "DirectRLEnv")
# a direct env's own methods -> stage (the manager-based env's names for the same work)
DIRECT_STAGES = {"_pre_physics_step": "actions", "_apply_action": "actions",
                 "_get_observations": "observations", "_get_rewards": "rewards",
                 "_get_dones": "terminations", "_reset_idx": "resets"}

_bench = None
patched: dict[str, str] = {}       # "Class.method" -> module
calls: dict[str, int] = {}         # "Class.method" -> calls (the run's summary: per method)


def _wrap(cls, method: str, stage: str):
    fn = cls.__dict__[method]
    bench = _bench

    if cls.__name__ == "DirectRLEnv" and method == "__init__":
        @functools.wraps(fn)
        def init(self, *a, **k):
            fn(self, *a, **k)
            _patch_direct(type(self))
        return init

    if cls.__name__ in ENV_STEPS and method == "step":
        return _env_step(fn, stage, bench, f"{cls.__name__}.{method}")

    if cls.__name__ == "PPO" and method == "update":
        @functools.wraps(fn)
        def update(self, *a, **k):
            calls["PPO.update"] = calls.get("PPO.update", 0) + 1
            bench.timer.start(stage)
            try:
                out = fn(self, *a, **k)
            finally:
                bench.timer.stop(stage)
            bench.end_iteration()
            return out
        return update

    if method == "learn":
        # Kit's close ends the process without atexit, so a run is closed here, when training
        # returns, and summary.json's "closed": true tells a finished run from a crashed one
        @functools.wraps(fn)
        def learn(self, *a, **k):
            out = fn(self, *a, **k)
            bench.close()
            return out
        return learn

    return _timed(fn, stage, bench, f"{cls.__name__}.{method}")


def _env_step(fn, stage, bench, key):
    """An env's step: the env_step stage, and its rewards and dones into the learning curve. A
    step inside a step (a subclass's override calling DirectRLEnv's) is counted once, outside."""
    @functools.wraps(fn)
    def step(self, *a, **k):
        if getattr(self, "_wheeledlab_bench_in_step", False):
            return fn(self, *a, **k)
        calls[key] = calls.get(key, 0) + 1
        bench.begin()
        bench.timer.start(stage)
        self._wheeledlab_bench_in_step = True
        try:
            out = fn(self, *a, **k)
        finally:
            self._wheeledlab_bench_in_step = False
            bench.timer.stop(stage)
        _obs, rew, terminated, truncated = out[:4]
        bench.on_env_step(rew, terminated | truncated)
        return out
    step._wheeledlab_bench = True
    return step


def _timed(fn, stage, bench, key):
    @functools.wraps(fn)
    def timed(self, *a, **k):
        calls[key] = calls.get(key, 0) + 1
        bench.timer.start(stage)
        try:
            return fn(self, *a, **k)
        finally:
            bench.timer.stop(stage)
    timed._wheeledlab_bench = True
    return timed


def _patch_direct(cls) -> None:
    """A direct env's own stage methods, on its class: each resolved through the MRO (the task's
    override, or DirectRLEnv's), wrapped once on `cls`, so a super() call inside one is not timed
    twice. A step the subclass overrides is the env_step stage too."""
    if cls.__dict__.get("_wheeledlab_bench_patched"):
        return
    for method, stage in {**DIRECT_STAGES, "step": "env_step"}.items():
        fn = getattr(cls, method, None)
        if fn is None or getattr(fn, "_wheeledlab_bench", False):
            continue
        owner = next(c for c in cls.__mro__ if method in c.__dict__)
        wrap = _env_step if method == "step" else _timed
        setattr(cls, method, wrap(fn, stage, _bench, f"{cls.__name__}.{method}"))
        patched[f"{cls.__name__}.{method}"] = f"{owner.__module__}.{owner.__name__}"
    cls._wheeledlab_bench_patched = True


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
