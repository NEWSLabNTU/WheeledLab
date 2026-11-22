"""The harness: a StageTimer, the per-iteration record, and the learning curve.

A side uses it one of two ways:

- *instrumented*: `hooks.install(bench)` wraps Isaac Lab's manager-based env and rsl_rl's PPO by
  class, when they are imported, so a training script runs unchanged (stock, both forms);
- *explicit*: a side calls `bench.stage(name)`, `bench.on_env_step(reward, done)` and
  `bench.end_iteration()` itself (ours, whose env is not Isaac Lab's).

Either way the row is the same (`record.COLUMNS`).
"""
from __future__ import annotations

import os
import time

from .record import RunRecord
from .timers import StageTimer


class Bench:
    def __init__(self, out_dir: str, form: str, n_envs: int, seed=None, mode: str = "sync",
                 warmup: int = 20, smi_every: int = 10, settings: dict | None = None, gpu: int = 0):
        self.timer = StageTimer(mode)
        self.record = RunRecord(out_dir, form, n_envs, seed, dict(settings or {}, timers=mode),
                                warmup=warmup)
        self.n_envs, self.smi_every, self.gpu = n_envs, smi_every, gpu
        self.iteration, self.steps = 0, 0
        self._t0 = None                        # this iteration's start
        self._torch = None
        try:
            import torch
            self._torch = torch if torch.cuda.is_available() else None
        except ImportError:
            pass
        self._ret = self._len = None           # per env, on the env's device
        self._done_ret = self._done_len = self._done_n = None
        self._last_smi = 0
        self.closed = False
        self.summary_extra = None             # a callable giving more summary fields (hooks)

    # -- explicit API -------------------------------------------------------------------------
    def stage(self, name: str):
        return self.timer.stage(name)

    def begin(self) -> None:
        """Mark the iteration's start, if it is not marked yet (the first env step does)."""
        if self._t0 is None:
            self._sync()
            self._t0 = time.perf_counter()

    def on_env_step(self, reward, done) -> None:
        """One vectorised env step: the rewards and the done flags of every env (tensors)."""
        self.begin()
        self.steps += 1
        if not self.n_envs:
            self.n_envs = self.record.n_envs = int(reward.numel())
        t = self._torch_mod(reward)
        # the accumulators are inference tensors when the rollout runs under inference_mode
        # (rsl_rl's does), so every update to them runs under it too
        with t.inference_mode():
            if self._ret is None:
                self._ret = t.zeros_like(reward, dtype=t.float32)
                self._len = t.zeros_like(reward, dtype=t.float32)
                self._done_ret = t.zeros((), dtype=t.float64, device=reward.device)
                self._done_len = t.zeros((), dtype=t.float64, device=reward.device)
                self._done_n = t.zeros((), dtype=t.int64, device=reward.device)
            self._ret += reward.reshape(self._ret.shape).float()
            self._len += 1
            d = done.reshape(self._ret.shape).bool()
            self._done_ret += (self._ret * d).sum(dtype=self._done_ret.dtype)
            self._done_len += (self._len * d).sum(dtype=self._done_len.dtype)
            self._done_n += d.sum()
            self._ret *= ~d
            self._len *= ~d

    def end_iteration(self) -> None:
        """The iteration's end (after the policy update): write its row."""
        self.begin()
        self._sync()
        wall = time.perf_counter() - self._t0
        t_rec = time.perf_counter()
        total, own, _calls = self.timer.take()
        if self.timer.enabled:
            total.setdefault("sample", 0.0)
            own.setdefault("sample", 0.0)
            inner = total.get("returns", 0.0) + total.get("update", 0.0)
            total["rollout"] = own["rollout"] = wall - inner
        peak = smi = ""
        # the host's load, every row; the GPU's, with nvidia-smi's sample (the run's own
        # processes are not counted as others)
        load = {"load1": f"{os.getloadavg()[0]:.2f}"}
        if self._torch is not None:
            peak = round(self._torch.cuda.max_memory_allocated() / 2**20)
        if self.smi_every and (self.iteration - self._last_smi >= self.smi_every or self.iteration == 0):
            from . import gpu
            try:
                smi = gpu.own_used_mib()
                o = gpu.others(self.gpu)
                load.update(gpu_util_pct=o["util_pct"], other_procs=o["procs"], other_mib=o["procs_mib"])
                load["sync_p50_ms"] = gpu.sync_probe(self._torch.cuda.current_device(), n=50)["p50_ms"] \
                    if self._torch is not None else ""
            except Exception:
                smi = ""
            self._last_smi = self.iteration
        n = ret = ln = None
        if self._done_n is not None:
            n = int(self._done_n.item())
            if n:
                ret = float(self._done_ret.item()) / n
                ln = float(self._done_len.item()) / n
            with self._torch_mod(self._done_n).inference_mode():
                self._done_ret.zero_(), self._done_len.zero_(), self._done_n.zero_()
        self.record.row(self.iteration, self.steps, wall, total, own, peak, smi, n or 0, ret, ln,
                        time.perf_counter() - t_rec, load)
        self.iteration += 1
        self.steps = 0
        # the summary at every iteration: an app that exits hard (Kit's close) runs no atexit
        self.record.summary(iterations=self.iteration, closed=False,
                            **(self.summary_extra() if self.summary_extra else {}))
        self._sync()
        self._t0 = time.perf_counter()        # the record's own time is outside every row

    def close(self, **summary) -> None:
        if not self.closed:
            self.closed = True
            extra = self.summary_extra() if self.summary_extra else {}
            self.record.close(iterations=self.iteration, closed=True, **{**extra, **summary})

    # -- helpers ------------------------------------------------------------------------------
    def _sync(self) -> None:
        if self._torch is not None:
            self._torch.cuda.synchronize()

    @staticmethod
    def _torch_mod(t):
        import torch
        return torch


def from_env() -> Bench | None:
    """A Bench from WLB_* variables (set by `wheeledlab_bench.launch`), or None."""
    out = os.environ.get("WLB_OUT")
    if not out:
        return None
    return Bench(out, os.environ.get("WLB_FORM", "unnamed"), int(os.environ.get("WLB_N_ENVS", "0")),
                 seed=os.environ.get("WLB_SEED"), mode=os.environ.get("WLB_TIMERS", "sync"),
                 warmup=int(os.environ.get("WLB_WARMUP", "20")))
