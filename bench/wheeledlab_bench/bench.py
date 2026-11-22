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
                 warmup: int = 20, smi_every: int = 10, settings: dict | None = None, gpu: int = 0,
                 provenance: dict | None = None, load_limit: float | None = None,
                 clean_rows: int = 0, max_rows: int = 0, outside_limit: float | None = None,
                 clean_sync: float | None = None, sync_probe_n: int = 20):
        self.timer = StageTimer(mode)
        self.record = RunRecord(out_dir, form, n_envs, seed, dict(settings or {}, timers=mode),
                                warmup=warmup, provenance=provenance)
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
        from .host import LoadTrack
        self.load = LoadTrack(load_limit, warmup)
        from .host import CleanRows, OutsideCpu
        try:
            self.outside = OutsideCpu()
        except OSError:                       # no /proc: no outside-CPU columns
            self.outside = None
        # the continuous gate: `clean_rows` clean measured rows, at most `max_rows` measured
        self.gate = (CleanRows(clean_rows, max_rows or 4 * clean_rows, outside_limit,
                               sync_limit=clean_sync)
                     if clean_rows and self.outside is not None and outside_limit is not None
                     else None)
        # --clean-sync: a short kernel+sync probe on the run's GPU at every measured row, outside
        # the row's time (it runs while the record is written)
        self.clean_sync, self.sync_probe_n = clean_sync, sync_probe_n

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
        load1 = os.getloadavg()[0]
        load = {"load1": f"{load1:.2f}"}
        self.load.add(self.iteration, load1)
        if self.outside is not None:
            load.update(self.outside.sample())
            if self.gate is not None and self.iteration >= self.record.warmup:
                sync_ms = None
                if self.clean_sync is not None:
                    sync_ms = self.row_sync_probe()
                    load["row_sync_p90_ms"] = "" if sync_ms is None else sync_ms
                load["clean"] = int(self.gate.add(load["outside_cpu"], sync_ms))
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
        self.record.summary(iterations=self.iteration, closed=False, host_load=self.load.summary(),
                            **self._gate_summary(),
                            **(self.summary_extra() if self.summary_extra else {}))
        verdict = self.gate.verdict() if self.gate is not None else None
        if verdict is not None:
            # the run has its clean rows (or never will): end it here, its record closed; a
            # hard exit, since the training loop and Kit's close would run on
            self.close(stopped_by="clean_rows", exit_code=0 if verdict == "complete" else 7)
            os._exit(0 if verdict == "complete" else 7)
        self._sync()
        self._t0 = time.perf_counter()        # the record's own time is outside every row

    def close(self, **summary) -> None:
        if not self.closed:
            self.closed = True
            extra = self.summary_extra() if self.summary_extra else {}
            self.record.close(iterations=self.iteration, closed=True, host_load=self.load.summary(),
                              **self._gate_summary(), **{**extra, **summary})

    def row_sync_probe(self):
        """This row's kernel+sync p90 (ms), or None when there is no GPU to probe."""
        if self._torch is None:
            return None
        from . import gpu
        try:
            return gpu.sync_probe(self._torch.cuda.current_device(), n=self.sync_probe_n)["p90_ms"]
        except Exception:
            return None

    # -- helpers ------------------------------------------------------------------------------
    def _gate_summary(self) -> dict:
        return {"clean_rows": self.gate.summary()} if self.gate is not None else {}

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
