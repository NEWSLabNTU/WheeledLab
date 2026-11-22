"""The host's CPU, for `--require-idle`: an exclusive GPU is not enough when the host's CPUs are
busy (the env's host-side work, the Python loop and the launches compete for them). On 241, the
P3 commit check's runs at a 1-minute load of 24 to 46 on 24 cores ran 15 % below those at 15 to
22, the GPU idle throughout (094 P3, 2026-11-22).

The gate is the 1-minute load average against a fraction of the cores this process may run on.
The run's own load is not in it yet when the gate is taken; the run adds about 1 to 2.
"""
from __future__ import annotations

import os
import time


def cores() -> int:
    try:
        return len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        return os.cpu_count() or 1


def load1() -> float:
    return os.getloadavg()[0]


def gate(max_frac: float, wait_s: float = 0.0, poll_s: float = 10.0, _load=None,
         _sleep=None) -> dict:
    """Wait up to `wait_s` for the 1-minute load to fall to `max_frac` x cores. Returns the
    check: load1, cores, limit, waited_s, ok (and every sample taken while waiting)."""
    _load, _sleep = _load or load1, _sleep or time.sleep
    n = cores()
    limit = max_frac * n
    samples, waited = [round(_load(), 2)], 0.0
    while samples[-1] > limit and waited < wait_s:
        _sleep(poll_s)
        waited += poll_s
        samples.append(round(_load(), 2))
    return {"load1": samples[-1], "cores": n, "max_frac": max_frac, "limit": round(limit, 2),
            "waited_s": waited, "ok": samples[-1] <= limit, "samples": samples}


class LoadTrack:
    """The load of each measured row against the gate's limit, for the summary: a run whose rows
    went over it was contended part of the time, whatever the gate said at its start."""

    def __init__(self, limit: float | None, warmup: int):
        self.limit, self.warmup = limit, warmup
        self.values: list[float] = []

    def add(self, iteration: int, value: float) -> None:
        if iteration >= self.warmup:
            self.values.append(value)

    def summary(self) -> dict:
        v = sorted(self.values)
        if not v:
            return {"limit": self.limit, "rows": 0}
        out = {"limit": self.limit, "rows": len(v), "median": v[len(v) // 2], "max": v[-1]}
        if self.limit is not None:
            out["rows_over_limit"] = sum(x > self.limit for x in v)
        return out
