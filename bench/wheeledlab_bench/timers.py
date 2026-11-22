"""Stage timers: wall-clock time per named stage, nested, with an optional device synchronisation
at every stage boundary so that asynchronous GPU work is charged to the stage that launched it.

Synchronising costs time. That cost is part of what the harness reports: a run with timers
`off` records only the iteration's wall time, and the difference to a run with timers `sync` is
the harness's overhead.
"""
from __future__ import annotations

import threading
import time
from collections import defaultdict
from contextlib import contextmanager

MODES = ("sync", "nosync", "off")


class StageTimer:
    def __init__(self, mode: str = "sync"):
        if mode not in MODES:
            raise ValueError(f"timer mode {mode!r}: one of {MODES}")
        self.mode = mode
        self._sync = None
        if mode == "sync":
            try:
                import torch
                if torch.cuda.is_available():
                    self._sync = torch.cuda.synchronize
            except ImportError:
                pass
        self._stack: list[list] = []          # [stage, t0, children_s]
        self.total = defaultdict(float)       # stage -> inclusive seconds, this iteration
        self.own = defaultdict(float)         # stage -> exclusive seconds (children taken out)
        self.calls = defaultdict(int)
        self._lock = threading.Lock()         # add() comes from other threads

    @property
    def enabled(self) -> bool:
        return self.mode != "off"

    def _now(self) -> float:
        if self._sync is not None:
            self._sync()
        return time.perf_counter()

    def start(self, stage: str) -> None:
        if not self.enabled:
            return
        self._stack.append([stage, self._now(), 0.0])

    def stop(self, stage: str) -> None:
        if not self.enabled:
            return
        name, t0, children = self._stack.pop()
        if name != stage:
            raise RuntimeError(f"stage {stage!r} stopped inside {name!r}")
        dt = self._now() - t0
        if self._stack:
            self._stack[-1][2] += dt
        if any(s[0] == stage for s in self._stack):   # re-entered (a reset inside a reset)
            return
        self.total[stage] += dt
        self.own[stage] += dt - children
        self.calls[stage] += 1

    def add(self, stage: str, dt: float) -> None:
        """Time spent in `stage` on another thread (a background refill): counted in full, in
        both columns (it has no children), never on the nesting stack, never synchronised."""
        if not self.enabled:
            return
        with self._lock:
            self.total[stage] += dt
            self.own[stage] += dt
            self.calls[stage] += 1

    @contextmanager
    def stage(self, name: str):
        self.start(name)
        try:
            yield
        finally:
            self.stop(name)

    def take(self) -> tuple[dict, dict, dict]:
        """This iteration's (inclusive, exclusive, calls) per stage, and start the next."""
        with self._lock:
            out = dict(self.total), dict(self.own), dict(self.calls)
            self.total.clear(), self.own.clear(), self.calls.clear()
        return out
