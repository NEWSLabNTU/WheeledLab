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


def _stat_cpu():
    """(busy, total) jiffies over every CPU, from /proc/stat."""
    with open("/proc/stat") as f:
        v = [int(x) for x in f.readline().split()[1:]]
    idle = v[3] + (v[4] if len(v) > 4 else 0)          # idle + iowait
    return sum(v) - idle, sum(v)


def _tree_jiffies(root: int) -> int:
    """CPU time of `root` and every descendant, from /proc: utime + stime (threads included),
    plus cutime + cstime, the time of children that exited and were waited for, so a
    short-lived child's time is not lost from the tree when it ends."""
    kids, times = {}, {}
    for d in os.listdir("/proc"):
        if not d.isdigit():
            continue
        try:
            with open(f"/proc/{d}/stat") as f:
                st = f.read()
        except OSError:
            continue
        rest = st[st.rindex(")") + 2:].split()      # after the comm field
        pid, ppid = int(d), int(rest[1])
        kids.setdefault(ppid, []).append(pid)
        times[pid] = sum(int(x) for x in rest[11:15])   # utime, stime, cutime, cstime
    total, todo = 0, [root]
    while todo:
        p = todo.pop()
        total += times.get(p, 0)
        todo.extend(kids.get(p, []))
    return total


class OutsideCpu:
    """The host's CPU in use outside this run's own process tree, in cores, over each
    interval between two calls: every CPU's busy time from /proc/stat, less the tree's
    utime + stime. Unlike load1 (a 1-minute average of the run queue, the run's own threads
    in it), it is instantaneous and leaves the run out (133, 2026-11-22)."""

    def __init__(self, root: int | None = None):
        self.root = root or os.getpid()
        self.n = cores()
        self._last = self._read()

    def _read(self):
        busy, total = _stat_cpu()
        return busy, total, _tree_jiffies(self.root)

    def sample(self) -> dict:
        busy, total, own = self._read()
        b0, t0, o0 = self._last
        self._last = (busy, total, own)
        dt = max(total - t0, 1)
        # /proc/stat counts every CPU: dt / ncpu jiffies of wall time
        wall = dt / (os.cpu_count() or self.n)
        host = (busy - b0) / wall
        mine = (own - o0) / wall
        return {"outside_cpu": round(max(host - mine, 0.0), 2), "own_cpu": round(mine, 2)}


class CleanRows:
    """The continuous gate (--clean-rows): a measured row whose outside CPU was over the limit
    is marked unclean; the run ends once it has `need` clean rows, or is refused after
    `ceiling` measured rows. A load spike costs time, not the point."""

    def __init__(self, need: int, ceiling: int, limit: float, sync_limit: float | None = None):
        self.need, self.ceiling, self.limit = need, ceiling, limit
        self.sync_limit = sync_limit          # --clean-sync: the row's kernel+sync p90, ms
        self.clean = self.measured = 0
        self.over_cpu = self.over_sync = 0

    def add(self, outside: float, sync_ms: float | None = None) -> bool:
        """One measured row: clean when its outside CPU is within the limit and, with a sync
        limit, its kernel+sync probe too (a probe that did not run is not clean)."""
        cpu_ok = outside <= self.limit
        sync_ok = (self.sync_limit is None
                   or (sync_ms is not None and sync_ms <= self.sync_limit))
        ok = cpu_ok and sync_ok
        self.measured += 1
        self.clean += ok
        self.over_cpu += not cpu_ok
        self.over_sync += not sync_ok
        return ok

    def verdict(self):
        """None (go on), "complete" or "refused"."""
        if self.clean >= self.need:
            return "complete"
        if self.measured >= self.ceiling:
            return "refused"
        return None

    def summary(self) -> dict:
        out = {"need": self.need, "ceiling": self.ceiling, "limit_cores": self.limit,
               "clean": self.clean, "measured": self.measured, "verdict": self.verdict()}
        if self.sync_limit is not None:
            out.update(sync_limit_ms=self.sync_limit, over_cpu=self.over_cpu,
                       over_sync=self.over_sync)
        return out


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
