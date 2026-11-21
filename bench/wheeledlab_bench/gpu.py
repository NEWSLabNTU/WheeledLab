"""The GPU's identity and load, from nvidia-smi: what a run names, and the idle check before it."""
from __future__ import annotations

import os
import subprocess


def _smi(*args: str) -> list[list[str]]:
    out = subprocess.run(["nvidia-smi", *args, "--format=csv,noheader,nounits"],
                         capture_output=True, text=True, timeout=30, check=True).stdout
    return [[c.strip() for c in line.split(",")] for line in out.strip().splitlines() if line.strip()]


def gpus() -> list[dict]:
    rows = _smi("--query-gpu=index,uuid,name,driver_version,memory.total,memory.used,utilization.gpu")
    return [dict(index=int(r[0]), uuid=r[1], name=r[2], driver=r[3], total_mib=int(r[4]),
                 used_mib=int(r[5]), util_pct=int(r[6])) for r in rows]


def apps() -> list[dict]:
    rows = _smi("--query-compute-apps=gpu_uuid,pid,used_memory")
    return [dict(uuid=r[0], pid=int(r[1]), used_mib=int(r[2]) if r[2].isdigit() else 0) for r in rows]


def own_used_mib(pid: int | None = None) -> int:
    """The memory nvidia-smi charges to this process (all GPUs), MiB."""
    pid = pid or os.getpid()
    return sum(a["used_mib"] for a in apps() if a["pid"] == pid)


def others(index: int, pid: int | None = None) -> dict:
    """The other processes on GPU `index`: their count, memory, and the GPU's utilisation."""
    pid = pid or os.getpid()
    g = next(g for g in gpus() if g["index"] == index)
    mine = {pid, os.getppid()}
    theirs = [a for a in apps() if a["uuid"] == g["uuid"] and a["pid"] not in mine]
    return dict(gpu=index, name=g["name"], driver=g["driver"], util_pct=g["util_pct"],
                procs=len(theirs), procs_mib=sum(a["used_mib"] for a in theirs),
                pids=[a["pid"] for a in theirs])
