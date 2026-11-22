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


# a display server's own context on every GPU is not another run
DISPLAY_SERVERS = ("Xorg", "Xwayland", "gnome-shell")


def parse_processes(xml_text: str) -> list[dict]:
    """Every process holding a context on any GPU, from `nvidia-smi -q -x`: compute, graphics
    (G) or both (C+G), display servers left out."""
    import xml.etree.ElementTree as ET
    out = []
    for g in ET.fromstring(xml_text).iter("gpu"):
        uuid = g.findtext("uuid") or ""
        for p in g.iter("process_info"):
            name = (p.findtext("process_name") or "").rsplit("/", 1)[-1]
            if name in DISPLAY_SERVERS:
                continue
            mem = (p.findtext("used_memory") or "").split()
            out.append(dict(uuid=uuid, pid=int(p.findtext("pid") or 0), type=p.findtext("type") or "",
                            name=name, used_mib=int(mem[0]) if mem and mem[0].isdigit() else 0))
    return out


def apps() -> list[dict]:
    """The processes on every GPU, graphics contexts included: --query-compute-apps lists no
    graphics-only context, so a renderer on the wrong GPU would not show (094 P3, 2026-11-22)."""
    xml_text = subprocess.run(["nvidia-smi", "-q", "-x"], capture_output=True, text=True,
                              timeout=30, check=True).stdout
    return parse_processes(xml_text)


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


def util_watch(index: int, seconds: int = 10) -> list[int]:
    """The GPU's utilisation, %, sampled once a second for `seconds`."""
    import time
    out = []
    for k in range(seconds):
        out.append(next(g for g in gpus() if g["index"] == index)["util_pct"])
        if k + 1 < seconds:
            time.sleep(1.0)
    return out


def sync_probe(index: int, n: int = 200) -> dict:
    """The latency of a one-element kernel plus a device synchronisation on GPU `index`, ms.

    On a free GPU it is about 0.01 ms. A GPU time-sliced with another process's busy context
    makes every synchronising call wait for that context's slice (2.3 ms measured on 241 against
    a 100 %-busy job), and Isaac Lab's articulation step makes several such calls per physics
    substep, so its throughput falls about 100x while Cartpole's barely moves (scensim 094)."""
    import time
    import torch
    dev = torch.device(f"cuda:{index}")
    x = torch.ones(8, device=dev)
    torch.cuda.synchronize(dev)
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        x.add_(1)
        torch.cuda.synchronize(dev)
        ts.append((time.perf_counter() - t0) * 1e3)
    ts.sort()
    return dict(p50_ms=round(ts[n // 2], 4), p90_ms=round(ts[int(n * 0.9)], 4), max_ms=round(ts[-1], 4))
