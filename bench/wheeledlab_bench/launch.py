"""Run a training script unchanged, with the harness hooked in.

    python -m wheeledlab_bench.launch --out DIR --form NAME [--timers sync|nosync|off]
        [--warmup 20] [--seed S] [--gpu 0] [--require-idle]
        [--sample-hook MOD:CLASS.METHOD ...] -- SCRIPT [ARGS...]

The script runs as `__main__` with ARGS as its argv. `--require-idle` refuses to start while
another process holds GPU `--gpu`, unless `--idle-others-ok` and the GPU's utilisation stays at
0 % over a 10 s watch (another user's process that holds memory and runs nothing, such as a
loaded model server); the check's result goes into the run's settings either way, and the CSV
records the other processes and the utilisation every few iterations, so one that wakes up
during the run shows.
"""
from __future__ import annotations

import argparse
import atexit
import json
import os
import runpy
import sys

from . import hooks
from .bench import Bench


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if "--" not in argv:
        print(__doc__, file=sys.stderr)
        return 2
    i = argv.index("--")
    ap = argparse.ArgumentParser(prog="wheeledlab_bench.launch")
    ap.add_argument("--out", required=True)
    ap.add_argument("--form", required=True)
    ap.add_argument("--timers", default="sync", choices=("sync", "nosync", "off"))
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--seed", default=None)
    ap.add_argument("--n-envs", type=int, default=0, help="0: from the env's first step")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--require-idle", action="store_true",
                    help="refuse while another process holds the GPU")
    ap.add_argument("--idle-others-ok", action="store_true",
                    help="with --require-idle: accept other processes when the GPU's utilisation "
                         "stays at 0 %% over a 10 s watch")
    ap.add_argument("--max-sync-ms", type=float, default=0.05,
                    help="refuse when a kernel+sync takes longer (p90: an other user busy in bursts "
                         "leaves p50 low); a free GPU takes ~0.01 ms")
    ap.add_argument("--allow-shared", action="store_true",
                    help="run anyway on a shared GPU (a smoke, not a measurement); the probe is recorded")
    ap.add_argument("--sample-hook", action="append", default=[], metavar="MOD:CLASS.METHOD",
                    help="time the `sample` stage around this method (repeatable): a side that "
                         "draws its episode starts on the host, e.g. a pool's refill and draw")
    a = ap.parse_args(argv[:i])
    script, rest = argv[i + 1], argv[i + 2:]

    from . import gpu
    try:
        idle = gpu.others(a.gpu)
    except Exception as e:
        idle = {"error": str(e)}
    if a.require_idle and idle.get("procs") and a.idle_others_ok:
        idle["util_watch_pct"] = gpu.util_watch(a.gpu)
    if a.require_idle and idle.get("procs") and (not a.idle_others_ok or max(idle["util_watch_pct"]) > 0):
        print(f"wheeledlab_bench: GPU {a.gpu} is not idle: {json.dumps(idle)}", file=sys.stderr)
        return 3
    try:
        # the GPU index as this process sees it (CUDA_VISIBLE_DEVICES renumbers)
        probe = gpu.sync_probe(0 if os.environ.get("CUDA_VISIBLE_DEVICES") else a.gpu)
    except Exception as e:
        probe = {"error": str(e)}
    if probe.get("p90_ms", 0) > a.max_sync_ms and not a.allow_shared:
        print(f"wheeledlab_bench: GPU {a.gpu} is time-sliced with another context: kernel+sync "
              f"p50 {probe['p50_ms']} ms, p90 {probe['p90_ms']} ms > {a.max_sync_ms} ms ({json.dumps(idle)}); a measurement "
              "needs an exclusive GPU (--allow-shared runs anyway, as a smoke)", file=sys.stderr)
        return 4

    bench = Bench(a.out, a.form, a.n_envs, seed=a.seed, mode=a.timers, warmup=a.warmup,
                  settings=dict(script=script, args=rest, gpu_before=idle,
                                load_before=os.getloadavg(), sync_probe_before=probe,
                                idle_others_ok=a.idle_others_ok,
                                allow_shared=a.allow_shared), gpu=a.gpu)
    for spec in a.sample_hook:
        hooks.add_target(spec, "sample")
    hooks.install(bench)
    bench.summary_extra = lambda: dict(patched=dict(hooks.patched), unpatched=hooks.unpatched(),
                                        calls=dict(hooks.calls))

    def done():
        bench.close()
    atexit.register(done)

    sys.argv = [script, *rest]
    sys.path.insert(0, os.path.dirname(os.path.abspath(script)))
    try:
        runpy.run_path(script, run_name="__main__")
    finally:
        done()
    return 0


if __name__ == "__main__":
    sys.exit(main())
