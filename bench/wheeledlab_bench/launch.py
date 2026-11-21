"""Run a training script unchanged, with the harness hooked in.

    python -m wheeledlab_bench.launch --out DIR --form NAME [--timers sync|nosync|off]
        [--warmup 20] [--seed S] [--gpu 0] [--require-idle] -- SCRIPT [ARGS...]

The script runs as `__main__` with ARGS as its argv. `--require-idle` refuses to start while
another process holds GPU `--gpu`; the check's result goes into the run's settings either way.
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
    ap.add_argument("--require-idle", action="store_true")
    a = ap.parse_args(argv[:i])
    script, rest = argv[i + 1], argv[i + 2:]

    from . import gpu
    try:
        idle = gpu.others(a.gpu)
    except Exception as e:
        idle = {"error": str(e)}
    if a.require_idle and idle.get("procs"):
        print(f"wheeledlab_bench: GPU {a.gpu} is not idle: {json.dumps(idle)}", file=sys.stderr)
        return 3

    bench = Bench(a.out, a.form, a.n_envs, seed=a.seed, mode=a.timers, warmup=a.warmup,
                  settings=dict(script=script, args=rest, gpu_before=idle,
                                load_before=os.getloadavg()), gpu=a.gpu)
    hooks.install(bench)
    bench.summary_extra = lambda: dict(patched=dict(hooks.patched), unpatched=hooks.unpatched())

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
