"""Run a training script unchanged, with the harness hooked in.

    python -m wheeledlab_bench.launch --out DIR --form NAME [--timers sync|nosync|off]
        [--warmup 20] [--seed S] [--gpu 0] [--require-idle [--max-load 0.5] [--wait-idle S]]
        [--clean-rows N [--max-rows M] [--clean-sync MS]]
        [--sample-hook MOD:CLASS.METHOD ...] -- SCRIPT [ARGS...]

The script runs as `__main__` with ARGS as its argv. `--require-idle` refuses to start while
another process holds GPU `--gpu`, unless `--idle-others-ok` and the GPU's utilisation stays at
0 % over a 10 s watch (another user's process that holds memory and runs nothing, such as a
loaded model server); the check's result goes into the run's settings either way, and the CSV
records the other processes and the utilisation every few iterations, so one that wakes up
during the run shows.

An idle GPU is not an idle host: `--require-idle` also refuses (exit 6) while the host's 1-minute
load is above `--max-load` x its cores (default 0.5), after waiting up to `--wait-idle` seconds
for it to fall. Every row records the load, and the summary's `host_load` counts the measured
rows above that limit, so a run the gate let start and the host then crowded shows it.

Every row also records the CPU in use outside the run's own process tree (`outside_cpu`, in
cores, over the row) and the tree's own (`own_cpu`). With `--clean-rows N` the gate is
continuous: a measured row whose outside CPU was above `--max-load` x cores is marked
`clean=0`, and the run ends once it has N clean rows (exit 0), or is refused after `--max-rows`
measured rows (exit 7). Its summary's `clean_rows` gives the counts; a reader takes the clean
rows only.

With `--clean-sync MS` as well, every measured row also runs a short kernel+sync probe on the
run's GPU (20 calls, while the row's record is written, outside its time) and is clean only
when its outside CPU and the probe's p90 (`row_sync_p90_ms`) are both within their limits: an
idle gap on a shared GPU, not only an idle host. The summary's `clean_rows` then adds
`sync_limit_ms` and the unclean rows by cause (`over_cpu`, `over_sync`).
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


def _where(prov, key):
    """A dirty part named by provenance.dirty, with its path and files."""
    if key.startswith("deps."):
        d = prov["deps"][key[5:]]
    elif ".submodules." in key:
        top, path = key.split(".submodules.", 1)
        d = next(m for m in prov[top]["submodules"] if m["path"] == path)
        d = dict(d, path=os.path.join(prov[top].get("repo", ""), path))
    else:
        d = prov[key]
    return f"{key} ({d.get('path')}: {', '.join(d.get('dirty_files', []))})"


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
    ap.add_argument("--max-load", type=float, default=0.5, metavar="FRAC",
                    help="with --require-idle: refuse while the host's 1-minute load is above FRAC x "
                         "its cores (default 0.5)")
    ap.add_argument("--wait-idle", type=float, default=0.0, metavar="S",
                    help="with --require-idle: wait up to S seconds for the load to fall first")
    ap.add_argument("--clean-rows", type=int, default=0, metavar="N",
                    help="the continuous gate: a measured row while the CPU outside the run's own "
                         "process tree is over --max-load x cores is marked unclean; the run ends "
                         "(exit 0) once it has N clean rows, or is refused (exit 7) after "
                         "--max-rows measured rows. Give the script enough iterations")
    ap.add_argument("--max-rows", type=int, default=0, metavar="M",
                    help="with --clean-rows: the measured rows before refusing (default 4 N)")
    ap.add_argument("--clean-sync", type=float, default=None, metavar="MS",
                    help="with --clean-rows: also probe kernel+sync (20 calls) at every measured "
                         "row; a row is clean only when its outside CPU and its probe's p90 are "
                         "both within their limits (the probe is the row's row_sync_p90_ms)")
    ap.add_argument("--max-sync-ms", type=float, default=0.05,
                    help="refuse when a kernel+sync takes longer (p90: an other user busy in bursts "
                         "leaves p50 low); a free GPU takes ~0.01 ms")
    ap.add_argument("--allow-shared", action="store_true",
                    help="run anyway on a shared GPU (a smoke, not a measurement); the probe is recorded")
    ap.add_argument("--sample-hook", action="append", default=[], metavar="MOD:CLASS.METHOD",
                    help="time the `sample` stage around this method (repeatable): a side that "
                         "draws its episode starts on the host, e.g. a pool's refill and draw")
    ap.add_argument("--task-package", default=None, metavar="NAME",
                    help="the task's top-level package (wheeledlab_tasks, p4_leadbrake, "
                         "scensim_batch): its commit and dirty flag go into meta.json")
    ap.add_argument("--dep-package", action="append", default=[], metavar="NAME",
                    help="a package the task imports from another checkout (repeatable): its "
                         "commit and dirty flag go into meta.json's provenance.deps")
    ap.add_argument("--allow-dirty", action="store_true",
                    help="run although the harness or the task package has uncommitted changes "
                         "(recorded); by default such a run is refused")
    a = ap.parse_args(argv[:i])
    if a.clean_sync is not None and not a.clean_rows:
        ap.error("--clean-sync needs --clean-rows (it marks rows, the gate counts them)")
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
    from . import host
    if a.require_idle:
        load = host.gate(a.max_load, a.wait_idle)
        if not load["ok"]:
            print(f"wheeledlab_bench: the host is busy: 1-minute load {load['load1']} > "
                  f"{a.max_load} x {load['cores']} cores = {load['limit']} after waiting "
                  f"{load['waited_s']:.0f} s; a measurement needs an idle host, not only an idle GPU "
                  "(--wait-idle S waits longer)", file=sys.stderr)
            return 6
    else:
        load = {"load1": round(host.load1(), 2), "cores": host.cores(), "limit": None,
                "checked": False}
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

    from . import provenance
    prov = {"harness": provenance.harness()}
    if a.task_package:
        prov["task"] = provenance.of_package(a.task_package)
    if a.dep_package:
        prov["deps"] = {n: provenance.of_package(n) for n in a.dep_package}
    unclean = provenance.dirty(prov)
    if unclean and not a.allow_dirty:
        print("wheeledlab_bench: uncommitted changes in " + "; ".join(
              _where(prov, k) for k in unclean) + "; commit them, or --allow-dirty (recorded)",
              file=sys.stderr)
        return 5
    prov["allow_dirty"] = a.allow_dirty

    bench = Bench(a.out, a.form, a.n_envs, seed=a.seed, mode=a.timers, warmup=a.warmup,
                  settings=dict(script=script, args=rest, gpu_before=idle,
                                load_before=os.getloadavg(), host_load_gate=load,
                                sync_probe_before=probe,
                                idle_others_ok=a.idle_others_ok,
                                clean_sync_ms=a.clean_sync,
                                allow_shared=a.allow_shared), gpu=a.gpu,
                  provenance=prov, load_limit=load.get("limit"),
                  clean_rows=a.clean_rows, max_rows=a.max_rows,
                  outside_limit=a.max_load * host.cores() if a.clean_rows else None,
                  clean_sync=a.clean_sync if a.clean_rows else None)
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
