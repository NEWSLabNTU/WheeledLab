"""One CSV schema for every run, every side (stock as shipped, stock ported, ours).

A row per training iteration. The stage columns are fixed, so every side writes the same
columns; a stage a side does not have is empty, never zero. `meta.json` beside the CSV names
the host, the GPU, the driver, the releases and the run's settings.
"""
from __future__ import annotations

import csv
import json
import os
import platform
import socket
import time
from importlib import metadata

# the stages, in the order they run; `env_step` holds the env's own stages, `rollout` holds
# env_step, `inference` and `storage`
STAGES = ("rollout", "inference", "env_step", "actions", "physics", "render", "scene",
          "terminations", "rewards", "resets", "commands", "events", "observations",
          "storage", "returns", "update")

COLUMNS = (["form", "n_envs", "seed", "iteration", "phase", "steps", "samples", "wall_s",
            "samples_per_s", "env_steps_per_s"]
           + [f"{s}_s" for s in STAGES] + [f"{s}_own_s" for s in STAGES]
           + ["torch_peak_mib", "smi_used_mib", "episodes", "ep_return_mean", "ep_len_mean",
              "record_s"])

PACKAGES = ("isaacsim", "isaaclab", "isaaclab_rl", "rsl-rl-lib", "torch", "warp-lang", "numpy",
            "gymnasium", "wheeledlab", "wheeledlab_tasks", "wheeledlab_rl", "wheeledlab-bench")


def releases() -> dict:
    out = {}
    for p in PACKAGES:
        try:
            out[p] = metadata.version(p)
        except metadata.PackageNotFoundError:
            pass
    return out


class RunRecord:
    def __init__(self, out_dir: str, form: str, n_envs: int, seed, settings: dict,
                 warmup: int = 20):
        os.makedirs(out_dir, exist_ok=True)
        self.out_dir, self.form, self.n_envs, self.seed, self.warmup = out_dir, form, n_envs, seed, warmup
        self.path = os.path.join(out_dir, "iterations.csv")
        self._f = open(self.path, "w", newline="")
        self._w = csv.DictWriter(self._f, fieldnames=COLUMNS)
        self._w.writeheader()
        self.settings = settings
        self.started = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        self._meta_written = False

    def write_meta(self) -> None:
        """meta.json, at the first row: the env count is known by then, and the releases are
        those the run imported."""
        from . import gpu
        try:
            g = gpu.gpus()
        except Exception as e:  # no nvidia-smi: say so
            g = [{"error": str(e)}]
        meta = dict(host=socket.gethostname(), platform=platform.platform(), gpus=g,
                    releases=releases(), form=self.form, n_envs=self.n_envs, seed=self.seed,
                    warmup=self.warmup, settings=self.settings, started=self.started,
                    pid=os.getpid(), schema=list(COLUMNS))
        with open(os.path.join(self.out_dir, "meta.json"), "w") as f:
            json.dump(meta, f, indent=1, default=str)
        self._meta_written = True

    def row(self, iteration: int, steps: int, wall_s: float, total: dict, own: dict,
            torch_peak_mib, smi_used_mib, episodes: int, ep_return_mean, ep_len_mean,
            record_s: float) -> None:
        if not self._meta_written:
            self.write_meta()
        samples = steps * self.n_envs
        r = dict(form=self.form, n_envs=self.n_envs, seed=self.seed, iteration=iteration,
                 phase="warmup" if iteration < self.warmup else "measure", steps=steps,
                 samples=samples, wall_s=f"{wall_s:.6f}",
                 samples_per_s=f"{samples / wall_s:.1f}" if wall_s > 0 else "",
                 env_steps_per_s=f"{steps / wall_s:.2f}" if wall_s > 0 else "",
                 torch_peak_mib=torch_peak_mib, smi_used_mib=smi_used_mib, episodes=episodes,
                 ep_return_mean="" if ep_return_mean is None else f"{ep_return_mean:.6g}",
                 ep_len_mean="" if ep_len_mean is None else f"{ep_len_mean:.6g}",
                 record_s=f"{record_s:.6f}")
        for s in STAGES:
            r[f"{s}_s"] = f"{total[s]:.6f}" if s in total else ""
            r[f"{s}_own_s"] = f"{own[s]:.6f}" if s in own else ""
        self._w.writerow(r)
        self._f.flush()

    def close(self, **summary) -> None:
        if not self._meta_written:
            self.write_meta()
        self._f.close()
        with open(os.path.join(self.out_dir, "summary.json"), "w") as f:
            json.dump(summary, f, indent=1, default=str)
