"""Where a run's code came from: the harness's commit and the task package's, each with a dirty
flag (scensim 133, 2026-11-22). meta.json carries them as `provenance`, and `launch` refuses a
run from a dirty tree unless `--allow-dirty`, which is recorded.

A package is found without importing it (`importlib.util.find_spec`, so Kit's own imports are
not disturbed). Its real path is resolved through symlinks, as colcon's `--symlink-install`
uses them, to the checkout it lives in. "Dirty" is scoped to the package's directory: a
modified or untracked (and not ignored) file under it. Other parts of the checkout do not
count. A package that is not in a git checkout (an installed wheel) has no commit, and says so.
"""
from __future__ import annotations

import importlib.util
import os
import subprocess

MAX_FILES = 20


def _git(cwd, *args):
    return subprocess.run(["git", "-C", cwd, *args], capture_output=True, text=True, timeout=30)


def of_path(path: str, package: str | None = None) -> dict:
    """The provenance of the directory `path`."""
    d = os.path.realpath(path)
    out = {"package": package, "path": d, "commit": None, "dirty": None}
    top = _git(d, "rev-parse", "--show-toplevel")
    if top.returncode != 0:
        out["error"] = "not in a git checkout"
        return out
    repo = top.stdout.strip()
    out["repo"] = repo
    head = _git(d, "rev-parse", "HEAD")
    out["commit"] = head.stdout.strip() if head.returncode == 0 else None
    br = _git(d, "rev-parse", "--abbrev-ref", "HEAD")
    out["branch"] = br.stdout.strip() if br.returncode == 0 else None
    st = _git(repo, "status", "--porcelain", "--untracked-files=normal", "--", d)
    files = [line[3:] for line in st.stdout.splitlines() if line.strip()]
    out["dirty"] = bool(files)
    if files:
        out["dirty_files"] = files[:MAX_FILES] + ([f"... {len(files) - MAX_FILES} more"]
                                                 if len(files) > MAX_FILES else [])
    return out


def of_package(name: str) -> dict:
    """The provenance of the top-level package `name`, found without importing it."""
    try:
        spec = importlib.util.find_spec(name)
    except (ImportError, ValueError) as e:
        return {"package": name, "commit": None, "dirty": None, "error": str(e)}
    if spec is None:
        return {"package": name, "commit": None, "dirty": None, "error": "not found"}
    if spec.submodule_search_locations:
        path = list(spec.submodule_search_locations)[0]
    elif spec.origin:
        path = os.path.dirname(spec.origin)
    else:
        return {"package": name, "commit": None, "dirty": None, "error": "no location"}
    return of_path(path, name)


def harness() -> dict:
    return of_path(os.path.dirname(os.path.abspath(__file__)), "wheeledlab_bench")


def dirty(prov: dict) -> list:
    """The names of the dirty parts of a provenance mapping ({"harness": ..., "task": ...})."""
    return [k for k, v in prov.items() if isinstance(v, dict) and v.get("dirty")]
