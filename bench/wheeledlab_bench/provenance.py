"""Where a run's code came from: the harness's commit and the task package's, each with a dirty
flag (scensim 133, 2026-11-22). meta.json carries them as `provenance`, and `launch` refuses a
run from a dirty tree unless `--allow-dirty`, which is recorded.

A package is found without importing it (`importlib.util.find_spec`, so Kit's own imports are
not disturbed). Its real path is resolved through symlinks, as colcon's `--symlink-install`
uses them, to the checkout it lives in. "Dirty" is scoped to the package's directory: a
modified or untracked (and not ignored) file under it. Other parts of the checkout do not
count. A package that is not in a git checkout (an installed wheel) has no commit, and says so.

A task's checkout also names its submodules (`submodules`): each one's path, the commit checked
out, the commit the checkout pins, and a dirty flag. A package the task imports from another
checkout (`launch --dep-package`, say aib from a submodule) is recorded under `deps` with its
own commit and dirty flag. This is the code that ran, which may differ from the pin (scensim
094 P4, 2026-11-28: a run's plant was inferred from a pin, not recorded). A dirty dependency
or submodule is refused like a dirty task. A submodule checked out off its pin is recorded
(`at_pin: false`), not refused: a bisect does that on purpose.
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
        out["dirty_files"] = _cap(files)
    out["submodules"] = submodules(repo)
    return out


def _cap(files):
    return files[:MAX_FILES] + ([f"... {len(files) - MAX_FILES} more"]
                                if len(files) > MAX_FILES else [])


def submodules(repo: str) -> list:
    """Each submodule of the checkout `repo` (recursively): path, HEAD, pin, dirty."""
    ls = _git(repo, "submodule", "status", "--recursive")
    out = []
    for line in ls.stdout.splitlines():
        if not line.strip():
            continue
        flag, rest = line[0], line[1:].split()
        path = rest[1]
        pinned = None
        parent, rel = repo, path
        # the pin is in the parent checkout's tree: the innermost one holding the path
        parts = path.split("/")
        for k in range(len(parts) - 1, 0, -1):
            cand = os.path.join(repo, *parts[:k])
            if os.path.exists(os.path.join(cand, ".git")):
                parent, rel = cand, "/".join(parts[k:])
                break
        tree = _git(parent, "ls-tree", "HEAD", "--", rel)
        if tree.returncode == 0 and tree.stdout.split():
            pinned = tree.stdout.split()[2]
        m = {"path": path, "commit": None if flag == "-" else rest[0], "pinned": pinned}
        if flag == "-":
            m.update(at_pin=None, dirty=None, error="not checked out")
        else:
            m["at_pin"] = rest[0] == pinned
            st = _git(os.path.join(repo, path), "status", "--porcelain", "--ignore-submodules=all",
                      "--untracked-files=normal")
            files = [x[3:] for x in st.stdout.splitlines() if x.strip()]
            m["dirty"] = bool(files)
            if files:
                m["dirty_files"] = _cap(files)
        out.append(m)
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
    """The names of the dirty parts of a provenance mapping ({"harness": ..., "task": ...,
    "deps": {name: ...}}), each submodule of the harness and task checkouts included."""
    out = []
    for k, v in prov.items():
        if not isinstance(v, dict):
            continue
        if k == "deps":
            out += [f"deps.{n}" for n, d in v.items() if isinstance(d, dict) and d.get("dirty")]
            continue
        if v.get("dirty"):
            out.append(k)
        out += [f"{k}.submodules.{m['path']}" for m in v.get("submodules", ()) if m.get("dirty")]
    return out
