"""wheeledlab_bench.provenance and launch's dirty-tree refusal, on throwaway git checkouts."""
import json
import subprocess
import sys

import pytest

from wheeledlab_bench import launch, provenance


def git(cwd, *a):
    subprocess.run(["git", "-C", str(cwd), *a], check=True, capture_output=True,
                   env={"GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t", "PATH": "/usr/bin:/bin",
                        "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t", "HOME": str(cwd)})


@pytest.fixture
def repo(tmp_path):
    r = tmp_path / "repo"
    (r / "mytask").mkdir(parents=True)
    (r / "mytask" / "__init__.py").write_text("X = 1\n")
    (r / "other.txt").write_text("a\n")
    (r / ".gitignore").write_text("logs/\n")
    git(r, "init", "-q")
    git(r, "add", "-A")
    git(r, "commit", "-q", "-m", "init")
    return r


def test_clean_package(repo):
    p = provenance.of_path(str(repo / "mytask"), "mytask")
    assert len(p["commit"]) == 40 and p["dirty"] is False and p["repo"] == str(repo)


def test_dirty_is_scoped_to_the_package(repo):
    (repo / "other.txt").write_text("b\n")            # outside the package: not counted
    (repo / "logs").mkdir()
    (repo / "logs" / "x").write_text("ignored\n")     # ignored: not counted
    assert provenance.of_path(str(repo / "mytask"))["dirty"] is False
    (repo / "mytask" / "__init__.py").write_text("X = 2\n")
    (repo / "mytask" / "new.py").write_text("")
    p = provenance.of_path(str(repo / "mytask"))
    assert p["dirty"] is True and len(p["dirty_files"]) == 2


def test_not_a_checkout(tmp_path):
    p = provenance.of_path(str(tmp_path))
    assert p["commit"] is None and p["error"] == "not in a git checkout"


def test_of_package_finds_without_importing(repo, monkeypatch):
    monkeypatch.syspath_prepend(str(repo))
    p = provenance.of_package("mytask")
    assert p["package"] == "mytask" and p["dirty"] is False
    assert "mytask" not in sys.modules
    assert provenance.of_package("no_such_package_xyz")["error"] == "not found"


def test_launch_refuses_a_dirty_task(repo, tmp_path, monkeypatch, capsys):
    monkeypatch.syspath_prepend(str(repo))
    (repo / "mytask" / "__init__.py").write_text("X = 3\n")
    script = tmp_path / "train.py"
    script.write_text("print('ran')\n")
    monkeypatch.setattr(launch.hooks, "install", lambda bench: None)
    monkeypatch.setattr(provenance, "harness", lambda: {"commit": "h" * 40, "dirty": False})
    from wheeledlab_bench import gpu
    monkeypatch.setattr(gpu, "others", lambda g: {"procs": []})
    monkeypatch.setattr(gpu, "sync_probe", lambda g, n=200: {"p50_ms": 0.0, "p90_ms": 0.0})
    out = tmp_path / "run"
    rc = launch.main(["--out", str(out), "--form", "t", "--task-package", "mytask", "--",
                      str(script)])
    assert rc == 5 and "uncommitted changes in task (" in capsys.readouterr().err
    rc = launch.main(["--out", str(out), "--form", "t", "--task-package", "mytask",
                      "--allow-dirty", "--", str(script)])
    assert rc == 0
    meta = json.load(open(out / "meta.json")) if (out / "meta.json").exists() else None
    summary = json.load(open(out / "summary.json"))
    assert summary["closed"] is True
    if meta:   # written at the first row; a script with no env steps writes none
        assert meta["provenance"]["task"]["dirty"] is True
        assert meta["provenance"]["allow_dirty"] is True


# --- the code a task loads besides itself, and the task checkout's submodules (scensim 094 P4:
# --- a run's plant was inferred from a pin, not recorded; 133, 2026-11-28)

@pytest.fixture
def super_repo(tmp_path):
    """A task checkout with one submodule `dep` holding the package `mydep`."""
    d = tmp_path / "dep_origin"
    (d / "mydep").mkdir(parents=True)
    (d / "mydep" / "__init__.py").write_text("Y = 1\n")
    git(d, "init", "-q")
    git(d, "add", "-A")
    git(d, "commit", "-q", "-m", "dep")
    r = tmp_path / "super"
    (r / "mytask").mkdir(parents=True)
    (r / "mytask" / "__init__.py").write_text("X = 1\n")
    git(r, "init", "-q")
    git(r, "-c", "protocol.file.allow=always", "submodule", "add", "-q", str(d), "dep")
    git(r, "add", "-A")
    git(r, "commit", "-q", "-m", "super")
    return r


def test_a_dependency_package_is_its_own_checkout(super_repo, monkeypatch):
    monkeypatch.syspath_prepend(str(super_repo / "dep"))
    p = provenance.of_package("mydep")
    head = subprocess.run(["git", "-C", str(super_repo / "dep"), "rev-parse", "HEAD"],
                          capture_output=True, text=True).stdout.strip()
    assert p["commit"] == head and p["dirty"] is False
    assert p["repo"] == str(super_repo / "dep")


def test_submodules_record_head_pin_and_dirty(super_repo):
    s = provenance.submodules(str(super_repo))
    assert [m["path"] for m in s] == ["dep"]
    (m,) = s
    assert m["commit"] == m["pinned"] and m["at_pin"] is True and m["dirty"] is False
    (super_repo / "dep" / "mydep" / "__init__.py").write_text("Y = 2\n")
    git(super_repo / "dep", "commit", "-q", "-am", "moved")
    (super_repo / "dep" / "loose.txt").write_text("x\n")
    (m,) = provenance.submodules(str(super_repo))
    assert m["at_pin"] is False and m["commit"] != m["pinned"]
    assert m["dirty"] is True and m["dirty_files"] == ["loose.txt"]


def test_task_provenance_carries_its_checkout_s_submodules(super_repo, monkeypatch):
    monkeypatch.syspath_prepend(str(super_repo))
    p = provenance.of_package("mytask")
    assert [m["path"] for m in p["submodules"]] == ["dep"]


def test_a_repo_without_submodules_has_an_empty_list(repo):
    assert provenance.submodules(str(repo)) == []


def test_a_dirty_dependency_or_an_off_pin_submodule_counts_as_unclean():
    prov = {"harness": {"dirty": False}, "task": {"dirty": False, "submodules": [
        {"path": "dep", "at_pin": True, "dirty": False}]},
        "deps": {"mydep": {"dirty": True}}}
    assert provenance.dirty(prov) == ["deps.mydep"]
    prov["deps"]["mydep"]["dirty"] = False
    prov["task"]["submodules"][0]["dirty"] = True
    assert provenance.dirty(prov) == ["task.submodules.dep"]
    prov["task"]["submodules"][0]["dirty"] = False
    prov["task"]["submodules"][0]["at_pin"] = False      # recorded, not refused: a bisect does it
    assert provenance.dirty(prov) == []


def test_launch_records_dep_packages(super_repo, tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(super_repo))
    monkeypatch.syspath_prepend(str(super_repo / "dep"))
    script = tmp_path / "train.py"
    script.write_text("print('ran')\n")
    monkeypatch.setattr(launch.hooks, "install", lambda bench: None)
    monkeypatch.setattr(provenance, "harness", lambda: {"commit": "h" * 40, "dirty": False})
    from wheeledlab_bench import gpu
    monkeypatch.setattr(gpu, "others", lambda g: {"procs": []})
    monkeypatch.setattr(gpu, "sync_probe", lambda g, n=200: {"p50_ms": 0.0, "p90_ms": 0.0})
    seen = {}
    real = launch.Bench
    def spy(*a, **k):
        b = real(*a, **k)
        seen["prov"] = k.get("provenance")
        return b
    monkeypatch.setattr(launch, "Bench", spy)
    (super_repo / "dep" / "mydep" / "__init__.py").write_text("Y = 3\n")
    out = tmp_path / "run"
    rc = launch.main(["--out", str(out), "--form", "t", "--task-package", "mytask",
                      "--dep-package", "mydep", "--", str(script)])
    assert rc == 5      # the dependency is dirty
    rc = launch.main(["--out", str(out), "--form", "t", "--task-package", "mytask",
                      "--dep-package", "mydep", "--allow-dirty", "--", str(script)])
    assert rc == 0
    assert seen["prov"]["deps"]["mydep"]["dirty"] is True
    assert [m["path"] for m in seen["prov"]["task"]["submodules"]] == ["dep"]
