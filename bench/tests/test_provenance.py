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
