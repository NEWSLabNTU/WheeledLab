"""The host-load gate of --require-idle: an idle GPU on a busy host is not a measurement."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from wheeledlab_bench import host, launch  # noqa: E402


def test_an_idle_host_passes_at_once(monkeypatch):
    monkeypatch.setattr(host, "cores", lambda: 24)
    g = host.gate(0.5, _load=lambda: 8.0, _sleep=lambda s: None)
    assert g["ok"] and g["limit"] == 12.0 and g["waited_s"] == 0 and g["samples"] == [8.0]


def test_a_busy_host_is_waited_for_then_passes(monkeypatch):
    monkeypatch.setattr(host, "cores", lambda: 24)
    loads = iter([30.0, 20.0, 11.0])
    g = host.gate(0.5, wait_s=60, poll_s=10, _load=lambda: next(loads), _sleep=lambda s: None)
    assert g["ok"] and g["waited_s"] == 20 and g["samples"] == [30.0, 20.0, 11.0]


def test_a_host_that_stays_busy_fails_after_the_wait(monkeypatch):
    monkeypatch.setattr(host, "cores", lambda: 24)
    g = host.gate(0.5, wait_s=30, poll_s=10, _load=lambda: 25.0, _sleep=lambda s: None)
    assert not g["ok"] and g["waited_s"] == 30 and len(g["samples"]) == 4


def test_the_track_counts_measured_rows_over_the_limit():
    t = host.LoadTrack(12.0, warmup=2)
    for i, v in enumerate([40.0, 40.0, 10.0, 13.0, 11.0, 20.0]):
        t.add(i, v)
    s = t.summary()
    assert s["rows"] == 4 and s["rows_over_limit"] == 2 and s["max"] == 20.0


def test_launch_refuses_a_busy_host_with_exit_6(monkeypatch, tmp_path, capsys):
    from wheeledlab_bench import gpu
    monkeypatch.setattr(gpu, "others", lambda i: {"procs": 0})
    monkeypatch.setattr(gpu, "sync_probe", lambda i, **k: {"p50_ms": 0.0, "p90_ms": 0.0})
    monkeypatch.setattr(host, "cores", lambda: 24)
    monkeypatch.setattr(host, "load1", lambda: 30.0)
    rc = launch.main(["--out", str(tmp_path / "r"), "--form", "x", "--require-idle", "--",
                      "nonexistent.py"])
    assert rc == 6
    assert "the host is busy: 1-minute load 30.0 > 0.5 x 24 cores = 12.0" in capsys.readouterr().err
    assert not (tmp_path / "r").exists()
