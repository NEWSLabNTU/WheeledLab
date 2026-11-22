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


def test_outside_cpu_leaves_the_run_out():
    """A busy loop in this process is own CPU, not outside: own_cpu x the interval is the CPU
    time the loop took (compared with the process's own count, so a loaded host does not
    matter)."""
    import time
    o = host.OutsideCpu()
    w0, c0 = time.perf_counter(), time.process_time()
    while time.process_time() - c0 < 0.5:
        pass
    s = o.sample()
    wall, cpu = time.perf_counter() - w0, time.process_time() - c0
    assert abs(s["own_cpu"] * wall - cpu) < 0.15
    assert s["outside_cpu"] >= 0.0


def test_the_tree_counts_children():
    """A child's CPU time is the run's own, also after it exited and was waited for."""
    import resource
    import subprocess
    import time
    o = host.OutsideCpu()
    w0 = time.perf_counter()
    r0 = resource.getrusage(resource.RUSAGE_CHILDREN)
    subprocess.run([sys.executable, "-c",
                    "import time\nc=time.process_time()\nwhile time.process_time()-c<0.5: pass"])
    r1 = resource.getrusage(resource.RUSAGE_CHILDREN)
    s = o.sample()
    wall = time.perf_counter() - w0
    child = (r1.ru_utime + r1.ru_stime) - (r0.ru_utime + r0.ru_stime)
    assert child >= 0.45
    assert s["own_cpu"] * wall >= child - 0.15


def test_clean_rows_end_complete_or_refused():
    g = host.CleanRows(need=3, ceiling=6, limit=12.0)
    for v in (5, 20, 6, 30):
        assert g.verdict() is None
        g.add(v)
    assert (g.clean, g.measured) == (2, 4) and g.verdict() is None
    g.add(7)
    assert g.verdict() == "complete"
    r = host.CleanRows(need=3, ceiling=4, limit=12.0)
    for v in (5, 20, 20, 20):
        r.add(v)
    assert r.verdict() == "refused" and r.summary()["clean"] == 1


def test_a_bench_ends_its_run_at_its_clean_rows(tmp_path, monkeypatch):
    """The gate ends the run itself: its record closed, the exit code complete (0)."""
    import json
    from wheeledlab_bench import bench as bmod
    exits = []

    class Exit(Exception):
        pass

    def fake_exit(code):
        exits.append(code)
        raise Exit
    monkeypatch.setattr(bmod.os, "_exit", fake_exit)
    b = bmod.Bench(str(tmp_path / "r"), "x", 4, warmup=1, mode="off", smi_every=0,
                   clean_rows=2, max_rows=5, outside_limit=1e9)
    b._torch = None
    try:
        for _ in range(10):
            b.begin()
            b.steps = 1
            b.end_iteration()
    except Exit:
        pass
    assert exits == [0]
    s = json.loads((tmp_path / "r" / "summary.json").read_text())
    assert s["closed"] is True and s["clean_rows"]["verdict"] == "complete"
    assert s["clean_rows"]["clean"] == 2 and s["iterations"] == 3
    rows = (tmp_path / "r" / "iterations.csv").read_text().splitlines()
    assert rows[0].endswith("outside_cpu,own_cpu,clean,row_sync_p90_ms")
    assert rows[1].endswith(",,") and rows[2].endswith(",1,")  # warm-up: no flag; no probe


SMI_XML = """<?xml version="1.0" ?><nvidia_smi_log>
<gpu id="00000000:01:00.0"><uuid>GPU-a</uuid><processes>
 <process_info><pid>1698</pid><type>G</type><process_name>/usr/lib/xorg/Xorg</process_name><used_memory>4 MiB</used_memory></process_info>
 <process_info><pid>111</pid><type>C+G</type><process_name>/venv/bin/python</process_name><used_memory>2939 MiB</used_memory></process_info>
 <process_info><pid>222</pid><type>G</type><process_name>kit</process_name><used_memory>310 MiB</used_memory></process_info>
</processes></gpu>
<gpu id="00000000:02:00.0"><uuid>GPU-b</uuid><processes>
 <process_info><pid>333</pid><type>C</type><process_name>python</process_name><used_memory>10258 MiB</used_memory></process_info>
</processes></gpu></nvidia_smi_log>"""


def test_apps_include_graphics_only_contexts_and_leave_out_xorg():
    from wheeledlab_bench import gpu
    ps = gpu.parse_processes(SMI_XML)
    assert [(p["uuid"], p["pid"], p["type"]) for p in ps] == [
        ("GPU-a", 111, "C+G"), ("GPU-a", 222, "G"), ("GPU-b", 333, "C")]
    assert ps[1]["used_mib"] == 310


def test_apps_reads_this_hosts_table():
    from wheeledlab_bench import gpu
    import shutil
    if shutil.which("nvidia-smi") is None:
        return
    assert all({"uuid", "pid", "type", "used_mib"} <= set(p) for p in gpu.apps())


def test_clean_sync_needs_both_limits():
    g = host.CleanRows(need=10, ceiling=20, limit=12.0, sync_limit=0.05)
    assert g.add(5, 0.02) is True          # both within
    assert g.add(20, 0.02) is False        # the host busy
    assert g.add(5, 0.4) is False          # the GPU time-sliced
    assert g.add(5, None) is False         # no probe: not clean
    s = g.summary()
    assert (s["clean"], s["measured"], s["over_cpu"], s["over_sync"]) == (1, 4, 1, 2)
    assert s["sync_limit_ms"] == 0.05
    # without a sync limit the summary and the verdicts are as before
    assert "over_sync" not in host.CleanRows(3, 6, 12.0).summary()
    assert host.CleanRows(3, 6, 12.0).add(5, 9.9) is True


def test_a_bench_with_clean_sync_probes_every_measured_row(tmp_path, monkeypatch):
    import json
    from wheeledlab_bench import bench as bmod
    probes = iter([0.01, 0.9, 0.02, 0.03])

    class Exit(Exception):
        pass

    def fake_exit(code):
        raise Exit
    monkeypatch.setattr(bmod.os, "_exit", fake_exit)
    b = bmod.Bench(str(tmp_path / "r"), "x", 4, warmup=1, mode="off", smi_every=0,
                   clean_rows=3, max_rows=10, outside_limit=1e9, clean_sync=0.05)
    b._torch = None
    calls = []
    monkeypatch.setattr(b, "row_sync_probe", lambda: calls.append(1) or next(probes))
    try:
        for _ in range(10):
            b.begin()
            b.steps = 1
            b.end_iteration()
    except Exit:
        pass
    s = json.loads((tmp_path / "r" / "summary.json").read_text())
    assert len(calls) == 4                 # measured rows only: the warm-up row is not probed
    assert s["clean_rows"]["verdict"] == "complete" and s["clean_rows"]["over_sync"] == 1
    rows = (tmp_path / "r" / "iterations.csv").read_text().splitlines()
    assert rows[0].endswith("clean,row_sync_p90_ms")
    assert [r.split(",")[-2:] for r in rows[2:]] == [["1", "0.01"], ["0", "0.9"], ["1", "0.02"],
                                                     ["1", "0.03"]]


def test_clean_sync_needs_clean_rows(capsys):
    import pytest
    with pytest.raises(SystemExit) as e:
        launch.main(["--out", "x", "--form", "y", "--clean-sync", "0.05", "--", "s.py"])
    assert e.value.code == 2 and "--clean-sync needs --clean-rows" in capsys.readouterr().err
