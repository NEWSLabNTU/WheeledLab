"""wheeledlab_bench's hooks, without Isaac: a stand-in for Isaac Lab's DirectRLEnv (as a module
named like Isaac Lab's), a task subclass of it as scensim_batch's ScensimEnv is, and the stages
they time (scensim roadmap 094 P2)."""
import sys
import types

import pytest

from wheeledlab_bench import hooks
from wheeledlab_bench.timers import StageTimer


class FakeBench:
    def __init__(self):
        self.timer = StageTimer("nosync")
        self.steps = 0

    def begin(self):
        pass

    def on_env_step(self, rew, done):
        self.steps += 1


def direct_module():
    """isaaclab.envs.direct_rl_env as far as the hooks see it: DirectRLEnv's step calls the
    subclass's stage methods, as Isaac Lab's does."""
    mod = types.ModuleType("isaaclab.envs.fake_direct_rl_env")

    class DirectRLEnv:
        def __init__(self, cfg=None):
            self.cfg = cfg

        def step(self, action):
            self._pre_physics_step(action)
            for _ in range(2):  # decimation
                self._apply_action()
            dones = self._get_dones()
            rew = self._get_rewards()
            self._reset_idx(None)
            return self._get_observations(), rew, dones, dones, {}

        def _reset_idx(self, env_ids):
            pass

    DirectRLEnv.__module__ = mod.__name__
    mod.DirectRLEnv = DirectRLEnv
    return mod


@pytest.fixture
def bench(monkeypatch):
    b = FakeBench()
    monkeypatch.setattr(hooks, "_bench", b)
    monkeypatch.setattr(hooks, "patched", {})
    monkeypatch.setattr(hooks, "calls", {})
    return b


def task_of(base):
    class ScensimEnv(base):
        def _pre_physics_step(self, action):
            self.action = action

        def _apply_action(self):
            pass

        def _get_observations(self):
            return {"policy": 0}

        def _get_rewards(self):
            return 1.0

        def _get_dones(self):
            return False

        def _reset_idx(self, env_ids):
            super()._reset_idx(env_ids)
    return ScensimEnv


def test_a_direct_envs_own_methods_are_timed_as_the_manager_based_stages(bench):
    mod = direct_module()
    hooks._patch_module(mod)
    env = task_of(mod.DirectRLEnv)()
    env.step(0.5)
    total, own, calls = bench.timer.take()
    assert calls["env_step"] == 1
    assert calls["actions"] == 3            # _pre_physics_step once, _apply_action per substep
    assert calls["observations"] == calls["rewards"] == calls["terminations"] == 1
    assert calls["resets"] == 1             # the task's _reset_idx; its super() call not again
    assert bench.steps == 1
    assert hooks.patched["ScensimEnv._apply_action"].endswith("ScensimEnv")
    assert hooks.patched["ScensimEnv._reset_idx"].endswith("ScensimEnv")
    # per method, for the run's summary
    assert hooks.calls["ScensimEnv._apply_action"] == 2
    assert hooks.calls["ScensimEnv._pre_physics_step"] == 1
    assert hooks.calls["DirectRLEnv.step"] == 1


def test_a_method_the_task_leaves_to_directrlenv_is_timed_there(bench):
    mod = direct_module()
    hooks._patch_module(mod)
    cls = task_of(mod.DirectRLEnv)
    del cls._reset_idx
    cls()
    assert hooks.patched["ScensimEnv._reset_idx"].endswith(".DirectRLEnv")


def test_a_step_the_task_overrides_is_counted_once(bench):
    mod = direct_module()
    hooks._patch_module(mod)

    class Override(task_of(mod.DirectRLEnv)):
        def step(self, action):
            return super().step(action)

    env = Override()
    env.step(0.1)
    env.step(0.1)
    _, _, calls = bench.timer.take()
    assert calls["env_step"] == 2 and bench.steps == 2
    assert hooks.calls["Override.step"] == 2 and "DirectRLEnv.step" not in hooks.calls


def test_two_envs_of_one_class_are_patched_once(bench):
    mod = direct_module()
    hooks._patch_module(mod)
    cls = task_of(mod.DirectRLEnv)
    cls(), cls()
    cls().step(0.0)
    _, _, calls = bench.timer.take()
    assert calls["actions"] == 3


# ---- the sample stage (a side that draws its episode starts on the host) ---------------------

def pool_module():
    mod = types.ModuleType("scensim_batch.fake_pool")

    class StartPool:
        def refill(self):
            return "filled"

        def draw(self, n):
            return list(range(n))

    StartPool.__module__ = mod.__name__
    mod.StartPool = StartPool
    return mod


@pytest.fixture
def sample_targets(monkeypatch):
    monkeypatch.setattr(hooks, "TARGETS", {k: dict(v) for k, v in hooks.TARGETS.items()})
    monkeypatch.setattr(hooks, "MODULES", {})
    monkeypatch.setattr(hooks, "PREFIXES", hooks.PREFIXES)


def test_sample_hook_times_draw_inside_a_reset(bench, sample_targets):
    hooks.add_target("scensim_batch.fake_pool:StartPool.draw", "sample")
    hooks.add_target("scensim_batch.fake_pool:StartPool.refill", "sample")
    assert "scensim_batch" in hooks.PREFIXES
    mod = pool_module()
    hooks._patch_module(mod)
    pool = mod.StartPool()
    bench.timer.start("resets")
    assert pool.draw(3) == [0, 1, 2]
    bench.timer.stop("resets")
    total, own, calls = bench.timer.take()
    assert calls["sample"] == 1 and total["sample"] <= total["resets"]
    assert own["resets"] == pytest.approx(total["resets"] - total["sample"])
    assert hooks.calls == {"StartPool.draw": 1}
    assert not {"StartPool.draw", "StartPool.refill"} & set(hooks.unpatched())


def test_sample_on_a_background_thread_is_counted_off_the_stack(bench, sample_targets):
    import threading
    hooks.add_target("scensim_batch.fake_pool:StartPool.refill", "sample")
    mod = pool_module()
    hooks._patch_module(mod)
    pool = mod.StartPool()
    bench.timer.start("env_step")
    th = threading.Thread(target=pool.refill)
    th.start()
    th.join()
    bench.timer.stop("env_step")         # the stack is untouched by the thread
    total, own, calls = bench.timer.take()
    assert calls["sample"] == 1 and total["sample"] == own["sample"] > 0
    assert own["env_step"] == pytest.approx(total["env_step"])   # not charged as a child


def test_a_named_target_in_another_module_is_left_alone(bench, sample_targets):
    hooks.add_target("scensim_batch.pool:StartPool.refill", "sample")
    mod = pool_module()                    # scensim_batch.fake_pool, not .pool
    hooks._patch_module(mod)
    assert "StartPool.refill" not in hooks.patched
    assert "StartPool.refill" in hooks.unpatched()


def test_stock_rows_carry_sample_zero(tmp_path):
    import csv
    from wheeledlab_bench import Bench
    b = Bench(str(tmp_path), "stock", 4, mode="nosync", smi_every=0)
    b.timer.start("env_step")
    b.timer.stop("env_step")
    b.end_iteration()
    b.close()
    row = next(csv.DictReader(open(tmp_path / "iterations.csv")))
    assert row["sample_s"] == "0.000000" and row["sample_own_s"] == "0.000000"
