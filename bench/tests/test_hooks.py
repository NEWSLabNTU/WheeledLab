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
