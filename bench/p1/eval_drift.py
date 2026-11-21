"""094 P1's pass bar for the MuSHR drift task: a trained policy, deterministic, N envs for T
seconds of sim time, on the run's own training env configuration (its noise, perturbations and
domain randomisation) with the out-of-bounds termination off and the episode longer than T.
Upstream's play configuration (MushrDriftPlayEnvCfg) is not used: its reset term passes
`dist_noise`, which the reset function does not take, so it does not build.

The bar is scensim's, not the paper's. The paper (arXiv 2502.07380v2) reports no result in
simulation; its real-car numbers (controlled slip up to 58 deg, about 1.6 m/s, laps without
spinning out) set the levels below. Per env:

- laps: the unwrapped angle of the car's position about the oval's centre, over 2 pi
  (counter-clockwise, the task's direction);
- off track: the task's own termination test (`cart_off_track`), at any step;
- spin-out: the slip angle |atan2(v_y, v_x)| in the body frame above 100 deg, or the car moving
  clockwise along the track, for 0.5 s on end, at speed above 0.5 m/s;
- peak controlled slip: the largest slip angle while on track, not spun out, moving
  counter-clockwise with v_x above 0.5 m/s;
- mean speed: the planar speed's mean over the run.

A policy passes when at least 90 % of envs complete 3 laps with no off-track and no spin-out,
the median peak controlled slip is at least 40 deg, and the mean speed is at least 1.2 m/s.

    python bench/p1/eval_drift.py --run-path <run dir> [--checkpoint N] [--envs 32] [--seconds 30] --out <json>
"""
from wheeledlab_rl.startup import startup
import argparse

parser = argparse.ArgumentParser(description="094 P1: the drift policy's pass bar.")
parser.add_argument("--run-path", required=True)
parser.add_argument("--checkpoint", type=int, default=None, help="default: the last")
parser.add_argument("--envs", type=int, default=32)
parser.add_argument("--seconds", type=float, default=30.0)
parser.add_argument("--out", required=True)
parser.add_argument("--run-config", default="RSS_DRIFT_CONFIG",
                    help="the run config, rebuilt by name: a pickled one is tied to its Isaac Lab release")
simulation_app, args_cli = startup(parser=parser)

import glob
import json
import math
import os
import re
from importlib import metadata

import gymnasium as gym
import torch
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
from rsl_rl.runners import OnPolicyRunner

from wheeledlab_rl.utils import ClipAction
from wheeledlab_tasks.drifting.mushr_drift_env_cfg import (
    CORNER_IN_RADIUS, CORNER_OUT_RADIUS, STRAIGHT, cart_off_track)

BAR = dict(laps=3.0, share_clean=0.90, slip_deg=40.0, speed=1.2)
SPIN_SLIP = math.radians(100.0)
SPIN_HOLD_S = 0.5
MIN_SPEED = 0.5


def checkpoint(run_path, n):
    models = glob.glob(os.path.join(run_path, "models", "model_*.pt"))
    if not models:
        raise SystemExit(f"no checkpoints in {run_path}/models")
    by_it = {int(re.search(r"model_(\d+)\.pt$", m).group(1)): m for m in models}
    it = max(by_it) if n is None else n
    return it, by_it[it]


def main():
    from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry
    import wheeledlab_rl.configs.runs as runs
    run_cfg = getattr(runs, args_cli.run_config)()
    task = run_cfg.env_setup.task_name
    env_cfg = load_cfg_from_registry(task, "env_cfg_entry_point")
    agent_cfg = load_cfg_from_registry(task, run_cfg.agent_setup.entry_point)
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, metadata.version("rsl-rl-lib"))
    it, path = checkpoint(args_cli.run_path, args_cli.checkpoint)

    env_cfg.scene.num_envs = args_cli.envs
    env_cfg.sim.device = run_cfg.train.device
    env_cfg.terminations.out_of_bounds = None           # counted here instead
    env_cfg.rewards.term_pens = None                    # it reads that termination
    env_cfg.curriculum = None                           # fixed reward weights; no effect on the policy
    env_cfg.episode_length_s = args_cli.seconds + 5.0   # no time-out inside the window
    env = gym.make(task, cfg=env_cfg)
    env.action_space.low, env.action_space.high = -1.0, 1.0
    env = RslRlVecEnvWrapper(ClipAction(env))
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=env.unwrapped.device)
    runner.load(path)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    u = env.unwrapped
    robot = u.scene["robot"]
    dt = u.step_dt
    steps = int(round(args_cli.seconds / dt))
    n, dev = u.num_envs, u.device
    hold = max(1, int(round(SPIN_HOLD_S / dt)))

    def planar():
        p = robot.data.root_pos_w[:, :2] - u.scene.env_origins[:, :2]
        return p

    obs = env.get_observations()
    p0 = planar()
    ang_prev = torch.atan2(p0[:, 1], p0[:, 0])
    ang_total = torch.zeros(n, device=dev)
    off = torch.zeros(n, dtype=torch.bool, device=dev)
    spun = torch.zeros(n, dtype=torch.bool, device=dev)
    bad_run = torch.zeros(n, dtype=torch.long, device=dev)
    peak_slip = torch.zeros(n, device=dev)
    speed_sum = torch.zeros(n, device=dev)
    with torch.inference_mode():
        for _ in range(steps):
            obs, _, _, _ = env.step(policy(obs))
            p = planar()
            ang = torch.atan2(p[:, 1], p[:, 0])
            d = torch.remainder(ang - ang_prev + math.pi, 2 * math.pi) - math.pi
            ang_total += d
            ang_prev = ang
            v = robot.data.root_lin_vel_b[:, :2]
            speed = torch.linalg.norm(v, dim=-1)
            speed_sum += speed
            slip = torch.abs(torch.atan2(v[:, 1], v[:, 0]))
            backwards = d < 0
            bad = (speed > MIN_SPEED) & ((slip > SPIN_SLIP) | backwards)
            bad_run = torch.where(bad, bad_run + 1, torch.zeros_like(bad_run))
            spun |= bad_run >= hold
            off |= cart_off_track(u, STRAIGHT, CORNER_IN_RADIUS, CORNER_OUT_RADIUS)
            controlled = (~off) & (~spun) & (~backwards) & (v[:, 0] > MIN_SPEED)
            peak_slip = torch.maximum(peak_slip, torch.where(controlled, slip, torch.zeros_like(slip)))

    laps = (ang_total / (2 * math.pi)).cpu()
    off, spun = off.cpu(), spun.cpu()
    clean = (laps >= BAR["laps"]) & ~off & ~spun
    slip_deg = torch.rad2deg(peak_slip).cpu()
    mean_speed = (speed_sum / steps).cpu()
    res = dict(
        run_path=os.path.abspath(args_cli.run_path), checkpoint=it, envs=n, seconds=args_cli.seconds,
        step_dt=dt, bar=BAR,
        share_clean=float(clean.float().mean()), laps_median=float(laps.median()),
        laps_min=float(laps.min()), off_track=int(off.sum()), spun=int(spun.sum()),
        slip_deg_median=float(slip_deg.median()), slip_deg_max=float(slip_deg.max()),
        speed_mean=float(mean_speed.mean()),
        per_env=dict(laps=laps.tolist(), off=off.tolist(), spun=spun.tolist(),
                     slip_deg=slip_deg.tolist(), speed=mean_speed.tolist()),
    )
    res["passed"] = bool(res["share_clean"] >= BAR["share_clean"] and res["slip_deg_median"] >= BAR["slip_deg"]
                         and res["speed_mean"] >= BAR["speed"])
    os.makedirs(os.path.dirname(os.path.abspath(args_cli.out)), exist_ok=True)
    with open(args_cli.out, "w") as f:
        json.dump(res, f, indent=1)
    print("EVAL", json.dumps({k: v for k, v in res.items() if k != "per_env"}))
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
