"""The lead-brake env's physics, scripted (no policy): 4 envs, each one manoeuvre, printed every
second of sim time. A sanity check of the car and the task before training.

  env 0  coast (0, 0): the speed decays slowly, the car stays in its lane at its height
  env 1  full brake (-1, 0) from the start: the stop's deceleration (dry, mu 1: about 9 m/s^2)
  env 2  half throttle (0.5, 0): it accelerates, the lead brakes ahead: a collision ends it
  env 3  steer left (0, 0.3): it leaves the road to the left: off_road ends it

    python tools/check_env.py      # in the Isaac Lab overlay's python, OMNI_KIT_ACCEPT_EULA=YES
"""
import argparse

from isaaclab.app import AppLauncher

ap = argparse.ArgumentParser()
ap.add_argument("--seconds", type=float, default=12.0)
AppLauncher.add_app_launcher_args(ap)
args = ap.parse_args()
app = AppLauncher(args).app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import p4_leadbrake  # noqa: E402,F401
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry  # noqa: E402
from p4_leadbrake import car, mdp  # noqa: E402

cfg = load_cfg_from_registry("P4-LeadBrake-v0", "env_cfg_entry_point")
cfg.scene.num_envs = 4
env = gym.make("P4-LeadBrake-v0", cfg=cfg).unwrapped
env.reset()
act = torch.tensor([[0.0, 0.0], [-1.0, 0.0], [0.5, 0.0], [0.0, 0.3]], device=env.device)
st = mdp.state(env)
robot = env.scene["robot"]
names = ["coast", "brake", "throttle", "steer"]
done_at = {}
steps = int(args.seconds / env.step_dt)
for k in range(steps):
    s, y, yaw = mdp.ego_s_y_yaw(env)
    v = mdp.T(robot.data.root_lin_vel_b)[:, 0]
    z = mdp.T(robot.data.root_pos_w)[:, 2]
    g = mdp.gap(env)
    if k % int(1.0 / env.step_dt) == 0:
        sus, _ = robot.find_joints(car.SUSPENSION_JOINTS)
        q = mdp.T(robot.data.joint_pos)[:, sus]
        wv = mdp.T(robot.data.joint_vel)[:, robot.find_joints(car.SPIN_JOINTS)[0]]
        print(f"   suspension q (FL FR RL RR) env0 {[round(float(x), 3) for x in q[0]]}  "
              f"env1 {[round(float(x), 3) for x in q[1]]}; wheel spin env1 "
              f"{[round(float(x), 1) for x in wv[1]]}, env2 {[round(float(x), 1) for x in wv[2]]}")
        print(f"t={k * env.step_dt:5.1f}  " + "  ".join(
            f"{names[i]}: s={s[i]:6.1f} v={v[i]:5.2f} y={y[i]:+5.2f} yaw={yaw[i]:+5.2f} "
            f"z={z[i]:+5.3f} gap={g[i]:6.1f} lead_v={st.lead_v[i]:4.1f}" for i in range(4)),
            flush=True)
    _, rew, term, trunc, _ = env.step(act)
    for i in range(4):
        if (term[i] or trunc[i]) and i not in done_at:
            tm = env.termination_manager
            why = [n for n in tm.active_terms if bool(tm.get_term(n)[i])]
            done_at[i] = (round((k + 1) * env.step_dt, 2), why)
print("ended:", {names[i]: v for i, v in done_at.items()})
print("lead params:", {n: [round(float(x), 2) for x in getattr(st, n)] for n in mdp.RANGES})
print("front", car.FRONT, "rear", car.REAR)
env.close()
app.close()
