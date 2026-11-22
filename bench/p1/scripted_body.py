"""Stock MuSHR (Isaac-MushrDriftRL-v0's body and ground) on scripted commands, so a rebuilt body
is checked against it before training (scensim roadmap 094 P2), on scensim 139's protocol:

  MUSHR_SUS_2WD_CFG on stock's plane (static 1.1, dynamic 1.0, multiply), solver iterations as
  stock's (4/0), physics 1/200 s commanded every step (decimation 1); every randomisation event
  off (the USD's wheel friction, no added mass, stock's rear damping, no pushes), unless
  --wheel-friction sets the wheels' (training draws 0.3 to 0.5); terminations and the
  curriculum off; 4 envs from rest at their origins, heading +x, all on the same commands.

  Commands through stock's action (MushrRWDActionCfg, scale (3.0, 0.488)), which gives the rear
  throttle joints a velocity target of v / 0.05 rad/s and both front steer joints tan(d):

    settle 0 m/s 0 rad 0.5 s | straight 2.0, 0, 2 s | turn 2.0, 0.3, 2 s |
    hard_turn 3.0, 0.488, 2 s | brake 0, 0, 3 s (target 0 is MuSHR's brake)

    python bench/p1/scripted_body.py [--wheel-friction 0.4] [--out trace.json]

Prints each phase's last v (body x) and yaw rate, and the brake's start speed, the time to
|v| < 0.02 m/s and the distance from its start to its end; the trace (env 0, every 10 steps:
t, x, y, v, yaw rate) goes to --out.
"""
import argparse
import json
import math

from isaaclab.app import AppLauncher

ap = argparse.ArgumentParser()
ap.add_argument("--wheel-friction", type=float, default=None)
ap.add_argument("--out", default=None)
AppLauncher.add_app_launcher_args(ap)
args = ap.parse_args()
app = AppLauncher(args).app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import wheeledlab_tasks  # noqa: E402,F401
from isaaclab.managers import EventTermCfg, SceneEntityCfg  # noqa: E402
from isaaclab.envs import mdp  # noqa: E402
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry  # noqa: E402

TASK = "Isaac-MushrDriftRL-v0"
cfg = load_cfg_from_registry(TASK, "env_cfg_entry_point")
cfg.num_envs = 4
cfg.scene.num_envs = 4
cfg.scene.env_spacing = 20.0
cfg.episode_length_s = 100.0
cfg.decimation = 1
cfg.sim.render_interval = 1
cfg.curriculum = None
# a reward term reads out_of_bounds, so it stays, never firing (the car leaves the track's area)
def never(env, straight=None, corner_in_radius=None, corner_out_radius=None):
    return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)


cfg.terminations.out_of_bounds.func = never
ev = cfg.events
for name in ("change_wheel_friction", "randomize_gains", "push_robots_hf", "push_robots_lf",
             "add_base_mass"):
    if hasattr(ev, name):
        setattr(ev, name, None)
if args.wheel_friction is not None:
    ev.change_wheel_friction = EventTermCfg(
        func=mdp.randomize_rigid_body_material, mode="startup",
        params={"static_friction_range": (args.wheel_friction, args.wheel_friction),
                "dynamic_friction_range": (args.wheel_friction, args.wheel_friction),
                "restitution_range": (0.0, 0.0), "num_buckets": 1,
                "asset_cfg": SceneEntityCfg("robot", body_names=".*wheel_link"),
                "make_consistent": True})
env = gym.make(TASK, cfg=cfg).unwrapped
env.reset()
r = env.scene["robot"]
T = lambda x: x.torch if hasattr(x, "torch") else x  # noqa: E731
dev = env.device

# from rest at each env's origin, heading +x
ids = torch.arange(4, device=dev)
pose = torch.zeros(4, 7, device=dev)
pose[:, :3] = env.scene.env_origins + torch.tensor(T(r.data.default_root_state)[0, :3].tolist(),
                                                   device=dev) * torch.tensor([0, 0, 1.0], device=dev)
pose[:, 6] = 1.0
r.write_root_pose_to_sim_index(root_pose=pose, env_ids=ids)
r.write_root_velocity_to_sim_index(root_velocity=torch.zeros(4, 6, device=dev), env_ids=ids)
jp = T(r.data.default_joint_pos).clone()
r.write_joint_position_to_sim_index(position=jp, env_ids=ids)
r.write_joint_velocity_to_sim_index(velocity=torch.zeros_like(jp), env_ids=ids)

PHASES = [("settle", 0.0, 0.0, 0.5), ("straight", 2.0, 0.0, 2.0), ("turn", 2.0, 0.3, 2.0),
          ("hard_turn", 3.0, 0.488, 2.0), ("brake", 0.0, 0.0, 3.0)]
vmax, smax = 3.0, 0.488
dt = env.step_dt
assert abs(dt - 0.005) < 1e-9, dt
trace, last, k = [], {}, 0
brake = {}
for name, v, d, dur in PHASES:
    a = torch.tensor([[v / vmax, d / smax]] * 4, device=dev)
    for j in range(int(round(dur / dt))):
        env.step(a)
        k += 1
        vx = float(T(r.data.root_lin_vel_b)[0, 0])
        wz = float(T(r.data.root_ang_vel_b)[0, 2])
        p = T(r.data.root_pos_w)[0] - env.scene.env_origins[0]
        x, y = float(p[0]), float(p[1])
        if name == "brake":
            if j == 0:
                brake = {"start_speed_mps": round(last["hard_turn"]["v"], 3), "x0": x, "y0": y,
                         "time_to_stop_s": None}
            if brake["time_to_stop_s"] is None and abs(vx) < 0.02:
                brake["time_to_stop_s"] = round((j + 1) * dt, 3)
        if k % 10 == 0:
            trace.append([round(k * dt, 3), round(x, 4), round(y, 4), round(vx, 4), round(wz, 4)])
        last[name] = {"v": vx, "yaw_rate": wz, "x": x, "y": y}
brake["distance_m"] = round(math.hypot(last["brake"]["x"] - brake.pop("x0"),
                                       last["brake"]["y"] - brake.pop("y0")), 3)
out = {"task": TASK, "body": "MUSHR_SUS_2WD_CFG", "dt": dt, "wheel_friction": args.wheel_friction,
       "phases": {n: {"v": round(last[n]["v"], 3), "yaw_rate": round(last[n]["yaw_rate"], 3)}
                  for n, *_ in PHASES[1:4]},
       "brake": brake}
print("BODY " + json.dumps(out), flush=True)
if args.out:
    json.dump({**out, "trace_fields": ["t", "x", "y", "v", "yaw_rate"], "trace": trace},
              open(args.out, "w"))
env.close()
app.close()
