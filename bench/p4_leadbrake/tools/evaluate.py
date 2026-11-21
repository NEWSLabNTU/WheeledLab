"""The pass bar of scensim roadmap 094 P4: one episode in each of 256 conditions drawn with a
fixed seed, judged by the task's own terms. An episode succeeds with no collision and no lane
exit, and either the route's end or a safe stop behind the stopped lead (>= 2 m, at rest) at
time-out. A checkpoint passes at >= 95 % successes.

    python tools/evaluate.py --checkpoint <model_N.pt> [--conditions 256] [--seed 20261122]

Prints one JSON line: successes, the end reasons, and the episodes' end gap and lateral offset.
"""
import argparse
import json

from isaaclab.app import AppLauncher

ap = argparse.ArgumentParser()
ap.add_argument("--checkpoint", required=True)
ap.add_argument("--conditions", type=int, default=256)
ap.add_argument("--seed", type=int, default=20261122)
AppLauncher.add_app_launcher_args(ap)
args = ap.parse_args()
app = AppLauncher(args).app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import p4_leadbrake  # noqa: E402,F401
from isaaclab.utils.seed import configure_seed  # noqa: E402
from isaaclab_rl.rsl_rl import (RslRlVecEnvWrapper, check_rsl_rl_version,  # noqa: E402
                                 create_rsl_rl_runner, handle_deprecated_rsl_rl_cfg)
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry  # noqa: E402
from p4_leadbrake import mdp  # noqa: E402

env_cfg = load_cfg_from_registry("P4-LeadBrake-v0", "env_cfg_entry_point")
agent_cfg = load_cfg_from_registry("P4-LeadBrake-v0", "rsl_rl_cfg_entry_point")
agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, check_rsl_rl_version())   # as train and play do
env_cfg.scene.num_envs = args.conditions
env_cfg.seed = args.seed
configure_seed(args.seed)
env = RslRlVecEnvWrapper(gym.make("P4-LeadBrake-v0", cfg=env_cfg), clip_actions=agent_cfg.clip_actions)
runner = create_rsl_rl_runner(env, agent_cfg)
runner.load(args.checkpoint)
policy = runner.get_inference_policy(device=env.unwrapped.device)
u = env.unwrapped
n = args.conditions
st = mdp.state(u)
cond = {k: getattr(st, k).clone().tolist() for k in mdp.RANGES}   # the first episodes' samples
done = torch.zeros(n, dtype=torch.bool, device=u.device)
why = [None] * n
ok = torch.zeros(n, dtype=torch.bool, device=u.device)
end_gap = torch.zeros(n, device=u.device)
end_y = torch.zeros(n, device=u.device)
obs = env.get_observations()
with torch.inference_mode():
    while not bool(done.all()):
        obs, _, dones, _ = env.step(policy(obs))
        new = dones.bool() & ~done
        if bool(new.any()):
            tm = u.termination_manager
            terms = {t: tm.get_term(t) for t in ("collided", "off_road", "route_done", "time_out")}
            for i in new.nonzero().flatten().tolist():
                why[i] = next((t for t, v in terms.items() if bool(v[i])), "other")
                ok[i] = (why[i] == "route_done") or (why[i] == "time_out" and bool(st.end_safe[i]))
                end_gap[i], end_y[i] = st.end_gap[i], st.end_y[i]
            done |= new
reasons = {}
for w in why:
    reasons[w] = reasons.get(w, 0) + 1
rate = float(ok.float().mean())
print(json.dumps({
    "checkpoint": args.checkpoint, "conditions": n, "seed": args.seed,
    "successes": int(ok.sum()), "success_rate": round(rate, 4), "pass": rate >= 0.95,
    "end_reasons": reasons,
    "end_gap_m": {"min": round(float(end_gap.min()), 2), "median": round(float(end_gap.median()), 2)},
    "end_abs_y_m": {"max": round(float(end_y.abs().max()), 2)},
    "conditions_sampled": {k: [round(min(v), 2), round(max(v), 2)] for k, v in cond.items()},
}), flush=True)
env.close()
app.close()
