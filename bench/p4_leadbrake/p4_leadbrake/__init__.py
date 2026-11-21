"""scensim roadmap 094 P4's stock side: the lead car brakes ahead, a hand-written Isaac Lab
environment of a Tesla Model 3 on a straight lane, as a user writes one without scensim.

    import p4_leadbrake   # registers P4-LeadBrake-v0 (Isaac Lab must be running)
"""
try:
    import gymnasium as gym
except ImportError:      # outside Isaac Lab's python (the constants' tests): nothing to register
    gym = None

if gym is not None:
    gym.register(
        id="P4-LeadBrake-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.env_cfg:LeadBrakeEnvCfg",
            "rsl_rl_cfg_entry_point": f"{__name__}.agent:LeadBrakePPORunnerCfg",
        },
    )
