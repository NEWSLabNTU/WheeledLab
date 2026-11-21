import os

from rsl_rl import runners


class OnPolicyRunner(runners.OnPolicyRunner):
    """rsl_rl's runner with WheeledLab's logging choices (rsl-rl >= 5).

    Upstream's runner was a copy of rsl-rl 2.3's `learn()` loop, kept to choose the logger.
    rsl-rl 5 changed that loop's API (one observation TensorDict, `act(obs)`, the logger
    object), so the loop is rsl_rl's own now and only the choices stay here:

    - `log_cfg.no_log`: no log directory, so nothing is written or saved;
    - `log_cfg.no_wandb`: TensorBoard instead of Weights & Biases;
    - checkpoints go to `<run_log_dir>/<model_save_dirname>/model_<it>.pt`, where
      `play_policy.py` looks for them.
    """

    def __init__(self, env, agent_cfg, log_cfg, device="cpu"):
        cfg = dict(agent_cfg)
        self.no_log = log_cfg.no_log
        self.no_wandb = log_cfg.no_wandb
        self.model_dir = log_cfg.model_save_path
        if not self.no_wandb:
            cfg["logger"] = {"class_name": "WandbLogWriter", "project_name": log_cfg.wandb_project}
        else:
            cfg["logger"] = "tensorboard"
        super().__init__(env, cfg, None if self.no_log else log_cfg.run_log_dir, device)

    def save(self, path, infos=None):
        os.makedirs(self.model_dir, exist_ok=True)
        super().save(os.path.join(self.model_dir, os.path.basename(path)), infos)
