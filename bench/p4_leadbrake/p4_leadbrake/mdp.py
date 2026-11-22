"""The lead-brake task's terms (scensim roadmap 094 P4, the stock side), in Isaac Lab's
manager-based style: a reset event that samples the episode, a command term that drives the lead,
an action term for the car, and observation, reward and termination functions.

The road is each env's own +x axis from its origin: one straight lane, 3.5 m wide, centred on
y = 0, `ROUTE` metres long. Quaternions are Isaac Lab 3.0's (x, y, z, w).
"""
from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from isaaclab.managers import ActionTerm, ActionTermCfg, CommandTerm, CommandTermCfg
from isaaclab.utils import configclass

from . import car

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

ROUTE = 300.0            # m
LANE_HALF = 1.75         # m: the lane is 3.5 m wide
OFF_ROAD = 3.0           # m from the lane's centre
SPEED_LIMIT = 15.0       # m/s, above the lead's highest
BRAKE_DAMPING = 6000.0   # N m s/rad: the brake saturates within 0.5 rad/s of a stop
SAFE_GAP = (2.0, 8.0)           # m: the safe stop behind a stopped lead
# per reset (CS2's family, simplified: straight, dry, day)
RANGES = {"lead_speed": (8.0, 14.0), "headway": (1.0, 3.0), "decel": (3.0, 7.0),
          "onset": (40.0, 120.0)}
LEAD_BOX = car.CONSTANTS["chassis"]["box"]   # the lead has the ego's footprint


def T(x):
    """A torch view of an Isaac Lab 3.0 data array (ProxyArray) or a tensor."""
    return x.torch if hasattr(x, "torch") else x


def yaw_xyzw(q):
    x, y, z, w = q.unbind(-1)
    return torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def ackermann(delta, wheel_base, tread):
    """The left and right front wheels' angles for the bicycle angle `delta` (+ left)."""
    t = torch.tan(delta)
    left = torch.atan2(wheel_base * t, wheel_base - 0.5 * tread * t)
    right = torch.atan2(wheel_base * t, wheel_base + 0.5 * tread * t)
    # atan2 keeps a negative angle's sign right only through the numerator's sign
    return torch.where(delta.abs() < 1e-9, torch.zeros_like(delta), left), \
        torch.where(delta.abs() < 1e-9, torch.zeros_like(delta), right)


class _State:
    """The episode's samples and the lead, per env, on the env (env.p4)."""

    def __init__(self, n, device):
        z = lambda: torch.zeros(n, device=device)  # noqa: E731
        self.lead_speed, self.headway, self.decel, self.onset = z(), z(), z(), z()
        self.lead_s, self.lead_v, self.lead_a = z(), z(), z()   # the lead's rear, on the route
        self.s_prev = z()                                       # the ego's progress, last step


def state(env) -> _State:
    if not hasattr(env, "p4"):
        env.p4 = _State(env.num_envs, env.device)
    return env.p4


def ego_s_y_yaw(env):
    robot = env.scene["robot"]
    p = T(robot.data.root_pos_w) - env.scene.env_origins
    return p[:, 0], p[:, 1], yaw_xyzw(T(robot.data.root_quat_w))


def gap(env):
    """The ego's front to the lead's rear, along the road (m)."""
    s, _, yaw = ego_s_y_yaw(env)
    return state(env).lead_s - (s + car.FRONT * torch.cos(yaw))


# ---- the reset event: the episode's samples, the ego at speed, the lead ahead ------------------

def reset_episode(env: ManagerBasedRLEnv, env_ids):
    if isinstance(env_ids, slice) or env_ids is None:   # Isaac Lab 3.0's first reset: slice(None)
        env_ids = torch.arange(env.num_envs, device=env.device)[env_ids or slice(None)]
    st = state(env)
    n = len(env_ids)
    for name, (lo, hi) in RANGES.items():
        getattr(st, name)[env_ids] = torch.empty(n, device=env.device).uniform_(lo, hi)
    v = st.lead_speed[env_ids]
    robot = env.scene["robot"]
    origins = env.scene.env_origins[env_ids]
    pose = torch.zeros(n, 7, device=env.device)
    pose[:, :3] = origins
    pose[:, 6] = 1.0                                   # (x, y, z, w): no rotation
    robot.write_root_pose_to_sim_index(root_pose=pose, env_ids=env_ids)
    vel = torch.zeros(n, 6, device=env.device)
    vel[:, 0] = v
    robot.write_root_velocity_to_sim_index(root_velocity=vel, env_ids=env_ids)
    jpos = T(robot.data.default_joint_pos)[env_ids].clone()
    jvel = torch.zeros_like(jpos)
    sus, _ = robot.find_joints(car.SUSPENSION_JOINTS)
    spin, _ = robot.find_joints(car.SPIN_JOINTS)
    jpos[:, sus] = 0.0          # the design height: the springs' target is preloaded by the sag
    jvel[:, spin] = (v / car.CONSTANTS["wheel"]["radius"]).unsqueeze(1)
    robot.write_joint_position_to_sim_index(position=jpos, env_ids=env_ids)
    robot.write_joint_velocity_to_sim_index(velocity=jvel, env_ids=env_ids)
    st.lead_s[env_ids] = car.FRONT + st.headway[env_ids] * v
    st.lead_v[env_ids] = v
    st.lead_a[env_ids] = 0.0
    st.s_prev[env_ids] = 0.0


# ---- the lead: cruise, then brake at a constant rate to a stop at its onset -------------------

class LeadCommand(CommandTerm):
    """The lead's state on the route, [rear s, speed, acceleration], stepped once a policy step
    (its kinematic box is placed there). Its episode samples come from the reset event."""

    cfg: LeadCommandCfg

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.lead = env.scene[cfg.asset_name]
        self._cmd = torch.zeros(self.num_envs, 3, device=self.device)

    @property
    def command(self):
        st = state(self._env)
        self._cmd[:, 0], self._cmd[:, 1], self._cmd[:, 2] = st.lead_s, st.lead_v, st.lead_a
        return self._cmd

    def _update_metrics(self):
        pass

    def _resample_command(self, env_ids):
        self._place(env_ids)

    def _update_command(self):
        st = state(self._env)
        dt = self._env.step_dt
        braking = (st.lead_s >= st.onset) & (st.lead_v > 0.0)
        st.lead_a = torch.where(braking, -st.decel, torch.zeros_like(st.lead_a))
        st.lead_v = torch.clamp(st.lead_v + st.lead_a * dt, min=0.0)
        st.lead_s = st.lead_s + st.lead_v * dt
        self._place(slice(None))

    def _place(self, env_ids):
        st = state(self._env)
        ids = torch.arange(self.num_envs, device=self.device)[env_ids]
        pose = torch.zeros(len(ids), 7, device=self.device)
        pose[:, :3] = self._env.scene.env_origins[ids]
        pose[:, 0] += st.lead_s[ids] + 0.5 * LEAD_BOX[0]
        pose[:, 2] += 0.5 * LEAD_BOX[2] + 0.138       # the box's underside at the chassis's
        pose[:, 6] = 1.0
        self.lead.write_root_pose_to_sim_index(root_pose=pose, env_ids=ids)


@configclass
class LeadCommandCfg(CommandTermCfg):
    class_type: type = LeadCommand
    asset_name: str = "lead"
    resampling_time_range: tuple = (1.0e9, 1.0e9)     # only at resets


# ---- the car: longitudinal torque and steering --------------------------------------------

class CarAction(ActionTerm):
    """Two actions in [-1, 1]: longitudinal (+ the rear wheels' drive torque up to the spin
    joints' effort limit, - a brake torque on every wheel up to the file's max brake torque)
    and steering (x the max steer angle, to both front wheels' Ackermann angles). The
    suspension holds its sag target."""

    cfg: CarActionCfg

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        a = self._asset
        self._sus, _ = a.find_joints(car.SUSPENSION_JOINTS)
        self._steer, _ = a.find_joints(car.STEER_JOINTS)
        self._spin, spin_names = a.find_joints(car.SPIN_JOINTS)
        self._driven_mask = torch.tensor([float(j in car.DRIVEN_JOINTS) for j in spin_names],
                                         device=self.device)
        self._raw = torch.zeros(env.num_envs, 2, device=self.device)
        self._proc = torch.zeros_like(self._raw)
        c = car.CONSTANTS
        self._sus_target = torch.full((env.num_envs, len(self._sus)), c["suspension"]["target"],
                                      device=self.device)
        self._drive_max = c["spin"]["effort_limit"]
        self._brake_max = c["actuation"]["max_brake_torque"]
        self._steer_max = c["actuation"]["max_steer_angle"]

    @property
    def action_dim(self):
        return 2

    @property
    def raw_actions(self):
        return self._raw

    @property
    def processed_actions(self):
        return self._proc

    def process_actions(self, actions):
        self._raw[:] = actions
        self._proc[:] = torch.clamp(actions, -1.0, 1.0)
        # the brake is the spin joints' implicit damping toward zero speed, capped at the brake
        # fraction of the max brake torque: stable at any gain, where a torque computed from the
        # last wheel speed chatters (3000 N m swings a 1.37 kg m^2 wheel 18 rad/s per 120 Hz step)
        b = torch.clamp(-self._proc[:, 0], min=0.0).unsqueeze(1).expand(-1, len(self._spin))
        braking = b > 0.0
        self._asset.write_joint_damping_to_sim_index(
            damping=torch.where(braking, BRAKE_DAMPING, 0.0).contiguous(), joint_ids=self._spin)
        self._asset.write_joint_effort_limit_to_sim_index(
            limits=torch.where(braking, b * self._brake_max, self._drive_max).contiguous(),
            joint_ids=self._spin)

    def apply_actions(self):
        a = self._asset
        lon, steer = self._proc[:, 0], self._proc[:, 1]
        left, right = ackermann(steer * self._steer_max, car.CONSTANTS["actuation"]["wheel_base"],
                                car.CONSTANTS["actuation"]["wheel_tread"])
        a.set_joint_position_target_index(target=torch.stack([left, right], 1), joint_ids=self._steer)
        a.set_joint_position_target_index(target=self._sus_target, joint_ids=self._sus)
        # drive torque on the rear wheels (the brake is in process_actions)
        drive = torch.clamp(lon, min=0.0) * self._drive_max
        a.set_joint_effort_target_index(target=drive.unsqueeze(1) * self._driven_mask,
                                        joint_ids=self._spin)


@configclass
class CarActionCfg(ActionTermCfg):
    class_type: type = CarAction
    asset_name: str = "robot"


# ---- observations --------------------------------------------------------------------------

def obs_ego(env):
    """Speed, yaw rate, lateral offset, heading error."""
    robot = env.scene["robot"]
    _, y, yaw = ego_s_y_yaw(env)
    v = T(robot.data.root_lin_vel_b)[:, 0]
    r = T(robot.data.root_ang_vel_b)[:, 2]
    return torch.stack([v, r, y, yaw], 1)


def obs_lead(env):
    """The gap (front to rear), the closing speed, the lead's acceleration."""
    st = state(env)
    v = T(env.scene["robot"].data.root_lin_vel_b)[:, 0]
    return torch.stack([gap(env), v - st.lead_v, st.lead_a], 1)


# ---- the reward: CaRL's, route progress times multiplicative penalties ---------------------

def carl_reward(env):
    st = state(env)
    s, y, _ = ego_s_y_yaw(env)
    progress = (s - st.s_prev) / ROUTE
    st.s_prev = s.clone()
    v = T(env.scene["robot"].data.root_lin_vel_b)[:, 0]
    m_speed = 1.0 - torch.clamp((v - SPEED_LIMIT).clamp(min=0.0) / SPEED_LIMIT, max=1.0)
    m_lane = 1.0 - torch.clamp(y.abs() / LANE_HALF, max=1.0)
    r = progress * m_speed * m_lane
    return torch.where(gap(env) <= 0.0, torch.zeros_like(r), r)   # no reward on a collision


# ---- terminations ----------------------------------------------------------------------------

def collided(env):
    return gap(env) <= 0.0


def off_road(env):
    _, y, _ = ego_s_y_yaw(env)
    return y.abs() > OFF_ROAD


def route_done(env):
    s, _, _ = ego_s_y_yaw(env)
    return s >= ROUTE


def safe_stop_reward(env):
    """1 in the step an episode times out standing in the bar's window (`safe_stop`), else 0:
    the return then rewards what the pass bar measures. Rewards are computed after the step's
    terminations, so the time-out is known here."""
    out = env.termination_manager.get_term("time_out") & safe_stop(env)
    return out.float()


def safe_stop(env):
    """Standing behind the stopped lead, its gap within SAFE_GAP: a success at time-out. The
    bar was set after the smoke runs: run 2's policy stopped 13.9-15.4 m back, which a bare
    >= 2 m passed; a stop that far back is timid, not good, so it fails now."""
    v = T(env.scene["robot"].data.root_lin_vel_b)[:, 0]
    st = state(env)
    g = gap(env)
    return (st.lead_v <= 0.0) & (v.abs() < 0.1) & (g >= SAFE_GAP[0]) & (g <= SAFE_GAP[1])


def record_outcome(env):
    """Never terminates: keeps each env's state as of this step, before the step's resets, so an
    evaluator can judge the episodes that just ended (safe stop, how far, where in the lane)."""
    st = state(env)
    s, y, _ = ego_s_y_yaw(env)
    st.end_safe = safe_stop(env)
    st.end_s, st.end_y, st.end_gap = s.clone(), y.clone(), gap(env)
    return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
