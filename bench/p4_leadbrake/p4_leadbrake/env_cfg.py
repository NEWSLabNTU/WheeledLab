"""The lead-brake environment's configuration (scensim roadmap 094 P4, the stock side): Isaac Lab
3.0's manager-based style, as its cartpole is written."""
from __future__ import annotations

import os

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.envs import mdp as base
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass

from . import car, mdp

# the car's USD, written at its first spawn (car.spawn_car), keyed by the source file's hash
CACHE = os.environ.get("P4_LEADBRAKE_CACHE", os.path.expanduser("~/.cache/p4_leadbrake"))
CAR_USD = os.path.join(CACHE, f"model3_{car.SOURCE['physics_hash'][:12]}.usda")


@configclass
class SceneCfg(InteractiveSceneCfg):
    # one ground for every env; each env's road is its own +x axis (filter_collisions keeps the
    # envs apart, as they cross each other's cells)
    # a dry road: friction 1.0, which the tyre's 1.0 multiplies (Isaac Lab's ground is 0.5)
    ground = AssetBaseCfg(prim_path="/World/ground", spawn=sim_utils.GroundPlaneCfg(
        size=(4000.0, 4000.0), physics_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=1.0, dynamic_friction=1.0, restitution=0.0)))
    robot = car.car_cfg(CAR_USD).replace(prim_path="{ENV_REGEX_NS}/Car")
    lead = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Lead",
        spawn=sim_utils.CuboidCfg(
            size=mdp.LEAD_BOX,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=1847.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(30.0, 0.0, 0.813)),
    )
    light = AssetBaseCfg(prim_path="/World/light",
                         spawn=sim_utils.DistantLightCfg(intensity=2000.0))


@configclass
class ActionsCfg:
    car = mdp.CarActionCfg()


@configclass
class CommandsCfg:
    lead = mdp.LeadCommandCfg()


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        ego = ObsTerm(func=mdp.obs_ego)
        lead = ObsTerm(func=mdp.obs_lead)
        last_action = ObsTerm(func=base.last_action)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    episode = EventTerm(func=mdp.reset_episode, mode="reset")


@configclass
class RewardsCfg:
    carl = RewTerm(func=mdp.carl_reward, weight=1.0)
    # a terminal infraction costs a whole route: CaRL's own cost of ending early, the progress
    # given up, is nothing here, as the lead always stops before the route's end (run 1: the
    # crash rate rose to 72 % without it)
    infraction = RewTerm(func=base.is_terminated_term, weight=-1.0,
                         params={"term_keys": ["collided", "off_road"]})


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=base.time_out, time_out=True)
    collided = DoneTerm(func=mdp.collided)
    off_road = DoneTerm(func=mdp.off_road)
    route_done = DoneTerm(func=mdp.route_done)
    outcome = DoneTerm(func=mdp.record_outcome)      # a probe for evaluation; never fires


@configclass
class LeadBrakeEnvCfg(ManagerBasedRLEnvCfg):
    scene: SceneCfg = SceneCfg(num_envs=1024, env_spacing=10.0)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()

    def __post_init__(self):
        self.decimation = 6              # 20 Hz policy over 120 Hz physics
        self.episode_length_s = 30.0
        self.sim.dt = 1.0 / 120.0
        self.sim.render_interval = self.decimation
