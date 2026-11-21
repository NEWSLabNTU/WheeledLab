"""The ego car: a Tesla Model 3 as an Isaac Lab articulation, built as a user without scensim
would build one (Isaac Lab 3.0 ships no car): a USD of rigid links and joints written with
pxr, spawned by Isaac Lab's UsdFileCfg, its joints driven by Isaac Lab actuators.

The numbers are copied from sim_vehicle_spec's generated articulation of the same car, so the
two sides of scensim roadmap 094 P4 drive one body (139's engine B builds from that file):

    sim_vehicle_spec generated/tesla_model3/articulation_carla-0.9.16.yaml
    spec_hash    dffb1d3df1f4ae564235a87c84c1a5a4b402f4d41d265186976561b96d31273c
    physics_hash 33635ad5c8611570ffbb9a00b495df5bdc1be06f8f6227280a4c34033acdc267

Nothing here imports scensim; `CONSTANTS` is the copy, and `test_car.py` holds it to the file
when the file is at hand.

Frame: base_link, x forward, y left, z up, the rear axle's centre on the ground.
"""
from __future__ import annotations

import os

SOURCE = {
    "file": "sim_vehicle_spec generated/tesla_model3/articulation_carla-0.9.16.yaml",
    "spec_hash": "dffb1d3df1f4ae564235a87c84c1a5a4b402f4d41d265186976561b96d31273c",
    "physics_hash": "33635ad5c8611570ffbb9a00b495df5bdc1be06f8f6227280a4c34033acdc267",
}

# link: (mass kg, com xyz, diagonal inertia, position xyz); the chassis's position is base_link
_CORNERS = {"front_left": (3.005, 0.8335), "front_right": (3.005, -0.8335),
            "rear_left": (0.0, 0.8335), "rear_right": (0.0, -0.8335)}
WHEEL_Z = 0.37
CONSTANTS = {
    "chassis": {"box": (4.792, 2.163, 1.35), "box_center": (1.416, 0.0, 0.813), "mass": 1680.0,
                "com": (1.408057143, 0.0, 0.464285714), "inertia": (779.8, 3500.0, 3731.0)},
    "upright": {"mass": 2.0, "inertia": (0.008005208, 0.011408333, 0.008005208)},
    "knuckle": {"mass": 2.0, "inertia": (0.008005208, 0.011408333, 0.008005208)},
    "front_wheel": {"mass": 16.0, "inertia": (0.758133333, 1.369, 0.758133333)},
    "rear_wheel": {"mass": 18.0, "inertia": (0.7673375, 1.369, 0.7673375)},
    "wheel": {"radius": 0.37, "width": 0.235, "static_friction": 1.0, "dynamic_friction": 1.0,
              "restitution": 0.0, "combine": "multiply"},
    "suspension": {"limits": (-0.1, 0.1),
                   "front": {"stiffness": 39779.085232832, "damping": 3561.208901144},
                   "rear": {"stiffness": 45115.303983578, "damping": 4038.93204642},
                   "target": -0.097066486},
    "steer": {"limits": (-0.7, 0.7), "stiffness": 238373.19, "damping": 854.697345635},
    "spin": {"effort_limit": 3000.0, "velocity_limit": 159.174027782},
    "actuation": {"wheel_base": 3.005, "wheel_tread": 1.667, "wheel_radius": 0.37,
                  "max_steer_angle": 0.7, "max_brake_torque": 3000.0, "drive_axle": "rear"},
}
# the body's extent about base_link, for the gap to the lead
FRONT = CONSTANTS["chassis"]["box_center"][0] + CONSTANTS["chassis"]["box"][0] / 2   # 3.812 m
REAR = CONSTANTS["chassis"]["box"][0] / 2 - CONSTANTS["chassis"]["box_center"][0]    # 0.98 m

SUSPENSION_JOINTS = [f"{c}_suspension" for c in _CORNERS]
STEER_JOINTS = ["front_left_steer", "front_right_steer"]
SPIN_JOINTS = [f"{c}_spin" for c in _CORNERS]
DRIVEN_JOINTS = ["rear_left_spin", "rear_right_spin"]


def build_usd(path: str) -> str:
    """Write the car's USD at `path` (once; it is the same for every env) and return it."""
    if os.path.exists(path):
        return path
    os.makedirs(os.path.dirname(path), exist_ok=True)
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade

    c = CONSTANTS
    stage = Usd.Stage.CreateNew(path)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    root = UsdGeom.Xform.Define(stage, "/Car")
    stage.SetDefaultPrim(root.GetPrim())
    UsdPhysics.ArticulationRootAPI.Apply(root.GetPrim())

    tyre = UsdShade.Material.Define(stage, "/Car/tyre_material")
    tm = UsdPhysics.MaterialAPI.Apply(tyre.GetPrim())
    tm.CreateStaticFrictionAttr(c["wheel"]["static_friction"])
    tm.CreateDynamicFrictionAttr(c["wheel"]["dynamic_friction"])
    tm.CreateRestitutionAttr(c["wheel"]["restitution"])
    tyre.GetPrim().CreateAttribute("physxMaterial:frictionCombineMode",
                                   Sdf.ValueTypeNames.Token).Set(c["wheel"]["combine"])

    def body(name, pos, mass, com_world, inertia):
        x = UsdGeom.Xform.Define(stage, f"/Car/{name}")
        x.AddTranslateOp().Set(Gf.Vec3d(*pos))
        UsdPhysics.RigidBodyAPI.Apply(x.GetPrim())
        m = UsdPhysics.MassAPI.Apply(x.GetPrim())
        m.CreateMassAttr(mass)
        m.CreateCenterOfMassAttr(Gf.Vec3f(*(a - b for a, b in zip(com_world, pos))))
        m.CreateDiagonalInertiaAttr(Gf.Vec3f(*inertia))
        return x

    ch = c["chassis"]
    body("chassis", (0.0, 0.0, 0.0), ch["mass"], ch["com"], ch["inertia"])
    box = UsdGeom.Cube.Define(stage, "/Car/chassis/collision")
    box.CreateSizeAttr(1.0)
    box.AddTranslateOp().Set(Gf.Vec3d(*ch["box_center"]))
    box.AddScaleOp().Set(Gf.Vec3f(*ch["box"]))
    UsdPhysics.CollisionAPI.Apply(box.GetPrim())

    def joint(kind, name, parent, child, pos, axis, limits):
        j = (UsdPhysics.PrismaticJoint if kind == "prismatic" else UsdPhysics.RevoluteJoint).Define(
            stage, f"/Car/joints/{name}")
        j.CreateBody0Rel().SetTargets([f"/Car/{parent}"])
        j.CreateBody1Rel().SetTargets([f"/Car/{child}"])
        ppos = (0.0, 0.0, 0.0) if parent == "chassis" else pos
        j.CreateLocalPos0Attr(Gf.Vec3f(*(a - b for a, b in zip(pos, ppos))))
        j.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
        j.CreateAxisAttr(axis)
        # a drive on every joint, as PhysX's articulation drives need it; Isaac Lab's actuators
        # then set its gains (SI) and targets, so the USD's values here are placeholders
        drive = UsdPhysics.DriveAPI.Apply(j.GetPrim(), "linear" if kind == "prismatic" else "angular")
        drive.CreateTypeAttr("force")
        drive.CreateStiffnessAttr(0.0)
        drive.CreateDampingAttr(0.0)
        if limits != (None, None):    # a wheel's spin has none
            lo, hi = limits
            if kind == "revolute":    # USD's revolute limits are in degrees
                lo, hi = lo * 57.29577951308232, hi * 57.29577951308232
            j.CreateLowerLimitAttr(lo)
            j.CreateUpperLimitAttr(hi)
        return j

    w = c["wheel"]
    for corner, (x, y) in _CORNERS.items():
        pos = (x, y, WHEEL_Z)
        front = corner.startswith("front")
        body(f"{corner}_upright", pos, c["upright"]["mass"], pos, c["upright"]["inertia"])
        joint("prismatic", f"{corner}_suspension", "chassis", f"{corner}_upright", pos, "Z",
              c["suspension"]["limits"])
        hub = f"{corner}_upright"
        if front:
            body(f"{corner}_knuckle", pos, c["knuckle"]["mass"], pos, c["knuckle"]["inertia"])
            joint("revolute", f"{corner}_steer", hub, f"{corner}_knuckle", pos, "Z",
                  c["steer"]["limits"])
            hub = f"{corner}_knuckle"
        wh = c["front_wheel" if front else "rear_wheel"]
        body(f"{corner}_wheel", pos, wh["mass"], pos, wh["inertia"])
        cyl = UsdGeom.Cylinder.Define(stage, f"/Car/{corner}_wheel/collision")
        cyl.CreateRadiusAttr(w["radius"])
        cyl.CreateHeightAttr(w["width"])
        cyl.CreateAxisAttr("Y")
        UsdPhysics.CollisionAPI.Apply(cyl.GetPrim())
        UsdShade.MaterialBindingAPI.Apply(cyl.GetPrim()).Bind(
            tyre, UsdShade.Tokens.weakerThanDescendants, "physics")
        joint("revolute", f"{corner}_spin", hub, f"{corner}_wheel", pos, "Y", (None, None))
    stage.GetRootLayer().Save()
    return path


def spawn_car(prim_path, cfg, translation=None, orientation=None, **kwargs):
    """Isaac Lab's USD spawner, after writing the car's USD: inside Kit, where pxr is Kit's
    (an import of pxr before Kit starts loads another USD and breaks Kit's)."""
    from isaaclab.sim.spawners.from_files import spawn_from_usd

    build_usd(cfg.usd_path)
    return spawn_from_usd(prim_path, cfg, translation, orientation, **kwargs)


def car_cfg(usd_path: str):
    """The ArticulationCfg: Isaac Lab actuators with the file's gains (SI: N/m, N m/rad)."""
    import isaaclab.sim as sim_utils
    from isaaclab.actuators import ImplicitActuatorCfg
    from isaaclab.assets import ArticulationCfg

    c = CONSTANTS
    s = c["suspension"]
    return ArticulationCfg(
        spawn=sim_utils.UsdFileCfg(
            func=spawn_car,
            usd_path=usd_path,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(max_depenetration_velocity=10.0),
            # the chassis box encloses the wheels (the file's choices): no self-collisions
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False, solver_position_iteration_count=8,
                solver_velocity_iteration_count=1),
        ),
        init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
        actuators={
            "front_suspension": ImplicitActuatorCfg(
                joint_names_expr=["front_.*_suspension"], stiffness=s["front"]["stiffness"],
                damping=s["front"]["damping"]),
            "rear_suspension": ImplicitActuatorCfg(
                joint_names_expr=["rear_.*_suspension"], stiffness=s["rear"]["stiffness"],
                damping=s["rear"]["damping"]),
            "steer": ImplicitActuatorCfg(
                joint_names_expr=STEER_JOINTS, stiffness=c["steer"]["stiffness"],
                damping=c["steer"]["damping"]),
            # torque in, as the file's consumer applies the powertrain's and brakes'
            "spin": ImplicitActuatorCfg(
                joint_names_expr=SPIN_JOINTS, stiffness=0.0, damping=0.0,
                effort_limit_sim=c["spin"]["effort_limit"],
                velocity_limit_sim=c["spin"]["velocity_limit"]),
        },
    )
