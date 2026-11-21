"""The car's constants against the file they were copied from (when it is at hand: set
P4_ARTICULATION_YAML to sim_vehicle_spec's generated/tesla_model3/articulation_carla-0.9.16.yaml),
and the Ackermann helper. Runs on system python: PYTHONPATH=. python3 -m pytest tests"""
import math
import os

import pytest

from p4_leadbrake import car

YAML = os.environ.get("P4_ARTICULATION_YAML")


@pytest.mark.skipif(not YAML or not os.path.exists(YAML), reason="P4_ARTICULATION_YAML not set")
def test_constants_are_the_files():
    import yaml

    f = yaml.safe_load(open(YAML))
    assert f["spec_hash"] == car.SOURCE["spec_hash"]
    assert f["physics_hash"] == car.SOURCE["physics_hash"]
    c, L, J = car.CONSTANTS, f["links"], f["joints"]
    assert tuple(L["chassis"]["shape"]["box"]) == c["chassis"]["box"]
    assert tuple(L["chassis"]["shape"]["center"]) == c["chassis"]["box_center"]
    assert L["chassis"]["mass"] == c["chassis"]["mass"]
    assert tuple(L["chassis"]["com"]) == c["chassis"]["com"]
    assert tuple(L["chassis"]["inertia"]) == c["chassis"]["inertia"]
    for corner, (x, y) in car._CORNERS.items():
        front = corner.startswith("front")
        w = L[f"{corner}_wheel"]
        assert tuple(w["position"]) == (x, y, car.WHEEL_Z)
        assert w["mass"] == c["front_wheel" if front else "rear_wheel"]["mass"]
        assert tuple(w["inertia"]) == c["front_wheel" if front else "rear_wheel"]["inertia"]
        assert w["shape"]["cylinder"]["radius"] == c["wheel"]["radius"]
        assert w["shape"]["cylinder"]["width"] == c["wheel"]["width"]
        for k in ("static_friction", "dynamic_friction", "restitution", "combine"):
            assert w["contact"][k] == c["wheel"][k]
        assert L[f"{corner}_upright"]["mass"] == c["upright"]["mass"]
        s = J[f"{corner}_suspension"]
        assert tuple(s["limits"]) == c["suspension"]["limits"]
        side = c["suspension"]["front" if front else "rear"]
        assert s["drive"]["stiffness"] == side["stiffness"]
        assert s["drive"]["damping"] == side["damping"]
        assert s["drive"]["target"] == c["suspension"]["target"]
        sp = J[f"{corner}_spin"]
        assert sp["driven"] == (f"{corner}_spin" in car.DRIVEN_JOINTS)
        assert sp["drive"]["effort_limit"] == c["spin"]["effort_limit"]
        assert sp["drive"]["velocity_limit"] == c["spin"]["velocity_limit"]
        if front:
            st = J[f"{corner}_steer"]
            assert tuple(st["limits"]) == c["steer"]["limits"]
            assert st["drive"]["stiffness"] == c["steer"]["stiffness"]
            assert st["drive"]["damping"] == c["steer"]["damping"]
    a = f["actuation"]
    for k, v in c["actuation"].items():
        assert a[k] == v


def test_extent():
    assert car.FRONT == pytest.approx(3.812)
    assert car.REAR == pytest.approx(0.98)
