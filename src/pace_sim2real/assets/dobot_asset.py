"""Fixed-base, contact-free Dobot Rover asset for PACE identification."""

from __future__ import annotations

from pathlib import Path

import mujoco
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg

from pace_sim2real.dobot import (
    DOBOT_DEFAULT_JOINT_POS,
    DOBOT_EFFORT_LIMITS,
    DOBOT_JOINT_ORDER,
    DOBOT_LEG_INDICES,
    DOBOT_LEG_JOINTS,
    DOBOT_XML_SHA256,
    normalize_leg,
)
from pace_sim2real.utils import PaceDCMotorCfg

ASSET_DIR = Path(__file__).with_name("dobot")
DOBOT_XML = ASSET_DIR / "dobot.xml"


def get_spec() -> mujoco.MjSpec:
    """Load the audited model, fix its trunk, and disable every contact geom."""
    spec = mujoco.MjSpec.from_file(str(DOBOT_XML))
    spec.assets = {
        f"{spec.meshdir}/{path.name}": path.read_bytes()
        for path in (ASSET_DIR / "assets").glob("*.STL")
    }
    free_joint = next((joint for joint in spec.joints if joint.name == "joint_fixed_world"), None)
    if free_joint is None or free_joint.type != mujoco.mjtJoint.mjJNT_FREE:
        raise ValueError("Dobot model must contain free joint 'joint_fixed_world'")
    spec.delete(free_joint)
    for geom in spec.geoms:
        geom.contype = 0
        geom.conaffinity = 0
    spec.modelname = "dobot_rover_pace"
    return spec


def _initial_state() -> EntityCfg.InitialStateCfg:
    return EntityCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.65),
        joint_pos=dict(zip(DOBOT_JOINT_ORDER, DOBOT_DEFAULT_JOINT_POS, strict=True)),
        joint_vel={".*": 0.0},
    )


def get_dobot_robot_cfg(leg: str) -> EntityCfg:
    """Return a fresh fixed-base robot with PACE actuators for one selected leg."""
    leg = normalize_leg(leg)
    indices = DOBOT_LEG_INDICES[leg]
    effort_limits = tuple(DOBOT_EFFORT_LIMITS[index] for index in indices)
    actuator = PaceDCMotorCfg(
        joint_names_expr=DOBOT_LEG_JOINTS[leg],
        saturation_effort=effort_limits,
        effort_limit=effort_limits,
        velocity_limit=(20.0, 20.0, 20.0),
        stiffness=(25.0, 25.0, 25.0),
        damping=(1.3, 1.3, 1.3),
        encoder_bias=0.0,
        armature=(0.000074, 0.000074, 0.000074),
        frictionloss=(0.02, 0.02, 0.02),
        viscous_damping=(0.02, 0.02, 0.02),
        max_delay=10,
    )
    return EntityCfg(
        spec_fn=get_spec,
        articulation=EntityArticulationInfoCfg(
            actuators=(actuator,), soft_joint_pos_limit_factor=1.0
        ),
        init_state=_initial_state(),
    )


__all__ = [
    "DOBOT_JOINT_ORDER",
    "DOBOT_LEG_JOINTS",
    "DOBOT_XML",
    "DOBOT_XML_SHA256",
    "get_dobot_robot_cfg",
    "get_spec",
]
