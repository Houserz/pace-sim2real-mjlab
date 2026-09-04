"""Dependency-free Dobot Rover identification contract."""

from __future__ import annotations

DOBOT_JOINT_ORDER = (
    "joint_front_left_abad",
    "joint_front_left_thigh_pitch",
    "joint_front_left_calf_pitch",
    "joint_front_right_abad",
    "joint_front_right_thigh_pitch",
    "joint_front_right_calf_pitch",
    "joint_rear_left_abad",
    "joint_rear_left_thigh_pitch",
    "joint_rear_left_calf_pitch",
    "joint_rear_right_abad",
    "joint_rear_right_thigh_pitch",
    "joint_rear_right_calf_pitch",
)

DOBOT_LEG_INDICES = {
    "FL": (0, 1, 2),
    "FR": (3, 4, 5),
    "RL": (6, 7, 8),
    "RR": (9, 10, 11),
}
DOBOT_LEG_JOINTS = {
    leg: tuple(DOBOT_JOINT_ORDER[index] for index in indices)
    for leg, indices in DOBOT_LEG_INDICES.items()
}
DOBOT_TASK_IDS = {leg: f"Dobot-Pace-{leg}-v0" for leg in DOBOT_LEG_INDICES}

DOBOT_DEFAULT_JOINT_POS = (
    0.0,
    0.84,
    -1.30,
    0.0,
    0.84,
    -1.30,
    0.0,
    0.75,
    -1.22,
    0.0,
    0.75,
    -1.22,
)
DOBOT_JOINT_LOWER = (-0.6632, -2.618, -2.53) * 4
DOBOT_JOINT_UPPER = (0.6632, 2.618, 2.53) * 4
DOBOT_EFFORT_LIMITS = (23.0, 23.0, 55.0) * 4
DOBOT_PHYSICS_DT = 0.0025
DOBOT_XML_SHA256 = "ebb45f4cd4697cef2f24659675affbd788bdc25d49a4e2f115ab81e514b7fd55"


def normalize_leg(leg: str) -> str:
    """Return one supported single-leg identifier."""
    normalized = leg.upper()
    if normalized not in DOBOT_LEG_INDICES:
        raise ValueError(f"unknown Dobot leg {leg!r}; choose from {tuple(DOBOT_LEG_INDICES)}")
    return normalized


def task_id_for_leg(leg: str) -> str:
    """Return the registered mjlab task for one Dobot leg."""
    return DOBOT_TASK_IDS[normalize_leg(leg)]
