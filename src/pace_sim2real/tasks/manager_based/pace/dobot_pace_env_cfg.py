"""Single-leg Dobot Rover PACE task configurations."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from mjlab.entity import EntityCfg

from pace_sim2real.assets.dobot_asset import DOBOT_LEG_JOINTS, get_dobot_robot_cfg
from pace_sim2real.dobot import DOBOT_PHYSICS_DT, normalize_leg

from .pace_sim2real_env_cfg import PaceCfg, PaceSim2realEnvCfg, PaceSim2realSceneCfg


def dobot_bounds(joint_count: int = 3) -> torch.Tensor:
    """Return conservative Dobot bounds in PACE's 4N+1 layout."""
    bounds = torch.zeros((4 * joint_count + 1, 2), dtype=torch.float32)
    bounds[:joint_count, 0] = 1.0e-6
    bounds[:joint_count, 1] = 1
    bounds[joint_count : 2 * joint_count, 1] = 7.0
    bounds[2 * joint_count : 3 * joint_count, 1] = 0.7
    bounds[3 * joint_count : 4 * joint_count, 0] = -0.2
    bounds[3 * joint_count : 4 * joint_count, 1] = 0.2
    bounds[-1, 1] = 10.0
    return bounds


@dataclass(kw_only=True)
class DobotPaceCfg(PaceCfg):
    leg: str = "FL"
    robot_name: str = "dobot_fl"
    data_dir: str = "dobot/fl/chirp_data.pt"
    joint_order: list[str] = field(default_factory=lambda: list(DOBOT_LEG_JOINTS["FL"]))
    bounds_params: torch.Tensor = field(default_factory=dobot_bounds)


@dataclass(kw_only=True)
class DobotPaceSceneCfg(PaceSim2realSceneCfg):
    robot: EntityCfg = field(default_factory=lambda: get_dobot_robot_cfg("FL"))
    terrain: None = None


@dataclass(kw_only=True)
class DobotPaceEnvCfg(PaceSim2realEnvCfg):
    scene: DobotPaceSceneCfg = field(default_factory=DobotPaceSceneCfg)
    sim2real: DobotPaceCfg = field(default_factory=DobotPaceCfg)

    def __post_init__(self) -> None:
        super().__post_init__()
        self.sim.dt = DOBOT_PHYSICS_DT


def dobot_pace_env_cfg(
    leg: str = "FL", *, play: bool = False, num_envs: int = 4096
) -> DobotPaceEnvCfg:
    """Create a registered fixed-base task for one Dobot leg."""
    leg = normalize_leg(leg)
    sim2real = DobotPaceCfg(
        leg=leg,
        robot_name=f"dobot_{leg.lower()}",
        data_dir=f"dobot/{leg.lower()}/chirp_data.pt",
        joint_order=list(DOBOT_LEG_JOINTS[leg]),
    )
    cfg = DobotPaceEnvCfg(
        scene=DobotPaceSceneCfg(num_envs=num_envs, robot=get_dobot_robot_cfg(leg)),
        sim2real=sim2real,
    )
    if play:
        cfg.episode_length_s = 1.0e9
    return cfg


__all__ = ["DobotPaceCfg", "DobotPaceEnvCfg", "dobot_bounds", "dobot_pace_env_cfg"]
