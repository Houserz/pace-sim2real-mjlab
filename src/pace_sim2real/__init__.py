"""PACE: Precise Adaptation through Continuous Evolution, powered by mjlab.

The public imports deliberately match the Isaac Lab edition so existing PACE
projects can move their system-identification workflow with minimal changes.
"""

__all__ = [
    "CMAESOptimizer",
    "PaceCfg",
    "PaceSim2realEnvCfg",
    "PaceSim2realSceneCfg",
]


def __getattr__(name: str):
    """Keep hardware-only imports independent from Torch and mjlab."""
    if name == "CMAESOptimizer":
        from .optim import CMAESOptimizer

        value = CMAESOptimizer
    elif name in {"PaceCfg", "PaceSim2realEnvCfg", "PaceSim2realSceneCfg"}:
        from .tasks.manager_based.pace import pace_sim2real_env_cfg

        value = getattr(pace_sim2real_env_cfg, name)
    else:
        raise AttributeError(name)
    globals()[name] = value
    return value
