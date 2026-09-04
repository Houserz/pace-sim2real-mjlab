"""Dobot single-leg hold and chirp trajectory generation."""

from __future__ import annotations

from typing import Any

import numpy as np

from pace_sim2real.dobot import (
    DOBOT_JOINT_LOWER,
    DOBOT_JOINT_UPPER,
    DOBOT_LEG_INDICES,
    normalize_leg,
)
from pace_sim2real.hardware.config import vector


def _baseline(value: np.ndarray) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (12,) or not np.isfinite(result).all():
        raise ValueError("baseline must be a finite 12-joint vector")
    return result


def generate_hold(
    config: dict[str, Any], baseline: np.ndarray, leg: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    baseline = _baseline(baseline)
    indices = np.asarray(DOBOT_LEG_INDICES[normalize_leg(leg)], dtype=np.int64)
    rate = 1.0 / float(config["physics_dt"])
    hold = config["hold"]
    target = baseline.copy()
    reviewed = vector(config, "hold", "target_joint_pos", 12)
    target[indices] = reviewed[indices]
    lower = np.asarray(DOBOT_JOINT_LOWER)
    upper = np.asarray(DOBOT_JOINT_UPPER)
    violation = np.maximum(lower - baseline, baseline - upper)
    if np.any(violation[indices] > float(hold["max_start_limit_violation_rad"])):
        raise ValueError("selected leg starts outside the recoverable configured joint range")
    if np.any(target[indices] <= lower[indices]) or np.any(target[indices] >= upper[indices]):
        raise ValueError("selected hold target must lie strictly inside joint limits")

    ramp_count = max(2, round(float(hold["ramp_s"]) * rate))
    hold_count = max(2, round(float(hold["duration_s"]) * rate))
    blend = np.linspace(0.0, 1.0, ramp_count)
    blend = 0.5 - 0.5 * np.cos(np.pi * blend)
    targets = np.concatenate(
        (
            baseline[None, :] + blend[:, None] * (target - baseline)[None, :],
            np.repeat(target[None, :], hold_count, axis=0),
        )
    )
    velocity = float(np.max(np.abs(np.diff(targets, axis=0)) * rate))
    if velocity > float(hold["max_command_velocity_rad_s"]):
        raise ValueError(f"hold transition reaches {velocity:.6f} rad/s")
    phases = np.concatenate(
        (np.full(ramp_count, "approach", dtype="<U8"), np.full(hold_count, "hold", dtype="<U8"))
    )
    return np.arange(len(targets)) / rate, targets, phases


def generate_chirp(
    config: dict[str, Any], center: np.ndarray, leg: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    center = _baseline(center)
    indices = np.asarray(DOBOT_LEG_INDICES[normalize_leg(leg)], dtype=np.int64)
    chirp = config["chirp"]
    rate = 1.0 / float(config["physics_dt"])
    pre = round(float(chirp["pre_hold_s"]) * rate)
    count = round(float(chirp["duration_s"]) * rate)
    post = round(float(chirp["post_hold_s"]) * rate)
    targets = np.repeat(center[None, :], pre + count + post, axis=0)
    phases = np.full(len(targets), "posthold", dtype="<U8")
    phases[:pre] = "prehold"
    local_time = np.arange(count) / rate
    duration = float(chirp["duration_s"])
    phase = (
        2.0
        * np.pi
        * (
            float(chirp["min_frequency_hz"]) * local_time
            + (float(chirp["max_frequency_hz"]) - float(chirp["min_frequency_hz"]))
            * local_time**2
            / (2.0 * duration)
        )
    )
    ramp = float(chirp["ramp_s"])
    envelope = np.sin(0.5 * np.pi * np.clip(local_time / ramp, 0.0, 1.0)) ** 2
    envelope *= np.sin(0.5 * np.pi * np.clip((duration - local_time) / ramp, 0.0, 1.0)) ** 2
    offsets = (
        (envelope * np.sin(phase))[:, None]
        * vector(config, "chirp", "amplitude_rad", 3)[None, :]
        * vector(config, "chirp", "direction", 3)[None, :]
    )
    targets[pre : pre + count, indices] += offsets
    phases[pre : pre + count] = "chirp"
    lower = np.asarray(DOBOT_JOINT_LOWER)[indices]
    upper = np.asarray(DOBOT_JOINT_UPPER)[indices]
    if np.any(targets[:, indices] <= lower) or np.any(targets[:, indices] >= upper):
        raise ValueError("generated chirp reaches a configured joint limit")
    return np.arange(len(targets)) / rate, targets, phases


def generate_identification(
    config: dict[str, Any], baseline: np.ndarray, leg: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    _, hold_targets, hold_phases = generate_hold(config, baseline, leg)
    _, chirp_targets, chirp_phases = generate_chirp(config, hold_targets[-1], leg)
    targets = np.concatenate((hold_targets, chirp_targets))
    phases = np.concatenate((hold_phases, chirp_phases))
    dt = float(config["physics_dt"])
    return np.arange(len(targets)) * dt, targets, phases
