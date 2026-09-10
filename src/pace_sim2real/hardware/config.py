"""Load and validate the portable Dobot hardware configuration."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from pace_sim2real.dobot import DOBOT_PHYSICS_DT

SCHEMA = "pace_dobot_hardware_v1"


def vector(config: dict[str, Any], section: str, key: str, length: int) -> np.ndarray:
    value = np.asarray(config[section][key], dtype=np.float64)
    if value.shape != (length,) or not np.isfinite(value).all():
        raise ValueError(f"{section}.{key} must be a finite vector with {length} values")
    return value


def load_config(path: str | Path) -> dict[str, Any]:
    source = Path(path).expanduser().resolve()
    config = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(config, dict) or config.get("schema") != SCHEMA:
        raise ValueError(f"expected config schema {SCHEMA!r}: {source}")
    config["_path"] = str(source)
    for key in ("config", "cyclonedds_uri"):
        value = Path(config["dds"][key]).expanduser()
        if not value.is_absolute():
            value = source.parent / value
        config["dds"][key] = str(value.resolve())
    config["chirp"].setdefault("phase_deg", [0.0, 0.0, 0.0])
    validate_config(config)
    return config


def validate_config(config: dict[str, Any]) -> None:
    dt = float(config["physics_dt"])
    if not np.isclose(dt, DOBOT_PHYSICS_DT, rtol=0.0, atol=1.0e-12):
        raise ValueError(f"physics_dt must be {DOBOT_PHYSICS_DT:g} s (400 Hz)")
    for key in ("config", "cyclonedds_uri"):
        if not Path(config["dds"][key]).is_file():
            raise FileNotFoundError(config["dds"][key])
    abs2hw = np.asarray(config["dds"]["abs2hw"], dtype=np.int64)
    if abs2hw.shape != (12,) or len(set(abs2hw.tolist())) != 12 or np.any(abs2hw < 0):
        raise ValueError("dds.abs2hw must contain 12 unique non-negative motor indices")
    offsets = vector(config, "dds", "motor_offset", 16)
    if int(abs2hw.max()) >= len(offsets):
        raise ValueError("dds.abs2hw references a missing motor_offset")
    for section, key, length in (
        ("control", "kp", 3),
        ("control", "kd", 3),
        ("hold", "target_joint_pos", 12),
        ("chirp", "amplitude_rad", 3),
        ("chirp", "phase_deg", 3),
        ("chirp", "direction", 3),
    ):
        vector(config, section, key, length)
    if np.any(vector(config, "control", "kp", 3) < 0) or np.any(
        vector(config, "control", "kd", 3) < 0
    ):
        raise ValueError("control gains must be non-negative")
    if np.any(vector(config, "chirp", "amplitude_rad", 3) < 0) or not np.all(
        np.isin(vector(config, "chirp", "direction", 3), (-1.0, 1.0))
    ):
        raise ValueError("chirp amplitude must be non-negative and direction must be +/-1")
    positive = {
        "dds.first_state_timeout_s": config["dds"]["first_state_timeout_s"],
        "control.damping_kd": config["control"]["damping_kd"],
        "control.shutdown_damping_s": config["control"]["shutdown_damping_s"],
        "hold.ramp_s": config["hold"]["ramp_s"],
        "hold.duration_s": config["hold"]["duration_s"],
        "hold.max_command_velocity_rad_s": config["hold"]["max_command_velocity_rad_s"],
        "hold.stability_window_s": config["hold"]["stability_window_s"],
        "chirp.min_frequency_hz": config["chirp"]["min_frequency_hz"],
        "chirp.max_frequency_hz": config["chirp"]["max_frequency_hz"],
        "chirp.duration_s": config["chirp"]["duration_s"],
        "chirp.ramp_s": config["chirp"]["ramp_s"],
        "safety.state_timeout_ms": config["safety"]["state_timeout_ms"],
        "safety.max_state_interpolation_gap_ms": config["safety"]["max_state_interpolation_gap_ms"],
        "safety.max_schedule_lag_ms": config["safety"]["max_schedule_lag_ms"],
        "safety.max_abs_joint_velocity_rad_s": config["safety"]["max_abs_joint_velocity_rad_s"],
        "safety.max_abs_tracking_error_rad": config["safety"]["max_abs_tracking_error_rad"],
        "safety.max_motor_temperature_c": config["safety"]["max_motor_temperature_c"],
    }
    for name, value in positive.items():
        if float(value) <= 0:
            raise ValueError(f"{name} must be positive")
    for key in ("pre_hold_s", "post_hold_s"):
        value = float(config["chirp"][key])
        if not np.isfinite(value) or value < 0:
            raise ValueError(f"chirp.{key} must be finite and non-negative")
    if float(config["chirp"]["max_frequency_hz"]) < float(config["chirp"]["min_frequency_hz"]):
        raise ValueError("chirp.max_frequency_hz must be >= chirp.min_frequency_hz")
    if 2.0 * float(config["chirp"]["ramp_s"]) > float(config["chirp"]["duration_s"]):
        raise ValueError("chirp.ramp_s cannot exceed half of chirp.duration_s")
    tolerance = float(config["safety"]["command_rate_tolerance_fraction"])
    if not 0.0 <= tolerance < 1.0:
        raise ValueError("safety.command_rate_tolerance_fraction must be in [0, 1)")
