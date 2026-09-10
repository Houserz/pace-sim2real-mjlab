"""Dobot DDS adapter with no writer until an explicit active command."""

from __future__ import annotations

import os
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Any

import numpy as np

from pace_sim2real.dobot import DOBOT_JOINT_ORDER, DOBOT_LEG_INDICES, normalize_leg
from pace_sim2real.hardware.config import vector


@dataclass(frozen=True)
class StateSample:
    host_time_ns: int
    q: np.ndarray
    dq: np.ndarray
    motor_temp: np.ndarray
    tau_est: np.ndarray


def _copy(sample: StateSample) -> StateSample:
    return StateSample(
        sample.host_time_ns,
        sample.q.copy(),
        sample.dq.copy(),
        sample.motor_temp.copy(),
        sample.tau_est.copy(),
    )


class DobotDDS:
    """Read lower state and create the lower-command writer only on request."""

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.dds_config = config["dds"]
        self._lock = threading.Lock()
        self._first_state = threading.Event()
        self._samples: deque[StateSample] = deque()
        self._state_indices: np.ndarray | None = None
        self._state_offsets: np.ndarray | None = None
        self._middleware: Any = None
        self._dds: Any = None
        self._writer_enabled = False

    @property
    def writer_enabled(self) -> bool:
        return self._writer_enabled

    def start_reader(self) -> None:
        self._state_indices = np.asarray(self.dds_config["abs2hw"], dtype=np.int64)
        self._state_offsets = vector(self.config, "dds", "motor_offset", 16)
        configured_uri = _path_uri(self.dds_config["cyclonedds_uri"])
        existing_uri = os.environ.get("CYCLONEDDS_URI", "").strip()
        if existing_uri and existing_uri != configured_uri:
            raise RuntimeError(
                "CYCLONEDDS_URI conflicts with the Dobot config: "
                f"environment={existing_uri!r}, config={configured_uri!r}"
            )
        os.environ["CYCLONEDDS_URI"] = configured_uri
        import dds_middleware_python as dds

        self._dds = dds
        self._middleware = dds.PyDDSMiddleware(str(self.dds_config["config"]))
        self._middleware.subscribeLowerState(
            str(self.dds_config["state_topic"]), self._state_callback
        )

    def _state_callback(self, state: Any) -> None:
        motors = state.motor_state()
        if self._state_indices is None or self._state_offsets is None:
            raise RuntimeError("state callback received before reader initialization")
        indices = self._state_indices
        offsets = self._state_offsets
        sample = StateSample(
            host_time_ns=time.monotonic_ns(),
            q=np.asarray([motors[hw].q() - offsets[hw] for hw in indices]),
            dq=np.asarray([motors[hw].dq() for hw in indices]),
            motor_temp=np.asarray([motors[hw].motor_temp() for hw in indices]),
            tau_est=np.asarray([motors[hw].tau_est() for hw in indices]),
        )
        with self._lock:
            self._samples.append(sample)
        self._first_state.set()

    def wait_for_first_state(self) -> StateSample:
        timeout = float(self.dds_config["first_state_timeout_s"])
        if not self._first_state.wait(timeout):
            raise TimeoutError(f"no {self.dds_config['state_topic']} sample within {timeout:.1f} s")
        return self.latest()

    def latest(self) -> StateSample:
        with self._lock:
            if not self._samples:
                raise RuntimeError("no Dobot lower-state sample is available")
            return _copy(self._samples[-1])

    def snapshot(self) -> list[StateSample]:
        with self._lock:
            return [_copy(sample) for sample in self._samples]

    def reset_capture_buffer(self) -> int:
        """Discard pre-arm history while retaining one state for fail-closed checks."""
        with self._lock:
            if not self._samples:
                raise RuntimeError("no Dobot lower-state sample is available")
            latest = _copy(self._samples[-1])
            discarded = len(self._samples) - 1
            self._samples = deque((latest,))
        return discarded

    def wait_for_state_after(self, host_time_ns: int) -> StateSample:
        """Wait for a state callback newer than ``host_time_ns`` without creating a writer."""
        deadline = time.monotonic() + float(self.dds_config["first_state_timeout_s"])
        while time.monotonic() < deadline:
            with self._lock:
                if self._samples and self._samples[-1].host_time_ns > host_time_ns:
                    return _copy(self._samples[-1])
            time.sleep(0.001)
        raise TimeoutError(
            f"no fresh lower-state sample within {self.dds_config['first_state_timeout_s']} s"
        )

    def recent(self, window_s: float) -> list[StateSample]:
        cutoff = time.monotonic_ns() - round(window_s * 1.0e9)
        with self._lock:
            samples = [sample for sample in self._samples if sample.host_time_ns >= cutoff]
        return [_copy(sample) for sample in samples]

    def baseline(self, minimum_samples: int = 20) -> np.ndarray:
        deadline = time.monotonic() + float(self.dds_config["first_state_timeout_s"])
        while time.monotonic() < deadline:
            samples = self.snapshot()
            if len(samples) >= minimum_samples:
                return np.median(
                    np.asarray([sample.q for sample in samples[-minimum_samples:]]), axis=0
                )
            time.sleep(0.01)
        raise TimeoutError(
            f"only {len(self.snapshot())} lower-state samples; need {minimum_samples}"
        )

    def enable_writer(self) -> None:
        if self._middleware is None or self._dds is None:
            raise RuntimeError("start the state reader before enabling the command writer")
        if self._writer_enabled:
            return
        self._middleware.createLowerCmdWriter(
            str(self.dds_config["command_topic"]),
            {
                "reliability": "reliable",
                "history_kind": "keep_last",
                "history_depth": 1,
                "durability": "volatile",
            },
        )
        self._writer_enabled = True

    def publish_position(self, target_q: np.ndarray, leg: str, *, damping: bool = False) -> None:
        if not self._writer_enabled:
            raise RuntimeError("refusing to publish: command writer is disabled")
        target = np.asarray(target_q, dtype=np.float64)
        if target.shape != (12,) or not np.isfinite(target).all():
            raise ValueError("target_q must be a finite 12-joint vector")
        controlled_indices = DOBOT_LEG_INDICES[normalize_leg(leg)]
        controlled = np.zeros(12, dtype=bool)
        controlled[list(controlled_indices)] = True
        kp_selected = vector(self.config, "control", "kp", 3)
        kd_selected = vector(self.config, "control", "kd", 3)
        if damping:
            kp_selected = np.zeros(3)
            kd_selected = np.full(3, float(self.config["control"]["damping_kd"]))
        offsets = vector(self.config, "dds", "motor_offset", 16)
        command = self._dds.LowerCmd()
        for index, hw in enumerate(self.dds_config["abs2hw"]):
            motor = command[int(hw)]
            motor.mode(0)
            motor.q(float(target[index] + offsets[int(hw)]))
            motor.dq(0.0)
            motor.kp(float(kp_selected[index % 3]) if controlled[index] else 0.0)
            motor.kd(float(kd_selected[index % 3]) if controlled[index] else 0.0)
            motor.tau(0.0)
        self._middleware.publishLowerCmd(command)

    def publish_damping(self, leg: str) -> None:
        if not self._writer_enabled:
            return
        target = np.zeros(12)
        dt = float(self.config["physics_dt"])
        deadline = time.monotonic()
        for _ in range(max(1, round(float(self.config["control"]["shutdown_damping_s"]) / dt))):
            self.publish_position(target, leg, damping=True)
            deadline += dt
            remaining = deadline - time.monotonic()
            if remaining > 0:
                time.sleep(remaining)


def _path_uri(path: str) -> str:
    """Return the file URI expected by CycloneDDS."""
    from pathlib import Path

    return Path(path).resolve().as_uri()


def safety_reason(
    config: dict[str, Any], sample: StateSample, target_q: np.ndarray, leg: str, now_ns: int
) -> str | None:
    safety = config["safety"]
    age_ms = (now_ns - sample.host_time_ns) / 1.0e6
    values = (sample.q, sample.dq, sample.motor_temp, sample.tau_est, target_q)
    if age_ms > float(safety["state_timeout_ms"]):
        return f"lower-state age {age_ms:.3f} ms exceeds limit"
    if not all(np.isfinite(value).all() for value in values):
        return "non-finite state or target"
    velocity = float(np.max(np.abs(sample.dq)))
    if velocity > float(safety["max_abs_joint_velocity_rad_s"]):
        return f"joint velocity {velocity:.3f} rad/s exceeds limit"
    indices = np.asarray(DOBOT_LEG_INDICES[normalize_leg(leg)])
    errors = np.abs(sample.q[indices] - np.asarray(target_q)[indices])
    worst = int(np.argmax(errors))
    error = float(errors[worst])
    if error > float(safety["max_abs_tracking_error_rad"]):
        return (
            f"{DOBOT_JOINT_ORDER[indices[worst]]}: tracking error {error:.3f} rad "
            f"exceeds {float(safety['max_abs_tracking_error_rad']):.3f} rad"
        )
    temperature = float(np.max(sample.motor_temp))
    if temperature > float(safety["max_motor_temperature_c"]):
        return f"motor temperature {temperature:.1f} C exceeds limit"
    return None


def hold_stability(
    config: dict[str, Any], samples: list[StateSample], target_q: np.ndarray, leg: str
) -> tuple[str | None, dict[str, Any]]:
    leg = normalize_leg(leg)
    if leg == "ALL":
        per_leg = {}
        failure = None
        for name in ("FL", "FR", "RL", "RR"):
            reason, metrics = hold_stability(config, samples, target_q, name)
            per_leg[name] = metrics
            if reason and failure is None:
                failure = f"{name}: {reason}"
        return failure, per_leg
    window_s = float(config["hold"]["stability_window_s"])
    if len(samples) < 2:
        return "fewer than two samples in the hold window", {}
    duration = (samples[-1].host_time_ns - samples[0].host_time_ns) / 1.0e9
    if duration < 0.9 * window_s:
        return f"hold window covers only {duration:.3f} s", {}
    indices = np.asarray(DOBOT_LEG_INDICES[normalize_leg(leg)])
    positions = np.asarray([sample.q for sample in samples])[:, indices]
    edge = max(1, len(positions) // 5)
    span = float(np.max(np.ptp(positions, axis=0)))
    drift = float(
        np.max(np.abs(np.median(positions[-edge:], axis=0) - np.median(positions[:edge], axis=0)))
        / duration
    )
    error = float(
        np.max(np.abs(np.median(positions[-edge:], axis=0) - np.asarray(target_q)[indices]))
    )
    metrics = {
        "window_s": duration,
        "max_position_span_rad": span,
        "max_drift_rate_rad_s": drift,
        "max_final_tracking_error_rad": error,
    }
    limits = config["hold"]
    if span > float(limits["max_position_span_rad"]):
        return f"hold position span {span:.6f} rad exceeds limit", metrics
    if drift > float(limits["max_drift_rate_rad_s"]):
        return f"hold drift {drift:.6f} rad/s exceeds limit", metrics
    if error > float(limits["max_final_tracking_error_rad"]):
        return f"hold tracking error {error:.6f} rad exceeds limit", metrics
    return None, metrics
