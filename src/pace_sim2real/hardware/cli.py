"""Portable, operator-gated Dobot hardware collection commands."""

from __future__ import annotations

import argparse
import gc
import json
import platform
import secrets
import subprocess
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from pace_sim2real.dobot import (
    DOBOT_JOINT_ORDER,
    DOBOT_LEG_INDICES,
    DOBOT_XML_SHA256,
    normalize_leg,
)
from pace_sim2real.hardware.config import load_config, vector
from pace_sim2real.hardware.data import align_samples, convert_capture, save_capture, sha256
from pace_sim2real.hardware.dobot import DobotDDS, hold_stability, safety_reason
from pace_sim2real.hardware.excitation import generate_hold, generate_identification

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = ROOT / "config" / "dobot_hardware.json"
DEFAULT_MANIFEST = ROOT / "runtime" / "MANIFEST.sha256"
DOBOT_XML = ROOT / "src" / "pace_sim2real" / "assets" / "dobot" / "dobot.xml"


@contextmanager
def _suspend_cyclic_gc() -> Iterator[dict[str, bool | int]]:
    """Keep cyclic GC out of the active control window and always restore it."""
    was_enabled = gc.isenabled()
    collected = gc.collect() if was_enabled else 0
    if was_enabled:
        gc.disable()
    state = {
        "was_enabled": was_enabled,
        "collected_before_active": collected,
        "disabled_during_active": not gc.isenabled(),
    }
    try:
        yield state
    finally:
        if was_enabled:
            gc.enable()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--config", type=Path, default=DEFAULT_CONFIG)

    subparsers.add_parser("doctor", parents=[common], help="Read-only installation checks.")
    observe = subparsers.add_parser("observe", parents=[common], help="Read lower state only.")
    observe.add_argument("--duration", type=float, default=2.0)

    for name in ("hold", "collect-chirp"):
        active = subparsers.add_parser(name, parents=[common], help=f"Active {name} command.")
        active.add_argument(
            "--leg",
            type=str.upper,
            choices=tuple(DOBOT_LEG_INDICES),
            required=True,
            help="One leg, or ALL for simultaneous mirrored four-leg motion.",
        )
        active.add_argument("--output", type=Path, default=None)
        active.add_argument("--overwrite", action="store_true")

    convert = subparsers.add_parser("convert", help="Convert an eligible NPZ capture to PACE PT.")
    convert.add_argument("source", type=Path)
    convert.add_argument("--output", type=Path, default=None)
    convert.add_argument("--overwrite", action="store_true")
    return parser


def _verify_manifest(path: Path) -> list[str]:
    failures: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        expected, relative = line.split(maxsplit=1)
        artifact = ROOT / relative
        if not artifact.is_file():
            failures.append(f"missing {relative}")
        elif sha256(artifact) != expected:
            failures.append(f"hash mismatch {relative}")
    return failures


def doctor(config: dict[str, Any]) -> int:
    """Check the host without importing DDS or opening an endpoint."""
    failures: list[str] = []
    if platform.system() != "Linux" or platform.machine() not in {"x86_64", "AMD64"}:
        failures.append("hardware runtime requires Linux x86_64")
    if sys.version_info[:2] != (3, 10):
        failures.append("hardware runtime requires CPython 3.10")
    if not DEFAULT_MANIFEST.is_file():
        failures.append(f"missing manifest {DEFAULT_MANIFEST}")
    else:
        failures.extend(_verify_manifest(DEFAULT_MANIFEST))
    try:
        result = subprocess.run(
            ["ip", "-json", "address", "show"],
            check=True,
            capture_output=True,
            text=True,
        )
        addresses = {
            item.get("local")
            for interface in json.loads(result.stdout)
            for item in interface.get("addr_info", [])
        }
        if "192.168.5.100" not in addresses:
            failures.append("no local interface has required address 192.168.5.100")
    except (FileNotFoundError, subprocess.CalledProcessError, json.JSONDecodeError) as error:
        failures.append(f"cannot inspect network interfaces: {error}")
    report = {
        "status": "PASS" if not failures else "FAIL",
        "config": config["_path"],
        "state_topic": config["dds"]["state_topic"],
        "command_topic": config["dds"]["command_topic"],
        "failures": failures,
    }
    print(json.dumps(report, indent=2))
    return 0 if not failures else 2


def observe(config: dict[str, Any], duration: float) -> int:
    if duration <= 0:
        raise ValueError("--duration must be positive")
    transport = DobotDDS(config)
    transport.start_reader()
    first = transport.wait_for_first_state()
    start = time.monotonic()
    while time.monotonic() - start < duration:
        time.sleep(min(0.05, duration))
    samples = transport.snapshot()
    elapsed = (
        (samples[-1].host_time_ns - samples[0].host_time_ns) / 1.0e9 if len(samples) > 1 else 0.0
    )
    report = {
        "mode": "observe",
        "writer_enabled": transport.writer_enabled,
        "samples": len(samples),
        "observed_rate_hz": (len(samples) - 1) / elapsed if elapsed > 0 else 0.0,
        "first_q": first.q.tolist(),
        "max_abs_dq_rad_s": float(np.max(np.abs(samples[-1].dq))),
        "max_temperature_c": float(np.max(samples[-1].motor_temp)),
    }
    print(json.dumps(report, indent=2))
    return 0


def _default_output(mode: str, leg: str) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return ROOT / "data" / "dobot" / leg.lower() / f"{leg.lower()}_{mode}_{stamp}.npz"


def _trajectory_report(targets: np.ndarray, leg: str, dt: float) -> None:
    indices = DOBOT_LEG_INDICES[leg]
    print(f"Selected leg: {leg}; active logical joints: {indices}")
    for index in indices:
        values = targets[:, index]
        velocity = float(np.max(np.abs(np.diff(values) / dt)))
        print(
            f"  {DOBOT_JOINT_ORDER[index]}: q=[{values.min():.5f}, {values.max():.5f}] rad, max|dq_des|={velocity:.5f} rad/s"
        )


def _confirm_active(leg: str, mode: str) -> None:
    if not sys.stdin.isatty():
        raise PermissionError("active Dobot commands require an interactive terminal")
    token = secrets.token_hex(2).upper()
    phrase = f"ARM {leg} {token}"
    print(f"WARNING: {mode} creates a writer on rt/lower/cmd.")
    if leg == "ALL":
        print("ALL moves FL, FR, RL and RR simultaneously (12 joints).")
        print("Fix the trunk and keep all four legs airborne with their sweep volumes clear.")
        print("Review the mirrored rear-leg pose; vertical reactions are not cancelled.")
    else:
        print(f"Only {leg} is controlled; support the trunk and other legs, sweep volume clear.")
    print("the emergency stop ready, and no competing lower-command writer active.")
    if input(f"Type {phrase} to create the writer: ").strip() != phrase:
        raise PermissionError("confirmation did not match; no writer was created")


def active(config: dict[str, Any], mode: str, leg: str, output: Path, overwrite: bool) -> int:
    leg = normalize_leg(leg)
    if output.exists() and not overwrite:
        raise FileExistsError(f"output exists: {output}")
    transport = DobotDDS(config)
    transport.start_reader()
    first = transport.wait_for_first_state()
    baseline = transport.baseline()
    if mode == "hold":
        time_s, targets, phases = generate_hold(config, baseline, leg)
    else:
        time_s, targets, phases = generate_identification(config, baseline, leg)
    dt = float(config["physics_dt"])
    _trajectory_report(targets, leg, dt)
    if leg == "ALL":
        print("ALL mirrors hold.target_joint_pos[:3] (FL); other hold entries are unused.")
        print("Mirroring applies after approach; all four hold gates must pass before chirp.")
        print(
            f"Approach duration: {np.count_nonzero(phases == 'approach') * dt:.3f} s "
            "(automatically extended if needed to respect the hold speed limit)."
        )
        hold_target = targets[np.flatnonzero(phases == "hold")[-1]].reshape(4, 3)
        for name, pose in zip(("FL", "FR", "RL", "RR"), hold_target, strict=True):
            print(f"  {name} hold [abad, thigh, calf]: {pose.tolist()} rad")
    print(
        f"PD gains [abad, thigh, calf]: Kp={config['control']['kp']}, Kd={config['control']['kd']}"
    )
    if mode == "collect":
        chirp = config["chirp"]
        print(
            f"Chirp [abad, thigh, calf]: amplitude={chirp['amplitude_rad']} rad, "
            f"phase={chirp['phase_deg']} deg, direction={chirp['direction']}"
        )
        print(
            f"Sweep: {chirp['min_frequency_hz']} -> {chirp['max_frequency_hz']} Hz "
            f"over {chirp['duration_s']} s; smooth ramp {chirp['ramp_s']} s."
        )
    print(f"Capture output: {output}")
    print(
        f"State ready: max|dq|={np.max(np.abs(first.dq)):.4f} rad/s, "
        f"max temperature={np.max(first.motor_temp):.1f} C"
    )
    _confirm_active(leg, mode)

    command_times: list[int] = []
    commanded: list[np.ndarray] = []
    logged_phases: list[str] = []
    status = "running"
    hold_checked = False
    hold_metrics: dict[str, Any] | None = None
    active_error: BaseException | None = None
    pre_capture_samples_discarded = transport.reset_capture_buffer()
    pre_gc_state_time_ns = transport.latest().host_time_ns
    with _suspend_cyclic_gc() as gc_state:
        transport.wait_for_state_after(pre_gc_state_time_ns)
        pre_capture_samples_discarded += transport.reset_capture_buffer()
        print(
            "Active capture timing guard: "
            f"discarded {pre_capture_samples_discarded} pre-arm state samples; "
            "cyclic GC suspended."
        )
        transport.enable_writer()
        start = time.monotonic()
        try:
            for index, target in enumerate(targets):
                deadline = start + index * dt
                lag_ms = (time.monotonic() - deadline) * 1.0e3
                if lag_ms > float(config["safety"]["max_schedule_lag_ms"]):
                    raise RuntimeError(f"control schedule lag {lag_ms:.3f} ms exceeds limit")
                if (
                    mode == "collect"
                    and not hold_checked
                    and index > 0
                    and phases[index - 1] == "hold"
                    and phases[index] == "prehold"
                ):
                    reason, hold_metrics = hold_stability(
                        config,
                        transport.recent(float(config["hold"]["stability_window_s"]) + 0.25),
                        targets[index - 1],
                        leg,
                    )
                    if reason:
                        raise RuntimeError(f"hold gate failed: {reason}")
                    hold_checked = True
                    print(f"Hold gate PASS: {json.dumps(hold_metrics, sort_keys=True)}")
                    start = time.monotonic() - index * dt
                sample = transport.latest()
                reason = safety_reason(config, sample, target, leg, time.monotonic_ns())
                if reason:
                    raise RuntimeError(f"hardware safety abort: {reason}")
                transport.publish_position(target, leg)
                command_times.append(time.monotonic_ns())
                commanded.append(target.copy())
                logged_phases.append(str(phases[index]))
                remaining = start + (index + 1) * dt - time.monotonic()
                if remaining > 0:
                    time.sleep(remaining)
            if mode == "hold":
                reason, hold_metrics = hold_stability(
                    config,
                    transport.recent(float(config["hold"]["stability_window_s"]) + 0.25),
                    targets[-1],
                    leg,
                )
                if reason:
                    raise RuntimeError(f"hold gate failed: {reason}")
                hold_checked = True
            status = "completed"
        except BaseException as error:
            status = (
                "operator_stop"
                if isinstance(error, KeyboardInterrupt)
                else f"aborted:{type(error).__name__}"
            )
            active_error = error
        finally:
            print(f"Sending bounded damping shutdown to {leg}.")
            transport.publish_damping(leg)

    command_time = np.asarray(command_times, dtype=np.int64)
    if len(command_time) < 2:
        assert active_error is not None
        raise active_error
    command_q = np.asarray(commanded)
    samples = transport.snapshot()
    _, measured, desired, mask, gap_ms = align_samples(samples, command_time, command_q)
    intervals = np.diff(command_time) / 1.0e9
    rate_hz = float(1.0 / np.mean(intervals))
    expected_rate = 1.0 / dt
    rate_valid = abs(rate_hz - expected_rate) / expected_rate <= float(
        config["safety"]["command_rate_tolerance_fraction"]
    )
    gap_valid = gap_ms <= float(config["safety"]["max_state_interpolation_gap_ms"])
    if mode == "collect" and status == "completed" and not rate_valid:
        status = f"invalid_command_rate:{rate_hz:.3f}Hz"
    elif mode == "collect" and status == "completed" and not gap_valid:
        status = f"invalid_interpolation_gap:{gap_ms:.3f}ms"
    eligible = (
        mode == "collect" and status == "completed" and hold_checked and rate_valid and gap_valid
    )
    metadata = {
        "schema": "pace_dobot_capture_metadata_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "mode": mode,
        "status": status,
        "eligible_for_fit": eligible,
        "leg": leg,
        "selected_indices": list(DOBOT_LEG_INDICES[leg]),
        "control": {key: vector(config, "control", key, 3).tolist() for key in ("kp", "kd")},
        "trajectory_mode": "mirrored" if leg == "ALL" else "single_leg",
        "chirp": dict(config["chirp"]) if mode == "collect" else None,
        "error": str(active_error) if active_error is not None else None,
        "physics_dt": dt,
        "observed_command_rate_hz": rate_hz,
        "max_state_interpolation_gap_ms": gap_ms,
        "hold_stability": hold_metrics,
        "pre_capture_samples_discarded": pre_capture_samples_discarded,
        "gc_was_enabled_before_active": gc_state["was_enabled"],
        "gc_collected_before_active": gc_state["collected_before_active"],
        "gc_disabled_during_active_capture": gc_state["disabled_during_active"],
        "config": config["_path"],
        "config_sha256": sha256(config["_path"]),
        "model": str(DOBOT_XML),
        "model_sha256": sha256(DOBOT_XML),
    }
    saved = save_capture(
        output,
        time_s=np.arange(len(measured)) * dt,
        dof_pos=measured,
        des_dof_pos=desired,
        phases=np.asarray(logged_phases)[mask],
        metadata=metadata,
        raw={
            "command_host_time_ns": command_time,
            "state_host_time_ns": np.asarray([sample.host_time_ns for sample in samples]),
            "state_q": np.asarray([sample.q for sample in samples]),
            "state_dq": np.asarray([sample.dq for sample in samples]),
            "state_motor_temp": np.asarray([sample.motor_temp for sample in samples]),
            "state_tau_est": np.asarray([sample.tau_est for sample in samples]),
        },
        overwrite=overwrite,
    )
    print(
        json.dumps(
            {"status": status, "eligible_for_fit": eligible, "capture": str(saved)}, indent=2
        )
    )
    if active_error is not None:
        raise active_error
    return 0 if status == "completed" else 2


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "convert":
        output = args.output or args.source.with_suffix(".pt")
        converted, sidecar = convert_capture(args.source, output, overwrite=args.overwrite)
        print(json.dumps({"data": str(converted), "manifest": str(sidecar)}, indent=2))
        return 0
    config = load_config(args.config)
    if sha256(DOBOT_XML) != DOBOT_XML_SHA256:
        raise RuntimeError(f"Dobot XML hash mismatch: {DOBOT_XML}")
    if args.command == "doctor":
        return doctor(config)
    if args.command == "observe":
        return observe(config, args.duration)
    mode = "collect" if args.command == "collect-chirp" else args.command
    output = (args.output or _default_output(mode, args.leg)).expanduser().resolve()
    return active(config, mode, args.leg, output, args.overwrite)
