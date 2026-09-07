"""Replay saved PACE parameters on an independent trajectory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from pace_sim2real.hardware.config import load_config
from pace_sim2real.hardware.data import load_capture, sha256
from pace_sim2real.utils import PaceDCMotor, load_pace_artifact, require_tensor

from ._common import (
    make_env,
    pace_joint_ids,
    pace_position_action,
    prepare_pace_model,
    resolve_device,
    validate_pace_trajectory_data,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", type=Path, help="Held-out PACE .pt data.")
    parser.add_argument("parameters", type=Path, help="Saved 4N+1 parameter tensor.")
    parser.add_argument("--task", default="Isaac-Pace-Anymal-D-v0")
    parser.add_argument("--task-module", "--task_module", dest="task_module", default=None)
    parser.add_argument("--reference-parameters", type=Path, default=None)
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Dobot hardware config used for the held-out capture and simulated PD gains.",
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--plot", type=Path, default=None)
    return parser


def _load_parameters(path: Path, joint_order: list[str], device: str) -> torch.Tensor:
    value = load_pace_artifact(path, map_location=device)
    if isinstance(value, dict):
        saved_joint_order = value.get("joint_order")
        if saved_joint_order is not None and saved_joint_order != joint_order:
            raise ValueError(
                f"parameter joint_order {saved_joint_order} does not match task {joint_order}"
            )
        value = value.get("params")
    joint_count = len(joint_order)
    params = (
        require_tensor(value, name="parameters").to(device=device, dtype=torch.float32).reshape(-1)
    )
    if params.shape != (4 * joint_count + 1,) or not torch.isfinite(params).all():
        raise ValueError(f"parameters must be a finite {4 * joint_count + 1}-vector")
    return params


def _replay(
    env,
    robot,
    joint_ids,
    target: torch.Tensor,
    initial_encoder_position: torch.Tensor,
    params: torch.Tensor,
) -> torch.Tensor:
    count = len(joint_ids)
    env.reset()
    prepare_pace_model(
        env,
        robot,
        joint_ids,
        armature=params[:count].reshape(1, count),
        damping=params[count : 2 * count].reshape(1, count),
        friction=params[2 * count : 3 * count].reshape(1, count),
        bias=params[3 * count : 4 * count].reshape(1, count),
        delay=params[-1:].reshape(1, 1),
        initial_encoder_position=initial_encoder_position.reshape(1, count),
    )
    positions = torch.empty_like(target)
    for step in range(len(target)):
        positions[step] = robot.data.joint_pos[0, joint_ids]
        action = pace_position_action(env, robot, joint_ids, target[step].reshape(1, count))
        env.step(action)
    return positions - params[3 * count : 4 * count]


def _metrics(simulated: torch.Tensor, measured: torch.Tensor) -> dict[str, object]:
    error = simulated - measured
    return {
        "rmse": float(torch.sqrt(torch.mean(torch.square(error))).item()),
        "mae": float(torch.mean(torch.abs(error)).item()),
        "joint_rmse": torch.sqrt(torch.mean(torch.square(error), dim=0)).cpu().tolist(),
        "joint_mae": torch.mean(torch.abs(error), dim=0).cpu().tolist(),
    }


def _load_hardware_config(data_path: Path, config_path: Path) -> dict[str, object]:
    """Load the exact Dobot config recorded by conversion of this capture."""
    sidecar = data_path.expanduser().resolve().with_suffix(data_path.suffix + ".json")
    if not sidecar.is_file():
        raise ValueError(f"--config requires the conversion manifest: {sidecar}")
    manifest = json.loads(sidecar.read_text(encoding="utf-8"))
    source = Path(str(manifest.get("source", ""))).expanduser()
    if not source.is_file():
        raise ValueError(f"converted data source capture is unavailable: {source}")
    if manifest.get("source_sha256") != sha256(source):
        raise ValueError(f"source capture hash does not match conversion manifest: {source}")
    expected = load_capture(source)["metadata"].get("config_sha256")
    config = load_config(config_path)
    actual = sha256(config["_path"])
    if expected != actual:
        raise ValueError(
            f"hardware config does not match the held-out capture ({actual} != {expected})"
        )
    return config


def _apply_pd_gains(
    robot, joint_ids: torch.Tensor, config: dict[str, object]
) -> dict[str, list[float]]:
    """Apply the held-out hardware PD gains to the matching PACE actuator."""
    actuator = next(
        (
            item
            for item in robot.actuators
            if isinstance(item, PaceDCMotor) and torch.equal(item.target_ids, joint_ids)
        ),
        None,
    )
    if actuator is None:
        raise ValueError("--config requires one PaceDCMotor matching the fitted joints")
    control = config["control"]
    result: dict[str, list[float]] = {}
    for name, key in (("stiffness", "kp"), ("damping", "kd")):
        field = getattr(actuator, name)
        values = torch.as_tensor(control[key], dtype=field.dtype, device=field.device)
        if values.shape != (len(joint_ids),):
            raise ValueError(f"control.{key} must contain one value per fitted joint")
        field.copy_(values.expand_as(field))
        getattr(actuator, f"default_{name}").copy_(field)
        result[key] = values.cpu().tolist()
    return result


def _plot_comparison(
    path: Path,
    time: torch.Tensor,
    joint_order: list[str],
    target: torch.Tensor,
    measured: torch.Tensor,
    simulated: torch.Tensor,
) -> None:
    import matplotlib.pyplot as plt

    time_values = time.cpu().numpy()
    fig, axes = plt.subplots(len(joint_order), 1, figsize=(11, 3 * len(joint_order)), sharex=True)
    axes = [axes] if len(joint_order) == 1 else axes
    for index, (axis, joint_name) in enumerate(zip(axes, joint_order, strict=True)):
        axis.plot(time_values, target[:, index].cpu().numpy(), label="target", linewidth=1.0)
        axis.plot(time_values, measured[:, index].cpu().numpy(), label="real", linewidth=1.0)
        axis.plot(time_values, simulated[:, index].cpu().numpy(), label="sim", linewidth=1.0)
        axis.set_ylabel("position [rad]")
        axis.set_title(joint_name)
        axis.grid(True, alpha=0.3)
        axis.legend()
    axes[-1].set_xlabel("time [s]")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def run(args: argparse.Namespace) -> dict[str, object]:
    hardware_config = (
        _load_hardware_config(args.data, args.config) if args.config is not None else None
    )
    device = resolve_device(args.device)
    env = make_env(args.task, 1, device, task_module=args.task_module)
    try:
        robot = env.scene["robot"]
        joint_order = list(env.cfg.sim2real.joint_order)
        joint_ids = pace_joint_ids(robot, joint_order, device)
        data = load_pace_artifact(args.data, map_location=device)
        time, measured, target = validate_pace_trajectory_data(
            data, physics_dt=env.physics_dt, joint_count=len(joint_order)
        )
        time = time.to(device)
        measured = measured.to(device)
        target = target.to(device)
        control = (
            _apply_pd_gains(robot, joint_ids, hardware_config)
            if hardware_config is not None
            else None
        )
        fitted = _load_parameters(args.parameters, joint_order, device)
        simulated = _replay(env, robot, joint_ids, target, measured[0], fitted)
        report: dict[str, object] = {
            "task": args.task,
            "joint_order": joint_order,
            "samples": len(measured),
            "fitted": _metrics(simulated, measured),
        }
        if control is not None:
            report["control"] = control
            report["config_sha256"] = sha256(hardware_config["_path"])
        if args.reference_parameters is not None:
            reference = _load_parameters(args.reference_parameters, joint_order, device)
            report["reference"] = _metrics(
                _replay(env, robot, joint_ids, target, measured[0], reference), measured
            )
        if args.plot is not None:
            _plot_comparison(args.plot, time, joint_order, target, measured, simulated)
            report["plot"] = str(args.plot.expanduser().resolve())
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(report, indent=2))
        return report
    finally:
        env.close()


def main(argv: list[str] | None = None) -> None:
    run(build_parser().parse_args(argv))


if __name__ == "__main__":
    main()
