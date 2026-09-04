"""Replay saved PACE parameters on an independent trajectory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from pace_sim2real.utils import load_pace_artifact, require_tensor

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
    parser.add_argument("--device", default=None)
    parser.add_argument("--output", type=Path, default=None)
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


def run(args: argparse.Namespace) -> dict[str, object]:
    device = resolve_device(args.device)
    env = make_env(args.task, 1, device, task_module=args.task_module)
    try:
        robot = env.scene["robot"]
        joint_order = list(env.cfg.sim2real.joint_order)
        joint_ids = pace_joint_ids(robot, joint_order, device)
        data = load_pace_artifact(args.data, map_location=device)
        _, measured, target = validate_pace_trajectory_data(
            data, physics_dt=env.physics_dt, joint_count=len(joint_order)
        )
        measured = measured.to(device)
        target = target.to(device)
        fitted = _load_parameters(args.parameters, joint_order, device)
        report: dict[str, object] = {
            "task": args.task,
            "joint_order": joint_order,
            "samples": len(measured),
            "fitted": _metrics(
                _replay(env, robot, joint_ids, target, measured[0], fitted), measured
            ),
        }
        if args.reference_parameters is not None:
            reference = _load_parameters(args.reference_parameters, joint_order, device)
            report["reference"] = _metrics(
                _replay(env, robot, joint_ids, target, measured[0], reference), measured
            )
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
