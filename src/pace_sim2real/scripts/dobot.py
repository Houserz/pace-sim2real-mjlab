"""One entry point for the Dobot hardware, fitting, and evaluation workflow."""

from __future__ import annotations

import os
import sys
from pathlib import Path

from pace_sim2real.dobot import normalize_leg, task_id_for_leg

ROOT = Path(__file__).resolve().parents[3]
HARDWARE_ENV = ROOT / ".venv-hardware"
HARDWARE_COMMANDS = {"doctor", "observe", "hold", "collect-chirp"}

USAGE = """usage: pace-dobot COMMAND [ARGS]

Commands:
  doctor                         Offline host and runtime checks
  observe [--duration S]         Read lower state; never creates a writer
  hold --leg LEG                 Active hold-only trial; LEG = FL/FR/RL/RR/ALL
  collect-chirp --leg LEG        Active hold, gate, and chirp; ALL mirrors four legs
  convert SOURCE [--output PT]   Convert a raw capture to PACE tensors
  fit [--leg LEG] [--data PT] [--num_envs N] [--device DEV]
  evaluate [--leg LEG] DATA PARAMS [--output JSON] [--device DEV]

For fit/evaluate, --leg is inferred from DATA.pt.json when present. If both are
supplied, they must agree. Active commands always require an explicit --leg.
ALL mirrors the FL hold pose and 3-joint chirp across four airborne legs.
New captures carry their PD gains. Old captures require --config for fit/evaluate.
"""


def _extract_leg(args: list[str]) -> tuple[str | None, list[str]]:
    leg: str | None = None
    remaining: list[str] = []
    index = 0
    while index < len(args):
        arg = args[index]
        if arg == "--leg":
            if index + 1 >= len(args):
                raise SystemExit("--leg requires FL, FR, RL, RR, or ALL")
            value = args[index + 1]
            index += 2
        elif arg.startswith("--leg="):
            value = arg.split("=", 1)[1]
            index += 1
        else:
            remaining.append(arg)
            index += 1
            continue
        try:
            normalized = normalize_leg(value)
        except ValueError as error:
            raise SystemExit(str(error)) from error
        if leg is not None and leg != normalized:
            raise SystemExit("choose one selection: FL, FR, RL, RR, or ALL")
        leg = normalized
    return leg, remaining


def resolve_leg(selected: str | None, data: Path | None) -> str:
    """Keep CLI imports light while sharing the capture's leg validation."""
    from pace_sim2real.hardware.data import resolve_leg as resolve_capture_leg

    return resolve_capture_leg(selected, data)


def _hardware_python() -> Path:
    python = HARDWARE_ENV / "bin" / "python"
    if not python.is_file():
        raise SystemExit(
            f"hardware environment is missing: {python}\n"
            "Follow docs/examples/dobot.md once, then rerun this command."
        )
    return python


def _run_hardware(command: str, args: list[str]) -> int:
    from pace_sim2real.hardware.cli import main as hardware_main

    return hardware_main([command, *args])


def _dispatch_hardware(command: str, args: list[str]) -> int:
    if "-h" in args or "--help" in args:
        return _run_hardware(command, args)
    leg, remaining = _extract_leg(args)
    if command in {"hold", "collect-chirp"}:
        if leg is None:
            raise SystemExit(f"{command} requires an explicit --leg")
        remaining.extend(("--leg", leg))
    elif leg is not None:
        raise SystemExit(f"{command} does not accept --leg")

    if Path(sys.prefix).absolute() == HARDWARE_ENV.absolute():
        return _run_hardware(command, remaining)
    python = _hardware_python()
    os.execv(
        python,
        [str(python), "-m", "pace_sim2real.hardware.cli", command, *remaining],
    )
    return 0


def _dispatch_fit(args: list[str]) -> int:
    leg, remaining = _extract_leg(args)
    if "--task" in remaining or any(arg.startswith("--task=") for arg in remaining):
        raise SystemExit("pace-dobot derives --task from --leg; use pace-fit for non-Dobot tasks")
    from pace_sim2real.scripts.fit import build_parser, main

    parsed = build_parser().parse_args(remaining)
    try:
        resolved = resolve_leg(leg, Path(parsed.data) if parsed.data else None)
    except ValueError as error:
        raise SystemExit(str(error)) from error
    main([*remaining, "--task", task_id_for_leg(resolved)])
    return 0


def _dispatch_evaluate(args: list[str]) -> int:
    leg, remaining = _extract_leg(args)
    if "--task" in remaining or any(arg.startswith("--task=") for arg in remaining):
        raise SystemExit("pace-dobot derives --task from --leg; use pace-evaluate for other tasks")
    from pace_sim2real.scripts.evaluate import build_parser, main

    parsed = build_parser().parse_args(remaining)
    try:
        resolved = resolve_leg(leg, parsed.data)
    except ValueError as error:
        raise SystemExit(str(error)) from error
    main([*remaining, "--task", task_id_for_leg(resolved)])
    return 0


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] in {"-h", "--help"}:
        print(USAGE)
        return 0
    command, remaining = args[0], args[1:]
    if command in HARDWARE_COMMANDS:
        return _dispatch_hardware(command, remaining)
    if command == "convert":
        leg, forwarded = _extract_leg(remaining)
        if leg is not None:
            raise SystemExit("convert reads the leg from its raw capture and does not accept --leg")
        return _run_hardware(command, forwarded)
    if command == "fit":
        if "-h" in remaining or "--help" in remaining:
            print(USAGE)
            return 0
        return _dispatch_fit(remaining)
    if command == "evaluate":
        if "-h" in remaining or "--help" in remaining:
            print(USAGE)
            return 0
        return _dispatch_evaluate(remaining)
    raise SystemExit(f"unknown Dobot command {command!r}\n\n{USAGE}")


if __name__ == "__main__":
    raise SystemExit(main())
