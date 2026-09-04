"""Raw Dobot capture alignment and conversion to the PACE tensor contract."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from pace_sim2real.dobot import (
    DOBOT_JOINT_ORDER,
    DOBOT_LEG_INDICES,
    DOBOT_LEG_JOINTS,
    normalize_leg,
)
from pace_sim2real.hardware.dobot import StateSample

CAPTURE_SCHEMA = "pace_dobot_capture_v1"


def align_samples(
    samples: list[StateSample], command_time_ns: np.ndarray, command_q: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    if len(samples) < 2:
        raise ValueError("at least two hardware samples are required")
    state_time = np.asarray([sample.host_time_ns for sample in samples], dtype=np.int64)
    state_q = np.asarray([sample.q for sample in samples], dtype=np.float64)
    if np.any(np.diff(state_time) <= 0):
        raise ValueError("hardware timestamps must be strictly increasing")
    mask = (command_time_ns >= state_time[0]) & (command_time_ns <= state_time[-1])
    if np.count_nonzero(mask) < 2:
        raise ValueError("fewer than two commands lie inside the state interval")
    selected_time = command_time_ns[mask]
    right = np.clip(np.searchsorted(state_time, selected_time, side="left"), 1, len(state_time) - 1)
    left = right - 1
    gap_ms = float(np.max(state_time[right] - state_time[left]) / 1.0e6)
    measured = np.column_stack(
        [np.interp(selected_time, state_time, state_q[:, joint]) for joint in range(12)]
    )
    return selected_time, measured, command_q[mask], mask, gap_ms


def save_capture(
    path: str | Path,
    *,
    time_s: np.ndarray,
    dof_pos: np.ndarray,
    des_dof_pos: np.ndarray,
    phases: np.ndarray,
    metadata: dict[str, Any],
    raw: dict[str, np.ndarray],
    overwrite: bool = False,
) -> Path:
    output = Path(path).expanduser().resolve()
    if output.exists() and not overwrite:
        raise FileExistsError(f"output exists: {output}")
    time_s = np.asarray(time_s, dtype=np.float64)
    measured = np.asarray(dof_pos, dtype=np.float64)
    desired = np.asarray(des_dof_pos, dtype=np.float64)
    phases = np.asarray(phases)
    if time_s.ndim != 1 or len(time_s) < 2 or np.any(np.diff(time_s) <= 0):
        raise ValueError("time must be a strictly increasing vector")
    if measured.shape != (len(time_s), 12) or desired.shape != measured.shape:
        raise ValueError("dof_pos and des_dof_pos must have shape [samples, 12]")
    if (
        phases.shape != time_s.shape
        or not np.isfinite(measured).all()
        or not np.isfinite(desired).all()
    ):
        raise ValueError("capture phases or position arrays are invalid")
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        schema=np.asarray(CAPTURE_SCHEMA),
        time=time_s,
        dof_pos=measured,
        des_dof_pos=desired,
        phase=phases,
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
        **raw,
    )
    return output


def load_capture(path: str | Path, *, require_eligible: bool = True) -> dict[str, Any]:
    source = Path(path).expanduser().resolve()
    with np.load(source, allow_pickle=False) as archive:
        result = {key: archive[key] for key in archive.files}
    if str(result.get("schema")) != CAPTURE_SCHEMA:
        raise ValueError(f"unsupported Dobot capture schema: {source}")
    metadata = json.loads(str(result["metadata_json"]))
    if require_eligible and not bool(metadata.get("eligible_for_fit", False)):
        raise ValueError(f"capture is not eligible for fitting: {metadata.get('status')}")
    result["metadata"] = metadata
    result["path"] = source
    return result


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def convert_capture(
    source: str | Path, output: str | Path, *, overwrite: bool = False
) -> tuple[Path, Path]:
    """Convert one eligible capture to selected-joint, tensor-only PACE data."""
    capture = load_capture(source)
    leg = normalize_leg(str(capture["metadata"]["leg"]))
    indices = np.asarray(DOBOT_LEG_INDICES[leg])
    phases = np.asarray(capture["phase"]).astype(str)
    mask = np.isin(phases, ("prehold", "chirp", "posthold"))
    if not np.any(phases[mask] == "chirp"):
        raise ValueError("capture has no eligible chirp phase")
    destination = Path(output).expanduser().resolve()
    sidecar = destination.with_suffix(destination.suffix + ".json")
    if not overwrite and (destination.exists() or sidecar.exists()):
        raise FileExistsError(f"output exists: {destination} or {sidecar}")
    import torch

    dt = float(capture["metadata"]["physics_dt"])
    sample_count = int(np.count_nonzero(mask))
    payload = {
        "time": torch.arange(sample_count, dtype=torch.float32) * dt,
        "dof_pos": torch.as_tensor(capture["dof_pos"][mask][:, indices], dtype=torch.float32),
        "des_dof_pos": torch.as_tensor(
            capture["des_dof_pos"][mask][:, indices], dtype=torch.float32
        ),
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, destination)
    manifest = {
        "schema": "pace_dobot_conversion_v1",
        "source": str(capture["path"]),
        "source_sha256": sha256(capture["path"]),
        "output": str(destination),
        "output_sha256": sha256(destination),
        "leg": leg,
        "joint_order": list(DOBOT_LEG_JOINTS[leg]),
        "source_joint_order": list(DOBOT_JOINT_ORDER),
        "samples": sample_count,
        "physics_dt": dt,
        "included_phases": ["prehold", "chirp", "posthold"],
    }
    sidecar.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return destination, sidecar
