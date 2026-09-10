"""Fit and replay share the same trajectory and captured-control contract."""

import json

import numpy as np
import pytest
import torch

from pace_sim2real.dobot import DOBOT_LEG_JOINTS
from pace_sim2real.hardware.data import convert_capture, save_capture
from pace_sim2real.scripts._common import load_pace_trajectory
from pace_sim2real.scripts.dobot import resolve_leg


@pytest.mark.parametrize("leg", ["FL", "ALL"])
def test_shared_trajectory_loader_validates_converted_data(tmp_path, leg):
    capture = save_capture(
        tmp_path / "capture.npz",
        time_s=np.arange(4) * 0.0025,
        dof_pos=np.zeros((4, 12)),
        des_dof_pos=np.ones((4, 12)),
        phases=np.full(4, "chirp"),
        metadata={
            "eligible_for_fit": True,
            "leg": leg,
            "physics_dt": 0.0025,
            "control": {"kp": [25.0] * 3, "kd": [1.3] * 3},
        },
        raw={},
    )
    source, sidecar = convert_capture(capture, tmp_path / "trajectory.pt")
    capture.unlink()  # New conversions must remain portable without their NPZ.
    order = list(DOBOT_LEG_JOINTS[leg])
    data, control = load_pace_trajectory(
        source, joint_order=order, physics_dt=0.0025, device="cpu", require_dobot=True
    )
    assert resolve_leg(None, source) == leg
    assert data["dof_pos"].shape == (4, len(order))
    assert control == {"kp": [25.0] * len(order), "kd": [1.3] * len(order)}
    manifest = json.loads(sidecar.read_text())
    for key, value, message in (
        ("joint_order", list(reversed(order)), "joint_order"),
        ("leg", "RR", "joint_order"),
        ("output_sha256", "wrong", "hash"),
    ):
        sidecar.write_text(json.dumps({**manifest, key: value}))
        with pytest.raises(ValueError, match=message):
            load_pace_trajectory(
                source, joint_order=order, physics_dt=0.0025, device="cpu", require_dobot=True
            )
    sidecar.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="sampling interval"):
        load_pace_trajectory(
            source, joint_order=order, physics_dt=0.005, device="cpu", require_dobot=True
        )
    sidecar.unlink()
    with pytest.raises(ValueError, match="manifest"):
        load_pace_trajectory(
            source, joint_order=order, physics_dt=0.0025, device="cpu", require_dobot=True
        )
    # Generic PACE trajectories retain the tensor-only format without a sidecar.
    generic, control = load_pace_trajectory(
        source, joint_order=order, physics_dt=0.0025, device="cpu"
    )
    assert torch.equal(generic["dof_pos"], data["dof_pos"])
    assert control is None
