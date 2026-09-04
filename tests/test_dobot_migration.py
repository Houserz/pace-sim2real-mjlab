from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from pace_sim2real.assets.dobot_asset import DOBOT_XML_SHA256, get_dobot_robot_cfg, get_spec
from pace_sim2real.dobot import (
    DOBOT_DEFAULT_JOINT_POS,
    DOBOT_LEG_INDICES,
    DOBOT_LEG_JOINTS,
    task_id_for_leg,
)
from pace_sim2real.hardware.cli import build_parser as build_hardware_parser
from pace_sim2real.hardware.config import load_config
from pace_sim2real.hardware.data import convert_capture, load_capture, save_capture, sha256
from pace_sim2real.hardware.dobot import DobotDDS
from pace_sim2real.hardware.excitation import generate_identification
from pace_sim2real.scripts.dobot import resolve_leg
from pace_sim2real.tasks.manager_based.pace.dobot_pace_env_cfg import (
    dobot_bounds,
    dobot_pace_env_cfg,
)
from pace_sim2real.utils import load_pace_artifact

ROOT = Path(__file__).resolve().parents[1]


def test_dobot_asset_is_fixed_contact_free_and_audited() -> None:
    assert sha256(ROOT / "src/pace_sim2real/assets/dobot/dobot.xml") == DOBOT_XML_SHA256
    model = get_spec().compile()
    assert (model.nq, model.nv) == (12, 12)
    assert np.all(model.geom_contype == 0)
    assert np.all(model.geom_conaffinity == 0)


def test_four_single_leg_tasks_share_the_13_parameter_contract() -> None:
    for leg, joint_order in DOBOT_LEG_JOINTS.items():
        cfg = dobot_pace_env_cfg(leg, num_envs=2)
        assert cfg.sim2real.joint_order == list(joint_order)
        assert cfg.sim2real.bounds_params.shape == (13, 2)
        assert cfg.scene.num_envs == 2
        actuator = cfg.scene.robot.articulation.actuators[0]
        assert actuator.joint_names_expr == joint_order
    assert dobot_bounds(6).shape == (25, 2)


def test_identification_trajectory_moves_only_the_selected_leg() -> None:
    config = load_config(ROOT / "config/dobot_hardware.json")
    baseline = np.asarray(DOBOT_DEFAULT_JOINT_POS)
    for leg, indices in DOBOT_LEG_INDICES.items():
        _, targets, phases = generate_identification(config, baseline, leg)
        selected = set(indices)
        for joint in range(12):
            if joint not in selected:
                np.testing.assert_allclose(targets[:, joint], baseline[joint])
        assert np.all(np.ptp(targets[:, list(indices)], axis=0) > 0)
        assert "hold" in phases and "chirp" in phases


def test_capture_conversion_selects_leg_and_records_hashes(tmp_path: Path) -> None:
    count = 8
    values = np.arange(count * 12, dtype=np.float64).reshape(count, 12) / 100.0
    capture = save_capture(
        tmp_path / "capture.npz",
        time_s=np.arange(count) * 0.0025,
        dof_pos=values,
        des_dof_pos=values + 0.01,
        phases=np.asarray(
            ["hold", "prehold", "chirp", "chirp", "posthold", "posthold", "hold", "hold"]
        ),
        metadata={
            "eligible_for_fit": True,
            "status": "completed",
            "leg": "FR",
            "physics_dt": 0.0025,
        },
        raw={},
    )
    output, sidecar = convert_capture(capture, tmp_path / "chirp.pt")
    data = load_pace_artifact(output)
    assert data["dof_pos"].shape == (5, 3)
    assert torch.equal(data["dof_pos"][0], torch.tensor(values[1, 3:6], dtype=torch.float32))
    manifest = json.loads(sidecar.read_text())
    assert manifest["joint_order"] == list(DOBOT_LEG_JOINTS["FR"])
    assert manifest["source_sha256"] == sha256(capture)
    assert load_capture(capture)["metadata"]["leg"] == "FR"


def test_writer_is_created_only_explicitly_and_nonselected_legs_are_passive() -> None:
    config = load_config(ROOT / "config/dobot_hardware.json")

    class Motor:
        def __init__(self) -> None:
            self.values = {}

        def __getattr__(self, name):
            return lambda value: self.values.__setitem__(name, value)

    class LowerCmd:
        def __init__(self) -> None:
            self.motors = [Motor() for _ in range(16)]

        def __getitem__(self, index):
            return self.motors[index]

    class FakeDDS:
        pass

    FakeDDS.LowerCmd = LowerCmd

    class Middleware:
        def __init__(self) -> None:
            self.writer_calls = 0
            self.command = None

        def createLowerCmdWriter(self, *_args) -> None:
            self.writer_calls += 1

        def publishLowerCmd(self, command) -> None:
            self.command = command

    transport = DobotDDS(config)
    middleware = Middleware()
    transport._dds = FakeDDS
    transport._middleware = middleware
    assert not transport.writer_enabled and middleware.writer_calls == 0
    transport.enable_writer()
    transport.publish_position(np.asarray(DOBOT_DEFAULT_JOINT_POS), "FL")
    assert middleware.writer_calls == 1
    assert middleware.command[0].values["kp"] == 10.0
    assert middleware.command[4].values["kp"] == 0.0
    assert middleware.command[4].values["kd"] == 0.0
    assert middleware.command[4].values["tau"] == 0.0


def test_robot_cfg_contains_only_selected_leg_actuator() -> None:
    cfg = get_dobot_robot_cfg("RL")
    assert len(cfg.articulation.actuators) == 1
    assert cfg.articulation.actuators[0].joint_names_expr == DOBOT_LEG_JOINTS["RL"]


def test_workflow_infers_one_leg_and_derives_its_task(tmp_path: Path) -> None:
    data = tmp_path / "chirp.pt"
    data.with_suffix(".pt.json").write_text(json.dumps({"leg": "rr"}))
    assert resolve_leg(None, data) == "RR"
    assert resolve_leg("RR", data) == "RR"
    assert task_id_for_leg("rr") == "Dobot-Pace-RR-v0"
    with pytest.raises(ValueError, match="disagrees"):
        resolve_leg("FL", data)


def test_hardware_commands_require_leg_only_when_active() -> None:
    parser = build_hardware_parser()
    assert parser.parse_args(["doctor"]).command == "doctor"
    with pytest.raises(SystemExit):
        parser.parse_args(["doctor", "--leg", "FL"])
    with pytest.raises(SystemExit):
        parser.parse_args(["collect-chirp"])
    args = parser.parse_args(["collect-chirp", "--leg", "FR"])
    assert args.leg == "FR"
