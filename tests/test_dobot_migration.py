from __future__ import annotations

import gc
import json
import threading
import time
from collections import deque
from pathlib import Path

import mujoco
import numpy as np
import pytest
import torch

from pace_sim2real.assets.dobot_asset import DOBOT_XML_SHA256, get_dobot_robot_cfg, get_spec
from pace_sim2real.dobot import (
    DOBOT_DEFAULT_JOINT_POS,
    DOBOT_LEG_INDICES,
    DOBOT_LEG_JOINTS,
    DOBOT_MIRROR_SIGNS,
    task_id_for_leg,
)
from pace_sim2real.hardware.cli import (
    _suspend_cyclic_gc,
)
from pace_sim2real.hardware.cli import (
    build_parser as build_hardware_parser,
)
from pace_sim2real.hardware.config import load_config
from pace_sim2real.hardware.data import (
    convert_capture,
    load_capture,
    load_dobot_control,
    load_hardware_config,
    save_capture,
    sha256,
)
from pace_sim2real.hardware.dobot import DobotDDS, StateSample, hold_stability, safety_reason
from pace_sim2real.hardware.excitation import generate_chirp, generate_identification
from pace_sim2real.scripts._common import apply_pd_gains
from pace_sim2real.scripts.dobot import resolve_leg
from pace_sim2real.tasks.manager_based.pace.dobot_pace_env_cfg import (
    dobot_bounds,
    dobot_pace_env_cfg,
)
from pace_sim2real.utils import PaceDCMotor, load_pace_artifact

ROOT = Path(__file__).resolve().parents[1]


def test_dobot_asset_is_fixed_contact_free_and_audited() -> None:
    assert sha256(ROOT / "src/pace_sim2real/assets/dobot/dobot.xml") == DOBOT_XML_SHA256
    model = get_spec().compile()
    assert (model.nq, model.nv) == (12, 12)
    assert np.all(model.geom_contype == 0)
    assert np.all(model.geom_conaffinity == 0)


def test_dobot_tasks_use_the_4n_plus_1_parameter_contract() -> None:
    for leg, joint_order in DOBOT_LEG_JOINTS.items():
        cfg = dobot_pace_env_cfg(leg, num_envs=2)
        assert cfg.sim2real.joint_order == list(joint_order)
        assert cfg.sim2real.bounds_params.shape == (4 * len(joint_order) + 1, 2)
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
            "config_sha256": sha256(ROOT / "config/dobot_hardware.json"),
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
    loaded = load_hardware_config(output, ROOT / "config/dobot_hardware.json")
    assert (
        loaded["control"]["kp"] == load_config(ROOT / "config/dobot_hardware.json")["control"]["kp"]
    )

    mismatched = json.loads((ROOT / "config/dobot_hardware.json").read_text())
    for key in ("config", "cyclonedds_uri"):
        mismatched["dds"][key] = str((ROOT / "config" / mismatched["dds"][key]).resolve())
    mismatched["control"]["kp"][0] += 1.0
    mismatched_path = tmp_path / "mismatched.json"
    mismatched_path.write_text(json.dumps(mismatched))
    with pytest.raises(ValueError, match="does not match the capture"):
        load_hardware_config(output, mismatched_path)
    order = list(DOBOT_LEG_JOINTS["FR"])
    with pytest.raises(ValueError, match="pass --config"):
        load_dobot_control(output, order, required=True)
    assert load_dobot_control(output, order, ROOT / "config/dobot_hardware.json") == {
        key: loaded["control"][key] for key in ("kp", "kd")
    }


def test_hardware_pd_gains_are_applied_to_the_pace_actuator() -> None:
    actuator = object.__new__(PaceDCMotor)
    actuator._target_ids = torch.tensor([1, 2, 3])
    actuator.stiffness = torch.zeros((2, 3))
    actuator.damping = torch.zeros((2, 3))
    actuator.default_stiffness = torch.zeros((2, 3))
    actuator.default_damping = torch.zeros((2, 3))
    robot = type("Robot", (), {"actuators": [actuator]})()
    config = {"control": {"kp": [10.0, 20.0, 30.0], "kd": [1.0, 2.0, 3.0]}}
    gains = apply_pd_gains(robot, torch.tensor([1, 2, 3]), config["control"])
    assert gains == {"kp": [10.0, 20.0, 30.0], "kd": [1.0, 2.0, 3.0]}
    assert actuator.stiffness.tolist() == [[10.0, 20.0, 30.0]] * 2
    assert actuator.default_damping.tolist() == [[1.0, 2.0, 3.0]] * 2


@pytest.mark.parametrize("leg", ["FL", "FR", "RL", "RR", "ALL"])
def test_writer_is_created_only_explicitly_and_nonselected_legs_are_passive(leg) -> None:
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
    target = np.asarray(DOBOT_DEFAULT_JOINT_POS)
    with pytest.raises(RuntimeError, match="disabled"):
        transport.publish_position(target, leg)
    transport.enable_writer()
    transport.enable_writer()
    transport.publish_position(target, leg)
    assert middleware.writer_calls == 1
    for index, hw in enumerate(config["dds"]["abs2hw"]):
        values = middleware.command[hw].values
        selected = index in DOBOT_LEG_INDICES[leg]
        assert values["kp"] == (config["control"]["kp"][index % 3] if selected else 0)
        assert values["kd"] == (config["control"]["kd"][index % 3] if selected else 0)
        assert values["q"] == target[index] + config["dds"]["motor_offset"][hw]
        assert values["tau"] == 0
    original_control = json.dumps(config["control"], sort_keys=True)
    config["control"]["shutdown_damping_s"] = config["physics_dt"]
    transport.publish_damping(leg)
    for index, hw in enumerate(config["dds"]["abs2hw"]):
        values = middleware.command[hw].values
        assert values["kp"] == 0
        assert values["kd"] == (
            config["control"]["damping_kd"] if index in DOBOT_LEG_INDICES[leg] else 0
        )
    config["control"]["shutdown_damping_s"] = json.loads(original_control)["shutdown_damping_s"]
    assert json.dumps(config["control"], sort_keys=True) == original_control


def test_capture_buffer_reset_discards_history_and_keeps_latest_state() -> None:
    transport = DobotDDS({"dds": {"first_state_timeout_s": 0.1}})
    samples = [
        StateSample(
            host_time_ns=stamp,
            q=np.full(12, stamp, dtype=np.float64),
            dq=np.zeros(12),
            motor_temp=np.zeros(12),
            tau_est=np.zeros(12),
        )
        for stamp in (1, 2, 3)
    ]
    transport._samples = deque(samples)

    assert transport.reset_capture_buffer() == 2
    retained = transport.snapshot()
    assert len(retained) == 1
    assert retained[0].host_time_ns == 3
    assert retained[0] is not samples[-1]


def test_wait_for_state_after_returns_only_a_new_callback() -> None:
    transport = DobotDDS({"dds": {"first_state_timeout_s": 0.1}})
    old = StateSample(1, np.zeros(12), np.zeros(12), np.zeros(12), np.zeros(12))
    fresh = StateSample(2, np.ones(12), np.zeros(12), np.zeros(12), np.zeros(12))
    transport._samples = deque((old,))

    def append_fresh_state() -> None:
        time.sleep(0.01)
        with transport._lock:
            transport._samples.append(fresh)

    worker = threading.Thread(target=append_fresh_state)
    worker.start()
    try:
        received = transport.wait_for_state_after(old.host_time_ns)
    finally:
        worker.join()

    assert received.host_time_ns == fresh.host_time_ns
    assert received is not fresh


def test_cyclic_gc_guard_restores_the_previous_state_after_an_error() -> None:
    initially_enabled = gc.isenabled()
    gc.enable()
    try:
        with pytest.raises(RuntimeError, match="test error"):
            with _suspend_cyclic_gc() as state:
                assert state["was_enabled"] is True
                assert state["disabled_during_active"] is True
                assert not gc.isenabled()
                raise RuntimeError("test error")
        assert gc.isenabled()
    finally:
        if not initially_enabled:
            gc.disable()


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
    assert parser.parse_args(["hold", "--leg", "all"]).leg == "ALL"


def test_all_trajectory_has_mirrored_feet_and_speed_limited_approach() -> None:
    config = load_config(ROOT / "config/dobot_hardware.json")
    dt = config["physics_dt"]
    _, targets, phases = generate_identification(config, np.asarray(DOBOT_DEFAULT_JOINT_POS), "ALL")
    hold = targets[phases == "hold"][-1]
    reference = np.asarray(config["hold"]["target_joint_pos"][:3])
    np.testing.assert_allclose(hold, (reference * np.asarray(DOBOT_MIRROR_SIGNS)).reshape(12))
    approach = targets[np.isin(phases, ("approach", "hold"))]
    assert (
        np.max(np.abs(np.diff(approach, axis=0))) / dt
        <= config["hold"]["max_command_velocity_rad_s"]
    )
    model = get_spec().compile()
    data = mujoco.MjData(model)
    sites = [model.site(name).id for name in ("FL", "FR", "RL", "RR")]
    reflections = np.array([[1, 1, 1], [1, -1, 1], [-1, 1, 1], [-1, -1, 1]])
    mirrored_targets = targets[phases != "approach"]
    for q in mirrored_targets[::40]:
        data.qpos[:] = q
        mujoco.mj_forward(model, data)
        feet = data.site_xpos[sites] - data.xpos[model.body("link_trunk").id]
        np.testing.assert_allclose(feet, feet[0] * reflections, atol=1e-12)


def test_chirp_phase_defaults_and_validation(tmp_path: Path) -> None:
    config = load_config(ROOT / "config/dobot_hardware.json")
    config["chirp"].pop("phase_deg")
    path = tmp_path / "old_config.json"
    path.write_text(json.dumps(config))
    loaded = load_config(path)
    assert loaded["chirp"]["phase_deg"] == [0.0, 0.0, 0.0]
    center = np.asarray(DOBOT_DEFAULT_JOINT_POS)
    _, original, _ = generate_chirp(loaded, center, "FL")
    config["chirp"]["phase_deg"] = [0.0, 0.0, 0.0]
    np.testing.assert_array_equal(generate_chirp(config, center, "FL")[1], original)
    for invalid in ([0, 90], [0, float("nan"), 90], [0, 90, float("inf")], [[0, 90, 90]]):
        config["chirp"]["phase_deg"] = invalid
        path.write_text(json.dumps(config))
        with pytest.raises(ValueError, match="chirp.phase_deg"):
            load_config(path)


@pytest.mark.parametrize("phase_deg", [[0, 0, 0], [0, 90, 90], [30, -45, 120]])
def test_chirp_joint_phases_and_smooth_boundaries(phase_deg) -> None:
    config = load_config(ROOT / "config/dobot_hardware.json")
    config["chirp"].update(
        phase_deg=phase_deg, min_frequency_hz=0.5, max_frequency_hz=0.5, duration_s=4.0, ramp_s=0.5
    )
    center = np.asarray(DOBOT_DEFAULT_JOINT_POS)
    _, targets, phases = generate_chirp(config, center, "FL")
    swing = targets[phases == "chirp", :3] - center[:3]
    dt = config["physics_dt"]
    amplitudes = np.asarray(config["chirp"]["amplitude_rad"])
    directions = np.asarray(config["chirp"]["direction"])
    # Quarter-cycle samples in the full-amplitude region, away from both ramps.
    for t in (0.5, 1.0, 1.5, 2.0):
        expected = amplitudes * directions * np.sin(np.pi * t + np.deg2rad(phase_deg))
        np.testing.assert_allclose(swing[round(t / dt)], expected, atol=1e-14)
    np.testing.assert_array_equal(swing[0], np.zeros(3))
    np.testing.assert_allclose(
        targets[phases != "chirp"], np.tile(center, (np.count_nonzero(phases != "chirp"), 1))
    )
    # Both boundary steps vanish quadratically with dt, even for cosine offsets.
    limit = amplitudes * (np.pi * dt / (2 * config["chirp"]["ramp_s"])) ** 2
    assert np.all(np.abs(swing[1]) <= limit + 1e-14)
    assert np.all(np.abs(swing[-1]) <= limit + 1e-14)


@pytest.mark.parametrize("phase_deg", [[0, 0, 0], [0, 90, 90]])
def test_mirrored_chirp_reduces_model_horizontal_reaction(phase_deg) -> None:
    """Prescribed-motion inverse dynamics; this is not hardware acceptance."""
    config = load_config(ROOT / "config/dobot_hardware.json")
    config["chirp"].update(
        phase_deg=phase_deg,
        amplitude_rad=[0.2, 0.1, 0.45],
        direction=[1.0, 1.0, 1.0],
        min_frequency_hz=0.1,
        max_frequency_hz=3.0,
        duration_s=20.0,
        ramp_s=2.0,
    )
    center = (np.array([0.05, 0.78, -1.2]) * np.asarray(DOBOT_MIRROR_SIGNS)).reshape(12)
    _, mirrored, phases = generate_chirp(config, center, "ALL")
    copied = center + np.tile(mirrored[:, :3] - center[:3], (1, 4))
    model = mujoco.MjModel.from_xml_path(str(ROOT / "src/pace_sim2real/assets/dobot/dobot.xml"))
    model.geom_contype[:] = 0
    model.geom_conaffinity[:] = 0
    joints = [model.joint(name).id for name in DOBOT_LEG_JOINTS["ALL"]]
    rms = []
    for q in (copied, mirrored):
        # Keep the floating base stationary; its inverse force is the required
        # support force, opposite to the legs' reaction on the support.
        data = mujoco.MjData(model)
        dq = np.gradient(q, config["physics_dt"], axis=0)
        ddq = np.gradient(dq, config["physics_dt"], axis=0)
        horizontal_force = []
        for i in np.flatnonzero(phases == "chirp"):
            data.qpos[model.jnt_qposadr[joints]] = q[i]
            data.qvel[model.jnt_dofadr[joints]] = dq[i]
            data.qacc[model.jnt_dofadr[joints]] = ddq[i]
            mujoco.mj_inverse(model, data)
            horizontal_force.append(data.qfrc_inverse[:2].copy())
        rms.append(np.sqrt(np.mean(np.sum(np.square(horizontal_force), axis=1))))
    assert rms[0] > 1.0
    assert rms[1] < 0.01 * rms[0]


def test_all_hold_and_tracking_abort_identify_failed_leg() -> None:
    config = load_config(ROOT / "config/dobot_hardware.json")
    target = np.zeros(12)
    samples = [
        StateSample(int(t * 1e9), target.copy(), target.copy(), target.copy(), target.copy())
        for t in np.linspace(1, 3, 21)
    ]
    assert hold_stability(config, samples, target, "ALL")[0] is None
    samples[-1].q[6] = 2 * config["hold"]["max_position_span_rad"]
    reason, metrics = hold_stability(config, samples, target, "ALL")
    assert reason.startswith("RL:")
    assert set(metrics) == {"FL", "FR", "RL", "RR"}
    samples[-1].q[11] = 2 * config["safety"]["max_abs_tracking_error_rad"]
    assert "joint_rear_right_calf_pitch" in safety_reason(
        config, samples[-1], target, "ALL", samples[-1].host_time_ns
    )


def test_all_conversion_carries_gains_without_source_capture(tmp_path: Path) -> None:
    source = tmp_path / "all.npz"
    values = np.arange(120).reshape(10, 12) / 100
    save_capture(
        source,
        time_s=np.arange(10) * 0.0025,
        dof_pos=values,
        des_dof_pos=values,
        phases=np.full(10, "chirp"),
        metadata={
            "leg": "ALL",
            "eligible_for_fit": True,
            "physics_dt": 0.0025,
            "control": {"kp": [40, 41, 42], "kd": [1, 2, 3]},
        },
        raw={},
    )
    output, _ = convert_capture(source, tmp_path / "all.pt")
    source.unlink()
    assert load_pace_artifact(output)["dof_pos"].shape == (10, 12)
    assert resolve_leg(None, output) == "ALL"
    gains = load_dobot_control(output, list(DOBOT_LEG_JOINTS["ALL"]), required=True)
    assert gains == {"kp": [40, 41, 42] * 4, "kd": [1, 2, 3] * 4}
    with pytest.raises(ValueError, match="joint_order"):
        load_dobot_control(output, list(DOBOT_LEG_JOINTS["FL"]), required=True)
    with output.open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="hash"):
        load_dobot_control(output, list(DOBOT_LEG_JOINTS["ALL"]), required=True)


@pytest.mark.parametrize("abort", [False, True])
def test_all_active_capture_gates_chirp_and_always_damps(tmp_path, monkeypatch, abort) -> None:
    from pace_sim2real.hardware import cli

    config = load_config(ROOT / "config/dobot_hardware.json")
    target = np.zeros(12)
    phases = np.array(["approach", "hold", "hold", "prehold", "chirp", "posthold"])
    instances = []

    class FakeTransport:
        def __init__(self, config):
            self.samples = []
            self.published = []
            self.damped = []
            self.writer_enabled = False
            instances.append(self)

        def latest(self):
            sample = StateSample(
                time.monotonic_ns(), target.copy(), target.copy(), target.copy(), target.copy()
            )
            self.samples.append(sample)
            return sample

        def start_reader(self):
            assert not self.writer_enabled

        wait_for_first_state = latest

        def wait_for_state_after(self, stamp):
            return self.latest()

        def baseline(self):
            return target.copy()

        def reset_capture_buffer(self):
            self.samples = [self.latest()]
            return 0

        def enable_writer(self):
            self.writer_enabled = True

        def publish_position(self, q, leg):
            assert self.writer_enabled and leg == "ALL"
            self.published.append(q.copy())
            self.latest()

        def publish_damping(self, leg):
            self.damped.append(leg)
            self.latest()

        def recent(self, window):
            return self.samples

        def snapshot(self):
            return self.samples

    def confirm(leg, mode):
        assert leg == "ALL" and not instances[-1].writer_enabled

    monkeypatch.setattr(cli, "DobotDDS", FakeTransport)
    monkeypatch.setattr(cli, "_confirm_active", confirm)
    monkeypatch.setattr(
        cli,
        "generate_identification",
        lambda *args: (
            np.arange(len(phases)) * config["physics_dt"],
            np.zeros((len(phases), 12)),
            phases,
        ),
    )
    monkeypatch.setattr(
        cli,
        "hold_stability",
        lambda *args: ("RL: hold position span exceeds limit" if abort else None, {}),
    )
    output = tmp_path / "capture.npz"
    if abort:
        with pytest.raises(RuntimeError, match="hold gate failed: RL"):
            cli.active(config, "collect", "ALL", output, False)
    else:
        assert cli.active(config, "collect", "ALL", output, False) == 0
    transport = instances[-1]
    assert transport.damped == ["ALL"]
    assert len(transport.published) == (3 if abort else len(phases))
    capture = load_capture(output, require_eligible=False)
    assert capture["metadata"]["eligible_for_fit"] == (not abort)
    assert capture["metadata"]["control"]["kp"] == config["control"]["kp"]
    assert capture["metadata"]["chirp"] == config["chirp"]
    if abort:
        assert "RL" in capture["metadata"]["error"]
        assert "chirp" not in capture["phase"]
