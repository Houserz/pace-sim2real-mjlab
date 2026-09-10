from __future__ import annotations

from pathlib import Path

import mujoco
import pytest
import torch

from pace_sim2real import CMAESOptimizer, PaceCfg, PaceSim2realEnvCfg, PaceSim2realSceneCfg
from pace_sim2real.assets.anymal_d_asset import JOINT_ORDER, get_spec
from pace_sim2real.tasks.manager_based.pace.anymal_pace_env_cfg import AnymalDPaceCfg
from pace_sim2real.utils import PaceDCMotorCfg, load_pace_artifact, project_root


def test_public_api_and_paths_are_available():
    assert PaceCfg is not None
    assert PaceSim2realEnvCfg is not None
    assert PaceSim2realSceneCfg is not None
    assert CMAESOptimizer is not None
    assert (project_root() / "pyproject.toml").exists()

    cfg = PaceDCMotorCfg(
        joint_names_expr=("joint",),
        saturation_effort=140.0,
        effort_limit=89.0,
        velocity_limit=8.5,
        stiffness=85.0,
        damping=0.6,
        max_delay=10,
    )
    assert cfg.target_names_expr == ("joint",)
    # PACE delay is a custom torque buffer; the native command-delay buffer is
    # intentionally disabled to preserve the upstream physical model.
    assert cfg.max_delay == 10
    assert cfg.delay_max_lag == 0
    assert cfg.delay_hold_prob == 0.0


def test_anymal_asset_and_original_parameter_layout():
    model = get_spec().compile()
    assert [model.joint(i).name for i in range(12)] == list(JOINT_ORDER)
    assert model.nv >= 12

    cfg = AnymalDPaceCfg()
    assert cfg.joint_order == list(JOINT_ORDER)
    assert cfg.bounds_params.shape == (49, 2)
    assert torch.all(cfg.bounds_params[:12, 0] > 0)
    assert torch.equal(cfg.bounds_params[36:48, 0], torch.full((12,), -0.1))


def test_cmaes_log_format_and_stopping(tmp_path: Path):
    joint_order = ["left", "right"]
    bounds = torch.tensor([[0.0, 1.0]] * 9)
    data = {
        "time": torch.linspace(0.0, 0.01, 3),
        "dof_pos": torch.zeros(3, 2),
        "des_dof_pos": torch.zeros(3, 2),
    }
    optimizer = CMAESOptimizer(
        bounds=bounds,
        population_size=4,
        log_dir=tmp_path,
        joint_order=joint_order,
        max_iteration=1,
        data=data,
        device="cpu",
        epsilon=None,
    )
    try:
        for _ in range(3):
            optimizer.tell(torch.zeros(4, 2), torch.zeros(4, 2))
        optimizer.evolve()
        assert optimizer.finished()
        run_dirs = list(tmp_path.iterdir())
        assert len(run_dirs) == 1
        assert (run_dirs[0] / "config.pt").exists()
        assert (run_dirs[0] / "mean_000.pt").exists()
        assert (run_dirs[0] / "best_trajectory.pt").exists()
        assert (run_dirs[0] / "best_trajectory_params.pt").exists()
        assert (run_dirs[0] / "population_best_000.pt").exists()
        assert (run_dirs[0] / "best_params.pt").exists()
        best = load_pace_artifact(run_dirs[0] / "best_params.pt")
        assert best["joint_order"] == joint_order
        assert best["params"].shape == (9,)
    finally:
        optimizer.close()


def test_packaged_urdf_compiles_without_visual_meshes():
    # This guards the self-contained MjSpec import path used at runtime.
    spec = get_spec()
    assert isinstance(spec, mujoco.MjSpec)


@pytest.mark.parametrize("save_interval", [0, 1])
@pytest.mark.parametrize(
    "errors",
    [[[3.0, 2.0, 1.0, 4.0], [4.0, 2.0, 3.0, 5.0]], [[4.0, 2.0, 3.0, 5.0], [3.0, 2.0, 1.0, 4.0]]],
)
def test_cmaes_best_is_lowest_evaluated_candidate_across_generations(
    tmp_path: Path, save_interval: int, errors: list[list[float]]
):
    optimizer = CMAESOptimizer(
        bounds=torch.tensor([[0.0, 1.0]] * 5),
        population_size=4,
        log_dir=tmp_path,
        joint_order=["joint"],
        max_iteration=2,
        data={
            "time": torch.tensor([0.0, 0.01]),
            "dof_pos": torch.zeros(2, 1),
            "des_dof_pos": torch.zeros(2, 1),
        },
        device="cpu",
        save_interval=save_interval,
    )
    try:
        generations = []
        with pytest.raises(RuntimeError, match="before evolve"):
            optimizer.get_best_sim_params()
        for generation_errors in errors:
            residual = torch.tensor(generation_errors).unsqueeze(1)
            trajectories = torch.stack(
                [
                    optimizer.sim_params[:, optimizer.bias_idx] + scale * residual
                    for scale in (1.0, 2.0)
                ],
                dim=1,
            )
            for sample in trajectories.unbind(dim=1):
                optimizer.tell(sample, torch.zeros_like(sample))
            score, index = (2.5 * residual.square().squeeze(1)).min(dim=0)
            generations.append(
                (optimizer.sim_params[index].clone(), trajectories[index].clone(), score.clone())
            )
            optimizer.evolve()

        assert optimizer.finished()
        expected_params, expected_trajectory, expected_score = min(
            generations, key=lambda generation: generation[2].item()
        )
        torch.testing.assert_close(optimizer.get_best_sim_params(), expected_params)
        # Callers must not be able to mutate the optimizer's retained best result.
        optimizer.get_best_sim_params().zero_()
        torch.testing.assert_close(optimizer.get_best_sim_params(), expected_params)
        run_dir = Path(optimizer.writer.log_dir)
        best = load_pace_artifact(run_dir / "best_params.pt")
        torch.testing.assert_close(best["params"], expected_params)
        torch.testing.assert_close(best["score"], expected_score)
        torch.testing.assert_close(
            load_pace_artifact(run_dir / "best_trajectory.pt"), expected_trajectory
        )
        torch.testing.assert_close(
            load_pace_artifact(run_dir / "best_trajectory_params.pt"), expected_params
        )
        for generation in [0, 1] if save_interval else [1]:
            population = load_pace_artifact(run_dir / f"population_best_{generation:03}.pt")
            for field, expected in zip(
                ("params", "trajectory", "score"), generations[generation], strict=True
            ):
                torch.testing.assert_close(population[field], expected)
        mean = load_pace_artifact(run_dir / "mean_001.pt")
        torch.testing.assert_close(
            mean, optimizer._params_to_sim_params(torch.tensor(optimizer.optimizer._mean))
        )
        assert not torch.allclose(mean.float(), expected_params)
    finally:
        optimizer.close()
