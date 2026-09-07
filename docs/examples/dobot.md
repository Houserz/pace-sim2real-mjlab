# Dobot Rover single-leg identification

The Dobot integration keeps real-time DDS collection separate from GPU fitting.
The first supported hardware target is one airborne, mechanically isolated leg:
`FL`, `FR`, `RL`, or `RR`.  Simultaneous multi-leg motion needs a separate
fixture and excitation review and is intentionally not exposed as an active
command.

The migrated historical hardware evidence covers FL.  The other three task and
collector selections share the audited joint mapping, but each still requires a
new operator-reviewed hold-only trial before chirp collection; software symmetry
is not physical validation.

## Files and responsibilities

| Path | Responsibility |
| --- | --- |
| `src/pace_sim2real/dobot.py` | Dependency-free joint order, leg groups, limits, and timestep. |
| `src/pace_sim2real/assets/dobot_asset.py` | Fixed-base, contact-free MJLab model and selected-leg PACE actuator. |
| `src/pace_sim2real/tasks/manager_based/pace/dobot_pace_env_cfg.py` | Four registered single-leg fitting tasks and 13-parameter bounds. |
| `config/dobot_hardware.json` | DDS mapping, offsets, gains, trajectory, and abort thresholds. |
| `src/pace_sim2real/hardware/dobot.py` | Lower-state reader and explicitly enabled lower-command writer. |
| `src/pace_sim2real/hardware/excitation.py` | Smooth approach, hold gate, and tapered chirp. |
| `src/pace_sim2real/hardware/data.py` | Raw NPZ schema, timing alignment, and selected-leg PT conversion. |
| `src/pace_sim2real/scripts/dobot.py` | One workflow entry point and leg-to-task routing. |
| `src/pace_sim2real/hardware/cli.py` | Isolated hardware command implementation. |

## Build the two environments

The normal mjlab environment performs conversion, fitting, and evaluation:

```bash
uv sync
```

Add `--group dev` only when pytest and Ruff are needed.

The hardware collector supports Ubuntu x86_64 and CPython 3.10.  First copy the
separately supplied vendor `.deb` and `.whl` into `runtime/dds/dist/`.  From the
repository root, verify and install them, then create the small environment:

```bash
sha256sum -c runtime/MANIFEST.sha256
sudo dpkg -i runtime/dds/dist/dds-middleware-with-thirdparty_0.24.4_amd64.deb
sudo ldconfig
uv venv --python 3.10 .venv-hardware
uv pip install --python .venv-hardware/bin/python -r runtime/dds/requirements-hardware.txt
uv pip install --python .venv-hardware/bin/python \
  runtime/dds/dist/dds_middleware_python-0.24.4-cp310-cp310-linux_x86_64.whl
uv pip install --python .venv-hardware/bin/python --no-deps -e .
```

The collection computer does not need Torch, CUDA, mjlab, or CMA-ES.  When only
the hardware environment is installed, invoke hardware commands directly with
`.venv-hardware/bin/pace-dobot`; using the repository-level `uv run` command may
sync the full mjlab project environment.

| Command | Required environment | DDS behavior | Main output |
| --- | --- | --- | --- |
| `doctor` | Hardware only | No endpoint | Host check report |
| `observe` | Hardware only | State reader only | State report |
| `hold --leg LEG` | Hardware only | Command writer; moves the selected leg | Raw hold NPZ |
| `collect-chirp --leg LEG` | Hardware only | Command writer; moves the selected leg | Raw collection NPZ |
| `convert` | Normal mjlab environment | No DDS | PACE PT and JSON sidecar |
| `fit` | Normal mjlab environment | No DDS | CMA-ES parameters and logs |
| `evaluate` | Normal mjlab environment | No DDS | Held-out metrics |

Assign the laptop's robot-facing interface `192.168.5.100/24`.  The bundled
CycloneDDS XML binds by address rather than by interface name, so different
laptops do not edit tracked configuration merely because the NIC is named
differently.

## Read-only checks

`doctor` does not import the vendor DDS module and opens no endpoint:

```bash
.venv-hardware/bin/pace-dobot doctor
```

`observe` subscribes to `rt/lower/state` but has no writer call:

```bash
.venv-hardware/bin/pace-dobot observe --duration 2
```

## Operator-run hardware commands

The following commands move hardware.  They must be run by the operator only
after mechanical support, sweep clearance, emergency stop, and absence of a
competing `rt/lower/cmd` writer are confirmed.

An optional hold-only trial performs the smooth approach and stability gate but
does not run the chirp:

```bash
.venv-hardware/bin/pace-dobot hold --leg FL
```

The `collect-chirp` command performs approach, hold, automatic stability
gate, chirp, and bounded damping exit:

```bash
.venv-hardware/bin/pace-dobot collect-chirp --leg FL
```

Both commands display the live state and trajectory envelope, then require one
short `ARM <LEG> <TOKEN>` confirmation.  No writer exists before that prompt is
matched.  The other nine joints are present in the whole-body message with
`Kp=0`, `Kd=0`, and `tau=0`; they require mechanical support.

The collection command writes raw data by default to:

```text
data/dobot/<leg>/<leg>_collect_<timestamp>.npz
```

This NPZ can be copied to a separate fitting computer.  It is not yet
`chirp_data.pt`; conversion is an offline step in the normal environment.

## Convert and fit

Raw captures keep all twelve joints, host callback timestamps, temperature,
estimated torque, phase labels, configuration hash, and model hash.  Conversion
keeps only `prehold`, `chirp`, and `posthold`, avoiding approach-heavy loss:

```bash
uv run python scripts/pace/dobot.py convert \
  data/dobot/fl/fl_collect_<timestamp>.npz \
  --output data/dobot/fl/chirp_data.pt

uv run python scripts/pace/dobot.py fit \
  --data data/dobot/fl/chirp_data.pt \
  --num_envs 64
```

The conversion sidecar records the source/output hashes and joint order.  The
workflow infers the leg and internal mjlab task from this sidecar.  Supplying
`--leg` is optional for fitting and evaluation, but a conflicting value is
rejected.  Without a sidecar, `--leg` is required.
`host_time_ns` is callback arrival time rather than a robot sampling timestamp,
and `tau_est` is retained only as a diagnostic signal, not calibrated torque
ground truth.

## Held-out validation

Create a local ignored configuration with different safe amplitude, frequency,
or direction, collect a second eligible capture, and convert it.  Replay the
exact saved parameter tensor on that independent trajectory:

```bash
uv run python scripts/pace/dobot.py evaluate \
  data/dobot/fl/held_out.pt \
  logs/pace/dobot_fl/<run>/best_params.pt \
  --config config/dobot_hardware.local.json \
  --plot logs/pace/dobot_fl/<run>/held_out.png \
  --output logs/pace/dobot_fl/<run>/held_out.json
```

The optional `--config` applies the capture's `control.kp` and `control.kd` to
the simulation.  It also verifies the configuration hash recorded in the raw
capture, so preserve one local configuration per experiment until evaluation
is complete.  The desired positions already contain the exact center,
amplitude, and frequency sent to hardware; evaluation replays those samples
rather than regenerating a nominal chirp.  `--plot` overlays target, real, and
simulated joint positions.

`best_params.pt` is the best sampled population member.  `mean_*.pt` is the
CMA-ES distribution mean.  Each `population_best_*.pt` bundles a score,
parameter vector, and its matching trajectory, avoiding ambiguity between a
mean parameter vector and another candidate's trajectory.

Passing unit tests, bundle hashes, or a low training score does not establish a
physical identification result.  Acceptance requires an eligible independent
capture, improvement over a declared nominal reference, parameter-boundary
review, and repeatability across runs.
