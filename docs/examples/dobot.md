# Dobot Rover identification

Use `FL`, `FR`, `RL`, or `RR` for one leg, or explicitly select `ALL` for
simultaneous mirrored four-leg identification. Collection stays in the small
hardware environment; conversion, fitting and evaluation use the mjlab environment.

`ALL` requires a fixed trunk and four airborne legs. Review and run a hold-only
trial before collecting. Software and model symmetry are not hardware acceptance.

## Four-leg operator workflow

The existing config is the only experiment configuration. In `ALL` mode:

- `hold.target_joint_pos[:3]` is the FL reference pose `[abad, thigh, calf]`.
  The other nine hold entries are used only for single-leg commands.
- `chirp.amplitude_rad` and `chirp.direction` stay three-element vectors in that
  same joint order. All legs share the frequency, duration and envelope.
- `chirp.phase_deg` sets the three joint phases in **degrees**. `[0, 90, 90]`
  gives `sin / cos / cos`; `[0, 0, 0]` restores the original in-phase sweep.
  Older configs without this field default to `[0, 0, 0]`. Phase settings also
  apply to single-leg collection.
- `control.kp` and `control.kd` also stay three-element vectors, shared by all legs.
- The complete reference pose plus chirp is mirrored as below, in logical joint
  coordinates before motor offsets are added. Rear pitch centers change sign.

| Leg | abad | thigh | calf |
| --- | --- | --- | --- |
| FL | a | b | c |
| FR | -a | b | c |
| RL | a | -b | -c |
| RR | -a | -b | -c |

For example, the current three-joint excitation is configured as:

```json
"amplitude_rad": [0.2, 0.1, 0.45],
"phase_deg": [0.0, 90.0, 90.0],
"direction": [1.0, 1.0, 1.0]
```

The reference target is `center + envelope * amplitude * direction * sin(phase +
phase_offset)`. The envelope smoothly introduces and removes the cosine offsets,
so the sweep starts and ends at the hold pose. The horizontal foot trace can be
approximately elliptical; its shape and changing height follow the leg geometry.
Four-leg mirroring applies to the entire target, irrespective of joint phase.

The approach starts from measured positions and can be asymmetric. Its duration
is automatically extended when needed to respect `hold.max_command_velocity_rad_s`.
The final hold and sweep targets are mirrored. Each leg must pass the existing
hold gate before the sweep begins. Mirroring targets reduces horizontal reactions
and moments in the model; vertical reactions are not cancelled or an acceptance
criterion here. Real tracking and fixture motion still require operator validation.

After the installation and read-only checks below, run these commands **manually**:

```bash
.venv-hardware/bin/pace-dobot hold --leg ALL
.venv-hardware/bin/pace-dobot collect-chirp --leg ALL
```

Each command prints the actual four-leg hold pose, joint sweep limits, approach
duration, PD gains and output path, plus the sweep amplitude, phase and frequency
before collection, then requires `ARM ALL <TOKEN>` once before
creating one DDS writer. Inspect the mirrored rear-leg pose and clearance before
confirming. All 12 joints are sent in one command at 400 Hz. Any safety or hold-gate
failure aborts the collection and sends bounded damping to all four legs.
The raw NPZ metadata records the actual `chirp` settings, including `phase_deg`.
Conversion and fitting replay the saved joint targets rather than regenerating
them from the current config.

Collection prints a timestamped `data/dobot/all/all_collect_<timestamp>.npz` path.
Replace the placeholders below with actual paths; these steps are offline:

```bash
uv run python scripts/pace/dobot.py convert data/dobot/all/all_collect_<timestamp>.npz
uv run python scripts/pace/dobot.py fit \
  --data data/dobot/all/all_collect_<timestamp>.pt --num_envs 64
uv run python scripts/pace/dobot.py evaluate \
  data/dobot/all/held_out.pt logs/pace/dobot_all/<run>/best_params.pt \
  --plot logs/pace/dobot_all/<run>/held_out.png \
  --output logs/pace/dobot_all/<run>/held_out.json
```

For held-out validation, collect and convert a second mirrored sweep with a
reviewed different amplitude or frequency. New PT files carry their actual PD
gains in `.pt.json`; copy **both files** to the fitting computer. Fitting and
validation infer `ALL` and use those gains automatically. The result has 49
parameters: 12 armatures, 12 passive damping values, 12 friction values, 12 encoder
biases and one shared effective PACE torque delay. `--num_envs` counts candidate
parameter sets, not legs. The report includes per-leg errors; the figure has four
rows (FL/FR/RL/RR) and three columns (abad/thigh/calf).

Older captures without recorded gains require `--config <original-config.json>`
for both `fit` and `evaluate`; a hash mismatch is rejected. For old conversion
manifests without a config hash, the original NPZ must also remain available.

## Files and responsibilities

| Path | Responsibility |
| --- | --- |
| `src/pace_sim2real/dobot.py` | Dependency-free joint order, leg groups, limits, and timestep. |
| `src/pace_sim2real/assets/dobot_asset.py` | Fixed-base, contact-free MJLab model and selected-leg PACE actuator. |
| `src/pace_sim2real/tasks/manager_based/pace/dobot_pace_env_cfg.py` | Single-leg (13 parameters) and ALL (49 parameters) fitting tasks. |
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
| `hold --leg LEG` | Hardware only | Command writer; moves the selected leg(s) | Raw hold NPZ |
| `collect-chirp --leg LEG` | Hardware only | Command writer; moves the selected leg(s) | Raw collection NPZ |
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

kill controller
```bash
cd /home/dobot/code/dobot_quad_sdk/high_level/python
  /home/dobot/miniconda3/envs/dobot-one-leg-hw/bin/python \
    examples/kill_robot.py 192.168.5.2:50051
```

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
matched.  In single-leg mode, the other nine joints are present in the whole-body message with
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
rejected. Dobot fitting and evaluation require the conversion sidecar to recover
joint order and captured gains.
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

New captures automatically supply their recorded `control.kp` and `control.kd`
to fitting and evaluation. The optional `--config` is needed for older captures;
it verifies the recorded configuration hash and rejects disagreement with any
recorded gains. Preserve one local configuration per experiment until validation
is complete.  The desired positions already contain the exact center,
amplitude, and frequency sent to hardware; evaluation replays those samples
rather than regenerating a nominal chirp.  `--plot` overlays target, real, and
simulated joint positions.

`best_params.pt` contains the lowest-loss evaluated candidate across all
completed generations and its score, matching the parameters printed and
returned by fitting.  `best_trajectory.pt` and `best_trajectory_params.pt`
refer to that same candidate at the latest checkpoint.  `mean_*.pt` remains
the CMA-ES distribution mean, which is not necessarily an evaluated candidate.
Each `population_best_*.pt` bundles the score, parameter vector, and matching
trajectory of that generation's best candidate.

Passing unit tests, bundle hashes, or a low training score does not establish a
physical identification result.  Acceptance requires an eligible independent
capture, improvement over a declared nominal reference, parameter-boundary
review, and repeatability across runs.
