# Dobot Rover model provenance

`dobot.xml` and the STL files are copied byte-for-byte from the audited Dobot
MJLab v2 model used by `dobot_one_leg_sysid`.

- Source XML SHA-256: `ebb45f4cd4697cef2f24659675affbd788bdc25d49a4e2f115ab81e514b7fd55`
- Nominal robot mass: `17.2352 kg`
- Runtime identification timestep: `0.0025 s`

The XML is a nominal CAD/URDF-derived model, not an identified real-robot
model.  The PACE task removes its free joint in memory, disables contact, and
adds only the selected leg's PACE actuators.

The corresponding Dobot license is bundled as `LICENSE` in this directory and
again under `runtime/dds/licenses/` in a source checkout.
