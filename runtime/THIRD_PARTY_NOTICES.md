# Third-party Dobot runtime files

The Dobot model is vendored in this repository.  The DDS binary artifacts are
supplied separately; copy them into `runtime/dds/dist/` so their hashes can be
verified before opening a DDS endpoint.

- `src/pace_sim2real/assets/dobot/`: Dobot Rover nominal model and meshes.
- `runtime/dds/dist/dds-middleware-with-thirdparty_0.24.4_amd64.deb`: separately supplied vendor
  native DDS runtime for Ubuntu amd64.
- `runtime/dds/dist/dds_middleware_python-0.24.4-cp310-cp310-linux_x86_64.whl`:
  separately supplied matching CPython 3.10 binding.
- License texts are retained under `runtime/dds/licenses/` and with the model
  package.
