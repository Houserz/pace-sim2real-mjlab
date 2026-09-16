# Dobot 第三方运行时文件

Dobot 模型随仓库提供。DDS 二进制保留在 `runtime/dds/dist/`，供打开 DDS 端点前校验哈希。

- `src/pace_sim2real/assets/dobot/`：Dobot Rover 标称模型与网格。
- `runtime/dds/dist/dds-middleware-with-thirdparty_0.24.4_amd64.deb`：Ubuntu amd64 原生 DDS 运行时。
- `runtime/dds/dist/dds_middleware_python-0.24.4-cp310-cp310-linux_x86_64.whl`：匹配的 CPython 3.10 绑定。
- `runtime/dds/licenses/` 和模型目录保留相应许可原文。

二进制目录随本仓库保留，克隆后按 `runtime/MANIFEST.sha256` 校验。部署步骤见 [环境部署](../docs/dobot/installation.md)。
