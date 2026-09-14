# 参与 PACE mjlab 开发

欢迎提交问题报告、中文文档改进、自定义机器人示例和 PACE 核心改进。较大的 API 或机器人接入变更，建议先创建 issue 讨论设计。

## 开发环境

进入自己的源码工作区：

```bash
uv sync --locked --group dev --group docs
```

本地预览文档：

```bash
uv run --group docs mkdocs serve
```

`site/` 为自动生成页面；文档源文件在 `docs/`，首页只维护阅读入口，具体操作在 `docs/dobot/` 中按主题维护。

代码变更提交前运行：

```bash
uv run ruff format --check src scripts tests
uv run ruff check src scripts tests
uv run pytest
uv run --group docs mkdocs build --strict
```

`pytest` 中的物理仿真集成测试需要可用的 mjlab 运行时，会运行 CPU 测试，并在 CUDA 可用时增加 CUDA 测试。这不是实机运动测试。仅修改文档时，优先检查文档构建、链接和命令与源码的一致性，无需执行采集或训练。

## 提交变更

创建功能分支，围绕明确问题修改，并在 PR 中说明最终行为及验证命令。自定义机器人示例应涵盖数据采集、辨识、可视化和文档。说明性文档统一使用中文，代码标识符、命令、论文题名及许可原文保留原样。

仓库保留经审核的全腿原始数据和 `runtime/dds/dist/` 配套安装包；数据范围见 [数据与归档](docs/dobot/data.md)。不要提交虚拟环境、缓存或生成的训练日志。实验报告可以跟踪，但应说明数据和参数归档位置；不要把缺少原始证据的报告写成已经完成硬件验收。

## 问题报告

请提供提交号、mjlab 版本、操作系统、Python 版本、相关 CUDA/GPU 信息、完整命令和报错日志。功能请求应说明目标机器人与具体工作流。

## 许可

贡献采用与本仓库及所保留上游 PACE 代码一致的 Apache-2.0 许可。
