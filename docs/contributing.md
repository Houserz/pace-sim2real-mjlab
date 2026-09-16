# 参与贡献

开发环境使用 `uv sync --locked --group dev --group docs`。代码变更运行 Ruff 与 pytest，文档变更运行 `uv run --group docs mkdocs build --strict` 并核对链接和命令。

完整流程见源码根目录的 `CONTRIBUTING.md`，包括问题报告、PR 说明和实验产物归档要求。文档正文统一使用中文；标识符、命令、引用题名和法定许可原文保持原样。
