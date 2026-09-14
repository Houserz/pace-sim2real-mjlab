# Dobot 数据与归档

## 仓库保留范围

仓库保留 8 份全腿有效原始 NPZ（约 28.18 MiB），位于 `data/dobot/all/`。筛选要求为 `leg=ALL`、`mode=collect`、`status=completed`、`eligible_for_fit=true`，并检查 12 关节数组、有限值、时间戳及扫频阶段。有效表示采集与格式检查通过，不代表辨识效果已验证。

| 2026-09-10 采集时间（文件名） | Kp / Kd |
| --- | --- |
| 174730、174929、175119、175639、180145 | 40 / 1.8 |
| 180339、180630、181114 | 25 / 1.3 |

各文件扫频时长和频率范围不同，以 NPZ 内嵌元数据为准；不能用当前默认配置替代历史采集设置。训练与验证数据应按实际实验来源划分，这 8 份文件不自动构成独立验证集。

`data/dobot/all/MANIFEST.json` 记录保留文件的哈希、大小、完整采集元数据和其他文件的排除原因。校验命令：

```bash
sha256sum -c data/dobot/all/MANIFEST.sha256
sha256sum -c runtime/MANIFEST.sha256
```

单腿数据、保持试验、失败/中断采集及转换后的 PT/JSON 被 Git 忽略，本地已有文件仍保留。新 NPZ 默认忽略，审核后再更新 `.gitignore` 白名单及数据清单。使用原始数据时按[辨识流程](identification.md)生成 PT 和 `.pt.json`。

## 配置与运行时

- `config/dobot_hardware.json`：默认采集、控制与安全配置。
- `runtime/dds/config/`：DDS 和 CycloneDDS 配置。
- `runtime/dds/dist/`：匹配的原生安装包与 Python wheel。
- `runtime/dds/requirements-hardware.txt`、`runtime/dds/licenses/`、`runtime/MANIFEST.sha256`：依赖、许可和校验清单。

## 实验归档与复现

每次实验至少保存以下内容：

- Git 提交号及未提交的源码/配置差异，`uv.lock` 和相关运行命令。
- 实际采集配置、原始 NPZ、转换 PT 与 `.pt.json`。
- 完整拟合运行目录，明确选定的参数文件及 SHA-256。
- 独立评估数据、参考参数、评估 JSON 和图像。
- 关节顺序、PD 增益、时间步、拟合边界、设备与运行日期。

转换文件、`logs/` 和 `config/dobot_hardware.local.json` 不随 Git 保存，需要另行备份。历史报告保存在 `docs/analysis/`，其中引用的完整训练日志和参数文件不随源码提供。
