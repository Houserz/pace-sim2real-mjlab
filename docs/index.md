# Dobot 使用文档

按当前任务直接进入对应页面，所有操作命令均从仓库根目录执行。

| 资料 | 内容 |
| --- | --- |
| [环境部署](dobot/installation.md) | 辨识环境、采集环境、DDS 安装与网络 |
| [配置与参数](dobot/configuration.md) | 关节顺序、采集配置、49 项参数及源码位置 |
| [数据采集](dobot/collection.md) | 主机检查、状态读取、保持与扫频 |
| [辨识与评估](dobot/identification.md) | NPZ 转换、CMA-ES 拟合、独立评估与画图 |
| [数据与归档](dobot/data.md) | 仓库保留的原始数据、筛选规则与复现材料 |
| [常见问题](dobot/troubleshooting.md) | 环境、网络、采集与辨识排查 |
| [历史实验分析](analysis/dobot_bias_comparison_20260912.md) | 2026-09-12 五组实验与 bias 分析；不作为当前配置说明 |

首次使用按“环境部署 → 配置与参数 → 数据采集 → 辨识与评估”阅读。已有原始数据时，可直接完成辨识环境安装并进入辨识流程。

PACE 使用固定躯干、关闭接触的 Dobot 模型做悬空腿辨识；离线误差不等于行走或实机部署验收。

[其他机器人](other-robots.md)仅作兼容性说明；开发接口在导航的“技术参考”中，来源与研究引用见[许可与致谢](legal.md)。
