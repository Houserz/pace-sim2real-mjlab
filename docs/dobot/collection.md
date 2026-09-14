# Dobot 数据采集

先完成[环境部署](installation.md)，按[配置说明](configuration.md)创建并检查 `config/dobot_hardware.local.json`。以下命令在仓库根目录执行。

## 只读检查

```bash
.venv-hardware/bin/pace-dobot doctor --config config/dobot_hardware.local.json
.venv-hardware/bin/pace-dobot observe --duration 2 --config config/dobot_hardware.local.json
```

`doctor` 不导入厂商 DDS 模块，也不创建端点；PASS 仅说明其检查项通过。`observe` 订阅 `rt/lower/state`，可核对状态是否到达及关节角是否合理。它不会创建 `rt/lower/cmd` 发布端，也不验证控制器已经退出。

开始主动试验前，操作者应确认躯干固定、所选腿悬空、扫频全范围无碰撞、急停可用，且没有其他 `rt/lower/cmd` 发布者。ALL 要求四条腿都悬空；单腿模式下，其余九关节在整机消息中为 `Kp=0、Kd=0、tau=0`，同样需要机械支撑。

旧文档中的控制器退出脚本位于另一台机器的厂商 SDK 路径，并非本仓库提供的通用命令。请使用实际部署系统已验证的控制器退出流程，不能仅根据某个旧路径直接执行。

## 保持与扫频采集（由操作者执行）

先执行四腿保持试验：

```bash
.venv-hardware/bin/pace-dobot hold --leg ALL --config config/dobot_hardware.local.json
```

确认保持稳定后，再执行扫频：

```bash
.venv-hardware/bin/pace-dobot collect-chirp --leg ALL --config config/dobot_hardware.local.json
```

单腿实验将两条命令中的 `ALL` 改为 `FL`、`FR`、`RL` 或 `RR`。主动命令必须显式选择腿；不能让程序自动猜测。

程序显示实时状态、保持中心、扫频范围、PD 增益、轨迹时长和输出路径，并要求输入 `ARM <LEG> <TOKEN>`。匹配确认前不创建指令发布端。之后依次执行平滑进入、保持及稳定性检查、扫频和阻尼退出；保持试验本身不执行扫频。ALL 中每腿都必须通过保持检查。

采集默认输出如下，具体路径以终端打印为准：

```text
data/dobot/all/all_collect_<timestamp>.npz
data/dobot/fl/fl_collect_<timestamp>.npz
```

检查结束报告中的 `status` 与 `eligible_for_fit`。不合格或中断采集不能直接拿来拟合；转换器会拒绝不合格数据。保持试验数据也不能替代正式扫频数据。

原始 NPZ 保存 12 关节状态、目标、主机回调时间、温度、估计力矩、阶段标签及配置/模型哈希等信息。`host_time_ns` 是主机回调到达时间，不是机器人内部采样时间；`tau_est` 是诊断信号，不是已标定的力矩真值。
