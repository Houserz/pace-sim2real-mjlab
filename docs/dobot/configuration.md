# Dobot 配置与参数

统一入口支持以下选择：

| 选择 | 关节范围 | 内部任务 | 参数向量长度 |
| --- | --- | --- | --- |
| `FL` | 左前腿 | `Dobot-Pace-FL-v0` | 13 |
| `FR` | 右前腿 | `Dobot-Pace-FR-v0` | 13 |
| `RL` | 左后腿 | `Dobot-Pace-RL-v0` | 13 |
| `RR` | 右后腿 | `Dobot-Pace-RR-v0` | 13 |
| `ALL` | 四腿同时 | `Dobot-Pace-ALL-v0` | 49 |

四腿顺序固定为 **FL、FR、RL、RR**；每腿依次为 **abad（外展）、thigh（大腿）、calf（小腿）**。完整名称由 `src/pace_sim2real/dobot.py` 定义。数据列、参数块和迁移配置都必须沿用此顺序，不能按文件名排序猜测。

默认采集配置为 `config/dobot_hardware.json`。建议首次实验复制一份本地配置，再检查和调整：

```bash
cp -n config/dobot_hardware.json config/dobot_hardware.local.json
```

该本地文件被 Git 忽略。[采集流程](collection.md)中的命令统一显式传入它，避免误用另一份配置。配置中的 DDS 相对路径以配置文件所在目录解析，因此本地副本放在 `config/` 下。

以下是编写本文时（2026-09-14）默认配置的快照，不是任意机器人可直接使用的运动许可：

| 设置 | 默认配置值 | 含义 |
| --- | --- | --- |
| `physics_dt` | 0.0025 s | 400 Hz 控制与回放时间步 |
| `control.kp / kd` | 每关节 25 / 1.3 | 三元素向量，四腿共享；不是被辨识的被动阻尼 |
| `hold.target_joint_pos` | 前腿各 `[0, -0.7, 1.20]`，后腿各 `[0, 0.75, -1.22]` rad | 12 个实际保持中心 |
| `hold.ramp_s / duration_s` | 6 / 3 s | 进入保持姿态与保持时长；进入时间可因速度约束延长 |
| `chirp.duration_s` | 30 s | 扫频段时长 |
| `chirp.min_frequency_hz / max_frequency_hz` | 0.1 / 4 Hz | 扫频范围 |
| `chirp.amplitude_rad` | `[0.2, 0.25, 0.4]` rad | 三类关节激励幅值 |
| `chirp.phase_deg` | `[0, 90, 90]` 度 | 外展为正弦，大腿、小腿相移 90 度 |
| `chirp.pre_hold_s / post_hold_s` | 2 / 2 s | 扫频前后静止段 |

ALL 使用配置里的全部 12 个中心角，**只镜像扫频偏移，不替换中心角**：

| 腿 | abad 偏移 | thigh 偏移 | calf 偏移 |
| --- | --- | --- | --- |
| FL | a | b | c |
| FR | -a | b | c |
| RL | a | -b | -c |
| RR | -a | -b | -c |

目标为 `中心角 + 包络 × 幅值 × 方向 × sin(扫频相位 + 关节相移)`。包络让激励平滑进入和退出。中心角不满足镜像时，不能仅凭偏移镜像推断整机受力抵消。旧配置缺少 `phase_deg` 时按 `[0, 0, 0]` 处理。

## 参数排列与单位

令 `N` 为辨识关节数，向量按参数块排列，并非按每个关节连续存四个参数：

```text
[0:N]       armature       等效关节惯量，kg·m²
[N:2N]      damping        被动黏性阻尼，N·m·s/rad
[2N:3N]     frictionloss   库仑摩擦，N·m
[3N:4N]     encoder_bias   编码器偏置，rad
[4N]        delay          共享力矩延迟，仿真步数
```

ALL 为 `12×4+1=49`；单腿为 `3×4+1=13`。bias 约定是 `q_encoder = q_sim - bias`。DDS 配置中的 `motor_offset` 是硬件坐标转换，不能直接等同于拟合 bias。

delay 先截断为非负整数步，再作用于已经计算并完成限幅的力矩。例如 `4.5889` 实际为 4 步，即 `4×0.0025=0.010 s`，不是四舍五入到 5 步，也不是连续的 11.47 ms 延迟。

编写本文时的**工作区边界**为 armature `[1e-6, 1]`、damping `[0, 5]`、frictionloss `[0, 1]`、bias `[0, 0]`、delay `[0, 5]`。bias 固定为零时仍保留 49/13 项文件布局，但 bias 不再是自由变量。历史实验请读取各自 `config.pt`；当前任务配置以源码为准，不要用当前源码反推历史设置。

## 源码位置

| 文件 | 职责 |
| --- | --- |
| `src/pace_sim2real/dobot.py` | 关节顺序、腿分组、映射和时间步 |
| `src/pace_sim2real/assets/dobot_asset.py` | 固定躯干、关闭接触及 PACE 执行器配置 |
| `src/pace_sim2real/tasks/manager_based/pace/dobot_pace_env_cfg.py` | 辨识边界与任务配置 |
| `src/pace_sim2real/hardware/cli.py` | 主机检查、状态观察、交互确认和采集流程 |
| `src/pace_sim2real/hardware/dobot.py` | DDS 状态读取、指令发布、保持与安全检查 |
| `src/pace_sim2real/hardware/excitation.py` | 平滑进入和扫频轨迹 |
| `src/pace_sim2real/hardware/data.py` | 采集格式、转换清单和增益恢复 |
| `src/pace_sim2real/scripts/dobot.py` | 统一命令分发 |
| `src/pace_sim2real/scripts/fit.py` / `evaluate.py` | 辨识与离线回放评估 |
