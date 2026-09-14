# 接入自定义机器人

`PaceSim2realEnvCfg`、`PaceSim2realSceneCfg`、`PaceCfg` 和 `PaceDCMotorCfg` 保留公开名称，但实现已使用 mjlab。`AnymalDPaceEnvCfg` 是可继承的配置类，可在子类 `__post_init__()` 中调用父类方法后修改时间步和动作配置。

下面是接入结构示例。`assets/my_robot.xml`、关节名和执行器参数都需要按实际机器人替换，不能直接当作 Dobot 的配置使用。

## 1. 定义关节顺序与参数边界

`joint_order` 同时决定数据列、CMA-ES 参数块和后续参数迁移的顺序，必须保持稳定。`bounds_params` 应有 `4 * len(joint_order) + 1` 行。

```python
from dataclasses import dataclass, field
import torch
from pace_sim2real import PaceCfg

JOINT_ORDER = ["hip_left", "knee_left", "hip_right", "knee_right"]

@dataclass(kw_only=True)
class MyRobotPaceCfg(PaceCfg):
    robot_name: str = "my_robot"
    data_dir: str = "my_robot/chirp_data.pt"
    joint_order: list[str] = field(default_factory=lambda: list(JOINT_ORDER))
    bounds_params: torch.Tensor = field(default_factory=lambda: torch.tensor([
        *[[1e-5, 1.0]] * 4,  # 等效关节惯量
        *[[0.0, 7.0]] * 4,   # 被动黏性阻尼
        *[[0.0, 0.5]] * 4,   # 库仑摩擦
        *[[-0.1, 0.1]] * 4,  # 编码器偏置
        [0.0, 10.0],         # 力矩延迟，单位为物理步
    ], dtype=torch.float32))
```

## 2. 创建模型与 PACE 执行器

使用由 MuJoCo `MjSpec` 或 URDF 加载器提供的 `EntityCfg`。`PaceDCMotorCfg` 支持标量、逐关节列表和正则表达式字典；`joint_names_expr` 是 `target_names_expr` 的兼容别名。

```python
import mujoco
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from pace_sim2real.utils import PaceDCMotorCfg

PACE_ACTUATOR = PaceDCMotorCfg(
    joint_names_expr=JOINT_ORDER,
    saturation_effort={".*": 140.0}, effort_limit={".*": 89.0},
    velocity_limit={".*": 8.5}, stiffness={".*": 85.0}, damping={".*": 0.6},
    armature={".*": 0.0}, friction={".*": 0.0}, viscous_friction={".*": 0.0},
    encoder_bias={".*": 0.0}, max_delay=10,
)

def get_robot_cfg() -> EntityCfg:
    return EntityCfg(
        spec_fn=lambda: mujoco.MjSpec.from_file("assets/my_robot.xml"),
        articulation=EntityArticulationInfoCfg(actuators=(PACE_ACTUATOR,)),
        init_state=EntityCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.5), joint_pos={".*": 0.0}, joint_vel={".*": 0.0}
        ),
    )
```

`max_delay` 延迟已经计算并限幅的力矩。CMA-ES 的连续延迟坐标截断为非负整数物理步，执行器容量应至少覆盖延迟搜索上界截断后的值。

每个 `joint_order` 关节都必须由 `PaceDCMotor` 控制；不能直接用延迟位置指令的普通执行器替换。采集和拟合会拒绝不匹配的执行器或不足的延迟容量。

## 3. 配置环境

设置 `scene.robot` 后，`PaceSim2realEnvCfg` 提供位置动作、关节位置/速度/上一动作观测、零权重关节限位奖励和不触发的超时项。

```python
from pace_sim2real import PaceSim2realEnvCfg, PaceSim2realSceneCfg

@dataclass(kw_only=True)
class MyRobotPaceSceneCfg(PaceSim2realSceneCfg):
    robot: EntityCfg = field(default_factory=get_robot_cfg)

@dataclass(kw_only=True)
class MyRobotPaceEnvCfg(PaceSim2realEnvCfg):
    scene: MyRobotPaceSceneCfg = field(default_factory=MyRobotPaceSceneCfg)
    sim2real: MyRobotPaceCfg = field(default_factory=MyRobotPaceCfg)

    def __post_init__(self) -> None:
        super().__post_init__()
        self.sim.dt = 0.0025
```

动作向量包含全部受控执行器关节，而 `joint_order` 可以只包含待辨识子集。封装脚本把待辨识目标写入完整动作向量，其他动作列保持零。替换 `actions.joint_pos` 时，需要覆盖所有待辨识关节。

如需观测只包含辨识关节，可用 `SceneEntityCfg("robot", joint_names=tuple(self.sim2real.joint_order))` 构造观测项。

## 4. 注册并运行

注册代码必须在创建环境前导入：

```python
from mjlab.tasks.registry import register_mjlab_task
from pace_sim2real.tasks.manager_based.pace.agents.rsl_rl_ppo_cfg import ppo_runner_cfg

register_mjlab_task(
    task_id="Mjlab-Pace-My-Robot-v0",
    env_cfg=MyRobotPaceEnvCfg(),
    play_env_cfg=MyRobotPaceEnvCfg(episode_length_s=1.0e9),
    rl_cfg=ppo_runner_cfg(),
)
```

```bash
uv run python scripts/pace/data_collection.py --task Mjlab-Pace-My-Robot-v0 --task-module my_robot.tasks --device cuda:0
uv run python scripts/pace/fit.py --task Mjlab-Pace-My-Robot-v0 --task-module my_robot.tasks --num_envs 64 --device cuda:0
```

`--task-module` 导入负责注册的模块。请先安装自定义包，或把其父目录加入 `PYTHONPATH`。`pace-train`、`pace-play`、`scripts/zero_agent.py` 和 `scripts/random_agent.py` 也接受该选项。

保存的 PT 包含 `time`、`dof_pos`、`des_dof_pos`，角度列按 `joint_order` 排列。每个样本间隔一个物理步；不匹配的时间基准会被拟合程序拒绝。

## 5. 将参数用于后续仿真

先选定明确的参数文件并进行独立轨迹验证。`mean_*.pt` 是优化分布均值，不一定对应已评估候选；若使用它，应明确记录并单独评估。

以下展示历史均值张量如何按块拆分，仅演示配置映射：

```python
import torch

mean = torch.load("logs/pace/my_robot/<run>/mean_000.pt", weights_only=True)
n = len(JOINT_ORDER)
DEPLOY_ACTUATOR = PaceDCMotorCfg(
    joint_names_expr=JOINT_ORDER,
    saturation_effort=140.0, effort_limit=89.0, velocity_limit=8.5,
    stiffness=85.0, damping=0.6,
    armature=mean[:n].tolist(),
    viscous_friction=mean[n:2*n].tolist(),
    friction=mean[2*n:3*n].tolist(),
    encoder_bias=mean[3*n:4*n].tolist(),
    max_delay=int(mean[4*n].item()),
)
```

已创建的实体可通过 PACE 执行器的 `update_encoder_bias(...)` 和 `update_time_lags(...)` 更新对应项，这些值跨完整或部分重置保留。改变惯量、阻尼、摩擦等模型参数时，应重新创建相应环境。配置迁移不等于硬件验收，详见[执行器模型](actuators.md)和[优化器 API](optim.md)。
