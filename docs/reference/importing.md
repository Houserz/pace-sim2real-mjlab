# Python 导入与环境绑定

公开导入名称与上游 PACE 保持兼容：

```python
from pace_sim2real import (
    CMAESOptimizer,
    PaceCfg,
    PaceSim2realEnvCfg,
    PaceSim2realSceneCfg,
)
from pace_sim2real.utils import PaceDCMotor, PaceDCMotorCfg, bind_environment
import pace_sim2real.tasks
```

导入 `pace_sim2real.tasks` 后，ANYmal-D 和 Dobot 任务会注册到 mjlab。通过 `mjlab.tasks.registry.load_env_cfg` 和 `mjlab.envs.ManagerBasedRlEnv` 创建环境，不需要 Isaac Sim 应用启动器。

PACE 不会动态修改 mjlab 类。封装脚本使用的 `make_env()` 会自动登记环境。如果自行创建 `ManagerBasedRlEnv`，并希望使用历史三参数 `update_simulator(robot, ...)`，需要显式绑定：

```python
env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0")
bind_environment(env)
optimizer.update_simulator(env.scene["robot"], joint_ids, initial_encoder_position)
```

新应用建议直接使用包含 `env` 的四参数形式，详见[高级优化](advanced.md)。
