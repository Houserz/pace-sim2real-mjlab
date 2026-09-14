# 高级优化

`CMAESOptimizer` 保留上游归一化 `[-1, 1]` 搜索空间和参数排列。目标函数接收原始仿真关节角与编码器坐标下的实测角，在内部应用一次候选 bias。

手工创建环境后，先绑定环境即可继续使用历史三参数调用。仓库封装脚本已经处理绑定：

```python
from pace_sim2real.utils import bind_environment

bind_environment(env)
optimizer.update_simulator(robot, joint_ids, initial_encoder_position)
```

新代码建议显式传入环境，无需依赖绑定注册表：

```python
optimizer.update_simulator(env, robot, joint_ids, initial_encoder_position)
```

如需加权误差或速度相关损失，可派生并改写 `CMAESOptimizer.tell()`。信任新目标函数前，应使用已知参数生成数据，验证参数恢复行为。

`tell()` 的仿真和实测输入都应是有限浮点张量，形状为 `[population_size, joint_count]`。非法样本在修改 CMA-ES 状态之前被拒绝。边界必须有限且有序，最后的延迟范围必须非负。
