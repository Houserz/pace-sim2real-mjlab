# 仿真迁移与参数映射

PACE 参数映射到 mjlab / MuJoCo-Warp 的以下字段：

| PACE 参数 | 仿真字段或实现 |
| --- | --- |
| 等效关节惯量 armature | `dof_armature` |
| 被动黏性阻尼 | `dof_damping` |
| 库仑摩擦 | `dof_frictionloss` |
| 编码器偏置 | `EntityData.encoder_bias` |
| 力矩延迟 | `PaceDCMotor` 的力矩延迟缓冲区 |

PACE 采用 `q_encoder = q - bias`，mjlab 内部采用 `q_encoder = q + encoder_bias`。兼容层在边界转换符号，因此 PACE 配置和辨识值继续保持原有语义。

ANYmal-D 保留上游固定基座设置：`dt=0.0025 s`、`decimation=1`、基座高度 `z=1.0 m`。Dobot 使用自身模型和初始高度 `0.65 m`，运行时移除自由关节并关闭所有接触；不要把两个任务的初始条件混为一谈。

PACE 先根据当前状态计算 PD 力矩并按力矩—转速限制裁剪，再对所得力矩施加延迟。这与直接延迟位置目标不同，尤其在延迟期间关节仍然运动时，两者的反馈含义不同。
