# Dobot 数据转换、辨识与评估

以下命令在辨识机的仓库根目录执行。`<timestamp>`、`<run>` 等占位符须替换为实际文件。已有仓库保留的原始数据时，可从本页开始；文件范围见[数据与归档](data.md)。

## 转换原始数据

将合格 NPZ 复制到辨识机，在正常 `.venv` 环境转换：

```bash
uv run python scripts/pace/dobot.py convert \
  'data/dobot/all/all_collect_<timestamp>.npz' \
  --output data/dobot/all/train.pt
```

输出文件必须成对保留：

```text
data/dobot/all/train.pt
data/dobot/all/train.pt.json
```

转换只保留 `prehold`、`chirp`、`posthold`，排除进入保持姿态的阶段。单腿提取对应 3 列，ALL 保留 12 列。PT 内容为：

| 字段 | 形状 | 单位与含义 |
| --- | --- | --- |
| `time` | `[T]` | 秒；从零开始，按 0.0025 s 时间网格生成 |
| `dof_pos` | `[T, N]` | rad；按统一关节顺序的实测编码器角 |
| `des_dof_pos` | `[T, N]` | rad；采集时实际发送的同步位置目标 |

JSON 记录腿选择、关节顺序、源数据和输出哈希、时间步，以及新采集记录的 PD 增益。拟合和评估从它推断任务；显式传入冲突的 `--leg` 会报错。

迁移时至少复制 NPZ、PT 和 `.pt.json`，并保存本次实际配置。新数据自动使用采集增益；旧数据缺少增益时，要在 `fit/evaluate` 传入 `--config` 指向采集时的原配置，且哈希必须匹配。更旧的转换清单若无配置哈希，还需要原始 NPZ，不应手工伪造元数据绕过校验。

## CMA-ES 参数辨识

先做小规模离线流程检查：

```bash
uv run python scripts/pace/dobot.py fit \
  --data data/dobot/all/train.pt --num_envs 4 --max_iterations 1 --device cuda:0
```

一代只能证明流程能够运行，不能作为有效辨识结论。正式运行例如：

```bash
uv run python scripts/pace/dobot.py fit \
  --data data/dobot/all/train.pt --num_envs 64 --device cuda:0
```

`--num_envs` 是 CMA-ES 候选数，不是腿数，最少为 4。默认值 4096 显存需求较大，建议显式从 64 开始，根据启动时打印的轨迹内存和实际显存余量调整。迭代数由任务配置决定，也可以通过 `--max_iterations` 显式指定。

拟合结果写入 `logs/pace/dobot_all/<run>/`；单腿目录相应为 `dobot_fl` 等。保留终端打印的本次运行目录，不要默认“最新目录”就是需要评估的实验。

| 文件 | 用途 |
| --- | --- |
| `config.pt` | 本次优化配置、边界、关节顺序和训练轨迹等 |
| `control.json` | 使用采集元数据时写出的 PD 增益记录 |
| `best_params.pt` | 已完成各代中，实际评估过的最低损失候选及分数 |
| `best_trajectory.pt` / `best_trajectory_params.pt` | 最新保存检查点中相匹配的最佳轨迹和参数 |
| `population_best_*.pt` | 对应代的最佳候选、分数和轨迹 |
| `mean_*.pt` | CMA-ES 分布均值，不一定被实际评估过 |
| `progress.pt` | 仅启用完整优化过程保存时产生 |

参数含义与排列见[配置与参数](configuration.md)。

## 独立数据评估

另行采集未参与拟合的轨迹，记录它的配置与采集时间。可在机械约束内改变幅值或频率，转换为 `held_out.pt`。重复使用训练文件只能得到训练回放误差，不能称为独立验证。

```bash
uv run python scripts/pace/dobot.py convert \
  'data/dobot/all/all_collect_<held_out_timestamp>.npz' \
  --output data/dobot/all/held_out.pt

uv run python scripts/pace/dobot.py evaluate \
  data/dobot/all/held_out.pt 'logs/pace/dobot_all/<run>/best_params.pt' \
  --device cuda:0 \
  --output logs/analysis/dobot_held_out.json \
  --plot logs/analysis/dobot_held_out.png
```

程序重放保存的 `des_dof_pos`，不会按当前配置重新生成扫频。新数据自动使用自身采集时的 PD 增益；旧数据的 `--config` 应对应这份评估数据，不能误用训练数据配置。

| 报告字段 | 含义 |
| --- | --- |
| `fitted.rmse / mae` | 全部样本、全部辨识关节的总体误差，单位 rad |
| `fitted.joint_rmse / joint_mae` | 按 `joint_order` 排列的逐关节误差，单位 rad |
| `legs` | ALL 模式下的分腿指标 |
| `reference` | 显式传入 `--reference-parameters` 时才有的参考参数结果 |
| `control` | 从数据或原配置恢复的 PD 增益 |

rad 换算成度用 `误差 × 180 / π`。曲线叠加目标、实测和仿真角度；ALL 为四行三列。总体误差应与逐关节、分腿结果一起看，避免平均值掩盖某一关节问题。

若要证明相对标称模型的改善，需要先明确参考参数的来源和语义，并在同一数据上增加 `--reference-parameters <reference.pt>`。默认 `evaluate` 不自动建立标称参考，也不会直接给出硬件验收结论。

综合判断至少包括：独立数据是否合格、相对参考是否改善、参数是否贴边、不同采集的结果是否可重复。低 RMSE 可能由参数互相补偿产生，不能直接证明零位正确或解决外八姿态。历史例子见[中文实验分析](../analysis/dobot_bias_comparison_20260912.md)。

## 轨迹与结果查看

训练日志的二维轨迹可用通用工具查看，显式指定机器人和运行目录：

```bash
uv run python scripts/pace/plot_trajectory.py \
  --robot_name dobot_all --folder_name '<run>' --plot_trajectory \
  --save_dir logs/analysis/dobot_training_plot
```

通用绘图工具会打印 `mean_*.pt` 的均值参数，而轨迹图读取 `best_trajectory.pt` 及其配套参数；不要把终端打印的均值误认为图中候选。`--save_dir` 将图保存到文件，适合无桌面的辨识机。

`--plot_score` 需要拟合时启用 `sim2real.cmaes.save_optimization_process = True`，已有运行缺少 `progress.pt` 时不能事后仅靠绘图命令补出来。

本地的 `scripts/pace/view_comparison.py` 是历史实验专用三维回放脚本，绑定了特定的 `logs/analysis/` 路径和数据哈希。它不属于通用评估流程，不能直接套用到任意新实验；常规流程优先使用 `evaluate --plot`。
