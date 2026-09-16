# 策略训练与回放

本仓库主要用于 PACE 系统辨识。RSL-RL 入口为已有项目模板保留；内置 PACE 任务的奖励权重为零，适合检查环境和 API 兼容性，增加有效任务奖励前不能用于有意义的行走训练。

默认使用本地 TensorBoard 日志，不需要登录或上传 W&B：

```bash
uv run python scripts/rsl_rl/train.py \
  --task Isaac-Pace-Anymal-D-v0 --num_envs 64 --max_iterations 1 \
  --logger tensorboard
```

兼容封装会转换历史参数，包括 `--num_envs`、`--max_iterations`、`--seed`、`--logger`、`--checkpoint`、`--device`、`--headless` 和布尔参数 `--video`。原生 mjlab 参数形式通过 `--help` 查看。仅显式使用 `--logger wandb` 时启用云日志。

未提供检查点时，`play` 在 `logs/rsl_rl` 中寻找最新本地 `model_*.pt`。也可以指定明确文件：

```bash
uv run python scripts/rsl_rl/play.py \
  --task Isaac-Pace-Anymal-D-v0 --checkpoint 'logs/rsl_rl/pace_sim2real/<run>/model_1.pt'
```

不使用检查点的环境检查可传 `--agent zero` 或 `--agent random`。这里的策略检查点与 PACE 的 `best_params.pt` 是不同类型的文件，不能互换。
