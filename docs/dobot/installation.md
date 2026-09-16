# Dobot 环境部署

本仓库有两套环境。可以部署在同一台机器，也可以由采集机保存原始数据，再传到 GPU 辨识机处理。以下命令均在仓库根目录运行。

| 环境 | 用途 | 主要要求 |
| --- | --- | --- |
| `.venv` | 数据转换、辨识、离线评估、文档 | 64 位 Linux；项目声明 Python 3.10–3.13；mjlab 1.5.2；拟合建议使用 NVIDIA GPU |
| `.venv-hardware` | Dobot 主机检查、状态读取、保持、扫频采集 | Ubuntu x86_64、CPython 3.10、厂商 DDS 原生库及匹配 wheel |

## 1. 准备源码与 uv

安装 `uv` 后获取源码，或进入已存在的工作区：

```bash
git clone git@gitee.com:inffni-robotics/dobot-pace-sim2real-mjlab.git
cd dobot-pace-sim2real-mjlab
```

部署应使用已推送的同一提交；本地未提交的修改不会随克隆复制。

## 2. 辨识环境

```bash
uv sync --locked
uv run python scripts/list_envs.py
uv run python -c 'import torch; print(torch.__version__); print("CUDA available:", torch.cuda.is_available())'
```

任务列表应包含 `Isaac-Pace-Anymal-D-v0` 和 `Dobot-Pace-FL-v0`、`FR`、`RL`、`RR`、`ALL` 五个 Dobot 任务。`pyproject.toml` 固定 `mjlab==1.5.2`；`--locked` 用于检查并遵循当前锁文件。

需要开发检查或文档时，增加对应依赖组：

```bash
uv sync --locked --group dev --group docs
```

建议为虚拟环境和内核缓存预留约 8–10 GB 磁盘，长时间采集与拟合日志需要额外空间。确保 NVIDIA 驱动与安装的 PyTorch/CUDA 运行时匹配。首次 CUDA 运行会编译内核；正式拟合前先以少量候选检查显存。`--device cpu` 适合小规模离线调试。

## 3. Dobot 采集环境

仅采集的机器无需执行上一节的 `uv sync`。DDS 配套二进制不纳入 Git，随交付单独提供；取得后放入 `runtime/dds/dist/`：

- `dds-middleware-with-thirdparty_0.24.4_amd64.deb`
- `dds_middleware_python-0.24.4-cp310-cp310-linux_x86_64.whl`

安装前校验文件完整性；校验失败时先核对文件与清单。

```bash
sha256sum -c runtime/MANIFEST.sha256
sudo dpkg -i runtime/dds/dist/dds-middleware-with-thirdparty_0.24.4_amd64.deb
sudo ldconfig
uv venv --python 3.10 .venv-hardware
uv pip install --python .venv-hardware/bin/python -r runtime/dds/requirements-hardware.txt
uv pip install --python .venv-hardware/bin/python \
  runtime/dds/dist/dds_middleware_python-0.24.4-cp310-cp310-linux_x86_64.whl
uv pip install --python .venv-hardware/bin/python --no-deps -e .
```

`--no-deps` 避免将 Torch、mjlab 等辨识依赖装入采集环境。以后直接运行 `.venv-hardware/bin/pace-dobot`，避免在采集机上使用可能自动同步完整项目的 `uv run`。

## 4. 网络与只读检查

将连接机器人的本机网卡配置为 `192.168.5.100/24`，并核对实际机器人地址。当前 CycloneDDS XML 按 IP 地址绑定，网卡名称变化不需要修改配置。

```bash
ip -br address
ip route get 192.168.5.2
printenv CYCLONEDDS_URI
.venv-hardware/bin/pace-dobot doctor
.venv-hardware/bin/pace-dobot observe --duration 2
```

最后两条的边界不同：`doctor` 检查系统、文件清单及本机地址，不创建 DDS 端点；`observe` 创建状态订阅端，不创建指令发布端。收到状态不证明机器人控制权已释放。

程序根据配置设置 `CYCLONEDDS_URI`；已有环境变量若与配置不一致会报错。应核对并修正当前终端的旧值，不能把其他机器人会话的配置直接套用过来。

完成后按[配置与参数](configuration.md)检查本地配置，再进入[数据采集](collection.md)。
