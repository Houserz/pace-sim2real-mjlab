# Dobot Rover 模型来源

`dobot.xml` 和 STL 网格逐字节复制自 `dobot_one_leg_sysid` 所用、已核对的 Dobot MJLab v2 模型。

- 源 XML SHA-256：`ebb45f4cd4697cef2f24659675affbd788bdc25d49a4e2f115ab81e514b7fd55`
- 标称机器人质量：`17.2352 kg`
- 辨识运行时间步：`0.0025 s`

XML 是来自 CAD/URDF 的标称模型，不是辨识后的实机模型。PACE 任务在内存中移除自由关节、关闭接触，只为选择的单腿或 ALL 关节添加 PACE 执行器。

Dobot 对应许可原文保存在本目录 `LICENSE`，源码工作区的 `runtime/dds/licenses/` 中也保留一份。
