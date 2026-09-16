# ANYmal-D 机器人描述（URDF）

本目录包含由 ANYbotics 开发的 ANYmal-D 简化机器人描述。原作者与维护者署名：**Linus Isler，ANYbotics**。上游完整机器人描述、仿真及控制软件的获取方式以 ANYmal Research 社区说明为准。

本仓库使用简化 URDF 中的惯性和碰撞信息，加载时移除视觉 Collada 网格引用，使 PACE 流程不依赖 ROS 的网格路径解析。这里的文件用于 mjlab 模型加载。

上游 README 中的 ROS `load.launch`、`standalone.launch` 和 RViz 用法依赖完整的上游描述包；这些不是本仓库的运行入口。请使用 [ANYmal-D 仿真示例](../../../../docs/other-robots.md)。

模型沿用 [BSD-3-Clause 许可](LICENSE)，许可和版权原文保持不变。
