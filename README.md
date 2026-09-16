# Dobot PACE Sim2Real（mjlab）

基于 PACE、mjlab / MuJoCo-Warp 的 **Dobot Rover 动力学辨识项目**。通过实机关节扫频数据与 CMA-ES 优化，辨识关节惯量、阻尼、摩擦、编码器偏置和力矩延迟，用于改进仿真模型。

支持四腿 12 关节及单腿流程：**原始数据 → 转换 → 参数辨识 → 独立数据评估**。仓库保留 8 份有效全腿原始数据、Dobot 模型、硬件配置与 DDS 配置/许可（二进制单独交付）；生成的辨识结果另行归档。ANYmal-D 作为上游兼容示例保留。

辨识使用固定躯干、关闭接触的模型；离线结果仍需独立验证。

## 文档

- [Dobot 使用文档](docs/index.md)：部署、配置、采集、辨识与评估。
- [原始数据与归档](docs/dobot/data.md)：保留范围、校验与复现材料。
- [历史实验分析](docs/analysis/dobot_bias_comparison_20260912.md)：五组实验与 bias 分析。

开发与文档预览见[贡献指南](CONTRIBUTING.md)。项目沿用 Apache-2.0，模型及第三方文件按各自许可提供；[许可、上游致谢与研究引用](docs/legal.md)。
