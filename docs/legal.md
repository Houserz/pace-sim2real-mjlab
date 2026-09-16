# 许可与署名

本仓库是上游 [PACE Sim2Real](https://github.com/leggedrobotics/pace-sim2real) 的 mjlab 移植版本。以下说明沿用仓库内已保存的版权记录，法定许可文本不作翻译替换。

| 材料 | 版权或来源记录 | 许可文本位置 |
| --- | --- | --- |
| mjlab 移植及文档 | Ziqi Fan，2026；Apache-2.0 | 根目录 `LICENSE` 和 `LICENSES/ZIQI-FAN-APACHE-2.0.txt` |
| 保留的上游 PACE 代码 | ETH Zurich, Robotic Systems Lab, Filip Bjelonic，2025；Apache-2.0 | `LICENSES/UPSTREAM-PACE-NOTICE.txt` 和根目录 `LICENSE` |
| 简化 ANYmal-D 描述 | ANYbotics AG，2023；BSD-3-Clause | `src/pace_sim2real/assets/anymal_d/LICENSE` |
| Dobot 模型及 DDS 运行时 | 仓库中保留各自来源记录；DDS 二进制单独交付、不纳入 Git（校验值见 runtime/MANIFEST.sha256） | `src/pace_sim2real/assets/dobot/LICENSE`、`runtime/dds/licenses/`、`runtime/THIRD_PARTY_NOTICES.md` |

根目录 `LICENSE` 保存完整 Apache License 2.0 条款；`LICENSES/` 中的独立署名记录用于明确上游和移植部分的来源。中文化不会移除原版权、署名或许可文本。

## 上游致谢与研究引用

上游项目为 [leggedrobotics/pace-sim2real](https://github.com/leggedrobotics/pace-sim2real)，由 ETH Zurich Robotic Systems Lab 的 Filip Bjelonic、René Zurbrügg 和 Oliver Fischer 维护。保留上游致谢：RSL Learning Group，以及 Konrad、Matthias、Zichong、Stephan、Efe、Yuntao、René、Clemens、Ryo、Alexander、Fabio、Oliver Fischer、René Zurbrügg，和早期测试者 Oliver、Clemens、Yasmine。

研究引用信息沿用上游记录：

```bibtex
@article{bjelonic2025towards,
  title         = {Towards Bridging the Gap: Systematic Sim-to-Real Transfer for Diverse Legged Robots},
  author        = {Bjelonic, Filip and Tischhauser, Fabian and Hutter, Marco},
  journal       = {arXiv preprint arXiv:2509.06342},
  year          = {2025},
  eprint        = {2509.06342},
  archivePrefix = {arXiv},
  primaryClass  = {cs.RO},
}
```
