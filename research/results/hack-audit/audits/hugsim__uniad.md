# HUGSIM × UniAD

- 榜单来源：[WA-JEPA 论文 Table 2](https://arxiv.org/abs/2608.20974)，436 场景同协议重测，HD-Score 0.3124。
- 代码：[hyzhou404/UniAD_SIM](https://github.com/hyzhou404/UniAD_SIM/tree/5fb279e39912a5ac7f58e00d56b065cadcd0a749)，固定 commit `5fb279e39912a5ac7f58e00d56b065cadcd0a749`，公开 HUGSIM 适配推理客户端；训练和完整榜单重测配置不在此 fork，B 档。原方法论文为 [UniAD](https://arxiv.org/abs/2212.10156)，另参照 [HUGSIM 论文](https://arxiv.org/abs/2412.01718)。
- 只读审计；未下载 checkpoint/场景资产，未运行模型。

## 阅读范围与整体印象

- 已读：`tools/closeloop/e2e.py`、`dataparser.py`、`README.md`，以及 UniAD 论文规划与命令段落、WA-JEPA HUGSIM Table 2 和 Appendix A。
- 未读：UniAD 全部模型头与训练脚本、HUGSIM 场景资产、436 场景复测日志。
- 整体印象：公开客户端能说明适配输入/输出，但固定代码的六相机配置与 WA-JEPA 复测论文所述四相机不一致，无法把该客户端直接等同于 0.3124 的提交。

## 代码路径和论文对照

`tools/closeloop/e2e.py` 从 FIFO 读 HUGSIM 帧，经 `dataparser.py` 转成 UniAD 相机张量、内外参、位姿和车辆状态，然后加载 checkpoint 运行规划头，并把 `sdc_traj` 写回控制管道。[入口 231–273 行](https://github.com/hyzhou404/UniAD_SIM/blob/5fb279e39912a5ac7f58e00d56b065cadcd0a749/tools/closeloop/e2e.py#L231-L273)。UniAD 论文 §3.4 与附录 F.5 说明规划头使用高层命令；此客户端 [dataparser.py 99–106 行](https://github.com/hyzhou404/UniAD_SIM/blob/5fb279e39912a5ac7f58e00d56b065cadcd0a749/tools/closeloop/dataparser.py#L99-L106) 从 HUGSIM `info['command']` 读取，和 WA-JEPA Appendix A 披露的共同 ground-truth command 协议一致。

## 核验结果与限制

- 未找到有代码与论文共同支持、且能论证提高这项 HUGSIM HD-Score 的榜单专用做法，因此本单元无 `findings.jsonl` 条目。共同的命令输入按基准协议记录，不当作 UniAD 独有得分手段。
- 固定客户端 [e2e.py 210–211 行](https://github.com/hyzhou404/UniAD_SIM/blob/5fb279e39912a5ac7f58e00d56b065cadcd0a749/tools/closeloop/e2e.py#L210-L211) 设六个相机；WA-JEPA Appendix A 称同表除 LTF 外的方法使用四个相机。公开客户端不能独立证明 0.3124 所用传感器配置，须获得该次复测的精确配置后才能归因。
- [数据适配 91–105 行](https://github.com/hyzhou404/UniAD_SIM/blob/5fb279e39912a5ac7f58e00d56b065cadcd0a749/tools/closeloop/dataparser.py#L91-L105) 把 `scene_token` 固定为 `062`。单场景客户端每次启动可能缓解跨场景状态混淆，但缺运行证据与分数消融，暂列复现问题而非有利分数发现。
- [推理异常分支 247–281 行](https://github.com/hyzhou404/UniAD_SIM/blob/5fb279e39912a5ac7f58e00d56b065cadcd0a749/tools/closeloop/e2e.py#L247-L281) 捕获 RuntimeError 后，后处理在检查 `results is not None` 前就索引 `results[0]`，可能导致客户端直接退出。该路径看起来是鲁棒性缺陷，不构成已证实得分增益。
- 此客户端的 0.3124 是 WA-JEPA 论文复测数字；仓库未附完整复测日志和对应 HUGSIM 控制器版本，不能复核绝对分数。
