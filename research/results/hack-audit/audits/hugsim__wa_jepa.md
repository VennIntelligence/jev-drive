# HUGSIM × WA-JEPA

- 榜单来源：[WA-JEPA 论文 Table 2](https://arxiv.org/abs/2608.20974)，436 个场景，HD-Score 0.4462。论文 Appendix A 说明四数据源、难度加权、统一重测控制器；不能与早期 345 场景的 DrivoR 数值直接比较。
- 代码：[AFARI-Research/WA-JEPA](https://github.com/AFARI-Research/WA-JEPA/tree/bec29660f5ea46ac73db8e2d0c33c8d1a72c23ad)，固定 commit `bec29660f5ea46ac73db8e2d0c33c8d1a72c23ad`；训练、推理、NAVSIM/HUGSIM 评测路径均在仓库，A 档。论文存于 `papers/wa_jepa.pdf`。
- 只读审计；未下载权重或场景资产，未运行模型和模拟器。

## 阅读范围与整体印象

- 已读：`scripts/evaluation/run_hugsim_benchmark.sh`、`close_loop/run_fixed_controller.py`、`run_upstream_compat.py`、`hugsim_client.py`、`hugsim_planner.py`、`aggregate_results.py`、`prep_scenarios.py`；论文正文 Table 2、Appendix A/相关 Table 5。
- 未读：全部训练数据处理、模型每层实现、HUGSIM 场景资产和逐场景运行日志。
- 整体印象：仓库对 HUGSIM 控制器和异常计分路径作了可见修正/防护。控制器修正已在论文披露并统一重测；异常制动是否进入主分数无运行证据。

## 代码路径和计分关系

`scripts/evaluation/run_hugsim_benchmark.sh` 选择场景、调用 `close_loop/run_fixed_controller.py`，默认再经 `run_upstream_compat.py` 修复上游崩溃。`close_loop/hugsim_client.py` 从 FIFO 收取 `(obs, info)`，`hugsim_planner.py` 将相机帧、历史位姿、速度及高层命令转成模型输入，预测的前向/左向轨迹再转成 HUGSIM 控制器坐标。`aggregate_results.py` 计算全部结果均值，也列出有推理失败或缺统计文件的非可信场景及另算的可信均值。论文 Appendix A 给出 436 场景与主聚合方式；Table 5 报告另外两种聚合，WA-JEPA 为 0.4483、0.4464，排序未变。

## 已编码发现

1. `HUG-WA-001`，`simulator_protocol_dependency`：评测入口把 HUGSIM 的轨迹转航向函数替换为修正 `arctan2` 参数顺序的版本。[代码 77–93 行](https://github.com/AFARI-Research/WA-JEPA/blob/bec29660f5ea46ac73db8e2d0c33c8d1a72c23ad/close_loop/run_fixed_controller.py#L77-L93)；[调用脚本 118–139 行](https://github.com/AFARI-Research/WA-JEPA/blob/bec29660f5ea46ac73db8e2d0c33c8d1a72c23ad/scripts/evaluation/run_hugsim_benchmark.sh#L118-L139)。该修正改变模型轨迹到车控的仿真链，不能视为网络自身能力；但论文 Appendix A 明确披露采用修正后 commit `ead17f2` 并对所有方法统一重测，不能据此指称对同表基线的不公平优势。没有修正前后的配对分数。
2. `HUG-WA-002`，`inference_failure_brake_fallback`：推理或轨迹合法性检查异常后，客户端发送全零轨迹作为强制制动。[代码 192–207 行](https://github.com/AFARI-Research/WA-JEPA/blob/bec29660f5ea46ac73db8e2d0c33c8d1a72c23ad/close_loop/hugsim_client.py#L192-L207)。仓库自身聚合器说明：即使一直原地不动，低碰撞与保持可行驶区域仍可形成一定 HD-Score，故另列失败场景可信均值。[聚合说明 24–33 行](https://github.com/AFARI-Research/WA-JEPA/blob/bec29660f5ea46ac73db8e2d0c33c8d1a72c23ad/close_loop/aggregate_results.py#L24-L33)。未提供本次 0.4462 的逐场景失败日志，因此只确认潜在污染路径，不断言已报分数实际含失败回退。论文正文、Appendix A 与评测说明未见该回退。

## 其他核验与边界

- `hugsim_planner.py` 使用 `info['command']`，而 HUGSIM 官方环境把基于最近录制位姿选出的命令放在 `info` 中；论文 Appendix A 明示所有方法使用 ground-truth driving commands。这是共同协议条件，尚无证据支持将其认定为 WA-JEPA 独有增益。
- `prep_scenarios.py` 只把 3DGS 场景文件存在的 YAML 写入 `runnable.txt`，运行脚本优先读该清单。论文声称 436 个场景；本次没有场景资产或运行记录，不能判断实际是否有场景被剔除，也不据此记录为已发生的分数增益。
- 仓库的控制器修补与评分代码是源代码证据，非本审计复现实验。论文报告的模型效能仍需同一场景、同一控制器的配对消融才能拆分。
