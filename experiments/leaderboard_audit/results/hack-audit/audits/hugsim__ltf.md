# HUGSIM × Latent TransFuser（LTF）

- 榜单来源：[WA-JEPA 论文 Table 2](https://arxiv.org/abs/2608.20974) 的同一 436 场景重测，HD-Score 0.2310；原 HUGSIM 与 NAVSIM 论文/代码定义了基线，不能把不同场景数的旧分数与此值并列。
- 代码：[hyzhou404/NAVSIM](https://github.com/hyzhou404/NAVSIM/tree/ca0ca7e4368646d8f7b86fb1fdaa1862c946176f)，固定 commit `ca0ca7e4368646d8f7b86fb1fdaa1862c946176f`；提供 LTF 模型及 HUGSIM 测试客户端，但并非 436 场景重测的完整提交记录，B 档。模型论文为 [NAVSIM](https://arxiv.org/abs/2406.15349)，另参照 [HUGSIM 论文](https://arxiv.org/abs/2412.01718)。
- 只读审计；未下载 checkpoint/场景资产，未运行模型。

## 阅读范围与整体印象

- 已读：`ltf_e2e.py`、`hugsim/dataparser.py`、`navsim/agents/transfuser/transfuser_agent.py`、`transfuser_features.py`、`transfuser_model.py`，以及 NAVSIM 论文的输入/指标段落和 WA-JEPA HUGSIM Table 2、Appendix A。
- 未读：全部 NAVSIM 数据下载/训练脚本、HUGSIM 场景资产、436 个场景的复测日志。
- 整体印象：固定仓库足以静态审计 LTF 的 HUGSIM 推理适配；固定直行命令与论文共同命令协议不完全一致，但无证据表明它提高了分数。

## 代码路径和论文对照

`ltf_e2e.py` 用 HUGSIM 配置加载 LTF checkpoint，从 FIFO 读 `(obs, info)`，通过 `hugsim/dataparser.py` 构造三相机 `AgentInput`，再把输出轨迹转换成模拟器坐标。异常时传 `None` 并退出，而不是发送一条可继续计分的备用规划。`navsim/agents/transfuser/transfuser_features.py` 把高层命令、速度、加速度拼接成 8 维状态特征，模型确实消费该特征。NAVSIM 论文 §3.1–3.2 定义左、直、右导航目标和 ego 状态；WA-JEPA Appendix A 说明 LTF 使用三个前视相机且所有方法共用 ground-truth driving commands。

## 观察但未编码为分数增益

本单元无已确认的有利分数发现，因而最终 `findings.jsonl` 不列 LTF 条目；下述实现偏差保留供复现实测。

- [数据适配 36–51 行](https://github.com/hyzhou404/NAVSIM/blob/ca0ca7e4368646d8f7b86fb1fdaa1862c946176f/hugsim/dataparser.py#L36-L51) 注释掉 `info['command']` 的读取，固定 `command[1] = 1`；[特征构造 45–51 行](https://github.com/hyzhou404/NAVSIM/blob/ca0ca7e4368646d8f7b86fb1fdaa1862c946176f/navsim/agents/transfuser/transfuser_features.py#L45-L51) 会把这个命令送入模型。这使转弯场景的 HUGSIM 高层命令无法进入 LTF，属于实现与论文共同协议的偏差。没有直行固定值与真实命令的配对分数；它可能降低成绩，不能称为有利于分数的 hack。
- `parse_raw` 从模拟器 `info` 读取 ego 位姿、速度和加速度；这些输入对应 NAVSIM 的 ego status。是否符合真车传感器前提取决于定位系统，没有本次分数的独立影响证据。
- 当前仓库只显示公开客户端和通用 checkpoint 路径，WA-JEPA 同协议复测的具体运行配置与逐场景输出不在此仓库。结论限制为静态代码路径，不能追溯 0.2310 的逐场景结果。
