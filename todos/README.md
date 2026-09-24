# todos/

实验规划与执行记录。简单计划使用 `YYYY-MM-DD-<slug>.md`；需要协作、日志与图表的实验使用同名文件夹。
新建时复制 `TEMPLATE.md`。

用中文写。临时、随手的东西放 `tmp/`（已 gitignore），不要放这里。

- [2026-09-22 B2D controller](2026-09-22-b2d-controller.md)：已完成轨迹控制器实施、反馈迭代与Dev10验收；无合格替代默认，全部结果见目录。

- [2026-09-23 真实TCP与转弯控制](2026-09-23-tcp-controller/README.md)：真实模型纵向对照、静止速度反馈修复，以及按转弯窗口单独优化横向。

- [2026-09-23-lateral-followup/](2026-09-23-lateral-followup/README.md)：逐弯控制诊断、Hermite负面闭环、固定后轴传播补偿及全部中间图表。

- [2026-09-23 横向 v2](2026-09-23-controller-next/lateral-v2-report.md)：后轴速度系 pursuit + Ackermann 反解的预注册闭环（[协议](2026-09-23-controller-next/lateral-v2-protocol.md)）；CTE 主目标与 held-out 全部改善，动作保护量失败，候选不通过。

- [2026-09-23 W2：TFv6 + 我们的控制器](2026-09-23-tfv6-controller/report.md)：TFv6 waypoint 交给 production 控制器与作者 waypoint PID 的四臂预注册闭环（[协议](2026-09-23-tfv6-controller/protocol.md)）；route+speed 表示比 waypoint 高约 14 DS；W2 的 C/D 被静止时的坐标变换缺陷污染；修正后重跑（[W2b](2026-09-23-tfv6-controller/report-w2b.md)），C−B 合并 +1.0 [−12, +15]，仍未检出。
- [2026-09-23 P4 CARLA 特征差距](2026-09-23-p4-carla-feature-gap.md)：Waymo 训的 head 在 CARLA 帧上能不能用（domain AUC、词表覆盖、head 迁移），预登记判据决定 P5 的前提。
- [2026-09-24 P5 CARLA 配对考卷 v0](2026-09-24-p5-carla-pairs-v0.md)：Bench2Drive 10 个 family 造 165 对 ego 逐帧相同的反事实配对 + 55 个天气 null，expert 两侧重跑出标签，考 TFv6 与 CARLA 内训的 Qwen 薄 head（结论见 decisions 第 32 条）。
- [2026-09-24 openpilot smoke run](2026-09-24-openpilot-smoke/README.md)：small / Cinque v3 / Lebowski 三个 openpilot 驾驶模型在 box 上用 onnxruntime + TensorRT 跑通（batch 1 p50 1.0 / 2.3 / 3.3 ms），数值与 CPU 参考等价，comma1M 4 段真实视频上 plan 明显好于 constant velocity；附 zero-shot 上 WOD-E2E / NAVSIM 的缺口。
- [2026-09-24 Alpamayo 1.5 smoke run](2026-09-24-alpamayo-smoke/README.md)：10B reasoning VLA 在 box 上跑通（权重走 ModelScope、clip 按 zip 成员稀疏拉取），batch 1 默认 n=1 p50 983 ms / n=6 2722 ms，SDPA+compile+flow 5 步降到 697 / 2184 ms 且输出基本不变；31 个 clip 上 minADE_6 0.74 m（card 0.916 m）；Cosmos-Reason2-8B gate 未通过需用户 accept；附 zero-shot WOD-E2E / NAVSIM 的缺口。
- [2026-09-24 榜单 hack 审计](2026-09-24-hack-audit/README.md)：7 个榜单各抽 3–5 个公开代码的方法，只读代码、对照论文，找出分数里不属于驾驶能力的部分，分类由审计自己长出来；Codex 在 Tokyo box 上执行。
- [2026-09-24 zoo 进 CARLA](2026-09-24-zoo-in-carla.md)：已 dropped，并入进行中的 zeroshot-exam；只有「无 scenario 路段」这一项移到 R 层测量。
- [2026-09-24 R 层测量](2026-09-24-r-layer-routine.md)：去掉 scenario 的路线上测 routine 驾驶能力，看排序是否等于榜单排序，并把 TFv6 的分数拆成接口、手写规则、网络三部分。
- [2026-09-24 P5 v1：E 层测量](2026-09-24-p5-v1-e-layer.md)：在 v0 上加绕行类 family 和横向标签、剔除规则兜住的题、接入 PDM-Lite 与 zoo 考生，测突发事件时反应的方向和方式。
