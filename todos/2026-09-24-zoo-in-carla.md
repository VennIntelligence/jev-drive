# Zoo 进 CARLA：让 openpilot 和 Alpamayo 在 CARLA 里被正确驱动，并过一道「会不会正常开」的关卡

状态: draft
执行者: Opus 执行代理（在 GPU box 上）
主题: ../research/capability-vs-leaderboard.md

## 目标

把我们已经在 GPU box 上跑通的真实世界驾驶模型接进 CARLA 0.9.15 / Bench2Drive，让它们**用各自训练时的传感器配置**看 CARLA，
并且能被开环 shadow（车由 expert 开，只读模型输出）和闭环（车由模型开）两种方式调用。
然后过一道关卡：在没有突发事件的路上，它们能不能正常地开。

这是 [R 层测量](2026-09-24-r-layer-routine.md) 和 [E 层测量（P5 v1）](2026-09-24-p5-v1-e-layer.md) 的共同前提。
P4 已经证明跨域是真问题（Waymo 训的 head 在 CARLA 帧上 domain AUC 为 1.000），所以不先过这一关，
后面测到的低分就分不清是「不会开」还是「看不懂 CARLA」。

## 对象

| 模型 | 已有的东西 | 训练时的输入 |
|---|---|---|
| openpilot Lebowski（877M，0.11.2 出货） | [smoke run](2026-09-24-openpilot-smoke/README.md)：onnxruntime + TensorRT，`jevdrive/openpilot/`，输入管线移植自 master | narrow 路面相机 + wide 相机，按 calibration warp 到 model frame；20 Hz；desire、traffic convention、v_ego |
| openpilot Cinque Terre v3（382M） | 同上 | 同上 |
| openpilot small（30M，车端） | 同上 | 同上 |
| Alpamayo 1.5（10B reasoning VLA） | [smoke run](2026-09-24-alpamayo-smoke/README.md)：`jevdrive/alpamayo/`，batch 1 约 0.7–1 s | 4 路相机（cross-left、front-wide、cross-right、front-tele）× 4 帧 @10 Hz，16 步 egomotion 历史 |
| TFv6（参照，已接好） | P5 与第 31 条里的 shadow / 闭环 wrapper | 它自己的 3 相机 + LiDAR + radar |

## 硬约束

- **第三方模型按原样运行，只测量、不改写**（CLAUDE.md）：不 fine-tune，不改模型代码。可以改的只有 wrapper：
  CARLA 里的传感器摆放、分辨率和内参，输入打包，以及输出到 control 的转换。
- **传感器要对齐训练分布**：相机的内参、外参、分辨率、帧率尽量贴近官方配置，每一项偏差都写进文档。
  openpilot 需要 calibration（device_from_calib）；Alpamayo 需要它那 4 路相机的几何关系。查不到官方参数的项写明是估计。
- **导航条件**：B2D 给的是 50 m 降采样后的 route 和 command。每个模型能吃什么样的导航输入（openpilot 的 desire、
  Alpamayo 是否接受导航或文本条件）要查清楚并写明。模型本身不支持按路线转弯的，如实记录，不要为它另写一个路径规划器兜底。
- **输出到 control**：openpilot 输出 plan / desired curvature / desired accel，Alpamayo 输出轨迹。
  统一用 production 控制器（它已经有 waypoint 接口），或者写一个最简单、有文档的转换；同一个转换对所有模型一致。
  注意第 31 条的教训：车静止时近端 waypoint 的抖动会造出假目标点，要用 W2b 修正后的切向规则。
- **时延**：闭环用 CARLA synchronous mode，模型推理时仿真暂停，所以 Alpamayo 的约 1 s 推理不影响结果；
  但要单独报告每个模型的实际推理时延，并注明「同步闭环不代表能实时运行」。
- 实验在 GPU box 上跑（CARLA 多实例的接法见 [benchmarks-and-evaluation.md](../research/benchmarks-and-evaluation.md) 第 3 节）；
  Tokyo box 只用来看画面、录屏核对传感器摆放。

## 关卡（跑之前写死）

路线：Bench2Drive 的 Dev10 路线 + 第 31 条的保留集，共 16 条，**去掉 scenario**（XML 里不放 scenario 元素，只剩背景车流），每条 1 个 TM seed。
参照：BehaviorAgent（特权 expert），以及 TFv6 的 A 臂（route + target speed + 作者 PID）。

| 指标 | 定义 |
|---|---|
| RC | Route Completion，完成的路线比例 |
| 系统性失败 | 在 ≥ 3 条路线上出现的同一种失败：不起步、持续压线、路口不转、原地打转 |
| 开环合理性 | 在 BehaviorAgent 驾驶的同一批帧上，模型 2 s 轨迹与 expert 未来的 ADE，和匀速 baseline 比 |

**通过**：平均 RC ≥ TFv6 A 臂同路线 RC 的 50%，没有系统性失败，且开环 ADE 好于匀速 baseline。
**不通过**的模型仍然可以参加后面的开环测量，但所有结果都要标注「domain 混杂」，不能拿来比较能力。
达到但只是勉强过线的，同样标出。

## 交付物

- wrapper 代码放在 `jevdrive/zoo/`（每个模型一个模块，接口统一：`reset(route)`、`step(sensors, ego) -> plan, control`），
  配套一个 recorder，能在 P5 的配对生成里同时挂上这些模型各自的传感器。
- 每个模型的传感器配置表（官方值、我们用的值、偏差），以及一张 CARLA 画面和它训练数据画面并排的对照图。
- 关卡结果表（16 条路线 × 模型），通过或不通过，以及失败模式的描述。
- 报告写在本文件的「结果」一节；小结果文件放 `research/results/zoo-in-carla/`。

## 预算

每个模型的集成大约 1 天（传感器 + 导航 + 输出转换），关卡闭环 16 条 × 5 个模型，在 GPU box 上并行跑约半天。
openpilot 三个模型共用一个 wrapper，只是换 checkpoint。总计 3–4 天。集成卡住超过 1 天的模型，写明卡在哪里，先做其他模型。

## 结果

跑完再填。
