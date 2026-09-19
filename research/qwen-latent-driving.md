# Qwen latent 里有没有驾驶智能

状态: 第一个研究主题，2026-09-19 开题。

## 核心问题

> Qwen 的视觉/语言预训练 latent，本身是否已经包含足够的 driving intelligence，
> 使得一个极小、非生成式的 decision head 就能读出未来驾驶行为？

不把 Jev-like 模型当成要训练的本体。候选 logits + softmax 的 decision 输出层社区已经做过
（LitJev、Jev Visual 等），不值得再证明一次。我们研究它前面的 z。

## 任务形式

过去的视频 → 几个 Jev-style fixed decision：

- 转向: left / straight / right
- 纵向: 减速 / 保持 / 加速
- 未来横向位移或曲率的类别（比如 2s 后 large-left … large-right 五档）

先只做 ego future maneuver，因为 ground truth 最干净。cut-in、pedestrian intention 这类 interaction
任务等这个问题回答清楚以后再做。

## 四个系统

| 实验 | 模型 | 在测什么 |
|---|---|---|
| A | Qwen-VL autoregressive prompt | 原始 VLM intelligence |
| B | Qwen-VL direct candidate logits | 不生成文字之后还剩多少能力 |
| C | Frozen Qwen latent + Linear/MLP | latent 本身有没有驾驶信息 |
| D | Frozen Qwen latent + tiny temporal Transformer | 缺的是不是 temporal dynamics |

C vs D 最关键：

- D ≫ C：Qwen 有视觉语义，但连续 dynamics 需要重新对齐。
- C ≈ D 且都好：multimodal pretraining 已经把大量 driving-dynamics 信息塞进 latent。
- A 好但 C/D 差：智能在完整的 transformer computation 里，抽 latent 的层选错了。

## Layer probe

从不同层取 z：vision encoder 输出、LLM early / middle / late。
猜测：越靠后越懂"这是什么"，越不记得"具体怎么动"。
如果成立，就支撑"VLM 不是没有 physical information，而是语言化过程中把它压掉了"。
这可能是论文的一部分。

## 评价指标

Accuracy / F1、NLL、Brier、ECE（在 val 上学一个 temperature T），以及 latency。
研究动机是：在 20–50 Hz 下还能保留多少 foundation-model intelligence。

## 必须有的 baseline（Claude 补充）

- **Majority / prior**：数据里绝大多数是 straight，不和它比，accuracy 没有意义。
- **Ego-state only**：只用过去轨迹、速度、加速度去预测未来 maneuver。
  未来 2s 的行为很大程度由当前运动状态决定，Qwen latent 必须**超过**这个 baseline
  才能说明它从画面里读出了东西。这是整个故事成立的前提。
- **小视觉 backbone**（比如 ImageNet ResNet/DINOv2 feature + 同样的 probe）：
  区分"Qwen 特有的智能"和"任何视觉 feature 都能做到"。

## 数据

- nuScenes mini（已在 box 上 `$DATA_DIR/datasets/nuscenes`）：10 个 scene，CAM_FRONT 404 个
  keyframe（2Hz），1938 个 sweep（~12Hz）。**只够 debug pipeline**，val 只有 2 个 scene，
  得不出任何结论。
- nuScenes trainval：850 个 scene，`/autodl-pub` 上已有，解压即可，不用下载。
  可以作为 mini 之后、Waymo 之前的第一个有统计意义的实验。
- Waymo E2E（正式实验候选）：8 cameras、10Hz、4s past + 5s future trajectory、
  velocity/acceleration、left/straight/right route command，专挑 long-tail 场景。
  4021 个 segment（train 2037 / val 479）。需要申请 license 并下载。

## Open questions

- z 怎么取：vision token mean-pool、最后一个 token、还是带 prompt 的某个 token？
  带不带 prompt 本身就是一个变量。
- 多帧怎么喂：多图输入 Qwen 一次过，还是每帧单独抽 z 再给 temporal model？
  两者测的东西不一样。
- 标签阈值（±5° yaw 等）怎么定，类别不平衡怎么处理。
- A/B/C/D 的 latency 口径怎么统一，才能公平比较。

## 参考

- Qwen3-VL-4B-Instruct: https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct
- Jev Visual（shared prefill → fork → candidate scoring）: https://github.com/hr98w/jev-visual
- LitJev: https://huggingface.co/spaces/multimodalart/jev-reproductions-tracker/discussions/2
- OpenJev（DiffusionGemma，需要 Blackwell，不适合 4090 起步）: https://github.com/razorback16/openjev
- nuScenes CAN bus expansion: https://www.nuscenes.org/tutorials/can_bus_tutorial.html
- Waymo E2E: https://waymo.com/open/data/e2e/
