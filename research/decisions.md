# 决定记录

跨 session 的共同决定都记在这里，一条一个小节。多个 Claude session 同时在这个 repo 上工作，
口头达成的东西不落盘就会丢，或者被下一个 session 用不同的假设覆盖掉。

每条写四件事：**决定**、**理由**、**状态**、**怎么才能定下来**。

状态只有两种：

- **待定**：已经按它执行，但支撑它的证据还不够，随时可能翻案。所有下游结论都要跟着标注不确定。
- **已确认**：证据够了，可以写进论文。改它需要新的证据，并在这里记下改动。

---

## 1. 论文框架：视觉是在 ego-state prior 之上的增量

**决定**：主张写成「在一个很强的 ego-state prior 之上，视觉带来了多少增量」，不写成「VLM 会开车」。
每张表里 ego-only 单独占一行；另外专门划一个 pre-maneuver-onset 子集（车还没开始转的时刻），
因为 prior 在那里最弱。

**理由**：nuScenes trainval 上，只用 11 维 ego 运动状态的 NLL 是 0.310，最好的冻结视觉 feature 是 0.400，
两者拼接 0.308，视觉在全体样本上几乎没有增量。planner 的 ADE 差距更大：ego-only 约 0.8 m，
视觉 feature 3.3–4.3 m。但在 hard subset 上，ego probe 的 turn recall 塌到 0.010，
Qwen 中间层是 0.246。详见 [qwen-latent-driving.md](qwen-latent-driving.md)。

**状态**：待定。

**怎么才能定下来**：需要 Waymo 上重复出同样的模式（ego-only 强、pre-maneuver-onset 子集上视觉有增量），
并且 nuScenes 那个 0.192 → 0.185 的增量要有 scene-level bootstrap CI 支撑。
现在的差距很小，还没做 CI，有可能不显著。

---

## 2. 指标：RFS 优先，ADE 并列汇报

**决定**：Waymo 上的开发循环盯 RFS（Rater Feedback Score，官方指标，把预测和人工评分过的轨迹比），
ADE 只作为并列数字报出来。

**理由**：官方 ADE 是对着评分最高的 rater 轨迹算的。logged future 本身得分 2.63 m，
而公开最好的 test ADE 是 2.65 m，说明 ADE 已经饱和，只用 ego 的 baseline 也能逼到一米以内。
另外约一半的 ego-only 预测落在所有 rater trust region 之外，被罚到 4.0，RFS 那里才有空间。
`jevdrive/waymo.py` 里有官方 RFS 的 bit-exact 移植。

**状态**：待定。

**怎么才能定下来**：2.63 m 这个数字目前是在**仅有的 68 帧 val** 上算的，样本很小。
b5 会在 val 下到 46 个 shard（约 240 帧）时给出更新的数字。如果那时 logged future 的得分明显偏离 2.65，
「ADE 已饱和」这个前提要重新判断，指标策略也要跟着改。

---

## 3. Waymo 上的 pre-maneuver-onset 子集

**决定**：Waymo 版本的 hard subset 用 past-state window 定义（当前 yaw rate 低，但未来要转），
和 nuScenes 的 `jevdrive/labels.py` 保持同一套定义，不用 nuScenes 的 CAN 数据，
这样两个数据集的表格能直接对齐。

**理由**：两边的子集定义不一致的话，跨数据集的对比就没有意义。

**状态**：待定（代码已进 `14944c7`，阈值和产生的样本数还没核对过）。

**怎么才能定下来**：核对 Waymo 上这个子集的样本数和转弯比例，确认和 nuScenes 的 4% 是同一量级；
如果差太多，说明阈值需要按数据集重新标定，那就要在论文里写清楚两边用了不同阈值。

---

## 4. 分辨率默认用 800 px

**决定**：抽 frozen feature 默认用 800 px 宽的输入，不用 1600 px 原分辨率。

**理由**：nuScenes trainval 上两者最好的 NLL 差不到 0.006，hard subset 上完全相同（0.213），
而 800 px 抽 feature 快 4.6 倍（30.8 对 142.4 ms/frame）。

**状态**：待定。

**怎么才能定下来**：这个结论来自转向三分类。Waymo 的轨迹任务、以及需要看清远处小目标的场景，
可能对分辨率更敏感。在 Waymo 上做一次同样的两分辨率对比即可确认。

---

## 5. 取中间层

**决定**：frozen feature 取 Qwen3-VL 解码器的中间层（nuScenes 上最好的是 L20–L24，共 36 层），
pooling 用 image-token mean，不用 last token。

**理由**：转向 probe 的 layer 曲线是清楚的倒 U 形，L20–L24 最低，到 L36 退回早期层水平。
见 [qwen-latent-driving.md](qwen-latent-driving.md) 的图。

**状态**：待定。

**怎么才能定下来**：planner 的 layer curve 上，这个倒 U 形要弱得多（轨迹回归上 L13–L22 只是略好）。
所以「中间层最好」在转向分类上成立，在轨迹回归上还不确定。需要 Waymo 的 layer curve 确认。

---

## 6. 下载顺序和带宽

**决定**：box 的下载串行执行，顺序是 navsim → HF 模型（Qwen3-VL-32B、AutoVLA）→ Waymo val → Waymo small。
Waymo train 要等 val 完成后再确认，test 最后（提交限制是每 30 天 6 次，不急）。

**理由**：box 的总下行带宽只有约 12–18 MB/s，所有任务共享。并行跑的时候 Waymo 只有 1.3 MB/s，
ETA 336 小时；串行之后单个任务能拿到约 10–16 MB/s。

**状态**：已确认（队列在 box 上 `$DATA_DIR/tmp/dlq.sh`，窗口 `jev:dlq`）。

---

## 7. 谁改哪个文件

**决定**：`jevdrive/plots.py` 里 `layer_curve` 和 `k_sweep` 归 planner 那条线维护，
`probe_layer_curve` 归 probe 这条线。改别人的函数之前先打招呼。

**理由**：多个 session 同时编辑同一个文件会互相覆盖，git 层面看不出冲突。

**状态**：已确认。
