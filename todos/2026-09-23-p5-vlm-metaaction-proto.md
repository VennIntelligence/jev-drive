# P5 原型：冻结 VLM 零样本给 meta-action，不用 CARLA

状态: running（预登记 2026-09-23，写于任何 VLM 回答被读之前）
主题: ../research/prediag-2026-09/README.md（P5 行）；决策背景是 ../research/decisions.md 第 25 条「第二个标签来源」

## 目标

第 25 条把「VLM 当慢老师、reaction decoder 当快学生」列成与 expert 重跑并列的第二个标签来源，开关是 P5 里
「VLM 零样本 meta-action」那一列的答对率。CARLA 的配对生成器排在 P4 之后，这里先在 Waymo E2E val 上用
logged future 当裁判做一个原型，只回答一件事：**在 pre-onset 帧上（车还没开始转、ego 运动还看不出意图的帧），
一个冻结的 Qwen3-VL 不经训练，能不能说出正确的离散反应？** 说得出，这条线值得接到 CARLA 配对上；
说不出，标签回到 expert 重跑。

meta-action（离散的驾驶意图标签，纵向一个、横向一个，而不是连续轨迹）是本文的输出形式。

## Setup

### 帧（已冻结）

从 P2/P3 共用的冻结子集 `p2p3_v1` 里取，只要 4 帧 × stride 2 的 clip 严格完整的帧（第 13 条）。
冻结文件 `$DATA_DIR/processed/waymo_e2e/subsets/p5vlm_v1.parquet`，
frame name 的 sha256 为 `a4e4b4393b572d9803d04783e58397d7a45fbd8fe3428f54e9b1fb5dd6f6959b`，共 3758 帧、478 个 sequence。

| 组 | n | 怎么取 | 用途 |
|:--|--:|:--|:--|
| pre_onset | 1458 | 全取（1510 帧里 clip 完整的） | 主判 |
| straight_yaw | 1458 | 按 pre_onset 的 v0 十分位逐档配平抽样 | 对照：直行帧上不能凭空编出反应 |
| turn_yaw | 400 | 随机抽 | 已经在转的帧，答案看得见，是 sanity 上限 |
| rater | 478 | 全取（479 帧里 clip 完整的） | 和 rater_best 的一致性 |

组之间有少量重叠（rater ∩ pre_onset 约 6 帧，第 3b 条）。pre_onset 再按 route command 拆成两行：
route 是 go straight 的 588 帧上，route command 基线在横向上注定答错，答对只能来自相机。

### 输入

和 P3(d″) 选定的 backbone 输入完全相同：三个相机（front、front-left、front-right），每个相机一段 4 帧、
stride 2（跨 0.6 s）、最老的在前的 clip，走 Qwen3-VL 自己的 video 通路，原生分辨率（每帧 6120 个 video token）。
`video_metadata` 按 5 fps 传进去，所以模型看到的时间戳是真实的 0.2 s 间隔，而不是默认的 24 fps。
另加一段紧凑的 ego 文字：当前速度和 1/2/3 s 前的速度、加速度、yaw rate、过去 3 s 的行进方向、route command
（Waymo E2E 给每个 planner 的导航意图，ego readout 的输入里本来就有它）。

### 输出和解析

一行严格 JSON：`{"reason": "<一句话>", "longitudinal": "<label>", "lateral": "<label>"}`。
reason 放在最前面，相当于一句话的 chain-of-thought，这是有意的设计选择，写死不再改。

| 轴 | 标签 |
|:--|:--|
| longitudinal（未来 3 s） | keep / slow / stop / accelerate |
| lateral（未来 5 s） | keep_lane / nudge_left / nudge_right / lane_change_left / lane_change_right / turn_left / turn_right |

解析是**严格**的：取第一个 `{...}` 做 `json.loads`，两个标签必须和上表逐字相同，不做修补、不做模糊匹配。
不合格的算答错，并单独报 invalid-JSON rate。greedy decoding，`max_new_tokens=96`，
prompt 的 sha256（用固定占位 ego 文本渲染）写进每个 run 的 `meta_<variant>.json`。

### Judge：从 logged future 自动打标签（阈值写死在 `jevdrive/waymo_p5vlm.py`）

同一个函数 `meta_labels` 给三样东西打标签：logged future（裁判）、rater_best 轨迹（rater 一致性）、
ego 历史的 CTRA 外推（一个基线）。所以它们是用同一把尺子量的。

- 纵向（3 s）：v3 = 2.5–3.0 s 的平均段速度。v3 < 1.0 m/s → stop（包括一直停着）；
  否则 v3 比当前速度 v0 低 max(1.5 m/s, 0.15·v0) 以上 → slow，高出同样多 → accelerate，其余 keep。
- 横向（5 s）：5 s 弦长 < 3 m → keep_lane（车几乎不动，没有横向动作）。否则取末端航向（最后 0.5 s 的位移方向，
  不足 1 m 时用 2 倍弦方位角，即等曲率近似）：|航向| ≥ 30° → turn；否则按 5 s 处的横向偏移，
  ≥ 2.0 m → lane_change，0.75–2.0 m → nudge，其余 keep_lane。
- lat3：把横向合成三类 keep / left / right（nudge、lane change、turn 按方向归并）。**这是主判的横向口径**，
  因为 pre-onset 问的就是「往哪边」。7 类横向只进 confusion matrix。
- 已知弱点：没有地图，**弯道跟车和换道在 judge 眼里是一样的**。末端航向落在 10°–30° 且不是 turn 的帧标为 ambiguous
  （pre_onset 22.5%、straight 2.8%、turn 24.5%、rater 11.9%），另报一行去掉它们的敏感性结果。

阈值只在 logged future 的标签分布上做过 sanity check（CPU run `waymo_p5vlm/select/20260923-121353`），
**没有看过任何 VLM 回答**：

| 组 | 纵向 keep / slow / stop / accel | lat3 keep / left / right | ambiguous |
|:--|:--|:--|--:|
| pre_onset | 721 / 346 / 36 / 355 | 23 / 573 / 862 | 0.225 |
| straight_yaw | 808 / 199 / 171 / 280 | 1194 / 111 / 153 | 0.028 |
| turn_yaw | 201 / 23 / 38 / 138 | 39 / 132 / 229 | 0.245 |
| rater | 182 / 62 / 130 / 104 | 288 / 86 / 104 | 0.119 |

直行帧 82% 是 keep_lane、只有 1.4% 被判成 turn，pre_onset 98% 是 left/right，说明 judge 在两端的行为是对的。

### 基线（都映射到同一个标签空间）

| 基线 | 纵向 | 横向 |
|:--|:--|:--|
| majority | 该组该轴的众数（组内样本内取，对基线偏乐观） | 同左 |
| continuation | 停着就 stop，否则 keep | keep_lane |
| ego kinematic | CTRA 外推轨迹过同一个 judge | 同左 |
| route command | CTRA 的纵向 | GO_LEFT → turn_left，GO_RIGHT → turn_right，其余 keep_lane |

每一行取所有基线里**最好的那个**做配对比较（事后挑最好的基线，对 VLM 是保守的）。

### 模型和 variant

| variant | 输入 | 作用 |
|:--|:--|:--|
| main | 三相机 clip + ego 文本 | 主结果 |
| text | 只有 ego 文本（明说看不到相机） | **视觉贡献的对照**：VLM 可能只是在复述 route command |
| front | 只有 front 一路 clip | 扰动一致性（flip 的替代品），可选 |
| shift1 | 三相机 clip 整体提前一帧（0.1 s），ego 文本不变 | 扰动一致性，可选 |

Qwen3-VL-4B-Instruct 先跑全部 3758 帧的 main 和 text。32B 用同一个 harness、同一个 prompt，
按 `RESOURCE_LEDGER.md` 只在 ≥ 72 GB 空闲时跑，帧数按 4B 的 profiling 结果决定（优先 pre_onset + straight）。

- 模型：Qwen3-VL-4B-Instruct、Qwen3-VL-32B-Instruct（bf16，`~/data/cache/huggingface`）
- 算力预算：4B ≤ 15 GB VRAM，时长在 profiling 后填；32B 见上

### 统计

准确率的 95% CI 按 sequence 重采样 bootstrap（1000 次，`traj.boot_ci`），和 ladder 一致。
VLM 对最好基线、VLM main 对 VLM text 都用同一帧上的配对差、同一个 sequence bootstrap。

## 步骤

- [x] `jevdrive/waymo_p5vlm.py` + `scripts/waymo_p5vlm.sh`：选帧、judge、生成、报告
- [x] 冻结帧集合，judge 阈值在 log 标签分布上 sanity check
- [ ] 4B profiling（随机 48 帧）：s/帧、峰值显存、invalid 率。**这一步只允许为格式问题（invalid JSON）改 prompt，
      不看准确率**；若改了 prompt，在这里记下改了什么和为什么
- [ ] 4B main + text 全量
- [ ] 4B front / shift1（可选，pre_onset + straight）
- [ ] 32B main（VRAM 允许时）
- [ ] 报告，结果填到下面

## 成功标准（跑之前写死）

要回答的是「这一列能不能当标签来源」，所以门槛是**提案**，由 lead 拍板。提议的门槛，四条全过才算「清过」：

1. **比基线强**：pre_onset 上 lat3 准确率，VLM main 减最好基线 ≥ +0.05，且配对 CI 下界 > 0。
2. **是视觉带来的**：pre_onset 上 lat3，VLM main 减 VLM text，配对 CI 下界 > 0。
   不过这一条，说明 VLM 只是在读 ego 文本和 route command，那它不是一个新的标签来源。
3. **不在直行帧上编反应**：straight_yaw 上 lat3 准确率不低于最好基线 0.05 以上（非劣）。
   第 25 条要求修正项在直行帧上为零，一个在直行帧上乱报反应的老师会直接污染这个约束。
4. **格式可用**：invalid-JSON rate ≤ 2%。

纵向准确率照报、带 CI，但不设门槛：pre_onset 的定义只关于横向，纵向在这里没有「先验失灵」的构造保证。

| 结果 | 读法 |
|:--|:--|
| 四条全过 | VLM 这条线留着，下一步把同一个 harness 接到 CARLA 配对上，用 expert 当裁判复测 |
| 1 过、2 不过 | VLM 在复述文本，不是视觉老师；这条线关掉（或者先去掉 route command 再测一次，看它自己看不看得出） |
| 1 不过（4B） | 4B 不够；若 32B 跑了且过，线留着但老师必须是 32B（成本进 todo）；32B 也不过，线关掉，标签回到 expert 重跑 |
| 1、2 过但 3 不过 | 老师会报反应但不会说「没事」，只能当 pre_onset 上的标签，不能当全集标签 |

## 结果

跑完再填。run dir 在 box 的 `$DATA_DIR/runs/waymo_p5vlm/`。
