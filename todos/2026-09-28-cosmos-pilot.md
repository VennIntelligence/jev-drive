# Cosmos-Transfer2.5 把 CARLA 配对重画成真实感视频：pilot（10 对）

状态: registered（判据写于任何 Cosmos 输出之前，2026-09-28 13:20 CST）
主题: [research/survey-counterfactual-video-gen.md](../research/survey-counterfactual-video-gen.md) §6；[research/decisions.md](../research/decisions.md) 第 32、42、44、48 条

## 目标

adapted openpilot 要在 CARLA 配对数据（同一世界有 / 没有 hazard 行人，expert 动作标签来自 CARLA）上训练。用户的原则：模型真学会了，就应该在 sim 和 real 里都能开。
这个 pilot 回答一件事：**Cosmos-Transfer2.5（NVIDIA 的 ControlNet 式视频 sim2real 模型）把一对 x⁺ / x⁻ 各自重画之后，这一对唯一的差别还是不是那个行人**——行人要留下来（x⁺ 里认得出来、x⁻ 里没有凭空冒出人），行人以外不能多出新的差别（风格、光照、纹理在两边不同，就是给模型的捷径）。

名词：
- **x⁺ / x⁻**：P5 v1 的配对世界，x⁻ 把 hazard 行人藏到地下，其余 XML、TM seed、expert 全同（第 32 条）。
- **控制输入（control）**：Cosmos 生成时逐帧跟随的结构信号，这里全部由 CARLA 渲染给出：edge（边缘图）、seg（实例分割着色图）、depth（深度图）。
- **Edge Distilled**：Cosmos-Transfer2.5 唯一的蒸馏变体，4 步采样、严格 93 帧；seg / depth 只有 35 步的 base 模型（官方表：同卡 7.7 倍慢）。
- **噪声地板**：同一个成员用同一 seed 再画一遍（确定性地板），以及换一个 seed 再画一遍（seed 间距）；配对差要拿它们当尺子。

## Setup

- **配对**：P5 v1 BehaviorAgent 集，行人 4 个 family，小地图（Town01–07、11）上全部 12 条行人路线里取 10 条，seed 0，每条一对（`jevdrive/cosmos_pilot.py select`，表 [research/results/cosmos/pairs.csv](../research/results/cosmos/pairs.csv)）。
  窗口 = 两个 ego 最后一个共同 tick 之前的 93 tick（4.65 s，20 Hz）：窗口内 ego 逐 tick 相同，两边的差只有行人。
- **重渲染**：P5 录制器原样重开这 20 个世界（同 XML、同 TM seed、同 BehaviorAgent + TFv6 shadow），在窗口内额外挂一台 openpilot 式前视相机（位置与 P5 的 Waymo 前视相机相同，车顶、离地 1.81 m；1280 × 704，水平 FOV 64°，覆盖 openpilot road 与 wide 两个 model frame），
  20 Hz 存 RGB、depth、instance segmentation（`scripts/cosmos_pair_agent.py`、`scripts/cosmos_gen.sh`）。使用前逐 tick 核对 ego 位姿与 P5 v1 原录像一致（差 < 1 cm）。
- **Cosmos**：Cosmos-Transfer2.5-2B（权重来自 ModelScope 镜像 `nv-community`，文件大小与 HF 一致；Reason1-7B 文本编码器的 sha256 与 HF 固定 revision 逐个一致），envs/cosmos-transfer（官方 lock，cu128 / torch 2.7），GPU 1。
  两个成员同一 prompt、同一 seed、同一设置；不给 image context（风格参考帧），guardrail 关。
- **控制变体（在第 1 对上选，选完冻结）**：(a) Edge Distilled，edge 由 Cosmos 在 CARLA RGB 上现算 Canny；(b) Edge Distilled，edge 由 CARLA 实例分割与深度的边界给出（不带 CARLA 纹理）；(c) seg base（35 步），实例分割按实例 id 固定着色（两成员同色）。
  选法：第 1 对上检查 1 与 2 都过的变体里行人召回最高的；并列取便宜的。第 1 对的数字照报，但在 10 对的汇总里单列。
- **检测器**：YOLO26x-seg 640 fp16（第 45 条），COCO person，conf ≥ 0.25。
- **openpilot**：Cinque，20 Hz 原生逐帧 step（不 hold），从零状态起跑；road / wide model frame 由这台单相机按 camgeom 渲染；比较只用第 40–92 帧（前 2 s 是 warm-up）。
- **算力**：GPU 1 独占（scheduler 行 `cosmos-pilot`，CPU 24–47）；CARLA 1 个 server（index 110）约 1 h，之后只有 Cosmos。

## 步骤

- [ ] 下载权重、建 env（进行中）
- [ ] select 10 对，写窗口
- [ ] CARLA 重渲染第 1 对 → 核对确定性、看控制视频
- [ ] 第 1 对上跑三个变体 + 噪声地板，按上面的选法冻结变体
- [ ] 其余 9 对，同一变体；每对 x⁻ 再画两遍（同 seed、换 seed）
- [ ] 检查 1–4，WebP 并排图，结论

## 成功标准（跑之前写死）

GT：hazard 行人的逐帧 mask 来自 CARLA 实例分割（hazard 的 3-D 框投影到图像里、框内行人类像素的主实例），「可见」= mask ≥ 300 px（1280 × 704）。
走廊：ego 在 dense route 上前方 0–40 m、左右各 3 m 的地面带，投影成图像多边形；检测的接地点（框底边中点）落在多边形内算「在走廊里」。

1. **行人保真**（主判：所有 10 对合并；另报 near = mask ≥ 2 000 px、far = 300–2 000 px）
   - x⁺ 召回：在 GT 可见帧上，Cosmos x⁺ 里有一个 person 检测与 GT 框 IoU ≥ 0.3 的帧比例 R_T；同一检测器在 CARLA 原图 x⁺ 上的 R_C 作参照。
     **过：R_T ≥ 0.9 · R_C 且 R_T ≥ 0.70**；且可见帧 ≥ 10 的每一对 R_T ≥ 0.5 · R_C。
   - x⁻ 幻觉：Cosmos x⁻ 里走廊内、且不压在任何 GT 行人像素上的 person 检测，出现这种检测的帧比例 H_T；CARLA 原图 x⁻ 上的 H_C 作参照。
     **过：H_T ≤ 2% 且 H_T ≤ H_C + 1 pp。**
2. **没有风格捷径**（mask 区 = 逐帧 hazard mask 与其投影框的并集，膨胀 24 px；x⁺ / x⁻ 任一边有 hazard 的像素都算）
   - 像素：mask 区外的 PSNR；LPIPS（AlexNet）在「x⁺ 的 mask 区填入 x⁻ 的像素」的合成图与 x⁻ 之间算，逐帧取中位数。
     对照：同 seed 重画的 x⁻ 对 x⁻（确定性地板）、换 seed 的 x⁻ 对 x⁻（seed 间距）、CARLA 原图配对（原图地板）。
     **过：配对 LPIPS ≤ 1/3 × seed 间距 LPIPS，且配对 PSNR ≥ 28 dB。**
   - 特征：openpilot `temporal` 的 d = 1 − cos，配对用上面的 mask 区合成图（mask 外的差），对照同上。
     **过：配对 d ≤ max(确定性地板 d, 1/3 × seed 间距 d)。** 整帧配对 d 与 CARLA 原图配对 d 并排报，不进判据。
3. **时间稳定**
   - 闪烁：灰度二阶时间差 E|I₍t+1₎ − 2 I₍t₎ + I₍t−1₎| 在 mask 区外取均值，Cosmos 对 CARLA 原图的比值。**过：中位比值 ≤ 1.5。**
   - 开起来一样：openpilot 在 Cosmos x⁻ 与 CARLA 原图 x⁻ 上逐帧比较：plan 2 s 处速度差 |Δv|、2 s 处横向位置差 |Δy|、lead prob > 0.5 的一致率。
     **过：|Δv| 中位 ≤ 0.5 m/s 且 p90 ≤ 1.5 m/s；|Δy| 中位 ≤ 0.3 m；lead 一致率 ≥ 85%。**
4. **成本**（本卡 RTX 6000D）：每个 93 帧 clip 的秒数（不含模型加载）、峰值显存；2 000 对 = 4 000 clip 的 GPU·h（另报 CARLA 重渲染的 server·h）。
   **预算线：≤ 250 GPU·h**（全箱 7 卡约 1.5 天）。

**判决**：1、2、3 全过且 4 在预算内 → go；1 或 2 不过 → no-go（这一对的差已经不止行人，数据不能用来训配对差分），写明要改什么；只有 3 或 4 不过 → go with changes。

## 结果

跑完再填。run dir：box 上 `$DATA_DIR/runs/cosmos/`。
