# 配方登记表（recipe registry）

2026-10-12 写，覆盖决策 170–264。只汇总，不新增读数：每个数都来自 `research/decisions.md` 的行、`decisions/NNN.md` 页或注明的结果文件。
记号：d### = 决策号；CI 都是 95%（多数为按 log 聚类的 bootstrap，以原文为准）；「推」= 由记录的数算出、原文没写。

## 先读这几个词（每个只解释一次）

- **Cinque**：openpilot 出厂的视觉 encoder + temporal policy，冻结不训；**adapter**：接在它后面、我们训练的小网络（加 ego / 位姿 / 指令输入，op_parity）。所有配方只改 adapter 与 plan 通路。
- **配方**：一次训练的全部选择（训练行、损失项、输入标准、步数）；**serving 开关**：不重训、只改 driver 喂入或输出的处理。
- **hinge**：只在越线时才罚的损失。**可行驶 hinge**：plan 的车身足迹离可行驶区域边界不足 margin 时罚，λ 是它的权重。
- **navtest / navhard**：NAVSIM 开环榜（EPDMS 分，navhard 两阶段）；**HUGSIM**：3DGS 闭环榜（HD 分）；**WOD-E2E**：Waymo 开环榜（RFS 分）。
- **AlpaSim nuPlan**：比赛的闭环模拟器，公开 1491 个 scene（就是 navtest 的 log）；**private**：主办方不公开的正式评测集；**PAI**：AlpaSim 的另一条赛道（不提交，d261）。
- **决策**：driver 每 0.5 s 被调用一次、返回 4 s plan；公开 scene 原样 10 个决策，重建缓存后可跑 18 个（d262 ALPASIM-LONG）。
- **零分**：scene 得 0，分三类：at-fault 碰撞、offroad（出路）、corridor（离日志路径横向 ≥ 4 m）。**GT distance**：离日志路径的最大横向距离的均值。
- **held 460 / train 1031**：d259 起按 log 划分的公开 scene，train 出训练行，held 只作判定；它是行的留出，不是配方选择的留出。
- **on-policy 行**：driver 自己在闭环里开到的状态（离日志可能已偏）上取的训练行；**ON-NONE**：同起点同步数续训、不加这些行的对照。

## 1. 活着的配方

| 代号 | 一句话是什么（改了哪一部分） | 底座 | 最好的读数 | 当前用途 | 决策 |
|:--|:--|:--|:--|:--|:--|
| SH30-F | 强可行驶 hinge（λ 30、margin 0.5 m）的 openpilot 配方（损失） | P2（adapter，NAVSIM 输入） | navtest EPDMS 89.55，对 P2H10 +0.87 [+0.64, +1.15]，2 seed；navhard +1.83；NAVSIM v1 PDMS 89.35 | NAVSIM / navhard / HUGSIM 上的论文臂；AlpaSim 上已弃（见弱点） | 170, 172, 258 |
| P2H10-F | 可行驶 hinge λ 10（margin 0.3 m）（损失） | P2 | AlpaSim held 300 对 SH30 +0.0369 [+0.0087, +0.0668]；700 scene 0.9483（2 seed）；1491 scene 0.9196 | 所有 AlpaSim 读数的登记基线；P2H10S / YR 的底座 | 205, 212, 253 |
| P2H10S-F | P2H10 + 车身课 S：自己 plan 的扫掠足迹对车辆框、对道路车道栅格两个 hinge，梯度只看 plan 形状；加 hinge-only 离轨行（数据行 + 损失） | P2H10 | PAI 281 scene，4 seed S 对 2 seed base +0.1100 [+0.0760, +0.1458]；navhard 33.46 对 31.54，+1.91 [+0.13, +3.74]；navtest +0.38 [+0.20, +0.57] | 闭环训练 P 族底座；论文臂（车身课） | 232, 236, 247, 253, 255 |
| YR10m10-F | P2H10 + 10% yaw-rate 行：注入的 yaw rate 与航向偏置解耦的合成行（数据行） | P2H10 | **private 0.80**（提交 #2，PCS 1007，at-fault 0.529 km，GT 1.733 m）；公开 1100 scene 对 P2H10 +0.0095 [+0.0024, +0.0166] | **提交 #2 的 checkpoint（`YR10m10-F-s0`）**；最后名额的对照；Y 族底座 | 213, 216, 260 |
| SSELFnoL-ON | P2H10S-F 续训 10 000 步，每 batch 13 行换成 10 决策 on-policy 行，行上只加 S 的两个 hinge（数据行 + 损失） | P2H10S-F | held 460：0.9392 对 control 0.9248，+0.0143 [+0.0017, +0.0276]，零分 −7.0 [−13.2, −1.2]；18 决策 0.821，对 YR10m10-F +0.044 [+0.018, +0.068] | 闭环训练线的第一个正读数；最后名额候选（规则见 plan.md 2.3，见弱点） | 259, 262 |
| YS-L18 | 同上的 S-only，但行取自 18 决策 rollout，底座换成 YR（数据行 + 损失） | YR10m10-F-s0 | held 452、18 决策：0.838，对 ON-NONE +0.1446 [+0.1187, +0.1730]，对底座 +0.0529 [+0.0241, +0.0823]；at-fault 零分 23.5 对底座 43；GT 1.357 对 1.495 m | 第 2 轮起点（`YS-L18-s1`）；最后名额候选 | 264 补记 3（**未入 decisions.md**） |
| Rnom（serving） | driver 不信 session 上报的相机安装姿态，改用固定名义旋转（serving） | 任意 openpilot 学生 | 96 个公开 scene −0.003 [−0.034, +0.028]（免费） | 提交前的保险选项，待用户定 | 263 |
| PAI (a)+(b)（serving） | `JEV_VCONT` 速度连续 + `JEV_LEAD` openpilot 前车 MPC 作限速（serving） | P2H10-F 起 | PAI 60 scene × 2 seed：0.215 → 0.363，+0.148 [+0.061, +0.244] | PAI 的 served 配置（d247 等读数都用它）；PAI 不提交 | 226 |
| PAI_KEYWARP（serving） | PAI driver 用 2 Hz keyframe + warp 造 slot，复现训练输入（serving） | 任意 | 33 scene × 2 seed：on − off 四个 checkpoint +0.0125 [−0.107, +0.131]，分不开 | 默认关；只作失配测试 | 252b |

每个配方的要点（训练数据 / 损失 / 关键超参 / 怎么跑 / 弱点）：

**SH30-F**：navtrain 全量 103 k token，protocol W（warp 帧）。损失 = 模仿 + anchor 蒸馏 + 可行驶 hinge λ 30 / margin 0.5 m；10 000 步 × 128，seed 0 / 1。
训练 `experiments/op_parity/scripts/strong_hinge_chain.sh`（`pp_train.py --arm P2 --frames warp --hinge-lam 30 --hinge-margin 0.5`）。
弱点：AlpaSim 里在直路上航向漂移，强 hinge 是触发（d205；700 scene 0.9140 对 P2H10 0.9484）；> 45° 转不过去没变（d170）；WOD 无跨域收益（d172、d217）。

**P2H10-F**：同 SH30 但 λ 10 / margin 0.3 m（`pp_train.py --arm P2 --hinge-lam 10`，`pp_hinge_chain.sh`）；navtest 88.67（op_parity `results/hinge.md`，d170 的对照行）。
AlpaSim：`SH30_TAG=P2H10-F-s0 run.sh <dir> sh30`。弱点：放弃 SH30 的开环收益（d205）；它的 49 个零分全部在出事前就在 served plan 里（d227）；
无 hinge 的 P2-F 不比它好但它也只比 P2-F 高 0.0074（d212）。

**P2H10S-F**：P2H10 + agent hinge（shape-only）与道路车道栅格 hinge，作用在每个模仿行的 own plan 与 hinge-only 离轨行上（`ot1` 4 / `yr1` 4 / `bd4` 5 行每 batch，权重 w = 3）。
`experiments/body1/scripts/bd4_train.py`，或 `experiments/alpasim/scripts/joint_train.py train --ho ot1:4,yr1:4,bd4:5 --ho-w 3 --agent-lam 10 --shape`（逐位复现，d254）。
弱点：AlpaSim nuPlan 1491 上对 P2H10 只 +0.0047 [−0.0016, +0.0115]，corridor 零分最多（27.25，d253）；> 45° plan 外移、新增零分在 48–70° 右转（d233、d244）；
PAI 绝对分带 10 Hz slot 的 caveat（d247）。

**YR10m10-F**：P2H10 配方 + 每 batch 13 / 128 行 yaw-rate 行（缓存 `yr1`；40% 只有最近 0.5 s 的注入 yaw，60% 三段 random walk，±2.5° 截断）。
`experiments/alpasim/scripts/ot3_rows.py train --std navsim --ot-mass 0.10`；镜像 `jev-alpasim:yr10m10-f-s0-073dea33`。
弱点：未碰过的 part012–015 上对 P2H10 −0.0069 [−0.0110, −0.0016]，确认读数为否（d216）；18 决策上 at-fault 零分 42、三者最多（d262）；
private 0.80 与 go-straight 0.799 同分（d260）；对上报俯仰 2–3° 的误差很敏感（d263）。

**SSELFnoL-ON**：行池 `on1_train`（P2H10S-F s0 / s1 在 1031 train scene 上 10 决策滚出的 20 386 行），每 batch 13 行 on-policy、只给 S 两个 hinge。
`experiments/onpol_nuplan/scripts/on_rows.py` → `on_train.sh full on1_train` → `on_read.sh held`。弱点：2 seed、460 scene；登记判词整体是「分不开」，只有这一臂保有登记地位（d259）；
`-ON` checkpoint 见过 navtest log 的渲染，永不报 navtest / navhard；提交需在全部 1491 上重训；18 决策 GT 对 YR −0.038 [−0.153, +0.081]，按 plan.md 2.3 的 GT 规则还换不掉 YR。

**YS-L18**：池 `on1_L18Y`（YR10m10-F-s0 在 train 上 18 决策滚出的 17 306 行），从 YR10m10-F-s0 起续训 10 000 步，13 行 / batch，只给 S 两项（无 K、无 C）。
`experiments/onpol_long/scripts/ol_chain.sh`（补记 3 的 S 阶段），读数在 box `$DATA_DIR/runs/onpol_long/report/Y-S/report_YS.md`。
弱点：读数还没写进 d264（d264 仍写「S-only 拆分臂在跑」），本表的数直接取自 box 报告；两个 seed 同一起点；对 ON-NONE 的差被「续训本身伤底座」放大（ON-NONE 对底座 −0.092），对底座才 +0.053；
同一做法在 P 族（`PS-L18`）at-fault 零分 36 对 35.5，硬闸差半个 scene 不过。任务书说它「刚过线」，box 报告是三条线都有余量地过（零分 44 对 120），不是刚过。

## 2. 已死的配方

| 代号 | 一句话是什么 | 死在哪个读数上 | 为什么死 | 决策 |
|:--|:--|:--|:--|:--|
| AP2 / AB 输入标准（AP2-AB、AP2H10-AB） | 按 AlpaSim 的投喂方式重训：1–4 个 keyframe、冷启动 backwarp、AlpaSim 自己的 route command | 公开 1491 +0.010，private −0.16（两次提交的唯一差别） | 方向在 private 上反了；λ 10 上公开 +0.0023 [−0.0124, +0.0179] 本来也没过线 | 189, 212, 260 |
| APY10m10-AB（提交 #1） | AB 输入标准 + 10% yaw-rate 行 | private 0.640、at-fault 0.132 km、GT 2.29 m（公开 0.935） | 被提交 #2 的 0.80 取代；18 决策上零分 89、漂移最陡 | 216, 253, 260 |
| JAPS（APS） | JOINT-1 联合训练：AB 标准 + S + yaw-rate 行 | 零分对 P2H10-F +12.5 [−4.0, +29.5]，corridor +14.0 [+3.5, +24.5] | 三样成分负交互；没过登记线 | 254 |
| JAPSnoY | APS 去掉 yaw 行（= AB 标准 + S） | 均分 0.9360 最高，但 corridor +12.0 [+2.0, +22.0]，navtest −0.32 破护栏；4 seed 与 APY 持平 | 带着 AB 标准，而 AB 在 private 上是负的 | 254, 259, 260 |
| JAPSnoA | APS 去掉 AB 标准（= S + yaw 行，NAVSIM 输入） | 0.9193，与 P2H10-F 0.9194 一样，零分 77 | 无增益 | 254 |
| 离轨行 ±0.5 m（OT10a05-F；λ 30 上的 OT30、APO-a05m10） | 10% 静态重投影的侧偏 ±0.5 m / ±2° 训练行 | λ 10 上 −0.0007；OT30 在 1491 上 +0.0170 但底座是 SH30；APO-a05m10 0.9409 低于 P2H10 | 收益只在强 hinge 底座上出现，而强 hinge 在 AlpaSim 已弃 | 198, 208, 210, 212 |
| 离轨行 ±1.5 m（OT10a15-F） | 同上，±1.5 m / ±5° | −0.106，零分 98.5，航向 sd 12° | 闭环里发散（navhard 却最高） | 212 |
| ENS-OT30 | OT30 两个 seed 的 plan 平均 | −0.0022 [−0.0097, +0.0045]，线 +0.008 | 两 seed 几乎同一策略，best-of-two 只是噪声最大值 | 211 |
| P2-F（无 hinge） | 去掉可行驶 hinge | 1100 scene 对 P2H10 −0.0074 [−0.0146, −0.0010] | 不比 λ 10 好 | 212 |
| corridor hinge K（PSKC-L18、YSKC-L18） | on-policy 行上加「离日志路径超出 1.5 m 死区」的 hinge | P 族 0.590 对 ON-NONE 0.771，at-fault 零分 97.5 对 35.5；Y 族 at-fault 80 对 79.5 | 18 决策下把车拉出路（P 族 offroad 零分 79 对 24）；训练行上降了，held 上不迁移 | 264 |
| lane-centre hinge C | 车道中心偏移 hinge | 训练前被标签闸排除 | 日志自己的未来在 46% 的行上违反它 | 264 |
| L 纵向课（SSELF-ON） | 前车接触、必停点、入弯速度三个 hinge + 进度护栏 | 加 L 对不加 −0.0237 [−0.0380, −0.0106]，零分 +7.0 | 抵掉 S 的增益；pilot dev-ADE 闸也没过 | 259 |
| ON-IMIT / ON-BOTH | 同样 on-policy 状态上用回到日志的恢复轨迹作模仿目标（DAgger 式）/ 模仿 + S + L | at-fault 零分 20 / 15 对 control 12.75，硬闸不过 | 模仿标签增加 offroad（18 对 12）；PAI 上 ON-BOTH 停在 pilot 闸 | 259, 261 |
| 不加行的续训（ON-NONE） | 同起点同步数再训 10 000 步，不加行 | 18 决策上对底座 −0.028（P）/ −0.092（Y） | 新优化器下续训本身伤底座（10 决策时持平：−0.0036） | 259, 264 |
| body1 刹停（4.1） | 接触预测头在 driver 里触发刹停 | −0.0067 [−0.0159, +0.0011]，偏慢 43 → 50 | 7 个误刹停抵掉 2 个救回 | 224 |
| body1 横向重规划（4.2） | 被 flag 的决策做一次横向 ramp | 零分 10 → 11，−0.0043 | 偏一次后不回线；一半基线零分从未被 flag | 225 |
| body1 loss 臂全梯度（`P2H10B`，w = 3） | S 的前身：hinge 沿 heading 也有梯度 | 700 × 2：−0.0046 [−0.0131, +0.0052]，偏慢 217 → 277 | agent hinge 向后拉，丢 progress；改成 shape-only 后成 P2H10S | 229, 230 |
| P2H10R（路线 hinge） | S + 到日志路径 2.5 m 死区的 hinge | navtest > 45° 外偏 2 m 以上 138 对 base 110 | 管道够不着外移，够得着的 1.5 m 破 dev ADE | 234 |
| P2H10SM（日志一致 margin） | S 的 margin 封顶在日志自身净空 − 0.05 m | 142 对 110（线 122） | 固定 margin 不是外移机制 | 257 |
| body1 stage 2（CARLA 行） | CARLA 里 student 在环生成行 | P2H10-F 在 CARLA 87% 的 tick 站着 | pre-check P0 关闭，未建 | 231 |
| selector 门 B | 冻结特征上的转弯候选选择器，只在 plan 转角 ≥ 20° 时用 | navtest +0.53 [+0.24, +0.85]；HUGSIM +0.002，WOD −0.0004 | 开环收益在闭环与 WOD 都不兑现；plan.md 第 5 节停 | 193, 194, 195 |
| left-hand-traffic flag | 告诉 driver 左行 | 220 个新加坡 scene +0.012 [−0.014, +0.034] | driver 不知道国家，不能全局开（`experiments/alpasim/results/aud1_lht_singapore.md`，无单独决策号） | — |
| WA-Cf memory | WA-JEPA 的 encoder 作 adapter memory | pilot navtest +0.95 | 「不借别人的强模型」（plan.md 0.3），只留参照行 | 160 |
| SH30_MOTION | driver 输入侧的 yaw 阻尼 | 去掉 M 类零分但转弯 −0.0014 | 伤转弯 | 205 |
| FIX1 nuPlan 开关 | (a) 速度连续、(b) 前车限速、(c) 底模速度剖面 | nuPlan 700：+a −0.007、+b −0.086、(c) −0.08 到 −0.15 | 九臂全负；只在 PAI 有用（见活表） | 226 |
| TR1 / LG | 训练里去掉 adapter 的 ego-only 速度先验 | AlpaSim 700 −0.0566 [−0.0804, −0.0315]，navtest −1.85 | 两头都输 | 221 |
| λ 30 / margin 1.0 | 更宽 margin 的强 hinge | navtest +0.02，直行 −0.54 | 已饱和 | 172 |
| 停车门训进配方（P2HGA） | 低速时 adapter 偏置置零，训练里也这样 | navtest −0.21，HUGSIM −0.020 | 起步先验只是搬家 | 174 |
| 几何 tokenizer memory | 真值 SDF / agent 占用编码进 memory 通道 | 被读也只值 +0.13 [−0.04, +0.29]，上限 < +0.3 | 不是 WA-Cf 增益的来源 | 197, 200 |
| HEAD1b 转弯描述头 | heading head 经 memory 喂回 / 作辅助 loss | −0.03 [−0.21, +0.16] / +0.04 [−0.04, +0.12] | heading 误差降了，出界率不动 | 243, 245 |
| FLOW1 | flow-matching plan head | > 45° 出界 −4.98 pp，但全板 EPDMS −1.10 | 直行与 EP 退步 | 246 |
| VT 续训视觉 | SH30 上续训视觉权重（分支 / 原地 / 三视图） | 分支 A 对 A0 −0.38 [−0.71, −0.05]；R 从头训 RB −1.01 | 没有臂不付代价地降 > 45° 出界 | 248 |
| 侧视图（EYES） | 三视图冻结 token bank 进联合训练 | 叠 S 时 > 45° 切内侧 +1.02 [+0.19, +1.91] | 侧视图线关闭 | 255 |
| stateless HUGSIM serving | HUGSIM 上按训练协议喂帧 | HD −0.047 [−0.104, +0.004] | 帧 / 时钟失配不是限制项 | 251 |
| op_lead（HUGSIM） | openpilot 前车通路接进 HUGSIM 闭环 | HD −0.005 [−0.068, +0.049] | 前车碰撞修好，起步停滞 1 → 14 | 239 |

## 3. 容易混的名字

- **P2**：op_parity 的第 2 臂，adapter 加 ego / 位姿 / 指令输入；**AP2**：P2 按 AlpaSim 输入标准重训。**APY** = AP2 + yaw-rate 行；**APO** = AP2 + 离轨行。
- **H10 / SH30 / λ**：H 后的数是可行驶 hinge 的 λ；H10 的 margin 0.3 m，SH30（strong hinge）是 λ 30、margin 0.5 m。λ 也用于 agent hinge（S 里是 10）。
- **10m10**：YR10m10 = yaw-rate 行、λ 10、行占 batch 的 10%（m25 = 25%，13 / 128 对 32 / 128）。**OT10a05 / a15**：离轨行、λ 10、幅度 0.5 / 1.5 m；**OT30** = 离轨行在 λ 30 上。
- **-F / -P（后缀）**：-F = 全量训练（navtrain 全量，10 000 步 × 128）；-P = pilot 规模（如 `P2H10SM-P-s0`）。注意不要和下面的族前缀 P 混。
- **-AB**：AP2 预登记里的 AB 臂，m < 4 个 keyframe 时用 backwarp 规则训练冷启动；带 -AB 的都是 AlpaSim 输入标准。
- **s0 / s1 / s2 / s3**：训练 seed。「4 seed」读数的 s2 / s3 是后补的。
- **S / K / C / L**：S = 车身课两项（扫掠足迹对车辆框、对道路车道栅格，只看 plan 形状）；K = corridor hinge（离日志路径 1.5 m 死区外罚）；
  C = 车道中心偏移 hinge（被闸排除）；L = 纵向课（前车 A、必停点 B、入弯速度 C、进度护栏 P；L 里的 C 和 hinge C 不是一回事）。
- **SKC**：onpol_long 臂名，计划是 S + K + C，C 被排除后实际是 S + K。**P2H10S / P2H10B / P2H10R / P2H10SM**：车身课的 shape-only / 全梯度 / 路线 hinge / 日志 margin 变体。
- **-ON**：10 决策 on-policy 行续训（onpol_nuplan）；**-L18**：18 决策 on-policy 行续训（onpol_long）。**-L18 的 P / Y 前缀**是底座族：P = P2H10S-F，Y = YR10m10-F。
- **NONE / SELF / IMIT / BOTH / SELFnoL**：行上不给标签（只续训）/ 自身后果 hinge（S + L）/ 模仿恢复轨迹 / 两者 / 只有 S 不要 L。
- **JAPS / noY / noA**：JOINT-1 一个 trainer 的联合配方（AB 标准 + S + yaw 行）及去掉 yaw 行 / 去掉 AB 标准的 drop-one 臂。
- **YS2 / YS2NONE**：第 2 轮从 `YS-L18-s1` 出发的臂与对照（未训）。

## 4. 下一步会新增的（全部未读）

- **第 2 轮闭环训练**（onpol_long 补记 5，`ol_chain.sh submit-r2`，工作区未提交）：从 `YS-L18-s1` 起，在 train 上滚 18 决策建池 `L18YS`，训 `YS2-L18` / `YS2NONE-L18`，读 held 与 1491 个 10 决策确认读数。**未读。**
  另有 onpol_nuplan 第 2 轮（`YSELFnoL-ON` / `YNONE-ON`，10 决策）池已建、未训。
- **PAI 迁移读数**（onpol_long 补记 1、4，Tokyo，281 scene）：队列 Y 底座 → Y ON-NONE → Y SKC → P 底座 → P ON-NONE → 过线的 S-only 臂（YS-L18 符合）。**未读。**
- **三臂判别实验**（`research/next-round/closed-loop.md` 第 7 节）：同样的 on-policy 行与四项后果，(a) 作输出 hinge、(b) 作前视支路的预测目标、(c) 两者、(b0) 支路无目标；
  判「动作选错」还是「离轨状态上看错」。预算约 15 + 5 + 6 card-h。**未读、未开始。**

## 来源之间的不一致

1. **第 262 号用了两次**：decisions.md 里 PROXY（公开 scene 当代理）与 ALPASIM-LONG（18 决策）都编为 262；本表引用 262 时两者都可能，按内容区分。
2. **SSELFnoL-ON 的 10 决策零分**：d259 / onpol_nuplan 写 14.5，alpasim_long held 表写 20.5；均分相同（0.939）。推：差的 6 个是路线折返、从不起 session 的 6 个 scene，后者算作零分，未核实。
3. **YS-L18 / PS-L18**：box 报告（2026-10-12 06:27）已出，d264 与 onpol_long README 仍写「在跑」；ol_chain.sh 的补记 5 只在工作区，未提交。
4. **P2H10S-F 的 1491 均分**：d253 为 4 seed 0.9243，JOINT-1 表为 2 seed 0.9226；种子数不同，不是矛盾。APY10m10-AB 公开均分 0.9335（2 seed，d253）、0.935（d260 取整）、0.9326（4 seed，d259）同理。
5. **`SH30_COMPILE`**：d226 写「镜像默认 `SH30_COMPILE=0`」，alpasim README（LAT1）写 box driver 默认编译、`=0` 用于复现旧 run；两处说的是不同的运行环境，但没有一处明说镜像为何不同。
