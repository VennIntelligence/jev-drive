# 历史帧质量是否限制 openpilot 在 NAVSIM 上的分（lane C，2026-10-04，预登记）

写于任何本 lane 的分数之前。已读到的只有数据侧事实（下面「数据」一节）。

## 问题
1. 出货的 GIMM 合成历史离真实帧有多远？真实中间帧作为上限，GIMM 还差多少 PDMS？
2. 历史质量阶梯：hold（无合成）< CPU warp < GIMM（出货）< 真实帧，各差多少；能否找到一个可部署的、比 GIMM 好的插帧。
3. navhard stage 2 的合成视图历史由榜单生成：哪些能改、哪些不能改。

## 数据（已核实，未看任何分数）
- NAVSIM / OpenScene 的 CAM_F0 只有 2 Hz（日志帧间隔 0.500 s，sensor_blobs 只有关键帧）。
- nuPlan v1.1 原始数据的 CAM_F0 是 10 Hz（一个 test 日志的 DB：730 张 / 73 s，间隔 100 ms），公开 S3 可按字节区间读：
  `nuplan-v1.1_test.zip`（每日志一个 sqlite，给图像时间戳）和 `sensor_blobs/test_set/nuplan-v1.1_test_camera_{0..11}.zip`（实为未压缩 tar）。
  navtest 与 navhard stage 1 都来自 nuPlan test 日志，所以两者都能拿到真实中间帧；navtrain 来自 trainval，本 lane 不取。
- Cinque 在 t0 只读 4 个关键帧 + t0 − 0.2k（k = 1, 2, 3, 4, 6, 7）这 6 个中间帧（op_lb `SYN_T`）。真实臂就是把这 6 帧换成
  nuPlan 里离 T0_cam − 0.2k 最近的 CAM_F0 图（T0_cam = t0 关键帧图像自己的时间戳），关键帧不动，渲染与关键帧同一个
  `OpenpilotMaps`（t0 的 CAM_F0 标定）。检查：关键帧在 DB 里的时间差应为 −0.5 / −1.0 / −1.5 s（±15 ms），中间帧时间误差
  ≤ 15 ms；抽 40 个 t0 关键帧与归档里的同名图逐字节相同。不满足就停下报告。

## 样本
- S_nt：navtest，按种子 20261004 打乱 147 个日志，每日志取 ≤ 20 个 token，取满 500 为止：26 个日志、504 个 token
  （`runs/op_lb/hq/sel.json`）。下载拿不到的 token 剔除并报告数量。
- S_nh1：同 26 个日志里全部 navhard stage-1 token（102 个），只做描述性读数。
- 这是筛查规模（「小样本先行」）：真实臂需要额外数据，不是可提交的做法，只作上限，不进 headline。

## 臂（全部：原生 Cinque、TensorRT、desire none、0 状态、31 步 20 Hz、adapter base，与出货管线相同）
| 臂 | 6 个中间帧 |
|:--|:--|
| hold | 最近一个 ≤ t 的关键帧（`synth_cpu hold`） |
| warp | CPU 自车运动 warp，真实高度（`op_lb run --vcam 0`，即第 104 条的 vh187 对照） |
| gimm | 出货 GIMM（`lb_navtest/gimm.npy` 的同一行，不重算） |
| real | nuPlan 10 Hz 真实 CAM_F0 |
探针（不是规则）：gimm、real、warp 各加 `rotL` / `rotR`（全部 9 个历史帧注入 ±10°/s 假偏航，第 92 条 E1），G = (ψ3s(L) − ψ3s(R)) / 2。

## 读数
- 官方 v1 PDMS（navtest metric cache，TOKENS_FILE 限定 S_nt），逐 token 配对差，token bootstrap 10 000 次 95% CI；
  按日志的 cluster bootstrap 作敏感性。
- 解释性读数（每臂）：起步计划速度比 pv0 / v（v > 3 m/s）；车道宽比（第 104 条 sections，模型车道宽 / 地图）；
  计划相对 real 的 4 s 位置差与 3 s 朝向差；G 按速度档（stop < 0.5、0.5–3、3–8、> 8 m/s）；
  中间帧对 real 的 Y 通道 PSNR（road / wide）。
- S_nh1：gimm 与 real 的计划差、同上解释性读数；若评分器能只跑 stage-1 子集，再给 stage-1 EPDMS 配对差（描述）。

## 判读线（现在定）
- **H1（历史质量是否限制分数）**，看 real − gimm：
  - 「限制，值得追」：点估计 ≥ +1.0 且 CI 下界 > 0；
  - 「不限制（在 1 分量级上）」：CI 上界 < +1.0；
  - 其余为「小 / 未定」，照实报。
  预测（写在数之前）：+1 到 +4。依据：WOD 上 GIMM 收回 hold→real 的 89%，NAVSIM 上 hold→GIMM 是 33 分，若同比例，剩约 4 分；
  但第 104 条显示 NAVSIM 的尺度错误把计划压慢，可能让历史精度不那么要紧。
- **阶梯**：预测 hold ≪ warp < gimm ≤ real；gimm − warp 应与第 104 条的 +1.83 同号（不同号就报告样本差异）。
- **H2（合成历史放大偏航跟随，第 92 条限定）**：3–8 m/s 档 G(gimm) − G(real) ≥ 2° 且 CI 下界 > 0 判「放大」；
  |差| < 1° 且 CI 在 ±2° 内判「不放大」。
- **可部署候选**：只有 H1 判「限制」时才做。候选 C1 = 在原始 CAM_F0 分辨率上做 GIMM 插帧再渲染（模型帧之前插帧，
  光流在更大、更清晰的视野上估计）。navtrain lb_navtrain 先 300 token pilot、再 3 000 token 筛查，过线（对 GIMM 配对 Δ > 0
  且 CI 下界 > 0）才跑 navtest 全量与 navhard 两阶段全量，与出货配对。H1 不判「限制」时不做，写明原因。
- 本 lane 之后任何新加的臂都标「事后」。

## navhard stage 2 能改什么（问题 3，事前的理解，结果里核实）
合成场景只给 4 张 2 Hz 的 3DGS 渲染历史（`synthetic_scene_pickles`，相邻合成场景共用渲染帧），没有中间帧、没有渲染器接口。
能改：4 张之间怎么插帧（hold / warp / GIMM / 其他），以及对这 4 张渲染做什么输入变换。不能改：渲染质量、2 Hz、合成轨迹本身。
所以真实臂在 stage 2 上不存在，stage 2 的上限只能用 stage 1 的 real − gimm 间接估计。

## 预算
一张卡（card 2，按 lease），约 5 h。下载约 1–2 GB（box 经 Clash）；rollout 每臂 504 token 约 1 min；评分 CPU。

## 补充 1（读到 504 个 token 的 real − gimm 之后，扩样本之前写）
已读：S_nt 上 real 86.87、gimm 85.64，real − gimm +1.23 [−0.50, +2.95]（token bootstrap），按登记线是「小 / 未定」；
差主要在 EP（78.5 对 74.5）。没有读 hold / warp 的分，没有读任何解释性读数。
- 扩样本：同一种子、同一抽法（日志打乱顺序不变，每日志 ≤ 20 个），把 n 从 500 加到 1 500（前 504 个 token 不变，
  新的约 1 000 个来自后续日志）。H1 的主检验改在合并的 S_nt+（约 1 500）上做，判读线不变；504 的结果照实并列，
  合并样本里包含已看过的 504 个，这一点在结果里写明。
- 只为 H1 扩样本：阶梯其余臂与 G 探针在合并样本上也跑（同一管线，成本小），读法不变。
- 可部署候选 C1 仍只在合并样本判「限制」时才做。
