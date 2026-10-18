# NTV0 + ACC + GAP + holdlead 预登记（PAI 281，serve 侧，不训练）

日期 2026-10-19，写于任何读数之前。问题：第 327 条 holdlead（`JEV_VHIST=holdlead`）在 g10 上 +0.040 [+0.020, +0.060]，但从没和 ACC / ACC + GAP
一起读过（第 328 条「ACC 与 holdlead 未合读」）。PAI 候选是 NTV0-L18-s0 + ACC + GAP（第 366 条，0.7443，零分 59）。叠上 holdlead 是否还有用？

## 组合检查（代码，读数之前）
- `experiments/alpasim/lib/vhist.py`：holdlead 只改喂给 adapter 的自车速度历史行（输入侧）；served 轨迹不变。
- `experiments/alpasim/lib/pai_driver.py`：`VH.feed` 在 `PC.plan` 的 `ego_fn` 里；`VH.observe(s.vh, o)` 在 `FX.apply`（ACC / GAP / LEAD，`serve_fix.py`）之前，
  读的是模型原始 plan 与 lead 输出；`FX.apply` 用量得的车速 `V[-1]`，不用喂入的速度。`core.lead_out = FX.NEED_LEAD or VH.NEED_LEAD`，两者共用 lead 头。
  driver 的互斥 assert 只管 TURN_GAIN / TRACK / EXPERT / INTENT_HOLD / TTS，不含 VHIST 与 ACC。`docker/build.sh` 也写了 `VHIST=holdlead ACC=1` 同用。→ **可组合，不冲突**。
- 预期的相互作用（不判，只记）：ACC 已经把纵向交给 openpilot 式 MPC，holdlead 的增益原本来自对照与被追尾（61 对 75）；ACC + GAP 已把被追尾压到 28（第 330 条），
  所以叠加的空间可能小，也可能因为模型 plan 变快而让 ACC 的 stop guard 少开。

## 读数
- 臂：NTV0-L18-s0，serve `JEV_VCONT=1.0 JEV_LEAD=1 JEV_ACC=1 JEV_GAP=0.7,2.7 JEV_VHIST=holdlead`，`controller.gains.rel_front_steering_angle_weight=10`，CONC 4，
  ALPASIM_FAST=1，chunk 表取自已存读数 `$DATA_DIR/runs/defect_fix/pai_accgap/NTV0-L18-s0/chunks`（同一 8 个表），优先级 12，pool。
  box `$DATA_DIR/runs/defect_fix/pai_accgap/NTV0-L18-s0-hl`；脚本 `experiments/defect_fix/scripts/ntv0hl_read.sh`。
- 对照：已存 NTV0-L18-s0 + ACC + GAP（同路径，代码 2e2d0e46；此后 `experiments/alpasim/{lib,scripts}` 无改动）与 RSG + ACC + GAP（`route_follow/acc_gap/read/s2`）。
  `pai281.py`，按场景配对，log 聚类 bootstrap 10 000，seed 0，剔除 driver RPC 错误场景。
- 结果 `experiments/defect_fix/results/ntv/NTV0-L18-s0-hl/pai281.md`。

## 判据（一条线）
**过** = 对已存 NTV0 + ACC + GAP 的配对均值差点估计 **≥ +0.01**，且零分 **≤ 61**（59 + 2）。两条都满足才算 holdlead 进 PAI 候选（需新镜像，提交由用户定）。
CI 照报不判（1 seed、一次读）；同时报对 RSG + ACC + GAP 的差、零分类型、被追尾。不过 = PAI 候选保持 NTV0 + ACC + GAP。不追加读数、不调 holdlead 参数。
