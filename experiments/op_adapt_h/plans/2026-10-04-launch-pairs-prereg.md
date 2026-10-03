# op-adapt H 第 2 轮：起步工作点的配对（lane L3b，2026-10-04）

写于任何训练和读数之前。依据：第 98 条（第一轮适配把 10°/s 的 G 削 71–78%，HUGSIM 打转不降）、第 100 条（闭环在起步工作点放大：
5 s 静止 + 第一帧运动后，模型对最初约 1° 历史偏航的响应近似阶跃，局部增益 step 1 约 5°/°，大信号增益原生 4.3、pilot 3.4、it_dw3 3.8，
环路每步 ×1.6–1.8；要切断环路大信号增益约需降到 0.2×）、第 94 条（真实转动是信号）。代码沿用 op_adapt_h 的 lib、trunk bank、h_chain。

## 改什么

在 it_dw3（蒸馏权重 3，过 drift 护栏）的配方上加两类行，教师一律是日志未来：

| 行 | 内容 | 素材 |
|:--|:--|:--|
| L（起步） | 历史 = 静止前缀 + 最后 m ∈ {1,2,3} 帧运动；再加小的假偏航 δ ∈ ±[0.3, 3]°：静止前缀整体转 δ，运动帧从 δ 线性降到 t0 的 0（与 HUGSIM 去旋转视角下的历史偏航同构） | 真实起步：WOD train 中静止 ≥ 1.8 s（v < 0.1 m/s）后第一次 v > 0.1 的 onset，t0 = onset 后第 m 帧（新域 `lwod`）；CARLA P6（route 不在 bench2drive220）同样定义（新域 `lcarla`）。合成起步：nav / wod / carla 池里 0.5 ≤ v0 < 5 m/s 的样本，把前 10−m 帧冻结成第 10−m 帧 |
| S（小偏航率） | 整段历史假偏航率 ±[0.3, 2]°/s | nav / wod / carla 池的 stop / low / mid 档 |

- 假偏航符号：日志 3 s 朝向 |ψ| > 3° 时取反向（未来不跟随），否则随机。未扰动的真实起步帧（含真实转动）也以 U / L 行的形式在训练里，教师 = 日志未来（第 94 条）。
- navtrain 没有真实起步：op_lb 的历史只有 1.5 s（4 关键帧 + GIMM），放不下 ≥ 1.8 s 的静止前缀，nav 只用合成起步。这是偏离任务书「三域都用真实起步」的地方。
- 考题不进训练：navtest、navhard、bench2drive220、HUGSIM 场景、第 92 条探针样本（WOD val、CARLA P4、navtrain lb_navtrain）。`lwod` 只取 op_adapt_l 的 train / dev 切分，`lcarla` 的 route 按已冻结的 `b2d/op-adapt-h-carla-{train,dev}` 归属。
- 新变体放在单独的 bank2（`$H/bank2/<dom>/`），旧 bank 不动。

## 臂（pilot，2 500 步，seed 0，约 15 min 训练）

`ln1`：dw 3；每批 U 10 / D 10 / H 8（假偏航 3–15°/s、repeat、single）/ O 6 / L 10 / S 4。L 行的域权重 lwod 0.35 / lcarla 0.15 / nav 0.2 / wod 0.15 / carla 0.15；L 行里 25% 是不加假偏航的起步（真实起步原样或合成冻结）。

参照：shipped（O）、it_dw3（同配方无 L / S 行），所以 ln1 − it_dw3 的差归于新增的两类行与份额变化（不单设 control）。

## 读数与线

| | 读数 | 线 |
|:--|:--|:--|
| (a) | 起步大信号增益：lean_probe replay（第 100 条的 9 份 HUGSIM 打转日志，3 场景 × 3 份），s = (φ1 正常 − φ1 去旋转) / H，1 ≤ \|H\| ≤ 15° 的步汇总取中位；比值 = s / s(shipped)；同报 lean_report 的 window kernel 每步增长 z(c s)，c 用同一个 c | **成立**：比值 ≤ 0.5。第 100 条说要 ≈ 0.2 才切断环路，报 z 是否 < 1。另报 local 模式 step 1 的局部增益（不设线） |
| (b) | h_rate_probe（第 92 条探针池，stop + low 档）在 0.5 / 1 / 2°/s 的 G 比值（新 / shipped） | **成立**：三个率上三域平均比值都 ≤ 0.5（削 ≥ 50%）；分域 CI 同报 |
| (c) | navtest PDMS Δ（对 port O，op_adapt_l 管线） | Δ 点估计 ≥ −0.3 |
| (d) | HUGSIM 全 64 场景，不加 derot 规则，PR #57 控制器，h_hugsim.sh SCEN=all64 | **成立**：打转 < 10（shipped 10）。只在 (a) 比值 ≤ 0.75 时跑（比 pilot 0.79 / it_dw3 0.85 有实质下降） |
| (e) | WOD val start / stop 捕获、静止帧假起步率（op_adapt_l_readout） | 捕获点估计 ≥ −0.02；假起步 Δ ≤ +0.02 |
| (f) | 54 个非打转场景 HD 配对差（对 base） | 点估计 ≥ −0.02，CI 同报 |

## 迭代规则

- 只看 dev 选：新的起步 dev（lwod / lcarla dev 的真实起步 + 三池 dev 的合成起步 m = 2，δ = ±1°；小率 ±1°/s）的 G_L1（3 s 朝向与 1 s 方向，每度假偏航）、drift 中位 ≤ 0.15 m、ADE；外加 (a)(b)（离线、便宜，任务书要求先读）。navtest / HUGSIM 只考选中的模型，每轮至多 2 个。
- 旋钮：L 行份额、δ 范围、dw。每次迭代前在文末追加「改了什么、期望什么」。

## 追加记录
