# op_parity：plan 足迹对日志车辆框的 hinge（agent hinge）预登记草稿（2026-10-07，未启动，待审）

## 问题与依据

[results/four_dirs.md](../results/four_dirs.md) 方向 3：P2H 在 navtest 的碰撞失败（NC < 1 或 TTC < 1）266 对 WA-JEPA 142 token，WA 只挂其中 30%（P2H 特有）；替换上限 (a) 1.16 [0.88, 1.44]、Shapley 0.87（NC 0.60，TTC 0.17）。最大一类是前方静止车 36%（0.50 分，其中 64% 在转弯 token 上），49% 的碰撞 plan 比日志快 10% 以上（全体基线 13%），这一子集 0.61 分。navhard 同类上限 2.56（与 WA 共享 81%）。HUGSIM 64 的 D3b（前方静止 / 慢车，10 场景）上限 (a) 3.8，plan 的终点落在静止车里面（plan 1.5 s 内避开静止车只有 0.13）。P2H 的损失里没有任何 agent 项：plan 对车辆的速度只靠模仿。本登记问：**在 P2H 微调里加一个「plan 足迹不进日志车辆框」的 hinge，碰撞失败与 EPDMS 是否改善，进度不掉。**

## 臂

- **基线**：pilot 配方 HP-F-s0（turn-train 的 H，已训：navtrain s0of12 + s1of12，W 帧，视觉冻结，P2 + drivable hinge λ 10，3 000 步 × batch 64）；全量阶段基线 P2H10-F-s0 / s1。
- **A10**（tag `HA-F-s0`）：H + agent hinge，λ_a = 10，margin 0.5 m。
- 数值事先写定，不调；λ_a = 3 只在小读 EP 掉超线（< −0.2）而 NC + TTC 失败已降 ≥ 0.3 pp 时开。

## agent hinge 定义（写代码前定死）

- 标签（CPU，`lib/agent_hinge.py` 新建，label 构建仿 `opb_labels.py`）：每个 navtrain token，取 NAVSIM 场景标注里 t0 后 0.5 .. 4 s 的 8 个时刻的非 ego 物体框（车辆、静止车、通用静态物；行人 / 自行车同收），转到 t0 后轴系，按 t0 时离 ego 最近保留 K = 16 个。与 scorer 一致：日志轨迹、非反应式。
- 只罚「在 ego 前方」的物体（物体中心在 ego 当前位姿的前半平面，且横向 < 3 m 或与 ego 足迹重叠）：对应 NC 的 at-fault 规则，不罚被追尾。
- 损失：plan 8 个位姿线性插值到 0.1 s（同 `lib/drivable_hinge.py`），每步把物体框按相邻两个标注时刻线性插值；ego 足迹（nuPlan Pacifica，前沿外扩 margin）与物体框的有符号距离用「ego 四角到物体框的 SDF 与物体四角到 ego 框的 SDF 取最小」近似；hinge = mean(relu(margin − d))，只在模仿行上算，与 drivable hinge 并列。
- 验证（训练前）：在 300 个 navtest token 上，日志人类轨迹的 hinge 违反率 < 1%；P2H 的 NC 失败 token 中 hinge > 0 的比例 ≥ 70%（否则是几何没对上，先修再训）。

## 读数与统计

navtest 12 146 token（bench，W 帧，同 devkit）：EPDMS、NC + TTC 失败率、EP、DAC；按 four_dirs 的 D3 子类（静止车、前车、超速子集）与转弯分层；按 136 log 聚类的配对 bootstrap（B 10 000）。全量阶段加 navhard（G 帧）与 HUGSIM 64 `spec_plan_smooth`（单独报 D3b 的 10 个场景与 31 个 fg 场景），对 P2H10 与 WA-JEPA。

## 小读闸门（seed 0，pilot 规模）

- **过**：NC + TTC 失败 −0.3 pp 以上（点估计）且 EPDMS ≥ +0.2 且 EP ≥ −0.2 → 进全量。
- **λ 3**：NC + TTC 已降 ≥ 0.3 pp 但 EP < −0.2 → 同规则跑 λ_a = 3 一次。
- **停**：NC + TTC 降幅 < 0.3 pp（不论分数）→ 停，报告。

## 全量（仅小读过时）

P2H10 配方（12 分片，10 000 步 × batch 128，warmup 300）+ agent hinge，seed 0 / 1。判据：navtest 对 P2H10 EPDMS CI 在 0 之上且 NC + TTC 失败降；navhard 不掉超 −1；HUGSIM 64 HD 不掉（全 64 差 ≥ −0.02），D3b 碰撞数报告。一句话结论：三项全过 →「agent 项进默认配方」；只 navtest 过 →「开环有效、闭环未动」。

## 代价与时间

标签几分钟 CPU；pilot 训练约 10–20 min（turn-train 实测 205–395 s 每臂）+ navtest 打分约 10 min；全量 2 × 30–60 min + navtest / navhard / HUGSIM 约 1.5 h。全程 < 3 h，按 docs/long-runs.md 分阶段（标签 → 1 → 闸门 → 其余），全部经 GPU 池与 `jevdrive.bench`。

## 限定

日志车辆不反应，hinge 学到的是「别撞非反应式日志车」，与 scorer 同口径，但 HUGSIM 的脚本车（D3a 对向车）不在其覆盖内；超速可能部分是视觉（近距离测距，HUGSIM lead 头 < 3 m 偏远 +1.95 m），hinge 只改 plan 目标；pilot 规模的效应在全量可能不同。

## 执行补记与声明的偏离（2026-10-07，执行者；写于任何 HA 臂训练和打分之前）

用户已批准本登记。下面是执行中遇到的含糊处与不可行处，按「最接近原意」取读法，全部在训练 / 打分前写定。

1. **hinge 的归约**（原文 mean(relu(margin − d)) 未说对谁取均值）：每 (行, 0.1 s 步) 取被计入物体中 relu(m − d) 的最大值（= 对最近物体的 hinge，没有物体为 0），再对模仿行 × 41 步取均值。与 drivable hinge 的「每行每步」口径一致，不随 K 稀释。
2. **「前半平面」的参照点**：ego 在该插值步位姿下、未外扩足迹的中心（后轴前 1.461 m）；横向 < 3 m 也相对这个中心量。
3. **几何校验第一轮不过，按登记「先修再训」**（`lib/agent_hinge.py check`，结果 `$DATA_DIR/runs/op_parity/agent_labels/check*.json`）：
   - 字面版（K 16、四类、前沿外扩 0.5 m、四周 relu(0.5 − d)）：人类日志 300 token 违反率 **1.33%**（4 / 300；全 navtest 0.92%），**不过**；P2H10-F-s0 的 NC 失败 token 中 hinge > 0 74%、HP-F-s0 78%（过）。
   - 违反的人类 token 全是**侧向近距会车**：相邻车道的车在 ego 中心前 1.5–7 m、横向 2.5–2.9 m，箱间侧隙 0.1–0.5 m（没有一例是前方间距）。3 m 横向窗口本意是收进路径上的物体，0.5 m 侧向余量罚的是相邻车道正常会车，不是 NC 的 at-fault 碰撞。
   - 只把「走廊外」物体（物体框与 ego 横向范围 |y| < 1.1485 m 不相交）的余量设 0（只罚真实接触），走廊内保持字面（前沿外扩 0.5 + relu(0.5 − d)）：人类 0.33%（全 navtest 0.07%），过；但 P2H10 NC 覆盖降到 **67.0%**（HP 72.1%），P2H10 不过。
   - 覆盖不足来自 K：92% 的 token K = 16 被占满（行人多），远一点的前车被挤掉。K = 32（同一「t0 最近」规则）：P2H10 76.6%、HP 81.6%，人类 0.33% / 0.11%，**两项都过**。加入锥桶 / 隔离栏 / 施工牌（登记未列的类）还能到 79% / 83%，但登记写定了类别，不加。
   - **采用**：K = 32；走廊内余量 0.5 m（前沿外扩 0.5 m 不变），走廊外余量 0（`--agent-side-margin 0`）；类别仍为 vehicle / generic_object / pedestrian / bicycle。标签 `runs/op_parity/agent_labels/navtrain_all-k32.npz`、`navtest-k32.npz`。λ_a、margin、闸门数值都不动。
4. **NC + TTC 失败** = token 的 NC < 1 或 TTC < 1（four_dirs D3 的定义），百分比 = 失败 token 占比；EP = devkit ego_progress × 100 的 token 均值。
5. **闸门的空档**：NC + TTC 降 ≥ 0.3 pp、EP ≥ −0.2 但 EPDMS < +0.2 不在三条规则里；读作「未过」→ 停（λ 3 只为 EP 掉线而设）。λ 3 若开，用同一规则判；过则全量用 λ 3。小读参照是单 seed 的 HP-F-s0（同 row stream）。
6. **D3 子类分层**：four_dirs 已存的 P2H10（s0 或 s1）D3 失败 token 集合与其子类（`results/four_dirs/nav_tokens_navtest.csv`，打分前固定），报新臂与参照在这些 token 上的 NC + TTC 失败率与 EPDMS；「超速子集」= 该文件 spd > 1.1（plan 比日志快 10% 以上）。另报全体 token 按转弯分桶（bench 的 turn bins）。
7. **全量**：tag `P2HA<λ>-F-s0 / s1`，P2H10 配方原样（12 分片、10 000 步 × 128、warmup 300、drivable hinge λ 10）+ agent hinge。navhard 用 G 帧（`@gimm`，与 hinge 车道同）。HUGSIM 64 只跑 `spec_plan_smooth`（登记所列，也是第 149 条的横向路径）；P2H10 参照用已存的 `spec_plan_smooth` 运行（seed 均值，有 r0 / r1 / r2 三次重复则取三次均值），WA-JEPA 只有 exam 口径。D3b 10 场景、fg 31 场景按 `four_dirs/hugsim_fg_events.csv` 中 P2H10 的事件定（打分前固定）。
8. **全量判据的读法**：navtest「CI 在 0 之上」= 对 P2H10 的 seed 均值配对差 95% CI 下限 > 0，「NC + TTC 失败降」= 点估计 < 0；navhard「不掉超 −1」与 HUGSIM「全 64 差 ≥ −0.02」都按点估计。
