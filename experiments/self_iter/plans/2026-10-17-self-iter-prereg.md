# SELF-ITER 预登记：开环自迭代（难例挖掘），标签 = 人的日志（2026-10-17）

- 来源：用户要一个自迭代循环——模型在全部数据上跑一遍，开错的行拿出来加重再训，开对的行照常混回去保住已有能力；标签用人的日志（在这些同分布状态上可信）。
  2026-10-17 上午用户改了范围（主会话 11:30 转达）：**不等 trainval 全量下载，用盘上已有的数据把第一轮整条跑通，identity 与 smoke 过了就直接起**，
  用户已批准起跑，本条记下。全量（183 个归档，约 2026-10-18 早上）到齐后同一条链重跑（`SFX=2 si.sh launch-full`）。
- 代码：`experiments/self_iter/scripts/si.py`、`si.sh`，`lib/self_iter.py`，`experiments/route_mem/scripts/rm_train.py` 的 SELF-ITER 块（`--si-scan`，关 = 逐位不变）。
  行工具沿用 WAIT-TRAIN（wt.py index / teacher、pp_prep.py、rm_build.py nav、rf_pait.py merge），读数沿用 nx.sh cont-reads 的同一批仪器。
- 接第 326、331、335、337 条：P2-F 起的模仿阶段洗掉了「看画面减速」一族；RTB1 的保留项买到了到达时停，但起步变迟（静止起步 4 s 弧长 −0.44 m）、
  公开集 12 个 scene 原地不动被追尾（explore_f.md）。本线不加新损失，只改「哪些行被抽到」，回答：只靠重抽人的日志里开错的行，能不能把错的地方修掉而不丢别的。

## 1. 行

- navtrain 训练行：RSG-L18 配方的 101 499 个正常行（dev 1 789 行不训练，照旧做 dev ADE）。
- trainval 新行（盘上 11:26 时 28 个 trainval 相机归档已解压，sensor_blobs/trainval 1 205 个 log 目录）：`si.py select` 对 navsim_logs/trainval 的每个 log 跑
  第 331 条的逐帧行（led.log_cells；格子 led.cells_of + wc.cell_of），新行 = 4 帧历史 CAM_F0 都在盘上、token 不在 navtrain、log 不含 navtrain dev token、
  不是任何评估 log（wt.leak：held 460、公开 1491、PAI 281、waitcap_test；navsim_logs/test 的 log 一个都不要）；**每帧都要（2 Hz），不做稀疏**。
  crc32(log) % 10 == 0 的 log 整个留出（tv_hold），永不训练，做闸的未见行。行数由 select 记录（`results/select.json`），写进第 6 节补记。
- 新行进合并池 on1_L18RBSI = on1_L18RB0（`s`，S 行，mode self，原样）++ on1_SITV（`t`，mode both：只有日志模仿，agent / road / route 标签 ok 全 False，
  所以没有任何 hinge），`--on-mode mixed`。新行在 128 行 batch 里的期望槽位 k_tv = min(round(102 × n_tv / (101 499 + n_tv)), 51)（按行数比例，上限一半：全量时 navtrain 行不至于只剩个位数），从 102 个正常槽里让出
  （`--on-k 13 + k_tv`），S 行期望仍是 13；新行的 route memory 和正常行一样按 --rm-drop 丢（不按池行的不丢规则）。

## 2. 扫描与抽样

- 扫描策略：RSG-L18-s0（默认；`--policy` 可换，例如 RTB1O-L18-s0）。前向 = 训练里的同一调用（route memory 开、人的历史），每行存
  ADE、沿轨 / 横向误差（2 s、4 s，投影到人路径的切向）、4 s 弧长偏差（plan − 人，带符号）、人的行为类（stopped / starting / turning / braking / cruising，
  lib/self_iter.classes）与第 331 条格子（cells.npz，按 token 连）。
- 难度 = 扫描 ADE，**只在同一集合内排名**：navtrain 训练行是策略自己的训练行，trainval 新行没见过，两者误差不可比。
- 抽样：每个集合里 ADE 前 10 % 的行抽取权重 × m，其余 × 1（`--si-mode top --si-top 0.1 --si-mult m`）；正常行用 numpy 加权抽取（rm_train 自己的 rng 流），
  池行用池抽取的 p（S 行质量 13 / on-k、新行其余、留出行 0）。m = 1 是「同样的行、均匀抽」的数据臂。
- 训练 = RSG-L18 原命令（`--init RB0-F-s0 --warmup 1 --eval-every 500 --yr-mass 0.1 --label-bank --agent-lam 10 --agent-on-only --shape --ho-w 3 --rm-lr 3e-4
  --on-split alpasim_nuplan/onpol-train --lr-scale 0.1`）+ 合并池 + 抽样，其余不变。P0 保留项（`--kd-lam`）本轮不开（可选开关在 rm_train 里，规则不变）。
- 一轮 = 扫描 → 重新加权 → 训练。第 2 轮用第 1 轮的模型重扫（`si.py scan --policy <round-1 tag>`，`--si-scan` 指向新扫描），这就是自迭代。

## 3. 先决检查（不过则停，链上用 --after 卡住）

1. 代码逐位：RSG-L18 配方 60 步，改动前代码（d7675c5a）对新代码（--si-scan 关）：标量、最终权重 max |diff| 0，on_hash_60 = RSG-L18-s0 的 `ea8cacc3e451c577`（`results/identity_code.json`）。
2. 池逐位：同配方，合并池 on1_L18RBSI、新行权重 0、`--on-mode mixed`（`--on-weight pool=t:0`）对 1 的新代码臂，同上三条（`results/identity_w0.json`）。
3. smoke：SISMOKE-L18-s0，m = 3，40 步，跑完不出非有限损失。

## 4. Pilot（1 000 步，seed 0）与登记闸

- 臂：SIU1P-L18-s0（m 1，只加数据）、SIM3P-L18-s0（m 3）、SIM6P-L18-s0（m 6）。对照 C = WTCP-L18-s0（RSG-L18 配方 1 000 步 seed 0，已存）。
- 读数：每臂与 C 在 navtrain dev（1 789 行）+ tv_hold 上的扫描（`si.py scan --sets dev`）；难例十分位由 RSG-L18-s0 的扫描定（不是由 C 或臂自己定，避免回归均值）。
- 线（全过 = PASS）：
  1. dev ADE（navtrain dev）≤ C + 0.01 m；
  2. tv_hold 上 RSG-L18-s0 难例前 10 % 的行，臂的平均 ADE 比 C 低 ≥ 3 %（相对）；配对差的 log 聚类 bootstrap CI 只报；
  3. 自由路（第 331 条 C 格）4 s 弧长偏差在 C ± 0.1 m 内；
  4. 横向不变：全部 dev 行 2 s 横向误差绝对值的中位数 ≤ C + 0.05 m；
  5. 静止起步（v0 < 0.5 且人起步）4 s 弧长偏差 |臂| ≤ |C| + 0.02 m（不能出现 RTB1 那种起步迟滞）。
- 读法：SIM 对 SIU = 挖掘本身的作用；SIU 对 C = 多出来的行的作用。闸只判 SIM / SIU 各自对 C。

## 5. 闸之后（链自动提交）

- 每个 PASS 臂：闭环 held 460 @18（6 块）、静止 scene @10、PAI stop17（PAI 配置）、到达重放、公开 1491 @10（10 块），对 RSG-L18-s0。
  闭环描述线：held 零分 J ≤ C + 3 且转弯完成 J ≥ C − 2（C = RSG-L18-s0）；静止 scene、stop17、公开均分 / 零分 / 被追尾只报。
- 长跑 L：PASS 臂里难例降幅最大的两个，各 25 000 步（RSG-L18 续训 2 500 步的 10 倍），同配方同抽样 → 同一开环闸（C 换成 RSG-L18-s0）→ PASS 则同一套闭环。
  L 回答「更多行 + 更长训练是否真有帮助」。
- 没有臂 PASS：不跑 L、不跑闭环，判 STOP，报读数。任何结果都不是提交候选，除非闭环全部读完后另行决定。

## 6. 补记

（行数、扫描结果、成本估计在这里补，先于对应读数之后的任何改动。）
