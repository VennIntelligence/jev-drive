# 预登记：rm_train 的 torch.compile 是否在 eager 的 seed 散布之内（2026-10-17，写于任何输出之前）

用户 2026-10-17 批准验证。docs/speed.md 的探针：RSG-L18 配方一步 196 → 99 ms（2.0x），但不逐位一致（300 步后权重差到 5.3e-3）。

**问题.** 编译训练和 eager 训练在同 seed 下的差，是否落在 eager 两个 seed 之间的差之内，从而能在新 lane 的边界上把它设成默认。

**实现.** `rm_train.py --compile`：每步主训练 forward 走 `torch.compile(model, dynamic=False)`（和探针、pp_train `--compile` 同一写法）；
第二次 forward（kd / awr / d2）、dev eval、reliance 读数仍是 eager。不加 flag 时不增加任何操作。

**臂（同一起点 RB0-F-s0，RSG-L18 续训配方 2 500 步，`nx.sh train rsg`）.**
- E0 / E1：已存的 RSG-L18-s0 / s1（eager）。复用条件：今天代码不加 flag 重训 RSGE-L18-s0（2 500 步），`nx.py bitcmp --hash ea8cacc3e451c577`
  对 RSG-L18-s0 逐位一致（标量、权重、dev ADE、on_hash_60）。不一致则 E0 / E1 的读数不复用，按今天代码重训重读后再判（另记补记）。
- C0 / C1：RSGC-L18-s0 / s1，同一命令加 `--compile`。
- 三个训练都 `--exclusive`、限在卡 1–4，墙钟可比。

**读数（全部复用已有脚本）.** dev ADE（训练 DONE）；wc.py plans（等 W、走 G）；led.py plans F（S_L）；dc.py plans nav / pai（desire 4 s，
`dt.desire_read` 的 mean4，lcl4 / lcr4 只报告）；wt.py devplans（横向 2 s：`wt.dev_read` 的 2 s 横向中位绝对差）；held 460 @18
（`rm_chain.sh read`，6 块）；PAI 281 提交配置（`pai_box.sh`，CHUNKS_FROM r281/rsg，SERVE `JEV_VCONT=1.0 JEV_LEAD=1 JEV_ACC=1`，
WIZ rel_front_steering_angle_weight=10，ALPASIM_FAST=1）读 C0、C1、E1，并把 E0 在今天的配置下重读一次作对照。

**判据（写死，结果留空）.** 散布 spread(X) = |X(E0) − X(E1)|；比例类格子的 spread 下限为一行（1 / n）。每条线对两个 seed 都要成立。
1. dev ADE：|ADE(Cs) − ADE(Es)| ≤ spread(ADE)。
2. 开环账本格子：等 W、走 G、S_L、desire 4 s（mean4）各自 |X(Cs) − X(Es)| ≤ spread(X)；横向 2 s：lat2s_med(Cs 对 Es) ≤ lat2s_med(E1 对 E0)。
3. 闭环 held 460 @18：Cs 对 Es 配对均分（session 聚类 CI，`wt.xpair`）CI 含 0，且零分差 |zeros(Cs) − zeros(Es)| ≤ 3。
4. PAI 281 提交配置：Cs 对 Es 配对均分 CI 含 0。E0 用已存读数 `runs/route_follow/pai_g10/acc`；若今天的 E0 重读与它有任何 scene 分数不同，
   E0 改用今天的重读（两者都报告）。E1 只有今天的读数。

**决定规则.** 四条线两个 seed 全过 → 建议从下一个 lane 边界起把 `--compile` 设为新 lane 的默认（从不在 lane 中途切换；老 lane 保持 eager）。
任一条不过 → 不切默认，`--compile` 保持可选，并写明哪条不过、差多大。另报：墙钟、每次训练省下的卡时、编译预热时间。

**说明.** 只有两个 eager seed，散布是单对估计，证据最多为「中」。
