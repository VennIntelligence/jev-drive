# 转弯学校 示范行：带人类时机的 PAI 转弯行 `on1_PAIT` 与合成减速的 `on1_CFTv`（2026-10-13）

脚本 [rf_pait.py](../../experiments/route_follow/scripts/rf_pait.py)、[rf_pait_chain.sh](../../experiments/route_follow/scripts/rf_pait_chain.sh)、
[rf_cftv.py](../../experiments/route_follow/scripts/rf_cftv.py)；登记 [2026-10-13-turning-school-prereg.md](../../experiments/route_follow/plans/2026-10-13-turning-school-prereg.md)；
闸 [pait_gates_PAIT80.json](../../experiments/route_follow/results/pait_gates_PAIT80.json)、[cftv_gates_CFTv.json](../../experiments/route_follow/results/cftv_gates_CFTv.json)、
[pait_route_check.json](../../experiments/route_follow/results/pait_route_check.json)；图 `experiments/route_follow/figs/pait_bev_PAIT80.png`、`cftv_bev_{0..5}.png`。没有训练。

## 1. 为什么要带纵向

用户今天的判断：PAI 上主要的失败是 14–17 m/s 冲进路口不减速。rf_pai_follow 的读数印证：RSG 在路口入口前减速的 0 / 20，人 10 / 20；
入口车速中位 ego 9.0、人 5.1 m/s（直行出口的那些 11.2 对 4.9）。只给形状（路线 hinge、on1_CF）不告诉模型「什么时候、多快」；
所以示范行的目标必须是一条带时间的轨迹：0.5–4 s 每个点是人在那个时刻真实到达的位置。

## 2. on1_PAIT：PAI 人类日志上的转弯行

- **状态**：onpol_pai 第 2 遍（日志跟随：driver 返回日志、MPC 跟踪，模型照常喂入并记录）里自车所在的状态，就是人当时的状态
  （横向偏差 p95 0.03 m，沿程 p95 0.22 m）。画面是这些状态下真实渲染、按 driver 方式编码的 token（onp_cache，G-id-ulp 已在该遍过工作线）。
- **选哪些行**：转弯用 rf_pai_follow 搬过来的严格定义（路线折线 30 m 内净转角 ≥ 60°，入口 / 出口取 10 % / 90 %），
  窗口从「人到入口前 6 s」与「人到入口前 40 m」两者较早者开始，到人到出口为止：覆盖 40 m → 0 m 的减速段，不只路口里面。
  每个转弯按车道图分类（rf_pai_follow topo 的规则，自车车道 = 入口前 10 m 的路线车道）：straight_exit / straight_road / no_straight_exit /
  curve_single / lane_change，写进 tab 的 `topo`，训练时 `--on-weight topo=straight_exit:2,topo=straight_road:2` 把「前面路是直的、路线要转」的行加倍抽。
- **目标**：行的 `imit` = 日志路径上 s_ego + d_k 处的点（d_k = 人在 t0 → t0 + 0.5k s 实际走过的弧长），横向偏差在 max(8 m, 2 s × v0) 内淡出。
  模仿损失（pp_train dist 的 x、y、yaw）因此带着速度；`--shape` 只改 hinge 的梯度，不碰模仿项，不需要新开关。
- **路线记忆**：pass 2 的 driver 没有记录路线消息，所以用 runtime 自己的 RouteGenerator（MAP、40 m）按 runtime 的调用顺序重生成
  （每个决策一次 generate_route，位姿 = 行的 anchor，姿态取日志在 t0 的 z / roll / pitch 再转到 anchor 的 yaw），再用 driver 的 Tracker 得到记忆。
  核对：pass 2 的 driver 记下了每个决策的第一个路线点 route0，重生成的与之相差 ≤ 0.041 m（1 340 行，100 % ≤ 0.05 m，G-route 过）。
  在 RB0 重滚（记录了全部消息）上同法重生成：路线点差 ≤ 0.07 m（只用 yaw）；记忆的点位置中位差几厘米，转弯 scene 0810968d p99 0.32 m、7 % 行有一个点的有效位翻转；
  cos / sin 在折线短段处会跳（served 的记忆本身也同样敏感），这是记忆表示的脆弱，不是重生成的偏差。
- **数（on1_PAIT80，现在只有已渲染的 80 个 train160 scene）**：1 361 行、12 个 scene、13 个转弯（straight_exit 5 个 / 386 行，curve_single 6 / 747，
  no_straight_exit 1 / 158，lane_change 1 / 70）；离入口 > 40 m 54 行、40–20 m 321、20–0 m 668、路口内 318。人到入口前 40 / 20 / 0 m 的车速中位 9.0 / 7.7 / 4.6 m/s。
  hard_train 一行都还没有：那 59 个 scene 没有沿日志的渲染。
- **闸**：G-route 过；G-road（示范折线不穿路沿段）0 %（日志 0 %）；G-speed（4 s 弦长与日志差 ≤ 10 %）100 %，中位比 1.000；
  G-r1：池文件与 cache 逐位相同（front、ego），P2H10S-F-s0 在池 token 上对 driver 记录的 poses_model 的读数（max(0.03 m, 4 ulp)）在 Tokyo CPU 上跑，结果见最终报告；
  BEV 6 行图看过：示范沿日志路径、点距随减速变密，记忆点在路线上。

## 3. 补全：剩下 104 个 scene 的链

`rf_pait_chain.sh` 已在 Tokyo `jev:pait-chain` 等 `/data/runs/route_night/CARD0_FREE`：卡 0 一空就用 pass 2 的同一套代码
（dev 镜像的 onp_pass2.sh + onp_log_driver、pass 2 代码克隆、P2H10S-F-s0、默认增益）沿日志滚 38 个优先 scene（train160 里 14 个日志转弯 scene + hard_train 的 24 个转弯 scene）
和其余 66 个 train160 scene（估计单卡 1.5 h），再出行、token、gid-ulp、路线、打包（PAITn）、与 PAIT80 合并成 `on1_PAIT`，rsync 到主盒并在主盒合并 `on1_PAITC`。
hard_exam 与 control 的 scene 不在列表里（断言过）。

## 4. on1_CFTv：nuPlan 反事实行的合成减速

on1_CFT 的 e2 规则排掉了「人的速度配不上更急的反事实出口」的 1 435 行（其中只因 e2 被排的 942 行）。这些行在同一条示范路径上合成纵向：
曲率限速 v ≤ √(3 / |κ|)（κ 在 2 m 上平滑，再按 0.5 s 离散的实测横向加速度收紧到 ≤ 3.3）、上限为人在 4 s 内的最高速，
向后递推保证前方任一限速点前制动 ≤ 2.5 m/s²，起点是 token 的车速，加速 ≤ 1.5 m/s²。起点车速以 2.5 m/s² 赶不上限速的 62 行排掉，与他车框重叠的 8 行排掉，留 872 行
（j0 596、j1 248、lc 28），`src = synthetic`；on1_CFT 的 11 442 行原样并入（`src = human`），共 12 314 行。合成行 4 s 弧长中位是人的 0.85。
闸：G-v1 路面内 100 %；G-v2（横向 ≤ 3.3、制动 ≤ 2.75、首步均速与 v0 差 ≤ 10 %）100 %；G-v3 复算的 on1_CFT 选择与已存池逐名相同。
`--on-weight src=synthetic:W` 可以单独调合成行的权重。

RCFT-L18（只用人速度的 CFT 形状目标）刚读完：G-swap 0.31（on1_CF 0.55）、public 1491 比 RSG 多 13 个零分，两条都没过。
所以按协调方的决定，CFTv 只做到 e2 行的合成减速为止，不再在 on1_CFT 其余部分上花功夫；登记里 RPAIT（只用 on1_PAIT）是主臂，RPAITC（+ on1_CFTv）是次臂。

## 5. 风险

1. 行少：PAIT80 只有 13 个转弯；温和日程 2 500 步 × 12 行 / batch 每行约被抽 20 次以上，有记住个别 scene 的风险。全量链补到约 30 个转弯后再训是首选；等不及就用 `-80` 后缀的臂，不与全量混比。
2. 状态仍是人的状态：教的是「在正确的位置、正确的速度，看见这条路线时怎么开」，不教「已经开快了怎么救」。后者是 on-policy 行（RSG 的那类）的事；两者可以以后合并。
3. PAI 没有道路栅格：PAIT 行在 `both` 下只有 agent hinge（日志回放的他车框），没有道路 hinge。
4. 路线记忆在转弯处的 cos / sin 脆弱（见 2），训练与服务两侧同样存在，但在转弯附近这部分输入噪声偏大。

## 6. 结果（2026-10-13 晚）：全量池与阶梯

全量 `on1_PAIT` 5 160 行、53 个严格转弯（hard_train 2 078 行；straight_exit 21 个转弯），闸全过（G-route 0.997、G-road 0、G-speed 1.000，
R1：P2H10S-F-s0 在池 token 上对 driver 记录的 plan，max(0.03 m, 4 ulp) 超出 0.29 % / 0.44 %，错 checkpoint 99 %）。六个臂按登记训完、读完
（[tschool/summary.md](../../experiments/route_follow/results/tschool/summary.md)）：**没有一个臂过阶梯第 2 步**，所以没有离线排序、没有 Tokyo 考试，没写 EXAM_READ_REQUEST。

- 主臂 RSGPAIT-L18（RSG 的 S 池 + 30 % PAI 行，全部 both）：held 均分 0.905 / 0.906，比 RSG 高 0.04，零分 37 / 36（线 37，压线过）；
  公开 1479 零分 56 / 54，比 RSG 多约 10 个（线 +5），**护栏不过**。与 RSGboth（同 both、无 PAI 行）比，held 零分更少（37 / 36 对 41 / 46），公开零分相近：
  多出来的零分跟着 S 行上的 both 模式走，不是 PAI 行带来的；PAI 行以 30 % 混进 S 配方里不伤 nuPlan。
- 只用 PAI 行的 RPAIT / RPAITC 离开 nuPlan 分布（held −0.056、公开多 20–25 个零分），与 80 scene 试点一致。
- 离线 PAI 转弯读数（诊断，19 个 hard_train 转弯，对 PAIT 臂是训练内）：入口处 plan 比人快 RSG 0.34、RSGboth 0.35、RSGPAIT 0.23、RPAIT 0.12 m/s；
  在人的状态上所有臂都转得对（lat2 ≈ 1、side4 = 1）。开环在人的状态上速度差本来就小，闭环冲太快是策略自己把速度带上去的结果，这个读数看不见。

下一步要定的是 both 模式在 S 行上带来的零分（不是 PAI 行）：例如只在 PAI / 示范行上开模仿、S 行保持 self（`--on-mode mixed` 已有且过了已知答案测试），由主会话定。

## on1_SELF

转弯学校第 2 阶段（P4，自状态分支示范）的行。脚本 [rf_selfpool.py](../../experiments/route_follow/scripts/rf_selfpool.py)；
P3 存档上的开发池 [results/p4_self/pool_dev](../../experiments/route_follow/results/p4_self/pool_dev/README.md)（`on1_SELFdev`，含 hard_exam scene，只作开发，不训练）。
没有训练。

- **分支**：分支搜索（P3 rf_p3.sh / P4 rf_p4.sh）里 read.json `done` 为真的 rollout（P2 的完成判据）。一个 scene × D 的多个完成单元都保留（近乎同一状态上的不同示范），
  按 scene 报行数，训练时用 `--on-weight log=<scene>:w` 封顶。失败的 rollout 不出行，按 scene 列进 `failing.json`，留给以后的 critic 线。
- **窗口**：src log（自车跟日志到交接 h，P3 全是这种）从 h 开始；src own（策略自己开到那里，P4 的 self 单元和恒等单元）从 h 与「路线站点越过入口前 40 m 的决策」两者较早者开始；
  到出口后 10 m（P4 的 release 站点）为止，`phase` = approach（入口前）/ inside / exit（出口到出口 + 10 m）。站点是 anchor 在 plan 路线（rf_p4 plan 的 0.5 m 栅格）上的投影。
- **anchor 状态**：策略自己的状态。输入照 onp_rows 从 rollout.asl 经 pai_driver 原样重放（G-in 与 drive.jsonl 逐位对上），token 由 onp_cache 在这些帧上编码；
  帧就是闭环里渲染给策略看的那些，**不需要重渲染**，只需一次编码器（GPU 约 1 分钟 / 32 个分支）。
- **目标 `fut`**：实际驶过的轨迹（driver 的 egomotion anchor）在 t0 + 0.5 k s（k = 1..8），换到 anchor 的 rig 坐标；速度隐含在点距里。`served` = 策略自己的原始 plan。
  只要 8 个 slot 有效、在任何评分事件 / 结束之前、4 s 未来在 rollout 内且在第一个评分事件之前的行。
- **路线**：输入记忆用 driver 记下的路线消息经 rmem.Tracker（rf_pai_pool 的做法，G-rmem：与 served `rmem` 相同）；路线 hinge 标签与 G-route 用 rf_pait 的重生成。
  偏离任务书「记忆按 rf_pait 重生成」的理由：在策略自己的状态上，重生成的记忆有 34 % 的行与 served 差 > 0.02（p99 0.095）；重生成取的是人同一时刻的 z / 姿态，
  而 P3 / P4 的 driver 本来就记了消息。`--feat regen` 可切回。
- **tag**：log（scene）、branch（`<scene 8 位>~<session 8 位>`）、cell（P3 `g<g>-s<s>`，P4 `g<g>-<raw|ve|ve08>`）、D（D40..D0，Did = 无交接）、src（own / log）、phase、
  push（这一决策上 served plan 被变换过）、topo、part；d_entry（入口 − 站点，浮点，merge 后丢）。
- **闸**（`pool_gates_<name>.json`）：G-in；G-route（重生成首点对 route0 ≤ 0.05 m ≥ 99 %）；G-rmem；G-road（示范穿路沿 ≤ 1 %）；G-speed（示范 4 s 弦长与控制器轨迹差 ≤ 10 % ≥ 99 %）；
  G-r1（池 token / ego / 特征与 cache 逐位相同）；G-shape（决定 293：分支实际轨迹在 [入口 − 40, 出口] 上、截到第一个零分事件，本类 p95 曲率 / 曲率率、横向加速度 ≤ 4.42、
  入口实际速度 ≤ v_e + 1；阈值取 plan 的 `gate`，量用 rf_shape.track_shape / speeds；不过的分支整支丢、计数）；G-crawl（入口前的行，min(v0, 入口前的未来段速) < 0.5 v_e 就丢）。
- **开发读数（P3，666 行，13 个分支，6 个 scene）**：G-in、G-rmem、G-road、G-speed、G-r1 过；G-route 97.9 %（线 99 %，超的 2.1 % 在 0.05–0.10 m，只影响 hinge 标签）按字面不过。
  G-shape 17 / 32（几何部分与决定 293 的闸 B 相同 20 / 32，入口速度再排 3 个），复算的曲率 / 曲率率与 shape.json 逐个相同。G-crawl 排 370 行：速度推的分支到路口前在爬。
  eaba3f6a 一个 scene 占 58 % 的行；路口内的爬行（eaba3f6a d0 s0.6，最低 0.3 m/s）G-crawl 管不到。

### 主盒上 P4 的命令

P4 在主盒跑完（`rf_p4.sh read <P4 root>` 已写出 `<root>/read.json`，`<root>/plan.json` 在）后：

```
cd ~/data/jev-drive && git pull
.venv/bin/python -m jevdrive.cl submit --name selfpool --vram 6 --cpu 16 --ram 40 --tries 1 --log-dir $DATA_DIR/runs/route_follow/selfpool_SELF/pool -- \
  $DATA_DIR/envs/op-train/bin/python experiments/route_follow/scripts/rf_selfpool.py build --backend box --roots <P4 root> --tag on1_SELF
```

- 默认 `--read <root>/read.json`、`--plan <root>/plan.json`、`--served` = 本仓库（主盒的 P4 用活仓库服务；如果 pai_driver / pai_core 在 rollout 之后改过，给冻结的克隆，G-in 会暴露）、
  工作目录 `$DATA_DIR/runs/route_follow/selfpool_SELF`、结果 `<work>/results`、暂存 `$DATA_DIR/tmp/selfpool_SELF`（主盒 /tmp 是 30 GB 系统盘）。hard_exam / control scene 会被拒。
- 已在主盒原生环境上冒烟过 sim + cold 两段（一个 PAI 原生 rollout：AlpaSim .venv 重放 199 行 G-in 199 / 199，op-train 的 cold 段）；cache / pack 与 Tokyo 同一 op-train 路径。
- 如果 P4 在 Tokyo 跑（`BACKEND=tokyo`），同一命令加 `--backend tokyo --card <空卡>`，再把 `on1_SELF`、`on1_SELF@warp`、`feat/on1_SELF.npz` rsync 到主盒（rf_pait_chain.sh 的做法）。

合并与抽样权重（SELF 行占池抽样 30 %，rf_tschool.sh 的算法；dev 池上试过 merge）：

```
$DATA_DIR/envs/op-train/bin/python experiments/route_follow/scripts/rf_pait.py merge --out RSGSELF --pools on1_L18RB0:r:self on1_SELF:s:both
WS=$($DATA_DIR/envs/op-train/bin/python -c "import numpy as np; t = np.load('$DATA_DIR/runs/op_parity/cache/on1_RSGSELF/tab.npz'); s = t['pool'] == 's'; print(round(0.3 / 0.7 * (~s).sum() / s.sum(), 4))")
# trainer: --on on1_RSGSELF --on-mode mixed --on-weight pool=s:$WS
```

on1_L18RB0 有 17 429 行，所以 W = 7 470 / n_SELF（n_SELF = 1 000 时 W ≈ 7.5）。再加按 scene 封顶的因子时，像 rf_tschool.sh 那样把分母换成 SELF 行的因子之和。
