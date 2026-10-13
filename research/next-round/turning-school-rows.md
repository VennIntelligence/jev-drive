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
