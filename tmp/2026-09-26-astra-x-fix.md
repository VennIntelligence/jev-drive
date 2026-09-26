# X 自主修复（最多 10 轮）

执行员 Astra；用户于旧审计后明确授权自主技术修复、最多 10 轮，保留好的验证结果。科学判据、1.0 m/s 门、冻结 P7 参数、其他考生和运行链均保持。目录 `$DATA_DIR/runs/nq4/cx/x-fix/roundNN/`；独立 socket、PID、配置、输入 SHA-256、log/events/tb。root 只协调与验收。

## round01：补齐独立纵向目标接口

假设：head 原始几何本身可用；静止是 X 把 stop 改成全零路径造成。需要保留真实 head/shift 路径，给冻结 P7 执行器传独立纵向目标，不能仅取消 stop 分支后假装已经用了巡航。

实现：新增 X 专用 `XController` 子类；不改 `scripts/b2d_controller.py`、P7.json 或基线 agent。无独立目标时，控制值与诊断逐位同原 P7；独立目标时速度参考为常量、前馈为零，保留 P7 反馈、限幅、jerk 与安全检查。目标只随被接受的新 plan 生效；重复/乱序 plan 不得覆盖。禁用独立目标期间的 terminal approach，防止它把明确的 0 重新抬为 1.5 m/s。

语义取舍（按用户目标记录，非另定门槛）：

- 当前 speedometer 经原 `controller_speed` 转换的速度，在每次 5 Hz head 计划时作门控。raw stop/wait 且速度 ≥1 时目标为 0；低于 1 时 effective mode=keep，目标取现有 route cruise 配置（未提供时沿用父 agent 的 8 m/s 默认）。
- “路径不变”按当前 head 路径或尚未结束的 shift 路径逐点保留理解；高低速 stop 都不再写全零路径。keep/bypass 仍使用原 learned timing。没有可用几何时保持 P7 的安全刹停，不擅造路线或取消门槛。
- dump 同时记 raw/effective mode、门控速度、巡航值、目标、head 路径和实际传给 P7 的路径；离线同一 XState 重放需路径与元信息逐位一致，head check 仍校验 raw 模式，fold 由原 split 校验。

本机命令：`.venv/bin/python scripts/nq4_x_selftest.py`。PASS：9 个速度边界/平移状态与路径不变例、X dump 离线逐位重放、100 tick 与冻结 P7 的默认输出/诊断逐位相同、独立 0/8 m/s 目标及零前馈、重复更新不污染目标、无几何安全刹停。compileall 通过。此处只是真正的合成回归测试，不是 CARLA 或科学结果。

隔离验证脚本 `scripts/nq4_x_fix.py --round 1`：GPU1 容量空余后在 table.tsv 登记合法 3-index 重试段，检查 i±120 与实际 ss、sch_table check=0；启动前 pids+600≤16000、GPU1 CARLA<4、至少约24 GB空余。先正式 Q2 xfit（源 READY 后，如已存在则只读），1 route=2534 orig seed0；通过原 pilot checklist 后运行其余登记 rule8 路线 2668、1790，然后十条 obstacle 路线、原 checklist 与逐位检查。不会启动 full batch。

资源快照：初次读 pids16370/20480、GPU1六个 CARLA，`sch_table show` 返回1；不能启动新 CARLA。后续读 pids16270、GPU1用46.9GB。仅建设本机代码，没有停止任何已有进程；待 gate 放行后才计 CARLA 数字。当前暂无完成/崩溃/blocked/起步读数。

部署：`de39c51` 已 main push / box ff-only pull，box 同样回归 PASS；`jev:cx-x-fix-r01`，wrapper PID272666。首个 gate 快照 pids16149、GPU1五个 CARLA，因此自动等待，没有登记新 index、启动新 head 或 CARLA。等待不计一次科学失败，round01 仍进行中。

round01 的实际结束：资源稍后放行，但 box 没有 `ss`，端口检查调用在启动模型前抛 `FileNotFoundError`；wrapper 已退出。`round01/result.json` 明确 `infrastructure_error, model_verdict=false`。CARLA/模型请求数都是 0，不能报成模型起步失败。

## round02：补端口盘点，先完成 GPU-only 正式 head

假设/修改：round01 的控制实现本机与 box 均过回归；唯一实测失败是运维依赖。端口读数在有 ss 时仍用 ss；缺失时读相同内核的 `/proc/net/tcp` 与 `tcp6`，并包括所有已绑定 TCP 状态，比只查 LISTEN 更严格。i±120、index≤494、sch_table check 与资源阈值不变。根据用户指出“非 CARLA 的 GPU 任务可以先跑”，正式 Q2 xfit 导出移到 CARLA 容量门之前；仍按 GPU1 显存、CPU、pids+64 门和 q2.lock 防重写。独立 formal export 的输入/输出检查没有降低。

最新外部机械快照（Sol capacity）：pids15300，GPU1约10.5GB/18%、1 CARLA；CPU约95/175核。round02 仍自行复核，不把单次 utilization 当启动依据。round01 的等待/错误产物保留；round02 新目录与 PID。

round02 已部署（f6d9bc3，box回归同过）；wrapper343121、formal export344728，GPU1、204–207。导出自身门：pids15798+64≤16000、GPU1用15.64GB、CPU约92核；此时已有四个CARLA，未追加CARLA。正式导出约2分钟完成，READY为Cinque/A1轨迹/A2模式：R1训练13234行/2214对，R2训练13506行/2090对；各1024行导出检查，轨迹最大差分别2.8610e-5/2.6703e-5 m，模式误差均0，原1e-3门通过。这是导出检查，不是闭环rule8。导出后返回CARLA容量门；wrapper现有67线程已全部pin到204–207，子进程本来已pin并限制OMP/BLAS2。CARLA仍需自己的pids+600与<4现有server条件。

round02 闭环结果：自动容量门在pids15062、GPU1两个既有CARLA/19.22GB时通过；table登记index0–2、核204–207，内核端口盘点与sch_table check=0。head402033在64秒后READY，runner405595开始253490。三个尝试分别在129.1/67.0/68.0秒后server_died_rc139，**0/1完成、3/3启动崩溃、0 tick、0 agent请求**；引擎日志均属`GameThread timed out waiting for RenderThread after 60.00 secs`。这不是模型不起步、不是blocked驾驶结果；保留原始日志，result仍明确infrastructure_error/model_verdict=false。自己的进程已清理、表行释放，未改看门狗或门槛。

## round03：给 render startup 八核，保留同一控制候选

假设：单模型+CARLA/client挤在四核，可能触发文档记录的render-thread饥饿（`docs/bench2drive-cost.md`和todo K18:37；原因不是已证明的）。仅改变运行CPU资源：live affinity审计确认134–149没有窄affinity活动进程，Sol确认对应pull-forward任务已DONE，因此用134–141八核；不借D的180–199，保留D待修复资源。parent/root和Sol capacity均已通知。

修复wrapper的BLAS线程池上限（原wrapper67线程，子进程已是2），并在head加载之后和每个route stage真正起CARLA之前重查容量，模型已存活时按SCH原每worker400线程预算；总plan cap仍16000，GPU1至多再加到4个CARLA。启动命令将wrapper本身也taskset134–141、OMP/BLAS2。模型/路径/门控/看门狗/世界与判据一概未变；既有formal q2 READY复用。重新单路253490，不跳到pilot。

round03 的单路已通过：wrapper448786、GPU1/index0–2/CPU134–141，初门pids15131、三个已有CARLA；head加载后的真正开车门再次通过。253490首个attempt完成，668 tick、167次head请求（含warmup）、142次X执行，最高8.0276 m/s；完成1/1，启动崩溃0，blocked0，stalled0，原stage1 sanity PASS。142次X路径/元信息离线逐位一致（最大差0）、head check返回0、unseen fold正确。实际11个低速stop目标均8.0 m/s，4个高速stop目标均0；raw模式keep19/stop15/bypass_L108。这里终于有真实起步与两侧门控证据，前两轮仍只记运维/导出结果。

仍不能写“X能力判格通过”：此刻只完成登记的2534这一条；另两条2668、1790的rule8和约10路sanity在同一round继续自动执行，full从未启动。控制候选从de39c51保持不变；aa4133d只是运行资源与安全复核。小表：`research/results/nq4/x/fix/round03/stage1_pilot.json`、`stage1_behavior.json`。
