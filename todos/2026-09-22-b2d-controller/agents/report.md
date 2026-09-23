# CLI 与报告子代理工作记录

状态：实现与离线回归完成；2026-09-22。未启动 server，未做 git 操作。

## 文件与接口

- `scripts/b2d_run.py`：新增 drive/controller_preset/cruise_mps/controller_config/tm_seed CLI 与 subprocess 透传；校验正有限 cruise、正 decimate、JSON object config；controller 暂仅 policy=none / built-in agent。独立 invocation manifest 保留 requested route 与 CLI（兼容共享输出目录的多次启动）。
- `scripts/b2d_route.py`：相同 CLI 校验；agent config 写绝对 controller JSON 路径、out、seed；记录 tick cap 是否实际触发。controller 模式包装本地 StatisticsManager 实例的现有统计调用，把既有 criterion 事件 frame/type/message/percentage 写 `criterion_events.json`，不改变 evaluator 判分或驾驶。
- `scripts/b2d_report.py`：保留既有 per-town stdout、`--json` 行为，新增 `--controller-json` 完整结构；CSV 所有 attempt 与未启动路线占位，展开主要 tracking/pose 指标。
- `scripts/test_b2d_controller_report.py`：10 个离线回归用例；CARLA 不需要启动。

## 报告语义

`status/harness_finished` 与 `official_status/completion/driving_completed` 独立；`results.json` 缺失、半写、空 records、多 records 不被判成功。所有重试保留；正式选择首个 harness finished，若无则最新 attempt，不能取最佳分数。请求但未启动路线保留在分母，attempt 数不计占位。配置 tick cap、实际 capped、partial 分列。

MinimumSpeed 事件数与 percentage 原样报告；固定 0.0.4 的模式标 unused，整体 penalty 读取官方 score_penalty，单项 penalty 未序列化则 null，不从事件推算扣分。

truth route cross-track、estimated route cross-track、controller diagnostic cross-track 三者分开。速度相对 command/reference 两种 RMS；raw/fused pose median/p90/p95、heading、age、controller CPU p99；所有指标保留样本数及 sum_squared 便于重算。truth frame 不匹配的定位/真值样本不计入精度。JSONL 半写、非单调帧、漏帧、传感器 frame 不匹配、无效/互斥控制分别计数；missing truth 不会落到 estimated。

首碰撞前指标使用 criterion snapshot frame；旧结果缺事件帧时明确 unavailable。全程、首碰撞前均给 tick 加权 RMS 与 route 等权 mean RMS；完整/未完成路线的 ticks 分布分开；low target-progress 连续 ≥5 s 片段单列。全部 attempt wall 与 retry wall 保留。

## 验证证据

执行：

```text
DATA_DIR=/data /data/envs/carla/bin/python scripts/test_b2d_controller_report.py
Ran 10 tests in 0.008s — OK
DATA_DIR=/data /data/envs/carla/bin/python scripts/test_b2d_runtime.py
Ran 9 tests in 0.216s — OK
```

最初 runtime 的 __new__ agent fixture 暴露 `set_global_plan` 过早读取 drive；已通知 agent 子代理修复，再跑 9/9 通过。

`agents/report-fixture-summary.json` 为合成输入结果：3 条请求路线、3 次真实 attempt（1 次重试）、1 条未启动；1 条 completion=100，另1条 finished 但 completion=31，观测完成率均值 65.5，不把缺失第3条当成功。文件显式标记 synthetic，不能当实测成绩。

## 边界

实验显式使用 `--decimate 4`：现有 CLI 默认 1 保持兼容。报告没有杜撰未记录的 setup/cleanup 分项或 stop-position；当前 cost 由已有 profile、wall、duration_game 与 control telemetry 给出。Criterion snapshot 在 evaluator 正常计算统计时抓取，强制 SIGKILL 的原始 event frame 可能仍不可得，此时报告 unavailable。尚无 live controller 数据，根代理负责实测整合、冻结默认与归档。

## 根代理追加分工：research修订

已更新 `research/trajectory-to-control.md`：修正aim_y与法向cross-track、CARLA加法lookahead、FF+完整bearing PID重复转向、绝对时间速度裁剪、CARLA/TCP离散语义、MinimumSpeed unused、finished≠completion与成本预算边界；替换原“半天/必能跑完”建议为共享计划G1–G4。写入实际MKZ轮距/后轴偏移/70°/km/h转向曲线，直接读calibration-units JSON核对；明确10 m/s右转sweep速度塌陷不可拟合，不宣称真实plant精确或Dev10已通过。

## 根代理追加分工：复用 CARLA server 的 Python API

`Runner(a, routes, servers=None)` 现接受每 worker 一个已有 `Server` 对象。传入列表长度不匹配或同一对象重复时，在创建输出文件前拒绝；未传参数的CLI行为保持原样。外部所有的server在成功/正常退出后继续存活，跨preset/seed顺序复用；campaign调用者负责最终 `stop()`。worker取消、异常或重试耗尽时仍停止；重试move与recycle按现有策略执行，可改变该对象端口/进程。重试后成功会返还仍活着的新server。

`Server`构造时创建自己的log目录；Runner回收孤儿时排除传入且仍活着的server PID，避免同log目录顺序复用误杀。外部server开始工作前照常探测可用地图。**同一个Server不可交给多个同时运行的Runner，也不可在Runner运行中由campaign并发操作**；API为同进程顺序campaign设计，不宣称跨进程所有权锁。

新增10项假server测试覆盖两个runner顺序复用、默认最终cleanup、失败恢复move/restart、失败耗尽、取消/异常、stop flag、recycle、PID保护、参数校验及目录创建。2026-09-22本地回归：report suite 20/20（0.022s），runtime 9/9（0.217s）；未启动live server。

## Dev10多组比较与速度语义核查

新增 `scripts/b2d_controller_compare.py`，读取多个 `--campaign` 路径下的group report；未完成group从partial attempt只读重建。manifest里预定的preset×seed×route全保留，未开始pursuit也显示10条缺失。输出`comparison.json/md`、正式`routes.csv`及保留全部重试的`attempts.csv`。按相同route XML SHA256和seed形成配对，只在carla/tcp两组均有完整官方结果后给“最强观测参考”；按均值completion、真正完成条数、原始blocked+deviation、carla最终平局规则排序。这不自动宣布默认，因为G1/G2、两seed、保留集及失败归因仍另验。重复来源不挑最高分；G2各summary独立保留全部case和失败gate。

JSON/CSV包含全程/首碰撞前truth CTE、两种速度error、raw/fused pose、heading、age、step cost、ticks成功/失败分布；缺失infra计数留null与覆盖数，所有已完成官方失败仍留在分母。source report路径/SHA、campaign记录commit、route SHA、配置当前SHA和reporter源码SHA均保留；当前hash不冒称启动时snapshot。

根代理指出B2D内建4000 tick上限：`Failed - TickRuntime`现单列`official_tick_runtime`，`harness_capped`仅表示人为profile截断，`capped`为两者或。根代理保留旧report、决定versioned重算；compare可直接从旧版明确官方status升级这一标记。**官方TickRuntime仍是最终驾驶失败，不因为capped而从completion对照剔除**；人工harness cap阻止完整对照选择。`b2d_report.py`另修缺失infra list不再默认0。

速度核查证据：carla-seed0/3514第一条trajectory[0]=[3.474663,3.376833]；后续点间隔2m，固定巡航8m/s，但controller加入t0=[0,0]后首段速度为norm(first)/.25=19.38m/s。每.2s重规划使reference导数长期读取这个首段，横向/后轴错位可抬高速度命令。3514全程actual/reference/command均值6.39/8.97/8.61m/s；blocked的25424分别.26/8.94/8.66。报告数字算术正确，但reference是controller输入轨迹的定时导数，不能叫独立的8m/s巡航真值；G2 validator独立cruise指标另列。根代理确认不在冻结Dev10中途修改adapter/controller或回算“修正”旧指标。

验证：compare新5项fixture通过（0.007s），report累计21项通过（0.021s）。已生成`results/comparison-interim/`中间快照，当时carla10/10官方结果、TCP9/10、pursuit0/10，参考选择ready=false；这是活跃运行中间观察，不代表最终对照。

## 离线原始trace与论文图（根代理追加）

`scripts/b2d_controller_selftest.py`新增独立CLI `--trace-dir`（也可调用`run_suite(trace_dir=...)`）；原`b2d_controller.py --selftest`入口保持不变，不接受新flag。trace root必须不存在，case/manifest/summary文件exclusive-create，不覆盖旧运行。case ID由固定suite顺序+kind+完整默认参数SHA构成，重复参数case仍有不同ordinal，不相互覆盖。

每个trajectory test tick保存后轴plant前/后状态、motion观测、生成world/local轨迹与source/release时间、实际接收trajectory、command/applied control（含转向延迟）、controller diagnostics、独立解析error。circle的CPU timing仍只包住step；写盘发生在计时外。起点、延迟、plant与指标公式不变；没有用trace指标替换原结果。静态摩擦两项标量unit exercise仍在summary内，它们没有虚构成20Hz轨迹测试。

运行产物：`/data/runs/b2d/controller/offline-traced/`，49个trajectory case、13726条逐tick记录，约25MB；`manifest.json`/`cases.jsonl`含身份与hash，`summary.json`含原aggregate，`source/`保存本次执行的selftest/controller精确源码。运行仅数秒，不需要tmux，不涉及CARLA。

`verification.json`证据：与共享`results/offline-selftest.json`递归比较，仅排除`controller_p99_ms`，其余全部值**精确相同**（非近似）；原summary SHA256=`1309de125f9bcc6034e6d9daca79c449fe9483a7247671a47b78be81774d7043`。49个文件hash/行数/单调tick、前后plant state连续性验证通过；从原始行重算圆弧/S弯lateral RMS与stop-position通过；重复同trace-dir CLI明确退出2拒绝覆盖。

新增`scripts/b2d_controller_offline_plot.py`，只读trace并核对SHA，生成publication PNG/PDF与provenance，figure目录也必须新建。实际图在`offline-traced/figures/offline-ablations.{png,pdf}`，已视觉检查：上排pursuit near/future window停车速度与位置error；下排左右圆弧PP与完整bearing-PID相加的原始误差及固定2s transient后的RMS。所有曲线来自保留的raw tick，柱高直接取原summary；图明确标为synthetic而非CARLA实测。左右镜像曲线重合但均保留，使用实/虚线与各自柱子，不删其中一侧。
