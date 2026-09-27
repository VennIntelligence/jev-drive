# 统一持久控制器（2026-09-27）

控制器是 box 上每 30 秒运行的固定政策程序，入口 `scripts/cx_controller_supervise.sh`；不需要模型持续轮询，不依赖 SSH 长连接。它管理已有 B 队列的授卡和失主恢复，不扩充科学任务。`scripts/cx_controller_registry.json` 明列 A、B、C、D、GK、P3、OPL、maintenance，以及原独立组合 17 项的真实完成证据。

## 所有权与动作

- supervisor（监督进程）持有自身 flock；controller（控制器）同时持有 `runs/sched/controller/controller.lock`、`runs/sched/owner.lock`、原 dispatcher（派工进程）的 `runs/nq4/cx/orchestration/dispatcher.lock`。另一控制器或旧 dispatcher 无法争写。`sch_table.py grant/revoke` 也必须取得全局 owner 锁；`show/check` 仍可随时读。
- A、C 及 B 的链和输出仍由原进程独占。用 PID、Linux startticks（进程启动时刻）、进程后代、B 的实际输出目录认领；已变成孤儿的 B runner 也阻止重复启动。完全没有 resident（仍驻留的）进程才归还资源；完成依赖另需登记 DONE、结果及非空产物。组合结果要求原 `result.json` 的 `status=DONE, rc=0`，支持原 glob（文件匹配）和替代产物语义。
- A 的 CPU 后处理失败不会保留已经退出的 GPU 进程资源；C 的科学依赖仍阻塞。A 的 DONE 更新后旧 ERROR 不可盖过它，较新的 ERROR 则优先阻断完成。读取异常或半写 JSON 是 UNKNOWN（未知），禁止该任务撤卡与恢复。节点盘点失败也 UNKNOWN，不把失联当完成。
- B 按已有脚本的固定队列和 PASS / APPROVED 执行，controller 不创建第二个 route/output owner。恢复登记的双卡旧授卡时，允许 `60-109` 转为已审的 `0-47,60-109`，完整发布五卡 `0,2,3,4,5 × 6`、原 index map（端口索引映射）、CPU 和 expansion 参数，保留其他 GO extras（附加设置）。先核原登记、CPU affinity（可用核集合）、线程/显存实际占用加未实现 worker 需求、SCH 端口冲突，再一次原子替换完整 GO。
- B 已授卡与真正接卡分开：必须 CURRENT（当前臂/seed）对应的真实 runner 身份、workers 参数和目标卡的实际 GPU 进程均出现才写 `RUNNER_AND_GPU_ACK`。启动时仍在旧 bridgedrive 正常运行，只写 `WAIT_EXISTING_ARM_BOUNDARY`；跨入下一臂后 300 秒仍未接齐卡，写具体 durable inbox（持久告警收件箱），不自动再启动第二条链。
- 唯一控制器级自动重试是原交接已登记的「B owner 曾被真实认领、无 ERROR / DONE、所有后代/孤儿均退出」情况：连续两个盘点至少 60 秒，门禁与容量仍有效时最多重启一次，重试次数先落盘。未知 ERROR、CL10 ERROR、科学 assert 不重试。controller 本身异常退出由 supervisor 以 5 至 60 秒退避重启；进程重启不会重置 B 的预算。
- 不运行 GK、P3、OPL；不制造 F/main review、prep、科学 PASS；不改阈值、live chain（运行中的链脚本）或结果。maintenance 原 owner 继续 T1/T9/T11/小表工作；17 项旧 dispatcher 已退出，控制器只认领其完成证据，不重启。
- 23:59 JST 后控制器不新授卡/恢复启动，既有 B 的已授权队列继续由原链运行；不杀现有工作。无登记可启动的工作明确 `NO_READY_WORK`；已 PASS 的 B 任务显示 `READY_DELEGATED_TO_B`。不承诺满载或全局最优。

## 运维与告警

持久目录 `$DATA_DIR/runs/sched/controller/` 下有 `state.json`、`heartbeat.json`、`events.jsonl`、`log.txt`、`STATUS.md`、`inbox.jsonl` 和 supervisor 日志。events / inbox 写入后 fsync；状态采用原子替换。没有自动唤醒 Codex / Monitor 智能体的机制，不能把写 inbox 说成已通知模型。root 下次接入第一步读取 inbox，对未处理事件逐条复核；旧记录保留，不伪造确认。

```bash
# box，启动一次
DATA_DIR=/root/autodl-tmp/ujs scripts/tmux_run.sh cx-controller bash scripts/cx_controller_supervise.sh
# 每次人工/模型接入首先读
DATA_DIR=/root/autodl-tmp/ujs python3 scripts/cx_controller.py inbox
# 只读查看实际状态
cat /root/autodl-tmp/ujs/runs/sched/controller/STATUS.md
```

13 项模拟测试覆盖 A post 失败资源退出但依赖未完成、独立任务隔离、UNKNOWN/节点失联不能撤卡、PID 重用、孤儿 runner、组合 glob/半写结果、重启/重复锁、deadline 保留进程、B 当前输出与真实 GPU ack、科学 gate 不造假、固定 PASS 队列、旧双卡到新授卡完整 GO 并保留附加设置、重试预算。命令 `PYTHONPATH=scripts python3 -m unittest scripts/test_cx_controller.py -v`；另有 Python 编译和 bash 语法检查。

## 部署证据

部署后追加真实 commit、box 状态、A 回收和 B 接卡状态。不能将模拟测试或 GO 内容当成实际五卡接卡。
