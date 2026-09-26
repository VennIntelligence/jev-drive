# 独立组合队列编排

用户已授权专业编排智能体按既定组合无人值守执行，目标为 28 日之前完成可执行工作。这里只启动 Q4b / Q5 / Q6 的原命令，不动运行链，不解除 Q4a assertion，不写 D 整链 DONE。其他 A/B/C/Alp 与维护脚本继续原 owner；GPU1 与 CPU134–141 保留 X，P3/P2/OPL 不纳入本队列。

## 2026-09-26 22:14 CST 实施

新增 `scripts/cx_orchestrator.py` 和 `scripts/cx_orchestration.json`：17 个固定步骤，Q6 WOD/NAVSIM 后处理与 devkit 用 CPU180–191，Q4b Hydra/compat/exam 与 Q6 single-frame fits 按 CPU192–199 自然串行；Q5 原 child 占用180–199，等前槽释放后接续。GPU-only fits 选择非1卡，按即时显存、利用率和尚未启动的 SCH worker 承诺预算准入，留每卡8GB。

持续进程每45秒检查，27日23:59 JST 停止接新工作。进程身份包含 PID/start_ticks/PGID；每个输出有 flock claim，dispatcher 有独占锁，状态和结果原子落盘，完成须成功返回和全部产物齐全。重启复用 DONE，失败只使相应依赖 WAIT，其他独立支路继续。只对自己的进程组处理超时。已有缓存先核 provenance MD5，再读实际输入数组和 token/frame 覆盖；Q6 sf-check/extract/heads 和 Q4b 特征不重抽。已存在原 DONE 还须产物齐全才能跳过。

CPU实时用量与亲和分别检查；GPU1不授batch；不新增CARLA，不申请端口，不自动grant表。线程预算包括现有 pids、SCH尚未起server、X额外600和本队列已准入/候选线程；非CARLA准入上限 min(pids.max−1480,19000)，原16000 CARLA启动门不改。GPU pending/active预留会保守重复计部分当前显存，宁可短等，不挤压现有批量。

本机5项检查通过：claim排他、完成标记缺产物不得跳过、CPU释放后交接/GPU1排除/待起显存预算、PID复用排除、失败支路隔离且已完成不重跑。测试不改科学条件。等待 box 同步后实际运行，PID/首批命令将在下节补。

## 实际启动与初次资源修正

`7aa0e7f` 已main push/box pull。首次 dispatcher PID537142 没有准入任何子进程：快照发现 CPU196 被只读 GPU watcher PID840028 占用，SCH旧表还保留 K 已DONE 与 OPL 已ERROR 的 pilot 承诺。没有忽略这两个门或停止 watcher。

修正 `3cbe48d` 已main push/box pull：Q4b/Q6 GPU槽改192–195,197–199，Q5用180–195,197–199；只有 K 的 cl/DONE+READY 或 OPL ERROR 已存在且相应runner确已不在时，从本队列的只读容量计算排除该旧pilot承诺，原SCH表不改。现存CARLA始终计入实时probe。首次无子进程的dispatcher按记录start_ticks核对后退出，manifest旧hash保留。

正式窗口 `jev:cx-orchestration-v2`，dispatcher PID542401。14:17:57 UTC 已实际准入：Q6 inputs worker PID543399，Q4b inputs worker PID543400；两者分别使用CPU180–191和192–195,197–199，CUDA隐藏。运行目录 `$DATA_DIR/runs/nq4/cx/orchestration/{pid,identity.json,log.txt,events.jsonl,STATUS.md,state.json,resources.json,jobs/}`。每个job有自身命令PID、日志、结果与完成标记。Q5等待CPU槽释放。正式计算的首批PID在下条补记。
