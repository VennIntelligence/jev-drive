# 不明 SIGKILL 取证（2026-09-25）

状态：来源**未能确证**。最可能的是平台在 kernel OOM 之外的内存执法（杀容器里最大的进程），证据是间接的；
我们自己的代码和 Mac 上所有 agent 的命令都排除掉了。已加固两处能误杀别人进程的 reaper，已加取证采样，
下一次 kill 能直接看到死前几秒的内存和最大进程。

## 结论

两次 kill 形态完全一样：**单个 pid** 收到 SIGKILL（signal 9，不能被捕获或拦截），同一 process group 里的父 shell
都活着，而且事后读到 kernel 的 OOM 计数是 0。被杀的都是当时容器里 anonymous memory（匿名内存，即进程自己分配、
不能像 page cache 那样回收的内存）最大的一类进程：加载了 openpilot Cinque 模型的 Python 进程。两次都发生在
容器内存贴着 `memory.high`（cgroup v2 的软上限：超过后 kernel 限速并回收，但不杀进程）、并且刚有大内存作业启动的
几分钟内。

排除得比较干净的有：kernel OOM（memcg 的和整机的都算）、我们仓库里所有会杀进程的代码路径、Mac 上所有 Claude agent
在两个时间窗里发出的命令、RLIMIT、pids.max、SIGHUP / tmux。剩下两个解释：平台的内存执法（排第一），和我们看不到的
人工 `kill -9`（用户自己的终端、Tokyo 上的 Codex 等，排第二，用户已说不是他）。没有 root 就拿不到 SIGKILL 的发送者
（容器里没有 dmesg、audit、eBPF、tracefs；父进程的 wait status 只有信号编号，没有 `si_pid`），所以只能靠相关性定案，
下面的取证就是为此加的。

## 两次事件

| | 2026-09-25 约 02:15–02:22 | 2026-09-25 13:41:09 |
|---|---|---|
| 被杀进程 | `zeroshot_policy_server.py cinque`（pid 840644），op-b2d-smoke 第 4 阶段后 | `nusc_backbone_openpilot.py` 主进程（pid 444250），decision40-nusc-op |
| 死法 | `zeroshot_b2d_op.sh: line 50: 840644 Killed`；脚本本身活着，按预设判 rc 3 | `decision40.sh: line 7: 444250 Killed`；`decision40.sh` 和 `slot_run.sh` 都活着，写了 rc 137 |
| 同组其他进程 | 未受影响（server 当时没有 setsid，和脚本同组） | 未受影响（同组还有 tmux 窗口的 shell、slot_run、decision40.sh） |
| 事后读的 cgroup 计数 | 02:26：`memory.events` high 9 728 539、max 0、oom 0、oom_kill 0 | 13:59：max 0、oom 0、oom_kill 0 |
| 整机 `free -g`（host 视角） | 1007 GB 总量，612 GB free | 1007 GB 总量，154 GB free + 546 GB cache |
| 当时的容器上限 | 240 GB（19:45 重启后） | 360 GiB（09:40 重启后），`memory.high` 比 `memory.max` 低 2 GiB |
| 几分钟前启动的大作业 | 02:10 `alp-b2d-full`（Alpamayo server + 4 CARLA） | 13:34 decision40-nusc-op 以满速重启（主进程 + 4 个 worker）；13:19 simlingo-exp-extra，13:28–13:32 p5route 三个 slot |

两次的计数都是在下一次容器重启之前读的（重启会清零计数），所以 "oom_kill = 0" 是有效的。

## 逐项排除

| 假设 | 判定 | 证据 |
|---|---|---|
| memcg OOM（容器自己的 `memory.max`） | 排除 | `max` 事件 0：用量从没碰到 `memory.max`；memcg OOM 会同时记 `oom` 和 `oom_kill` |
| 整机 OOM | 排除 | kernel 5.15 的 `__oom_kill_process` 会在受害者所在 memcg 记 `oom_kill`（仍是 0）；且 host 当时有 154–612 GB free |
| 我们的 group kill（`os.killpg`、`kill -- -pgid`，含 pid 复用） | 排除 | 两次同组的父 shell 都活着并写了退出码；group kill 会连它们一起杀 |
| 我们的单 pid SIGKILL（`b2d_tfv6_campaign.clean_owned_server`） | 排除 | 只杀命令行含 `CarlaUE4` 的 pid，两个受害者都不是 |
| `b2d_run.reap_orphans`、`zeroshot_b2d_op.sh cleanup()`、`hugsim/zs_run.py`、`carla_server.sh` | 排除 | 都是 group kill 或 SIGTERM，且按命令行/pid 文件筛过 |
| Mac 上的 Claude agent（主会话 + 全部 subagent 的 transcript） | 排除 | 01:50–02:30 和 13:32–13:46 里没有任何 `kill`；唯一相关的是 13:33 decision40 agent 对**旧**实例的 `kill`（SIGTERM，rc 143），新实例 13:34 才起 |
| 容器里的 shell 历史 | 没有线索 | fish history 自 09-24 起没有 kill；bash_history 为空；13:11 有一次交互登录（pts/13） |
| RLIMIT_CPU / RLIMIT_AS（超硬限制会被 SIGKILL） | 排除 | tmux server、ssh shell、运行中作业的 `/proc/<pid>/limits` 全是 unlimited |
| pids.max | 排除 | fork 失败不是 SIGKILL；`pids.events` max 0 |
| tmux kill-window / ssh 断开 | 排除 | 那是 SIGHUP（rc 129），并且会带走整组 |
| GPU driver Xid / CUDA 致命错误 | 排除 | driver 不发 SIGKILL；CUDA 致命错误是 abort（rc 134） |
| **平台内存执法**（AutoDL host 侧 agent，或容器里以 root 跑的 `autopanel`） | **最可能，未证实** | 见下节 |
| 看不到的人工 `kill -9 <pid>` | 可能性低 | 形态吻合；但用户说不是他，Mac 上的 agent 都没发 |

## 为什么排第一的是平台内存执法

支持的证据：

1. **受害者是最大的进程。** 现在实测（`/proc/<pid>/smaps_rollup`）：每个加载了 Cinque 的 openpilot 进程占
   **约 27 GB 私有 anonymous memory**，Lebowski 约 5.4 GB。13:19 的 ps 快照里，最大的进程依次是 nusc-op 主进程
   55–64 GB、HUGSIM 的 Cinque server 27–51 GB、nusc-op 的 4 个 worker 各 27 GB（fork 来的 copy-on-write，与主进程共享）。
   02:15 被杀的是 Cinque policy server，同一类进程。这正是 kernel OOM killer 挑受害者的方式（按 RSS 打分），
   但 kernel 的计数说它没动手。
2. **时间点都在内存贴顶的时候。** 12:19–13:19 每 30 s 的快照里，`memory.current` 一直在 360–384 GB，
   `memory.high` = 384.4 GB，high 事件从 15.2 万涨到 37.1 万；02:26 时 high 事件已经累计 973 万（240 GB 那一期）。
   两次 kill 都在大作业启动后几分钟：02:10 起 Alpamayo 全量，13:34 起 nusc-op 满速重启（主进程再加载 Cinque + Lebowski，
   worker 各自再载 index），期间 simlingo-exp-extra 和 p5route 也刚起。
3. **平台显然在管内存。** `memory.high` 被设成 `memory.max − 2 GiB`（重启前后都是），容器里所有进程的
   `oom_score_adj` 都是 999；AutoDL 帮助文档只说"程序占用的内存容量超了被系统终止"，不说用什么机制。
   容器里有一个 root 进程 `autopanel`（AutoDL 的资源面板），我们没有权限看它在做什么。

反对的证据：

1. 12:19–13:19 整整一小时 `memory.current` 贴着 `memory.high`，却没有 kill。所以触发条件不是 "用量 ≥ high"。
   这段时间的用量里 page cache 占很大一块（现在是 anon 321 GB + file 232 GB），真正不可回收的 anon 可能离上限还远。
2. 13:41 那一刻的 anon 用量没有记录（op 的 watchdog 13:19 就随 smoke3 结束了），所以 "anon 贴顶" 只是推测。

一个可以检验的推论：如果执法看的是 anon（或 PSI "full" 压力），那么下一次 kill 之前几秒，boxwatch 会看到 anon 接近
`memory.max`、`memory.pressure` 的 full 累计在涨，而受害者是 top-1 RSS。若 kill 发生在 anon 远低于上限时，这条假设就倒了。

## 已做的加固（本次提交）

**1. reaper 的 pid 复用问题。** 这个盒子 `pid_max` = 1 000 000，而 09:40 重启后不到 4 小时 pid 就用到了 93 万，
13:34 的新进程已经是 44 万：**pid 每几个小时就回绕一次**，pid 文件里记的旧 pid 很快会落到别人的进程上。旧代码有两个
能因此误杀别人的地方：

| 位置 | 旧行为 | 风险 |
|---|---|---|
| `b2d_run.kill_group`（`reap_orphans` 用） | pid 活着且命令行含 `CarlaUE4` 或 `b2d_route.py`，就 `killpg(getpgid(pid))`，先 TERM 后 KILL | 复用这个 pid 的若是别人的 CARLA / route，或任何命令行里提到 `b2d_route.py` 的 shell，就杀掉**它所在的整个组**（可能是别人 tmux 窗口里的整个 slot） |
| `b2d_tfv6_campaign.clean_owned_server` | 命令行含 `CarlaUE4` 就对单 pid 先 TERM 后 KILL | 同上；而且只杀 wrapper，不杀 CARLA 本体 |

新做法（`b2d_run.owned_group_members` / `kill_owned_group`）：我们启动的进程都是 setsid 出来的组长，pid 就是 pgid，
且 stdout 指向 `--out` 下的日志、子进程继承同一个 stdout。只有当 pgid 组里**仍有 stdout 指向我们 `--out` 的进程**时
才杀这个组，每发一次信号前都重新确认。命令行匹配不能证明所有权，stdout 的目标可以。顺带修掉一个旧漏洞：
wrapper（记录的 pid）已经退出、CARLA 本体还活着时，旧 reaper 因为 pid 不在了就放过它，新 reaper 按组找得到。

测试（`scripts/test_b2d_reaper.py`，在盒子上用真实进程跑）：自己的组在组长先退出后仍被清掉；一个 argv 伪装成
`CarlaUE4`、占着记录 pid 的外人进程不被杀（旧代码会杀它，已验证旧代码在这条上失败）；一个不是组长的外人进程
（旧代码会 `killpg` 它所在的组，也就是测试进程自己的组）不被杀。`test_b2d_controller_report.py` 21 个用例照过。

**2. 取证。**

- `scripts/boxwatch.sh`：整机一个实例（flock），每 5 s 在 `$DATA_DIR/runs/boxwatch/<日期>.tsv` 记一行：
  `memory.current`、anon、file、`memory.events` 的 high / max / oom_kill、`memory.pressure` full 累计（µs）、
  `pids.current`、按 RSS 排的前三个进程；每 60 s 存一份完整 ps（保留 3 h）。开销是每 5 s 一次 `ps`，每天约 4 MB。
- `scripts/slot_run.sh`：每个 slot 启动时顺手拉起 boxwatch（已在跑就立即退出）；作业被信号杀死（rc > 128）时写
  `$DATA_DIR/runs/sched/<slot>.death-<HHMMSS>.txt`：退出时刻（毫秒）、cgroup 计数、`memory.stat` 的 anon/file、
  最大的 15 个进程、boxwatch 最后 2 分钟，并把路径写进 gpu-plan.md 的 FAILED 那一行。

## 建议下一步

1. **问 AutoDL 客服**：容器内存超限时平台是否会主动 kill 进程；用什么指标（`memory.current`、anon、PSI）、
   什么阈值、挑哪个进程。这是唯一能直接证实第一假设的途径。
2. **下一次 kill 出现时**先看 `<slot>.death-*.txt` 和 boxwatch 那几分钟：受害者是不是 top-1、anon 离 `memory.max`
   多远、pressure full 有没有在涨。连续两次吻合就可以定案。
3. **Cinque 每进程 27 GB 的 host 内存值得单独查。** 现在 8 个 navsim openpilot shard 就占了约 224 GB anon。
   它让每个 Cinque 进程都成为最大进程，也就是最可能的受害者；如果是 ORT / TensorRT 的 host 侧副本，可能省得下来。
4. 不建议在共享盒子上做 "把 anon 顶到上限看谁被杀" 的受控实验：执法者若按 RSS 挑人，被杀的可能是别人更大的进程。
   要做只能在盒子空闲、测试进程明显最大时，由 main 批准。
5. 排期上把容器 anon 控制在 `memory.max` 的约 85% 以内（按每个 Cinque 进程 27 GB 记账），在找到原因之前能降低再被杀的概率。
