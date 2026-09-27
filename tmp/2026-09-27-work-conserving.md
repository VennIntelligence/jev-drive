# B 尾部补位核查与最终队列兼容修复（2026-09-27）

## 当前事实

09:35–09:37 CST，B 仍是 BLUE seed0，仅 GPU0 的 runner764399/startticks800902333 剩两个 CARLA，GPU2/3/4/5 已空；pids2227/20480、实际CPU约5/175核。因此瓶颈确是整臂屏障，非资源不足。当时 SimLingo 的十路 pilot（小规模验证）已 PASS，可合法提前执行，但旧 `nq3_b.sh run_arm` 会删除下一臂 claims（路线占用锁），不能直接启动同输出目录 helper。

09:40:21，BLUE 自然完成220/220，原链953934自行进入最后一臂 SimLingo。没有执行接管或给这次自然提速记功。原链每20秒启动一张卡，每卡6个worker（路线执行器），全部仍用官方220路及原始CL10包装器、模型、参数、最大3次尝试。

| GPU | 实际 SimLingo runner PID | startticks（进程启动时钟，防止PID复用误认） |
|---|---:|---:|
| 0 | 935741 | 801410980 |
| 2 | 936497 | 801412980 |
| 3 | 937705 | 801414980 |
| 4 | 939009 | 801416981 |
| 5 | 940835 | 801418981 |

五卡开始后，现场30个live claims，BLUE的claims为0。旧链仍唯一持有B输出；GPU1/6、CPU180–189仍归P3唯一owner。B后继QUEUE已空。A/C与17项Q4b/Q5/Q6已完成；G/K仍缺原F/main和全量门，OPL禁止，Q4a原科学assert保留。没有为填空闲资源重复运行已完成组合或制造新科学许可。之前BLUE尾部的跨臂浪费并未修复；本次边界已过，追加一套无候选的调度器不会改善剩余工期。

## 已确认的技术异常与修复

最后一臂启动时，原B把空QUEUE写成单独换行。旧统一控制器01:40:34UTC开始报`ValueError: malformed original B queue`，使STATUS和heartbeat停止刷新、继续显示过时的READY_DELEGATED_TO_B。这不阻止原B继续，但妨碍持久控制状态。

新增 `scripts/cx_controller_v2.py`，继承原控制器，只忽略空白队列记录；非空畸形记录仍拒绝，PASS/SKIP/DONE逻辑保持。自带supervise入口沿用同一supervisor/global/controller锁与原state目录，子进程异常后5–60秒退避重启。没有改运行中的原controller/bash/实验脚本。

迁移只终止精确身份879209/startticks801166192（无trap的旧supervisor）与879213/startticks801166193（旧controller）。不向进程组发信号，全部B/P3 worker和原链保留。新控制器沿用 `runs/sched/controller/{state,events,heartbeat,STATUS}`，仍只有一个owner。

验证：`python3 -m unittest discover -s scripts -p test_cx_controller_v2.py`通过33项（原回归继承及3个新场景）：真实单换行QUEUE可完成tick、写出ONLINE/NO_READY_WORK同时保留RUNNING；混合空行不绕过PASS/SKIP/DONE；非空坏行仍报错。部署证据待下面追加。
