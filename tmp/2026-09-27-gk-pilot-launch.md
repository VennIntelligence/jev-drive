# GPU1 G/K 分级 pilot 启动（2026-09-27）

用户明确同意root建议的G/K分级pilot（小规模闭环检查）。此次范围是K0–K3优先的1→10路线，再推进原登记可独立G候选；不等于全量许可，不写prep/DONE/GO、不改科学判据。worldreuse（跨路线复用世界）保持关闭，shift仍原固定+15m。P3正式训练继续GPU6/CPU180–189。

10:51CST核查：原GK链及smoke/post均无活进程。B=SimLingo，GPU0/2/3/4/5满载；P3 GPU6约18.4GB/82%；全机pids18032超过计划16000（硬限20480）。GPU1旧Alpamayo587198/startticks797396286占24.6GB；socket仅LISTEN、无连接实例、无其他进程argv引用，B也不使用Alp。记录进程树后按精确PID分别停止它及9个compile子进程，没有group kill（按进程组杀进程）；记录在`runs/nq4/cx/gk-pilots/retired-alpamayo.json`。随后pids17203；仍不能因此越过16000容量门。

## 执行路径

- `scripts/cx_gk_pilots.py run`是有限pilot任务的唯一owner（执行进程），沿用已登记GPU1、CPU204–207、index150–159、最多2个CARLA。主统一控制器不变。
- 固定队列：K0/K1/K2/K3 unseen（未见该路线折）、PDM ghost、BridgeDrive ghost、BLUE ghost、Q2 ghost、PDM shift/swap；每个调用原`nq4_gk.sh pilot`，首阶段1路线，第二阶段10路线。TFv6/SimLingo/Cinque/mc历史失败没有原因修复，不擅自重跑。X已完成独立修复10路，不重复。
- 已有完整10路线pilot的原PASS可核后复用；2路线smoke不能当10路线完成。输出隔离在`runs/nq4/gk/cx_pilot_20260927*`，不覆盖旧失败证据。失败仅阻对应候选，结束即接下个独立候选。
- 每阶段重新检查实际CPU/pids/GPU1显存/调试CARLA数与端口、原SCH表冲突；把B现有claims减实际CARLA计数得到尚未兑现worker预留。`pids + 400×(pending_B+new_workers)+100 <=16000`；CPU≤165、GPU1≤88GB；不改硬限或b2d17000背压。容量等待不计入路线执行超时。
- 原`nq4_gk.sh`确认未运行后修其生命周期管理：在创建服务器/runner时记PID+startticks和已验证子树；停止使用pidfd逐个信号，禁止group kill/pkill，不碰外部所有者。旧identity缺失且PID活着则拒绝覆盖。僵尸进程不再被kill-0误认为未结束。
- `runs/nq4/cx/gk-pilots/{pid,state.json,STATUS.md,events.jsonl,log.txt}`持久记录，tmux继续无需模型轮询。重启认领原worker，禁止重复launch；未知/部分结果记TECHNICAL_WAIT，不生成PASS。

验证：4个针对性测试通过（B pending预算、PID复用不认领无关子进程、部分/失败不能PASS、2路线不能替10路）；bash语法通过。尚未由这些测试宣称真实pilot成功；部署实负载证据下面追加。
