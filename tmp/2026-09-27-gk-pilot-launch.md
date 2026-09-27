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

## 部署与首次现场状态（10:59 CST）

`8bbcda1` main已push、Box ff-only pull。两端4项针对测试通过；另在Linux实际创建父/子进程和同进程组无关sentinel（哨兵）执行owned stop测试，父子精确退出，sentinel存活。没有按组发信号。

已部署`jev:cx-gk-pilots`：owner97841；K0 pilot shell97843/startticks801884298；首阶段容量检查97888。原pilot已选定首路线并进入持久等待，但**没有启动CARLA或模型实负载**。现场pids18199、pending_B=3（B的30claims对27实际CARLA），一worker含pending/head预算后超过16000。GPU1已清到0.04GB/0CARLA；CPU62/175，B五卡100%，P3 GPU6 18.3GB/85%。唯一阻塞是全机线程预算，不是GPU1显存/端口，也不再是原F/main未授权。

队列每30秒在Box自行重新核查，容量足够即启动首K0路线，原检查通过才进10路；之后失败候选隔离、下一合法候选接续。没有为了制造首个PID越过16000门或停止B/P3科学工作。首个真实runner PID应由`runs/nq4/gk/cx_pilot_20260927_k/k0_unseen/s0/runner.pids`及`G/events.jsonl`的server_ready核查；等待shell PID不能当作路线已开始的证据。

## 13:30CST：通过后批量补位

用户再次明确要求已通过pilot立即进入空卡批量，不等其他K层全部检查完。K0原两stage真实PASS：10/10、10attempt、0crash、blocked7/10（原70%上限内），DS6.199对同10路CL3的6.891。13:30:11原pilot结算后K1已真实runner353656。此时pids约4549，线程不再阻塞；原pilot-only脚本没有批量接续，是GPU0/6等空闲的直接原因。

发现旧head329147及其包装shell329145仍活是cleanup兼容错误：tmux登录python3缺os.pidfd_open（SSH测试解释器有）。当前K1正在复用该head，不能杀。helper补Linux pidfd syscall兼容，保持精确身份且不使用groupkill；下一阶段清理自动使用修复。

新增`cx_gk_batch.py`持久补位和`cx_gk_batch_worker.sh`，不修改运行中的GK bash/pilot队列。每个已通过K层按原seed0/1/2及官方220路安排独立卡，顺序K0/K3/K1/K2；不等整臂/所有层完成。批量除原两stage PASS外，再核K READY已有的计划有限值、模型/fold身份、blocked≤CL3+2、v_target/g3/rules原检查；未知或缺证据不放行。各批量lease（资源租用）隔离head/socket/cfg/log/清理，结果进原arms_k，seed0硬链接复用已完成10路线，避免重跑。

每张卡需真实无居民且无B runner，独立合法8-index段经SCH表及±120/全TCP端口校验；B旧注册段保持，不抢它的尾部端口。按真实pids+尚未兑现worker×600+调试两worker余量核≤16000，CPU≤165；最多6worker/卡。P3正式scene0原主门失败，无合法后续；P3 owner明确释放GPU6/CPU180–189，6c626fc registry已Box拉取并迁移唯一controller生效，旧nq4-p3 table claim清为'-'。

`79c2ebb`已Box部署；登录python3.12缺os.pidfd_open现场重测fallback清理成功，无关同组进程存活。K0追加READY检查实际PASS（blocked7 vsCL3=5，恰在+2线上）；原PASS不因此扩到其他K层。

13:40:46 CST首batch已启动：dispatcher359401，seed0 worker359432/startticks802853522、GPU0 head359463；GPU6 seed1 worker359610/startticks802855529按20秒错峰接续。GPU0 index0–7、GPU6 index22–29经过全TCP占用和SCH±120检查；P3旧claim已清。此时head处于加载、不能把它写成路线完成；后续runner由原execute在ready后启动。

## 实际路线与资源验收（13:44–13:45 CST）

三个K0正式runner均已出现：seed0/GPU0=360626，seed1/GPU6=361390，seed2/GPU5=363521，各6worker、各6个claims。seed0/1最新真实heartbeat已到405/234 ticks；seed2刚开始第一个CARLA。seed0已完成计数10来自复用，**不是新批量已经新增完成10条**。现场pids4431、CPU21/175；GPU0/6已24.1/19.7GB并各3/2个CARLA，GPU5第1个CARLA起步，原runner继续20秒错峰铺满。

控制器注册`43ab2c0`已部署并仅迁移controller/supervisor，PID361054/startticks802869930，`GK RUNNING actual_gpus=[0,1,5,6]`；P3资源声明为空。B尾路GPU2/4和GPU1 K1 pilot保持原身份，GPU3暂无新的已完成K gate可分配。新持久batch每轮读取真实K门，一通过就给空卡排该层seed，不等K0三个seed全结束。G全量的原prep/科学门没有伪造；G pilot仍原队列继续。

## 14:51CST 静默空卡故障与修复

K1原pilot和READY追加门均PASS（blocked5/ref5、moving_target_median3.414，failed=[]）；GPU2/3/4闲、pids7437，不是科学门失败。真实原因是B虽14:01:21 DONE，旧SCH行仍占300–389/60–119等大块；新K0三段与其他pilot预留叠加后，`free_block`返回None，原batch代码静默continue。现场撤销前None，核完整B DONE/结果、无B链953934/任何B output writer后仅撤B表行，立即变合法index60；未杀953938孤立status watcher。

原dispatcher下一轮14:53即起K1 seed0/GPU2 worker463944；14:55起seed1/GPU3 worker468093。为防重复故障新增`cx_gk_batch_v2.py`：自动回收真实完成且无writer的B旧预留；每卡具体WAIT写events及admission.json；使用原注册400线程/worker、真实claims衡量未兑现workers，不把尾部空闲worker重复当未来负载；同一16000/165/88GB门下动态选1–6worker。GPU1只预留其尚未兑现的两pilot槽，已居民不再双算。supervisor重启仅调度器并认领原worker身份，不停止实验。

新增4项回归通过：7437真实快照可给三空卡各6worker、预算不足动态降worker但不越16000、尾部worker不虚占、B有活writer绝不回收、无端口WAIT可持久且去重。旧live dispatcher/bash均不修改；新版本迁移只精确旧dispatcher PID，保留K0/K1正在跑的worker。
