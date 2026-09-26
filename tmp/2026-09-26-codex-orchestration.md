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

14:18:42 UTC 两个输入审计均成功 DONE：WOD读入19,663帧、478 rater、1,458 pre_onset、11,597 straight_yaw；Q4b和Q6实际消费者的array/token覆盖与provenance核验通过。14:18:55 UTC **正式科学命令已经同时启动**：Q6 `rt-wod` worker PID546743、命令PID546751，CPU180–191；Q4b `hydra` worker PID546747、命令PID546752，GPU5、CPU192–195,197–199、预留20GB。Q6 refit与Q5按已占用CPU槽等待。之后由持久脚本接棒，不进行模型持续轮询。

## 用户要求吞吐修正（第二轮）

22:29左右状态：Hydra六次refit与NAV ridge全部DONE（Hydra navtest选择一致率1.0000），Q6 WOD/NAV后处理DONE，rt-score在跑且日志有2/6完成。没有科学FAIL/WAIT；ready的compat/refit被 `pending-inclusive threads 20484` 阻挡，而实时pids16316、CPU约91.6/175。初版把已达到自身启动背压的CARLA所有未来峰值，以及已经进入resident的评分线程重复预留；Q5又要求19核整块空闲。这两个限制由编排实现引入，确实延迟可执行工作。

新文件 `cx_orchestrator_v2.py` / `cx_orchestration_v2.json`，commit `a432315`，旧dispatcher/manifest不改。新队列在142–149、180–195、197–203范围按实时窄亲和及已有claim挑核（明确排除X134–141与watcher196）。Q5原child取10核、devkit每次6 worker；请求、arm、seed、阈值、预测和输出命名不改。Q6三个seed只有refit-check是真依赖，不再人为串行，report仍等全部三个seed。

pids按resident加尚未实现的任务claim预算；CARLA既有runner的17000背压以下可实现容量加一个400线程race，另留X600和hard limit1024的余量，不修改CARLA原16000计划门或任何runner。GPU resident内存不再与整个active预算双算，仍保留未加载部分、SCH pending显存与每卡8GB。GPU1永不分配。本机/box七项回归检查通过，包括实际16316pids的ready任务可入、hard余量拒绝、Q5与评分/fit共存、GPU不双算、动态核隔离、脱离PGID的Ray线程计数、科学命令与refit gate保持。

修正前7次采样测量（实际77.49秒，nvidia-smi查询阻塞使时长超过预定30秒）保存在远端 `before-throughput-v2.json`：CPU91.879/175（52.5%）；GPU0–6平均利用率82.7/6.7/90.7/58.9/55.3/51.0/86.3%，六张batch卡均值70.8%忙。这个时间窗不支持“70%GPU空闲”的概括，但明确证实CPU仍有余量，而三个ready任务被本调度器人为挡住。之前用户看到的瞬时图不用于回推此窗口。

14:37:55 UTC新窗口 `jev:cx-orchestration-v3` 接管。旧supervisor PID542401按start_ticks+argv核对后只发PID TERM；其代码没有supervisor TERM清理trap；全部worker有独立PGID。迁移前后的score worker556660/start_ticks797342398完全相同且仍活，结果锁没有释放/重建。`migration-before.json`与`successor.json`保存过程证据。

同一个准入周期实际启动：compat命令PID612023（CPU142–143），Q6 refit命令PID612024（CPU144–147、GPU3），Q5原child命令PID612016（CPU148–149,192–195,197–200、GPU3）。Q6已读46703行并拟合，Q5已开始frag-prep。compat已出双方seed0不可比的原判格，exam按原逻辑跳过；这不是基础设施失败。GPU3的两个任务当时都满足显存未来预留，每卡8GB余量保持，GPU1不使用。后续seed按可用显存/核继续排。

后测是7次采样、实际89.54秒：CPU79.554/175，GPU0–6均值78.0/2.4/76.3/45.3/61.0/20.1/85.9%（batch均值61.1%）；pids13858–14344，对比前测15360–16446。前后工作集变化，评分完成释放资源，不能声称“占用率提高”或凭此推导加速百分比。可靠的推进证据是DONE步从5增至9，原来卡住的compat/exam/refit完成；refit109秒，随后的等价检查预测最大差与帧差都是0。Q5继续图片准备，后测结束后三个single-frame seed worker622911/622912/622913均已实际并行启动。

最终补上重启检查：比较RUNNING任务自己的不可变launch.json中的实际命令，避免下次重启把Q5的resource-only SCORE_THREADS=6误判为相对旧模板14的命令变化。改文件前再次只停dispatcher610518，记录并核对Q5worker612015及三个seed worker仍活；不改/重启科学worker。第8项回归覆盖该重启情形和真正命令变化必须拒绝。测量输出新增真实elapsed_s，避免把30秒sleep意图当实测窗口长度。
