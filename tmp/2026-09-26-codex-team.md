# Codex 专用协作与维护台账

本文件由root维护，是任务与智能体的入口。每次状态改变更新；完成记录保留。上下文压缩后先重读 [master](2026-09-26-handoff-master.md)、[逐lane交接](2026-09-26-codex-handoff.md)、[状态与异常](2026-09-26-codex-status.md)，再读本文件。远端实际资源以 `$DATA_DIR/runs/sched/table.tsv`、GO和实时进程为准，台账不代替授权。

## 当前团队

| 智能体 | 负责 | 文件/写权限 | 状态与下一步 |
|:--|:--|:--|:--|
| root | 仅指挥、跨agent协调、资源/范围检查与验收 | 本台账、codex-status汇总 | 按用户要求正常每1–2h整体检查；异常/待人事项及时汇报 |
| lane_watch | 原A/B/C/SCH看护 | 已交接Sol | 已暂停，保留历史报告 |
| sol_watch（gpt-6-sol / medium） | A/B/C/D/SCH看护与T1明确机械FLAG处置 | sol-watch.md、codex-status append；T1 SKIP/APPROVED | 已接手；常规5–10min轮询，异常/完成即时通知root；不重复共享pilot实例 |
| sol_ops（gpt-6-sol / medium） | T10 SKIP/OPL收表，T6b P3安装，T7收尾与T9合并就绪检查 | sol-ops.md、codex-status append、授权输出；Git先协调 | T10优先；不得启动OPL/P3链；Alpamayo两run均退出才合并 |
| astra_x_audit（gpt-6-astra / high） | T8 X复审；用户新授权资源优化/隔离并行scheduler | astra-x-audit.md、授权scheduler代码（先方案与Git协调） | 先完成X证据报告；审计现有已验证任务并行/CPU-pids瓶颈，给reviewable扩容/分片方案，不改科学判据或live链 |
| p1_resume | P1已完成；T8 X stop条件审计与最小修复 | scripts/nq4_x_agent.py（只经批准最小条件）；Git先协调 | X stop规则未实现；低速路径必须保持，目标速度路线巡航；未授GPU验证资源，不启动 |
| o | O已完成；T3 W机械表与固定判格 | todo W结果；codex-status append | W判据1整体不过（仅obstacle过），判据2不过；CI缺项如实标未提供；decisions W对应旧条目不明确，待人 |

## 授权任务队列（master第4节为原文）

| ID | 触发/依赖 | 执行/下一步 | 状态 |
|:--|:--|:--|:--|
| T1 | ESCALATE新的FLAG及完整verdict | 原生cl2/7因blocked/不动→SKIP三seed；其余仅高DS或blocked→APPROVED；其他等人 | 当前未见FLAG |
| T2 | pilot PASS、端口/pids/显存满足、批量容量可用 | sch_table grant；完成revoke；先PASS先授，pilot让卡 | B CL3已自动过门开跑；F/OPL待核 |
| T3 | 各节链DONE、原登记判据与小表完整 | 数字/CI/固定判格写todo与明确对应decisions旧条目，标待复核 | W todo完成88ccd51并push/boxpull；decisions映射待人；P1已有表；其他等DONE |
| T4 | C全部判卷出数 | 已存预测按x01/x00同天气95%分位替代tau，规则7其余不变；描述性不进判格 | 未触发 |
| T5 | F/K/OPL/P3最后交接节齐 | 可启动就按指定命令；未建设完成不得自行补设计 | F旧禁止启动由T8覆盖；K/OPL/P3待最终节 |
| T6 | P3 null与ridge_late预测/对比图齐 | 算误翻率≤7%；保存图、残影待人；不扩60场景 | 未触发 |
| T7 | lane完成/规定收表周期 | scp核验、小表commit/push、关已退出窗口；不动其他人的文件 | W/P1/O已交付；CL1最新219路线四表25f5067已push/boxpull |
| T8 | X最小速度门+1路验证、合法index、pilot/GO配置 | prep DONE后启动nq4_gk.sh，内部1→10→full门不绕过，worldreuse仅PASS后开 | agent与P7目标速度接口不兼容，记录等人；没有修改/启动 |
| T9 | Alpamayo主runner和helper37228均退出 | consolidate --owner-pid 主pythonPID；记帧数前后；若finaljudge先跑过则合并后按单步命令重跑 | 未触发：主python PID684982（--deadline23:30），helper37228（最晚08:00）；sol_ops保存依赖 |
| T10 | main否定OPL partner占比89%>50% | 禁止nq4_opl.sh；cl2/cl7/cl5d全部seed立即SKIP；仅收表等smoke完成 | sol_ops执行中，共用SKIP.lock协调 |
| T11 | D到q4b/q6前 | 核provenance md5；不符仅删指定缓存；相符复用 | sol_ops跟踪，watch通知D转步 |

## 当前实验与资源（最近核查约19:24 CST）

| lane | 当前/完成 | GPU/CPU与资源归属 | 异常/门 |
|:--|:--|:--|:--|
| A | v0rr 461/522（19:23） | 3–5，核0–59，旧index600–689 | 无ERROR，预计即将转v1；未DONE不动结果副本 |
| B | CL1 219/220 DONE；CL3 PASS后19:23全量 | 0/2每卡9worker，核60–109，index300/360；扩卡GO未来段另核 | gate脚本已部署；不把旧3条smoke当自动通过 |
| C | q1_op完成→q1_heads | GPU6，核150–179；SCH helpers另表 | READY已产出；全部判卷完成后才T4 |
| D | q4a-fit失败停链 | AssertionError：lam_r=0不复现M-C | 交接要求等人，禁止重试/放宽；前移Q4b GPU5作业独立继续 |
| W | 三seed+report DONE | 额外seed2已释放GPU5；窗口已关闭 | 六表22f4b3c已推送，T3待填 |
| P1/O | 完成 | CPU计数结束，窗口关闭 | P1 33,490事件；O fallback1,320行 |
| F G/K/X | smoke/post在跑，完整链未启动 | GPU1旧pilot150–159；后续授权T8另核 | X需最小修复与单路线验证；worldreuse未PASS不得开启 |
| K/OPL/P3 | 原Opus执行员收尾 | 以最新SCH与最终交接为准 | 尚未接齐，不重复启动 |

## 执行纪律与恢复检查

- 不修改运行中的链脚本；Mac代码commit/push、box只pull/执行。Git串行：先消息协调再定向stage；不stash/reset/clean他人修改。
- GPU1保留基础设施/验证；pilot可借等gate时空闲batch卡，每卡3–4pilot/debug，受pids/端口限制；validatedbatch优先让卡。当前全卡分配不等于GPU/CPU满载，按实时吞吐与SCH容量补作业。
- 批量必须1→约10→清单PASS后开；任何新段i及i±120无冲突，index≤494，check=0且实际端口空。只按记录PID停止。
- 不做P2、新判据、新todo设计、专家问题；规则没覆盖的情况写status等人。master T1–T11固定规则高于旧笼统“不填判格”说明。
- 每轮：重读新增交接→查ESCALATE/verdict→查DONE/ERROR→SCH show/check→资源让卡/GO→收表→更新台账与status。

## 待人事项

| 项 | 证据/位置 | 等待原因 |
|:--|:--|:--|
| D assertion | codex-status19:20原文；runs/nq3/d/ERROR | 等价检查失败不在重试范围 |
| W decisions对应条目 | W无直接旧条目，48只是特征来源 | T3未授权凭猜测选旧条目/建新条目；todo可先完成 |
| T8实际端口快照不同于文字 | 19:24 table中OPL候选IDX0=120且无GO，B扩卡3:60；check=0 | 核最新SCH版本后再处理，不能按过期值盲改 |

## 19:26 CST 接手更新

- T9已读加入队列，主/补runner仍运行时绝不consolidate。
- SCH最终说明：并行3stream尚未实现，现脚本多个实例会共享B_DIR互杀server。须停止后隔离stream目录/claim再改；master T5未授权自行建设这条改造，记录等人，继续已有串行pilot。
- P3最终交接明确仍需Opus完成未验证的GPU/render管线，Codex仅机械看护CPUprep；不能直接训练启动。
- X实现缺口：P7 Controller.update没有独立目标速度参数，轨迹静止会安全刹停。main要求路径不变、目标巡航，不能用改路径或改P7代替。X最小条件与单路起步验证暂停，待明确实施范围。

## 用户指定的模型与管理方式

常规机械任务用Sol medium，困难代码审查用Astra；root仅指挥和检查，正常每1–2h整体验收，异常及时通知。当前4并发槽=root+astra_x_audit+sol_watch+sol_ops，其他历史智能体不继续重复分派。同一SKIP/GO/Git写入必须先协调避免共享文件lost update。T10覆盖此前OPL可授batch/原cl5d可运行的旧说明。

## 用户新增资源优化授权

用户明确要求：留1卡（GPU1）给其他智能体验证测试；已可完整执行且过pilot的任务，尽量并行跑完再接下一大任务；综合CPU、pids、GPU显存/计算和端口瓶颈。Astra先做实测审计和最小reviewable方案，Sol执行机械步骤，root只协调验收。此授权允许补齐SCH并行隔离实现，但只在对应旧脚本停止后改，不能修改运行链/科学条件。正常每1–2h整体检查，失败/冲突立即记录。

P3最新ops核查：prep22382已exit1，缺指定Waymo tfrecord，未prep.done；此分支等人。安装独立可按T6b重跑已补nvdiffrast的新脚本，失败按交接停，不自行修设计。

## 最新执行方式：一次性脚本，节省 Tokens

用户要求三天长任务以持久一次性脚本运行，执行而非模型诊断；root仅每隔几小时调度/验收。Sol ops优先把固定规则轮询/状态/收表/就绪门写为无人值守脚本并启动。Astra和Sol watch保存现有资料后结束模型轮询，实验本身不停。未知/规则外异常写WAIT；不给模型自动拉起诊断，不重复请求批准。诊断包按既定期限只收既有原始证据与覆盖标记。T2不能完全机械确定的授卡留READY_FOR_SCHEDULING供root低频处理。代码/运行PID/log/events/退出状态须由ops交付后记录到本台账。

## 无人值守部署完成

代码 `71eee9f` 已main push/boxpull，新增脚本无live链改动；6项安全检查Mac/box通过。远端 `jev:cx-maintenance` PID158078（每10min，执行固定机械规则），输出 `$DATA_DIR/runs/nq4/cx/maintenance/{log.txt,events.jsonl,STATUS.md,state.json,export.json,exports/}`。本机 `jev:cx-maint-collect` PID87740，每10min只读同步固定小表与证据，不调用模型；首轮10文件SHA核验，列READY_TO_COMMIT供root低频处理。

T1/T10锁写、T9双PID退出与finaljudge同step锁、T11producer DONE与D/producer记录进程均停才失效生效。T2仅READY_FOR_SCHEDULING，不自动猜分配。D/X/P3未知分支WAIT，其他任务继续。诊断原始packet/provenance/README自动保存；确定性工作目标27日23:59 JST。模型watch已结束，root每几小时检查调度/待提交表，不连续采样诊断。

## 用户新授权：Astra 解决 X，最多自主迭代 10 轮

`astra_x_fix`（gpt-6-astra / high）已分派，负责X起步问题的技术实现、独立单路线验证和offline/rule8检查；成功后才进入约10路线sanity，不绕过full gate。这次用户明确授权自主修复，覆盖先前仅审查/最小一条件导致的实施暂停；仍保持科学判据与目标行为，不修改其他考生、P7基线或运行链。GPU1先核实时空余与既有3–4pilot/debug上限；独立socket/PID/目录，合法index与pids检查，禁止复用F共享model server。记录round01–10的假设/变更/命令/结果；达到良好结果提前保留，否则完成10轮后证据化停止。root只协调/检查，维护脚本继续，不重启其他模型轮询。

## 21:02 CST 一次性容量核查（Sol）

| 工作 | 最新步骤/资源 | 独立 CPU/GPU 队列状态 |
|:--|:--|:--|
| A | v1 238/1320；GPU3–5，生成核0–47；offline核48–59 | BLUE已515/515，SimLingo330/515，三worker56977–56979各约119%CPU继续；不重复启动 |
| B | CL3 s0，GPU0/2各9worker，核60–109 | 后续臂受pilot/原链门控制，不能仅因CPU空闲复制 |
| C | current_step=q1_judge；GPU6，核150–179 | Alp主684982约111%CPU，亲和158–163；helper37228约121%，亲和118–133，原队列运行；T9尚未就绪 |
| D/forward | assertion停；q4b/Q6 DONE且md5一致 | C Qwen/WAJEPA、D navtrain及Q4b/Q6前移已完成，无剩余已授权独立GPU命令 |
| K/F | K STATUS step=done、READY存在，训练/开环导出完成 | 剩闭环CARLA；F全链仍需X原gate，不重复训练 |
| P3 | install已DONE；prep缺tfrecord、无prep.done | 没有可启动训练/render/CPUprep任务；缺项待人 |
| X | round01资源gate通过后ss缺失，infra_error，272666退出 | Astra修/proc/net端口核验并准备GPU1正式Q2导出；保留核204–207与GPU1，不另占 |
| 维护 | box158078、Mac87740均活，每600s无模型维护 | T7已有16固定小表可收；T9仍WAIT |

SCH snapshot pids15300/20480、计划cap16000；cgroup实际95/175核使用（cap165），负载约154–166，不把亲和已分配核等同实时占用。GPU0/1/2/3/4/5/6显存分别64.2/10.5/61.0/82.2/58.9/51.9/57.1GB、利用率92/18/70/63/81/98/92%；GPU1仅1CARLA。check=0。pids约剩700到计划门仅限制新CARLA预算，不能解释所有GPU任务；GPU-only已完成/已有owner/未来依赖未齐，现可加的独立GPU验证是Astra X正式导出。没有发现完整就绪、未运行、已授权的独立CPU固定命令，未为了填核启动无效作业。仅此机械快照，不持续模型轮询。

## 全任务盘点补正（约21:15 CST）

详见[全任务表](2026-09-26-codex-task-inventory.md)，143个子任务/臂×variant组行，覆盖Q1–Q6/CL/G/K/W/X/O/P1–P3/E与T1–T11。D的Q4b/Q5/Q6只受原串行顺序阻挡，不依赖Q4a；已有GPU缓存不代表整个科学节DONE。Q5 select独立CPU180–183/4threads执行成功22行，输出runs/nq4/cx/q5-select，未写D链DONE。C20:23 q1_judge实际ERROR，sol_c_repair接手；K-prep含规则8实际全DONE/pass，K批量未完成。

## 21:48 CST / 13:48 UTC 一次性效果核查

A v1实时388/1320，对比21:02的238增加150；21:42原STATUS吞吐170world/h（此前157），这是自然推进观测，不归因未执行的新调度。B CL3 seed0实时195/220，对比20:58的100增加95，CURRENT仍cl3 0，未DONE。CL4 pilot FAIL（crash6/15=.40，blocked5/9=.556）；CL2 FLAG（10/10完，moves0、blocked1、DS0），不撤原T10 SKIP。

C旧allow_pickle修复后q1_judge21:18:17 DONE（4.6min），随后q2_report因EmptyDataError停，report DONE尚无；由sol_c_repair唯一owner继续。D原lam_r=0 assertion保留。X round03 PID448786活，stage1单路253490已产生ticks.jsonl124397bytes（round02为0tick）；尚无summary/route DONE=0，没有模型PASS或科学结果。formalQ2导出已DONE复用。X使用CPU134–141，不再204–207。

资源瞬时：CPU.stat 1s增量75.5核，pids10765/20480（plan16000）；GPU0–6 used MiB/util分别22553/20%、34891/33%、36078/57%、68748/64%、50492/56%、51471/38%、58462/92%。不是CPU已分配核数。Q5 select DONE21:09:23（记录PID351840，22行小表）；Q4b/fullQ5/Q6 mixed-plan尚未实际launch，无对应进程，不能报这些优化已提速。Q1新小表本次检查目录未查得，不声称判格或空结果；q1_judge DONE可核。仅此快照，无新任务启动/持续模型轮询。

## 独立组合队列已部署（Astra orchestration）

[编排记录](2026-09-26-codex-orchestration.md)：`3cbe48d` main已push/boxpull，`jev:cx-orchestration-v2` PID542401。17步固定Q4b/Q6/Q5组合持久调度；Q6与Q4b输入验证PID543399/543400已实际启动。GPU1留X，CPU196留现有watcher；不改D assertion/ERROR/DONE、不改live链、不启动OPL/P3/P2。CPU/显存/pids按实时+pending预留，失败仅阻断其依赖。root低频验收，无模型持续轮询。

14:18:55 UTC编排首次正式双任务启动：Q6 rt-wod PID546751（CPU180–191），Q4b Hydra PID546752（GPU5/CPU192–195,197–199）。inputs均DONE；dispatcher542401独立继续。状态/证据见上述编排链接。

## 2026-09-26 Sol B queue successor (one-shot)
B chain953934 healthy CL3s1 runners508600/509884; no ERROR/stale T1 gate. CL2/CL7/CL5d T10SKIP retained, CL4FAILSKIP retained. Current shared pilot953785 CL6; no other fullyPASS batch waiting. CL5READY missing from pilot ARMS: root authorized durable successor600986 (jev:cx-cl5-successor, runs/nq4/cx/cl5-successor/once.py) verified alive, waits953785/replacement pilot exit then original sch_cl_pilots.sh cl5, guarded pids+1200<=16000/CPU+7.5<=165/GPU1free>=30GB/debugCARLA+3<=4/ports390–419 free/SCHcheck0. No immediate load gain; no GO/SCHtable/livechain/Git changes. GKprep absent; KREADY not K10pilotPASS, XnotoverallPASS; OPL/P3 blocked per master. [Details](2026-09-26-sol-queue-unblock.md). Model ends; persistent successor continues.

编排二次吞吐修正 `a432315`：旧未来线程峰值/active双算与19核整块门已修。新v3窗口接管，score556660不重启；14:37:55UTC compat612023、Q6refit612024、Q5child612016实际并行启动。动态池142–149,180–195,197–203，避开X134–141/watcher196，GPU1保留。见[编排记录](2026-09-26-codex-orchestration.md)。

编排最终supervisor更新：`1fb1352`、PID631108、jev:cx-orchestration-v4；8项两端回归通过。Q4b完成，Q6 real verdict完成，三个single-frame fit同起22:41:47，分别135/138/197秒成功结束。Q5worker612015原身份持续运行；随后report/table由持久队列接续。非模型连续轮询。

## 2026-09-27 早间恢复与统一控制器（此节替代上面的旧状态）

- A：`7b36bf8` 修复真实相机末帧到 tick 的兼容读取；CPU post 与 judge 均返回 0，v1/DONE、judge/DONE、a/DONE 已验证写出，无运行资源。
- B：GO 已授 GPU0/2/3/4/5，每卡6 worker，原 index/extras 保留，check=0。07:57 CST 仍在 BridgeDrive 原臂，五卡实际启动确认尚缺，不能将授卡当满载。
- C：07:57 CST 已进入 v1_prep，GPU6恢复工作。Alpamayo T9 已合并18142帧，仍缺640，final由原链顺序处理。
- 独立 Q4b/Q5/Q6 组合：17/17 DONE，旧 dispatcher 已退出，不重启已完成任务。
- unified_controller（Astra high）：唯一新控制器实现与部署 owner；box-local 30秒状态/资源闭环，不能越过科学门。
- controller_review（Sol medium）：独立故障复现与审核，不改 owner 实现，不操作实验进程。
- controller_inventory（Sol medium）：已完成只读真实身份/候选清单，见 `2026-09-27-controller-inventory.json`。
- G/K/X、P3、D原科学链仍有明确人工门；OPL禁跑。控制器必须将缺门与资源不足区分，不能制造 PASS/DONE。

root负责验收、协调和范围内Git检查。

统一控制器 `aa97fee` 已main push、box ff-only pull并启动 `jev:cx-controller`；首轮ONLINE，PID760047/start_ticks800835949。13项故障测试及独立审查通过。A/A-v1真实完成且释放资源，17/17组合产物完成，C已认领GPU6运行。B仍原BridgeDrive边界，控制器明确等待真实五卡runner+GPU确认，不能把GO当满载。告警为持久inbox，不会自动唤醒模型。部署完整证据见 `2026-09-27-unified-controller.md`。

## 2026-09-27 B 尾部补位核查

见[work-conserving记录](2026-09-27-work-conserving.md)。BLUE09:40:21 CST自然220/220完成；SimLingo最后一臂已五卡各6worker，PID935741/936497/937705/939009/940835。之前整臂屏障造成尾部空闲属实；本次未实施跨臂接管。最后空QUEUE触发controller解析异常，新增v2只修空白记录并迁移唯一控制器，保持B/P3进程。B后继QUEUE空，17组合已完成，未知科学门继续WAIT，不新增无任务调度器。

B控制器修复 `d55d9ed` 已Box部署；唯一v2 PID954435/startticks801438503，supervisor954433，jev:cx-controller-v2。B原链/五runner未变。09:45CST实际五batch GPU97–100%（GPU5 96%），pids14965、CPU48核；NO_READY_WORK准确反映后继队列空，P3 GPU1/6保留。详见work-conserving部署验收；这是自然换臂负载，不宣称跨臂优化已实施。

## 用户授权 G/K pilot（2026-09-27 10:51CST 后）

Astra orchestration负责GPU1 K0–K3原1→10pilot再合法G候选，见[gk-pilot-launch](2026-09-27-gk-pilot-launch.md)。P3保留GPU6/CPU180–189；旧Alp已核无调用者并精确停止。全机pids仍超过16000，持久队列必须等容量释放；不写全量GO/prepDONE，不降科学门。原GK无活链，先修精确PID清理再部署。

G/K `8bbcda1`已部署`jev:cx-gk-pilots`，owner97841、K0 pilot97843/start801884298，容量wait97888。10:59CST pids18199、B pending3，尚无GK真实CARLA负载；GPU1已清空但需保持16000计划门。Box持久队列30秒自行重试，不需模型轮询。P3GPU6原owner不变。
