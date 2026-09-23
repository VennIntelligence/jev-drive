# 真实 TCP 模型接入与控制器对照：只读审计和最小方案

状态：2026-09-23，只读检查完成；未运行模型、CARLA 或 GPU，未修改运行时代码。当前正在跑的 oracle-controller 正式实验与本文真实 TCP 实验分开。本文仅交付下一步方案；真实模型新增执行范围尚未确认，不自动启动9例或实施完整pursuit。用户关心实际驾驶行为，不把 driving score 作为唯一目标。

## 已经运行过的真实 TCP 是什么

- 官方本地入口：[tcp_b2d_agent.py](/data/third_party/Bench2DriveZoo/team_code/tcp_b2d_agent.py)，网络：[TCP/model.py](/data/third_party/Bench2DriveZoo/TCP/model.py)，配置：[TCP/config.py](/data/third_party/Bench2DriveZoo/TCP/config.py)。已有可视化包装：[b2d_tcp_visual_agent.py](/data/worktrees/jev-drive-controller-v2/scripts/b2d_tcp_visual_agent.py) 继承官方 `TCPAgent`，并调用 `super().run_step`，不是当前 `StubAgent` 的 oracle 路线控制器。
- 权重 `/data/models/bench2drive/tcp/tcp_b2d.ckpt` 实际存在，319,450,037 bytes，SHA256 `e6573ff1f8ea910b9a53eddfb68f69cac469bf5bfa253a516578f6126110b4fe`。本次仅读取文件并计算哈希，没有反序列化或推理。
- 官方 setup 读取 checkpoint 的 `state_dict`，去掉 `model.` 前缀后 `load_state_dict(..., strict=False)`。后续接入需记录 missing/unexpected keys；权重文件存在和哈希一致本身不能证明所有网络参数都加载匹配。
- [tokyo-box.md](/home/ujs/mycode/jev-drive/docs/tokyo-box.md:238) 记录官方仓库版本 `8a08b07883f10b7d83f6bf5dd475bda91a91c50a`、环境 `/data/envs/b2d-tcp`、`PLANNER_TYPE=only_traj`。本次没有调用 git；以当前实际源码哈希为准。历史 attempt JSON 证明使用该真实 agent 和 checkpoint，但没有完整保存 PLANNER_TYPE 环境，所以 `only_traj` 的历史来源是当时操作记录，而非每个 attempt 的独立环境快照。

| 历史执行 | route | ticks | attempt wall | 官方结果 |
|---|---:|---:|---:|---|
| tcp-fast | 24211 | 1889 | 169.1 s | Completed，completion/DS 100/100 |
| tcp-fast | 1711 | 1847 | 275.4 s | Completed，100/100 |
| tcp-fast | 1773 | 661 | 121.9 s | 用户中止，completion 28.82；未完成不能计通过 |
| tcp-smoke | 1711 | 1852 | 273.1 s | Completed，100/100 |
| tcp-smoke | 24211 attempt 3 | 1891 | 226.3 s | Completed，100/100；此前两次传感器校验失败保留 |
| tcp-smoke | 1773 | 272 | 74.9 s | 用户中止，completion 13.68 |

来源为各运行 `attempts/<route>/<attempt>/{attempt,results,route_result}.json`。这些是真实模型运行，却**不是**真实 TCP + 新控制器的配对实验；预处理优化还曾在 tcp-fast 运行中途更新，不能把历史 wall 差异归因于控制器。

## 模型、时间和坐标契约

1. **原生未来点是 4×2、时间 +0.5/+1.0/+1.5/+2.0 s。** `config.pred_len=4`；[gen_tcp_data.py](/data/third_party/Bench2DriveZoo/tools/gen_tcp_data.py:11) 明写 `4*5 # 10hz --> 2hz`，并采样 `i+5:i+FUTURE_FRAMES+5:5`。原生 PID 也用相邻点距离乘 2，取三段均值；它不把 origin→first point 纳入期望速度。
2. **每个仿真 tick 推理。** `seq_len=1` 只让第一 tick 返回中性控制，之后官方 `run_step` 每次都调用网络；历史配置 decimate=1。保持 0.05 s 仿真步长和原生 20 Hz 推理/控制，不把它改成 oracle-controller 的 5 Hz 规划。墙钟推理慢只会拖慢同步仿真，不意味着推理采样变成 5 Hz。
3. **真实模型输入不是 oracle demo 的三路 800×450。** 原生三相机 1600×900，经官方 JPEG quality20、裁剪拼接、bilinear resize 到 256×900、归一化；速度输入除以12；导航 target point 与 one-hot command 进入网络。现有 debug BEV/chase 不输入网络。为两组固定同一个已有像素等价预处理模式，不混用历史中途变更的版本。
4. **网络 raw waypoint 是 forward/right；本项目控制器是 forward/left。** 数据标注旋转 `R=[[cosθ,sinθ],[-sinθ,cosθ]]`，θ=compass−π/2；原生 `control_pid` 再交换两个轴，得到 `[right,forward]`，并用 `π/2−atan2(forward,right)` 产生右正 CARLA steer。对未交换的 raw prediction，方向变换应为 `[raw_x,-raw_y]`。PID metadata 已交换轴，不能直接当原始预测使用。
5. **物理原点需单独确认。** 在线导航 target point 明确以 GNSS 对应位置为原点，GNSS 安装 x=−1.4m；可视化 `prediction_origin=[-1.4,0]` 也是这一解释。但训练标签只显示从 annotation `x/y` 减去当前 `x/y`，本地生成脚本未证明 annotation 的物理原点就是 GNSS，而不是 actor/其他参考点。因此不能把 preview 假定当测量证明。若之后确认 GNSS 原点，与实测后轴 −1.388633m 的静态差是 −0.011367m；若标签跟踪未来 GNSS 点，未来杆臂还随未来航向旋转，不能不加说明地把所有点当未来后轴位置。
6. 当前 Controller 严格要求 20×2、0.25s、5s。**不能通过复制尾点、外推或拉长时间把 TCP 的2s变成5s。** 这会伪造停车尾段/速度/可用路径，并改变规划语义。全横纵向接入必须显式支持原生4点/0.5s有限时域，保留真实源帧与时戳；不存在2s以后的规划，就不能声称存在。

## 原生控制绝不只是当前 `preset=tcp`

原生 `control_pid` 在4点附近搜索 aim_dist=4m，并用导航 target point 的角度/突变条件进行目标选择；条件是更小的绝对目标角，或与末段角差>0.3且目标前向距离<10m。它然后用 turn PID .75/.75/.3、40窗；速度目标为三段的平均速度，speed PID5/.5/1、40窗，误差裁到[0,.25]，desired<.4或speed/desired>1.1触发制动。当前 oracle `preset=tcp` 仅复用了标量 PID 语义和固定4m前视，未复刻这些目标选择逻辑。

`PLANNER_TYPE` 三条路径：

| 分支 | 官方最终混合前的控制 |
|---|---|
| only_traj | 原生 trajectory PID |
| only_ctrl | 网络 action_index argmax 对应离散动作；不是 waypoint controller |
| merge_ctrl_traj | steer/throttle 各0.5×traj+0.5×direct；brake=max(two branches) |

**三条路径后都还有实际决定行为的尾部处理：**

- |steer|>0.07 时速度阈值1.0m/s，否则1.5m/s；超过阈值，throttle最多0.05，未超过最多0.5。
- 任意正 brake 变为1.0，然后有 brake 就清零 throttle。

这是当时文档已记录的官方低速限制。若新 PI 放在这层前面，连续小制动仍变全制动，油门也被压到0.05，不能期待它保留 oracle G2 中的平顺行为；若仅候选绕开尾部，就同时改变执行策略，不能称为纯 PI 因果对照。Hybrid 默认与 only_traj 也不能混称同一个基线。

## 最小正确接入：先隔离纵向，保持原生规划和横向

建议第一步做真实 TCP **纵向控制器对照**，避免先把4点规划硬接成5秒。只消费原生 `control_pid` 的同一期望速度与同一速度计输入，把其 speed PID 替换为已经冻结的 SI PI(Kp=.5,Ki=.25)，**保持原生 steering、target-point arbitration 和网络输出完全不变**。这样不依赖尚未证明的预测物理原点，也不需要重新定义未来时域。

最小代码工作应是一个明确命名的真实 TCP 包装器，而非给 StubAgent 增加 `--drive controller` 后误以为调用了模型：

- 每个 tick 只调用一次原生 `run_step` / 网络 forward。通过受控 `control_pid` hook 保存 native waypoint/desired speed/steer/throttle/brake，再算 PI；不要运行两遍 planner、两遍 PID 或两次网络。
- 在原生 `control_pid` 交换坐标前 `.detach().cpu().numpy().copy()` 保存 raw prediction；metadata 单独保存。CPU numpy 视图可能被原生交换操作改写，不能让日志或候选读到被覆盖的输入。
- 显式记录并冻结 `PLANNER_TYPE=only_traj`。原生网络仍可计算 direct head，但不让其混入 selected control；将来测 hybrid 时需保留0.5/0.5和max brake，并单独命名实验。
- 对相同 tick 更新 native PID 和候选 PI 各一次，用被选中的分支控制车辆，另一分支仅作 shadow 日志。PI按实际时间积分，路由/异常重置；未完成首帧初始化维持与基线相同的中性输出。
- 官方 TCP 对 NaN compass 置0会影响导航输入和网络 target point，当前 PoseFilter 的恢复补丁不会自动作用到真实 TCP 包装器。先在所有真实 TCP 组保持同一官方输入规则并记录raw异常；若需要改成预测，必须双方共同修改并明确这是另一项输入处理变更，不能只改候选。

### 公平分组：推荐三个明确的实验臂

| 实验臂 | 轨迹/导航/横向 | 纵向 | 尾部执行规则 | 可以回答 |
|---|---|---|---|---|
| A official_anchor | 原生、完全不变 | 原生PID | 原生低速限油门+binary brake | 是否保持真实官方基线行为 |
| B native_common_envelope | 同A | 原生PID | 公共throttle[0,.75]/brake[0,1]、互斥；不做速度专用限油门、不把小制动变1 | A→B是执行规则变化，不是PI收益 |
| C pi_common_envelope | 同A | PI .5/.25，用原生desired speed | 与B完全相同 | B→C才是纵向控制器差异 |

B/C保持原生 lateral clipping，暂不叠加新steer gain、rate limit或max前视。阶段预算若只允许两组，可以先B/C并明确它们不是未经修改的官方最终策略；已有A历史结果仅作背景，不能充当匹配的第三臂。也可以先A与“PI但保留官方尾部”做最窄6例测试，但只能回答**在原生尾部约束下**的表现，可能完全遮蔽PI作用。

全横纵向 pursuit/max 是第二步：确认数据原点；让控制器显式接收4×2/0.5s而默认20×2不变；2s之外禁止补点；只做坐标与时间适配，不用 dense route 替换/修复模型预测，也不使用 v2 oracle rejoin。选择是否保留native target arbitration本身属于控制器定义，必须在运行前写清。新增短时域 API、停车/时域耗尽和坐标测试通过后，才能把同一真实 TCP 规划器接上 pursuit/max。不能从阶段一纵向结果声称已验证完整 pursuit。

## 必要验证与固定三路线协议

实现前/离线测试：

- 同一输入tensor只forward一次；raw prediction复制前后不变，baseline native输出逐值等于官方；seq_len首帧、20Hz调用、route/reset均一致。单独证明common-envelope B/C的后处理相同。
- 原生desired speed用三个相邻0.5s段精确复算，排除首点origin桥；不拿controller自己target与自己speed定义互证。恒速、减速、停止、零点/非有限点、PI饱和恢复/互斥、暂停和新route清零。
- 若做full pursuit：raw `[forward,right]` 左右镜像、metadata交换陷阱、原点证据、2s时域边界、有限尾段不被当5s停车；绝不使用真实dense path馈入控制。
- 本次不重新开GPU。真正运行前由根代理在空闲GPU上做checkpoint key兼容检查和一帧真实模型契约检查，固定权重/源码/环境/优化开关并归档。

固定路线建议沿用**24211 Town01 DynamicObjectCrossing、1711 Town12 ParkingCutIn、1773 Town12 ParkedObstacle**，不因某组表现差更换。它们兼有动态参与者、切入与静态绕障；1773历史未完成必须显式保留。路线相邻执行全部实验臂，固定声明seed=0、相机/天气/地图/ego车型/同步步长、原生导航和一次推理/步；同地图复用server但每例重建ego/agent/model状态。最小新增执行建议B/C两臂×3路线=6例；若需完整官方锚点，A另加3例，必须明确增加范围，不能默默扩成9例。所有结果保留，不按更好停车时刻裁切。固定TM seed也不保证NPC生成完全一致（现有3514已观察到不同对手），所以不能宣称两臂经历相同世界状态。记录对手车型/生成失败/首个交互帧等差异；若结论取决于单个交互，在另行确认的小预算内只对该路线两臂各追加一次复跑，并原样保留首轮，不调参。不要自动扩展增益或路线网格。

每帧日志：源图像/运动frame IDs及时间、raw compass、网络raw4点、target point、command、原生desired speed、native和PI控制、尾部前/后控制、选择分支、积分/饱和/gear字段、实际控制和独立真值位置/速度/加速度。网络特征/图像哈希可以用于同帧replay验证，但**闭环不同控制会改变下一帧观察和模型规划**，不能要求两组整条路线预测逐点一致。

驾驶行为主表应同时包含：碰撞与首次接触帧、偏航/越线/blocked、完成进度与到达时间、停车/恢复事件、全程和moving CTE、实际速度与原生预测速度的差、速度波动、纵横向加速度/jerk、转向速率及饱和。CTE相对模型自身预测仅作跟踪诊断，独立路线距离不是绕障时唯一正确性指标；必须查看模型轨迹是否穿障碍以及实际执行是否偏离模型。DS/官方记录照列，但不是唯一优化目标。短轨迹和模型重规划会使“同一时刻误差”含义不同，统计定义写入报告。单seed三路线只能证明有限集行为差异，不能宣称跑榜增益或全模型泛化。

## 小规模真实模型 wall 预算

历史 tcp-fast 的完整两例约444.5s/7.4min；每profile tick分别87.254ms和132.526ms，包含网络/预处理/传感器等待，不能用3.8ms GPU forward估计整条路线。1773只跑661tick即被用户取消，没有完整时间；按已见约136ms/tick，若约2000tick结束是4–5min级，若走满现有4000tick guard是约9–10min级，另有启动/模型载入开销，且真实失败可能不同。

因此同规模三路线**一臂约12–18min规划区间**；B/C六例约24–36min，A/B/C九例约36–54min，再预留初次server/model启动及异常重试时间。这是基于旧观测的预算，不是保证；预处理版本和慢速执行尾部变化会改变tick数。每例600s墙钟上限、所有超时/未完成照列可作为停止边界；不要为省时间把4000tick失败变成成功。此估算不能外推220路线，也不能归因于某个控制器加速。

## 本次只读源码哈希

| 文件 | SHA256 |
|---|---|
| team_code/tcp_b2d_agent.py | 091b664963c6d7fca75c501df7945684759952614edd934b127b43b9db8e0bcc |
| TCP/model.py | 575ce747444ccb7cf288690393eb48963225668948d84ce5c293b667af522645 |
| TCP/config.py | 9bd25c9f61b2b568ba00cdb9f3b8a5e685487f662fbcefe8839f8652016d80fa |
| TCP/data.py | a7185775e47e8b4b98ebdc2d78d9df627eef12926f953ae0dd28b0da11520d91 |
| tools/gen_tcp_data.py | df913913b9306fd9ad0129d10d00d40a29ea85d4acd6418e6840ad463c636ea0 |
| scripts/b2d_tcp_visual_agent.py | 66a5b583ae09fea83739296fca761a71f33460285c4d626f9824a508778f35f5 |
