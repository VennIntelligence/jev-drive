# Zero-shot 闭环考试：Alpamayo 1.5 与 openpilot 在 Bench2Drive 上

状态: 预注册已冻结（2026-09-24 17:20，smoke 之前；plumbing 只用来查适配，不计分）；smoke 进行中；全量 220 条**等用户批准**
主题: ../../research/openpilot-and-open-driving-models.md、../../research/benchmarks-and-evaluation.md
相关: [Alpamayo smoke](../2026-09-24-alpamayo-smoke/README.md)、[openpilot smoke](../2026-09-24-openpilot-smoke/README.md)、
[控制器 API](../../docs/b2d-controller.md)、[闭环成本](../../docs/bench2drive-cost.md)

## 目标

把两个现成的开放驾驶模型不做任何训练、原样放进 Bench2Drive（CARLA 0.9.15 上的 220 条闭环路线，
指标是 DS（Driving Score，route completion 乘以违规惩罚系数）和 SR（Success Rate，无违规走完的比例））：
NVIDIA Alpamayo 1.5（10B reasoning VLA，输出 6.4 s 轨迹）和 comma openpilot 的 Lebowski（877M，输出 plan 与
desired curvature/accel）。这一轮只做 smoke（5 条路线）和全量成本估计；全量跑要用户看过估计之后批准。

闭环相对开环考试（WOD-E2E、NAVSIM）的优势是**传感器由我们摆**：相机的内参、FOV、安装位置和帧率都可以按模型
自己的车来生成，适配损失主要只剩渲染外观（domain gap）和车辆/道路环境本身。下面每一项都写清楚“复现了什么、
还剩什么妥协”，先冻结再跑。

## 预注册（冻结于 smoke 之前）

### 1. 传感器 rig

**Alpamayo 1.5。** 模型训练数据是 PhysicalAI-AV（NVIDIA Hyperion 车队）。我们下载了 chunk 3119 的标定
（`calibration/{camera_intrinsics,sensor_extrinsics,vehicle_dimensions}`，100 个 clip），取中位数作为 rig。
这批车的轴距 2.850 m、车长 4.87 m，与 Bench2Drive 的 ego（Lincoln MKZ 2020，轴距 2.860 m）几乎一样，
所以安装位置可以按“离后轴的距离”直接搬过来，不需要缩放。rig 坐标系原点在后轴中心地面，x 前 y 左 z 上，
这也正是模型 egomotion 和输出轨迹的坐标系（雷达与车长数据核对过：前保险杠在 x = 3.8 m）。

| 相机（模型 index） | 位置 x/y/z (m, 后轴地面) | yaw / pitch (°) | 原生镜头 | CARLA 渲染（pinhole） |
|---|---|---|---|---|
| cross-left (0) | 2.531 / 0.938 / 0.897 | 66.9 / 0.0 | f-theta 120°，1920×1080 | 1368×784，HFOV 135.6° |
| front-wide (1) | 1.779 / 0.000 / 1.433 | −0.4 / −0.5 | f-theta 120° | 1384×784，HFOV 136.2° |
| cross-right (2) | 2.535 / −0.926 / 0.902 | −66.6 / 0.7 | f-theta 120° | 1392×792，HFOV 136.3° |
| front-tele (6) | 1.729 / 0.093 / 1.446 | −0.1 / −0.2 | f-theta 30° | 608×352，HFOV 30.7° |

CARLA 只有 pinhole 相机，所以每路相机渲染一张覆盖整个 f-theta 视场的 pinhole 图，再在 policy server 的 GPU 上用
`grid_sample`（双线性）按该相机的 f-theta 多项式（`fw_poly/bw_poly`、主点 cx/cy，均为标定中位数）重采样成模型真正看到
的 576×320 图。576×320 就是 Qwen3-VL processor 对 1920×1080 原图的缩放结果（像素数在 163840–196608 之间、边长为 32
的倍数），所以我们直接生成这一尺寸，processor 不再缩放。pinhole 渲染的焦距取 f-theta 中心在 576 宽下的尺度
（279 px/rad），中心区域 1:1 采样、边缘过采样。几何上唯一没复现的是 roll（各相机都小于 0.25°）。
时序：4 帧 @10 Hz（CARLA `sensor_tick = 0.1`），与训练一致。

**openpilot（Lebowski）。** comma 3X 两路相机：road（1928×1208，焦距 2648 px，HFOV 40.0°）和 wide（同尺寸，
焦距 567 px，HFOV 118.9°）。openpilot 自己的 warp 把两者都当 pinhole 处理（`common/transformations/camera.py`），
我们在 CARLA 里按这两个内参原样渲染，再用 modeld 同款的最近邻 warp（`jevdrive/openpilot/frames.py`，
calibration rpy = 0，因为 CARLA 相机水平、正对前方，正是 calib frame）得到 512×256 的 model frame。
颜色：CARLA 的 RGB 按 BT.601 limited range 转 YUV420（comma1M 实测 Y 在 16–245，是 limited range），
chroma 取 2×2 平均，与 NV12 一致。

安装位置：与 Hyperion 的 front-wide 相机完全相同（后轴前 1.779 m、离地 1.433 m，即挡风玻璃上沿；同轴距的车，
位置可以直接搬）。plumbing 阶段试过两个更低的位置，都不行，数字留在这里：openpilot 的名义相机高 1.22 m 在 MKZ 上
让发动机盖占掉 road frame 底部约 20%（真实 comma road frame 看不到发动机盖）；1.35 m 时相机在 CARLA 这辆车的
挡风玻璃后面，玻璃是深色的，整幅图明显变暗（同一场景 Alpamayo 的相机亮度正常）。comma 装在玻璃内侧、离车顶线
约 8 cm，我们放在玻璃外同一 x 处、高约 8 cm，这 8 cm 的高度差是剩余妥协。
时序：两路都是 5 Hz（`sensor_tick = 0.2`），见下面第 4 节。

**两个模型共同的剩余妥协（都算 handicap，不修）：** CARLA 渲染外观（光照、材质、噪声、夜景）与真实相机 ISP 不同；
wide 相机在真车上是鱼眼，openpilot 按 pinhole 处理，我们渲染的是真 pinhole，所以 model frame 边缘的放大率与训练时略有
差别；Hyperion 车队的车身与 MKZ 不完全相同，cross 相机若被车身遮挡要在 smoke 的图里看出来。

### 2. 路线信息

Bench2Drive 给 agent 的是稠密路线（GPS + world 坐标 + RoadOption：LEFT/RIGHT/STRAIGHT/LANEFOLLOW/CHANGELANE*）。
我们只把模型在真车上能拿到的那一种形式交给它：

- **Alpamayo**：自然语言 nav（`<|route_start|>…<|route_end|>`）。用 GNSS+罗盘+里程计的位姿
  （`b2d_controller_adapter.PoseFilter`，只用传感器）在路线上求进度，往前找下一个 junction 命令；若是 LEFT/RIGHT
  且距离 ≤ 60 m（约模型 6.4 s 的视野），给 `"Turn left in {d}m"` / `"Turn right in {d}m"`（d 取整，在路口内为 0）。
  这是官方 notebook 和 `nav_demo_samples.json` 里唯一出现过的句式（TURN_LEFT/TURN_RIGHT，9–36 m）。
  直行路口、变道和一般跟车**不给 nav**（用默认 prompt），因为我们不知道训练里“直行”“变道”的句式，
  写一个模型没见过的句子比不写更糟。这意味着直行穿越路口时模型要自己判断，属于已知 handicap。
- **openpilot**：没有导航输入，只有 desire（8 维 one-hot，modeld 只喂上升沿脉冲）。路口 LEFT/RIGHT 前 20 m 起
  给 `turnLeft/turnRight`，在 CHANGELANELEFT/RIGHT 段内给 `laneChangeLeft/Right`，其余 `none`。openpilot 的
  turn desire 本来只在低速打灯时出现，模型是否会据此在路口转弯没有验证过；**预期它在需要转弯的路线上大量偏离路线**，
  这是这个模型在这个考试上的结构性 handicap，结果要单独标注。

### 3. 控制：两个模型都走同一个固定控制器

两个模型的轨迹都交给仓库的固定控制器 `scripts/b2d_controller.py`（`Controller(preset="carla")`，
纵向 vendor 模式，冻结的 MKZ 参数 `todos/2026-09-22-b2d-controller/results/controller_config.json`，
20 Hz 执行，按时间戳用里程计把轨迹重投影到当前位姿）。输入统一是后轴坐标、0.25–5.0 s 的 20 个点：
Alpamayo 的 64 点 @10 Hz 直接线性插值；openpilot 的 plan（33 个非均匀时间点，calib frame，原点在相机）先平移到后轴
（x + 1.779 m）、y 取反（device frame y 向右）再插值。

为什么 openpilot 也不用它自己的 desired curvature/accel：那两个量是给 openpilot 里按车型标定的横纵向控制器
（torque/angle 控制 + 车辆模型）用的，CARLA 的 MKZ 没有这层；把 curvature 换成方向盘角本来也要一个车辆模型，
那正是这个控制器已经标定过的部分。两个模型走同一个执行层，DS 的差别才能归到 planner 上。代价是 openpilot 的 plan
在它自己车上是由 MPC 跟踪的，这里换成 pure-pursuit/PID 式跟踪，属于轻微 handicap。curvature/accel 仍逐次记录，
方便事后对照。

### 4. 规划频率、同步与延迟

CARLA 以 20 Hz 同步步进，模型推理时仿真暂停，所以推理延迟只花墙钟，不影响驾驶；**不模拟延迟**（轨迹时间戳就是输入帧
的仿真时间）。这对 Alpamayo 是有利的偏差（真车上 0.7 s 的延迟会让轨迹变旧），写进结果的解读里。

- **Alpamayo：2 Hz 重规划**（每 5 个 10 Hz 相机组规划一次），两次之间控制器按 20 Hz 跟踪上一条轨迹（6.4 s 视野远大于
  0.5 s 间隔，控制器的 stale timeout 也是 0.5 s）。Bench2DriveZoo 的 AD-MLP 同样是 2 Hz 推理。
- **openpilot：5 Hz**，即它的 context 频率。Lebowski 的队列在 ONNX 外面，某一相位 t 的输出只依赖帧 t−4、t，
  隐状态 t−96…t−4（步长 4）和按 4 步对齐 max-pool 的 desire，所以**只跑一个相位与 20 Hz modeld 在该相位的输出完全
  相同**。已在 comma1M 段 `0045b4fe` 的 400 帧（含一个 turnLeft 脉冲）上验证：100 个相位步的输出最大差 **0**
  （`scripts/test_zeroshot_openpilot_context.py`，TRT fp16）。所以 5 Hz 不是近似，只是不计算用不到的相位。
- CARLA 的 `sensor_tick` 让不同相机的相位可能差一个 tick（生成顺序和 UE tick interval 的余量累积）。agent 把
  “每路都比上一组新”的帧拼成一组，组帧号取最新；各路帧号记录在 `plans.jsonl`。plumbing 实测（route 2390，
  40 次规划）：组间隔 117/117 都是 2 tick（正好 10 Hz），规划间隔 39/39 都是 10 tick；cross-left 在 38/40 组里比
  其他三路早一个 tick（50 ms），这一点不修，算剩余妥协（10 m/s 时约 0.5 m 的侧视图时差）。

### 5. 推理配置

- Alpamayo：SDPA + `torch.compile`（vision、expert）+ flow matching 5 步，n = 1 条轨迹，温度 0.6、top-p 0.98，
  seed = 规划序号。这是 Alpamayo smoke 里“输出基本不变”档的最快配置（p50 697 ms，与默认配置的轨迹差 0.17 m，
  远低于换 seed 的 1.28 m）。保留 CoC 推理（不关 reasoning）。
- openpilot：Lebowski，onnxruntime TensorRT EP fp16（smoke 数值等价已核），traffic convention 左舵。
- 一个 resident server 进程服务所有 CARLA worker（unix socket，`scripts/zeroshot_policy_server.py`），
  GPU 上用一把锁串行；openpilot 每个 worker 的递归状态在 server 里按连接分开保存。

### 6. Smoke 路线（按 id 预先固定）

| route | town | 场景 | 选它的理由 |
|---|---|---|---|
| 24211 | Town01 | DynamicObjectCrossing | 小镇、行人横穿；TCP 在我们这里 393 tick 跑完 |
| 2390 | Town12 | VanillaNonSignalizedTurn | 无信号路口转弯，考 nav/desire；控制器 smoke 的标准路线 |
| 1711 | Town12 | ParkingCutIn | 成本锚点：policy none 288 s / 1283 tick |
| 2373 | Town12 | VanillaSignalizedTurnEncounterRedLight | 红灯 + 转弯 |
| 3564 | Town13 | InvadingTurn | Town13 的每 tick 成本（全量里 Town12+13 占 69% 路线、83% 墙钟） |

TM seed 0，每条一次（`--max-attempts 1`，基础设施失败才重跑）。所有规划都存模型输入图（带预测轨迹投影）；
route 2390 另外合成视频。

### 7. 记什么、怎么判

- 官方 `results.json`：DS、RC（route completion）、违规列表、状态；SR 按 Bench2Drive `merge_route_json.py`
  的定义（完成且无违规）。
- 成本：每条路线的墙钟、tick 数、s/tick（`route_result.json` 的 tick profile），每次规划的往返时间和 server 推理时间。
- 适配是否正确（visual check）：模型输入图的地平线位置、车身遮挡、左右是否颠倒、投影后的轨迹是否落在可行驶区域；
  openpilot 与真实 comma1M model frame 对照。发现适配错误就修，修完**重跑受影响的 smoke**，不当作模型分数。
- 全量估计：用本次 smoke 实测的每 tick 成本（分模型、分 town）乘 full220 实测的每 town tick 数与 route 数，
  再按并发 worker 数和共享 GPU 的排程折算。

## Smoke 结果

（进行中，结果到了就写在这里。）
