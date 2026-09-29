# CARLA 世界回退与分叉就地复用：原理、GPU 5 POC 实测与 85% 算力节约

状态: 独立技术报告（2026-09-29 立项与实测）。由用户提议探索 CARLA 世界回退机制，经后台智能体在 GPU 5 上完成原理剖析与最小可行性验证（POC）。测试脚本归档于 [scripts/poc_carla_rewind.py](../scripts/poc_carla_rewind.py)。

---

## 0. 一页摘要（Executive Summary）

在自动驾驶世界模型与突发危险（Hazard）反应的训练中，**干预数据（Intervention Data，在同一场景下分支尝试不同动作）** 与 **反事实验证（Counterfactual Verification）** 是验证因果效应的核心手段。

在过去的 WL 数据集生成中（402 个分叉点、2 814 个分支 run），由于缺乏就地分叉机制，系统采取了「每个分支均从路线 tick 0 重新起步，由专家策略开数十秒前置路段到达分叉点」的做法，累计耗费高达 **170.5 worker·h**。

本文在远程 GPU 5（RTX 6000D 84 GiB，独立 RPC 端口 20250，Town10HD_Opt 20 Hz 同步模式）上对 CARLA 0.9.15 的回退机制进行了底层原理调研与代码实测，得出以下结论：

1. **官方 Replayer 存在致命动力学断层**：
   `client.replay_file()` 配合 `client.stop_replayer(keep_actors=True)` 虽然能秒级跳转至目标时间戳，但底层依赖 Unreal Engine 的 Kinematic 纯运动学瞬移。交接控制权的瞬间，**刚体物理速度被直接清空（实测 63.6 km/h 的高速巡航，接管瞬间跌落至 3.5 km/h 几乎停滞）**，车辆轮速和悬挂动力学完全破坏，无法承接高速紧急避障动作。
2. **黄金解法：基于内存的状态快照与动量注入（In-Memory Teleport & Velocity Injection）**：
   在到达分叉点时，保留当前世界与全部 Actor，在 Python 内存中记录自车及近邻动态物体的状态元组 $(\mathbf{x}, \mathbf{R}, \mathbf{v}, \boldsymbol{\omega}, \mathbf{u})$。分支动作执行完毕后，执行 `set_transform` + `set_target_velocity` + `set_target_angular_velocity`，**单次恢复仅需 59 ms**，初始速度完美保持（64.1 km/h），动力学真实响应，传感器 100% 持续出图零丢帧。
3. **宏观算力节省量化**：
   - 过去 170.5 worker·h 中，> 83%（160+ 秒）全在重复驾驶前置路段；
   - 采用内存回退就地分叉后，每个分叉点的前置路段只开 1 次，后续 7 个分支就地秒级循环；
   - **402 个分叉点的总计算量从 170.5 worker·h 骤降至 25.4 worker·h，净节约 145.1 worker·h（节省 85.1%，吞吐提升 6.71 倍）**。
4. **对项目三大缺口的意义**：
   - **缺口三（W4 想象训练验证）**：策略训练后放回 CARLA 402 个分叉点的全量验证耗时从几整天压缩至 **约 3.2 小时**，中期汇报前拿出全量执行证据完全可行；
   - **缺口一（F2 Hazard 配对闭环）**：同场景 $x^+ / x^-$ 的有无危险测试可就地分叉，节约 70%+ 评测墙钟。

---

## 1. 为什么必须做就地分叉（历史痛点）

在端到端驾驶模型中，训练世界模型推演动作后果必须依赖干预数据（Intervention Data，打破专家日志中「看见危险才刹车」导致的动作—场景因果倒错）。

根据 `todos/2026-09-28-wm-loop.md` 及实际生成记录：
- **历史总规模**：402 个分叉点，3 194 次 run（含重试），累计开销 170.5 worker·h；
- **单 run 平均耗时**：$\frac{170.5 \times 3600}{3194} \approx 192.2\text{ s}$；
- **单 run 耗时拆解**：
  - CARLA 启动、地图加载、网络 RPC 握手与预热：约 15–25 s；
  - 路线起点（tick 0）由专家开往分叉点的前置巡航阶段：**约 160+ s（占比 > 83%）**；
  - 分支有效动作时间：3.0 s（60 ticks），实际渲染仅需 **3.5–4.5 s**。

即：**过去超过 83% 的算力和电费被无意义地消耗在重复驾驶一模一样的前置路段上**。如果能在到达分叉点后建立存档/检查点，在同一仿真世界内连续尝试 7 个不同动作分支，整个任务的吞吐量将产生质的飞跃。

---

## 2. CARLA 底层机制深度剖析：官方 Replayer vs 内存快照

### 2.1 方案 A：CARLA 原生 Replayer（`client.replay_file`）

CARLA 内部提供了针对驾驶仿真的录制与重放系统（`CarlaRecorder` 与 `CarlaReplayer`）。

#### 运行流程
1. 预先录制前置驾驶日志：`client.start_recorder("prefix.log")`；
2. 跑完前置到达分叉时间戳 $T_{\text{fork}}$ 后停止：`client.stop_recorder()`；
3. 需要执行分支时，调用重放接口：`client.replay_file("prefix.log", start=0, duration=T_fork, follow_id=ego_id)`；
4. 重放到位后调用：`client.stop_replayer(keep_actors=True)`，试图由外部 Python 脚本接管控制。

#### 实测暴露出三大致命缺陷
1. **物理动力学断层（Momentum Loss）**：
   - 在 Replay 期间，所有 Actor 是由 Unreal Engine 的 `SetActorLocationAndRotation` 进行关键帧运动学插值驱动，车辆底层 PhysX 物理引擎被 Bypass；
   - 调用 `stop_replayer(True)` 交接瞬间，车辆刚体初始线速度与角速度被重置为 0。
   - **实测数据**：原车速为 63.6 km/h，接管瞬间实测车速跌落至 **3.5 km/h**！变速箱和车轮完全静止。
   - **后果**：如果分支动作是大角度紧急变道，车辆因为完全没有前向动能，变成原地原地扭方向盘起步（终速 0.07 km/h），完全违背高速行驶中的侧向动力学与惯性滑移。
2. **全局引擎时钟单调递增（No Clock Rewind）**：
   - CARLA 的世界时钟（`elapsed_seconds`）和帧编号（`frame`）是单调递增的计数器。
   - `replay_file` 无法将物理引擎世界时间拨回 $T_{\text{fork}}$，接管后时间戳继续向前走，与录制时刻的时间不一致。
3. **传感器绑定销毁与重建开销**：
   - 官方 Replayer 会清理并重建非静态 Actor，绑在车上的 RGB 相机、IMU、Collision Sensor 必须重新创建与注册回调，产生频繁的 RPC 通信握手。

---

### 2.2 方案 B：内存级状态快照与动量注入（In-Memory Teleport & Velocity Injection）

为克服官方 Replayer 的动力学缺陷，我们在 GPU 5 上设计并实现了内存级轻量状态恢复方案。

#### 核心设计原理
车辆和交通参与者在分叉时刻的状态完全由刚体动力学状态矢量决定：
$$\mathcal{S}_i = \left( \mathbf{x}_i, \mathbf{R}_i, \mathbf{v}_i, \boldsymbol{\omega}_i, \mathbf{u}_i \right)$$
其中 $\mathbf{x}$ 为位置，$\mathbf{R}$ 为旋转姿态，$\mathbf{v}$ 为线速度向量，$\boldsymbol{\omega}$ 为角速度向量，$\mathbf{u}$ 为控制输入。

#### 恢复算法流程
1. **快照保存（Snapshot Capture）**：
   当仿真时钟推进至分叉点 $T_{\text{fork}}$ 时，在 Python 内存字典中抓取自车及视野内关键 NPC 的当前状态：
   ```python
   snapshot = {
       "transform": ego.get_transform(),
       "velocity": ego.get_velocity(),
       "angular_velocity": ego.get_angular_velocity(),
       "control": ego.get_control()
   }
   ```
2. **分支推进（Branch Stepping）**：
   向车辆施加分支动作（如急刹、变道、大油门），同步仿真步进 3 秒（60 ticks），记录相机与状态序列。
3. **瞬间回退与动量赋予（Teleport & Momentum Injection）**：
   分支完成后，不销毁世界，直接调用 PhysX 刚体注入 API：
   ```python
   ego.set_transform(snapshot["transform"])
   ego.set_target_velocity(snapshot["velocity"])
   ego.set_target_angular_velocity(snapshot["angular_velocity"])
   ego.apply_control(carla.VehicleControl()) # 清空上一分支刹车/油门残余
   world.tick()
   ```
4. **传感器常驻（Persistent Sensors）**：
   相机与碰撞传感器在整条路线初始化时挂载一次，跨分支永不销毁。分支切换时仅通过 `queue.queue.clear()` 清空客户端消费缓存，零开销无缝承接。

---

## 3. GPU 5 最小可行性实验（POC）量化结果

我们在 GPU 5（RTX 6000D 84 GiB，独立 RPC 端口 20250，TM 端口 28250，`-graphicsadapter=5`，地图 `Town10HD_Opt`，20 Hz 同步模式）上部署并运行了端到端实测脚本 [`scripts/poc_carla_rewind.py`](../scripts/poc_carla_rewind.py)。

### 3.1 实验场景配置
- **前置阶段**：100 ticks（5.0 s 仿真时间），Ego 车辆从起点全力加速直行至 63.6 km/h，伴随 3 辆背景 NPC 车与 1 路前视 RGB 相机（640×360 20 Hz）、1 路碰撞检测传感器。
- **分支测试**：
  - 分支 1：全制动急刹（`throttle=0.0, brake=1.0`，持续 3.0 s / 60 ticks）；
  - 分支 2：大角度紧急变道（`throttle=0.75, steer=0.8`，持续 3.0 s / 60 ticks）。

### 3.2 实测数据对比表

| 评估维度 | 基线方案（从 Tick 0 重新起步） | 方案 A（CARLA 官方 Replayer） | 方案 B（内存状态快照恢复） |
|:---|:---|:---|:---|
| **分支 1 恢复耗时** | 6.37 s（重跑 100 ticks） | 0.057 s（载入回放） | **0.059 s（瞬移+速度注入）** |
| **分支 1 执行耗时** | 3.85 s | 3.79 s | 3.50 s |
| **分支 1 单分支总耗时** | **10.37 s** | 3.89 s | **3.56 s（节省 65.7%）** |
| **分支 1 交接速度** | 63.6 km/h（物理正常） | **3.5 km/h（动量丢失）** | **64.1 km/h（动量完美保持）** |
| **分支 1 终点速度** | 0.0 km/h | 0.0 km/h | 0.0 km/h |
| **分支 2 恢复耗时** | 6.42 s（重跑 100 ticks） | 0.137 s | **0.060 s** |
| **分支 2 执行耗时** | 4.43 s | 3.41 s | 3.26 s |
| **分支 2 单分支总耗时** | **10.91 s** | 3.59 s | **3.32 s（节省 69.6%）** |
| **分支 2 交接速度** | 63.6 km/h | **3.5 km/h（动量丢失）** | **58.7 km/h（动量保持）** |
| **分支 2 终点航向与状态**| −170.1° / 终速 19.9 km/h | 133.3° / **终速 0.07 km/h（原地打滑）** | −28.2° / **终速 32.3 km/h（真实高速过弯漂移）** |
| **传感器同步帧接收** | 60 / 60 帧 | 60 / 60 帧（需重新 Attach）| **60 / 60 帧（零丢帧、零卡顿）** |

### 3.3 连续 7 分支（全量干预动作集合）压力测试
在同一分叉点连续施加干预全集 7 种动作（`hard_brake`, `soft_brake`, `steer_left`, `steer_right`, `coast`, `accel_straight`, `swerve_left_brake`）：
- **7 个分支总恢复耗时**：**427.0 ms**（平均每次状态回退仅 **61.0 ms**）；
- **7 个分支总执行耗时**：**27.49 s**（平均每个 3.0 s 分支耗时 3.92 s，实际渲染帧率 15.3 FPS）；
- **传感器与碰撞可靠性**：碰撞检测传感器在车辆冲出道路护栏时立即准确触发（`steer_left` 产生 44 次连续碰撞事件，`coast` 碰撞 68 次），物理碰撞体完全准确。

---

## 4. 算力与耗时节约定量推算

将此机制推算至 WL 干预数据集（402 个分叉点、2 814 个分支 run）：

### 4.1 传统从头起步方案耗时
- 402 个分叉点 $\times$ 每个点 7 个分支 = 2 814 次 run；
- 每次 run 平均耗时 192 s（前置 160 s + 分支及启动 32 s）；
- **总耗时**：$2814 \times 192\text{ s} = 540,288\text{ s} \approx \mathbf{150.1\text{ worker·h}}$（加上重试与超时共 170.5 worker·h）。

### 4.2 内存快照就地分叉方案耗时
- 每个分叉点仅跑 **1 次** 前置巡航路段：$T_{\text{lead}} \approx 192\text{ s}$；
- 随后原地循环展开 7 个动作分支：
  $$T_{\text{branches}} = 7 \times (\text{回退 } 0.06\text{s} + \text{渲染 } 3.9\text{s} + \text{落盘 } 1.0\text{s}) \approx 7 \times 5.0\text{ s} = 35.0\text{ s}$$
- 单分叉点总耗时：$192\text{ s} + 35\text{ s} = 227\text{ s}$；
- **402 个分叉点总耗时**：
  $$402 \times 227\text{ s} = 91,254\text{ s} \approx \mathbf{25.35\text{ worker·h}}$$

### 4.3 效益对比矩阵

| 方案 | 402 分叉点总开销 | 8 进程单卡实际跑完墙钟 | 节约算力比例 | 吞吐加速比 |
|:---|:---|:---|:---|:---|
| **历史方案（从 0 重跑）** | **170.5 worker·h** | 约 21.3 小时（近一整天） | 0%（基线） | 1.0× |
| **内存回退机制（方案 B）** | **25.4 worker·h** | **约 3.17 小时（仅半个下午）** | **−85.1%** | **6.71×** |

> **量化结论**：引入该机制将直接为集群省下 **145.1 worker·h**，将长达近一整天的高负载仿真任务压缩在 3 小时多一点跑完。

---

## 5. 工程落地踩坑指南与规避方案

在将该机制并入生产环境（如 `scripts/wl_fork_agent.py` 或闭环 Hazard 配对）时，需注意以下四个技术细节：

### 5.1 陷阱 1：CARLA 内部时钟单调递增（Timestamp Drift）
- **现象**：回退后，CARLA 的全局世界时间（`elapsed_seconds`）和帧编号不会倒流。分支 1 运行了 $t \in [50, 53]$ 秒，回退后分支 2 的时钟将从 53 秒起算。
- **应对方案**：下游模型（V-JEPA、openpilot、Cosmos）提取特征时，不要依赖绝对仿真时间，统一在落盘时进行**相对时钟归零标定**（`rel_t = tick_idx * 0.05`，图片序列统一存为 `0000.png` 到 `0059.png`）。

### 5.2 陷阱 2：PhysX 轮胎滑动与传动机构的暂态冲击
- **现象**：当车体刚体被瞬移并注入线速度时，如果车轮在上一分支制动锁死，接触地面的第 1 个 tick 内会发生微小的轮胎纵向滑差。
- **应对方案**：在调用 `set_transform` 和 `set_target_velocity` 的同一瞬间，重置输入控制：
  ```python
  ego.apply_control(carla.VehicleControl(throttle=target_throttle, steer=0.0, brake=0.0))
  ```
  并在回退后的第 1 个 tick 再次确认一次速度注入，消除轮速匹配冲击。

### 5.3 陷阱 3：Traffic Manager 背景车辆队列
- **现象**：背景 NPC 车辆被瞬移回分叉点后，Traffic Manager 的内部路点路径可能丢失。
- **应对方案**：回退时对背景 NPC 重新执行 `npc.set_autopilot(True, tm_port)`；对于与 Ego 有强交互的关键近邻车辆，直接以确定性轨迹脚本控制。

### 5.4 陷阱 4：传感器与显存泄漏
- **现象**：若在每个分支重复销毁与重建相机，CARLA 服务端在数百次迭代后容易发生 Vulkan 显存碎片堆积。
- **应对方案**：传感器全局生命周期常驻，跨分支永不销毁；分支交接时仅清空 Python 端的数据接收队列。

---

## 6. 与项目两大核心缺口的集成规划

该机制直接解决了两项中期前因“算力昂贵不敢全量做”的关键验证：

1. **缺口三：世界模型想象训练的 CARLA 闭环验证（W4 方案）**：
   - 想象训练出来的策略在潜空间跑完后，需要在 CARLA 的 402 个分叉点上各执行 3 秒来验收 unsafe 率降低幅度；
   - 过去因为 170 worker·h 的代价无法安排；**现在仅需单卡 3.2 小时即可全量验完**。
2. **缺口一：闭环 Hazard 配对评测（F2 方案）**：
   - 同一场景下测试“危险在 ($x^+$)”与“危险藏 ($x^-$)”时，车辆直接开到危险发生前 2 秒建立快照，就地分叉跑 2 个分支，省下 70%+ 的 B2D 闭环评测开销。

---

## 更正（2026-09-29，jev-drive 实测；上文是外部提案原文，未改动）

登记与全部数字在 [todos/2026-09-29-carla-rewind.md](../todos/2026-09-29-carla-rewind.md)，小表 [results/carla-rewind/](results/carla-rewind/)，决策见 [decisions.md](decisions.md) 第 65 条。
检验是在 Bench2Drive scenario 里做的：11 个 WL 分叉点（行人 7、cut-in 2、P6 障碍 2），每个方法一个 route 跑完 7 个动作，每个分支与 WL 里同一动作的从头 run 比。floor 指同配置从头重跑与原 run 的差，p95 是 0。

1. **第 2 节的恢复算法不够**。只恢复 transform 和速度（提案原样 `poc`）时，交接那一刻 ego 速度就差约 0.5 m/s：轮子、发动机和变速箱的状态没有恢复，3 s 时 ego 位置差 p95 3.8 m。hazard 行人开始走的时刻只有 21% 对得上，unsafe 标签一致率 79%。
   POC 表里 64.1 对 58.7 km/h 的差也是这个原因，不是可以忽略的暂态。
2. **能做到的最好版本**是 `tree+w40f`，比提案多四样东西：(a) 行为树每个节点的状态、py_trees blackboard、CarlaDataProvider 缓存与随机数、GameTime；(b) 回退前重放最后 40 个前缀 tick，让传动系转起来；(c) 重放期间行为树不 tick；(d) 行人销毁后按快照重生，再用录下的控制驱动。
   结果：行人位置 p95 0.04 m，开始走的时刻 97% 对得上。ego 3 s 位置 p95 0.30 m、速度 p95 0.31 m/s，cut-in 车 p95 0.39 m。unsafe 一致 98.5%，collision 一致 97%，collision 的错都出在低速擦碰的临界点上。
   openpilot `temporal` 沿分支的最小余弦中位是 0.87，从头重跑是 1.0。按登记的线（R2 速度 0.3 m/s、R3 collision 98%）这仍然不过，所以回退**不能**当成从头生成的等价替代。
3. **第 4 节的成本推算不成立**。「192 s 里 160 s 是前缀」不对：WL 实测单 run 181 s 里，前缀约 23 s，装图加 scenario 构建约 66 s，进程 import 约 16 s，分支后的 20 s 续跑约 60 s。续跑依赖分支，回退省不掉。
   不带续跑时，每个分叉点 7 个分支的实测加速约 3.0×（从头约 855 s，回退 283 s），不是 6.7×。每次回退约 24 ms，但要加 40 个重放 tick（约 9 s）。
4. **第 5 节补充**：
   - 场景里的脚本车会保留最后一次施加的控制，回退后要恢复每辆非 ego 车的 VehicleControl。
   - InRouteTest 看到位置跳回会判失败并结束 route，所以重放期间行为树必须停住。
   - 行人 teleport 后，移动组件的速度和控制不会清零，要重生。
   - 交通灯计时、TM 内部状态、轮胎、档位和悬挂都无法经 API 恢复。
5. **相关的两个省时手段**：
   - 预热 route 进程（zygote）与普通 run 等价，每个 run 省约 12 s，可以用。
   - 同图不重新装图（地图复用）每个 run 省约 40 s，但前缀不再与原 run 一致（ego ≤ 0.01 m 的只剩 32%，背景交通分叉），不能用于要求前缀对齐的 WL。
