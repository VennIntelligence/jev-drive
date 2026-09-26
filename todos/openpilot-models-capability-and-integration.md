# OpenPilot 三大模型体系：能力边界、特征组合、配对训练与仿真策略接入全景指南

> **目标**：全面梳理系统中关于 openpilot 三大模型（small、Cinque、Lebowski）的实测能力边界、特征表征组合（`temporal` vs `vision`）、配对差分训练机制（Pair-Δ），以及如何无损接入闭环仿真器（CARLA / Bench2Drive）与底层策略执行器。  
> **文档位置**：`tmp/openpilot-models-capability-and-integration.md`

---

## 1. 三大模型谱系与基本架构（The Three Models）

在我们的评测与主线开发中，深入探究了 Comma.ai 演进史上的三个代表性端到端驾驶模型：

| 模型代号 | 架构与时钟特性 | 输入配置与感受野 | 核心定位与特点 |
|:---|:---|:---|:---|
| **small** | 轻量级早期模型，浅层 CNN + 循环时序单元 | 单路或双路低分辨率相机输入，时序记忆窗口短 | 资源开销极小，作为基线与极速边缘推理对照。 |
| **Cinque** (~0.9.7 时代主力) | **20 Hz 主时钟**，双目输入（road 120° + wide 180°），内部维护 **512 维 `temporal` 隐状态** | 512×256 model frame，透视投影校准（warp），RNN 时序记忆窗口达 5 秒 | **当前主线的主力 Backbone**。时序平滑度极高，动力学稳定性极佳，对前车与加塞极度敏感。 |
| **Lebowski** (最新大模型) | **5 Hz context rate**，深度网络，上下文容量大幅扩展 | 采用更大的模型容量与更长的 context step（0.2s 步长） | 拥有更强的长期上下文语义捕捉能力，但对时序抖动和输入延迟更敏感。 |

---

## 2. 真实能力边界（Capacity Boundaries & Empirical Findings）

通过在 Waymo (WOD-E2E)、NAVSIM 以及我们自研的 P5/P6 反事实考卷上的横向对比，摸清了其真实的物理能力边界：

### ① 开环真实道路水平（Waymo WOD-E2E）
* **结论**：**表现惊艳，逼近甚至匹敌专业大模型**。
  * 在 479 个人类驾驶仲裁帧（rater frames）上，Cinque 原生 Plan 的 RFS（人工偏好分）达到 **8.005**，Lebowski 达到 **7.886**，显著击败常规基线（如 CV 7.26），直接进入顶会顶尖模型梯队。
* **输入协议致命伤（NAVSIM 暴跌的真相）**：
  * openpilot 在 NAVSIM 上跑分暴跌，并不是模型不行，而是 **NAVSIM 的 2 Hz、1.5 秒采样并保持（sample-and-hold）协议对时序连续模型造成了降维打击**；
  * 实验证明：把 2 Hz 离散帧强行喂给 20 Hz 的 openpilot，其内部速度估计会被**虚假放大约 2.5 倍**，导致纵向剧烈超调；一旦恢复 10 Hz 真实时间流输入，模型能力立即恢复。

### ② 长尾避险能力边界（P5 考卷发现）
* **对车辆（Vehicles）极度敏感（天生就会）**：
  * 在高速加塞（HighwayCutIn）、盲区切入（StaticCutIn）、前车急刹上，Cinque 和 Lebowski 的原生定向反应翻转率高达 **86% ~ 89%**，甚至压制了 CARLA 专属强模型（如 TFv6 的 42%~46%）。
* **对行人（Pedestrians）几乎完全失灵（长尾盲区）**：
  * 原生模型在突发横穿行人题上，翻转率**直接跌到 0%**！
  * **根因分析**：Comma.ai 设计了约 700-bit 的视觉信息瓶颈，为了保证行车轨迹的高度稳定和平滑，极其稀疏的行人小像素被内部先验当成噪声过滤掉了。

---

## 3. 特征表征与读出头组合（Readout Combinations）

为了充分榨取 openpilot 的强大特征，我们对比了多种解耦组合：

```mermaid
flowchart TD
    Cam["双目相机 (Road 120° + Wide 180°)"] --> OP["openpilot 冻结骨干 (Cinque / Lebowski)"]
    
    OP -->|512 维隐状态| Temporal["temporal 特征 (时序平滑 + 动力学历史)"]
    OP -->|视觉特征图| Vision["vision 特征 (瞬时几何与物体外观)"]
    
    Temporal --> Native["① 原生规划头 (Native Plan)"]
    Temporal --> Ridge["② 线性岭回归 (ridge_late)"]
    Temporal --> Cls["③ 离散轨迹分类头 (cls_late K=1024)"]
    
    Temporal -.-> Fusion["④ 多流配对融合 (M-C 双流架构)"]
    Qwen["Qwen3-VL / YOLO26x"] -.-> Fusion
```

1. **`temporal` vs `vision` 特征**：
   * `temporal` 是 512 维包含时序动力学记忆的循环特征，作为规划头输入时平滑性最好；
   * `vision` 包含更未被时序平滑的物体细节，线性探针（Probing）表明它对静止障碍、锥桶、事故车的可分性极高（AUC 0.88~0.97）。
2. **读出头（Heads）对比**：
   * **`ridge_late`（连续回归残差）**：最保避险反应，能完整保留连续减速与微调；
   * **`cls_late`（K=1024 轨迹词表分类）**：公开榜单常用，但致命弱点在于**词表内没有避障绕行轨迹**，且离散 Anchor 会引入量化跳跃噪声，反而压制了微小减速反应。
3. **M-C 双流异构融合（Multi-Stream Architecture）**：
   * 采用 **`openpilot temporal` ⊕ `Qwen3-VL (L18_last)` / `YOLO26x-seg`**；
   * 解决“偏科”问题：openpilot 负责车道保持、车辆博弈与平顺控制；外部感知分支专门作为“哨兵”，负责在出现行人与静态障碍时发出激发信号。

---

## 4. 配对差分训练（Pair-Difference Training / Elicitation）

常规均匀模仿学习（Imitation Learning）会把 99% 的平稳巡航当成主导，导致模型在关键时刻“不刹车”。为此，团队落地了**反事实配对差分训练**：

### 核心原理
利用 P5 物理引擎生成的反事实配对（$x^+$ 有行人 vs $x^-$ 无行人）：
$$\Delta f = f(x^+) - f(x^-)$$
$$\Delta y = y_{\text{expert}}(x^+) - y_{\text{expert}}(x^-)$$

训练一个轻量的残差头（Reaction Head），只以 $\Delta f \to \Delta y$ 为目标函数进行拟合。

### 实测成效
* 在原本翻转率为 0% 的行人场景中，经过配对差分训练的 M-C 双流 Head，在拟人专家（BA）标签下的**行人反应翻转率直接飙升至 42.1% ~ 46.7%**！
* 证实了：**骨干网络里并非看不见行人，而是需要差分监督将其因果反应显式“激发”出来。**

---

## 5. 策略怎么接入仿真与实车（Policy & Integration Guide）

将 openpilot 接入 CARLA（Bench2Drive）闭环仿真器并连接底层控制器，曾遇到过严重的冲出车道事故。经过深入复盘，锁定了三大关键修复与标准接入流程：

### ① 坐标系变换：相机原点 $\to$ 后轴中心（致命 Bug 修复）
* **踩坑血泪史**：
  * openpilot 输出的规划轨迹（Plan）的原点在**前挡风玻璃上的相机位置**；
  * 底层控制器（如 P7 / Pure Pursuit）默认要求的输入轨迹必须在**后轮轴中心（Rear Axle）**；
  * 早期适配器仅粗暴地做了 $x + 1.78\text{ m}$ 的平移，导致控制器认为 $t=0$ 时刻车身落后了 1.78 米，瞬间换算成 **$7.1\text{ m/s}$ 的全油门起步命令**！车在静止状态被瞬间拽翻冲出车道。
* **标准数学变换公式**：
  设相机相对后轴偏置为 $\mathbf{d} = (1.779, 0)$，相机系轨迹为 $(p_x(t), p_y(t))$、航向为 $\psi(t)$，转入后轴系坐标 $\mathbf{r}(t)$ 必须包含刚体旋转项：
  $$\mathbf{q}(t) = \big(p_x(t), -p_y(t)\big), \quad \psi'(t) = -\psi(t)$$
  $$\mathbf{r}_{\text{rear}}(t) = \mathbf{d} + \mathbf{q}(t) - \mathbf{R}\big(\psi'(t)\big) \cdot \mathbf{d}$$
  变换后，$t=0$ 时 $\mathbf{r}_{\text{rear}}(0) = \mathbf{0}$，静止时控制器进入平稳的 `stop_hold` 状态。

---

### ② 相机时序严格配对（Temporal Synchronization）
* **问题**：road 相机与 wide 相机如果采用异步采样，会出现 1~3 帧（50~150ms）的时序抖动，导致时序横向误差暴增 83%。
* **规范**：
  * CARLA 必须开启同步模式（Synchronous mode），设置 `sensor_tick = 0.05s`（20 Hz）；
  * 保证每个仿真步同时渲染 road 和 wide，严格锁死同一物理时间戳。

---

### ③ 循环隐状态冷启动预热（Warm-up Protocol）
* openpilot 的 RNN 依赖历史帧积累。场景刚开始的前 1 秒隐状态为空，第一帧规划距离会异常拉长至 85~90 米。
* **接入规范**：
  * 仿真开始后执行 **5 秒刹车保持预热（Warm-up）**；
  * 先喂入 25 个 context 步让网络隐状态收敛，随后再将轨迹接入底盘控制器。

---

### ④ 底层执行器对接（P7 Controller）
* **控制节拍**：openpilot 原生在 20 Hz 运行，每 4 个 step（即 5 Hz）向 P7 控制器投递一次轨迹切片；
* **控制器配置**：P7 采用 Pure Pursuit 纯追踪算法，限制正向加速度上限（$\le 2\text{ m/s}^2$），忠实执行轨迹跟踪，不人为添加平滑滤波混淆模型本身的规划质量。

---

### ⑤ 高层交互信号对接（Desire 变道脉冲）
* openpilot 支持通过高层 Desire 信号触发变道（如 `LaneChangeLeft`）；
* 在绕行避障场景（P6）中，当上游模式决策头判定需要借道时，在对应帧打入 Desire 上升沿脉冲，车辆即可在闭环动力学下完成大于 2.5 米的标准车道级绕障动作。

---

## 6. 核心工程落盘文件速查

* **模型抽取与特征定义**：`jevdrive/openpilot/`、`jevdrive/p5_openpilot.py`
* **坐标系变换与 CARLA 适配 Agent**：`scripts/b2d_zeroshot_agent.py`（核心函数 `plan_origin: rear`）
* **开环配对评测总表**：`research/results/openpilot-openloop/`
* **闭环执行层与 P7 控制器**：`todos/2026-09-23-tfv6-controller/controller-eval/P7.json`
* **闭环评测与 Alpamayo 原始读数**：`research/results/zeroshot-b2d/smoke-alpamayo.csv`

---

## 7. 闭环实测战报与横向对照（Bench2Drive 成绩、作废复盘与 NVIDIA Alpamayo 1.5 专测）

在摸清开环特性之后，模型是否能在真实物理反馈的 **CARLA Bench2Drive（220 条高难官方路线）** 闭环环境中存活，是决定技术路线成败的终极试金石。

### 7.1 NVIDIA 专用大模型 Alpamayo 1.5 闭环实测成绩

Alpamayo 1.5 是 NVIDIA 推出的 10B 级推理型自动驾驶视觉-语言-动作模型（Reasoning VLA）。我们在 Bench2Drive 上对其进行了零样本（Zero-shot）闭环实测：

#### ① 5 条官方基准路线 Smoke 实测数据
| 路线 ID | 场景特征 | 状态 (Status) | 驾驶得分 (DS) | 路线完成度 (RC) | 扣分与违规记录 (Infractions) |
|:---|:---|:---|:---|:---|:---|
| **1711** | 城市直行与常规交互 | **Completed** | **100.0** | 100% | 无擦碰、无违规 |
| **24211**| 复杂避障与多车汇入 | **Completed** | **100.0** | 100% | 无擦碰、无违规 |
| **3564** | 雨夜转弯与车辆跟行 | **Completed** | **60.0** | 100% | 路线完整走完，发生 1 次车辆轻微碰撞 (Vehicle collision) |
| **2373** | 信号灯路口左转 | **Failed** | 15.1 | 21.6% | 闯红灯 1 次，路口转弯轨迹偏离规划路线 |
| **2390** | 复杂立体匝道与汇入 | **Failed** | 28.7 | 28.7% | 路线偏离 (Route deviation) |

* **总体成绩**：**平均驾驶得分（DS）60.8，路线完成度（RC）70.1，成功率（SR）2/5（走完 3 条）**；
* **学术价值**：证明了 10B 级通用 VLA 模型完全具备在合成仿真图像中“直接看懂环境、闭环连续控车”的能力，无微调即可跑出 60+ DS。

#### ② 为什么之前的 220 条全量跑了 13 条被紧急叫停？
* 在 2026-09-24 批准全量 220 条测试后，程序跑至第 13 条路线时被团队主动全量 Kill 叫停；
* **核心原因（严防基线污染）**：当时底层的通用控制器正在经历重大版本重构，如果底层执行器自身存在转向超调或纵向迟滞，产出的闭环成绩将无法区分究竟是“大模型脑子不行”还是“底盘手脚失灵”。为避免产出带毒的伪数据，前 13 条数据被全部作废封存；
* **最新安排**：Alpamayo 1.5 已正式收录进 Night-Queue-3 的 **CL8** 队列，改用标准验收合格的 P7 执行层重新跑全量 220 条。

---

### 7.2 OpenPilot 闭环实测历程与即时决战（Night-Queue-3 Lane B）

#### ① 早期 Smoke 测得 DS 2.7 的重大冤案平反
* 在适配初期，openpilot Lebowski 曾在 5 条路线上全部在 10 米内冲出车道撞墙，测得 DS 2.7 的荒谬低分；
* 逐帧复盘直接揪出了本文第 5 节记录的**坐标系平移 Bug**（把车顶相机原点直接当底盘后轴，误触发 7.1 m/s 全油门弹射）；
* **该分数已被正式撤回作废**，证实其失败绝非“模型看不懂 CARLA”，而是工程适配严重失真。

#### ② CL0 前置等价性验收（2026-09-26 17:48 CST 正式过线）
在最新一轮（Night-Queue-3）闭环大考前，团队设置了严密的 **CL0 阻断验收**：
* 在 3 条验收路线上连续截取 1,524 次轨迹请求，离线与在线代码在同一张 GPU 上进行对比，**所有位姿与隐状态特征达到 100% 逐位一致（Bit-identical）**；
* **原汁原味的原生短板发现**：在完全剥离任何外部作弊脚本（如 Route Oracle 强制起步）的情况下，openpilot 的纯原生 Plan 在完全静止时缺乏起步动力先验，容易触发超时停滞；而加载了我们自研的 **`temporal` + P5 避险 Head（CL3 / CL4 / CL5）** 后，起步与跟车动作完全恢复顺畅。

#### ③ 正在实跑的 220 条闭环全量战役梯队（CL1 ~ CL10）
目前 CARLA 6 卡集群正在全速推进以下全量闭环评测：

```mermaid
flowchart TD
    CL1["CL1 专家回放天花板<br/>(P7 控制器基准损耗, DS 36~60)"] --> CL2["CL2 / CL7 openpilot 原生 Plan<br/>(Cinque / Lebowski 真实能力基线)"]
    CL2 --> CL3["CL3 Cinque temporal + ridge_late<br/>(连续线性残差头)"]
    CL3 --> CL4["CL4 Cinque ⊕ Qwen M-C 双流<br/>(核心重头戏：验证开环 46% 行人避险能否兑现为闭环零碰撞)"]
    CL4 --> CL5["CL5 / CL5d 障碍绕行 Head + Desire 变道脉冲<br/>(验证闭环自主借道与对向博弈)"]
    CL5 --> CL8["CL8 NVIDIA Alpamayo 1.5 全量 220 路线<br/>(在 P7 统一执行层下重新决出官方终局跑分)"]
    CL8 --> CL10["CL10 公开明星模型同台摸底<br/>(TFv6, BridgeDrive, BLUE, SimLingo)"]
```

每当一个梯队跑完 220 条路线，系统将自动汇总其 Driving Score、Success Rate 以及在突发障碍场景下的碰撞削减率，为整篇论文提供最具说服力的闭环铁证。

