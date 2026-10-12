# 路线遵循：策略为什么不按导航走（诊断与方案稿）

2026-10-12 写（专家 A，「路线遵循」）。状态：**诊断 + 方案，没有训练，没有闭环 run。** 第 2 节的反事实读数在本节预登记提交之后才跑。
记号同 plan.md：d### 指 `research/decisions/###.md`；「测」= 有记录的读数；「推」= 由测得的数算出；「估」= 没有测量依据。盒上路径以 `$DATA_DIR`（`/root/autodl-tmp/ujs`）起。

前提（用户在 Tokyo box 上按分数从低到高看完全部 PAI 回放后的判断，本稿当作前提）：问题不在视野。凡是导航要求变道或在路口转 90°，车几乎都不做：
要么转早了撞路沿，要么直着走过去；它做成的 90° 转弯，都是路线本身一开始就已经在弯里的。它不按导航航点走。眼睛线（junction_eyes）停，d283 只作为发现保留。

## 0. 一句话

（读数回来后填）

## 1. 路线到底怎么进模型：端到端链条（读代码，带文件引用）

### 1.1 serve 侧：AlpaSim 发来什么，driver 喂给模型什么

| 环节 | 内容 | 出处 |
|:--|:--|:--|
| 模拟器发的 route | `RouteGeneratorMap`：rollout 开始时把 scene 自己的日志轨迹投到车道中心线，超出录制部分沿「朝向最接近的后继车道」延长；每个控制步从自车投影点起重采样 20 个点、间距 80 / 19 = 4.21 m，**弧长 < 40 m 的点丢掉**（`route_start_offset_m: 40`，nuPlan `e2e_challenge_nuplan_common/base.yaml`，PAI `e2e_challenge/dev.yaml`、`ec2.yaml` 同为 40），其余 NaN 补齐：实际是 40–80 m 处的约 10 个有效点 + 10 个 NaN，rig 系 | `third_party/alpasim/src/runtime/alpasim_runtime/route_generator.py`；docs/alpasim.md「What a nuPlan-track driver actually receives」 |
| driver 收到后 | `submit_route` → `command_from_route(req.route)`：取第一个离 rig 原点 ≥ 5 m 的点（实际就是 ≈ 40 m 弧长处那个），y > 2 m 记 LEFT、< −2 m 记 RIGHT、否则 STRAIGHT；没有点记 UNKNOWN。另外只存 `route0`（第一个点，写日志）。**其余 19 个点丢弃** | `experiments/alpasim/lib/sh30_driver.py` 219–226 行、`pai_driver.py` 177–181 行；规则本身 `e2e_challenge/sample_submission_simscale_navsim_transfuser/navsim_transfuser_challenge/navigation.py` |
| 进模型的张量 | `parity_adapter.ego_features`：20 维 ego 行 = [present, **cmd L, S, R**, vx, vy, ax, ay, 4 个历史位姿 × (x, y, yaw)]；UNKNOWN 记为三个 0。ego 行（去掉 present 的 19 维）经两层 MLP 变成 4 个 memory token，和其他 memory 一起被 32 个 slot query 的 2 层 decoder 读，输出一个 (32, 512) 的 bias **加到 9 个上下文帧的全部隐藏 token 上**，零初始化 | `lib/parity_adapter.py` 37–44、107–160 行；`experiments/alpasim/lib/sh30_core.py` 275–285 行、`pai_core.py` 145–160 行 |
| 频率 / 视界 | nuPlan：2 Hz，每个 `drive` 一次推理，route 每步一条新消息；PAI：10 Hz 推理（`PAI_EVERY=1`），route 每步更新。cmd 在两次 route 消息之间保持；会话开头第一条 route 到之前是 UNKNOWN（全 0） | `sh30_driver.py`、`pai_driver.py` |
| openpilot 自己的 desire 输入 | 不用（AlpaSim driver 只给 traffic convention，desire 为 0）。desire 在 openpilot 里只表示司机发起的变道，d92 / d121 已测它不是选择信号 | `docs/openpilot-interface.md` 第 24、27 行 |
| 输出被谁跟踪 | 8 个 0.5–4 s 的后轴位姿 → 10 Hz 轨迹；官方非线性 MPC 每 0.1 s 重解，**只对 1.0–2.0 s 处的位置与航向计代价**（`idx_start_penalty` = 10） | docs/alpasim.md「What the controller reads」 |
| ROUTE-MEM 支路（只在带 sidecar 的 checkpoint） | 全部 route 消息按自车位姿带到当前后轴系、合并成 −6…+72 m 的 40 点折线加掩码，进 adapter 的一个零初始化小 decoder | `experiments/route_mem/lib/rmem.py`；d282：被策略忽略 |

所以，**对现在 serve 的每一个 checkpoint（YS-L18、YS2G、YR10m10、APY10m10-AB、P2H10 家族），路线的全部信息 = 3 个数的 one-hot，由 40 m 外那一个点相对自车的横向符号决定。**
由这条规则直接推出的几件事（推，几何算术，不需要读数）：

1. **「哪里转」不在输入里。** 90° 左转（半径约 10 m）在转弯入口还在前方 D 处时，40 m 弧长处的点横向约为 max(0, 40 − D − 约 8) m，所以 D ≲ 30 m 时 cmd 变 L，
   入口在 30 m 处与在 3 m 处的 cmd 一模一样。d268 测到指令在日志转到一半前约 4.6 s 亮起（中位）。
2. **变道、路口转弯、普通弯道是同一个 token。** 一次 3.5 m 的变道只要在 40 m 弧长内做完就记 L；一条半径 < 400 m 的普通弯道在 40 m 处的横向 s² / 2R 就已 > 2 m，也记 L。
   d268：干净直行 rollout 任一 4 决策窗里出现非直行指令的基率 0.41–0.53。L 大多数时候只意味着「前面路往左弯」，画面本来就看得见。
3. **转弯开始后 cmd 很快回到 S。** 车头转过一半后，40 m 外的点已在出口直道上、接近车头方向，cmd 回 S；所以「转完这个弯」的后半段没有任何路线信号。
4. **偏移的路线（车已离开路线中心）不改变 cmd**，除非 40 m 处的点横向越过 2 m。把路线平移 1 m、换成另一条同向车道，对这些模型完全不可见。

### 1.2 train 侧：训练行里同一个位置放的是什么

| 行 | cmd 从哪来 | 有没有模仿目标 | 出处 |
|:--|:--|:--|:--|
| NAVSIM navtrain 行（P2H10、YR10m10、YS 的模仿部分） | OpenScene / NAVSIM 的 `driving_command`（当前帧，`index[...]["cmd"][-1]`）；**它的生成规则我们没有在代码里核到**（OpenScene 转换脚本不在盒上），只有经验对照：与 AlpaSim route 规则在 navtest 一致 69.8 %、navtrain 约 73 %，NAVSIM 直行的 8 025 个 navtest token 里 3 394 个被 route 规则判成左 / 右（d189、`experiments/alpasim/results/ap2_smoke.md`）；与按地图判的实际支路一致 78 %（d093） | 有：日志 4 s 未来 | `experiments/op_parity/scripts/pp_prep.py` 268–269 行 |
| AB 族 navtrain 行（APY10m10-AB、AP2H10-AB） | AlpaSim 自己的 `RouteGeneratorMap` 在 NAVSIM 日志轨迹上重建 route，再套同一 4 向规则（与真实 AlpaSim route 480 / 480 条一致） | 有 | `experiments/alpasim/scripts/ap2_route.py`，d189 |
| on-policy 行（YS / YS2G 的 S 课、onpol_long） | driver 当时喂的 ego 行原样（即 serve 侧的 cmd） | **没有**：只带 S 的两个 hinge（扫掠足迹对车、路面栅格），不含任何跟随路线的项 | `experiments/onpol_nuplan/scripts/on_rows.py` 357–463 行；d264 |
| ROUTE-MEM 行 | RouteGenerator 重建的记忆；NAVSIM 行按「跟日志走的 18 决策 rollout 第 k 步」造 | NAVSIM 行有；on-policy 行没有 | d282 |

### 1.3 train / serve 是否一致

- **定义不一致（测）。** serve 时的 cmd 是「40 m 外的路线点横向 > 2 m」；YS / YR / P2H10 训练时的 cmd 是 NAVSIM `driving_command`，两者 navtest 上只有 69.8 % 一致，
  不一致的主体是 route 规则判左 / 右而 NAVSIM 判直（d185：8 个不一致里 6 个）。模型在训练中学到的「L」语义与 serve 时「L」出现的位置不同。
- **AB 族把定义对齐了，私有榜反而 −0.16（d260）**：唯一差别是 AlpaSim 输入标准（route command + 冷启动），公开 +0.010、私有 −0.16。所以「把 cmd 定义对齐」本身不是修法；
  这一点也说明 cmd 对 plan 的作用方向在两族之间真的不同（若 cmd 完全无作用，两族在这一项上不应有差）——是否如此由第 2 节的跨 checkpoint 读数回答。
- **信息量一致地少。** 两种定义都是一个 3 值符号，都没有距离、没有几何、没有车道级（变道）信息。不一致的是语义，一致的是贫乏。
- **对齐了的部分**：ego 状态、历史、相机、冷启动（d189 / d263），与路线无关。

第 2 节的 nav 读数补一个训练侧的量：NAVSIM 训练行里 cmd 为 L / R 的行，有多少在 4 s 目标窗内已经能看到这个动作（转 ≥ 20° 或横移 ≥ 2 m）。若大部分都能看到，
则在模仿损失下 cmd 对图像 + 历史是冗余的，学不到「只有 cmd 才知道的事」（第 3 节的论证）。

## 2. 路线反事实诊断（serve 的模型，不训练）

### 2.1 预登记（2026-10-12，本节在任何读数之前提交；脚本 `experiments/route_follow/scripts/rf_cf.py`）

**要分开的三种情况**：(U) 路线不被使用；(W) 被使用但太弱；(L) 被使用但太晚。

**做法。** 因为路线只经 3 个数进入模型（1.1），对这些 checkpoint「换路线」= 换 cmd：在已缓存的 served 视觉 token 上（`$DATA_DIR/runs/road_probe/tok/{held,zero}`，d275 的
held 重跑带帧 dump，G-tok 已验 8 129 / 8 129 个决策的 plan 复现 ≤ 0.10 m），保持 token 与其余 ego 不变，把 cmd 换成 5 个值：served（原样）、L、S、R、none（全 0），
各前向一次，输出 8 个后轴位姿（0.5–4 s，y 左正）。「另一支」= 与路线方向相反的 cmd，「移除」= S 或 none，「平移」对这些模型按 1.1 第 4 条不可见（不读）。
标签 = `labels/{held,zero}.parquet`（d275 同一份：日志路径 40 m 内转角 `turn40`、方向 `turn_sign`、沿日志路径到路口的距离 `jdist`、10 / 20 / 40 m 处日志路径的横向 `ylog`）。
held 的状态来自 YS-L18-s1 自己的 18 决策 rollout（452 scene），日志路径 = route 的近处（route 就是投到车道中心的日志轨迹）。

**checkpoint**：主读 `YS-L18-s1`（这些 token 的 served 模型）；描述行 `YS2G-L18-s1`、`YR10m10-F-s0`（NAVSIM cmd 训练，私有提交 #2）、`APY10m10-AB-s0`（AlpaSim route cmd 训练，提交 #1）、
`P2H10-F-s0`（PAI serve 的族）；`P0`（出厂，无 adapter）作空对照。zero 组（98 个出路零分 rollout 的全部决策）只用各行自己的 served 模型（YS-L18-s1 / YR10m10-F-s0）。

**决策分桶**（held，8 129 个决策）：
- JT 路口转弯：`turn40` ≥ 45° 且 `jdist` ≤ 40 m；再按阶段分 JT_far（20–40 m）、JT_near（0–20 m，不含 0）、JT_in（已在路口内，`jdist` = 0）；JT_cmd_on = served cmd 与转向同侧。
- LC 变道代理：`turn40` < 15° 且 |`ylog40`| ≥ 2.5 m（40 m 内横移 ≥ 2.5 m、航向几乎不变）；LC_cmd_on 同上。
- BEND 无路口弯：`turn40` ≥ 20° 且 40 m 内无路口（参照：画面本身就看得见的弯）。
- ST 直行：`turn40` < 10° 且 |`ylog40`| < 1 m；再分 served cmd 为 S / 为 L 或 R。

**量**（每个决策，sgn = 路线所在一侧，+ = 朝路线那边）：
- swing@t = sgn ·（朝路线一侧的 cmd 下 plan 的 y(t) − 反向 cmd 下的 y(t)），t = 1.0 / 1.5 / 2.0 / 4.0 s。**主量 = swing@2.0s**（MPC 计代价的视界末端）。
- towardS@t = 朝路线 cmd 减 S；servedS@t = served 减 S；servedNone@t = served 减 none。
- 在固定距离 d = 10 / 20 m（相机前方横线，与 d275 同口径）：need@d = sgn · ylog_d（路线自己在那里的横向），swing@d、towardS@d、planS@d（cmd = S 时 plan 的横向，
  即「只靠图像 + 历史」）、planServed@d、errServed@d（+ = 越过路线一侧，− = 没够到）。
- ST 上另报 |y_L − y_R|@t（直路上 cmd 的总摆幅）。
- 统计：中位数，按 log 聚类 bootstrap 的 95 % CI（B 2000）。

**闸**：G-rep：YS-L18-s1 在 held、各自模型在 zero 上 served 变体复现存档 plan，≥ 95 % 的决策 8 个位姿最大差 ≤ 0.10 m；G-null：P0 五个变体之间最大差 < 1e-3 m；
G-n：held 的 JT_cmd_on ≥ 200 个决策（不足则只作描述）。任一闸不过，读数只作描述，不下判词。

**判词线（主读 YS-L18-s1，held，JT_cmd_on）**：
- **(U) 路线不被使用**：swing@2.0s 中位 < 0.10 m 且 CI 上界 < 0.20 m。（0.10 m 的依据：闭环跟踪偏差 p95 2–4 cm（d220），走廊半宽 2 m，90° 转弯在 2 s 内需要的横移是米级。）
- **(W) 被使用但太弱**：swing@2.0s 中位 ≥ 0.10 m，且 towardS@20m 中位 / need@20m 中位 < 0.25（cmd 供给的横移不到路线所需的四分之一）。
- **部分**：该比值 0.25–0.5；**够用**：≥ 0.5。
- **(L) 太晚**：JT_near 的 swing@2.0s 中位 < 0.10 m 而 JT_in 的 ≥ 2 倍于 JT_near，且 JT_near 的 errServed@20m 中位 < −1.0 m（够不到路线）。
- **路口与直路分不开**：JT_cmd_on 的 swing@2.0s CI 与 ST 的 |y_L − y_R|@2.0s CI 重叠。
- 变道（LC_cmd_on）用同一组线，n < 50 时只描述。
- 「谁在转弯」（描述）：JT 上 planS@20m / need@20m 的中位比；若 ≥ 0.8，转弯来自图像 + 历史，不来自路线。
- 跨 checkpoint 只描述；特别报 APY（route cmd 训练）与 YR10m10（NAVSIM cmd 训练）的 swing@2.0s 之差。
- 训练侧（nav，描述）：navtrain 12 个 shard 与 navtest 上，按日志 4 s 未来分桶（直行 / 横移 / 转 20–45° / ≥ 45°）的 NAVSIM cmd 分布与 route 规则 cmd 分布；
  L / R 行里 4 s 目标窗已显示该动作的比例；L / R 行落在 4 s 直行未来上的比例。

**预算**：一张卡，6 个 checkpoint × 2 组 × 5 变体的前向（约 10 万次 policy 前向）估 < 30 min；nav 与 read 为 CPU，估 < 20 min。全部经 GPU pool。

### 2.2 读数

（待填）

### 2.3 PAI 回放逐条（Tokyo box，低分路口 scene）

（待填）

## 3. 为什么会忽略路线，以及怎样让路线变得不可缺

（待填）

## 4. 没能核实的事

（待填）
