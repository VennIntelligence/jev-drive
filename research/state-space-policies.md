# 公开权重的 state-space 多智能体驾驶 RL policy：清单、装机、第一版行为读数

2026-09-26。「state-space policy」指不吃图像、只吃结构化状态（ego 状态 + 周围 agent 的相对位姿 / 速度 + 道路 polyline）的驾驶 policy。
目的：判断之后的行为层（bypass 绕行、negotiation 让行 / 谁先走、recovery 恢复）能不能借一个现成的 state-space policy，再经 openpilot 的 desire 离散决策执行。
所以先回答三件事：谁放了权重、在我们的数据上能不能跑、它到底会不会绕和让。评测的预登记、执行日志与完整结果在 [todos/2026-09-26-state-space-policies.md](../todos/2026-09-26-state-space-policies.md)；第三层的量具综述见 [behavior-layer-instruments.md](behavior-layer-instruments.md)。

## 0. 结论先行

1. **公开且能下载权重的 state-space 多智能体 RL policy 很少**：能装能跑的只有 BehaviorBench（Bosch / Freiburg，PufferDrive fork）随仓库发布的 `simple_ppo.pt` 与 `conditioned_ppo.pt`（Gigaflow 式 reward conditioning）；
   GPUDrive 的 HF policy（MIT）有权重但要 Madrona 构建，HR-PPO 在已被取代的 Nocturne 上，CaRL 有 nuPlan / CARLA checkpoint 但吃特权 BEV raster、要 nuPlan 原始 `.db` 或 CARLA 闭环。
   Gigaflow 本身闭源，两个公开复现（2608.30819、CounterPlay）都没放权重；PufferDrive、V-Max、Waymax 只给框架不给权重。
2. **本轮在 WOMD interactive val 上给 BehaviorBench 两个权重做了预登记考试**（考卷见 [todos/2026-09-26-state-space-policies.md](../todos/2026-09-26-state-space-policies.md)）：
   negotiation 上 PPO 对交互对手有明确的让行（43%，null 2%）和横向避让（37%，null 1.4%），是 IDM 的 3 倍；
   bypass 上只有 conditioned policy 在激进系数下干净地绕（79.5%，null 20%，碰撞 16%），保守系数下 40–57% 撞上；PPO 过障碍 81%，但它平时就在车道里晃出 1 m 以上（null 82.5%），按判据记不成「绕」；
   recovery 上没有一个会回到原轨迹。
3. **作为 openpilot desire 之后的执行层，两个权重都不能直接用**：PPO 不居中（平均离车道中心 0.71 m）、不回线；conditioned 在保守系数下碰撞率过高，默认系数的有效性前置（goal ≥ 80%、碰撞 ≤ 10%）也没过。
4. **一个会坑所有 BehaviorBench / PufferDrive WOMD 评测的仿真行为**：终点就在起点的车（路边停车）在 eval 第 1 步被标成 removed，从 obs 和 IDM / PDM 里消失，但仍参与碰撞。我们在造场景时把这类车的 goal 推远后，IDM 撞静止车的比例从 95% 降到 3.5%。

## 1. 候选清单

「权重」一列只认能直接下载的 checkpoint；「数据」指它原生吃的场景格式。装机难度：低 = pip / uv 装完即跑；中 = 要编 C / CUDA 扩展或转数据；高 = 要大体量数据或模拟器集群。

| 候选 | 出处 | 权重 | License | 输入状态 | 原生数据 | 报过的行为 | 装机 |
|:--|:--|:--|:--|:--|:--|:--|:--|
| **BehaviorBench（Bosch / Freiburg，PufferDrive fork）** | arXiv 2605.10034；github.com/boschresearch/behavior-bench | **有**，随仓库：`weights/simple_ppo.pt`（2.5 MB）、`conditioned_ppo.pt`（Gigaflow 式 reward conditioning，10 维系数进 obs）、`SMART_epoch_030.pt` | AGPL-3.0 | ego ℝ⁷ + 最近 31 个 agent ℝ^{31×8} + 128 段可见道路 ℝ^{128×7}，共 1120 维；动作 7 档加速度 × 13 档转角离散，或连续 | WOMD（GPUDrive JSON → .bin）；另带 nuPlan devkit / interPlan planner 适配（`pufferlib/nuplan_integration`）与 CARLA Town01/02/10 地图 | Interactive1k（最交互的 1000 个 WOMD val 场景）× 8 种 traffic regime；论文主结论是 PPO 对「和自己同分布的 traffic」过拟合，换 IDM / SMART / log 当 traffic 碰撞率明显上升。未单报 bypass | 中（本次实测，见 §2） |
| **GPUDrive（NYU Emerge Lab）** | arXiv 2502.14706（Cornelisse et al., *Building reliable sim driving agents by scaling self-play*）；github.com/Emerge-Lab/gpudrive | **有**，HF `daphne-cornelisse/policy_S10_000_02_27`（10k 场景训，51k 参数）与 `policy_S1000_02_27` | MIT | ego + partner + road graph，局部视野 | WOMD（GPUDrive JSON，HF `EMERGE-lab/GPUDrive(_mini)`） | 10k held-out 场景 goal 99.8%，碰撞 + off-road < 0.8%；未报 bypass / yielding | 中偏高：Madrona C++/CUDA 构建；obs 布局与 PufferDrive 不同，权重不能直接互换 |
| **PufferDrive（Emerge Lab / PufferAI）** | github.com/Emerge-Lab/PufferDrive | **无公开 checkpoint**（README 只有 `--load-model-path <your-trained-policy>.pt`） | MIT | 同 BehaviorBench 的祖先布局 | WOMD、ScenarioMax 转的数据 | WOSAC realism、human-compatibility 评测入口 | 训练框架本身低；无权重即要自训（论文口径 10¹¹ step、88×L40S 约 20 h） |
| **SPICED / human-regularized self-play（Emerge Lab）** | arXiv 2606.19370（*Human-like autonomy emerges from self-play and a pinch of human data*）；github.com/Emerge-Lab/spiced_self_play | 仓库是 PufferDrive fork，**README 未见权重链接**（未逐目录核实） | MIT | 同 PufferDrive | WOMD | 项目页视频：negotiation、排队、等 gap、保守 merge；无定量 bypass | 同 PufferDrive |
| **HR-PPO（nocturne_lab）** | arXiv 2403.19648（RLC 2024）；github.com/Emerge-Lab/nocturne_lab | **有**，`models_trained/hr_rl` | 仓库带 LICENSE（Nocturne 系 MIT，未核实 fork） | Nocturne 的锥形视野 partner + road obs | Nocturne 格式 WOMD 子集（~2000 场景，Dropbox） | success 93%、off-road 3.5%、碰撞 3%；「交互场景里与人协同更好」的代理指标 | 高：Nocturne 已被 GPUDrive 取代，C++ 老构建 |
| **CaRL（Tübingen autonomousvision）** | arXiv 2504.17838；github.com/autonomousvision/CaRL | **有**：nuPlan `checkpoints/`（`nuplan_51892_1B` 等 5 个）；CARLA `results/`（2 个 PyTorch seed + Roach 基线） | Civil-M（MIT 变体，禁军用） | 特权 BEV raster（GT 状态画成语义图），不是 vector | nuPlan devkit（需 nuPlan `.db` 日志）/ CARLA 0.9.15 | CARLA Longest6 v2 DS 73（v1.1，与 PDM-Lite 持平）；nuPlan Val14 CLS-NR / CLS-R；未单报 bypass | 高：nuPlan 需要原始 `.db`（我们只有 OpenScene pickle），CARLA 需闭环（本任务禁止） |
| **V-Max（Valeo）** | arXiv 2503.08388；github.com/valeoai/V-Max | **README 未提供 checkpoint** | 仓库有 LICENSE，README 未写类型 | Waymax 观测 wrapper（vector / BEV 可选） | WOMD、nuPlan（ScenarioMax 转，HF 有 mini 版） | nuPlan 式指标套件；SAC / PPO 训练管线 | 训练框架中（JAX）；无权重 |
| **Waymax（Google）** | arXiv 2310.20199；github.com/waymo-research/waymax | **无 RL 权重**；只带 IDM / expert / constant-velocity actor | Waymax 自有非商用 license | Waymax state | WOMD | — | — |
| **ScenarioMax（Valeo）** | github.com/valeoai/ScenarioMax | 不是 policy，是数据转换器（WOMD / nuPlan / nuScenes → GPUDrive JSON / TFExample） | — | — | — | — | 低，本次用它转 WOMD |
| **MetaDrive / ScenarioNet** | arXiv 2109.12674 / 2306.12241 | MetaDrive 包内自带一个 PPO `expert()` 权重 | Apache-2.0 | lidar 式 259 维状态（非 agent 列表） | 程序生成地图为主；ScenarioNet 可回放 WOMD / nuPlan | 程序化路网上 success 高；非真实分布训练 | 低，但输入不是 agent-list，且训练分布离真实远 |
| **Gigaflow（Apple）** | arXiv 2502.03349 | **闭源** | — | 最近 63 个物体 + 200 个道路元素（100 m 内）+ reward 系数 | 自造城市地图 | bypass、creeping、zipper merge、死锁解除（论文视频与定性） | — |
| Gigaflow 复现：*What Emerges and What Breaks in Self-Play Driving* | arXiv 2608.30819（Sisask, Tampuu, Matiisen） | **未放**（只有演示页） | — | 同 Gigaflow 布局 | PufferDrive + 塔尔图真实 HD map | 路口主路让行 59%；人行横道让行 39% / 85%；红灯处借对向车道「绕」停止线（reward hacking）；CARLA DS 29–30 vs Gigaflow 93 | — |
| CounterPlay | arXiv 2609.21617 | 未放 | — | BehaviorBench 布局 | BehaviorBench | 在 Interactive / Random 两个 split × 8 regime 上 SOTA，解决 anchor 的大部分 timeout | — |
| Gigapixel（Mila，像素 self-play） | arXiv 2606.19641；github.com/montrealrobotics/gigapixel | 计划 2026-10-30 ~ 11-13 放，**现在没有** | Apache-2.0 | 像素（不在本清单范围；若同时放特权 teacher 再看） | nuPlan | NAVSIM navhard / HUGSIM 评测（计划） | — |

## 2. 在我们的数据上能不能跑

| 候选 | WOD-E2E val | navtrain / navtest（OpenScene） | CARLA P5 v1 记录 | 说明 |
|:--|:--|:--|:--|:--|
| BehaviorBench PPO | 否（E2E 没有 HD map 和他车 track） | 理论上可：`nuplan_integration/observation_builder.py` 从 nuPlan 的 ego / detection / map 造 1120 维 obs；需要把 OpenScene 帧包成 devkit 的 `PlannerInput`，本轮没做 | 可离线造 obs（有 GT actor + Town 地图，仓库带 Town01/02/10 JSON），但 obs 的道路段要从 CARLA map 重采；本轮没做 | **本轮用 WOMD motion val_interactive**（box 上的授权 gcloud 账号可读，10 个 shard 2.5 GB） |
| GPUDrive policy | 否 | 否（只吃 GPUDrive JSON） | 否 | 需 WOMD |
| CaRL nuPlan | 否 | 否：需要 nuPlan 原始 `.db`，我们只有 OpenScene pickle + map | CaRL CARLA 需闭环 | 本任务不跑 |
| HR-PPO | 否 | 否 | 否 | Nocturne 格式 |

### 装机实测（BehaviorBench，box，2026-09-26）

| 步骤 | 做法 | 耗时 | 坑 |
|:--|:--|:--|:--|
| env | `~/data/envs/statepol`（uv，py3.11，torch 2.9.1 cu128）；`uv pip install --no-deps -e .` + 手装依赖；`python setup.py build_ext --inplace` 编 C 仿真 | ~40 min（大半是 PyPI 走 turbo 下 torch 慢） | 仓库 `setup.py` 同时钉 `tensorflow==2.20.0` 与 `waymo-open-dataset-tf-2-11-0`（要 TF 2.11），**依赖不可解**，只能 `--no-deps`；SMART 需 `torch-cluster`（pyg wheel） |
| 转换 env | `~/data/envs/statepol-conv`（py3.10，ScenarioMax[womd]，TF 2.11.1）；lane connectivity 的 helper 要 `waymo_open_dataset.protos.scenario_pb2`，用 ScenarioMax 自带 proto 做 shim | 5 min | — |
| 数据 | 授权 gcloud 账号拉 WOMD v1.3 `validation_interactive` shard 0–9（2.5 GB）→ ScenarioMax 转 GPUDrive JSON（2706 个）→ 过滤「两个交互对象都是车」2080 个场景 → 4160 个 ego episode | 下载 3 min，转换 25 min（按文件并行，只能 10 路） | — |
| 仿真接口 | 仓库 `pufferlib/ocean/benchmark/eval.py` 的 Evaluator；ego = 被考 planner，traffic = expert（log replay）或 PPO | — | split 名只接受白名单（每个变体单独一个 data root，split 统一叫 `validation_interactive`）；一个 split 至少 100 张图；`map_file[100]` 定长 buffer；eval 从 **log 第 0 帧**起步（`init_steps=0`），仿真坐标减过 world mean；neural planner 每张图都新建一个探测几百张图的临时 env（7–8 s / 图，缓存后 ~0.3 s） |
| smoke | 8 张图 PPO × expert traffic 跑通，goal 7/8 | — | ego 当 `expert` planner 不支持（只有 traffic 能 replay）；eval 里终点即起点的车第 1 步变成「不可见但可碰撞」的幽灵，造场景时把 goal 推远（`scripts/statepol_build.py` 的 `deghost`） |


## 3. 第一版行为读数（BehaviorBench 两个权重）

数字全部来自 v2（ghost 修复后）；判据、null 与每一项的读法见 todo。traffic = expert（其余车辆 log replay，不反应）。

| 考卷 | PPO（`simple_ppo`） | conditioned（`conditioned_ppo`） | IDM / PDM 参照 |
|:--|:--|:--|:--|
| 有效性（x⁻：goal / 碰撞） | 86.5% / 6.2%，过 | normal 55.9% / 15.2%，不过 | IDM 40.4% / 9.0% |
| A 让行反应率（null） | **43.0%**（2.0%） | normal 14.0%（0.2%） | IDM 13.8%（0.0%） |
| A 横向反应率（null） | **37.0%**（1.4%） | normal 3.7%（0.0%） | IDM 0.9%（0.0%） |
| A 与 log 通行顺序一致 | 89.7%（n = 378） | normal 84.4% | IDM 87.2% |
| A 碰撞 有 / 无对手 | 11.7% / 6.2% | normal 24.6% / 15.2% | IDM 18.1% / 9.0% |
| B bypass / stop / collide | 81.3% / 2.9% / 10.5%（null 82.5%，无结论） | aggr **79.5%** / 2.9% / 15.8%（会绕）；normal 28.7 / 4.7 / 40.4；caut 3.5 / 7.0 / 56.7 | IDM 2.9 / 92.4 / 3.5；PDM 4.1 / 91.8 / 3.5（只会停） |
| C 平移 1.5 m 后 3 s 回到 0.5 m 内 | 11.7%（不平移也只有 15.7%） | normal 6.1%（不平移 65.0%） | IDM 37.9%（不平移 72.6%） |

- A 的判据 (c)「对手在场不多撞 5 pp」三者都没过，IDM 也没过，主要因为 log replay 的对手不会反应；at-fault 碰撞的差只有 2.5–2.6 pp。反应式对手（traffic = PPO）的一臂本轮没跑。
- B 在同一个 conditioned 权重上，绕不绕由 reward 系数决定；保守系数下它慢速蠕行后撞上，既不绕也不及时停。
- 样本：A 1000 对（WOMD v1.3 `validation_interactive` shard 0–9），B 171、C 343 个直路 episode。

## 4. 出处

- BehaviorBench：arXiv 2605.10034，https://github.com/boschresearch/behavior-bench （branch `main-behavior-bench`）
- GPUDrive self-play：arXiv 2502.14706，https://github.com/Emerge-Lab/gpudrive ，https://huggingface.co/daphne-cornelisse/policy_S10_000_02_27
- PufferDrive：https://github.com/Emerge-Lab/PufferDrive
- SPICED：arXiv 2606.19370，https://github.com/Emerge-Lab/spiced_self_play ，https://spiced-self-play.com
- HR-PPO：arXiv 2403.19648，https://github.com/Emerge-Lab/nocturne_lab
- CaRL：arXiv 2504.17838，https://github.com/autonomousvision/CaRL
- V-Max：arXiv 2503.08388，https://github.com/valeoai/V-Max ；ScenarioMax：https://github.com/valeoai/ScenarioMax
- Waymax：arXiv 2310.20199
- Gigaflow：arXiv 2502.03349；复现 arXiv 2608.30819；CounterPlay arXiv 2609.21617；Gigapixel arXiv 2606.19641
