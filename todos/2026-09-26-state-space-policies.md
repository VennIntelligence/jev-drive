# 公开 state-space 驾驶 RL policy：判断之后的行为（bypass / negotiation / recovery）

状态: running
主题: ../research/decisions.md 第 25、42、44 条；候选清单与证据见 [tmp/2026-09-26-state-space-policies.md](../tmp/2026-09-26-state-space-policies.md)；
量法沿用 [tmp/2026-09-26-behavior-layer-survey.md](../tmp/2026-09-26-behavior-layer-survey.md) §3.3 的反事实配对读法。

## 目标
「判断之后的行为」如果交给一个现成的 state-space policy（只吃 ego + 周围 agent + 道路的结构化状态，不吃图像），
它到底会不会绕开静止障碍、会不会因为对手而让行、偏出车道后会不会回来。这里只量行为，不涉及我们的感知。

## Setup
- **Policy**：BehaviorBench（arXiv 2605.10034，PufferDrive fork，AGPL-3.0）随仓库发布的两个权重：
  `simple_ppo.pt`（自博弈 PPO，下称 PPO）和 `conditioned_ppo.pt`（Gigaflow 式 reward conditioning，按仓库自带的 normal / caut / aggr 三组系数各跑一次）。
  对照 planner：IDM（跟车规则，不会变道，bypass 的地板）、PDM（仓库实现，横向只有 ±1 m 三条 proposal）、constant_velocity（什么都不看）。
- **仿真**：PufferDrive C 仿真，10 Hz，episode 91 步（9.1 s）。ego 由被考 policy 控制；主设置 traffic = expert（其余车辆 log replay，不反应），
  negotiation 另跑一版 traffic = PPO（其余车辆也由 PPO 控制，会反应）。goal = ego 的 log 终点（仓库默认）。
- **数据**：WOMD v1.3 `validation_interactive` 的 shard 0–9（box 上 `~/data/datasets/womd/validation_interactive/`），
  经 ScenarioMax 转 GPUDrive JSON、BehaviorBench 的 `extract_interactive_benchmark.py` 过滤成「两个 tracks_to_predict 都是车」并各当一次 ego。
  变体场景（删对手、插障碍、平移起点）在 JSON 层生成后用仓库的 `save_map_binary` 写 .bin，代码 `scripts/statepol_*.py`。
- **env**：box 上 `~/data/envs/statepol`（uv venv，py3.11，torch 2.9.1 cu128），代码 `~/data/third_party/statepol/behavior-bench`；run dir `~/data/runs/statepol/`。
- **预算**：数据转换 ≤ 30 min（CPU），评测 ≤ 1.5 h（一张卡 + 多进程）；超过估计 2 倍即停下写日志。

## 考卷（跑之前写死）

坐标：所有横向量都在 ego 起始时刻 t₀ 的 ego 坐标系里算，x 朝前、y 朝左；t 指 t₀ 之后的秒数。
（2026-09-26 跑任何数字之前的实现修正：原写 t₀ = 第 10 帧；实测 BehaviorBench 的 eval 以 `init_steps=0` 从 log 第 0 帧起步，rollout 共 90 步，所以 t₀ = 第 0 帧，其余读法不变。仿真坐标减过每个场景的 world mean，读数前按 ego 第 0 帧平移回 log 坐标；用 constant_velocity ego + expert partner 验过：partner 与 log 逐帧误差 0.0 m，ego 第 1 步落在 log 第 1 帧附近。）

### A. negotiation：删掉交互对手的反事实对
- 场景：每个过滤后的交互对，ego = 其中一辆，partner = 另一辆。最多 1000 个 ego episode（固定 seed 抽样）。
- 三臂：x⁺ 原场景；x⁻ 删掉 partner；x⁰（null）删掉一辆「与 ego log 路径全程距离 ≥ 30 m」的无关车，没有这样的车则该 episode 不进 null。
- 读数：Δprog(t) = ego 沿 x⁻ 路径的弧长进度（x⁺ − x⁻），t = 3、5 s；Δv(3 s)；Δ_lat(t) = y⁺(t) − y⁻(t)，t = 2、3 s，记号取「远离 partner」为正。
  「让行反应」= Δprog(5 s) < −τ_prog；「横向反应」= Δ_lat(3 s) > τ_lat。τ = 同一 policy 在 x⁰ 臂上 |Δ| 的 p95，下限 τ_prog ≥ 1.0 m、τ_lat ≥ 0.3 m。
- 通行顺序（只在路径相交的对上）：ego 的 x⁺ 轨迹与 partner 的 log 轨迹最近距离 < 3 m 时，取冲突点，比较 ego 与 partner 先后到达冲突点 3 m 内的时刻；
  报「sim 顺序与 log 顺序一致率」和「ego 后走率」。
- **判据**：该 policy「会博弈」当且仅当 (a) 让行反应率 ≥ 2 × x⁰ 上的同口径率且 ≥ 10%，并且 (b) 路径相交对上与 log 顺序一致率 ≥ 70%，
  并且 (c) x⁺ 的碰撞率 ≤ x⁻ 碰撞率 + 5 pp（对手在场时不因博弈失败多撞）。只满足 (a) 叫「会让，但顺序不像人」；(a) 不满足叫「对对手不反应」。

### B. bypass：在 ego 的 log 路径上插一辆静止车
- 场景：ego 在 t₀ 的速度 ≥ 5 m/s、log 在 8 s 内前进 ≥ 40 m、总航向变化 < 20°（直路）的 episode，最多 600 个。
- x⁺：在 ego log 路径弧长 s_obs = max(20 m, 2.5 s × v₀) 处放一辆 4.8 × 2.0 m 的静止车，朝向取该处 log 航向，全程 valid，log replay 中不动；
  log 轨迹在任意时刻经过该位置 5 m 内的其他车（通常是同车道前车）从 B 的两臂里一起删掉，超过 3 辆则丢弃该 episode（实现修正：原写「换 s_obs + 10 m」，在跑数前的小样本上发现前车几乎总会经过障碍位置，只剩 4/29 可用）。x⁻ = 删掉这些车、不放障碍的同一场景（variant `obsctrl`）。
- 每个 x⁺ episode 归一类（按顺序判）：**collide**（ego 碰撞）、**offroad**、**bypass**（ego 纵向越过障碍后缘 + 3 m，且在障碍前后 ±10 m 内相对 log 路径的最大横向偏移 ≥ 1.0 m）、
  **stop**（没越过，末速 < 0.5 m/s）、**other**（没越过也没停，如慢速蠕行到 episode 结束）。
- null：x⁻ 上同一窗口内最大横向偏移 ≥ 1.0 m 的比例（policy 自然晃出来的「假绕行」率）。
- **判据**：「会绕」= bypass 率 ≥ 30%，且 bypass 率 − x⁻ null 率 ≥ 20 pp，且 collide ≤ 20%。「只会停」= stop ≥ 50% 且 bypass < 10%。
  collide ≥ 30% 叫「不处理静止障碍」。其余情况如实报四类占比，不下结论。IDM 必须落在「只会停」或 collide，否则检查插障碍的实现。

### C. recovery：平移起点
- 场景：取 B 的直路 episode，把 ego 的 t ≤ t₀ 历史整体横向平移 ±1.5 m（左右各一半，按 episode 序号交替），速度与航向不动。
- 读数：t = 1、3、5 s 时 ego 到其 log 路径的横向距离 |d|；offroad 率；碰撞率。null = 不平移时同一量。
- **判据**：「会恢复」= |d(3 s)| < 0.5 m 的比例 ≥ 70%，且 offroad ≤ 5%，且碰撞率不高于不平移 + 5 pp。

### 有效性前置（不过就不报行为结论）
- 基线（x⁻、traffic = expert）上 PPO 的 goal 率 ≥ 80% 且碰撞率 ≤ 10%；否则视为「适配 / 数据转换有问题」，只报问题，不报行为。
  参照：论文 Interactive1k 上 PPO ego × PPO traffic 的 score 59.5，碰撞在换 traffic 后上升。
- 同一 episode 重复跑两次输出逐步一致（确定性检查）。

## 步骤
- [ ] env + 构建 + 权重 smoke（仓库 eval.py 在少量场景上跑通）
- [ ] 下载 10 个 shard、转 JSON、过滤、写 .bin
- [ ] 变体生成（A 三臂、B、C）
- [ ] 评测：6 个 ego planner × traffic expert；A 另跑 traffic = PPO
- [ ] 读数与判定，写结果

## 预计耗时
env + 构建 20 min；下载 5 min；转换 15 min；评测：每 episode 约 0.3–1 s，
(1000×3 + 600×2 + 600×2) ≈ 5.4k episode × 6 个 planner ≈ 32k episode，32 进程并行约 15–30 min；读数 30 min。

## 执行日志
- 10:40 评测超过预计 2 倍，停下重估：预计 15–30 min，按实测会到约 2 h。原因两条：(1) box 同时被别的任务压到 load average 250–310（容器 125 核），
  同样 1000 个 episode 从 103 s 变成 230–310 s；(2) 每个 episode 约 7 CPU·s，大头是 evaluator 每步两次 `get_state()` 全量拷贝实体轨迹，加每个 shard 的 import 与 env 探测。
  处置：次要臂缩小（traffic = PPO 只跑 ppo / idm 两个 ego；B 的扩样 run 只跑 5 个 ego planner），其余照预登记继续，总时限内可完成；不改判据。

- 11:34 box 11:45 重启加第 6 张卡，按协调要求在 11:34 干净停下（所有已完成 chunk 都是原子写的 pkl，重跑自动跳过）。

## 停机时的状态与恢复命令（2026-09-26 11:34 box 时间）
- 已完成（`~/data/runs/statepol/main/*.pkl`，40 个）：expert traffic 下 base / nopartner 全 7 个 ego planner；nullrm 的 ppo、cond_normal/caut/aggr、idm；
  obstacle / obsctrl / shift 全 7 个。部分读数在 `~/data/runs/statepol/main/readout_partial.txt` 和 `readout.json`。
- 未完成：nullrm 的 pdm、cv（只影响 A 的对照列）；traffic = PPO 的 base / nopartner / nullrm × {ppo, idm}；B 的扩样（`variants_big`，obstacle / obsctrl，最多 5000 episode）。
- 恢复（box 回来后先 `nvidia-smi`；本评测只用 CPU，不占卡）：
  ```
  cd ~/data/jev-drive && git pull && scripts/tmux_run.sh statepol-main bash -c "bash ~/data/runs/statepol/full3.sh 2>&1 | tee ~/data/runs/statepol/main_log.txt"
  # full3.sh = statepol_eval.py（expert，obstacle,obsctrl,shift,nullrm；已完成的跳过）→ readout → statepol_eval.py（traffic ppo，base,nopartner,nullrm × ppo,idm）→ readout
  scripts/tmux_run.sh statepol-big bash -c "bash ~/data/runs/statepol/big2.sh 2>&1 | tee ~/data/runs/statepol/big_log.txt"   # 等 main 的 ALLDONE 后跑 B 扩样
  ```
- **停机前发现的两个疑点，结论下之前必须先查**：(1) B 里 IDM 和 PDM 撞上插入的静止车的比例是 95%，这对跟车规则不合理，
  很可能插入的车对 IDM / PDM 的前车检测不可见，或插入方式有问题。按预登记，这属于「检查插障碍的实现」，B 的数字暂不作数。
  (2) PPO 在不平移的 base 上，1 s 时离 log 路径的横向距离中位数就有 1.6 m，其他 planner 只有 0.04–0.2 m，要先排除坐标或起步的适配问题。

## 结果
（跑完再填）
