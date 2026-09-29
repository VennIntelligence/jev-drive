# G 扰动崩塌（shift / swap）初步读数（2026-09-28）

> **已被取代**：最终读数见 [tmp/2026-09-29-g-final.md](2026-09-29-g-final.md)（`research/decisions.md` 第 58 条）。本文仅留作过程记录。

> **预览（PRELIMINARY），不是最终判格。** 读数 3（扰动崩塌）登记在只写方向：`shift` / `swap` 目前只跑了 1 个 seed，
> 登记的判格「CI 下界 > 10 pp」在 1 seed 下不下结论，只看方向。设计、口径与 2026-09-28 补充（**卡死算场景失败**）见
> [night-queue-4 的 G 节](../todos/2026-09-26-night-queue-4.md)。计算用的是登记代码 `jevdrive.nq4_g.collect` / `g_tables`
> 原样，没有改任何逻辑或判据——只是把喂给它的行过滤到下面这批已跑完的格子。

## 用了哪些数据

跑在 box 的 `~/data/runs/nq4/g/prelim-shift-swap-20260928/`（8 核以内，taskset 168–175），生成脚本原样带在
[research/results/nq4/g/prelim_shift_swap/g_prelim_shift_swap.py](../research/results/nq4/g/prelim_shift_swap/g_prelim_shift_swap.py)，
表在 [research/results/nq4/g/prelim_shift_swap/](../research/results/nq4/g/prelim_shift_swap/)。lane 自己的 `runs/nq4/gk/results/g/`
和 `g-lane/STATUS.md` 没有动。

已跑完、纳入本次计算的格子（`cells_included.csv`）：

| 考生 | shift.0 | swap.0 | ghost.0 | ghost.1 | orig.0 | orig.1 |
|:--|:--:|:--:|:--:|:--:|:--:|:--:|
| TFv6 | 80/80 | 50/50 | 80/80 | 80/80 | 80/80 | 80/80 |
| BridgeDrive | 80/80 | 50/50 | 80/80 | 80/80 | 80/80 | 80/80 |
| BLUE | 80/80 | — | 80/80 | 80/80 | 80/80 | 80/80 |
| SimLingo | — | — | 80/80 | — | 80/80 | — |

**没纳入、仍在跑的格子（按用户交代排除）**：`blue.swap.0`（25/50）、`simlingo.shift.0`（8/80）、`simlingo.swap.0`（10/50）；
另外 `ghost` / `orig` 的 seed 2 大部分还在跑（BridgeDrive 48/80、BLUE 与 SimLingo 未开），`shift` / `swap` 登记只排了 1 seed，
两者都不在本次范围内。PDM-Lite 没有单独纳入这次的拉取（登记的 shift / swap 已撤给 PDM-Lite，这次的候选只有榜单四家）；
读数 1 的 `position_memory` 门槛这里只用对照窗口比，PDM-Lite 基线沿用 [seed-0 初览文档](2026-09-28-g-seed0-prelim.md) 里的
3.6% [0, 10.7]（seeds 0–2 池化），没有重新算。

## 读数 3：扰动崩塌（主读数）

`shift` = 触发点顺路线 +15 m；`swap` = 同类换 actor（施工区/事故/停车障碍三者轮换，行人↔自行车，cut-in 换成 van）。
通过 = 官方记录的路线进度到达场景区终点、且区内无碰撞（卡死提前结束记「未到达」= 失败，2026-09-28 补充）。
配对差 = orig 通过率 − 扰动后通过率（同路线同 seed），路线整组 bootstrap 95% CI。

| 考生 | 扰动 | orig 通过率 | 扰动后通过率 | 配对差 [95% CI] | 路线数 | seed 数 | 方向（1 seed，不判格） |
|:--|:--|--:|--:|--:|--:|--:|:--|
| TFv6 | shift +15 m | 95.0% (76/80) | 92.5% (74/80) | +2.5 pp [−5.0, +10.0] | 80 | 1 | 略降 |
| BridgeDrive | shift +15 m | 97.5% (78/80) | 90.0% (72/80) | +7.5 pp [0.0, +15.0] | 80 | 1 | 降 |
| BLUE | shift +15 m | 87.5% (70/80) | 87.5% (70/80) | 0 pp [−8.75, +8.75] | 80 | 1 | 持平 |
| TFv6 | swap（同类换 actor） | 98.0% (49/50) | 96.0% (48/50) | +2.0 pp [−4.0, +8.0] | 50 | 1 | 略降 |
| BridgeDrive | swap（同类换 actor） | 98.0% (49/50) | 98.0% (49/50) | 0 pp [−6.0, +6.0] | 50 | 1 | 持平 |

登记代码算出的 `collapse`（CI 下界 > 10 pp）五行全是 `False`——这本身不是最终判格，因为 1 seed 时的规则是只写方向，
不是拿这一个布尔值当结论；但点估计本身最高也只有 BridgeDrive shift 的 7.5 pp，离 10 pp 的门槛还有一截，
所以即使换算成「只写方向」也看不出哪一格是奔着崩塌去的。`BLUE.swap` 与 `SimLingo` 的两个扰动都还没跑完，缺口见上表。

## 读数 1：幽灵反应率（ghost，seed 0–1 平均；仅供参考，非本次主读数）

| 考生 | 触发窗口幽灵率 [95% CI] | 路线数 | 对照窗口同一比率 | 对照路线数 | position_memory |
|:--|--:|--:|--:|--:|:--|
| TFv6 | 3.3% [0, 8.3] | 30 | 13.6% | 11 | False |
| BridgeDrive | 11.7% [3.3, 23.3] | 30 | 10.0% | 10 | False |
| BLUE | 10.3% [3.4, 20.7] | 29 | 10.0% | 10 | False |
| SimLingo（仅 seed 0） | 21.4% [7.1, 35.7] | 28 | 20.0% | 10 | False |

加了 seed 1 之后（TFv6/BridgeDrive/BLUE 从 1 seed 变 2 seed 平均）数字和 seed-0 初览文档里的很接近（TFv6 6.9%→3.3%、
BridgeDrive 10.3%→11.7%、BLUE 10.7%→10.3%），SimLingo 还是 seed 0 一个数没变。四家都远没到「对照率 + 10 pp」的门槛。

## 怎么看

这批新格子（shift +15 m、同类换 actor）在四个榜单模型上都没有测出扰动崩塌：五个「考生 × 扰动」配对里通过率掉得最多的是
BridgeDrive 在 shift 下的 7.5 pp，其余都在 0–2.5 pp 或持平，全部远低于登记的 10 pp 判格，CI 也基本跨 0（除 BridgeDrive
shift 的下界正好贴着 0）。这和幽灵反应率读数一致：加了 seed 1 之后幽灵率没有系统性升高，仍然远低于对照段 + 10 pp 的门槛。
两个读数合起来暂时看不出「记住了具体实例」的迹象——扰动之后模型该刹车、该躲还是会刹车、会躲。但样本还很薄：`shift` / `swap`
只有 1 seed，BLUE.swap 与 SimLingo 的两个扰动完全没跑完，登记的判格明确要求「1 seed 时只写方向」，这份预览到此为止，
不写进 `research/decisions.md`。
