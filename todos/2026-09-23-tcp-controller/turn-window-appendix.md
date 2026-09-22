# 阶段二转弯窗口：几何预定义附录

2026-09-23。阶段一protocol已冻结，本附录不改变其六例纵向对照，不证明阶段二已运行或横向已改善。几何helper与初始证据由独立agent产生：[lateral_windows.py](agents/lateral_windows.py)、[manifest](agents/lateral-evidence-v1/manifest.json)、[精确窗口CSV](agents/lateral-evidence-v1/turn-windows.csv)、[frame索引](agents/lateral-evidence-v1/frame-windows.csv)。

## 固定几何规则

以原始route_reference的world_xy折线弧长每.5m重采样；使用s±2.5m的5m弦求展开heading，κ=dheading/ds。|κ|≥.02/m的点合并相距≤3m的间隙，累计绝对heading≥15°保留；窗口前后各pad5m。CARLA世界xy的正heading变化为右转。正负曲率各累计≥15°标S；单向净角≥60°或peak|κ|≥.08/m标sharp。

这些是“明显曲率core”，并非所有弯道的完整分割。低于阈值不能自动叫直线：24240整条heading范围约89.9°，core累计只有49.2°；26405有46.2°宽缓弯但没有该阈值core。必须同时保留整条几何概况，必要的新宽缓弯定义只能在相关控制结果前另冻结。

| 已有几何候选 | core弧长约m | 已知形态 |
|---|---|---|
| 26966 | 29–46 | 约88.4°右急弯 |
| 17563 | 32.5–43.5；78.5–90 | 两段S，各累计绝对转角约68°、净约0° |
| 24240 | 23–59 | 明显左转core约49° |
| 25854 | 3–34.5 | 明显右转core约49° |

精确边界以CSV为准。24211/1711/1773三条本阶段TCP XML没有该阈值窗口，且独立端点弦偏离均≤.075m；不能承诺第一阶段完成道路弯道验证。实际模型可能为绕障而转向，与道路几何窗口是不同对象。

旧G2真值frame索引只描述其已记录的oracle-controller运行，例如pursuit max26966的8315–8384、17563的5399–5468与5552–5622。它们不是未来真实TCP的frame编号，也不能直接当本轮比较。新实跑按其独立弧长/时戳生成对应窗口，并保留进入、退出与整个路线。

## 使用边界

根代理在阶段二执行前确认少量路线与窗口，不自动增加当前六例或从多个结果里选最有利窗口。先在冻结窗口内检查预测曲率/目标点选择/steer饱和与执行滞后，再做完整路线回归。横向改动不能只靠全路线均值，也不能仅靠局部窗口漂亮就通过。

窗口的全部帧保留，包括低速、碰撞、故障和恢复；speed≥2m/s只作额外moving诊断，不是主统计过滤器。独立agent现有Dev10索引仅使用pursuit seed0 attempt1做几何/frame索引，不能当跨preset选择或胜负证据。
