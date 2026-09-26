# P1：真实 log 的独立行人事件

CX 产出，待 main 复核。按已登记门槛，独立事件 **33 490 ≥ 2 000，判格为「开 P2」**。本项仅计数与写判格，未启动 P2。

## 数据与口径

GT（数据集真实标注）类别严格等于 `pedestrian`，使用框中心。走廊从当前 ego rear-axle（车辆后轴参考点）开始，沿 logged future（实际记录的未来轨迹）按弧长取前方 30 m，横向距离 ≤ 4 m；首尾端点外的投影排除，不跨 `sample_prev`（前一采样帧链接）断链。按 main 于 2026-09-26 18:27 CST 登记的边界决定，未来路径不足 30 m 时沿最后一段有效行进方向延长，完全没有有效线段时按当前 ego pose 朝向构造 30 m 直线。

同一 track token（行人持续身份标识）在连续帧中持续合格仅算一次；不再合格或采样断链后再次合格是新事件。城市取 `map_location`，速度取事件起始帧 `ego_dynamic_state[:2]` 的模。速度档为 `[0,2)`、`[2,5)`、`[5,10)`、`[10,∞)` m/s；原任务「>10」档依既有登记包含恰好 10 m/s。

数据根为 box 的 `$DATA_DIR/datasets/navsim/navsim_logs/`。开数前已写日志盘点：现有 OpenScene / nuPlan 两个 split，未找到另一个独立 nuPlan log 目录；不纳入合成场景。

| split | log 数 | 独立事件数 |
|:--|--:|--:|
| test | 147 | 1 608 |
| trainval | 1 310 | 31 882 |
| 合计 | 1 457 | 33 490 |

## 城市与起始速度

| 城市（map_location） | 0–2 m/s | 2–5 m/s | 5–10 m/s | ≥10 m/s | 合计 |
|:--|--:|--:|--:|--:|--:|
| sg-one-north | 118 | 134 | 144 | 6 | 402 |
| us-ma-boston | 866 | 287 | 397 | 22 | 1 572 |
| us-nv-las-vegas-strip | 20 596 | 4 674 | 3 870 | 1 953 | 31 093 |
| us-pa-pittsburgh-hazelwood | 160 | 77 | 168 | 18 | 423 |
| 合计 | 21 740 | 5 172 | 4 579 | 1 999 | 33 490 |

事件主要来自 Las Vegas，且主要落在起始速度低于 2 m/s 的档。这是数据分布读数，未作模型能力结论。

## 完整性与边界统计

| 项 | 数值 |
|:--|--:|
| 处理帧数 | 798 141 |
| GT 行人框总数（跨帧重复计） | 17 399 540 |
| 唯一 split / log / track 身份数 | 31 891 |
| 原始 future 有效但不足 30 m 的帧 | 58 903 |
| 上一项中有 GT 行人框的帧 | 54 939 |
| 原始 future 无有效线段的帧 | 1 465 |
| 上一项中有 GT 行人框的帧 | 1 280 |
| 计数墙钟 | 114.93 s |

边界统计描述延长前的原始路径；两类帧按 main 规则补足后正常参与计数。连续合格区间的事件数高于唯一身份数，因为同一行人离开走廊后再进入会开启新事件。

CSV 行数、唯一 event ID、城市 × 速度汇总、所有事件的纵向 / 横向几何边界均通过核对。抽取 test 的 `2021.05.25.14.16.10_veh-35_00083_00485.pkl`（44 个事件）和 trainval 的 `2021.05.12.19.36.12_veh-35_00416_00557.pkl`（64 个事件）复算，事件 ID、起始速度、投影弧长与横向距离全部一致。自检覆盖身后 / 超过 30 m 端点的排除、静止朝向补足、短路径方向补足、采样断链和同 track 连续去重 / 重入。

## 交付与复现

- 每个事件一行：[events.csv](events.csv)。汇总机器记录：[metrics.json](metrics.json)。
- 代码口径 commit：`a95a0e8`；运行时间为 2026-09-26 18:28:30–18:30:25 CST。
- box 运行日志与事件流：`$DATA_DIR/runs/nq4/cx/p1/{log.txt,events.jsonl}`；仅使用 CPU 196–197、2 worker，OMP / OpenBLAS / MKL 线程各 1。已核对退出码 0 并关闭自己的 `cx-p1` 完成窗口。
- 命令：`taskset -c 196-197 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 ~/data/envs/jevdrive/bin/python scripts/nq4_p1_count.py --logs-root "$DATA_DIR/datasets/navsim/navsim_logs" --output-dir "$DATA_DIR/runs/nq4/cx/p1" --workers 2`。
