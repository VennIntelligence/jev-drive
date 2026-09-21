# Large Map 的闭环成本，以及 220 条路线实测

状态: 进行中（2026-09-22 夜）。
目标文档: [docs/bench2drive-cost.md](../docs/bench2drive-cost.md)。
背景: [research/carla-efficiency.md](../research/carla-efficiency.md)。

## 为什么有这一轮

昨天的成本模型有两个洞，两个都只在小地图上量过：

1. **一个 tick 在 Large Map 上要多少钱。** `docs/bench2drive-cost.md` 里那个 71 ms/tick
   的优化配置（800×450、每 4 tick 渲染、Qwen3-VL-4B 真的在环）测的是 base package 的路线，
   全是 Town01–10HD。Town12 的 route 1711 跑出来是 288.2 s / 1283 tick，约 225 ms/tick，
   但那是**未优化**配置，两个数不可比。
2. **Large Map 上能并行开几个实例。** 4 个饱和、8 个优化后能跑，都是在 Town10HD 上量的。
   Large Map 要流式加载 tile、持有的 actor 多得多，天花板可能更低。

**220 条路线里 151 条是 Large Map**（Town12 104、Town13 47、Town11 和 Town15 各 7），
所以这两个洞直接决定"一轮评测到底几小时"。

用户后来加的范围：**不要再外推，直接把 220 条跑完，报实测 wall clock。**
于是顺序是：先定 Town12 上的并发上限（它决定用几个 worker），再用那个 worker 数跑满 220 条。
Task 1 的 per-town 成本从全量跑里免费掉出来——每条路线都有自己的 tick profile，
每条路线的 town 都知道——所以不另外测一遍。

## 方法

**只用 `scripts/b2d_run.py` 和未经修改的 Bench2Drive leaderboard。**
昨晚最贵的一个错误就是量了一个手写的 bench 脚本，然后把它自己的 bug 记到 CARLA 头上。
插桩和优化仍然是 `scripts/b2d_hooks.py` 里带开关的 monkeypatch，未优化路径保持可复现。

优化配置（和 `docs/bench2drive-cost.md` 里那 71.1 ms/tick 的一行逐字相同）：

```
--rig front3 --width 800 --height 450 --policy gpu --policy-socket /tmp/b2d-policy.sock \
--decimate 4 --overlap --no-spectator --zero-copy
```

policy server 一个进程供所有 worker：
`b2d_policy_server.py --backbone qwen --width 800 --n-images 3 --layers 18`，
每帧 1050 个 image token。

新增的只有 `scripts/b2d_report.py`：只读，把每次 attempt 的记录和路线 XML 的 town 拼起来，
按 town 报成本、可靠性（R6）和 server age 失败曲线（R7）。它不启动任何东西。

端口：`--server-index 40`，即 RPC 4000+50i、TM 10000+50i。
避开别的 session 正在用的 2000/8000/9000 段，
也顺手满足了 Bench2Drive README 那句「低于 10000 的端口不安全」。

## 负载条件（必须和数字一起报）

- **GPU 是共享的。** Waymo 特征抽取每 ~15 分钟跑一次 ~3.4 分钟的 burst，占空比约 23%，
  还要再持续大约一天。不能杀、不能停，所以它是本轮所有数字的背景负载的一部分。
  policy server 常驻约 9.6 GB 显存。
- **网络被一个下载占满**，到 23 号中午前后。和本轮无关，但一起记着。
- 所有数字都是**共享卡**口径，不能和 `docs/carla.md` 里那张 idle 卡的表合在一起看。

## 开跑之前写下的预测（CLAUDE.md 要求，且事后写的预测不值钱）

从已有的两组数拆出来的推理：

| 量 | Town10HD | Town12 | 比值 |
|---|--:|--:|--:|
| 未优化配置，非 sensor 的 per-tick 工作（`world_tick` + `tree`） | ~33 ms | 70.4 ms | 2.1× |
| 其中 `world_tick` | 7–9 ms | 33.8 ms | ~4× |
| 其中 scenario tree | 24–26 ms | 35.7 ms | 1.4× |
| 未优化配置的阻塞等相机 | 125–135 ms | 108.2 ms | 0.85× |
| 未优化配置的总 tick | 158–170 ms | 178.6 ms | ~1.1× |

**关键推理**：未优化配置下 Town12 只贵 1.1×，是因为四分之三的 tick 是在阻塞等相机帧，
而等相机的时间两张图一样——Large Map 多出来的 `world_tick` 和 tree 大部分藏在那个等待后面。
**优化配置把那个等待删掉了**（sensor wait 0.03 ms），于是藏在后面的东西全部露出来。
Town10HD 优化后 71.1 ms = 33 ms 非 sensor 工作 + 约 38 ms 落回 `world_tick` 的渲染残余；
Town12 的渲染残余应该一样（同样 3 相机、同样 800×450、同样 decimate 4），
所以预测 **Town12 优化后约 108 ms/tick，即小地图的 1.5×**。

220 条的预测（8 个 worker，若 Large Map 的并发上限确实还是 8）：

- 小地图 8 实例实测 54.5 ms/tick，Large Map 按 1.5× 取 81.8 ms/tick
- 每条路线 2615 tick（小地图实测均值，Town12 只有 route 1711 一个点是 1283，先不用）
- 每条路线 tick 之外的开销：Large Map 约 60 s（route 1711 的 288.2 s 减去 tick 时间），小地图约 25 s
- `151 × (2615×0.0818 + 60) + 69 × (2615×0.0545 + 25) = 52917 s`，除以 8 → **1.84 h**

加上重启、server 启动错峰和 Waymo 的 burst，**预测 2–3 小时**，
即文档里现在那个「约 1.1 小时」乐观了 2–3 倍。
如果 Large Map 的并发上限掉到 4，同样的算术给 **3.2 h**。

**预测区间：1.5–3.5 h，中心 2.5 h。** 超过 3 h 的部分就是 CLAUDE.md 说要先 profile 的那段，
而这一轮的 profiling pass 就是下面的并发梯子本身。

## 结果

（待填）
