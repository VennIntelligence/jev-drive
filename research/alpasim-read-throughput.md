# AlpaSim read 吞吐：常驻栈 / work stealing / renderer 副本（2026-10-18，perf lane）

接 [alpasim-read-profile.md](alpasim-read-profile.md)。目标是**每卡时完成的场景数**（分数逐位一致）。三个杠杆按顺序试，每个都有 15 % 的早停线。

**结论先行。** 没有一个杠杆过线。默认路径不变（`run.sh` / 各 lane 的 chunk 作业）。
- 杠杆 1+2（常驻栈 + work stealing）：可行，分数逐位一致；实测 +4 %（单 read）和 −11 %（两个 read，邻居更重的卡），都在 ±15 % 噪声内。按阶段算的上限：149 场景的 chunk 约 9 %，77 场景的 chunk 约 16 %。
- 杠杆 3（每栈多个 renderer 副本）：测了（见下“杠杆 3 实测”）。N=2 墙钟 −22 %，但 VRAM 峰值 +63 %，按卡份额 +6 %（不过线），且分数不逐位一致；N=3 更差。

代码留在仓库里，是可选路径，不是默认：`experiments/alpasim/scripts/read.sh` / `queue_read.py`。

## AlpaSim 允许什么（读源码，`0bb4c4b`）

- **runtime 有 daemon 模式**，随附发布：`wizard.run_mode=SERVER` → `alpasim_runtime.simulate --serve`，是一个长期运行的 gRPC 服务（`daemon/engine.py`、`scheduler.py`）。
  - 它能对已经在跑的 renderer / controller 反复接 `simulate` 请求，每个请求带一组场景。
  - 请求还可以带自己的 driver 地址（`available_drivers`）：换 checkpoint 只需重启 driver，renderer 和 runtime 不动。
  - config 里把 driver 设成 `runtime.endpoints.driver.skip=true`，runtime 启动时就不等 driver，所以 runtime 启动和 driver 预热能并行。
- **聚合**：AlpaSim 自己的 `run_aggregation_from_runtime` 只读 `<dir>/rollouts/**/metrics.parquet`。把完成的 rollout 目录移进各 read 的 out dir，再单独聚合一次即可。之后 out dir 和 `run.sh` 的布局一样。
- **随附 bug**：在 trajdata provider 上调用 `get_runtime_info` 会抛 AttributeError（`TrajdataSceneProvider` 没有 `scene_infos`）。所以就绪看端口，容量从 overrides 算。没有改 AlpaSim。
- **renderer 副本**：`services.renderer.replicas_per_container=N`。wizard 会在同一张卡上起 N 个 MTGS 进程，network config 里有 N 个 endpoint，runtime 的地址池在它们之间分配（只在 wizard 层验证过）。
- **场景缓存 / driver 状态**：renderer 的 FIFO 场景缓存和 runtime worker 的 trajdata dict 在 daemon 里跨请求保留；sh30 driver 的状态都是按 session 的。
  - **硬约束（实测）**：同一个场景的两个 session 同时跑在一个 MTGS renderer 上会互相破坏：渲染报 `torch.Size([8, N, 3])` 的 INTERNAL 错误，分数改变。
  - 所以 `queue_read.py` 在一个栈上从不并发同一场景，scene-major 排序（多个 read 的同一场景背靠背）不能用。

## 实测（pilot 1：`all_n10/c3`，149 场景，10-decision，`c0b_chain.OVERRIDES`，卡 1-4，priority 1，与 JTN lane 同卡）

卡时 = 作业墙钟 × 声明 VRAM 20 / 80 GB（池子 4 个/卡的记账口径）。

| 臂 | 内容 | GPU 作业 | rollout | 墙钟和 s | makespan s | 场景/卡时 | 相对 base | 分数 vs base |
|---|---|--:|--:|--:|--:|--:|--:|---|
| base | 现行 `run.sh`，RSG-L18-s0 和 RRED-L18-s0 各一个作业 | 2 | 298 | 1659 | 863 | 2587 | – | 与存档 `route_follow/guard/read10/RSG-L18-s0/b3` 149/149 逐位一致（score + 全部 metrics） |
| qs1 | 队列，1 个 slot，RSG | 1 | 149 | 796 | 796 | 2697 | +4 % | 149/149 逐位一致 |
| qr2 | 队列，2 个 slot，两个 read，read-major | 2 | 298 | 1859 | 930 | 2308 | −11 %（卡 2 当时 100 % util） | 298/298 逐位一致 |
| qs2 | 队列，2 个 slot，scene-major | 2 | 298 | 1525 | 773 | 2814 | +9 % | **不一致**：分数相同的只有 138/149 和 136/149，均值 −0.007，各有 1 个 renderer 失败；作废 |

各阶段（秒）：
- **base**：driver 就绪 40-41，服务就绪 21。runtime_s 767-784 里含约 31 s 的聚合。总时长 811-827，固定开销（总时长 − runtime_s）只有 44。
- **队列 slot**：runtime 就绪 24-29，第一个 rollout 完成在 66-79 s（driver 预热 41-64 s 是主要的等待）。最后一个 rollout 完成后 slot 立即退出，聚合放到 CPU 的 collect 作业里。

**为什么 1+2 过不了线。** 现行配置（2026-10-13 的启动重叠，4 并发）已经把每个作业的固定开销压到约 75 s（44 s 启动 + 31 s 聚合），只占 149 场景作业的 9 %。剩下的都是逐场景的成本，约 5.2 s/场景/slot，被单个 renderer 卡住，常驻栈动不了它。

**为什么 work stealing 也过不了线。** 尾巴只影响 makespan。按作业记卡时的话，提前结束的作业会把位置让给别人，所以尾巴不进卡时。只有池子整体没活的时候尾巴才是真浪费。

对 77 场景的 18-decision chunk（held 460 分 6 块），上限约 75 / (77 × 5.2 + 75) ≈ 16 %，未实测。

## 杠杆 3

wizard 输出验证过：2 个 MTGS 进程（端口 +0 / +1），network config 里有 2 个 renderer endpoint。三个臂（base 20 GB、2 副本 × 2 并发、2 副本 × 4 并发 + driver / controller 8）已提交，用户叫停后在 renderer 启动阶段取消，**没有数据**。

如果以后要测：同一列表，三臂同时跑，按实测 VRAM 峰值折算卡份额。理由是第二个 renderer 的 VRAM 会抵消一部分收益。

## 杠杆 3 实测（p4，2026-10-18，同一列表，三臂同时各占一张空卡，全部跑完）

`replicas_per_container=N`，renderer / driver / controller 的 n_concurrent_rollouts 都设为 4N（base 是 4）。base 声明 20 GB，r2 声明 34，r3 声明 48，卡 1 / 3 / 4。

| 臂 | 墙钟 s | 相对 base 速率 | VRAM 峰值 GB（池子实测） | 声明 GB | 场景/卡时（声明 base 20 / r2 24 / r3 36 GB 口径） | 分数 vs 存档 b3 |
|---|--:|--:|--:|--:|--:|---|
| base | 811 | – | 12.2 | 20 | 2647 | 149/149 逐位一致 |
| r2 | 635 | +28 % | 19.9 | 24 | 约 2800（+6 %） | 分数相同 141/149，metrics 相同 101/149，均值 0.953860 vs 0.953758，最大 |d| 0.011 |
| r3 | 723 | +12 % | 25.7 | 36 | 约 1800（−32 %） | 149/149 逐位一致 |

结论：过不了 +15 % 线。N=2 的墙钟收益被多出来的 VRAM（峰值 12.2 到 19.9 GB）基本抵消，按实测峰值折算反而更差；N=3 的墙钟比 N=2 还慢。r2 的分数漂移（r3 没有）说明多副本并发下的渲染时序会改分，不能无审计地用。默认不变。
注意：三臂不是同一张卡，邻居负载不同（单次，无重复），所以 ±15 % 内的差别不可信。原始数据 `$DATA_DIR/runs/perf/tput/p4/`，脚本是 p3.sh 的改版（每臂单独占卡）。

## 成本

约 0.5 卡时：smoke 2 次约 0.05，pilot 1 0.41，杠杆 3 取消前约 0.02。

有一次 smoke 绕过了 `env.sh`，gsplat 在 `~/.cache/torch_extensions` 重新 JIT 了 188 s，那个目录已删。`read.sh` 会 source `env.sh`，以后不会再出现。

原始数据：`$DATA_DIR/runs/perf/tput/{smoke,p1,p3}/`。脚本：`tmp/tput/{smoke.sh,p1.sh,p3.sh,ana.py,cmp.py}`（gitignored，Mac 和 box 都有）。

## 建议

默认不变。值得做的只剩两件，都要用户拍板：
1. 18-decision 的 77 场景 chunk 改走 `read.sh`（上限 16 %，未测）。
2. （杠杆 3 已测，不值得做。）
