# 闭环基础设施验收：CARLA harness 性能、控制器验收、不明 SIGKILL

状态: running（2026-09-25 15:30 开始）
主题: ../research/trajectory-to-control.md、[docs/carla.md](../docs/carla.md)、[docs/bench2drive-cost.md](../docs/bench2drive-cost.md)、[docs/hugsim.md](../docs/hugsim.md)
子文档: [profiling](2026-09-25-closed-loop-infra-acceptance/profiling.md)、[B2D 控制器验收](2026-09-25-closed-loop-infra-acceptance/b2d-controllers.md)、
[HUGSIM 控制器验收](2026-09-25-closed-loop-infra-acceptance/hugsim-controllers.md)、[SIGKILL 取证](2026-09-25-closed-loop-infra-acceptance/sigkill.md)

## 为什么做

到目前为止 Bench2Drive（CARLA）和 HUGSIM 的闭环分数主要是被基础设施和控制器决定的，而不是被模型决定的：
Zoo PID 只执行了 Alpamayo plan 横向偏移的 4%，碰撞后把车顶在障碍物上
（[Alpamayo 闭环诊断](2026-09-24-zeroshot-exam/alpamayo-closed-loop-diagnosis.md) 第 7 节）；TCP partner 的出厂低速油门上限
让它最高 1.5 m/s；HUGSIM 官方控制器让 openpilot 20/20 原地打转；openpilot 适配曾让停着的车倒车；box 的 CPU 长期被
cgroup 限流在约 68/75 核，却没人量过一个 CARLA server 到底要多少核；还有两次查不出来源的 SIGKILL（不是 OOM）。
用户因此暂停了所有闭环模型考试，直到这套基础设施验收通过。**本任务不跑任何模型考试。**

## 要回答什么

1. **CARLA harness 的成本与最佳布局。** 我们的 `scripts/b2d_run.py` + CARLA 0.9.15（按我们的启动方式）每个 server 在
   sync 模式下吃多少核、RAM、VRAM，tick rate 多少；每张卡、整台 box 上加 server 时吞吐怎么变；瓶颈是渲染、CPU、
   traffic manager、Python agent 还是 I/O；在现在这台 box（5 × RTX PRO 6000 96 GB、cgroup 125 核、600 GB RAM）上
   worker 怎么摆最好。瓶颈在我们自己代码里的就优化（在子集上验证行为一致，记前后数字）。
2. **控制器验收。** 给我们用过的每个控制器喂一份已知是好的 plan（专家轨迹），看它能不能跟住、在预注册的小路线集上
   分数接近专家。逐个控制器给出 pass / fail、跟踪误差、DS（或 HD-Score）对专家参考。
3. **不明 SIGKILL 的来源**（openpilot policy server；2026-09-25 13:41 的 nuScenes 特征提取 `decision40-nusc-op` rc 137）。

## 资源

GPU 0、GPU 4 归本任务（schedule.md 2026-09-25 15:15 版），两卡合计 ≤ 70 核；CARLA server index 700–799
（控制器验收 700–739，profiling 740–789）。GPU 1 在 hugsim-scored-op 结束后可按借卡规则借用（≤ 4 h）。

## 结果

（跑完再填；每一部分的预注册、数据与细节在各自的子文档里，这里只放结论与总表。）
