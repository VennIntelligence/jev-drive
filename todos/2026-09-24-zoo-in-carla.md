# Zoo 进 CARLA（已并入 zeroshot-exam，不单独执行）

状态: dropped（2026-09-24，写完当天发现与进行中的工作重复）

本文原计划把 openpilot 和 Alpamayo 1.5 按各自训练时的传感器接进 CARLA，做开环 shadow 和闭环两种调用，
并在去掉 scenario 的路线上过一道「会不会正常开」的关卡。这些工作已由
[zeroshot-exam](2026-09-24-zeroshot-exam/bench2drive.md) 在做，而且做得更细：rig 按标定中位数复现、
导航只给模型在真车上能拿到的形式、两个模型走同一个固定控制器，并且已经在 smoke 之前冻结了预注册；
开环部分见同目录下的 [wod-e2e.md](2026-09-24-zeroshot-exam/wod-e2e.md) 和 NAVSIM 的 zero-shot 脚本。

**唯一没有被覆盖、需要保留的一项**是「无 scenario 路段」的测量：zeroshot-exam 报的是完整路线上的 DS/SR，
routine 驾驶（R 层）和突发反应（E 层）混在一起。这一项移到 [R 层测量](2026-09-24-r-layer-routine.md) 里做，
直接复用 zeroshot-exam 的 policy server、rig 和控制器，不再另写 wrapper。
