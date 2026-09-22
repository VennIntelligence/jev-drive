# paired-v1：接入故障发现过程，不是最终对照

2026-09-23。根代理检测到真实 TCP 公共 guard 对微小负的静止 speed 误触发 reverse_motion，主动 SIGINT 停止第五例。这是代理维护干预，不是用户取消。原始 attempt.json 仍保留 harness 的 `cancelled_by_user` 字符串，不改写原件；事实依据为下列 intervention.json。

四例完整、第五例中断、pi_common/1773未运行。主比较将使用修复后 paired-v2 的完整六例，不能拼接，也不能择优替换。v1 所有原始故障帧和数字均保留。

| v1 案例 | telemetry 帧 | reverse_motion 帧 | raw speed在(-.01,0)帧 |
|---|---:|---:|---:|
| 24211 native | 480 | 84 | 80 |
| 24211 PI | 394 | 0 | 0 |
| 1711 native | 524 | 0 | 0 |
| 1711 PI | 505 | 0 | 0 |
| 1773 native，中断 | 1307 | 461 | 458 |

首对原报告的 PI−native 全程速度 RMS −.8452 m/s 与 jerk p95 −6.6294 m/s³ 已受公共 guard 接入故障污染，不能解释为 PI 效果。tiny-negative 与 reverse 计数不同，因为真正反向仍在 reverse 分类，原始符号与理由完整保留。

3210个受控帧的 GPS/IMU/SPEED/三路RGB均与记录frame一致，truth frame均一致，0解析错误。每例第一帧neutral无预测/PID是初始化；其余捕获的原生PID调用均1。没有独立网络source-frame字段，因此只能报告同run_step的传感器帧与一次forward/PID证据，不能额外声称未记录的时间戳。异步bev不参与模型同步判定。

selected与native steer、selected与所选comparison分支均无超过1e-7的差异，保留每帧原差。selected与official tail的油门/刹车差可来自**有意的公共envelope变更**，不是浮点舍入。两臂相对旧官方更快也不能归给PI或harness。

- [完整发现分析与逐帧CSV](analysis-edition-002-v1-discovery/comparison.json)，附manifest及当次分析脚本字节。
- [新图版](figures-edition-003-v1-discovery/README.md)：PNG/PDF、2秒抽样原预测点CSV、两种物理原点假设、输入/脚本/输出hash。
- [首对原分析](analysis-edition-001/comparison.json)、原图版001与002均原样保留；003只修正布局并补充其他闭合案例。
- [9项分析测试](analysis-tests-v3/tests.log)，覆盖world-first jerk、非连续帧、5秒实际跨度、nested PID计数及传感器同步/official-tail差异区分。
- 原始根：`/data/runs/b2d/tcp-controller/paired-v1`。
- 干预记录：`/data/runs/b2d/tcp-controller/standstill-replay-v1/intervention.json`。
- 输入重放只说明新guard在相同输入上不再误制动，不是闭环恢复证明；闭环需看独立paired-v2。
