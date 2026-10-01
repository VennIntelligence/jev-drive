# v4定位集成与运动日志验证

2026-09-23 JST。4份日志均逐字节复制，完整原路径、字节数及SHA256在[index.json](index.json)。未覆盖旧验证目录或原始日志。

| 日志 | 结果 |
|---|---|
| verification-v4-pose-integration.log | 11 tests OK |
| verification-v4-pose-full.log | 126 tests OK |
| verification-motion-capture.log | 10 tests，1 error；StubAgent缺_motion_log属性 |
| verification-motion-capture-corrected.log | 10 tests OK |

失败日志与修正日志共同保留，不把首次失败隐藏为最终通过。此处为软件验证证据，不能替代smoke-v4-recovery的官方完成记录或正式G4整组结果。
