# aim-g2-v1：服务器启动失败

CARLA server106未打开port7300，运行在首个route_start之前结束。0驾驶案例，无法生成六例CTE/控制/jerk/恢复结论，也不将此记为候选驾驶失败。start→end事件区间182.182474s计入基础设施成本；事件之后server.stop清理不在该区间内。

原始目录、终态异常与事件副本见status.json/events.jsonl。协议624a72…e6143、验收c94088…5fef3保持不变。后续根代理重试另建目录，不能覆盖此失败或把启动耗时忽略。本文不启动或更改CARLA。
