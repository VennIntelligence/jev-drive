# 渲染故障根因诊断：交接状态（2026-09-29，随阶段刷新）

任务（main 下达）：找 WL 分叉生成里 130 / 2814 个渲染故障 run 的根因（117 过曝、13 黄昏路灯不亮），在 GPU 6 上复现，
给出共享 harness 里的修复（默认关，命名清楚），跑失败例 + 正常对照验证，然后在 WL todo 写中文小节 + before/after 表，
decisions.md 加短条目，给 main 3–5 行（原因、选项名与取值、前后失败率、正常帧是否变）。
规矩：只用 GPU 6，≤ 3 GPU·h，≤ 6 个 CARLA，>1 min 的跑在 tmux `jev`（scripts/tmux_run.sh），只按精确 PID 停进程，
只提交自己的文件，不做 Artifact。已用 GPU 时间约 0.6 GPU·h（10:04 起，GPU 6 独占）。

## 已查清的事实

- 原诊断的「故障」是相对**来源 run**（旧 host、挂 TFv6 影子）判的。按分叉组内 7 个分支彼此比，130 个里 91 个是「7 个分支一致、来源不同」，
  即来源 run 本身不同，不是分支渲染坏；分支自身偏离组中位数 > 10 的只有 30 个（`/tmp/rf_branch.csv`，box）。
- 两个独立的现象：
  1. **白天整幅泛光（bloom / lens flare 爆掉）**：三个相机在同一帧一起跳变（如 90003260 tick 1→5，90003010 tick 33→37），之后整条 run 一直如此。
     只在 Town12 / Town13（Large Map）出现，且**只在同一 server 上一条 route 是同一张图时**出现（同图前驱 16 / 1638，异图前驱 0 / 315；
     前驱是夜里 5 / 140，白天 11 / 1498）。旧 host 1 312 个 run 扫描 0 个跳变；新 host D2 378 个 0 个。
  2. **Town03 黄昏（sun = 0）变暗**：**新起的 server 上的第一条 route** 必现（E1 2 / 2 次、逐帧相同，亮度 19 vs 复用 server 67）；
     原数据里 4 个分支（173 hold、175 op_stop / brake_hard、176 op）和来源 172 / 174 / 176 都是 server_age_routes = 0。
     tick 1 两者都亮（57），tick 5 起新 server 的变暗到 6–8。**把 RouteLightsBehavior 的 update 置空（E4）新 server 也保持亮（67.5）**；
     置空 RouteWeatherBehavior 不影响。`set_day_night_cycle(True)` 放在 set_weather 前（B2D_LIGHTS_FIX=1，E3）无效。
     无 leaderboard 的探针（render_probe.py）在静止相机上开关 LightManager 的灯（504 → 41 盏）亮度不变，说明变暗的不是远处路灯本身。
- E0（复用 server，7 条 route × 3 次）21 个 run 全部逐帧一致，未复现任何故障。

## 代码改动（都已 push，box 已 pull）

- `scripts/b2d_hooks.py`：`B2D_CAM_ATTRS`（JSON，给每个 `sensor.camera.rgb` 加 blueprint 属性，包住 `carla.World.spawn_actor`，写 `cam_attrs.json`）；
  `B2D_LIGHTS_TRUTH=N`（每 N tick 用新 client 读 server 真实灯状态 → `lights_truth.jsonl`）；`B2D_LIGHTS_FIX=1`（测试用，无效）；
  `B2D_MUTE_BEHAVIOR=weather,lights`（诊断）。
- `scripts/b2d_route.py`：导出 `B2D_CARLA_PORT`。
- `scripts/render_diag.sh`（重跑 WL route，全程存帧，REPS 份并行，RECYCLE=1 = 每条 route 新 server）、`scripts/render_diag.py`（frames / summary / scan）、
  `scripts/render_probe.py`（无 leaderboard 的调用顺序回放）。
- 输出都在 box `$DATA_DIR/runs/wl/renderfix/<exp>/`：e0_stock、e1_fresh_night、e3_fresh_night_fix、e4_mute_weather、e4_mute_lights、
  e2_night_then_day（运行中）、probe*_*、scan_oldhost.parquet、scan_newhost_wl.parquet。

## 正在跑

- tmux `jev:rf-e2`：`EXP=e2_night_then_day`，5 条链 × 1 worker（server index 460–469，port 25000+），每条链按顺序跑
  Town13 夜 → Town13 昼、Town12 夜 → Town12 昼 各两对（复现「同图前驱」的白天泛光）。结束时写 `renderfix/e2_night_then_day/DONE`。
  runner PID 见 `renderfix/e2_night_then_day/pids.txt`。
- 其余窗口（rf-e0/e1/e3/e4/scan/probe1）都已结束。

## 下一步

1. 黄昏变暗：看 RouteLightsBehavior 在新 / 复用 server 上实际开关了哪些灯（client 缓存 vs server 真值、LightGroup 分组，尤其 Building），
   找出「新 server 上被关掉而复用 server 上没被关掉」的那批；修复候选：warm-up（harness 在第一条 route 前先 load 一次图）或让灯的开关按 server 真值做一次。
2. 白天泛光：等 E2；若复现，试 `B2D_CAM_ATTRS`（exposure_mode=manual / bloom、lens_flare 置 0）与「每次 route 前 load 另一张图」两类修复。
3. 修复验证：失败例 + 正常对照，量化正常帧变化；写 WL todo 小节、decisions.md、给 main 3–5 行。
