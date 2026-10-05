# Junction turns with the action head alone, at the open-loop-aligned camera (and 1.22 m as a side row)

2026-10-05. Question: with the B2D camera moved from the legacy 1.433 m, does the action head alone (zones off, `"zones": false, "div_m": 1e9`, turn desire on) take the
junction turns of decisions 121 (choice) and 122 (forced)? One run per cell, seed 2, shipped Cinque, no `nored`. Lane [`experiments/op_closed_loop/scripts/junction_rig122_lane.py`](https://github.com/VennIntelligence/jev-drive/blob/1dc08a0aae9ce639995e644d8a90f27da1ae6d7c/experiments/op_closed_loop/scripts/junction_rig122_lane.py)
(runs `$DATA_DIR/runs/rig122`), report `scripts/junction_rig122_report.py`, full tables [junction_rig122_tables.md](junction_rig122_tables.md), per-turn
[junction_rig122_per_turn.csv](junction_rig122_per_turn.csv), every collision [junction_rig122_collisions.csv](junction_rig122_collisions.csv). Turn metrics are
`junction_cl_report.py`'s (took branch = came within 5 m of the dense exit; leaves lane = peak cross-track > 1.75 m).

**Plan change during the run.** The first runs used the 1.22 m bumper-line camera (decision 125). The user then ruled that the B2D camera must match the open-loop boards'
viewpoint instead of being special-cased. `spec` is now the open-loop-aligned camera **(x 1.59 m, y 0, z 1.86 m)**, the mean of the real extrinsics: NAVSIM CAM_F0 x 1.665,
z 1.862 m above the road (40 frames; ego origin 0.35 m above road + 1.512), WOD front x 1.519, z ~1.86 m (1984 segments; 1.8065 above the ground origin); both level after
openpilot's calibration (real pitch -1.3 / -0.2 deg). `interface.B2D_MOUNTS` holds it plus `bumper122` and `windshield143` as named presets (`spec_bumper122`,
`spec_windshield143`); docs/openpilot-interface.md and its test are updated. The 1.22 m runs that had finished stay as a side row (the last two shards were not run).

Arms: A = shipped `drive`, zones on, 1.433 m; D = `drive`, zones off, 1.433 m (decisions 121 / 122); **OL / OLnz** = `spec` at the open-loop camera, zones on / off;
side row 143nz = `spec` zones off 1.433 m (height-only control), 122 / 122nz = `spec` at (3.8, 0, 1.22). Turns: 25 on 20 val routes (13 choice, 12 forced, all
already in the 1.433 m per-turn CSVs; R_min 4.7-42 m, left and right, T-stems and curves). Choice turns on the 13 routes are a subset of the 38 of decision 121; the 12 forced
are all the forced turns that were run at 1.433 m.

## Result: the open-loop camera does not make the head turn

| turns took the intended branch | choice (13) | forced (12) | choice leaves lane | forced leaves lane |
|---|---|---|---|---|
| A  zones on, 1.433 m | 11 (85%) | 9 (75%) | 8% | 0% |
| D  zones off, 1.433 m | 2 (15%) | 4 (33%) | 91% | 70% |
| **OLnz  zones off, open-loop camera** | **0 (0%)** | **1 (8%)** | 100% | 78% |
| OL  zones on, open-loop camera | 12 (92%) | 8 (67%) | 8% | 0% |

- OLnz minus D: choice -2 turns (both D successes lost, none gained), forced -3 / +0 (rate -25 pp [-56, 0], route-cluster CI), i.e. **no better than 1.433 m, nominally worse**.
  The height-only control (143nz, `spec` op-path at 1.433 m) is 0/10 choice and 2/9 forced on its turns; OLnz is 0/10 and 1/9 on the same turns. At 1.433 m and at 1.86 m the head
  drives straight through the junction; the head's median peak desired curvature is 0.30 x the needed 1/R_min (choice) and 0.75 x (forced; the wide curves carry that median).
- By radius (took branch / turns; median head peak / needed): choice R < 10 m A 6/8, D 2/8, OLnz 0/8 (0.29), OL 8/8; 10-20 m A 4/4, D 0/4, OLnz 0/4 (0.47), OL 4/4. Forced R < 10 m
  3/3, 0/3, 0/3 (0.74), 3/3; 10-20 m 3/4, 1/4, 0/4 (0.29), 3/4; > 20 m 3/5, 3/5, 1/5 (4.15, wobble), 2/5. The wide forced curves (R > 20 m) are the only place the head alone ever
  followed, and OLnz is no better there (24758 -79 deg and 26153 +37 deg are lost; 24758 also hits the stopped car ahead 4 times, in every zones-off OL-camera run it is stuck behind it).
- With the zones on, the open-loop camera is as good as the shipped route-driven arm: OL - A choice +1 turn (8 pp [-15, 31]), forced -1 (-8 pp [-27, 0]); DS 69.6 [58.4, 80.5] vs A 67.7
  [54.2, 81.0] on the 20 routes (1.22 m `spec`: 70.6), RC 92.1 vs 86.8, completed 16 vs 15. So the camera move costs nothing with the dense route steering and gains nothing without it.
  Without zones DS is OLnz 26.3 / D 41.6 / 143nz 29.9 (16 routes) / 122nz 32.6 (16 routes).

## 1.22 m side row (decision 125's question, on more turns)

Zones off, turns where all of D / OLnz / 143nz / 122nz / 122 exist (choice 10, forced 9): took branch 122nz **5/10 choice, 3/9 forced** vs 143nz 0/10, 2/9 and OLnz 0/10, 1/9.
Paired 122nz - 143nz: choice +50 pp [+20, +80] (5 turns gained, none lost), forced +11 [0, 38]. So **1.22 m is the only height of the three at which the head alone gets onto the
exit branch at all**, direction-consistent with decision 125 (5/7 vs 0/7) on 19 independent turns, and the height trend is monotone-with-a-cliff, not a straight
line (1.433 and 1.86 m behave alike).
But taking the branch is not turning cleanly: 122nz leaves the lane in 90% of the entered choice turns (median peak cross-track 7.2 m), and its head asks for **5.7 x the needed
curvature** (median head peak / needed over the choice turns; per-turn 1.0-1.8 1/m against 0.07-0.17 needed). The car swings through the junction in a wobble that happens to reach the
exit point. Per-turn: 27297 and 10255 pass through with 2.1 / 1.9 m peaks, 34183 and 6999 with 6.7 / 7.6 m; zones on at 1.22 m (122) takes 9/10 and 9/9 with 0.4 / 0.3 m median peaks.
The 1.22 m head is therefore too jumpy to count as a lane-keeper through a junction, and the 1.433 / 1.86 m head is too passive: the head's curvature scales with camera height
(decision 108's scale law), and neither end gives a usable turn without the route.

## The collisions of decision 125 (10 at 1.22 m vs 3 at 1.433 m)

On the same 6 routes (28180, 24944, 27297, 9196, 6999, 34183), hits by class (ego distance to the dense centreline at the contact; 1.75 m = lane edge):

| arm | hits | vehicle, ego in lane | vehicle, ego off lane | static / layout (pole, fence, building, vegetation, traffic light) | ego v < 0.5 m/s at the hit |
|---|---|---|---|---|---|
| 122nz (decision 125's 10) | 10 | 2 | 2 | **6** | 2 |
| 143nz (its 3) | 3 | 1 | 2 | 0 | 1 |
| OLnz | 5 | 1 | 2 | 2 | 3 |
| D | 6 | 3 | 0 | 3 | 2 |
| 122 / OL / A (zones on) | 4 / 2 / 2 | 3 / 2 / 2 | 0 | 1 / 0 / 0 | 1 / 0 / 1 |

The extra seven of 122nz are **off-road after the over-steered turn, not yielding failures**: 6 of 10 are static objects (traffic lights 28180 and 27297, fences 34183, building 9196)
that the swinging car reaches after leaving the lane (27297's traffic light is logged twice at one instant), plus two vehicle contacts with the ego off the lane. The 1.433 m car
collides little because it barely moves in the junction (mean 0.76 m/s in decision 125, a car standing in the wrong lane gets hit 1-2 times: 27994 four times, 24758 one
stationary MKZ repeatedly) and never reaches a pole. Turning into traffic without yielding is the in-lane vehicle class (122nz 2, 122 3, A 2): the same count with zones on, so it
is the shared longitudinal / yield scheduler (a car crossing at 4-10 m/s: 9196's firetruck and 5423's mustang are hit in most arms), not the camera. Over all 25 turns / 20 routes hits are
A 6, OL 9, 122 13, D 13, 143nz 16 (16 routes), OLnz 21, 122nz 24 (16 routes); of OLnz's 21, 16 happen after the 15 s turn window (stuck or parked off-road, repeated hits on
24758 / 26153 / 27994 / 34183) and only 7 are vehicle contacts with the ego off the lane.

## Curvature desired vs needed

Per turn in the tables (head peak = max |act_k| in the window, before clip / delay). Zones-off head peaks: median 0.27 x needed at 1.433 m D on choice turns, 0.30 x at 1.86 m, 5.7 x at
1.22 m; forced turns D 1.30 x, OLnz 0.75 x, 122nz 2.66 x. With the zones on the head output is 0.8-0.9 x needed on the turns A / OL take (not what steers there). At 1.86 m the
head asks for the same tiny curvature as at 1.433 m (mostly 0.01-0.1 1/m where 0.07-0.21 is needed), consistent with 121 / 122: the head is not a turn generator, and moving the camera
up (to match the open-loop boards) does not change that.

## Figures

- `figs/junction_rig122_panels.png`: BEV, four turns, paths of D (1.433 m), 143nz, OLnz (1.86 m), OL, 122nz (1.22 m) over the dense lane: choice 10255 (tight right, 6 m), choice 34183 (left 13 m),
  forced 28180 (T-stem, 4.7 m), forced 24944 (curve 10.7 m). Look at: D / 143nz / OLnz go straight on in the choice junctions; 122nz reaches the exit but cuts and swings (34183: 6.7 m); OL
  (zones on) hugs the centreline.
- `figs/rig122_ol_28008.gif`, `figs/rig122_olnz_28008.gif`: route 28008 choice junction (turn 1, -90 deg, night), chase camera next to both model input frames (road above, wide below), no command drawn.
  OL turns left with the route; OLnz drives straight on past the junction. Both are re-runs for recording, not the scored cell (the recording agent is not bitwise equal);
  the OL clip is 6.4 MB because the encoder ladder could not shrink it further. Not frame-reviewed.

## Limits

n = 25 turns on 20 routes, one run per cell, seed 2; closed loop is not bitwise deterministic (route DS flips on identical runs), so single turns flip; the CIs are route-cluster bootstrap and
the discordant-pair counts are in the tables file. 143nz / 122nz / 122nz miss four routes (the last 1.22 m shards were stopped): their rows cover 16 routes / 10 + 9 turns, comparisons
are on those turns. The OL / OLnz runs use `spec` (op-path clip + 0.2 s delay) while D is the legacy raw-curvature `drive` arm; 143nz isolates this (143nz = D on forced, 0 vs 2 on choice).
The 1.86 m camera sits above the MKZ roof, so objects close in front of the car are seen from higher up than any real openpilot mount.

## Draft decision paragraph (Chinese, for main)

**B2D 相机对齐 open-loop 板（x 1.59 m、z 1.86 m）后 action 头仍不转弯：zones 关时 choice 0/13、forced 1/12，与 1.433 m 的 D（2/13、4/12）和 `spec` 1.433 m（0/10、2/9）无差别；1.22 m 是三档里唯一让头自己走上出口分支的高度（5/10、3/9），但转得不干净（**中偏弱**，25 转弯 / 20 路线，seed 2 单次）。**
2026-10-05。接第 121 / 122 / 125 条。用户裁定 B2D 相机要与 NAVSIM（CAM_F0 x 1.665、z 1.862 m）和 WOD（前相机 x 1.519、z 约 1.86 m）的视角对齐，不为 B2D 单独降到 1.22 m；`spec` 相机改为两者均值 (1.59, 0, 1.86)，1.22 m（`spec_bumper122`）和 1.433 m（`spec_windshield143`）留作命名预设。结果 [experiments/op_closed_loop/results/junction_rig122.md](../../experiments/op_closed_loop/results/junction_rig122.md)。（1）对齐相机 + zones 关：choice 0/13、forced 1/12，离开车道 100% / 78%，头的期望曲率峰值只有所需的 0.30 倍（choice）；同一批转弯上 1.433 m 的 D 为 2/13、4/12，`spec` 1.433 m 为 0/10、2/9。**第 121 / 122 条「头直行穿过路口、急弯不转」在 1.86 m 下成立，不是 1.433 m 的相机偶然。**（2）zones 开：对齐相机与 shipped 路线臂相当（choice 12/13 对 11/13，forced 8/12 对 9/12，DS 69.6 对 67.7），相机移动本身不伤、也不帮。（3）旁行 1.22 m：zones 关 choice 5/10、forced 3/9（对 143nz +50 pp [+20, +80]；+11 pp [0, 38]），方向与第 125 条一致；但 90% 的 choice 转弯离开车道（峰值中位 7.2 m），头要 5.7 倍所需曲率，是过冲摆过去而不是跟弯。头的曲率随相机高度变化，1.22 m 过猛、1.433 / 1.86 m 过弱，没有一档能不靠路线转路口。（4）第 125 条 10 对 3 的碰撞：6 次是 1.22 m 车摆出车道后撞杆、护栏、红绿灯、建筑，另 2 次出车道撞车，不是转弯不让行；1.433 m 少是因为车几乎不动。路口内真正「转进来车」的车内碰撞（同类 2–3 次）所有 zones 开的臂一样，归纵向让行调度而非相机。对方向的含义：把相机对齐开环榜后，路口转弯的能力缺口不会被相机高度解决，仍要靠路线折线输入 / 图像指令 + 微调教会模型转弯幅度，高度对齐留给微调统一处理。**状态**：待定。**限定**：n = 25 转弯，单 seed 单次；143nz / 122nz 缺 4 条路线（末两个 1.22 m 分片被计划变更停掉）；OL 用 `spec`（clip + 延迟）而 D 是旧 `drive` 原始曲率，143nz 为分离该因素的对照。**会推翻或推进本条的证据**：第二个 seed；在对齐相机上微调后的头在同一批转弯上的走对率与期望曲率；38 + 30 全集。
- `figs/spec_shipped_28180_sharp_turn.gif` (2026-10-06): shipped under the B2D `spec` preset (open-loop-aligned camera 1.86 m, op-path, zones
  off, desire on), route 28180 turn 0 (forced right turn at a T junction, R_min 4.7 m), chase camera next to both model input frames; made with
  `experiments/op_route_ft/scripts/rft_gif_lane.py --gif shipped:28180` + `scripts/junction_forced_gif.py --speedup 3 --fps 8 --width 480`. A
  re-run for recording, not the scored cell. Look at the car at the T: it does not turn right and runs straight over the kerb towards the pole.
