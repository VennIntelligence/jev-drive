# Stage 1: shipped Cinque (and P2H10-F-s0) driving Bench2Drive on the Tokyo monitor (2026-10-13)

Tokyo box, card 0, CARLA 0.9.15 windowed at the desktop unless noted, Bench2Drive 0.0.4 val XML, `--tm-seed 0`, one rollout per
route. Ticks/s = route ticks / route wall, map load excluded. Run dirs are under `$DATA_DIR/runs/tokyo_b2d_viz/runs/` on Tokyo
(the pre-landing location; runs after the landing go to `$DATA_DIR/runs/b2d_monitor/runs/`). Code: the pre-landing copies of
`lib/op_arb_viz_agent.py` (same logic as landed). Configs: `cfg/` (`drive` = op_arb preset `drive`, not landed).

| Run | Config | Route | DS | RC | Wall | Ticks | Ticks/s | Infractions |
|---|---|---|--:|--:|--:|--:|--:|---|
| r24721-plain | drive | 24721 Town01 HardBreakRoute | 100 | 100 | 62 s | 1227 | 19.9 | - |
| loop-1013-1712-p1 | drive | 24721 | 100 | 100 | 63 s | 1314 | 20.8 | - |
| loop-1013-1712-p1 | drive | 334 Town07 SignalizedJunctionLeftTurn | 60 | 100 | 41 s | 690 | 16.8 | vehicle collision |
| loop-1013-1712-p1 | drive | 28147 Town10HD SignalizedJunctionLeftTurnEnterFlow | 100 | 100 | 102 s | 1161 | 11.4 | - |
| loop-1013-1719-p1 | op_alone | 24721 | 100 | 100 | 101 s | 2053 | 20.3 | - |
| loop-1013-1719-p1 | op_alone | 24944 Town02 T_Junction | 9.8 | 20.5 | 214 s | 4000 | 18.7 | layout collision, off lanes |
| smooth-t2 (off-screen) | op_alone_smooth | 24721 | 100 | 100 | 91 s | 1449 | 16.0 | - |
| loop-1013-1740-p1 | op_alone_smooth | 27005 Town03 NonSignalizedJunctionLeftTurn | 37.3 | 53.3 | 80 s | 1142 | 14.3 | red light, route deviation |
| p2h10-t1 (off-screen) | p2h10_alone_smooth | 334 | 31.4 | 100 | 47 s | 651 | 14.0 | layout + vehicle collision |

`drive` steers by the route in turn zones (op_arb's arm up to 2026-10-05); `op_alone` is openpilot alone (preset `spec`, zones /
divergence / coast off) with op_arb's longitudinal (min(set, IDM, plan) + stop latch + 5 s timer); `op_alone_smooth` replaces the
longitudinal by the FIX1 port (`LonSmooth` in `lib/op_arb_viz_agent.py`).

## FIX1 longitudinal port on route 24721 (op_alone vs op_alone_smooth, same lateral)

| Run | Config | DS | Ticks to finish | Brake on/off switches | Stops (v falls below 0.3 m/s) | Car moves at |
|---|---|--:|--:|--:|--:|--:|
| loop-1013-1719-p1 | op_alone | 100 | 2053 | 97 | 18 | 5.80 s |
| loop-1013-1731-p1 | op_alone | 100 | 2100 | 95 | 21 | 5.80 s |
| smooth-t2 | op_alone_smooth | 100 | 1449 | 53 | 12 | 5.85 s |
| loop-1013-1740-p1 | op_alone_smooth | 100 | 1443 | 59 | 12 | 5.85 s |

Counted from `ticks.jsonl` (brake > 0 toggles between consecutive ticks). The port finishes the route in ~30 % fewer ticks with about
half the brake switching; DS is 100 in every row, so the gain is smoothness / time, not score. Shipped Cinque never asks to move from
standstill; the FIX1 (c2) launch ramp starts it right after the 5 s warm-up.

## First-plan stall and server warm-up

The first plan after an `op_arb_server` start took 8.8 s on the first route (CARLA is synchronous, so the window froze); later routes'
first plans take 25-33 ms. `scripts/server.sh` therefore runs 20 warm-up plans (`scripts/op_warm.py`) before the loop starts. One
off-screen CARLA instance once ran at 1.0-1.7 s per tick from tick 0 (GPU at 98 %), unexplained; a restart gave 26 ms per tick.

P2H10-F-s0 ran through the existing op_parity path (arm ONNX `$DATA_DIR/runs/op_parity/hugsim/onnx/pp-P2H10-F-s0.onnx`, sha256
3f63266f..., + the bias server, `parity_ego` fed every plan): the off-screen smoke above and one on-screen pass
(`loop-1013-1746-p1`), then stopped by the user.
