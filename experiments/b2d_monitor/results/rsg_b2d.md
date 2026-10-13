# Stage 2: RSG-L18-s0 (route memory) in the Bench2Drive loop (2026-10-13)

Tokyo box, card 0, windowed loop, Bench2Drive 0.0.4 val, `--tm-seed 0`, **one rollout per route, one seed**, and every config carries
`stop_on`: the route ends 0.5 s after the first collision or after 1 s more than 3.5 m from the dense route (simulator truth, evaluation
only), so DS is "up to the first event". Five turn routes: 334 Town07 left, 27005 Town03 left, 24948 Town07 right, 27870 Town04 right,
24944 Town02 left. Run dirs are under `$DATA_DIR/runs/tokyo_b2d_viz/runs/` on Tokyo (pre-landing location). Reader:
`scripts/rsg_turns.py <run dir>`; steer plots: `scripts/steer_plot.py`.

## Serving path (new)

RSG-L18-s0 (decision 286) had no B2D serving path: the 20 Hz ONNX + bias server feeds only the 20-dim ego row. Built here:
`scripts/rsg_server.py` (envs/op-train, ~1.3 GB on the card) = `sh30_core.Core` (ROUTE-MEM via the `route_mem.json` sidecar, `lead_out`) +
`pai_core.plan`'s slots, 8 real policy slots at 0.2 s from the CARLA road / wide renders packed as the family's training frames
(`lib/rsg_pack.py`: BT.601 full-range YCbCr, nearest model pixel, 2x2 chroma mean, no JPEG round trip). The agent (config key `rsg`)
sends every 0.2 s of sim time: the packed frame, P (4, 3) / V (4, 2) at -1.5 / -1.0 / -0.5 / 0 s in the t0 rear-axle frame (CARLA's
left-handed world mapped to (x, -y, -yaw)), acc, the 3-way command (route desire: L / R within ~20 m of the junction, else S), and the
route memory `rmem.memory` of the dense leaderboard route at the ego pose. The 8 served poses are the P7 tracker's path; longitudinal =
FIX1 on the plan's own speed / lead heads with c2 = the adapter's plan.

Checks: sign (in a left turn the fed history has y > 0, yaw < 0, NAVSIM's signature, matching CARLA truth -34 deg over 1.5 s);
memory geometry (fed points vs the dense route at the same arc offsets: 0.04-0.7 m while driving; `rsg-geom-334`, `rsg-geom2-334`);
index maps identical to `jevdrive.openpilot.frames` (0 of 131072 pixels differ). Not identical to PAI serving: P7 pursuit on the poses
instead of AlpaSim's MPC on 1-2 s positions; CARLA renders without JPEG; vy = ay = 0 in the ego row (B2D parity convention) except arm B;
the full route in memory from the start (PAI accumulates batches starting 42 m ahead).

## Server speed-up

| Stage per decision | First version | Landed (agent packs, per-slot token cache, `Core.compile`) |
|---|---|---|
| pack | 3.5-6 ms | 3.0-3.2 ms |
| encode | 66-100 ms | 7-8 ms |
| policy | 16-30 ms | 3.3-3.4 ms |
| export | - | 0.1-0.2 ms |
| agent end to end | 103-121 ms | 15-16 ms |

Exactness (every 50th full decision vs `pai_core.plan` interpreted, full re-encode): max pose difference 4-12 mm (compile / fp16
rounding). Sim speed stays ~21-22 ticks/s: render-bound (two 1928x1208 Epic cameras, windowed).

## Route memory ON vs OFF (aim 0.5 s, `rsg_smooth` / `rsg_smooth_rmemoff`; runs `rsg-on-t4`, `rsg-off-t4`)

| Route | Turn | Route onset | Car onset ON / OFF | 2 s plan miss ON / OFF | 4 s plan miss ON / OFF | DS ON / OFF | End ON / OFF |
|---|---|--:|---|---|---|---|---|
| 334 Town07 | left | +6.5 m | -12.6 / -14.6 m | 2.6 / 3.3 m | 8.3 / 12.6 m | 21.3 / 15.9 | vehicle / vehicle |
| 27005 Town03 | left | +5.4 m | -1.5 / -8.5 m | 2.2 / 4.2 m | 2.8 / 14.1 m | 16.1 / 16.8 | vehicle / vehicle |
| 24948 Town07 | right | +1.6 m | -3.4 / -1.4 m | 3.0 / 1.4 m | 5.7 / 0.5 m | 26.3 / 29.1 | pole / fence |
| 27870 Town04 | right | +4.4 m | -3.4 / -3.4 m | 4.3 / 4.5 m | 3.3 / 3.4 m | 9.7 / 9.7 | guardrail / guardrail |
| 24944 Town02 | left | +4.5 m | (pole before the turn) | - | - | 6.4 / 6.4 | pole / pole |

Onsets in m relative to the junction entry (first LEFT / RIGHT command point of the dense route; negative = before). Route onset = the
dense route's heading has turned 10 deg; car onset = the car's true heading has turned 10 deg. Plan miss = lateral distance of the
plan's 2 s / 4 s pose from the dense route at the decision nearest the car onset. Reading: the memory acts on left turns (plan closer
to the route, later onset: 27005 -1.5 vs -8.5 m), not on right turns; RSG still turns early and cuts the inside (at onset its 2 s pose
is 2-4 m off the route; on 334 the 4 s pose aims ~24 m past the entry). Every turn route ends in a collision.

## Early-turn arms (`rsg_lat12*`, all memory ON)

| Route (route onset) | t4: aim 0.5 s | A: aim 1.5 s | B = A + true vy / ay | C1 = A, L/R only within 5 m | C2 = A, command S |
|---|---|---|---|---|---|
| 334 left (+6.5 m) | -12.6 m, DS 21.3 | -10.6, 20.7 | -8.6, 21.3 | -6.6 (flip -4.6), 22.0 | -7.6, 22.0 |
| 27005 left (+5.4) | -1.5, 16.1 | -2.5, 17.2 | -3.5, 16.1 | -1.5 (flip -4.5), 16.1 | +4.2, 33.5 |
| 24948 right (+1.6) | -3.4, 26.3 | -3.4, 27.7 | -4.4, 26.3 | +0.8 (flip -4.4), 24.4 | no turn, 43.7 |
| 27870 right (+4.4) | -3.4, 9.7 | -3.4, 8.6 | -3.4, 9.7 | -3.4, 9.7 | -3.4, 9.7 |
| 24944 left (+4.5) | pole before the turn, 6.4 | 5.9 | off-route, 9.1 | 6.4 | 5.9 |

Cells: car onset (m vs entry), DS. First event: a collision in every cell except C2 on 24948 and 27005 (off route), so C2's higher DS
there (43.7, 33.5) is distance driven before leaving the route, not a completed turn. The default command switches to L / R 18.4-19.6 m
before the entry on 334, 27005, 24948 and 10.4 m on 27870 (`cmd_on_m` of `rsg_turns.py`). Runs: t4 = `rsg-on-t4`; A `loop-1013-1852-p1` (`cfg/rsg_lat12.json`, P7 aim
`cfg/P7_la15.json`); B `loop-1013-1903-p1` (`rsg_lat12_vy`; `loop-1013-1902-*` invalid, server bug); C1 `loop-1013-1907-p1`
(`rsg_lat12_c1`); C2 `loop-1013-1910-p1` (`rsg_lat12_c2`). "flip" = where the command switched to L / R in C1 (m vs entry).

Reading: the executor horizon (A vs t4) and vy / ay (B) change nothing material. The 18-20 m L/R command is the main onset lever: flipping
it at 5 m delays the onset by ~4 m on 334 (-10.6 -> -6.6) and 24948 (-3.4 -> +0.8; 24948 then enters within +-3 m of the route onset
+1.6). With no turn command (C2) the memory alone times the 27005 left turn (+4.2 vs route +5.4 m) but does not make the 24948 right turn;
334 turns early even with no command (scene / junction-mouth trigger). The steer is a ramp from ~2 s before onset in every arm (the car
follows a plan that already curves): `figs/steer_lat12_vs_t4.png`, `figs/steer_A_B_C1_C2.png`. The memory ON / OFF, C1 and C2 rows were re-read with the landed `scripts/rsg_turns.py`
on 2026-10-13 ~19:35 JST: identical.

## Start delay

| Model | Warm-up | What starts the car from standstill | Car moves at (sim) |
|---|---|---|---|
| Cinque shipped (`op_alone_smooth`) | 5 s | FIX1 (c2) launch ramp, at once | 5.80-5.95 s (every route) |
| RSG-L18-s0 (`rsg_*`, c2 = the adapter's plan) | 2 s | not the plan: the adapter launch rarely asks for > 0.3 m/s; decision 140's driver resume (registered 10 s standstill) fires at 12.05 s, or the 15 s timer at 17.0 s | 12.85-12.90 s (334, 24944, 27870: resume rule); 17.85-17.95 s (24948: resume at 17.0; 27005: timer at 17.0) |

From `lon.jsonl` events (`resume_rule_fire`, `timer_fallback_fire`) and `ticks.jsonl` (first tick with v > 0.5 m/s), identical in all
RSG runs above (t4 ON / OFF, A, B, C1, C2). The first agent tick is at sim 0.05 s (map load is before the sim clock). This is the same
standing d231 measured for the adapter family in CARLA (own-plan launch share 0.24).

Status: RSG on B2D closed by the user after arms C1 / C2 (2026-10-13 ~19:15 JST).
