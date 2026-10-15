# LON-INVENTORY: material for longitudinal envelope rows on the policy's own states (2026-10-16)

Plan: [plans/2026-10-16-lon-inventory.md](../../plans/2026-10-16-lon-inventory.md). Script: `scripts/rf_loninv.py count` (box, CPU pool job
`loninv-full2`, 61 s wall). Numbers: `inventory.json` (everything), `tables.md` (generated tables), one CSV per class with one line per
decision (`policy_{red,stop,qual,free,turn}.csv`), `pack_check.json`. No training, no simulation, nothing recomputed in a simulator.

## Stores

| store | what | box path (`$DATA_DIR/runs/...`) |
|:--|:--|:--|
| policy | RSG-L18-s0's 18-decision rollouts on onpol-train, one per scene: 18 352 decisions in 1 021 scenes; pool `on1_L18RSG` = 17 474 rows (1 020 scenes, 97 logs), every row with frames (`@warp/front.npy` (17 474, 8, 32, 512)), tab, agent / road / route labels and the on_lon pack | `onpol_nuplan/rows/L18RSG/{meta,all,lon}.npz`, `op_parity/cache/on1_L18RSG{,@warp}`, rollouts `onpol_long/roll/RSG-L18-s0/c0..c5` (label run `onpol_long/labels-L18RSG`) |
| context | LONG-PROBE context of the 1 491 public scenes (route, stop lines with type / arc / light state at 24 keyframes / ln_tstop / ln_elig, human speed by arc) | `onpol_nuplan/ctx/ctx.npz` |
| expert packs | rf_expert.py pack | `route_follow/rewind_train64/pack` (64 train scenes), `route_follow/rule_expert/pack_held` (held 460, none in train) |
| human | navtrain 12 shards, 103 288 tokens / 1 192 logs (long_contact ctx + body1 agent labels + log tabs) | `long_contact/ctx/ctx.npz`, body1 labels |
| PAI high-speed | `on1_PAITV` (SLOW S2 / S3 rows) | `op_parity/cache/on1_PAITV{,@warp}` |

The 878 decisions not in the pool (overlap 672, log_future 151, corridor 42, frame 13) have rollout dumps but no tokens and no labels;
they are not classified. Decisions k = 0, 1 are log-replay states (the human's state, not the policy's); the tables give them apart.
A row counts only when it is on its route (|lateral| <= 3.5 m; 17 409 rows). Decisions are 0.5 s apart, so a decision count is already
a 2 Hz row count.

## 1. Stop-line approaches (line 0-40 m ahead of the front bumper, policy speed >= 6 m/s)

decisions / scenes / logs; every decision has frames.

| class | all | replay k <= 1 | human stopped at the line | plan not <= 1 m/s before the line | plan crosses within 4 s | no intent (plan end speed >= 0.7 x speed) AND human stopped | plan cannot stop at <= 3 m/s^2 AND human stopped |
|:--|--:|--:|--:|--:|--:|--:|--:|
| red light now (first light line ahead) | 120 / 53 / 44 | 27 | 62 / 28 / 24 | 119 | 29 / 13 / 11 | 7 / 5 / 5 | 0 |
| stop-sign line | 73 / 24 / 14 | 14 | 7 / 2 / 2 | 73 | 19 / 9 / 7 | 1 | 0 |
| LONG-PROBE qualifying (ln_elig; stop sign, or light red now) | 77 / 36 / 31 | 19 | 64 / 28 / 24 | 76 | 1 | 6 / 5 / 5 | 0 |

By speed floor (qualifying lines; all / human stopped / no intent AND human stopped):

| speed | qualifying | red light now | stop sign |
|:--|:--|:--|:--|
| >= 0 | 1 147 / 763 / 337 (119 scenes) | 1 388 / 711 / 318 | 690 / 99 / 50 |
| >= 3 | 451 / 312 / 43 (85 scenes, 49 logs) | 554 / 280 / 37 | 392 / 50 / 14 |
| >= 6 | 77 / 64 / 6 (36 scenes, 31 logs) | 120 / 62 / 7 | 73 / 7 / 1 |
| >= 9 | 2 / 0 / 0 | 4 / 0 / 0 | 1 / 0 / 0 |
| >= 12 | 0 | 0 | 0 |

Reading:

- **On nuPlan the policy mostly does brake for the lines it reaches fast**: where the human stopped, the plan's mean deceleration over 4 s
  is 0.82 m/s^2 against 0.65 needed to stop at the line (median, own states); "plan not <= 1 m/s before the line" is true almost everywhere
  only because the lines are 20-40 m away (96 of 120) and a 4 s plan cannot reach them. No decision where the human stopped has a plan
  that can no longer stop. This is not PAI's failure (13 / 14 scenes with no intent at 12-17 m/s, decision 315).
- **The speed regime PAI fails in does not exist here**: 4 red-light approach decisions at >= 9 m/s, 0 at >= 12 (policy speed on the pool
  p50 4.8, p99 12.8, max 15.0 m/s). nuPlan on-policy rows can carry "stop at the red line" at 3-9 m/s only.
- The 29 red-light crossings within 4 s are on lines the logged driver did not stop at (28 of 29 not ln_elig: crossed on red by the log,
  turn arrows, or a line of another connector); the plans there mostly decelerate (end ratio 0.23-0.82) and roll through at 1.6-9.2 m/s.
  These are label noise for a stop course; use the qualifying class.
- Stop-sign lines on the route are mostly not stops for the human (7 of 73 at >= 6 m/s): lc_ctx likely matches stop polygons the route passes
  through in junctions. Only ln_elig stop signs are usable.
- Share of the pool: qualifying at >= 3 m/s is 2.6 % of rows (451 / 17 474), at >= 6 m/s 0.44 %. The 5-10 % sampling target of
  traffic-light-probe needs ~2-4x oversampling of the >= 3 m/s set (85 scenes) or more rollouts.

Human navtrain (rows / logs), same definitions on the human's own state: red light now 1 711 / 335 (human stopped 894; decelerating,
ax < -0.5, 1 191 = 70 %), stop-sign line 819 / 171 (stopped 124), qualifying 1 418 / 269 (stopped 960, decelerating 977). The human rows are
10-20x more numerous but 70 % already decelerate (decision 315: the label is predictable from the history).

## 2. Free road slower than the reference (rf_lonol free cell, speed >= 2 m/s, human speed at the station known)

| human speed at the station | free decisions | policy < 0.85 x human | < 0.85 x proxy v_ref | with expert pack | < 0.85 x expert (pack) |
|:--|--:|--:|--:|--:|--:|
| < 8 | 2 259 / 321 / 77 | 802 / 163 / 58 | 749 | 172 / 33 / 23 | 37 / 13 / 10 |
| 8-12 | 2 693 / 294 / 74 | 1 055 / 170 / 63 | 1 026 | 60 / 14 / 12 | 42 / 11 / 9 |
| 12-16 | 397 / 42 / 19 | 149 / 22 / 14 | 149 | 0 | 0 |
| >= 16 | 0 | 0 | 0 | 0 | 0 |
| all | 5 349 / 522 / 87 (577 replay) | **2 006 / 319 / 81** (0 replay) | 1 924 / 326 / 78 | 232 / 36 / 26 | 79 / 21 / 15 |

- 37 % of the policy's free-road decisions are more than 15 % slower than the human at the same station; none of them at a replay state.
- nuPlan has no free road at >= 16 m/s in either store (human navtrain free rows: < 8 9 949, 8-12 9 197, 12-16 1 735, >= 16 **8**). The
  PAI handover regime (median 21 m/s) can only come from PAI rows (section 6) or speed augmentation.
- The expert's v_ref on free road is the human's speed with a 3 m/s floor and a curvature cap, so "below the expert" and "below the
  human" are nearly the same set (proxy 1 924 vs human 2 006).

## 3. Turn approach (strict turn >= 60 deg, entry 0-40 m ahead of the rear axle)

| | decisions / scenes / logs | replay k <= 1 |
|:--|--:|--:|
| turn approach | 2 234 / 260 / 68 | 383 |
| faster than proxy expert + 1 m/s | 114 / 40 / 24 | 35 (own states 79) |
| with expert pack | 277 / 57 / 34 | 114 |
| faster than expert (pack) + 1 m/s | 23 / 19 / 15 | 22 (own states 1) |
| faster than human + 1 m/s | 60 / 25 / 20 | 0 |
| faster than proxy + 1 AND human + 1 | 29 / 12 / 9 | 0 |

On nuPlan's own states the policy is rarely faster than the expert's entry profile: about 80 decisions in ~40 scenes (proxy), and the
pack-covered scenes show one. The overspeed into turns (decision 310: 10.7-13.8 m/s straight-through on PAI) is again a high-speed PAI
regime. Human navtrain: 20 492 turn-approach rows (785 logs), 873 faster than the proxy + 1.

## 4. Can these be on-policy rows in the current format

Yes for every class above: each counted decision is a pool row with frames (`on1_L18RSG@warp/front.npy`), the fed ego vector, the served
plan, actor boxes, road raster, route features and the on_lon pack (route in the row frame, eligible light lines with red per dense step
`b_mu`). Missing: (a) an envelope label pack per row (v_ref(s) on the row's route in its own frame + the reason masks: red / qualifying
line with its arc, stop-sign eligibility, turn entry, lead), built from `ctx.npz` + `lon.npz` exactly as on_lon_pack builds B / C, and the
`--on-lon` module that applies it; (b) for decisions outside the pool (878), tokens (`on_rows.py tok`, one GPU job) and a relaxed
exclusion; overlap decisions (672) are collision states and not worth it. Stop-sign lines are not in `lon.npz` (B is light-only); they
are in `ctx.npz`.

## 5. Expert reference coverage

64 of the 1 021 rollout scenes have a pack (`rewind_train64/pack`); the held 460 packs do not overlap the train split. Frame check: the
decision-0 anchors lie 0.09 m (median, max 0.18) from the pack's log path, so the pack and the rollout share a frame. Proxy v_ref
(rule_expert.Scene.profile on the context route) against the real pack at the same rows: median |diff| 0.11 m/s (free rows 0.17, turn
rows 0.38), 86 % within 1 m/s. Packing the other 957 scenes: the two pack jobs ran 460 scenes in 49 s on 32 workers and 64 in 23 s on
16 (about 3.4-5.7 core-s per scene), so ~1-1.5 core-hours, under 2 min wall as one CPU pool job (`rf_expert.py pack --runs
$DATA_DIR/runs/onpol_long/roll/RSG-L18-s0/c{0..5}/run`). Not done here.

## 6. PAI high-speed straight rows (on1_PAITV)

7 301 rows / 75 train160 scenes, frames present; speed p10 12.9, p50 19.8, p90 31.4, max 34.8 m/s; 12-16 m/s 2 646, >= 16 m/s 4 655.
Imitation targets of the human's own log (not on-policy states). SLOW verdict (lines12.md): S2 (S1 + PAITV) fails held (zeros +7.5, GT
+0.254 m) and the public guard (zeros +5.5); S3 (PAITV alone) passes the guard (zeros +2.5, plan 2 s short +0.031 [+0.023, +0.038]) and
fails held (GT +0.062 m, rear-ended 0.92 x RSG, zeros +3.0); the PAI 281 line is not read. They are the only rows anywhere at the PAI
handover speeds.

## Not counted

- PAI red lights and stop lines: PAI has no light state; train160 has 4 red-stop events (traffic-light-probe).
- Leads as an envelope reason: not counted as a class (free road excludes them).
- Decisions outside the pool (878): no stored pose in the row tables.
- Only one rollout per scene (RSG-L18-s0); other stores (L18RB0, L18YS, ...) are earlier checkpoints' states.
