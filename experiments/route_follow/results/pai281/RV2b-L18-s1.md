# PAI 281: RV2b-L18-s1 vs RSG-L18-s0 (stored read r281/rsg), gains 20, JEV_VCONT 1.0, JEV_LEAD 1

Run `/data/runs/route_night/v2pai/RV2b-L18-s1` (3bc80aa8 rf_v2pai.sh chain: start the next 281 read on each card as it finishes the current one's last chunk (no idle card)), reference `/data/runs/route_night/r281/rsg`; rf_v2pai.sh read. Served memory check (rf_v2pai.py served):

- served memory = offline rebuild: 281 sessions, 55693 decisions, max |diff| 1.50e-01 (logged rounding 1e-5), mask differs at 0
- ordered start vs projection start (55693 decisions): median |diff| 0.00 m, p90 0.42 m, >= 5 m at 2100; first valid slot ordered / projection differ at 3961 (ordered further at 3215)

Notes: 4 rollouts of chunk c001 (901c2cff, 90ab8c24, 91d84204, 92af888e) failed at start (renderer port 6500 connection timeout when card 1
joined) and were rerun as `runs/rv2b-l18-s1_c001r`; rf_pai_follow extract skips failed rollouts without a controller trace (d3f590da).
Reproducibility: confirmed. An exact rerun of chunk c007 (29 scenes, 12 zeros, 1 spin; `/data/runs/route_night/v2pai/rerun_RV2b_c007`)
is bit-identical to the stored chunk at every decision of all 29 scenes (scores, spins, zeros equal). The RV2a non-reproduction of
Amendment 17 was RV2a's first run driving at the default gains (Amendment 18); this run carries gains 20 in every chunk.

## All 281 and the exam / control lists (rf_pai281.py; turn proxy = the 20 reference turn scenes)

Scenes scored by every label: 281; turn scenes (a scoring reference label turned 60-200 deg): 18 (left 10, right 8)

| label | mean | zeros | collision | offroad | corridor | handoff spins | turn done left | turn done right |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|
| rsg | 0.5720 | 70 | 9 | 14 | 47 | 2 | 10/10 | 8/8 |
| rv2b-l18-s1 | 0.5819 | 67 | 12 | 11 | 44 | 2 | 8/10 | 5/8 |

## all281: 281 scenes

| label | mean | zeros | collision | offroad | corridor | rear-ended | spins | plan 2 s deficit m | vs rsg (95 % CI) |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|:--|
| rsg | 0.5720 | 70 | 9 | 14 | 47 | 77 | 2 | -0.60 | +0.0000 [+0.0000, +0.0000] |
| rv2b-l18-s1 | 0.5819 | 67 | 12 | 11 | 44 | 78 | 2 | -0.57 | +0.0099 [-0.0120, +0.0313] |

## exam72: 72 scenes

| label | mean | zeros | collision | offroad | corridor | rear-ended | spins | plan 2 s deficit m | vs rsg (95 % CI) |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|:--|
| rsg | 0.3348 | 38 | 6 | 9 | 23 | 11 | 1 | -0.77 | +0.0000 [+0.0000, +0.0000] |
| rv2b-l18-s1 | 0.3573 | 35 | 6 | 7 | 22 | 13 | 1 | -0.69 | +0.0225 [-0.0254, +0.0710] |

## control75: 75 scenes

| label | mean | zeros | collision | offroad | corridor | rear-ended | spins | plan 2 s deficit m | vs rsg (95 % CI) |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|:--|
| rsg | 0.7779 | 0 | 0 | 0 | 0 | 30 | 0 | -0.50 | +0.0000 [+0.0000, +0.0000] |
| rv2b-l18-s1 | 0.7811 | 0 | 0 | 0 | 0 | 29 | 0 | -0.50 | +0.0032 [-0.0016, +0.0081] |

## Strict turns (36) and exam scenes (72): rf_pai_follow's turn rule on each label's own route (rf_v2pai.py turns / table)

### rv2b-l18-s1 vs rsg: per scene

Scenes 90; strict turns = 36 (the identity's strict turns); exam turn scenes 72.


## Counts

- strict turns completed: rsg 5 / 36, rv2b-l18-s1 7 / 36; gained 3 (b40f4729, e848c843, ea468790), lost 1 (132ae6cb)
- decision 293 shape gate (completed and kappa / dkappa / a_lat / entry speed within the class gate): rsg 1, rv2b-l18-s1 4
- exam turn scenes: 72 (18 with a strict turn on the identity's route): completed rsg 3, rv2b-l18-s1 4
- zeros over the 90 scenes: rsg 49, rv2b-l18-s1 47; mean score rsg 0.3219, rv2b-l18-s1 0.3246, paired +0.0027 [-0.0502, +0.0510]
- new zeros on the non-turn part (rv2b-l18-s1 zero, rsg not, event before the entry or no strict turn): 2 (a4d2f7cf, 7750265d); offroad among them (stop rule >= 3): 1

| scene | set | side | deg | rsg | rv2b-l18-s1 | rv2b-l18-s1 where (m rel. entry) | rv2b-l18-s1 v_0 | rv2b-l18-s1 shape gate | switch on / dec (first, m rel. entry) |
|:--|:--|:--|--:|:--|:--|:--|--:|:--|:--|
| 09528162 | strict+exam | R | -100 | off | event (corridor) | +2.3 | 3.3 | not completed |  /  |
| 132ae6cb | strict | L | +86 | completed | off |  | 6.5 | not completed |  /  |
| 1e8e939e | strict | R | -49 | event (collision) | event (collision) | -8.1 |  | not completed |  /  |
| 22fa5c92 | strict+exam | L | +91 | event (collision) | event (collision) | +3.9 | 2.6 | not completed |  /  |
| 268949d2 | strict | R | -88 | off | event (collision) | +2.2 | 3.7 | not completed |  /  |
| 2f227f2c | strict+exam | R | -66 | event (corridor) | event (corridor) | +1.1 | 10.6 | not completed |  /  |
| 302c5c99 | strict+exam | L | +163 | event (corridor) | event (corridor) | +1.3 | 6.5 | not completed |  /  |
| 3232fdb7 | strict | R | -87 | event (corridor) | event (corridor) | -2.9 |  | not completed |  /  |
| 33c8976b | strict | L | +107 | event (offroad) | event (offroad) | +4.7 | 2.6 | not completed |  /  |
| 37d660b1 | strict | R | -85 | event (corridor) | event (corridor) | +9.8 | 4.5 | not completed |  /  |
| 382a7247 | strict+exam | L | +83 | completed (collision) | completed (collision) | +123.6 | 8.7 | fail: alat,v_entry |  /  |
| 3c2822d9 | strict+exam | L | +81 | completed | completed |  | 6.5 | fail: v_entry |  /  |
| 3ce13eed | strict+exam | R | -100 | event (corridor) | event (corridor) | +6.9 | 3.6 | not completed |  /  |
| 403d558a | strict | R | -93 | event (corridor) | event (corridor) | +5.4 | 11.7 | not completed |  /  |
| 68e53160 | strict | L | +108 | event (corridor) | event (corridor) | +20.0 | 5.0 | not completed |  /  |
| 6fd8cf30 | strict | R | -88 | short_rear | short_rear |  |  | not completed |  /  |
| 8a143cb2 | strict | R | -76 | completed | completed |  | 1.8 | pass |  /  |
| 94c1d722 | strict+exam | R | -86 | event (offroad) | off |  | 3.1 | not completed |  /  |
| 9f393c8a | strict+exam | R | -84 | event (collision) | event (collision) | +17.4 | 5.9 | not completed |  /  |
| a174faed | strict | R | -90 | event (corridor) | event (corridor) | +4.2 | 6.2 | not completed |  /  |
| a1b166cc | strict+exam | L | +91 | event (corridor) | off |  | 3.8 | not completed |  /  |
| a4d2f7cf | strict | L | +110 | short_slow | event (corridor) | -6.1 |  | not completed |  /  |
| abd45a30 | strict+exam | L | +90 | short_rear | short_rear |  |  | not completed |  /  |
| b40f4729 | strict | R | -89 | off | completed |  | 2.9 | pass |  /  |
| c4a58f83 | strict+exam | R | -85 | short_slow | short_slow |  |  | not completed |  /  |
| ce161092 | strict+exam | L | +85 | event (offroad) | event (offroad) | +22.5 | 8.1 | not completed |  /  |
| d90fe123 | strict+exam | L | +85 | completed | completed |  | 8.2 | fail: v_entry |  /  |
| ddc3e8df | strict | R | -66 | event (corridor) | off |  | 2.6 | not completed |  /  |
| e0f6ebe1 | strict+exam | R | -90 | off | off |  | 7.8 | not completed |  /  |
| e37aa159 | strict+exam | L | +82 | event (corridor) | event (corridor) | +15.5 | 7.5 | not completed |  /  |
| e5e47fd2 | strict | L | +95 | event (corridor) | event (corridor) | +6.3 | 13.7 | not completed |  /  |
| e848c843 | strict+exam | R | -90 | off | completed |  | 3.1 | pass |  /  |
| ea468790 | strict | R | -83 | off | completed |  | 3.2 | pass |  /  |
| eaba3f6a | strict | R | -90 | event (corridor) | event (corridor) | +3.7 | 12.4 | not completed |  /  |
| f9e361aa | strict | R | -81 | event (corridor) | event (corridor) | +3.9 | 11.4 | not completed |  /  |
| fd01e9ab | strict+exam | L | +98 | event (corridor) | event (corridor) | +10.5 | 3.7 | not completed |  /  |
| 00fabf60 | +exam |  |  | no turn (corridor) | no turn (corridor) |  |  |  |  /  |
| 054b5901 | +exam |  |  | no turn (corridor) | no turn (corridor) |  |  |  |  /  |
| 05855d57 | +exam |  |  | no turn | no turn |  |  |  |  /  |
| 06b3e399 | +exam |  |  | no turn | no turn |  |  |  |  /  |
| 0feb3afa | +exam |  |  | no turn | no turn |  |  |  |  /  |
| 1089758a | +exam |  |  | no turn | no turn |  |  |  |  /  |
| 127c4f85 | +exam |  |  | no turn (corridor) | no turn (corridor) |  |  |  |  /  |
| 1a69f757 | +exam |  |  | no turn (collision) | no turn (collision) |  |  |  |  /  |
| 2466266e | +exam |  |  | no turn (offroad) | no turn (offroad) |  |  |  |  /  |
| 26004ca8 | +exam |  |  | no turn | no turn |  |  |  |  /  |
| 2b577fc2 | +exam |  |  | no turn | no turn |  |  |  |  /  |
| 2c263e19 | +exam |  |  | no turn | no turn |  |  |  |  /  |
| 2eaf1872 | +exam |  |  | no turn (offroad) | no turn |  |  |  |  /  |
| 300465f7 | +exam |  |  | no turn (offroad) | no turn (offroad) |  |  |  |  /  |
| 3fa5e06b | +exam |  |  | no turn | no turn |  |  |  |  /  |
| 404541da | +exam |  |  | no turn | no turn |  |  |  |  /  |
| 415e950b | +exam |  |  | no turn | no turn |  |  |  |  /  |
| 44df49fe | +exam |  |  | no turn (corridor) | no turn (corridor) |  |  |  |  /  |
| 4a863a41 | +exam |  |  | no turn | no turn |  |  |  |  /  |
| 4cb05d65 | +exam |  |  | no turn (offroad) | no turn (offroad) |  |  |  |  /  |
| 4cb7f7ac | +exam |  |  | no turn (corridor) | no turn (corridor) |  |  |  |  /  |
| 4f7cb072 | +exam |  |  | no turn | no turn |  |  |  |  /  |
| 5266c7c9 | +exam |  |  | no turn | no turn |  |  |  |  /  |
| 559d4ad6 | +exam |  |  | no turn (corridor) | no turn (corridor) |  |  |  |  /  |
| 5c4a277a | +exam |  |  | no turn | no turn |  |  |  |  /  |
| 66b15fbe | +exam |  |  | no turn | no turn |  |  |  |  /  |
| 6ee9ee78 | +exam |  |  | no turn (corridor) | no turn |  |  |  |  /  |
| 7381c9fa | +exam |  |  | no turn (offroad) | no turn (offroad) |  |  |  |  /  |
| 7750265d | +exam |  |  | no turn | no turn (offroad) |  |  |  |  /  |
| 7d994ed3 | +exam |  |  | no turn (corridor) | no turn (corridor) |  |  |  |  /  |
| 8120803d | +exam |  |  | no turn | no turn |  |  |  |  /  |
| 8b25a6ac | +exam |  |  | no turn | no turn |  |  |  |  /  |
| 98f88a4e | +exam |  |  | no turn (offroad) | no turn |  |  |  |  /  |
| 9b6c68b4 | +exam |  |  | no turn | no turn |  |  |  |  /  |
| 9f8acd00 | +exam |  |  | no turn (corridor) | no turn (corridor) |  |  |  |  /  |
| a2c5ae46 | +exam |  |  | no turn (corridor) | no turn (corridor) |  |  |  |  /  |
| a68ad579 | +exam |  |  | no turn | no turn |  |  |  |  /  |
| b44a4a44 | +exam |  |  | no turn (collision) | no turn (collision) |  |  |  |  /  |
| ba40be62 | +exam |  |  | no turn (corridor) | no turn (corridor) |  |  |  |  /  |
| bbc1699d | +exam |  |  | no turn | no turn |  |  |  |  /  |
| bbc53aa8 | +exam |  |  | no turn | no turn |  |  |  |  /  |
| bc309fd6 | +exam |  |  | no turn (corridor) | no turn (corridor) |  |  |  |  /  |
| bd97de34 | +exam |  |  | no turn (corridor) | no turn (corridor) |  |  |  |  /  |
| c5da4975 | +exam |  |  | no turn | no turn |  |  |  |  /  |
| ca173de1 | +exam |  |  | no turn | no turn |  |  |  |  /  |
| cbd87234 | +exam |  |  | no turn (corridor) | no turn (corridor) |  |  |  |  /  |
| cdf95fd1 | +exam |  |  | no turn (corridor) | no turn (corridor) |  |  |  |  /  |
| d8bfaaf2 | +exam |  |  | no turn (offroad) | no turn (offroad) |  |  |  |  /  |
| dc401449 | +exam |  |  | no turn | no turn |  |  |  |  /  |
| de9db5dc | +exam |  |  | no turn (collision) | no turn (collision) |  |  |  |  /  |
| e7db8971 | +exam |  |  | no turn | no turn |  |  |  |  /  |
| e904e9c0 | +exam |  |  | no turn (corridor) | no turn (corridor) |  |  |  |  /  |
| f8ee31cb | +exam |  |  | no turn | no turn |  |  |  |  /  |
| fc9fde87 | +exam |  |  | no turn (corridor) | no turn (corridor) |  |  |  |  /  |

## Lane changes and the approach offset (rf_laneoff.py, d299's reading B instruments; added 2026-10-15 11:40 JST)

Inputs: rf_tax extract and rf_progress rmem of this run (Tokyo `/data/runs/route_night/v2pai/lane/`), rf_tax report and rf_pai_follow report
(Mac), the PAI lane pickle of d299; one `rf_laneoff.py report` over RSG (r281/rsg, d299's inputs) + RV2a-L18-s0 + RV2b-L18-s1 (RSG reproduces
d299: reached 4, shifted 1, in lane 2). All three runs find the same 6 map-validated sites (route leaves its lane chain for an adjacent lane,
the human moved 2-6 m to that side).

Map-validated lane-change sites (R = ego reached the site; shift = ego's move toward the route's side in [s, s + 60] m, shifted = >= 2 m;
L = in the route's lane (|lat| <= 1.75 m) somewhere in [s + 30, s + 60]):

| scene | RSG-L18-s0 | RV2a-L18-s0 | RV2b-L18-s1 | human shift |
|:--|:--|:--|:--|--:|
| 1e8e939e | R +1.3 L | R +1.2 L | R +1.4 L | +2.9 |
| 415e950b | R +0.3 | R +0.0 | R **+8.1** (not in lane after; score 0.39) | +4.7 |
| 51242b4d | R +1.1 | R +1.0 | R +1.0 | +2.5 |
| 8a143cb2 | R +4.0 L | R +3.9 L | R +4.1 L | +2.6 |
| e904e9c0 | not reached (zero before) | not reached | not reached | +2.4 |
| f8ee31cb | not reached (site at the start) | not reached | not reached | +3.0 |
| **reached / shifted >= 2 m / in lane after** | **4 / 1 / 2** | **4 / 1 / 2** | **4 / 2 / 2** | |

Strict turns at entry - 20 m (d299's exploratory lead; inward = signed lateral offset to the route toward the turn's inside, - = the side
away from the turn; 22 of 36 turns defined in each run):

| run | median inward all | failed (n) | completed (n) | >= 0.75 m outside | >= 1.75 m outside | paired vs RSG (22 turns), mean [95 % CI] |
|:--|--:|:--|:--|--:|--:|:--|
| RSG-L18-s0 | -0.45 | -0.48 (19) | +0.06 (3) | 7 | 3 | |
| RV2a-L18-s0 | -0.44 | -0.44 (20) | -0.29 (2) | 7 | 3 | -0.01 [-0.03, +0.01] |
| RV2b-L18-s1 | -0.44 | -0.46 (19) | +0.03 (3) | 7 | 3 | +0.01 [-0.02, +0.05] |

Reading: lane changes did not improve. Neither arm makes a required lane change RSG misses; RV2b's one extra "shift" (415e950b) overshoots
by 8 m and does not end in the route's lane. The approach offset before strict turns is unchanged (median within 0.01 m, paired mean
within +-0.05 m): the lane hinge of arm b (weight 1) does not move the car toward the turn side on PAI.
