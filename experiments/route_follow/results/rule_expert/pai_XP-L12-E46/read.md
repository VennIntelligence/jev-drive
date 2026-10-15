# PAI 281: rule expert XP-L12-E46 vs RSG-L18-s0 (stored read r281/rsg), gains 20

Run `/data/runs/route_night/expert/XP-L12-E46` (d2b3cd80 decision 308: plan switching refuted; plan creep found (each new plan accepts the car's shortfall; 2-4 m outward over the approach)), reference `/data/runs/route_night/r281/rsg`; rf_expert.sh tokyo-read.

## All 281 and the exam / control lists (rf_pai281.py)

Scenes scored by every label: 281; turn scenes (a scoring reference label turned 60-200 deg): 18 (left 10, right 8)

| label | mean | zeros | collision | offroad | corridor | handoff spins | turn done left | turn done right |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|
| rsg | 0.5720 | 70 | 9 | 14 | 47 | 2 | 10/10 | 8/8 |
| xp-l12-e46 | 0.7646 | 36 | 7 | 8 | 21 | 2 | 5/10 | 7/8 |

## all281: 281 scenes

| label | mean | zeros | collision | offroad | corridor | rear-ended | spins | plan 2 s deficit m | vs rsg (95 % CI) |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|:--|
| rsg | 0.5720 | 70 | 9 | 14 | 47 | 77 | 2 | -0.60 | +0.0000 [+0.0000, +0.0000] |
| xp-l12-e46 | 0.7646 | 36 | 7 | 8 | 21 | 49 | 2 | -0.79 | +0.1926 [+0.1357, +0.2499] |

## exam72: 72 scenes

| label | mean | zeros | collision | offroad | corridor | rear-ended | spins | plan 2 s deficit m | vs rsg (95 % CI) |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|:--|
| rsg | 0.3348 | 38 | 6 | 9 | 23 | 11 | 1 | -0.77 | +0.0000 [+0.0000, +0.0000] |
| xp-l12-e46 | 0.6872 | 16 | 4 | 2 | 10 | 13 | 2 | -0.78 | +0.3524 [+0.2462, +0.4565] |

## control75: 75 scenes

| label | mean | zeros | collision | offroad | corridor | rear-ended | spins | plan 2 s deficit m | vs rsg (95 % CI) |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|:--|
| rsg | 0.7779 | 0 | 0 | 0 | 0 | 30 | 0 | -0.50 | +0.0000 [+0.0000, +0.0000] |
| xp-l12-e46 | 0.8116 | 8 | 1 | 1 | 6 | 13 | 0 | -0.81 | +0.0337 [-0.0624, +0.1288] |

## Strict turns (36) and exam scenes (72) (rf_v2pai.py turns / table)

### xp-l12-e46 vs rsg: per scene

Scenes 90; strict turns = 36 (the identity's strict turns); exam turn scenes 72.


## Counts

- strict turns completed: rsg 5 / 36, xp-l12-e46 29 / 36; gained 26 (09528162, 22fa5c92, 268949d2, 2f227f2c, 302c5c99, 3232fdb7, 33c8976b, 37d660b1, 3ce13eed, 403d558a, 68e53160, 94c1d722, 9f393c8a, a1b166cc, a4d2f7cf, b40f4729, c4a58f83, ce161092, ddc3e8df, e0f6ebe1, e5e47fd2, e848c843, ea468790, eaba3f6a, f9e361aa, fd01e9ab), lost 2 (382a7247, d90fe123)
- decision 293 shape gate (completed and kappa / dkappa / a_lat / entry speed within the class gate): rsg 1, xp-l12-e46 24
- exam turn scenes: 72 (18 with a strict turn on the identity's route): completed rsg 3, xp-l12-e46 14
- zeros over the 90 scenes: rsg 49, xp-l12-e46 19; mean score rsg 0.3219, xp-l12-e46 0.6837, paired +0.3618 [+0.2664, +0.4589]
- new zeros on the non-turn part (xp-l12-e46 zero, rsg not, event before the entry or no strict turn): 1 (f8ee31cb); offroad among them (stop rule >= 3): 0

| scene | set | side | deg | rsg | xp-l12-e46 | xp-l12-e46 where (m rel. entry) | xp-l12-e46 v_0 | xp-l12-e46 shape gate | switch on / dec (first, m rel. entry) |
|:--|:--|:--|--:|:--|:--|:--|--:|:--|:--|
| 09528162 | strict+exam | R | -100 | off | completed |  | 4.4 | pass |  /  |
| 132ae6cb | strict | L | +86 | completed | completed |  | 4.8 | pass |  /  |
| 1e8e939e | strict | R | -49 | event (collision) | event (offroad) | -51.9 |  | not completed |  /  |
| 22fa5c92 | strict+exam | L | +91 | event (collision) | completed |  | 2.3 | pass |  /  |
| 268949d2 | strict | R | -88 | off | completed |  | 0.3 | pass |  /  |
| 2f227f2c | strict+exam | R | -66 | event (corridor) | completed |  | 4.4 | pass |  /  |
| 302c5c99 | strict+exam | L | +163 | event (corridor) | completed |  | 3.7 | fail: kmax,dk |  /  |
| 3232fdb7 | strict | R | -87 | event (corridor) | completed |  | 4.5 | pass |  /  |
| 33c8976b | strict | L | +107 | event (offroad) | completed |  | 2.9 | fail: kmax |  /  |
| 37d660b1 | strict | R | -85 | event (corridor) | completed |  | 4.8 | pass |  /  |
| 382a7247 | strict+exam | L | +83 | completed (collision) | event (collision) | +5.8 | 4.6 | not completed |  /  |
| 3c2822d9 | strict+exam | L | +81 | completed | completed |  | 6.5 | fail: v_entry |  /  |
| 3ce13eed | strict+exam | R | -100 | event (corridor) | completed |  | 3.3 | pass |  /  |
| 403d558a | strict | R | -93 | event (corridor) | completed (collision) | +25.7 | 4.8 | pass |  /  |
| 68e53160 | strict | L | +108 | event (corridor) | completed |  | 3.6 | pass |  /  |
| 6fd8cf30 | strict | R | -88 | short_rear | short_rear |  |  | not completed |  /  |
| 8a143cb2 | strict | R | -76 | completed | completed |  | 1.8 | pass |  /  |
| 94c1d722 | strict+exam | R | -86 | event (offroad) | completed |  | 3.6 | pass |  /  |
| 9f393c8a | strict+exam | R | -84 | event (collision) | completed |  | 5.9 | fail: v_entry |  /  |
| a174faed | strict | R | -90 | event (corridor) | event (offroad) | +2.2 | 4.5 | not completed |  /  |
| a1b166cc | strict+exam | L | +91 | event (corridor) | completed |  | 4.6 | pass |  /  |
| a4d2f7cf | strict | L | +110 | short_slow | completed |  | 4.0 | pass |  /  |
| abd45a30 | strict+exam | L | +90 | short_rear | short_slow |  | 4.7 | not completed |  /  |
| b40f4729 | strict | R | -89 | off | completed |  | 1.2 | fail: kmax,dk |  /  |
| c4a58f83 | strict+exam | R | -85 | short_slow | completed |  | 4.4 | pass |  /  |
| ce161092 | strict+exam | L | +85 | event (offroad) | completed |  | 4.7 | pass |  /  |
| d90fe123 | strict+exam | L | +85 | completed | short_rear |  | 4.8 | not completed |  /  |
| ddc3e8df | strict | R | -66 | event (corridor) | completed |  | 2.6 | pass |  /  |
| e0f6ebe1 | strict+exam | R | -90 | off | completed |  | 4.6 | pass |  /  |
| e37aa159 | strict+exam | L | +82 | event (corridor) | short_rear |  | 4.6 | not completed |  /  |
| e5e47fd2 | strict | L | +95 | event (corridor) | completed |  | 5.5 | pass |  /  |
| e848c843 | strict+exam | R | -90 | off | completed |  | 3.2 | pass |  /  |
| ea468790 | strict | R | -83 | off | completed |  | 3.2 | pass |  /  |
| eaba3f6a | strict | R | -90 | event (corridor) | completed |  | 4.1 | pass |  /  |
| f9e361aa | strict | R | -81 | event (corridor) | completed |  | 4.7 | pass |  /  |
| fd01e9ab | strict+exam | L | +98 | event (corridor) | completed |  | 4.5 | pass |  /  |
| 00fabf60 | +exam |  |  | no turn (corridor) | no turn |  |  |  |  /  |
| 054b5901 | +exam |  |  | no turn (corridor) | no turn (corridor) |  |  |  |  /  |
| 05855d57 | +exam |  |  | no turn | no turn |  |  |  |  /  |
| 06b3e399 | +exam |  |  | no turn | no turn |  |  |  |  /  |
| 0feb3afa | +exam |  |  | no turn | no turn |  |  |  |  /  |
| 1089758a | +exam |  |  | no turn | no turn |  |  |  |  /  |
| 127c4f85 | +exam |  |  | no turn (corridor) | no turn (corridor) |  |  |  |  /  |
| 1a69f757 | +exam |  |  | no turn (collision) | no turn (offroad) |  |  |  |  /  |
| 2466266e | +exam |  |  | no turn (offroad) | no turn (corridor) |  |  |  |  /  |
| 26004ca8 | +exam |  |  | no turn | no turn |  |  |  |  /  |
| 2b577fc2 | +exam |  |  | no turn | no turn |  |  |  |  /  |
| 2c263e19 | +exam |  |  | no turn | no turn |  |  |  |  /  |
| 2eaf1872 | +exam |  |  | no turn (offroad) | no turn |  |  |  |  /  |
| 300465f7 | +exam |  |  | no turn (offroad) | no turn |  |  |  |  /  |
| 3fa5e06b | +exam |  |  | no turn | no turn |  |  |  |  /  |
| 404541da | +exam |  |  | no turn | no turn |  |  |  |  /  |
| 415e950b | +exam |  |  | no turn | no turn |  |  |  |  /  |
| 44df49fe | +exam |  |  | no turn (corridor) | no turn (corridor) |  |  |  |  /  |
| 4a863a41 | +exam |  |  | no turn | no turn |  |  |  |  /  |
| 4cb05d65 | +exam |  |  | no turn (offroad) | no turn |  |  |  |  /  |
| 4cb7f7ac | +exam |  |  | no turn (corridor) | no turn (corridor) |  |  |  |  /  |
| 4f7cb072 | +exam |  |  | no turn | no turn |  |  |  |  /  |
| 5266c7c9 | +exam |  |  | no turn | no turn |  |  |  |  /  |
| 559d4ad6 | +exam |  |  | no turn (corridor) | no turn (corridor) |  |  |  |  /  |
| 5c4a277a | +exam |  |  | no turn | no turn |  |  |  |  /  |
| 66b15fbe | +exam |  |  | no turn | no turn |  |  |  |  /  |
| 6ee9ee78 | +exam |  |  | no turn (corridor) | no turn |  |  |  |  /  |
| 7381c9fa | +exam |  |  | no turn (offroad) | no turn (collision) |  |  |  |  /  |
| 7750265d | +exam |  |  | no turn | no turn |  |  |  |  /  |
| 7d994ed3 | +exam |  |  | no turn (corridor) | no turn (corridor) |  |  |  |  /  |
| 8120803d | +exam |  |  | no turn | no turn |  |  |  |  /  |
| 8b25a6ac | +exam |  |  | no turn | no turn |  |  |  |  /  |
| 98f88a4e | +exam |  |  | no turn (offroad) | no turn |  |  |  |  /  |
| 9b6c68b4 | +exam |  |  | no turn | no turn |  |  |  |  /  |
| 9f8acd00 | +exam |  |  | no turn (corridor) | no turn |  |  |  |  /  |
| a2c5ae46 | +exam |  |  | no turn (corridor) | no turn |  |  |  |  /  |
| a68ad579 | +exam |  |  | no turn | no turn |  |  |  |  /  |
| b44a4a44 | +exam |  |  | no turn (collision) | no turn (collision) |  |  |  |  /  |
| ba40be62 | +exam |  |  | no turn (corridor) | no turn (corridor) |  |  |  |  /  |
| bbc1699d | +exam |  |  | no turn | no turn |  |  |  |  /  |
| bbc53aa8 | +exam |  |  | no turn | no turn |  |  |  |  /  |
| bc309fd6 | +exam |  |  | no turn (corridor) | no turn (corridor) |  |  |  |  /  |
| bd97de34 | +exam |  |  | no turn (corridor) | no turn |  |  |  |  /  |
| c5da4975 | +exam |  |  | no turn | no turn |  |  |  |  /  |
| ca173de1 | +exam |  |  | no turn | no turn |  |  |  |  /  |
| cbd87234 | +exam |  |  | no turn (corridor) | no turn |  |  |  |  /  |
| cdf95fd1 | +exam |  |  | no turn (corridor) | no turn |  |  |  |  /  |
| d8bfaaf2 | +exam |  |  | no turn (offroad) | no turn (offroad) |  |  |  |  /  |
| dc401449 | +exam |  |  | no turn | no turn |  |  |  |  /  |
| de9db5dc | +exam |  |  | no turn (collision) | no turn |  |  |  |  /  |
| e7db8971 | +exam |  |  | no turn | no turn |  |  |  |  /  |
| e904e9c0 | +exam |  |  | no turn (corridor) | no turn |  |  |  |  /  |
| f8ee31cb | +exam |  |  | no turn | no turn (collision) |  |  |  |  /  |
| fc9fde87 | +exam |  |  | no turn (corridor) | no turn (corridor) |  |  |  |  /  |

## Runway turns (22) (rf_appread.py)

### PAI 281 approach mechanism: xp-l12-e46 vs rsg (rf_appread.py, EXPLORATORY Amendment 4)

Runway turns where both labels define the strict turn: 22 of 22. Human entry speed there (median): 4.99 m/s.

| read | n | rsg median | xp-l12-e46 median | mean xp-l12-e46 - rsg [95 % CI] | median diff |
|:--|--:|--:|--:|:--|--:|
| v_entry | 18 | 7.38 | 4.57 | -3.20 [-4.61, -1.82] | -2.98 |
| excess26 | 20 | 0.24 | 0.00 | -1.13 [-1.99, -0.45] | -0.18 |
| inward20 | 20 | -0.44 | -0.00 | +0.29 [+0.12, +0.51] | +0.11 |

| subset | n | rsg completed | xp-l12-e46 completed |
|:--|--:|--:|--:|
| runway22 | 22 | 3 | 15 |
| other14 | 14 | 2 | 14 |

| scene | runway | rsg done / v_entry / excess26 / inward20 | xp-l12-e46 done / v_entry / excess26 / inward20 |
|:--|:--|:--|:--|
| 09528162 | Y | n / +2.65 / +0.00 / -1.86 | Y / +4.43 / +0.00 / -1.86 |
| 132ae6cb | Y | Y / +6.50 / +0.00 / +0.11 | Y / +4.81 / +0.01 / +0.13 |
| 1e8e939e | Y | n / - / +8.02 / -2.48 | n / - / - / - |
| 22fa5c92 |  | n / +2.55 / +0.00 / - | Y / +2.28 / +0.16 / - |
| 268949d2 |  | n / +2.65 / +0.30 / - | Y / +0.30 / +0.42 / - |
| 2f227f2c | Y | n / +10.69 / +2.49 / -0.41 | Y / +4.42 / +0.00 / -0.07 |
| 302c5c99 | Y | n / +6.62 / +0.28 / -0.15 | Y / +3.73 / +0.00 / -0.05 |
| 3232fdb7 | Y | n / - / +0.06 / -1.89 | Y / +4.54 / +0.27 / +0.00 |
| 33c8976b |  | n / +2.55 / +0.27 / - | Y / +2.91 / +0.40 / - |
| 37d660b1 | Y | n / +4.68 / +0.02 / +0.18 | Y / +4.77 / +0.02 / +0.18 |
| 382a7247 | Y | Y / +8.95 / +0.09 / +0.06 | n / +4.56 / +0.01 / +0.05 |
| 3c2822d9 |  | Y / +6.48 / +0.88 / - | Y / +6.48 / +0.88 / - |
| 3ce13eed |  | n / +3.62 / +0.03 / - | Y / +3.32 / +0.02 / - |
| 403d558a | Y | n / +11.79 / +1.62 / -0.83 | Y / +4.78 / +0.00 / -0.83 |
| 68e53160 |  | n / +5.07 / +0.02 / - | Y / +3.63 / +0.05 / - |
| 6fd8cf30 | Y | n / +2.61 / +0.00 / -1.00 | n / - / - / - |
| 8a143cb2 |  | Y / +1.82 / +0.03 / - | Y / +1.83 / +0.03 / - |
| 94c1d722 | Y | n / +3.20 / +0.00 / -0.89 | Y / +3.63 / +0.05 / -0.31 |
| 9f393c8a |  | n / +5.86 / +0.15 / - | Y / +5.86 / +0.15 / - |
| a174faed | Y | n / +6.16 / +0.40 / -0.48 | n / +4.50 / +0.00 / +0.02 |
| a1b166cc |  | n / +3.81 / +0.00 / - | Y / +4.57 / +0.12 / - |
| a4d2f7cf | Y | n / +3.83 / +0.00 / +0.08 | Y / +4.04 / +0.00 / +0.00 |
| abd45a30 | Y | n / - / +0.00 / -0.12 | n / +4.70 / +0.00 / -0.00 |
| b40f4729 |  | n / +2.97 / +0.43 / - | Y / +1.16 / +0.15 / - |
| c4a58f83 |  | n / - / - / - | Y / +4.43 / +0.00 / +0.02 |
| ce161092 | Y | n / +8.06 / +1.19 / +0.01 | Y / +4.74 / +0.00 / -0.00 |
| d90fe123 | Y | Y / +8.34 / +1.45 / +0.02 | n / +4.80 / +0.00 / -0.01 |
| ddc3e8df |  | n / +2.58 / +0.08 / - | Y / +2.58 / +0.08 / - |
| e0f6ebe1 | Y | n / +7.64 / +1.40 / -0.90 | Y / +4.57 / +0.01 / -0.00 |
| e37aa159 | Y | n / +7.38 / +0.00 / -0.72 | n / +4.61 / +0.00 / -0.05 |
| e5e47fd2 | Y | n / +13.78 / +3.44 / -0.11 | Y / +5.51 / +0.00 / +0.02 |
| e848c843 | Y | n / +3.41 / +0.19 / -0.60 | Y / +3.22 / +0.26 / -0.19 |
| ea468790 |  | n / +3.21 / +0.25 / - | Y / +3.17 / +0.44 / - |
| eaba3f6a | Y | n / +12.51 / +7.04 / -0.47 | Y / +4.12 / +0.11 / -0.09 |
| f9e361aa | Y | n / +11.36 / +3.63 / +0.03 | Y / +4.68 / +0.00 / +0.00 |
| fd01e9ab |  | n / +3.65 / +0.01 / - | Y / +4.47 / +0.04 / - |

## Lines and the per-turn failure list (rf_expert.py report)

### Lines (pai)

- E2: completed 29 / 36 (line 29 / 36)
- E3 (control, 75 scenes scored by every label): xp-l12-e46 0.8116; rsg 0.7779; xp-l12-e46 - rsg +0.0337 (line >= -0.05)
- xp-l12-e46: scored 281, mean 0.7646, zeros 36, collision at fault 7, offroad 8
- rsg: scored 281, mean 0.5720, zeros 70, collision at fault 9, offroad 14
- xp-l12-e46 zeros 36: shared with rsg 21, new 15, removed 49; new zeros by reason: collision_at_fault 3, left_corridor_laterally 8, offroad 4
- new zeros whose last expert decision is at or past the human log's end station - 2 m (the evaluator's corridor / GT reference is a straight extension there; the expert does not stop where the human stopped): 9

### Rule expert xp-l12-e46: the registered turn set (pai, 36 turns)

- completed: **29 / 36**; rsg 5 / 36
- failure classes: collision 1, corridor 2, speed 4

| scene | side | deg | outcome | v_entry | v_log_entry | shape_ok | ref_rsg | chain_bad | score | zero | cls | yield_dec | yield_all | nudges |
|:--|:--|:--|:--|:--|:--|:--|:--|:--|:--|:--|:--|:--|:--|:--|
| 09528162 | R | -100 | completed | 4.43 | 4.52 | True | off | False | 1.0 |  |  | 0 | 0 | 4 |
| 132ae6cb | L | 86 | completed | 4.8 | 8.21 | True | completed | False | 0.159 |  |  | 49 | 95 | 0 |
| 1e8e939e | R | -49 | event | nan | 4.61 | False | event | False | 0.0 | offroad | corridor | 0 | 0 | 0 |
| 22fa5c92 | L | 91 | completed | 2.3 | 3.36 | True | event | False | 1.0 |  |  | 53 | 77 | 9 |
| 268949d2 | R | -88 | completed | 0.31 | 1.94 | True | off | True | 1.0 |  |  | 100 | 164 | 3 |
| 2f227f2c | R | -66 | completed | 4.42 | 4.41 | True | event | False | 0.498 |  |  | 0 | 0 | 0 |
| 302c5c99 | L | 163 | completed | 3.73 | 3.73 | False | event | False | 0.977 |  |  | 31 | 67 | 0 |
| 3232fdb7 | R | -87 | completed | 4.55 | 5.3 | True | event | False | 1.0 |  |  | 0 | 8 | 0 |
| 33c8976b | L | 107 | completed | 2.93 | 1.36 | False | event | False | 1.0 |  |  | 0 | 2 | 1 |
| 37d660b1 | R | -85 | completed | 4.77 | 5.51 | True | event | False | 0.494 |  |  | 36 | 135 | 7 |
| 382a7247 | L | 83 | event | 4.57 | 8.38 | False | completed | False | 0.0 | collision_at_fault | collision | 105 | 105 | 0 |
| 3c2822d9 | L | 81 | completed | 6.48 | 6.5 | False | completed | False | 0.818 |  |  | 0 | 120 | 6 |
| 3ce13eed | R | -100 | completed | 3.33 | 3.35 | True | event | False | 1.0 |  |  | 59 | 59 | 2 |
| 403d558a | R | -93 | completed | 4.78 | 6.19 | True | event | False | 0.0 | collision_at_fault |  | 17 | 98 | 10 |
| 68e53160 | L | 108 | completed | 3.62 | 4.55 | True | event | False | 1.0 |  |  | 34 | 56 | 0 |
| 6fd8cf30 | R | -88 | short_rear | nan | 4.52 | False | short_rear | False | 0.055 |  | speed | 0 | 174 | 1 |
| 8a143cb2 | R | -76 | completed | 1.82 | 1.74 | True | completed | False | 0.83 |  |  | 0 | 12 | 0 |
| 94c1d722 | R | -86 | completed | 3.62 | 3.48 | True | event | False | 0.918 |  |  | 0 | 0 | 0 |
| 9f393c8a | R | -84 | completed | 5.86 | 5.87 | False | event | False | 0.955 |  |  | 0 | 0 | 0 |
| a174faed | R | -90 | event | 4.5 | 4.56 | False | event | False | 0.0 | offroad | corridor | 12 | 73 | 0 |
| a1b166cc | L | 91 | completed | 4.57 | 4.89 | True | event | False | 1.0 |  |  | 35 | 127 | 1 |
| a4d2f7cf | L | 110 | completed | 4.04 | 4.1 | True | short_slow | False | 1.0 |  |  | 0 | 1 | 0 |
| abd45a30 | L | 90 | short_slow | 4.7 | 6.17 | False | short_rear | False | 0.964 |  | speed | 85 | 148 | 3 |
| b40f4729 | R | -89 | completed | 1.16 | 2.01 | False | off | False | 1.0 |  |  | 48 | 48 | 1 |
| c4a58f83 | R | -85 | completed | 4.41 | 7.77 | True | short_slow | False | 1.0 |  |  | 0 | 0 | 0 |
| ce161092 | L | 85 | completed | 4.74 | 5.28 | True | event | False | 1.0 |  |  | 0 | 40 | 0 |
| d90fe123 | L | 85 | short_rear | 4.79 | 6.04 | False | completed | False | 0.809 |  | speed | 0 | 63 | 0 |
| ddc3e8df | R | -66 | completed | 2.57 | 2.54 | True | event | False | 1.0 |  |  | 0 | 0 | 0 |
| e0f6ebe1 | R | -90 | completed | 4.57 | 4.72 | True | off | False | 1.0 |  |  | 0 | 0 | 4 |
| e37aa159 | L | 82 | short_rear | 4.61 | 8.77 | False | event | False | 0.665 |  | speed | 7 | 99 | 0 |
| e5e47fd2 | L | 95 | completed | 5.5 | 6.47 | True | event | False | 1.0 |  |  | 0 | 0 | 4 |
| e848c843 | R | -90 | completed | 3.21 | 3.09 | True | off | False | 1.0 |  |  | 0 | 0 | 2 |
| ea468790 | R | -83 | completed | 3.17 | 2.46 | True | off | False | 0.746 |  |  | 106 | 106 | 0 |
| eaba3f6a | R | -90 | completed | 4.13 | 4.27 | True | event | False | 0.814 |  |  | 0 | 71 | 3 |
| f9e361aa | R | -81 | completed | 4.68 | 5.25 | True | event | False | 0.958 |  |  | 0 | 58 | 2 |
| fd01e9ab | L | 98 | completed | 4.49 | 5.75 | True | event | False | 1.0 |  |  | 0 | 0 | 0 |
