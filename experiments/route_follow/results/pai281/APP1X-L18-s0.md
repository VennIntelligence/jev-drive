# PAI 281: APP1X-L18-s0 vs RSG-L18-s0 (stored read r281/rsg), gains 20, JEV_VCONT 1.0, JEV_LEAD 1

Run `/data/runs/route_night/v2pai/APP1X-L18-s0` (80d62482 approach course prereg Amendment 4 (EXPLORATORY): APP1X-L18-s0 (A1, W 1, seed 0) mechanism read on PAI 281, held H1 only), reference `/data/runs/route_night/r281/rsg`; rf_v2pai.sh read. Served memory check (rf_v2pai.py served):

- ffe5b17d session 5c89d8bc: 199 decisions, layout old (200), max |served - offline| 1.70e-04, mask differs at 0
- served memory = offline rebuild: 281 sessions, 55693 decisions, max |diff| 1.60e-01 (logged rounding 1e-5), mask differs at 0

## All 281 and the exam / control lists (rf_pai281.py; turn proxy = the 20 reference turn scenes)

Scenes scored by every label: 281; turn scenes (a scoring reference label turned 60-200 deg): 18 (left 10, right 8)

| label | mean | zeros | collision | offroad | corridor | handoff spins | turn done left | turn done right |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|
| rsg | 0.5720 | 70 | 9 | 14 | 47 | 2 | 10/10 | 8/8 |
| app1x-l18-s0 | 0.5097 | 76 | 16 | 32 | 28 | 29 | 2/10 | 7/8 |

## all281: 281 scenes

| label | mean | zeros | collision | offroad | corridor | rear-ended | spins | plan 2 s deficit m | vs rsg (95 % CI) |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|:--|
| rsg | 0.5720 | 70 | 9 | 14 | 47 | 77 | 2 | -0.60 | +0.0000 [+0.0000, +0.0000] |
| app1x-l18-s0 | 0.5097 | 76 | 16 | 32 | 28 | 85 | 29 | -0.99 | -0.0623 [-0.1083, -0.0163] |

## exam72: 72 scenes

| label | mean | zeros | collision | offroad | corridor | rear-ended | spins | plan 2 s deficit m | vs rsg (95 % CI) |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|:--|
| rsg | 0.3348 | 38 | 6 | 9 | 23 | 11 | 1 | -0.77 | +0.0000 [+0.0000, +0.0000] |
| app1x-l18-s0 | 0.3082 | 37 | 7 | 12 | 18 | 15 | 9 | -1.08 | -0.0266 [-0.1123, +0.0625] |

## control75: 75 scenes

| label | mean | zeros | collision | offroad | corridor | rear-ended | spins | plan 2 s deficit m | vs rsg (95 % CI) |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|:--|
| rsg | 0.7779 | 0 | 0 | 0 | 0 | 30 | 0 | -0.50 | +0.0000 [+0.0000, +0.0000] |
| app1x-l18-s0 | 0.6467 | 6 | 2 | 2 | 2 | 32 | 4 | -0.85 | -0.1312 [-0.1831, -0.0848] |

## Strict turns (36) and exam scenes (72): rf_pai_follow's turn rule on each label's own route (rf_v2pai.py turns / table)

### app1x-l18-s0 vs rsg: per scene

Scenes 90; strict turns = 36 (the identity's strict turns); exam turn scenes 72.


## Counts

- strict turns completed: rsg 5 / 36, app1x-l18-s0 11 / 36; gained 8 (268949d2, 3ce13eed, b40f4729, ddc3e8df, e0f6ebe1, ea468790, eaba3f6a, f9e361aa), lost 2 (132ae6cb, d90fe123)
- decision 293 shape gate (completed and kappa / dkappa / a_lat / entry speed within the class gate): rsg 1, app1x-l18-s0 8
- exam turn scenes: 72 (18 with a strict turn on the identity's route): completed rsg 3, app1x-l18-s0 4
- zeros over the 90 scenes: rsg 49, app1x-l18-s0 40; mean score rsg 0.3219, app1x-l18-s0 0.3619, paired +0.0400 [-0.0431, +0.1268]
- new zeros on the non-turn part (app1x-l18-s0 zero, rsg not, event before the entry or no strict turn): 8 (05855d57, 26004ca8, 2c263e19, 3fa5e06b, 4a863a41, 8120803d, bbc53aa8, e7db8971); offroad among them (stop rule >= 3): 7

| scene | set | side | deg | rsg | app1x-l18-s0 | app1x-l18-s0 where (m rel. entry) | app1x-l18-s0 v_0 | app1x-l18-s0 shape gate | switch on / dec (first, m rel. entry) |
|:--|:--|:--|--:|:--|:--|:--|--:|:--|:--|
| 09528162 | strict+exam | R | -100 | off | event (collision) | +6.0 | 2.3 | not completed |  /  |
| 132ae6cb | strict | L | +86 | completed | off |  | 5.0 | not completed |  /  |
| 1e8e939e | strict | R | -49 | event (collision) | off |  | 4.1 | not completed |  /  |
| 22fa5c92 | strict+exam | L | +91 | event (collision) | event (collision) | +5.6 | 3.3 | not completed |  /  |
| 268949d2 | strict | R | -88 | off | completed |  | 2.7 | pass |  /  |
| 2f227f2c | strict+exam | R | -66 | event (corridor) | event (corridor) | +1.5 | 5.2 | not completed |  /  |
| 302c5c99 | strict+exam | L | +163 | event (corridor) | event (corridor) | +3.4 | 5.6 | not completed |  /  |
| 3232fdb7 | strict | R | -87 | event (corridor) | short_slow |  |  | not completed |  /  |
| 33c8976b | strict | L | +107 | event (offroad) | event (offroad) | +6.8 | 2.8 | not completed |  /  |
| 37d660b1 | strict | R | -85 | event (corridor) | off |  | 5.6 | not completed |  /  |
| 382a7247 | strict+exam | L | +83 | completed (collision) | completed (collision) | +125.9 | 7.4 | fail: v_entry |  /  |
| 3c2822d9 | strict+exam | L | +81 | completed | completed |  | 6.5 | fail: v_entry |  /  |
| 3ce13eed | strict+exam | R | -100 | event (corridor) | completed |  | 3.0 | pass |  /  |
| 403d558a | strict | R | -93 | event (corridor) | event (offroad) | +10.6 | 5.9 | not completed |  /  |
| 68e53160 | strict | L | +108 | event (corridor) | event (corridor) | +22.7 | 5.7 | not completed |  /  |
| 6fd8cf30 | strict | R | -88 | short_rear | short_rear |  | 2.7 | not completed |  /  |
| 8a143cb2 | strict | R | -76 | completed | completed |  | 1.8 | pass |  /  |
| 94c1d722 | strict+exam | R | -86 | event (offroad) | short_slow |  | 2.8 | not completed |  /  |
| 9f393c8a | strict+exam | R | -84 | event (collision) | event (collision) | +19.5 | 5.9 | not completed |  /  |
| a174faed | strict | R | -90 | event (corridor) | short_rear |  | 3.1 | not completed |  /  |
| a1b166cc | strict+exam | L | +91 | event (corridor) | event (corridor) | +20.6 | 5.5 | not completed |  /  |
| a4d2f7cf | strict | L | +110 | short_slow | short_slow |  |  | not completed |  /  |
| abd45a30 | strict+exam | L | +90 | short_rear | short_rear |  |  | not completed |  /  |
| b40f4729 | strict | R | -89 | off | completed |  | 2.8 | pass |  /  |
| c4a58f83 | strict+exam | R | -85 | short_slow | short_slow |  |  | not completed |  /  |
| ce161092 | strict+exam | L | +85 | event (offroad) | short_rear |  |  | not completed |  /  |
| d90fe123 | strict+exam | L | +85 | completed | short_rear |  |  | not completed |  /  |
| ddc3e8df | strict | R | -66 | event (corridor) | completed |  | 2.6 | pass |  /  |
| e0f6ebe1 | strict+exam | R | -90 | off | completed |  | 4.4 | pass |  /  |
| e37aa159 | strict+exam | L | +82 | event (corridor) | event (corridor) | -6.5 |  | not completed |  /  |
| e5e47fd2 | strict | L | +95 | event (corridor) | off |  | 5.4 | not completed |  /  |
| e848c843 | strict+exam | R | -90 | off | short_slow |  | 2.5 | not completed |  /  |
| ea468790 | strict | R | -83 | off | completed |  | 3.3 | pass |  /  |
| eaba3f6a | strict | R | -90 | event (corridor) | completed |  | 4.6 | pass |  /  |
| f9e361aa | strict | R | -81 | event (corridor) | completed |  | 5.8 | fail: v_entry |  /  |
| fd01e9ab | strict+exam | L | +98 | event (corridor) | event (corridor) | +10.8 | 4.1 | not completed |  /  |
| 00fabf60 | +exam |  |  | no turn (corridor) | no turn (corridor) |  |  |  |  /  |
| 054b5901 | +exam |  |  | no turn (corridor) | no turn (corridor) |  |  |  |  /  |
| 05855d57 | +exam |  |  | no turn | no turn (collision) |  |  |  |  /  |
| 06b3e399 | +exam |  |  | no turn | no turn |  |  |  |  /  |
| 0feb3afa | +exam |  |  | no turn | no turn |  |  |  |  /  |
| 1089758a | +exam |  |  | no turn | no turn |  |  |  |  /  |
| 127c4f85 | +exam |  |  | no turn (corridor) | no turn (corridor) |  |  |  |  /  |
| 1a69f757 | +exam |  |  | no turn (collision) | no turn (offroad) |  |  |  |  /  |
| 2466266e | +exam |  |  | no turn (offroad) | no turn (corridor) |  |  |  |  /  |
| 26004ca8 | +exam |  |  | no turn | no turn (offroad) |  |  |  |  /  |
| 2b577fc2 | +exam |  |  | no turn | no turn |  |  |  |  /  |
| 2c263e19 | +exam |  |  | no turn | no turn (offroad) |  |  |  |  /  |
| 2eaf1872 | +exam |  |  | no turn (offroad) | no turn (corridor) |  |  |  |  /  |
| 300465f7 | +exam |  |  | no turn (offroad) | no turn (offroad) |  |  |  |  /  |
| 3fa5e06b | +exam |  |  | no turn | no turn (offroad) |  |  |  |  /  |
| 404541da | +exam |  |  | no turn | no turn |  |  |  |  /  |
| 415e950b | +exam |  |  | no turn | no turn |  |  |  |  /  |
| 44df49fe | +exam |  |  | no turn (corridor) | no turn (corridor) |  |  |  |  /  |
| 4a863a41 | +exam |  |  | no turn | no turn (offroad) |  |  |  |  /  |
| 4cb05d65 | +exam |  |  | no turn (offroad) | no turn (offroad) |  |  |  |  /  |
| 4cb7f7ac | +exam |  |  | no turn (corridor) | no turn (corridor) |  |  |  |  /  |
| 4f7cb072 | +exam |  |  | no turn | no turn |  |  |  |  /  |
| 5266c7c9 | +exam |  |  | no turn | no turn |  |  |  |  /  |
| 559d4ad6 | +exam |  |  | no turn (corridor) | no turn (corridor) |  |  |  |  /  |
| 5c4a277a | +exam |  |  | no turn | no turn |  |  |  |  /  |
| 66b15fbe | +exam |  |  | no turn | no turn |  |  |  |  /  |
| 6ee9ee78 | +exam |  |  | no turn (corridor) | no turn |  |  |  |  /  |
| 7381c9fa | +exam |  |  | no turn (offroad) | no turn (offroad) |  |  |  |  /  |
| 7750265d | +exam |  |  | no turn | no turn |  |  |  |  /  |
| 7d994ed3 | +exam |  |  | no turn (corridor) | no turn (corridor) |  |  |  |  /  |
| 8120803d | +exam |  |  | no turn | no turn (offroad) |  |  |  |  /  |
| 8b25a6ac | +exam |  |  | no turn | no turn |  |  |  |  /  |
| 98f88a4e | +exam |  |  | no turn (offroad) | no turn |  |  |  |  /  |
| 9b6c68b4 | +exam |  |  | no turn | no turn |  |  |  |  /  |
| 9f8acd00 | +exam |  |  | no turn (corridor) | no turn (corridor) |  |  |  |  /  |
| a2c5ae46 | +exam |  |  | no turn (corridor) | no turn |  |  |  |  /  |
| a68ad579 | +exam |  |  | no turn | no turn |  |  |  |  /  |
| b44a4a44 | +exam |  |  | no turn (collision) | no turn (collision) |  |  |  |  /  |
| ba40be62 | +exam |  |  | no turn (corridor) | no turn (corridor) |  |  |  |  /  |
| bbc1699d | +exam |  |  | no turn | no turn |  |  |  |  /  |
| bbc53aa8 | +exam |  |  | no turn | no turn (offroad) |  |  |  |  /  |
| bc309fd6 | +exam |  |  | no turn (corridor) | no turn (corridor) |  |  |  |  /  |
| bd97de34 | +exam |  |  | no turn (corridor) | no turn |  |  |  |  /  |
| c5da4975 | +exam |  |  | no turn | no turn |  |  |  |  /  |
| ca173de1 | +exam |  |  | no turn | no turn |  |  |  |  /  |
| cbd87234 | +exam |  |  | no turn (corridor) | no turn |  |  |  |  /  |
| cdf95fd1 | +exam |  |  | no turn (corridor) | no turn |  |  |  |  /  |
| d8bfaaf2 | +exam |  |  | no turn (offroad) | no turn (offroad) |  |  |  |  /  |
| dc401449 | +exam |  |  | no turn | no turn |  |  |  |  /  |
| de9db5dc | +exam |  |  | no turn (collision) | no turn (collision) |  |  |  |  /  |
| e7db8971 | +exam |  |  | no turn | no turn (offroad) |  |  |  |  /  |
| e904e9c0 | +exam |  |  | no turn (corridor) | no turn |  |  |  |  /  |
| f8ee31cb | +exam |  |  | no turn | no turn |  |  |  |  /  |
| fc9fde87 | +exam |  |  | no turn (corridor) | no turn (corridor) |  |  |  |  /  |
