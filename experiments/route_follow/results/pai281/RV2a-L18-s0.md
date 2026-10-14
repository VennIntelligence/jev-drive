# PAI 281: RV2a-L18-s0 vs RSG-L18-s0 (stored read r281/rsg), gains 20, JEV_VCONT 1.0, JEV_LEAD 1

Run `/data/runs/route_night/v2pai/RV2a-L18-s0` (7ea5c717 Turn-gain smoke passed (tgcheck clean, unswitched identity 2 / 2 bit-identical to r281/rsg); rf_v2pai read: all-281 paired list, exam counts over scenes both labels drove), reference `/data/runs/route_night/r281/rsg`; rf_v2pai.sh read. Served memory check (rf_v2pai.py served):

- served memory = offline rebuild: 281 sessions, 55693 decisions, max |diff| 1.38e-01 (logged rounding 1e-5), mask differs at 0
- ordered start vs projection start (55693 decisions): median |diff| 0.00 m, p90 0.68 m, >= 5 m at 2110; first valid slot ordered / projection differ at 4293 (ordered further at 3253)

## All 281 and the exam / control lists (rf_pai281.py; turn proxy = the 20 reference turn scenes)

Scenes scored by every label: 281; turn scenes (a scoring reference label turned 60-200 deg): 18 (left 10, right 8)

| label | mean | zeros | collision | offroad | corridor | handoff spins | turn done left | turn done right |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|
| rsg | 0.5720 | 70 | 9 | 14 | 47 | 2 | 10/10 | 8/8 |
| rv2a-l18-s0 | 0.5367 | 87 | 12 | 22 | 53 | 13 | 9/10 | 5/8 |

## all281: 281 scenes

| label | mean | zeros | collision | offroad | corridor | rear-ended | spins | plan 2 s deficit m | vs rsg (95 % CI) |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|:--|
| rsg | 0.5720 | 70 | 9 | 14 | 47 | 77 | 2 | -0.60 | +0.0000 [+0.0000, +0.0000] |
| rv2a-l18-s0 | 0.5367 | 87 | 12 | 22 | 53 | 72 | 13 | -0.64 | -0.0353 [-0.0583, -0.0136] |

## exam72: 72 scenes

| label | mean | zeros | collision | offroad | corridor | rear-ended | spins | plan 2 s deficit m | vs rsg (95 % CI) |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|:--|
| rsg | 0.3348 | 38 | 6 | 9 | 23 | 11 | 1 | -0.77 | +0.0000 [+0.0000, +0.0000] |
| rv2a-l18-s0 | 0.2766 | 47 | 7 | 14 | 26 | 7 | 10 | -0.94 | -0.0582 [-0.1228, +0.0092] |

## control75: 75 scenes

| label | mean | zeros | collision | offroad | corridor | rear-ended | spins | plan 2 s deficit m | vs rsg (95 % CI) |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|:--|
| rsg | 0.7779 | 0 | 0 | 0 | 0 | 30 | 0 | -0.50 | +0.0000 [+0.0000, +0.0000] |
| rv2a-l18-s0 | 0.7803 | 0 | 0 | 0 | 0 | 30 | 0 | -0.50 | +0.0024 [-0.0009, +0.0059] |

## Strict turns (36) and exam scenes (72): rf_pai_follow's turn rule on each label's own route (rf_v2pai.py turns / table)

### rv2a-l18-s0 vs rsg: per scene

Scenes 90; strict turns = 36 (the identity's strict turns); exam turn scenes 72.


## Counts

- strict turns completed: rsg 5 / 36, rv2a-l18-s0 5 / 36; gained 1 (b40f4729), lost 1 (132ae6cb)
- decision 293 shape gate (completed and kappa / dkappa / a_lat / entry speed within the class gate): rsg 1, rv2a-l18-s0 2
- exam turn scenes: 72 (18 with a strict turn on the identity's route): completed rsg 3, rv2a-l18-s0 3
- zeros over the 90 scenes: rsg 49, rv2a-l18-s0 60; mean score rsg 0.3219, rv2a-l18-s0 0.2579, paired -0.0640 [-0.1228, -0.0068]
- new zeros on the non-turn part (rv2a-l18-s0 zero, rsg not, event before the entry or no strict turn): 10 (06b3e399, 26004ca8, 3fa5e06b, 404541da, 4a863a41, 4f7cb072, 7750265d, 8120803d, bbc53aa8, e7db8971); offroad among them (stop rule >= 3): 7

| scene | set | side | deg | rsg | rv2a-l18-s0 | rv2a-l18-s0 where (m rel. entry) | rv2a-l18-s0 v_0 | rv2a-l18-s0 shape gate | switch on / dec (first, m rel. entry) |
|:--|:--|:--|--:|:--|:--|:--|--:|:--|:--|
| 09528162 | strict+exam | R | -100 | off | event (corridor) | +3.0 | 3.4 | not completed |  /  |
| 132ae6cb | strict | L | +86 | completed | off |  | 6.4 | not completed |  /  |
| 1e8e939e | strict | R | -49 | event (collision) | event (collision) | -9.0 |  | not completed |  /  |
| 22fa5c92 | strict+exam | L | +91 | event (collision) | event (collision) | +3.8 | 2.7 | not completed |  /  |
| 268949d2 | strict | R | -88 | off | event (corridor) | +21.3 | 1.4 | not completed |  /  |
| 2f227f2c | strict+exam | R | -66 | event (corridor) | event (corridor) | +0.9 | 10.8 | not completed |  /  |
| 302c5c99 | strict+exam | L | +163 | event (corridor) | event (corridor) | +1.3 | 6.7 | not completed |  /  |
| 3232fdb7 | strict | R | -87 | event (corridor) | event (corridor) | -3.2 |  | not completed |  /  |
| 33c8976b | strict | L | +107 | event (offroad) | event (offroad) | +4.0 | 2.4 | not completed |  /  |
| 37d660b1 | strict | R | -85 | event (corridor) | event (corridor) | +5.5 | 4.4 | not completed |  /  |
| 382a7247 | strict+exam | L | +83 | completed (collision) | completed (collision) | +123.6 | 8.8 | fail: alat,v_entry |  /  |
| 3c2822d9 | strict+exam | L | +81 | completed | completed |  | 6.5 | fail: v_entry |  /  |
| 3ce13eed | strict+exam | R | -100 | event (corridor) | event (corridor) | +4.3 | 3.2 | not completed |  /  |
| 403d558a | strict | R | -93 | event (corridor) | event (corridor) | +5.5 | 11.7 | not completed |  /  |
| 68e53160 | strict | L | +108 | event (corridor) | event (corridor) | +17.4 | 5.0 | not completed |  /  |
| 6fd8cf30 | strict | R | -88 | short_rear | short_rear |  | 2.1 | not completed |  /  |
| 8a143cb2 | strict | R | -76 | completed | completed |  | 1.8 | pass |  /  |
| 94c1d722 | strict+exam | R | -86 | event (offroad) | event (corridor) | +9.2 | 3.0 | not completed |  /  |
| 9f393c8a | strict+exam | R | -84 | event (collision) | event (collision) | +17.0 | 5.9 | not completed |  /  |
| a174faed | strict | R | -90 | event (corridor) | event (corridor) | +4.2 | 6.1 | not completed |  /  |
| a1b166cc | strict+exam | L | +91 | event (corridor) | event (corridor) | +6.8 | 4.0 | not completed |  /  |
| a4d2f7cf | strict | L | +110 | short_slow | short_slow |  | 3.7 | not completed |  /  |
| abd45a30 | strict+exam | L | +90 | short_rear | short_rear |  |  | not completed |  /  |
| b40f4729 | strict | R | -89 | off | completed |  | 2.9 | pass |  /  |
| c4a58f83 | strict+exam | R | -85 | short_slow | short_slow |  |  | not completed |  /  |
| ce161092 | strict+exam | L | +85 | event (offroad) | event (offroad) | +22.2 | 8.0 | not completed |  /  |
| d90fe123 | strict+exam | L | +85 | completed | completed |  | 8.3 | fail: v_entry |  /  |
| ddc3e8df | strict | R | -66 | event (corridor) | event (corridor) | +15.9 | 2.6 | not completed |  /  |
| e0f6ebe1 | strict+exam | R | -90 | off | off |  | 7.9 | not completed |  /  |
| e37aa159 | strict+exam | L | +82 | event (corridor) | event (corridor) | +14.5 | 7.4 | not completed |  /  |
| e5e47fd2 | strict | L | +95 | event (corridor) | event (corridor) | +6.3 | 13.8 | not completed |  /  |
| e848c843 | strict+exam | R | -90 | off | off |  | 3.3 | not completed |  /  |
| ea468790 | strict | R | -83 | off | event (corridor) | +8.6 | 3.1 | not completed |  /  |
| eaba3f6a | strict | R | -90 | event (corridor) | event (corridor) | +3.7 | 12.5 | not completed |  /  |
| f9e361aa | strict | R | -81 | event (corridor) | event (corridor) | +3.8 | 11.4 | not completed |  /  |
| fd01e9ab | strict+exam | L | +98 | event (corridor) | event (corridor) | +2.6 | 3.4 | not completed |  /  |
| 00fabf60 | +exam |  |  | no turn (corridor) | no turn (corridor) |  |  |  |  /  |
| 054b5901 | +exam |  |  | no turn (corridor) | no turn (corridor) |  |  |  |  /  |
| 05855d57 | +exam |  |  | no turn | no turn |  |  |  |  /  |
| 06b3e399 | +exam |  |  | no turn | no turn (corridor) |  |  |  |  /  |
| 0feb3afa | +exam |  |  | no turn | no turn |  |  |  |  /  |
| 1089758a | +exam |  |  | no turn | no turn |  |  |  |  /  |
| 127c4f85 | +exam |  |  | no turn (corridor) | no turn (corridor) |  |  |  |  /  |
| 1a69f757 | +exam |  |  | no turn (collision) | no turn (collision) |  |  |  |  /  |
| 2466266e | +exam |  |  | no turn (offroad) | no turn (offroad) |  |  |  |  /  |
| 26004ca8 | +exam |  |  | no turn | no turn (offroad) |  |  |  |  /  |
| 2b577fc2 | +exam |  |  | no turn | no turn |  |  |  |  /  |
| 2c263e19 | +exam |  |  | no turn | no turn |  |  |  |  /  |
| 2eaf1872 | +exam |  |  | no turn (offroad) | no turn (corridor) |  |  |  |  /  |
| 300465f7 | +exam |  |  | no turn (offroad) | no turn (offroad) |  |  |  |  /  |
| 3fa5e06b | +exam |  |  | no turn | no turn (offroad) |  |  |  |  /  |
| 404541da | +exam |  |  | no turn | no turn (offroad) |  |  |  |  /  |
| 415e950b | +exam |  |  | no turn | no turn |  |  |  |  /  |
| 44df49fe | +exam |  |  | no turn (corridor) | no turn (corridor) |  |  |  |  /  |
| 4a863a41 | +exam |  |  | no turn | no turn (offroad) |  |  |  |  /  |
| 4cb05d65 | +exam |  |  | no turn (offroad) | no turn (offroad) |  |  |  |  /  |
| 4cb7f7ac | +exam |  |  | no turn (corridor) | no turn (corridor) |  |  |  |  /  |
| 4f7cb072 | +exam |  |  | no turn | no turn (offroad) |  |  |  |  /  |
| 5266c7c9 | +exam |  |  | no turn | no turn |  |  |  |  /  |
| 559d4ad6 | +exam |  |  | no turn (corridor) | no turn (corridor) |  |  |  |  /  |
| 5c4a277a | +exam |  |  | no turn | no turn |  |  |  |  /  |
| 66b15fbe | +exam |  |  | no turn | no turn |  |  |  |  /  |
| 6ee9ee78 | +exam |  |  | no turn (corridor) | no turn |  |  |  |  /  |
| 7381c9fa | +exam |  |  | no turn (offroad) | no turn (offroad) |  |  |  |  /  |
| 7750265d | +exam |  |  | no turn | no turn (collision) |  |  |  |  /  |
| 7d994ed3 | +exam |  |  | no turn (corridor) | no turn (corridor) |  |  |  |  /  |
| 8120803d | +exam |  |  | no turn | no turn (offroad) |  |  |  |  /  |
| 8b25a6ac | +exam |  |  | no turn | no turn |  |  |  |  /  |
| 98f88a4e | +exam |  |  | no turn (offroad) | no turn (offroad) |  |  |  |  /  |
| 9b6c68b4 | +exam |  |  | no turn | no turn |  |  |  |  /  |
| 9f8acd00 | +exam |  |  | no turn (corridor) | no turn (corridor) |  |  |  |  /  |
| a2c5ae46 | +exam |  |  | no turn (corridor) | no turn (corridor) |  |  |  |  /  |
| a68ad579 | +exam |  |  | no turn | no turn |  |  |  |  /  |
| b44a4a44 | +exam |  |  | no turn (collision) | no turn (collision) |  |  |  |  /  |
| ba40be62 | +exam |  |  | no turn (corridor) | no turn (corridor) |  |  |  |  /  |
| bbc1699d | +exam |  |  | no turn | no turn |  |  |  |  /  |
| bbc53aa8 | +exam |  |  | no turn | no turn (offroad) |  |  |  |  /  |
| bc309fd6 | +exam |  |  | no turn (corridor) | no turn (corridor) |  |  |  |  /  |
| bd97de34 | +exam |  |  | no turn (corridor) | no turn (corridor) |  |  |  |  /  |
| c5da4975 | +exam |  |  | no turn | no turn |  |  |  |  /  |
| ca173de1 | +exam |  |  | no turn | no turn |  |  |  |  /  |
| cbd87234 | +exam |  |  | no turn (corridor) | no turn (corridor) |  |  |  |  /  |
| cdf95fd1 | +exam |  |  | no turn (corridor) | no turn |  |  |  |  /  |
| d8bfaaf2 | +exam |  |  | no turn (offroad) | no turn (offroad) |  |  |  |  /  |
| dc401449 | +exam |  |  | no turn | no turn |  |  |  |  /  |
| de9db5dc | +exam |  |  | no turn (collision) | no turn (collision) |  |  |  |  /  |
| e7db8971 | +exam |  |  | no turn | no turn (corridor) |  |  |  |  /  |
| e904e9c0 | +exam |  |  | no turn (corridor) | no turn (corridor) |  |  |  |  /  |
| f8ee31cb | +exam |  |  | no turn | no turn |  |  |  |  /  |
| fc9fde87 | +exam |  |  | no turn (corridor) | no turn (corridor) |  |  |  |  /  |
