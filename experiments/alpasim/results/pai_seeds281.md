# PAI track, 281 curated-validation scenes: P2H10S-F (two seeds) against base, paired (2026-10-11)

Descriptive read, not registered; it promotes nothing (same standing as decision 235, whose 60 scenes it extends to 281). All runs: Tokyo box,
`pai_eval.sh` / `pai_run.sh`, served configuration `JEV_VCONT=1.0 JEV_LEAD=1`, CONC 4, unharmonised renderer, one rollout per scene. Scenes:
the 281 of `/data/runs/alpasim/pai_full/s_{b1a,b1b,b1c,q40,b2,n1,n2,n3}.tsv`. Base is `P2H10-F-s0` (one base seed).
Read-out: `pai_eval_report.py --rows base=... s0=... s1=... --mean both=s0,s1 --pair s0-base s1-base both-base s1-s0`; its output is
[pai281/paired281.md](pai281/paired281.md) (with the throughput of every run dir), one row per scene in [pai281/paired281_scenes.tsv](pai281/paired281_scenes.tsv).

**Extended on 2026-10-11 (lane PAI-281): the last section, "Four S seeds, two base seeds", adds `P2H10S-F` seeds 2 and 3 and base seed 1 on all 281 scenes; its numbers supersede "one base seed" above.**

**Caveat on the absolute scores.** `pai_core.slots` feeds rendered 10 Hz frames to all history slots of checkpoints trained on warped frames
([../../body1/results/served_plan_length.md](../../body1/results/served_plan_length.md), a diagnosis of 2026-10-11 without a second reader, fix
not applied). Every row below is served that way, so the absolute means are low by an unknown amount; the paired differences have the same input
on both sides.

| Row | Scenes | Mean scene score | Zeros | at-fault collision | offroad | left corridor | other zero |
|---|--:|--:|--:|--:|--:|--:|--:|
| base `P2H10-F-s0` | 281 | 0.3746 | 144 | 23 | 42 | 80 | 1 |
| `P2H10S-F-s0` | 281 | 0.4629 | 119 | 18 | 33 | 70 | 0 |
| `P2H10S-F-s1` | 281 | **0.4837** | **110** | 14 | 29 | 72 | 0 |
| mean of the two seeds (per scene) | 281 | 0.4733 | 99 (zero in both) | | | | |

Zero kinds are the scorer's flags on zero-score scenes; two flags can be set on one scene, so a row's kinds can sum to more than its zeros.

| Paired difference | Scenes | Mean | Scene-bootstrap 95 % CI | Better / worse / within 0.01 |
|---|--:|--:|---|---|
| s0 - base | 281 | +0.0883 | [+0.0472, +0.1304] | 60 / 38 / 183 |
| s1 - base | 281 | +0.1091 | [+0.0655, +0.1536] | 62 / 49 / 170 |
| two-seed mean - base | 281 | +0.0987 | [+0.0594, +0.1394] | 69 / 47 / 165 |
| s1 - s0 | 281 | +0.0208 | [-0.0098, +0.0516] | 38 / 50 / 193 |

Bootstrap: 10 000 resamples of scenes, seed 0. The CI covers scene sampling only: there is one base seed, and decision 235 measured a
base-to-base spread of up to 0.029 on 60 served scenes; a second base seed was then run on 139 of the scenes (section below).

Where the difference comes from (decomposition of the paired mean over the 281 scenes):

| | base zero -> arm scores | arm zero <- base scored | both score | both zero |
|---|---|---|---|---|
| s0 - base | 37 scenes, +0.1154 | 12 scenes, -0.0276 | 125 scenes, +0.0005 | 107 |
| s1 - base | 45 scenes, +0.1382 | 11 scenes, -0.0278 | 126 scenes, -0.0013 | 99 |

As in decision 235, the whole difference is zero-score scenes becoming scored ones (net 25 and 34 scenes); on scenes both drivers pass the
score is unchanged. All three zero kinds are lower in both seeds. The two seeds are not separable (CI includes 0); 99 scenes are zeros in both
seeds, 93 of them also in base.

## Base seed 1 on 139 of the scenes

A second base seed, `P2H10-F-s1`, same served configuration, on the 139 scenes of `s_{b1a,b1b,b1c,q40}` (run in the idle window before the
training data landed; not part of the plan of this read). Output: [pai281/baseseeds139.md](pai281/baseseeds139.md), per scene
[pai281/baseseeds139_scenes.tsv](pai281/baseseeds139_scenes.tsv).

| Row (139 scenes) | Mean | Zeros | at-fault collision | offroad | left corridor |
|---|--:|--:|--:|--:|--:|
| base s0 | 0.3245 | 79 | 12 | 21 | 47 |
| base s1 | 0.3121 | 82 | 13 | 25 | 46 |
| `P2H10S-F-s0` | 0.4086 | 69 | 12 | 19 | 40 |
| `P2H10S-F-s1` | **0.4440** | **60** | 9 | 15 | 40 |

| Paired difference (139 scenes) | Mean | Scene-bootstrap 95 % CI |
|---|--:|---|
| base s1 - base s0 | -0.0124 | [-0.0596, +0.0353] |
| s0 - base s0 | +0.0841 | [+0.0193, +0.1503] |
| s0 - base s1 | +0.0965 | [+0.0283, +0.1632] |
| s1 - base s0 | +0.1195 | [+0.0544, +0.1873] |
| s1 - base s1 | +0.1319 | [+0.0683, +0.1970] |
| two-seed mean - two-seed base mean | +0.1080 | [+0.0528, +0.1646] |

The two base seeds differ by 0.012 (CI includes 0), the two `P2H10S-F` seeds by 0.035 on these scenes; each of the four arm-against-base pairs
is positive with a CI excluding 0, and the smallest of them (+0.084) is seven times the base-to-base difference. On these 139 scenes the at-fault
collision zeros of `P2H10S-F-s0` equal base (12 against 12 / 13); only s1 has fewer (9).

On the 139 scenes that had a base rollout before this run (the earlier subset read): base 0.3245, s0 0.4086 (+0.0841 [+0.0193, +0.1503]),
s1 0.4440 (+0.1195 [+0.0544, +0.1873]), two-seed mean +0.1018 [+0.0419, +0.1646].

Run dirs (Tokyo box, `/data/runs/alpasim/`): base `pai_full/runs/{full1_ab,base_q40_c1}` (139 scenes) and `pai_base/runs/*` (142), base seed 1 `pai_base_s1/runs/*` (139); s0
`pai_full/runs/p2h10s-f-s0_*`; s1 `pai2c/runs/s1_b1a_c4` (33) and `pai_s1/runs/*` (248).

## Four S seeds, two base seeds (2026-10-11)

Lane PAI-281. A descriptive, mechanical extension of the read above, pre-registered before any new score was read
([../plans/2026-10-11-pai281-seeds-prereg.md](../plans/2026-10-11-pai281-seeds-prereg.md)). It promotes nothing; decision 247 keeps its standing and
gets an addendum ([../../../research/decisions/247.md](../../../research/decisions/247.md)). Everything is as above (Tokyo box, served
configuration `JEV_VCONT=1.0 JEV_LEAD=1`, `PAI_KEYWARP` unset, unharmonised renderer, CONC 4, one rollout per scene, the same 281 scenes of
`pai_full/s_*.tsv`) except that two runs were added: afterwards the base has two seeds and `P2H10S-F` four on every one of the 281 scenes.

**Caveat on the absolute scores, unchanged (decision 247 point 3).** `pai_core.slots` feeds rendered 10 Hz frames to all history slots of checkpoints
trained on warped frames ([../../body1/results/served_plan_length.md](../../body1/results/served_plan_length.md), a diagnosis of 2026-10-11 without a
second reader, fix not applied; decision 252 tested the keyframe + warp slot switch on 33 scenes and could not separate on from off). Every row below
is served that way, so the absolute means are low by an unknown amount; the paired differences have the same input on both sides.

| Run | Checkpoint | Scenes | Chunks (CHUNK 36) |
|---|---|--:|---|
| base s1, rest | `P2H10-F-s1` | 142 = `s_b2`, `s_n1`, `s_n2`, `s_n3` (the ones base s1 had not run) | 4 (36 / 36 / 36 / 34), the same chunks base s0's own `pai_base` run used |
| S s2 | `P2H10S-F-s2` | 281 | 8 (7 x 36, 29) |
| S s3 | `P2H10S-F-s3` | 281 | 8, the same chunks as S s2 |

S s2 and s3 are the main-box checkpoints of decision 235's four seeds, pulled with `tokyo_pull.sh ckpt` and sha256-equal to the box's. One queue
(`scripts/pai_queue.sh`, run from a frozen clone at 36669eda, two cards, one PAI stack per card) walked the three runs. Result: 20 chunks, 704 of 704
scenes with a result, `inference_error` + `input_error` = 0 in every run dir, no zero without a scorer flag except one in base s0 (as before). Timing,
card time and one interruption are under "Run notes" below. Pre-registered pipeline check, made before the new rows were read: the old rows re-read with
the extended script reproduce decision 247 to the last digit (base s0 0.3746, S s0 0.4629, S s1 0.4837, two-seed mean 0.4733 with the same CIs on 281;
base s1 0.3121 against base s0 0.3245 on the 139, -0.0124 [-0.0596, +0.0353]; the "139 scenes" column of the halves table below).

### Rows (n = 281)

| Row | Mean scene score | Score 1 | Slow (0 < s < 1) | Zeros | at-fault collision | offroad | left corridor | other zero | Driver errors |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| base s0 | 0.3746 | 67 | 70 | 144 | 23 | 42 | 80 | 1 | 0 |
| base s1 | 0.3730 | 70 | 66 | 145 | 23 | 45 | 81 | 0 | 0 |
| S s0 | 0.4629 | 87 | 75 | 119 | 18 | 33 | 70 | 0 | 0 |
| S s1 | 0.4837 | 90 | 81 | 110 | 14 | 29 | 72 | 0 | 0 |
| S s2 | 0.4937 | 90 | 82 | 109 | 17 | 28 | 70 | 0 | 0 |
| S s3 | 0.4950 | 91 | 83 | 107 | 12 | 28 | 70 | 0 | 0 |
| base mean (B2) | 0.3738 | | | 128 (zero in every seed) | | | | | |
| S mean (S4) | 0.4838 | | | 89 (zero in every seed) | | | | | |
| S s0, s1 mean | 0.4733 | | | 99 (zero in every seed) | | | | | |

Zero kinds are the scorer's flags on zero-score scenes; two flags can sit on one scene, so a row's kinds can sum to more than its zeros. Score 1 =
perfect scene, slow = scored between 0 and 1. A mean row is the per-scene mean of its seeds and is zero on a scene only if every seed is zero.
The two base seeds are the same driver to the second decimal (0.3746, 0.3730); the four S seeds are 0.4629, 0.4837, 0.4937, 0.4950 (range 0.0321,
SD 0.0148, s0 lowest). All four S seeds have fewer zeros than both base seeds (107 to 119 against 144 and 145) and more perfect scenes (87 to 91
against 67 and 70). At-fault collision zeros are 12 to 18 for S against 23 and 23 for the base, offroad zeros 28 to 33 against 42 and 45, left-corridor
zeros 70 to 72 against 80 and 81.

### Paired differences, the requested set (n = 281, scene-bootstrap 95 % CI)

| Difference (a - b) | Scenes | Mean | Scene-bootstrap 95 % CI | Better / worse / within 0.01 |
|---|--:|--:|---|---|
| S s0 - base s0 | 281 | +0.0883 | [+0.0472, +0.1304] | 60 / 38 / 183 |
| S s1 - base s0 | 281 | +0.1091 | [+0.0655, +0.1536] | 62 / 49 / 170 |
| S s2 - base s0 | 281 | +0.1191 | [+0.0765, +0.1628] | 74 / 37 / 170 |
| S s3 - base s0 | 281 | +0.1204 | [+0.0753, +0.1656] | 67 / 44 / 170 |
| S s0 - base mean (B2) | 281 | +0.0891 | [+0.0522, +0.1275] | 75 / 30 / 176 |
| S s1 - base mean (B2) | 281 | +0.1099 | [+0.0720, +0.1491] | 72 / 43 / 166 |
| S s2 - base mean (B2) | 281 | +0.1199 | [+0.0816, +0.1593] | 88 / 27 / 166 |
| S s3 - base mean (B2) | 281 | +0.1212 | [+0.0815, +0.1618] | 79 / 37 / 165 |
| S mean (S4) - base mean (B2) | 281 | +0.1100 | [+0.0760, +0.1458] | 91 / 40 / 150 |
| base s1 - base s0 | 281 | -0.0016 | [-0.0339, +0.0313] | 28 / 55 / 198 |

Bootstrap: 10 000 resamples of scenes, seed 0, percentile CI, as above; it covers scene sampling only, the seed-to-seed variation is read from the rows
and from the seed pairs below. Better / worse / within count scenes by the mean-row difference.

Every S seed is above both base seeds and above their mean, each with a CI that excludes 0: the eight (S seed, base seed) differences run from +0.0883
(S s0 - base s0) to +0.1220 (S s3 - base s1), and the smallest is about 54 times the base's own seed difference (-0.0016 [-0.0339, +0.0313] on all 281
scenes). The four-seed S mean against the two-seed base mean is +0.1100 [+0.0760, +0.1458]. Decision 247's headline (two S seeds against base s0) was
+0.0987 [+0.0594, +0.1394]; with the two-seed base mean on the same two S seeds it is +0.0995 [+0.0652, +0.1352], so the second base seed barely moves it,
and S s2 and s3 (the two highest seeds) raise it to +0.1100.

### Fixed against broken, and where the difference comes from

Fixed = zero on the second side and scored on the first (zero -> scored), broken = the reverse. A part is the summed difference of those scenes divided
by 281, so the three parts add up to the mean difference. Against a single base seed the status of a scene is clean. For the pairs with a mean row the
table averages the decomposition over all (seed, seed) pairs (the mean of eight single-seed differences is exactly the difference of the means);
the scene-level version, where a mean row is zero only if every seed is zero, follows.

| Difference (a - b) | zero -> scored: scenes | part | scored -> zero: scenes | part | both scored: scenes | part | both zero: scenes |
|---|--:|--:|--:|--:|--:|--:|--:|
| S s0 - base s0 | 37 | +0.1154 | 12 | -0.0276 | 125 | +0.0005 | 107 |
| S s1 - base s0 | 45 | +0.1382 | 11 | -0.0278 | 126 | -0.0013 | 99 |
| S s2 - base s0 | 45 | +0.1386 | 10 | -0.0244 | 127 | +0.0048 | 99 |
| S s3 - base s0 | 49 | +0.1515 | 12 | -0.0294 | 125 | -0.0018 | 95 |
| S s0 - base mean (B2) (mean of 2 seed pairs) | 37.50 | +0.1124 | 12.00 | -0.0302 | 124.50 | +0.0068 | 107.00 |
| S s1 - base mean (B2) (mean of 2 seed pairs) | 44.00 | +0.1293 | 9.50 | -0.0241 | 127.00 | +0.0047 | 100.50 |
| S s2 - base mean (B2) (mean of 2 seed pairs) | 45.00 | +0.1332 | 9.50 | -0.0252 | 127.00 | +0.0119 | 99.50 |
| S s3 - base mean (B2) (mean of 2 seed pairs) | 47.50 | +0.1424 | 10.00 | -0.0255 | 126.50 | +0.0043 | 97.00 |
| S mean (S4) - base mean (B2) (mean of 8 seed pairs) | 43.50 | +0.1293 | 10.25 | -0.0262 | 126.25 | +0.0069 | 101.00 |
| base s1 - base s0 | 16 | +0.0477 | 17 | -0.0405 | 120 | -0.0088 | 128 |

| Difference (a - b), mean rows | zero -> scored: scenes | part | scored -> zero: scenes | part | both scored: scenes | part | both zero: scenes |
|---|--:|--:|--:|--:|--:|--:|--:|
| S s0 - base mean (B2) | 27 | +0.0815 | 18 | -0.0302 | 135 | +0.0378 | 101 |
| S s1 - base mean (B2) | 32 | +0.0955 | 14 | -0.0241 | 139 | +0.0385 | 96 |
| S s2 - base mean (B2) | 33 | +0.0992 | 14 | -0.0252 | 139 | +0.0460 | 95 |
| S s3 - base mean (B2) | 35 | +0.1071 | 14 | -0.0255 | 139 | +0.0396 | 93 |
| S mean (S4) - base mean (B2) | 44 | +0.0958 | 5 | -0.0106 | 148 | +0.0248 | 84 |

As in decisions 235 and 247 the difference is zero-score scenes becoming scored ones: over the eight (S seed, base seed) pairs 43.5 scenes are fixed on
average (+0.1293), 10.25 broken (-0.0262), 126.25 are scored by both (+0.0069) and 101 are zero in both; at scene level 44 scenes are zero in both base
seeds and scored by at least one S seed (19 of them by all four), 5 are scored by a base seed and zero in all four S seeds (the net of 25 to 38 fixed
scenes per pair reads the same way). Scenes scored on both sides carry +0.0069 of the +0.1100: against base s0 their part is -0.0018 to +0.0048,
against base s1 +0.0104 to +0.0190, and the base's own both-scored part (base s1 - base s0) is -0.0088, so on scenes the base already passes the S
seeds are not distinguishable from the base-to-base spread. The zero set itself is seed-sensitive in both arms: two seeds of the same arm differ in the
zero status of 19 to 33 scenes (base s0 / s1 33 = 16 + 17; the S seed pairs 19 to 32), about one scene in ten, which the net S effect (25 to 38 more
fixed than broken scenes per pair) exceeds.

### Seed to seed, and the two halves of the scenes

Supplementary, same method (the rest of the pairs, each with its decomposition, is in [pai281/seeds281.md](pai281/seeds281.md)):

| Difference (a - b) | Scenes | Mean | Scene-bootstrap 95 % CI | Better / worse / within 0.01 |
|---|--:|--:|---|---|
| S s0 - base s1 | 281 | +0.0899 | [+0.0497, +0.1302] | 80 / 21 / 180 |
| S s1 - base s1 | 281 | +0.1108 | [+0.0720, +0.1506] | 73 / 30 / 178 |
| S s2 - base s1 | 281 | +0.1207 | [+0.0803, +0.1624] | 86 / 20 / 175 |
| S s3 - base s1 | 281 | +0.1220 | [+0.0815, +0.1640] | 76 / 29 / 176 |
| S mean (S4) - base s0 | 281 | +0.1092 | [+0.0699, +0.1498] | 75 / 49 / 157 |
| S s0, s1 mean - base s0 | 281 | +0.0987 | [+0.0594, +0.1394] | 69 / 47 / 165 |
| S s0, s1 mean - base mean (B2) | 281 | +0.0995 | [+0.0652, +0.1352] | 81 / 38 / 162 |
| S s1 - S s0 | 281 | +0.0208 | [-0.0098, +0.0516] | 38 / 50 / 193 |
| S s2 - S s0 | 281 | +0.0308 | [+0.0021, +0.0613] | 45 / 32 / 204 |
| S s3 - S s0 | 281 | +0.0321 | [+0.0051, +0.0606] | 35 / 43 / 203 |
| S s2 - S s1 | 281 | +0.0100 | [-0.0171, +0.0367] | 63 / 20 / 198 |
| S s3 - S s1 | 281 | +0.0113 | [-0.0135, +0.0363] | 38 / 28 / 215 |
| S s3 - S s2 | 281 | +0.0013 | [-0.0269, +0.0294] | 25 / 62 / 194 |

S s0 is the lowest S seed; s2 and s3 are above it by about 0.03 with CIs that just exclude 0, the other S seed pairs are not separable. The spread of the
S seeds (0.0321) is larger than the base's (0.0016), so a two-seed S mean understates the four-seed mean by 0.0105 (0.4733 against 0.4838).

The 281 scenes are the 139 on which base s1 had already run (the read above; the old half) and the 142 it had not (the evidence decision 247 listed;
the new half). Every S seed is above the base mean with a CI excluding 0 in both halves, and the base's own difference changes sign:

| Difference (a - b) | 281 scenes | 139 scenes (base s1 had run them) | 142 scenes (base s1 new) |
|---|---|---|---|
| S mean (S4) - base mean (B2) | +0.1100 [+0.0760, +0.1458] | +0.1270 [+0.0719, +0.1834] | +0.0934 [+0.0514, +0.1389] |
| S s0 - base mean (B2) | +0.0891 [+0.0522, +0.1275] | +0.0903 [+0.0292, +0.1524] | +0.0879 [+0.0460, +0.1332] |
| S s1 - base mean (B2) | +0.1099 [+0.0720, +0.1491] | +0.1257 [+0.0662, +0.1870] | +0.0945 [+0.0477, +0.1450] |
| S s2 - base mean (B2) | +0.1199 [+0.0816, +0.1593] | +0.1402 [+0.0799, +0.2020] | +0.1001 [+0.0502, +0.1523] |
| S s3 - base mean (B2) | +0.1212 [+0.0815, +0.1618] | +0.1520 [+0.0904, +0.2145] | +0.0911 [+0.0411, +0.1439] |
| base s1 - base s0 | -0.0016 [-0.0339, +0.0313] | -0.0124 [-0.0596, +0.0353] | +0.0088 [-0.0343, +0.0541] |
| S mean (S4) - base s0 | +0.1092 [+0.0699, +0.1498] | +0.1209 [+0.0614, +0.1833] | +0.0978 [+0.0455, +0.1526] |
| S s0, s1 mean - base mean (B2) | +0.0995 [+0.0652, +0.1352] | +0.1080 [+0.0528, +0.1646] | +0.0912 [+0.0501, +0.1356] |

| Row | 281: mean / zeros | 139: mean / zeros | 142: mean / zeros |
|---|---|---|---|
| base s0 | 0.3746 / 144 | 0.3245 / 79 | 0.4237 / 65 |
| base s1 | 0.3730 / 145 | 0.3121 / 82 | 0.4325 / 63 |
| S s0 | 0.4629 / 119 | 0.4086 / 69 | 0.5160 / 50 |
| S s1 | 0.4837 / 110 | 0.4440 / 60 | 0.5226 / 50 |
| S s2 | 0.4937 / 109 | 0.4585 / 60 | 0.5282 / 49 |
| S s3 | 0.4950 / 107 | 0.4703 / 57 | 0.5192 / 50 |
| base mean (B2) | 0.3738 / 128 | 0.3183 / 72 | 0.4281 / 56 |
| S mean (S4) | 0.4838 / 89 | 0.4454 / 50 | 0.5215 / 39 |

The new half is easier for every row (0.42 to 0.53 against 0.31 to 0.47). The per-half tables are [pai281/seeds281_old139.md](pai281/seeds281_old139.md)
and [pai281/seeds281_new142.md](pai281/seeds281_new142.md); the scene lists of the halves are `pai281/scenes_base_s1_{old139,new142}.txt`.

### Statement written before the runs

Decision 247's direction (S above base, all of it from zero-score scenes becoming scored ones) was to be said to hold if S4 - B2 is positive with a CI
excluding 0, every S seed is above B2, and |base s1 - base s0| is smaller than the smallest S seed - B2 difference. All three hold: +0.1100 [+0.0760,
+0.1458]; +0.0891 to +0.1212, every CI excluding 0; 0.0016 against 0.0891. It is a descriptive read: nothing is promoted, the 10 Hz slot caveat stands,
and the numbers were recomputed from the per-scene table with an independent script (counts and parts identical, CI endpoints within 1e-4), not by a
second reader.

### Run notes

- **Time.** Launch 13:41:27 JST, DONE 19:32:57 JST: 5 h 51 min wall for 704 scenes (120 scenes per hour including the interruption; the cards ran at
  67.1 and 66.2 scenes per hour of busy time). Card time: 2 x 5 h 51 min = 11 h 42 min, of which 10 h 32 min in the 20 finished chunks (card 0: 11 chunks,
  382 scenes, 5 h 41 min; card 1: 9 chunks, 322 scenes, 4 h 51 min), 34 min in the two killed attempts below, 14 min both cards waiting for the resume
  and 21 min of card 1 idle at the end (the last chunk, 29 scenes, ran on card 0; chunks are indivisible). Per chunk and per run:
  [pai281/queue_timing.md](pai281/queue_timing.md) (`pai_queue_timing.py`).
- **Interruption.** At 16:51:09 JST the queue was paused on the user's order (a check of the APY10m10-AB-s0 submission image came first): the queue
  processes were killed and the compose projects removed, and at 16:57:30 the same launch script was run again, which re-walks the queue and skips
  finished chunks. Two chunks were in flight and lost, `s2` c006 (card 1, started 16:20:14; its 36 rollouts had all completed, it was in the
  evaluation phase) and `s3` c000 (card 0, started 16:48:00); both were rerun from scratch, c006 on the other card. The killed c006 attempt had already
  written its rollouts into the run dir, so its `results-summary.json` holds 72 records for 36 scenes; the two records of every scene have identical
  score and scorer flags (36 of 36), so the per-scene tables do not depend on which is read. That is a second datum, next to decision 252's, that a rerun
  of the same scene list reproduces to the last digit, here also across the two cards.
- **Exploratory (added after the first read; everything else in this section is on the list written before the runs).** The seed-pair decomposition of
  the pairs with a mean row (the scene-level one, which the plan lists, puts a scene that some seeds fix into "both scored": +0.0248 for S4 - B2, which
  is why the seed-pair version was added), the zero-status flips between seed pairs, the scene counts 44 / 5 / 19, the SD of the S seeds and the ratio of
  the smallest S - base difference to the base's own.
- **Code.** The queue ran from a frozen clone at 36669eda (it holds `pai_queue.sh` and the `pai_eval.sh` fix for `JOIN=1 CARDS=" "`, which the first
  attempt of the queue hit: a bare `wait` with no card hangs on the log `tee`; no stack had started). The read-out ran from the box checkout at
  bfeab949 (`pai_eval_report.py --detail --only`, added for this read; its default output is unchanged). The code of the new runs includes decision 252's
  `PAI_KEYWARP` slot switch (default off, left unset; 252 found the off path identical to the rollouts stored before the switch existed on 33 of 33
  scenes in each of three reruns), the rows made earlier were made before it existed.

### Commands and files

```bash
# launch (tmux window `pai281` on the Tokyo box): the three runs through one two-card queue
CARDS='0 1' CHUNK=36 CONC=4 SERVE='-e JEV_VCONT=1.0 -e JEV_LEAD=1' bash experiments/alpasim/scripts/pai_queue.sh /data/runs/alpasim/pai281_ext/queue \
  "base_s1_rest|P2H10-F-s1|$L/s_b2.tsv $L/s_n1.tsv $L/s_n2.tsv $L/s_n3.tsv" "s2|P2H10S-F-s2|$L/s_*.tsv" "s3|P2H10S-F-s3|$L/s_*.tsv"   # L=/data/runs/alpasim/pai_full
# read-out (P=/data/runs/alpasim, X=$P/pai281_ext/queue)
python3 experiments/alpasim/scripts/pai_eval_report.py \
  --rows base0="$P/pai_full/runs/full1_ab,$P/pai_full/runs/base_q40_c1,$P/pai_base/runs/*" base1="$P/pai_base_s1/runs/*,$X/base_s1_rest/runs/*" \
         s0="$P/pai_full/runs/p2h10s-f-s0_*" s1="$P/pai2c/runs/s1_b1a_c4,$P/pai_s1/runs/*" s2="$X/s2/runs/*" s3="$X/s3/runs/*" \
  --mean b2=base0,base1 s4=s0,s1,s2,s3 s01=s0,s1 \
  --pair s0-base0 s1-base0 s2-base0 s3-base0 s0-b2 s1-b2 s2-b2 s3-b2 s4-b2 base1-base0 s0-base1 s1-base1 s2-base1 s3-base1 s4-base0 s01-b2 s01-base0 \
         s1-s0 s2-s0 s3-s0 s2-s1 s3-s1 s3-s2 --detail --scenes seeds281_scenes.tsv --out seeds281.md      # + --only scenes_base_s1_old139.txt / _new142.txt
```

- [pai281/seeds281.md](pai281/seeds281.md): the full output (rows, every pair with CI and counts, the decompositions, the throughput of every run dir);
  [seeds281_old139.md](pai281/seeds281_old139.md), [seeds281_new142.md](pai281/seeds281_new142.md): the same on the two halves.
- [pai281/seeds281_scenes.tsv](pai281/seeds281_scenes.tsv): one row per scene, the score of every row (six seeds and the three mean rows).
- [pai281/queue_timing.md](pai281/queue_timing.md), [pai281/scenes_base_s1_old139.txt](pai281/scenes_base_s1_old139.txt),
  [pai281/scenes_base_s1_new142.txt](pai281/scenes_base_s1_new142.txt).
- Run dirs (Tokyo box, `/data/runs/alpasim/`): base s0, S s0, S s1 and base s1 (139) as in the lines above; base s1, rest
  `pai281_ext/queue/base_s1_rest/runs/*` (142); S s2 `pai281_ext/queue/s2/runs/*`; S s3 `pai281_ext/queue/s3/runs/*`; queue log, STATUS, `gpu.csv`
  in `pai281_ext/queue/`.
