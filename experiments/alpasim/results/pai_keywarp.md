# PAI-KW: the PAI driver's slot switch `PAI_KEYWARP` (2 Hz keyframes + warps), on against off (2026-10-11)

Pre-registered: [../plans/2026-10-11-pai-keywarp-prereg.md](../plans/2026-10-11-pai-keywarp-prereg.md) (committed in cf934e75, before any
closed-loop number). Descriptive read on 33 scenes; it promotes nothing. No second reader has checked the numbers.

**Question.** The `P2H10` family was trained on warped history (2 Hz keyframes, the other slots warped, decision 142). The PAI driver puts a
rendered 10 Hz frame into every slot; offline that shortens the plan by 10 %
([../../body1/results/served_plan_length.md](../../body1/results/served_plan_length.md), item 7). Does the mismatch cost closed-loop score,
and does the paired `P2H10S` (S) minus base difference of decisions 235 / 247 depend on the frame source?

**The switch.** `PAI_KEYWARP=1` (off by default; `lib/pai_core.py` `key_slots`, `lib/pai_driver.py`): only the frames at
t0 - 1.5 / 1.0 / 0.5 / 0 s are read, as keyframes, and the 8 slots come from `sh30_core.lattice_gpu`, the function the nuPlan driver calls
(called, not reimplemented; cold start = its rule on the oldest keyframe). **It discards 6 of every 10 rendered frames** and replaces them by
warps: this is the train / serve mismatch test, not the serving ceiling of the family on PAI. With the switch off `slots` is the function it was
(`tests/test_pai_keywarp.py`, and check (i) below).

## Equivalence checks (all before the full run)

Code `scripts/pai_keywarp_check.py`; files in [pai_keywarp/checks/](pai_keywarp/checks/).

| Check | Where | Result |
|---|---|---|
| (i) switch off = the file before the switch (`ef2f829d`), recorded PAI stream | Tokyo, driver image, 3 scenes of COL1's `a10` messages replayed through `pai_driver.Driver`, 596 decisions (551 warm, 45 cold) | slot frames, validity and real mask identical in 596 / 596 decisions |
| (ii) switch on = the nuPlan driver / training path on the same keyframes, same stream | same replay | pixels against `sh30_core.lattice` (CPU reference, `op_interp.synth_cpu`): 0 of 1 874 853 888 differ, validity equal in 596 / 596; plan against `sh30_core.Core.plan` (synth cpu and gpu) on the 551 warm decisions: 0.000 m in every decision. Scale of a real difference: plan on against plan off, same decision, median 1.86 m (largest distance over the 8 poses), 100 % of decisions over 0.03 m |
| (ii) on navtest tokens | GPU box, pool job, `jevdrive.run.Run` (`runs/alpasim/pai-kw-check/20261011-094228`, split `navsim/navtest@v1:19cf783cbae9`) | pixels against the CPU lattice: 0 of 1 887 436 800 differ. Plans against the bench's warp predictions of the same checkpoint: median 0.008 m, max 0.060 m, 9.3 % of tokens over 0.03 m; the same path on real frames against its cached-token plans: median 0.008, max 0.059, 6.5 %; wrong-model control (the other seed's bench predictions): median 0.144 m, 95 % over |
| (iii) offline 600 tokens of `lb_hq_navtestX` (the draw of served_plan_length.md), real 10 Hz frames as a stream through `pai_core.slots` | same job | off: arc / log 0.8949, ADE 1.350 m (cached real tokens: 0.8949, 1.350). **On: 0.9934, 0.608 m** (cached warp tokens and bench predictions: 0.9934, 0.608). Registered gate 0.994 +- 0.01 and 0.61 +- 0.03 m: pass |

The 6 cm maximum in the navtest row is not the frames (they are the same bytes): the driver encodes one token's 8 image pairs per call,
the cache was encoded in batches of 128, both in fp16. It is the level `sh30_check.py` found for the nuPlan driver (0.045 m max), and the
real-frame row, which involves no warp, shows the same.

## Protocol as run

Tokyo box, `pai_eval.sh` (one `pai_run.sh` stack per RTX 3090), scene list `s_b1a` (33 scenes, one simulator launch per run), CONC 4,
unharmonised renderer, image `jev-alpasim:p2h10-f-s0-6f3d05d1`, served configuration `JEV_VCONT=1.0 JEV_LEAD=1`, one rollout per scene: the
protocol of decision 247. The on runs add `-e PAI_KEYWARP=1` and nothing else. Checkpoints: base `P2H10-F-s0` / `-s1`, S `P2H10S-F-s0` / `-s1`.
Code: a frozen copy of 2f0dd515 (`/data/runs/alpasim/keywarp/code`), chain `scripts/pai_keywarp_chain.sh`, read `scripts/pai_keywarp_report.py`.

**Deviation from the lane brief.** The closed-loop runs did not go through `jevdrive.bench` and the GPU pool of the main box: `jevdrive.bench`
has no AlpaSim bench, the `s_b1a` protocol and its stored rollouts live on the Tokyo box (no pool there), and the main box has neither these
33 scenes nor scene-for-scene agreement with Tokyo (39 of 40, pai2_box.md). The existing runner was used as it is. Only check (ii) / (iii) on
navtest ran on the main box (pool job, `jevdrive.run.Run`). Wall time on the two Tokyo cards: 1 h 57 min for 7 runs (28 to 30 min each).

**The switch-off side.** Three reruns with the new code and the switch off against the rollouts stored before the switch existed
([pai_keywarp/offcheck.md](pai_keywarp/offcheck.md)): S s0 (stored run on the same list), base s0 (stored inside a 99-scene launch), base s1
(stored in chunks of 34 / 35): 33 of 33 scene scores identical in each. So the off path is unchanged in closed loop, the PAI simulator did not
depend on the scene list here, and the stored S s1 run (`pai2c/runs/s1_b1a_c4`, not rerun) is the off reference of S s1.

## Stage 1 (one scene, base s0)

Switch on: 199 `drive` calls, 199 inferences, 0 inference / input errors, `keywarp` in every record, 15 cold decisions then 8 valid slots;
in the dumps the 0 s slot of one second is the -1.0 s slot of the next (17 of 17), the slots between differ from both. Latency alone on a
card: 31.5 ms median, 33.2 ms p95 (off on the other card: 30.0, 55.4).

## Runs

33 scenes each ([pai_keywarp/runs.md](pai_keywarp/runs.md), per scene [pai_keywarp/scenes.tsv](pai_keywarp/scenes.tsv)). Zero kinds are
the scorer's flags on zero-score scenes. "Mean, scored" = mean over the scenes with a score above 0.

| Run | Mean scene score | Zeros | at-fault collision | offroad | left corridor | Mean, scored |
|---|--:|--:|--:|--:|--:|--:|
| base s0 off | 0.3114 | 20 | 3 | 5 | 13 | 0.790 |
| base s0 on | 0.2975 | 18 | 3 | 7 | 8 | 0.655 |
| base s1 off | 0.3313 | 19 | 3 | 5 | 11 | 0.781 |
| base s1 on | 0.3282 | 16 | 3 | 5 | 9 | 0.637 |
| S s0 off | 0.3905 | 17 | 5 | 3 | 10 | 0.806 |
| S s0 on | 0.4150 | 13 | 3 | 4 | 8 | 0.685 |
| S s1 off | 0.4175 | 16 | 4 | 3 | 11 | 0.811 |
| S s1 on | **0.4602** | **11** | 2 | 4 | 7 | 0.690 |

## (a) Switch on minus off, paired over the 33 scenes

`jevdrive.stats.paired`, 10 000 resamples of scenes, seed 0, percentile 95 % CI ([pai_keywarp/onoff.md](pai_keywarp/onoff.md)). An arm's
scene score is the mean of its two checkpoint seeds. Fixed = zero with the switch off, scored with it on; broken = the reverse; for a
two-seed row the counts are over the 66 (scene, seed) rollouts.

| Contrast | off | on | on - off | 95 % CI | Fixed / broken | Better / worse / within 0.01 |
|---|--:|--:|--:|---|---|---|
| base, two-seed mean | 0.3214 | 0.3129 | -0.0085 | [-0.1609, +0.1407] | 11 / 6 of 66 | 7 / 10 / 16 |
| base s0 | 0.3114 | 0.2975 | -0.0139 | [-0.1696, +0.1412] | 5 / 3 | 6 / 9 / 18 |
| base s1 | 0.3313 | 0.3282 | -0.0031 | [-0.1628, +0.1557] | 6 / 3 | 7 / 10 / 16 |
| S, two-seed mean | 0.4040 | 0.4376 | +0.0336 | [-0.0995, +0.1674] | 12 / 3 of 66 | 10 / 8 / 15 |
| S s0 | 0.3905 | 0.4150 | +0.0245 | [-0.1247, +0.1809] | 6 / 2 | 7 / 9 / 17 |
| S s1 | 0.4175 | 0.4602 | +0.0426 | [-0.1043, +0.1973] | 6 / 1 | 8 / 9 / 16 |
| all four checkpoints | 0.3627 | 0.3752 | +0.0125 | [-0.1072, +0.1309] | 23 / 9 of 132 | 11 / 11 / 11 |

Registered outcome: **CI includes 0** in every row. On 33 scenes the mismatch has no measurable closed-loop cost; the CI half-width is 0.12 to
0.16 (the plan guessed 0.08), so an effect below about 0.13 is not visible here and "no cost" is not claimed. The seeds agree in sign per arm
(base -0.014 / -0.003, S +0.025 / +0.043).

The mean hides a change of composition that is the same in all four checkpoints. Over the 132 rollouts: 23 zeros become scored and 9 scored
become zeros (net 14 fewer zeros: 72 -> 58), and the 51 rollouts scored on both sides lose score, 0.801 -> 0.619 (28 lower, 5 higher by more
than 0.01, median difference -0.044). By the scorer's flag of the off rollout: left corridor 41 -> 16 scored, 23 still left corridor;
at-fault collision 15 -> 7 scored; offroad 16 -> 14 still offroad; scored 60 -> 6 offroad, 2 collision, 1 left corridor. Per seed the three
parts of the mean are about +0.12 to +0.14 (fixed), -0.02 to -0.07 (broken), -0.06 to -0.08 (both scored): they cancel. Why the score of
scenes that pass on both sides drops was not looked at (the score of a scene without a zero flag is its progress score; the on plans are 10 %
longer offline).

## (b) S minus base under each frame source

Same bootstrap ([pai_keywarp/sbase.md](pai_keywarp/sbase.md)). Fixed / broken: base zero and S scored / the reverse.

| Contrast | base | S | S - base | 95 % CI | Fixed / broken | Better / worse / within 0.01 |
|---|--:|--:|--:|---|---|---|
| off, two-seed means | 0.3214 | 0.4040 | +0.0827 | [-0.0344, +0.2037] | 9 / 3 of 66 | 7 / 5 / 21 |
| off, s0 - s0 | 0.3114 | 0.3905 | +0.0791 | [-0.0554, +0.2200] | 5 / 2 | 6 / 5 / 22 |
| off, s1 - s1 | 0.3313 | 0.4175 | +0.0862 | [-0.0318, +0.2119] | 4 / 1 | 8 / 4 / 21 |
| on, two-seed means | 0.3129 | 0.4376 | **+0.1247** | [+0.0366, +0.2331] | 10 / 0 of 66 | 11 / 5 / 17 |
| on, s0 - s0 | 0.2975 | 0.4150 | +0.1175 | [+0.0251, +0.2303] | 5 / 0 | 8 / 6 / 19 |
| on, s1 - s1 | 0.3282 | 0.4602 | +0.1319 | [+0.0350, +0.2469] | 5 / 0 | 11 / 5 / 17 |
| difference of differences (on - off), two-seed means | | | +0.0420 | [-0.1122, +0.1950] | | 12 / 8 / 13 |
| difference of differences, seed 0 / seed 1 | | | +0.0384 / +0.0457 | [-0.1264, +0.2013] / [-0.1245, +0.2175] | | |

Registered outcome: **the CI of the difference of differences includes 0 and S - base has the same sign under both frame sources**: S's effect
is stable under the frame source within the resolution of 33 scenes. Under the switch the S - base CI excludes 0 in both seeds and no scene goes
from scored to zero; with the switch off it does not exclude 0 on these 33 scenes (decision 247 reads +0.099 [+0.059, +0.139] on 281). On both
sides the difference is zeros becoming scored; the scenes both score contribute at most 0.004 to the mean difference.

## Latency

`total_ms` of every inference in `drive.jsonl`, CONC 4, driver on the renderer's card ([pai_keywarp/latency.md](pai_keywarp/latency.md)).
The track asks for a decision every 100 ms.

| | p50 | p95 | p99 | Over 100 ms | Warm stage medians: slots / encode / policy |
|---|--:|--:|--:|--:|---|
| off (5 runs incl. stored) | 49.0 to 51.3 | 97.2 to 104.0 | 133 to 153 | 4.4 to 5.9 % | 0.3 / 32 / 13 |
| on (4 runs) | 57.5 to 59.0 | 81.2 to 86.2 | 117 to 119 | 2.6 to 3.3 % | 8.2 / 33 / 12 |

The warp adds 8 ms to the median call under four concurrent rollouts (3.2 ms alone on a card; `slots` includes the wait for the card, the
warp runs under the driver's GPU lock as in the nuPlan driver). The tail is not worse: p95 and the share over 100 ms are lower with the
switch on in every run; the cause was not examined. The latency budget is not blown by the switch; it is already missed on 4 to 6 % of
calls without it at CONC 4 on a card shared with the renderer.

## What this does and does not say

- Verified: the switch does what it claims (checks i to iii); with it off nothing changes (unit test, 596 replayed decisions, three
  closed-loop reruns identical to the stored rollouts).
- Measured, 33 scenes, 2 checkpoint seeds per arm: no on - off difference outside the CI in any arm; S - base keeps its sign and the
  difference of differences is inside the CI.
- Not claimed: that the mismatch is free (resolution about 0.13); that the switch is the ceiling of warp-trained checkpoints on PAI (it
  discards 6 of 10 real frames); anything about why scored scenes score lower under the switch.
- The alternative fix, mixed warp + real-frame training of the family, is written down in the plan and not started. Its registered decision
  point, the absolute on - off change, reads "CI includes 0": no closed-loop evidence here for retraining the family for this mismatch.

Run dirs (Tokyo, `/data/runs/alpasim/keywarp/`): `{base,S}-s{0,1}-{on,off}/runs/*` (no `S-s1-off`), `stage1/`, `check/`; stored references
`pai_full/runs/{p2h10s-f-s0_b1a,full1_ab}`, `pai2c/runs/s1_b1a_c4`, `pai_base_s1/runs/*`. Compact per-run files: `pai_keywarp/<run>.json`.
