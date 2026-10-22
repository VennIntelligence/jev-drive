# LOG-SYNC: how far our models are from the log car at the same sim time on PAI, and whether it goes with failures (2026-10-22)

CPU diagnostic on stored reads, no GPU, no rendering, no training. Prereg (committed before the numbers, 8518e4cd):
[plans/2026-10-22-log-sync-prereg.md](../plans/2026-10-22-log-sync-prereg.md). Decision 448.

**Why it matters.** The PAI runtime sends the sim time as the render frame time, and NuRec draws the background as it was
reconstructed at that time. When the ego camera is far from where the recording car was at that time, the image degrades
([alpasim_human/results/render_lag.md](../../alpasim_human/results/render_lag.md): 52 dB, falling to 18-25 dB at 16-36 m of
log-camera displacement; lateral 1-2 m at the right time is clean). This happens whether the ego is behind the log car or ahead of it.

**Answer.** Offsets are common: about a quarter of all driving steps are more than 10 m off. They do not go with zeros, though,
and they show no blur-then-slower loop. Under the prereg lines this is "common but not enriched": a property of the track that
every model shares.
- **How common.** 22-25 % of pre-failure steps are more than 10 m from the log car along the path, and 57-60 % of scenes get there at some point.
- **Ahead vs behind.** Ahead is at least as common as behind: R6DAG drives faster than the human more often than slower.
- **Zeros are not enriched.** In the 3 s before the scored failure, 33-36 % of zeros are more than 10 m off, against 28-33 % of
  ok scenes over the same time windows.
- **Partials.** The large behind offsets sit on the slow and rear-hit partials, where being behind is the failure itself.
- **No loop.** After falling behind, the ego on average catches up rather than slows further.

The result: no arm is opened. A GPU calibration on 2-3 scenes from the ranked list below would tell how bad the image actually is
at these offsets. It is proposed, not run.

## Inputs and method

- Reads: PAI 281, ACC + GAP, Harmonizer off. These are the same roots that spin_events.py scanned: R6DAG-L18-s0
  (`round6/dag/pai281/R6DAG-L18-s0-ag`), R6DAG-L18-s1 (`.../R6DAG-L18-s1-ag`) and NTV0-L18-s0 (`defect_fix/pai_accgap/NTV0-L18-s0`);
  281 / 281 / 280 rollouts.
- Ego per step: the controller CSV (10 Hz, rollout local frame, rear axle; the same file spin_events.py reads), via the rf_tax
  extracts `$DATA_DIR/runs/pai_fix/tax/<tag>-281ag.pkl`.
- Log car: the same pkl's `log`, i.e. the scene's recorded rig trajectory. On 6 rollouts with a stored rollout.asl it equals
  `rollout_metadata.ego_rig_recorded_ground_truth_trajectory` to 0.0000 m (the asl is kept only for about 50 rollouts per read,
  so the pkl is the full source). Handoff check: |ds| at t <= 1.5 s (force-GT) p99 = 0.02 m in all three reads.
- Outcome: pai_fix facts (score, fclass, t_ev_s). The R6DAG classes come from
  [round7/results/loss_map.csv](../../round7/results/loss_map.csv).
- Definitions: as in the prereg.
  - ds = s_e - s_log(t): along-track offset on the log path; behind < 0.
  - tau = t_log(s_e) - t: time equivalent.
  - g: camera distance; v_log: the log car's speed.
  - pre-failure = t <= t_ev.
  - Steps with |d| > 8 m off the log path (4-6 %) are left out.
- Code: [scripts/log_sync.py](../scripts/log_sync.py) (extract, then `--report`). Small outputs are in [log_sync/](log_sync/):
  - `scenes.csv`: per read x scene, the max offset with its time, sign, tau and v_log, plus the largest moving-log offset.
  - `zeros.csv`: per zero.
  - `ranked.csv`
  - `summary.json`

  Per-step arrays: box `$DATA_DIR/runs/alpasim/log_sync/steps_<read>.npz`.

## Distribution (pre-failure steps on the log path)

| read | steps | p50 / p90 / p99 \|ds\| m | > 5 m | > 10 m | > 20 m | behind > 10 / > 20 | ahead > 10 / > 20 | scenes reaching > 5 / 10 / 20 m |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|
| R6DAG-s0 | 48 059 | 2.5 / 23.4 / 68.8 | 38.3 % | 23.8 % | 12.2 % | 9.5 % / 4.9 % | 14.4 % / 7.3 % | 81 % / 59 % / 37 % |
| R6DAG-s1 | 48 772 | 2.6 / 25.0 / 67.5 | 38.6 % | 24.6 % | 13.4 % | 9.5 % / 5.4 % | 15.1 % / 8.0 % | 81 % / 60 % / 41 % |
| NTV0-s0 | 47 962 | 2.5 / 22.8 / 81.8 | 36.2 % | 22.5 % | 11.5 % | 11.9 % / 6.3 % | 10.6 % / 5.2 % | 76 % / 57 % / 32 % |

**Time equivalent at |ds| > 10 m.**
- Behind: median -1.8 to -1.9 s (p90 -4.0 to -4.5 s).
- Ahead: median +2.1 s for R6DAG, +2.5 s for NTV0 (p90 +5.3 to +6.3 s).
- 8 % of ahead steps are past the end of the log path, where the log car never goes.

**By stratum** (share of scenes reaching > 10 m, R6DAG-s1):

| stratum | n | share reaching > 10 m | of them behind / ahead |
|:--|--:|--:|--:|
| straight | 77 | 82 % | 27 % / 58 % |
| turn | 36 | 70 % | 32 % / 43 % |
| launch | 37 | 65 % | 24 % / 46 % |
| stop/lead | 128 | 45 % | 20 % / 24 % |

R6DAG is ahead more than NTV0 on straights (58 % against 34 %).

**Reading.** At the probe's anchors, about 16 m behind at 8 m/s already costs 25 dB. A quarter of the steps the models are scored
on are therefore rendered from a camera far from the recording camera. The mean offset is small (median 2.5 m); the tail is long.

## Relation to outcome

Scene level (pre-failure max |ds| > 10 m; in brackets the scene went more than 10 m behind / more than 10 m ahead; median max |ds|):

| class | R6DAG-s0 | R6DAG-s1 | NTV0-s0 |
|:--|--:|--:|--:|
| ok (>= 0.98) | 199: 61 % (21 / 44), 14.6 m | 211: 63 % (20 / 45), 15.4 m | 184: 57 % (24 / 34), 11.2 m |
| lost (< 0.98) | 82: 55 % (38 / 22), 11.6 m | 70: 54 % (37 / 20), 11.8 m | 96: 58 % (39 / 22), 12.1 m |
| slow partial | 6: 100 % (100 / 0), 98.9 m | 6: 100 % (100 / 0), 71.0 m | 11: 91 % (91 / 9), 72.5 m |
| rear-hit, moving | 13: 92 % (92 / 23), 31.2 m | 13: 92 % (92 / 15), 39.0 m | 26 (all rear-hit): 88 % (88 / 4), 36.5 m |
| rear-hit, launch | 10: 70 % (60 / 20), 15.1 m | 8: 50 % (50 / 0), 10.7 m | (in the row above) |
| stop overshoot (zero) | 20: 30 % (5 / 25), 7.8 m | 19: 37 % (5 / 32), 7.6 m | 23: 43 % (4 / 39), 9.7 m |
| turn inside offroad (zero) | 10: 50 % (0 / 50), 9.5 m | 7: 43 % (0 / 43), 7.7 m | - |

This raw comparison is biased: ok scenes count 20 s, zeros only up to their failure. The time-matched control below decides.

**Time-matched control** (registered line: zeros at least 10 pp above ok scenes and a ratio of at least 2). Each zero's
[t_ev - 3, t_ev] window against the ok scenes over the same window, share with |ds| > 10 m (behind / ahead):

| read | zeros | ok scenes, same windows |
|:--|--:|--:|
| R6DAG-s0 | 36 % (13 / 23) | 33 % (7 / 26) |
| R6DAG-s1 | 33 % (7 / 26) | 33 % (7 / 26) |
| NTV0-s0 | 36 % (7 / 29) | 28 % (8 / 20) |

**Not enriched.** The gap is at most 8 pp and the ratio at most 1.3. At the failure time the median ds of zeros is +1 to +5 m
(overshoot +3 to +6 m, turn-inside offroad +1 to +2 m). Turn-inside zeros are never behind. Where they are off, they are ahead:
the model enters the curve faster than the human did. That is the known over-speed, and the offset is a consequence of it.

**Partials.** The slow and rear-hit partials are behind by construction: the score charges progress, and the replayed follower
drives into an ego that lags its log position. Three slow partials from the loss map show how the offset builds up
(first crossing of -10 m: ego / log speed):
- **bbc53aa8** (R6DAG 0.74; NTV0 1.0). R6DAG crosses -10 m at 8.7 s at 25.1 / 34.8 m/s and keeps slowing (21 m/s at -20 m,
  9.6 s); it ends 285 m behind. NTV0 crosses at 9.0 s at 27.5 / 34.7 m/s, speeds back up to 28.3 m/s and ends 39 m behind.
- **478d6ec4** (all three about 0.90). Crosses -10 m at 9.4 s at 25.4 / 28.2 m/s and ends 152-155 m behind.
- **2a0efada** (R6DAG-s0 0.81). Crosses at 10.4 s at 6.7 / 15.1 m/s.

In every one of these the speed fell before the offset grew. Whether the degraded image then keeps the ego slow is what the GPU
check on bbc53aa8 can answer.

## Subsequent speed (descriptive; confounded, because being behind is caused by slowness)

The table shows the change of v_e - v_log over the next 3 s (m/s) after a step where the log car moves (v_log >= 3, t >= 1.7 s,
t + 3 s still pre-failure). Each cell is mean [n]; in brackets the subset with v_e / v_log in [0.8, 1.2].

| ds bin | R6DAG-s0 | R6DAG-s1 | NTV0-s0 |
|:--|--:|--:|--:|
| < -20 m | +0.96 [1337] (+0.63) | +0.63 [1546] (+0.15) | +0.80 [1904] (+0.92) |
| -20..-10 | +0.10 [1575] (-0.01) | +0.29 [1517] (+0.13) | +0.27 [1849] (+0.10) |
| -10..-5 | +0.51 [2080] (+0.09) | +0.44 [2024] (+0.11) | +0.19 [2239] (+0.00) |
| -5..5 | +0.29 [16902] (+0.19) | +0.30 [17080] (+0.21) | +0.13 [17938] (+0.05) |
| 5..10 | -0.38 [2945] (-0.48) | -0.27 [3063] (-0.30) | -0.26 [2287] (-0.21) |
| 10..20 | -1.04 [2526] (-1.20) | -0.75 [2601] (-0.82) | -0.37 [1665] (-0.22) |
| > 20 m | -2.00 [2875] (-0.94) | -1.93 [3180] (-1.07) | -2.28 [1937] (-1.31) |

The registered loop signature was more relative deceleration of at least 0.5 m/s in the > 10 m behind bins than in the < 5 m bin.
It is absent: behind the log, the ego on average gains speed on the log car; ahead, it loses speed.
- Ahead: much of this is the log car speeding up after the ego has passed its stop or slow-down point.
- Behind: the ego's own ACC set speed pulling it back.

On average there is no blur -> slower loop. Single scenes like bbc53aa8 (R6DAG) stay possible and are not excluded by this table.

## Ranked scenes / times for a later GPU re-render check (not rendered)

The ranking uses moving-log steps before the failure, taking the largest camera distance per scene; 14 scene / time pairs are listed.
- Zeros give the most informative moments (just before the failure).
- Large-lag partials give the extreme cases.

| # | scene | read | t s | ds m | tau s | v_log m/s | outcome (R6DAG-s0 / s1 / NTV0) | why |
|--:|:--|:--|--:|--:|--:|--:|:--|:--|
| 1 | bbc53aa8 | R6DAG-s0 | 9.6 / 12 / 19.8 | -20 / ... / -285 | to -8.3 | 34.8 / 15.6 | 0.74 / 0.74 / 1.0 slow | ruler a >= 1 item; R6DAG slows on after falling behind, NTV0 recovers |
| 2 | 6e6cbe67 | R6DAG-s1 | 8.2 | -67.6 | -2.9 | 24.0 | 0 / 0 / 0.83 side / front collision | farthest behind before a zero |
| 3 | 596af0c7 | R6DAG-s0 | 10.6 | -48.2 | -1.8 | 26.8 | 0 / 0.62 / 0.62 offroad straight | behind at speed at the failure |
| 4 | a68ad579 | R6DAG-s0 | 19.3 | -48.7 | -3.6 | 12.6 | 0 offroad junction straight | behind at the failure |
| 5 | 404cc92f | R6DAG-s0 | 12.4 | +45.6 | +3.5 | 13.0 | 0 drift (t_ev 17.3) | farthest ahead before a zero |
| 6 | 7cfa81de | NTV0-s0 | 12.9 | +36.4 | +3.1 | 11.5 | NTV0 0 (zero-other) | ahead at the failure |
| 7 | 9f8acd00 | R6DAG-s1 | 10.2 | +29.2 | past log end | 9.5 | 0 overshoot | overshoot, the region the log car never reached |
| 8 | 7dc59d0d | R6DAG-s1 | 13.6 | +28.8 | +1.1 | 27.6 | 0 drift | ahead at high speed |
| 9 | 0feb3afa | R6DAG-s0 | 14.4 | +27.2 | +1.1 | 24.7 | 0 front collision | ahead at high speed |
| 10 | 478d6ec4 | R6DAG-s1 | 11.6 / 19.9 | -20 / -155 | to -5.5 | 28.6 / 14.7 | 0.90 all three, slow | shared slow partial |
| 11 | 98f88a4e | R6DAG-s1 | 19.9 | -149.8 | -6.6 | 14.2 | 0.57 all three, rear-hit | shared |
| 12 | 132ae6cb | R6DAG-s1 | 19.9 | -145.2 | -6.7 | 17.5 | 0.10 all three, rear-hit launch | shared |
| 13 | b65e55e9 | NTV0-s0 | 19.9 | -200.5 | -7.8 | 17.2 | 1 / 1 / 0.48 rear-hit | NTV0 only |
| 14 | e904e9c0 | R6DAG-s0 | 10.7 | -31.6 | -2.5 | 12.7 | 0 side collision | behind at the failure |

Proposed check (GPU, not run):
- Scenes: 2-3 of the list (bbc53aa8 at 9.6 and 12 s, 6e6cbe67 at 8.2 s, 404cc92f at 12.4 s).
- Renders: render the stored request as sent, and the same request with the frame time set to the log time whose pose is nearest
  the ego (render_lag's `lag_actors` condition).
- Measure: PSNR / sharpness between the two.

Only if the as-sent image is clearly degraded on a scene where the model then differs would a render-side diagnostic read follow.
That read would be a runtime change and diagnostic only, never in the submission serve (no-serve-hacks).

## Limits

- One read per model, one seed for NTV0.
- Offsets are along the log path. Steps more than 8 m to the side (offroad, other route) are excluded.
- "Degraded" is inferred from the two-scene probe, not measured here.
- The speed table is confounded by construction (the offset is integrated speed difference) and is descriptive only.
