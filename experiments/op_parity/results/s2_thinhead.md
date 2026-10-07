# System 2 as a thin head on VLM features (WOD-E2E val, F20 selection): the head that works needs no vision. Ego state + command + the plan give +0.15 RFS out of fold, all of it on the 52 turn-intent frames; frozen Qwen3 features (one frame or four) and Cinque features add nothing

Written 2026-10-08. Open loop, 479 rater frames, nothing submitted. Pre-registration (committed before any score):
[plans/2026-10-08-s2-thinhead-prereg.md](../plans/2026-10-08-s2-thinhead-prereg.md). Code: `scripts/s2_thinhead.py` (q0 / plans / extract / prep /
latency), `scripts/s2_thinhead_heads.py` (fit / report / q2 / figs), `scripts/s2_thinhead_ft.py` (the fine-tune pilot), `scripts/s2_thinhead_chain.sh`.
Tables: [s2_thinhead/](s2_thinhead/); figures: [../figs/s2_thinhead/](../figs/s2_thinhead/). Conventions of [s2_gohold.md](s2_gohold.md): cluster-mean
RFS, paired bootstrap over sequences (B 4 000), WP2 = per-frame mean over its two seeds; `stopped` v0 < 0.5 m/s (120), `moving` (359), `turn` =
left / right intent (52), `straight` (427). The script reproduces WP2 8.111 and the F20 oracle +1.068 (decision 168).

**Setting.** System 1 = WP2, the only trajectory generator. System 2 = a head that picks one of the 20 candidates F20 built from WP2's plan
(5 paths x 4 speed profiles). Privileged ceiling +1.068. Rater scores exist on val only: every number fitted on them is out of fold by sequence
(5 folds x 10 repeats, hyper-parameters by inner 4-fold; vision streams are standardised and PCA-whitened on WOD train rows, never on labels).

## Answer

1. **Q0, hindsight supervision cannot help on all frames.** Picking the F20 candidate nearest to the LOGGED future and scoring it with the raters:
   min-ADE candidate **-0.060 [-0.246, +0.130]** vs WP2 (min-FDE at 3 s / 5 s -0.091 [-0.283, +0.108]; semantic class -0.152 [-0.319, +0.019]; the log
   itself +0.020). It is positive at standstill (+0.507 [+0.135, +0.884], 37 % of that stratum's ceiling), negative when moving (-0.160 [-0.361,
   +0.040]; min-FDE -0.228 [-0.437, -0.010]). A selector that imitated the log perfectly would gain nothing overall; rater scores are the only
   supervision with headroom on all frames, hindsight is usable at standstill only.
2. **Q1, a thin head does deliver a small out-of-fold gain, but not from vision.** Ridge head on rater scores (supervision b), d RFS vs WP2:

   | input streams | all (479) | turn (52) | straight (427) | share of the F20 ceiling | in-sample |
   |:--|:--|:--|:--|--:|--:|
   | `E0`: ego state + command | +0.102 [+0.018, +0.196] | +0.898 [+0.473, +1.510] | -0.018 [-0.064, +0.022] | 10 % | +0.151 |
   | **`E`: ego + command + plan descriptor (the floor)** | **+0.148 [+0.062, +0.248]** | +1.124 [+0.643, +1.787] | +0.000 [-0.035, +0.038] | 14 % | +0.225 |
   | `C+E`: + Cinque `temporal` | +0.102 [+0.020, +0.194] | +0.943 | -0.026 [-0.068, +0.015] | 10 % | +0.185 |
   | `Q1+E`: + frozen Qwen3, one frame | +0.113 [+0.023, +0.211] | +0.960 | -0.021 [-0.072, +0.035] | 11 % | +0.214 |
   | `QV+E`: + frozen Qwen3, 4 frames x 0.2 s | +0.094 [-0.009, +0.207] | +0.921 | -0.038 [-0.121, +0.049] | 9 % | +0.272 |
   | `QL+E`: + frozen Qwen3, 4 frames x 0.5 s | +0.066 [-0.029, +0.168] | +0.879 | -0.062 [-0.133, +0.005] | 6 % | +0.246 |
   | `Q1+C+E` / `QV+C+E` | +0.101 [+0.007, +0.203] / +0.079 [-0.018, +0.184] | +0.962 / +0.934 | | 9 % / 7 % | +0.235 / +0.309 |

   - The whole gain sits on the turn-intent frames (WP2's RFS there is 6.11, the oracle ceiling +2.44): the head changes WP2's plan on 94 % of
     them (hold / creep and 1.2 m nudges) and on 11 % of the straight frames, where it is worth 0.000. It is a correction of System 1 by command
     and ego state, not a scene-conditioned selector.
   - **Every visual stream is at or below the floor**: `C+E - E` -0.046 [-0.084, -0.013], `Q1+E - E` -0.036 [-0.078, +0.005], `QV+E - E` -0.054
     [-0.125, +0.016], `QL+E - E` -0.082 [-0.141, -0.027]. Permuting only the visual stream across frames leaves the result unchanged (actual
     +0.08 .. +0.10 inside the null +0.05 .. +0.14, p 0.49-0.81). The extra inputs cost variance and bring no signal.
   - **Qwen - Cinque** (the design question): one frame +0.011 [-0.027, +0.048], 4 x 0.2 s -0.008 [-0.068, +0.056], 4 x 0.5 s -0.036 [-0.085,
     +0.006]; on top of Cinque -0.001 / -0.023 / -0.015 (all CIs contain 0). By the pre-registered rule **Qwen does not add.**
   - **Multi-frame - single-frame**: 4 x 0.2 s -0.019 [-0.067, +0.034], 4 x 0.5 s **-0.047 [-0.088, -0.008]**. Showing Qwen3 time does not help; the
     longer clip is slightly worse. (The clips do carry motion: v0 is read from the pooled feature with R2 0.70 / 0.72 against 0.49 for one frame.)
   - Verdict by the pre-registered family (ridge, b, {`C+E`, `Q1+E`, `QV+E`, `QL+E`}): `C+E` and `Q1+E` have 95 % CIs above 0 and are beyond the
     full-permutation null, but their Bonferroni CIs touch 0 (-0.002, -0.001): **weak**; `QV+E` / `QL+E` contain 0. No visual arm "delivers".
     The arm that does is the floor `E` (98.75 % CI [+0.038, +0.273], permutation p 0.005, moving +0.074 [-0.004, +0.152]), which was not in the family.
3. **Supervision.** (a) hindsight class on r2-train, applied to val without rater labels: `E0` +0.113 [+0.047, +0.193], `E` +0.052 [-0.025,
   +0.137], `Q1+E` +0.056, `QV+E` +0.044, `C+E` -0.016 [-0.101, +0.072]: positive at standstill and on turn frames, zero or negative on straight
   moving frames, as Q0 predicts. A better fit of the hindsight class is not a better selector: Cinque lowers the held-out hindsight NLL most
   (1.247 vs 1.317 for `E`) and scores worst. (c) a then b is below b alone for every vision arm with the ridge head (`E` -0.072 [-0.142,
   -0.005]); with the MLP head, c is the only setting that gets off zero (`E` +0.104 [+0.018, +0.202], `Q1+E` +0.091 [+0.004, +0.189]).
4. **Q2 (run: the floor's CI excludes 0).** F30 is not better than F20 (`E`: +0.122 vs +0.154, difference -0.032 [-0.072, +0.004]); choosing the
   speed alone keeps most of it (+0.124 [+0.049, +0.215]), the path alone is worth nothing (+0.016 [-0.037, +0.065]). A margin rule fitted out of
   fold (keep WP2 unless the predicted gain exceeds m) does not change the mean (-0.005 [-0.030, +0.016]) and removes the moving-frame doubt
   (+0.078 [+0.008, +0.147]). The same recipe on shipped as System 1: +0.047 [-0.046, +0.147] retrained, +0.065 [+0.016, +0.124] with the WP2-fitted
   head and the margin rule; shipped + selector stays below WP2 + selector (-0.193 [-0.335, -0.062]).
5. **Q3, joint fine-tune pilot** (LoRA on decoder layers 15-18 + the same multi-stream head, multi-frame input, a then b): **no gain from fine-tuning.** LoRA + head +0.055 [-0.022, +0.139] against the frozen feature with the same head +0.050 [-0.056, +0.166]; FT - FZ **+0.005 [-0.080, +0.086]**. On the data-rich hindsight task the LoRA pre-training (3 000 train rows) does not improve the held-out NLL (1.286 -> 1.299 on the same 400 r2-dev rows, accuracy 59.5 % -> 59.0 %).
6. **Latency** of the head path: batch 1, JPEG bytes to the selected candidate, on a card that was idle at the start: Qwen3 one frame x 3 cameras 107 ms preprocessing + 215 ms model (to layer 18, 3 060 tokens); 4 frames x 3 cameras 386 + 443 ms (6 120 tokens); the linear head 0.014 ms. The floor head needs none of it: its inputs are System 1's own (0.014 ms).

**What this says about the design.** On this board the information a selector can use out of fold with 479 labelled frames is System 1's own
state (command, speed, the plan it produced). Frozen Qwen3-VL-4B features, pooled, single- or multi-frame, do not carry a learnable signal for
the next-5 s choice beyond that, and neither does Cinque's own representation; with ample hindsight labels (90 k train rows) Qwen3 features still
add nothing to the hindsight class (held-out NLL 1.313 / 1.314 vs 1.317 without vision), so the limit is not only the label count. The learning
curve of every arm is still rising at 479 frames (`E`: +0.035 / +0.091 / +0.109 / +0.148 at 25 / 50 / 75 / 100 % of the training folds).

## Q0: the ceiling of hindsight supervision (`q0.md`, `q0_picks.md`)

| selection (uses the logged future; scored by the raters) | all | stopped | moving | turn |
|:--|:--|:--|:--|:--|
| `H-ade`: F20 candidate with the smallest ADE to the log | **-0.060 [-0.246, +0.130]** | +0.507 [+0.135, +0.884] | -0.160 [-0.361, +0.040] | +0.579 [-0.059, +1.242] |
| `H-fde`: smallest mean distance at 3 s and 5 s | -0.091 [-0.283, +0.108] | +0.581 [+0.205, +0.968] | -0.228 [-0.437, -0.010] | +0.680 [+0.007, +1.369] |
| `H-cls`: the log's V3 speed class (follow where WP2 agrees) + nearest path | -0.152 [-0.319, +0.019] | +0.338 [-0.101, +0.763] | -0.312 [-0.485, -0.132] | +0.608 [-0.065, +1.375] |
| the log itself | +0.020 [-0.205, +0.246] | +0.762 [+0.337, +1.206] | -0.138 [-0.390, +0.108] | +0.521 [-0.227, +1.249] |
| F20 oracle (privileged) | +1.068 [+0.891, +1.256] | +1.376 | +0.943 | +2.439 |

- The min-ADE candidate halves WP2's ADE to the log (1.06 m vs 1.89 m) and reaches the rater-best candidate on 60 % of frames (WP2 untouched: 56 %);
  it is better than WP2 on 21.5 % of frames and worse on 16.0 %. Cut_ins -2.09 [-3.31, -0.89] and Cyclist -0.48 [-0.90, -0.07] lose (the driver
  brakes, the raters continue: decision 164).
- By the pre-registered rule: the CI on all frames contains 0 with a point estimate <= +0.1, so hindsight supervision cannot help by construction
  on all frames; it is valid supervision on the standstill stratum (CI lower bound > 0).
- Hindsight class shares, keep / follow: r2-train 54.8 %, r2-dev 54.8 % (WP2 never saw r2-dev: no in-sample shift), val rater frames 39.7 %.
  The rater frames are the hard moment of each sequence, not a sample of the train distribution.

## Q1: frozen features (`q1.md`, `arms.md`, `contrasts.md`, `permutation.md`, `picks.md`, `hindsight_fit.md`, `learning_curve.md`)

**Streams.** Every arm is a multi-stream input: `ego` (the 20 numbers of `pp_wod.wod_ego`: command one-hot, 4-pose history, speed, acceleration)
and `plan` (15: arc length at 1-5 s, lateral position at 3 / 5 s, heading, WP2's own V3 class, the prototype distances for this speed), plus
one or two visual streams: `C` = shipped Cinque `temporal` (512), `Q1` = `qwen_front3` `L18_mean` (3 front cameras in one forward, 3 060 tokens,
layer 18 of 36, image-token mean: decision 5), `QV` = `qwenvid_train_t4` `L18_mean` (the same three cameras, 4 frames each at 0.2 s spacing through
Qwen3's video path, 6 120 tokens; cached for train and val), `QL` = the same recipe at 0.5 s spacing (t0 - 1.5 .. t0, the instants of the ego pose
history; extracted here for the rater frames, 478 of 479 have the window). The cached `qwen_front3` set is single-frame.
**Heads.** `L`: multi-output ridge on the 20 candidates' RFS gain over keep / follow, argmax (k in {8, 32, 128} PCs per visual stream and lambda by
inner CV; for a: multinomial logistic). `M`: one linear encoder per stream (32) -> GELU -> dropout 0.5 -> 20, listwise loss.

Ridge head, out of fold (b, c) or applied without rater labels (a); d RFS vs WP2 [95 % CI]:

| streams | sup. | all | stopped | moving | turn | straight | 98.75 % CI (all) | in-sample |
|:--|:--|:--|:--|:--|:--|:--|:--|--:|
| `E0` | b | +0.102 [+0.018, +0.196] | +0.314 [+0.114, +0.547] | +0.010 [-0.071, +0.081] | +0.898 [+0.473, +1.510] | -0.018 [-0.064, +0.022] | [-0.000, +0.219] | +0.151 |
| **`E`** | b | **+0.148 [+0.062, +0.248]** | +0.312 [+0.104, +0.555] | +0.074 [-0.004, +0.152] | +1.124 [+0.643, +1.787] | +0.000 [-0.035, +0.038] | [+0.038, +0.273] | +0.225 |
| `C+E` | b | +0.102 [+0.020, +0.194] | +0.232 [+0.023, +0.472] | +0.047 [-0.020, +0.115] | +0.943 [+0.494, +1.550] | -0.026 [-0.068, +0.015] | [-0.002, +0.218] | +0.185 |
| `Q1+E` | b | +0.113 [+0.023, +0.211] | +0.256 [+0.048, +0.497] | +0.047 [-0.036, +0.131] | +0.960 [+0.492, +1.606] | -0.021 [-0.072, +0.035] | [-0.001, +0.240] | +0.214 |
| `QV+E` | b | +0.094 [-0.009, +0.207] | +0.185 [-0.062, +0.452] | +0.054 [-0.042, +0.166] | +0.921 [+0.470, +1.521] | -0.038 [-0.121, +0.049] | [-0.043, +0.244] | +0.272 |
| `QL+E` | b | +0.066 [-0.029, +0.168] | +0.172 [-0.067, +0.429] | +0.015 [-0.066, +0.091] | +0.879 [+0.456, +1.471] | -0.062 [-0.133, +0.005] | [-0.052, +0.195] | +0.246 |
| `Q1+C+E` | b | +0.101 [+0.007, +0.203] | +0.212 [-0.014, +0.470] | +0.044 [-0.031, +0.123] | +0.962 [+0.470, +1.627] | | [-0.018, +0.230] | +0.235 |
| `QV+C+E` | b | +0.079 [-0.018, +0.184] | +0.161 [-0.096, +0.442] | +0.037 [-0.026, +0.105] | +0.934 [+0.486, +1.560] | | [-0.041, +0.211] | +0.309 |
| `QL+C+E` | b | +0.087 [-0.006, +0.190] | +0.191 [-0.050, +0.453] | +0.038 [-0.028, +0.108] | +0.930 [+0.479, +1.564] | | [-0.034, +0.217] | +0.220 |
| `E0` | a | +0.113 [+0.047, +0.193] | +0.277 [+0.100, +0.492] | +0.048 [-0.005, +0.110] | +0.610 [+0.251, +1.145] | +0.034 [-0.007, +0.080] | [+0.031, +0.217] | |
| `E` | a | +0.052 [-0.025, +0.137] | +0.199 [-0.036, +0.455] | -0.007 [-0.026, +0.015] | +0.626 [+0.243, +1.162] | -0.046 [-0.092, -0.013] | | |
| `C+E` | a | -0.016 [-0.101, +0.072] | +0.016 [-0.244, +0.286] | -0.022 [-0.061, +0.008] | +0.482 [+0.104, +1.053] | -0.085 [-0.154, -0.026] | | |
| `Q1+E` | a | +0.056 [-0.021, +0.141] | +0.198 [-0.037, +0.455] | -0.001 [-0.018, +0.018] | +0.626 | -0.042 [-0.087, -0.009] | | |
| `QV+E` | a | +0.044 [-0.029, +0.129] | +0.168 [-0.065, +0.429] | -0.009 [-0.020, -0.000] | +0.442 | -0.020 [-0.078, +0.043] | | |
| `E0` | c | +0.111 [+0.026, +0.207] | +0.328 [+0.123, +0.574] | +0.017 [-0.065, +0.091] | +0.918 | -0.017 | [+0.005, +0.233] | +0.137 |
| `E` | c | +0.076 [-0.027, +0.182] | +0.127 [-0.154, +0.423] | +0.054 [-0.009, +0.121] | +0.881 | -0.044 | | +0.122 |
| `C+E` / `Q1+E` / `QV+E` | c | +0.032 / +0.061 / +0.051 (all CIs contain 0) | +0.131 / +0.125 / +0.134 | +0.010 / +0.031 / +0.013 | +0.757 / +0.853 / +0.858 | | | +0.144 / +0.238 / +0.270 |
| constant candidate chosen in fold | b | 0.000 (always keep) | | | | | | |

MLP head (`M`; k = 64, 32 + 32 with two visual streams): b from a random initialisation is 0.000 for every arm (the inner CV picks the largest
weight decay, at which the head keeps WP2's plan everywhere; the training folds reach +0.39 .. +0.46 with vision and +0.04 without: pure
overfit). a: `E0` +0.079 [+0.027, +0.147], `E` +0.065 [-0.011, +0.150], `Q1+E` +0.065, `QV+E` +0.038, `C+E` -0.008. c (a's weights, L2-SP): `E`
+0.104 [+0.018, +0.202], `Q1+E` +0.091 [+0.004, +0.189], `QV+E` +0.068 [-0.009, +0.159], `E0` +0.036, `C+E` +0.004, `Q1+C+E` -0.008, `QV+C+E` +0.007.

Paired contrasts, ridge, supervision b (same frames, same folds):

| contrast | all | turn | straight |
|:--|:--|:--|:--|
| plan stream: `E - E0` | +0.047 [+0.006, +0.093] | +0.226 [+0.036, +0.461] | +0.019 [-0.015, +0.060] |
| Cinque over the floor: `C+E - E` | -0.046 [-0.084, -0.013] | -0.180 [-0.364, -0.025] | -0.027 [-0.059, +0.005] |
| Qwen one frame over the floor: `Q1+E - E` | -0.036 [-0.078, +0.005] | -0.163 [-0.348, +0.007] | -0.021 [-0.061, +0.021] |
| Qwen 4 x 0.2 s over the floor: `QV+E - E` | -0.054 [-0.125, +0.016] | -0.203 [-0.398, -0.038] | -0.038 [-0.113, +0.038] |
| Qwen 4 x 0.5 s over the floor: `QL+E - E` | -0.082 [-0.141, -0.027] | -0.245 [-0.438, -0.088] | -0.062 [-0.123, -0.006] |
| **Qwen - Cinque**: `Q1+E - C+E` | **+0.011 [-0.027, +0.048]** | +0.017 [-0.137, +0.180] | +0.006 [-0.030, +0.041] |
| Qwen - Cinque, 4 x 0.2 s: `QV+E - C+E` | -0.008 [-0.068, +0.056] | -0.023 [-0.170, +0.108] | -0.011 [-0.073, +0.056] |
| Qwen - Cinque, 4 x 0.5 s: `QL+E - C+E` | -0.036 [-0.085, +0.006] | -0.065 [-0.250, +0.087] | -0.036 [-0.086, +0.009] |
| Qwen on top of Cinque: `Q1+C+E` / `QV+C+E` / `QL+C+E` `- C+E` | -0.001 [-0.035, +0.031] / -0.023 [-0.065, +0.015] / -0.015 [-0.051, +0.019] | +0.019 / -0.009 / -0.014 | -0.009 / -0.030 / -0.021 |
| multi-frame - single-frame: `QV+E - Q1+E` | -0.019 [-0.067, +0.034] | -0.040 [-0.146, +0.043] | -0.017 [-0.070, +0.037] |
| multi-frame 0.5 s - single-frame: `QL+E - Q1+E` | -0.047 [-0.088, -0.008] | -0.082 [-0.207, +0.020] | -0.042 [-0.086, -0.001] |
| same whitening for both (fitted on the val rows): `QL+E - Q1v+E` / `QL+E - QVv+E` | -0.023 [-0.063, +0.013] / -0.011 [-0.038, +0.014] | | |
| c - b, per arm (`E0` / `E` / `C+E` / `Q1+E` / `QV+E`) | +0.009 / -0.072 [-0.142, -0.005] / -0.070 [-0.134, -0.010] / -0.052 / -0.043 | | |

Under supervision a the `Q1+E - C+E` contrast is +0.072 [+0.001, +0.154]: that is Cinque being worse than the floor there (`C+E - E` -0.068),
not Qwen being better (`Q1+E - E` +0.004 [-0.000, +0.012]; the selected k is 8 PCs and the picks are those of `E`).

Controls and diagnostics:

| | `E` | `C+E` | `Q1+E` | `QV+E` | `QL+E` |
|:--|--:|--:|--:|--:|--:|
| out-of-fold gain, 3 repeats | +0.132 | +0.091 | +0.079 | +0.099 | +0.080 |
| all inputs permuted across frames (200 permutations): null mean [95 % range]; p | -0.009 [-0.032, +0.005]; 0.005 | -0.012 [-0.037, +0.003]; 0.005 | -0.013 [-0.044, +0.004]; 0.005 | -0.014 [-0.043, +0.003]; 0.005 | -0.013 [-0.041, +0.005]; 0.005 |
| only the visual stream permuted: null mean [range]; p | | +0.105 [+0.065, +0.143]; 0.76 | +0.098 [+0.059, +0.144]; 0.81 | +0.098 [+0.046, +0.138]; 0.49 | +0.099 [+0.050, +0.140]; 0.79 |
| learning curve, 25 / 50 / 75 / 100 % of the training sequences | +0.035 / +0.091 / +0.109 / +0.148 | +0.020 / +0.056 / +0.103 / +0.102 | -0.042 / +0.054 / +0.094 / +0.113 | +0.029 / +0.061 / +0.069 / +0.094 | |
| hindsight class on r2-dev (4 746 rows; prior NLL 1.716, accuracy 54.8 %): NLL / accuracy | 1.317 / 58.0 % | 1.247 / 58.4 % | 1.313 / 58.0 % | 1.314 / 57.6 % | |
| share of frames whose plan is changed: all / turn | 20 % / 94 % | 30 % / 90 % | 30 % / 85 % | | |

- The in-sample column is the optimism: +0.08 for the floor, +0.10 .. +0.23 for the visual arms (more inputs, more optimism, less out of fold).
- Layer curve of the single-frame stream (b, ridge, descriptive): L09 +0.106, L18 +0.113, L27 +0.107, L36 +0.101, L18 last token +0.104; no layer
  stands out and none exceeds the floor.
- Equal input width for the two-stream arms (k / 2 PCs each): `Q1+C+E` +0.104, `QV+C+E` +0.086: unchanged.
- What the floor head does (`picks.md`, repeat 0): speed picks follow / hold / creep / go on turn frames 15 / 27 / 10 / 0 (WP2 seed 0), paths
  lane-left / nudge-left / keep / nudge-right / lane-right 0 / 15 / 21 / 9 / 7; on the other frames it keeps follow on 389 of 427. Its pick reaches the
  rater-best candidate on 54 % of turn frames (WP2 untouched: 23 %). It is more than one constant per command: a constant candidate per intent
  chosen in fold gives +0.069 [-0.019, +0.150] (`q2.md`), the ridge head is +0.085 [+0.013, +0.171] above it.

## Q2: candidate set, conservative rule, shipped as System 1 (`q2.md`, `q2_contrasts.md`)

Run because the floor's out-of-fold CI excludes 0. The plan stream is standardised on the val frames here (so that WP2's and shipped's descriptors
share one scale); the F20 / argmax row reproduces Q1 (+0.154 against +0.148).

| System 1 | candidate set | K | rule | `E` all | stopped | moving | turn | oracle | frames changed |
|:--|:--|--:|:--|:--|:--|:--|:--|--:|--:|
| WP2 | F20 | 20 | argmax | +0.154 [+0.064, +0.254] | +0.309 | +0.084 [+0.002, +0.168] | +1.123 | +1.068 | 18 % |
| WP2 | F20 | 20 | margin fitted out of fold (median 0.18) | +0.149 [+0.066, +0.243] | +0.313 | +0.078 [+0.008, +0.147] | +1.052 | | 15 % |
| WP2 | F30 | 30 | argmax / margin | +0.122 [+0.034, +0.218] / +0.124 [+0.043, +0.216] | +0.296 | +0.049 / +0.053 | +0.970 / +0.938 | +1.223 | 18 % / 14 % |
| WP2 | speed only | 4 | argmax / margin | +0.124 [+0.049, +0.215] / +0.122 [+0.049, +0.211] | +0.331 / +0.324 | +0.033 | +0.838 / +0.823 | +0.830 | 13 % / 12 % |
| WP2 | path only | 5 | argmax / margin | +0.016 [-0.037, +0.065] / +0.015 [-0.036, +0.064] | +0.066 | +0.006 | +0.243 | +0.284 | 12 % / 11 % |
| shipped | F20, head refitted on shipped | 20 | argmax / margin | +0.047 [-0.046, +0.147] / +0.062 [-0.014, +0.145] | +0.015 / +0.023 | +0.049 / +0.074 | +0.567 / +0.533 | +1.165 | 39 % / 25 % |
| shipped | F20, head fitted on WP2 | 20 | argmax / margin | +0.050 [-0.003, +0.113] / +0.065 [+0.016, +0.124] | +0.049 / +0.058 | +0.045 / +0.063 | +0.481 / +0.490 | | 16 % / 14 % |

- Vocabulary: a larger set does not help a head trained on 479 frames (F30 - F20 -0.032 [-0.072, +0.004]); speed only - F20 -0.030 [-0.079,
  +0.022]; path only - F20 -0.138 [-0.232, -0.052]. What is learnable is the speed choice; the lateral nudges the oracle uses are not found.
- With vision the margin helps a little (`C+E` margin - argmax +0.024 [+0.003, +0.046]) and the visual arms stay below the floor (`C+E - E` under the
  margin rule -0.027 [-0.050, -0.003], `Q1+E - E` -0.014 [-0.049, +0.026]).
- shipped + selector 8.07 against WP2 + selector 8.26: -0.193 [-0.335, -0.062]. The selector does not replace System 1's input parity.

## Q3: joint fine-tune pilot (`ft.md`, `ft_info.json`)

Run as planned (the frozen read shows nothing from Qwen3 beyond ego + Cinque, but the diagnostics do not say fine-tuning could not change it: the
frozen feature fails on the data-rich hindsight task too, which points at the feature, and the learning curves still rise). Pilot scale.

Recipe: Qwen3-VL-4B on the `QV` input (3 cameras x 4 frames at 0.2 s, video path, decoder cut at layer 18), LoRA rank 16 on the attention q / v
projections of decoder layers 15-18 (the last fusion blocks of the truncated model; 8 adapted matrices), the pooled image-token mean -> the `M`
head with its three streams (vision 2 560 standardised, ego, plan). `FZ` = the same head on the cached frozen feature. Both: supervision a first
(FZ head: all 89 792 r2-train rows from the cache; FT: starts from FZ's head and a zero LoRA, then 3 000 r2-train rows online, 750 steps of 4),
then b by sequence 5-fold (the folds of repeat 0, 3 epochs, batch 4, no inner selection), from the layer-14 hidden states cached for the 478
rater frames with a clip. One permutation run (rater targets of another frame, same folds).

| arm | all | stopped | moving | turn | train folds (in-sample) |
|:--|:--|:--|:--|:--|--:|
| FZ: frozen feature + head, a then b | +0.050 [-0.056, +0.166] | +0.261 [-0.017, +0.567] | -0.048 [-0.133, +0.030] | +0.579 [+0.071, +1.224] | +0.152 |
| **FT: LoRA + head, a then b** | **+0.055 [-0.022, +0.139]** | +0.112 [-0.115, +0.353] | +0.025 [-0.016, +0.069] | +0.638 [+0.303, +1.162] | +0.053 |
| **FT - FZ** | **+0.005 [-0.080, +0.086]** | -0.148 [-0.359, +0.042] | +0.073 [+0.001, +0.153] | +0.059 [-0.436, +0.544] | |
| a only (no rater labels): FZ / FT | +0.042 [-0.032, +0.130] / +0.028 [-0.033, +0.100] | +0.110 / +0.094 | 0.000 / 0.000 | +0.322 / +0.317 | |
| FT - FZ, a only | -0.015 [-0.064, +0.019] | | | | |
| permuted targets: FZ / FT | -0.105 [-0.218, -0.002] / -0.013 [-0.057, +0.023] | | | +0.058 / +0.084 | -0.061 / -0.019 |
| FT - FT permuted | +0.069 [+0.005, +0.147] | +0.204 [+0.015, +0.432] | +0.006 [-0.032, +0.047] | +0.554 [+0.207, +1.075] | |

Hindsight class on r2-dev (the data-rich read): FZ head on all 4 746 rows NLL 1.324, accuracy 57.8 %; on the 400 rows used online, frozen 1.286 /
59.5 %, after the LoRA pre-training 1.299 / 59.0 %. The training loss did not move (1.26 .. 1.40 per 100 steps).

- Fine-tuning the last fusion blocks changes nothing measurable: FT - FZ +0.005 with a CI of +/-0.08, and no improvement of the hindsight class
  with 3 000 rows. Neither arm's CI excludes 0 on all frames; both sit below the vision-free ridge floor (+0.148), and like it they have their
  whole gain on the turn frames (+0.58 / +0.64). The one stratum where the CI of FT - FZ excludes 0 is `moving` (+0.073 [+0.001, +0.153]),
  where FZ loses and FT is at +0.025: fewer harmful changes, not a gain over WP2.
- No overfit signature in FT (train folds +0.053, out of fold +0.055; permuted -0.013): three epochs at these learning rates barely move the
  head. FZ shows it (train folds +0.152, out of fold +0.050, permuted targets -0.105).
- Checks: the split forward (frozen to layer 14, then layers 15-18) reproduces the cached `L18_mean` to a relative L2 of 0.005 (median; max
  0.031: batch 2 against the cache's batch 8); the online feature with the LoRA at zero matches the cache to 0.005 (median).
- Cost: cache 15 min, pre-training 41 min (0.67 s per row on a shared card), k-fold with the permutation run 34 min: about 1.5 card-hours.
- Not tried: LoRA on the vision tower or on more layers, a token-grid read-out, more pre-training rows, a longer schedule. With this result the
  next step is not a longer run of the same recipe (see caveats).

## Latency (`latency.md`)

Batch 1, the rater frames, 40 frames after 5 warm-up; JPEG bytes -> processor -> Qwen3-VL-4B to layer 18 -> pooled feature (host sync) -> head.
The card was at 0 % utilisation and 1.9 GB at the start (`latency_cards.txt`); bf16, no compilation.

| path | LLM tokens | preprocessing ms (read + decode + processor) | model ms median / p95 | peak VRAM |
|:--|--:|--:|:--|--:|
| `Q1`: 3 cameras, 1 frame | 3 060 | 107 | 215 / 216 | 8.3 GB |
| `QV` / `QL`: 3 cameras x 4 frames, video path | 6 120 | 386 | 443 / 523 | 13.3 GB |
| head: linear, PCA folded into the weights (CPU) | | | 0.014 | |
| Cinque `temporal` (already computed by System 1; from the feature set's meta, shared card) | | | 5-20 | |

A feature-plus-head System 2 on Qwen3 is 0.32 s per decision with one frame and 0.83 s with four at native resolution (decision 85's
resolution lever, 559-1 153 tokens, was not applied); zero-shot prompting was 3-4 s (decision 168). The head that delivers here costs 0.014 ms.

## Figures

| figure | what to look at |
|:--|:--|
| [01_arms_oof.png](../figs/s2_thinhead/01_arms_oof.png) | Out-of-fold gain of every arm (ridge head; grey a, blue b, red c; crosses = in-sample fit) against the F20 oracle line, per stratum. The `turn` panel carries everything (about +1.0 for every arm, the oracle at +2.4); the `straight` panel is flat at 0. Adding a visual stream moves the cross up and the dot down. |
| [02_permutation_curve.png](../figs/s2_thinhead/02_permutation_curve.png) | Left panels: the out-of-fold gain (red line) against 200 fits with the inputs permuted across frames: every arm is far outside its null. Right: the learning curve, still rising at the full 479 frames; the floor `E` is on top at every size. |
| [03_frames.png](../figs/s2_thinhead/03_frames.png) | Eight frames of the floor head `E` (b, ridge, out-of-fold picks; left block the 4 largest gains, right block the 4 largest losses; FRONT camera at t0, BEV with the 20 candidates in grey, WP2 red, the head's pick blue, the oracle's pick green dashed, rater trajectories black / grey). Gains: turn or low-speed frames where WP2 runs wide or long and the head takes nudge + creep or hold (4.0 -> 10.0, the oracle's own pick). Losses: frames where WP2 was already at 10.0 and the head applied the same correction (go on a straight road, hold at a night junction). The head sees no image; the picture is there to show what it would have needed. Numbers: `s2_thinhead/figure_frames.csv`. |

## Checks

- Reproduces WP2 8.111, the F20 oracle +1.068, the log 8.131.
- Features are joined by `frame_name`; held-out R2 of v0 from the raw visual streams: Cinque 0.94, Qwen3 one frame 0.49, 4 x 0.2 s 0.70, 4 x 0.5 s 0.72.
- WP2's plan through the training-protocol token path (used for the train rows) against the harness plan on the 245 rater frames that have both:
  mean 0.012 m, p95 0.043 m.
- `wod/r2-train`, `wod/r2-dev`, `wod/val` are pairwise disjoint (`splits.check_disjoint`); the rater frames are inside `wod/val`.
- The label-permutation control uses the same code path as the real fit (3 repeats on both sides).

## Deviations from the pre-registration

- Train rows for supervision a: r2-train rows at frame % 4 == 0 (89 792 rows of 2 036 sequences, the rows that have the multi-frame cache), not
  every third row; r2-dev 4 746 rows.
- `H-cls` picks the path by the distance at 5 s among the five candidates at the chosen speed (the prereg said lateral offset).
- Standardisation of ego / plan streams and the PCA use WOD train rows, not the training folds (still label-free); `QL` has no train rows and is
  whitened on the 478 val rows, with `Q1v` / `QVv` (same treatment) as its like-for-like references.
- Model selection inside the folds uses the frame mean of the realised gain, the report the cluster mean.
- The `straight` stratum, the turn-only reading, the intent-conditional constant and the `E` arm's verdict are *post hoc* (the pre-registered
  family held the four visual arms only).
- The permutation control compares 3-repeat statistics; the MLP head has no permutation control (its b arm is 0).
- Q3 is a pilot: 3 000 pre-training rows (planned as "a train subset"; 6 000 was the first sizing, halved after the measured 0.7-0.8 s per row), 3
  epochs per fold (5 planned before timing), one repeat of the folds, fixed hyper-parameters, the head is the `M` head with a 2 560-wide vision
  encoder instead of 64 PCs. One permutation run, not a null distribution.
- The MLP head's tie rule (largest weight decay on ties) makes its b arm collapse to "keep"; reported as is.

## Caveats

- The fine-tune pilot does not settle whether Qwen3 can be made to carry the signal: it adapts 4 decoder layers on a pooled read-out with 3 000
  hindsight rows and 383 rater frames per fold. It does say that this cheap recipe gives nothing.
- 479 frames, 52 of them with a turn intent: the one positive result rests on those 52 (CI [+0.64, +1.79] there, robust across arms and heads,
  but one board and one System 1). Cluster rows are in `arms.md`, descriptive only.
- Open loop; the candidates of a standstill plan drive straight ahead (decision 168's caveat).
- Pooled features only: one 2 560-vector per frame (image-token mean) or Cinque's 512-vector. A spatial token grid with an attention read-out was
  not tried here (decision 43's Q9b tried it for pedestrians).
- The ridge target is the RFS gain of each candidate, which is discontinuous (4.0 floor); a classification of "which candidates are inside a
  trust region" might use the labels better. Not tried.
- No test submission; val rater labels are used for fitting only out of fold, the oracle rows are ceilings.
