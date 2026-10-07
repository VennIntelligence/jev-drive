# System 1 + System 2 late fusion on WOD-E2E val: a perfect selector among 20 path x speed candidates of WP2 is worth +1.07 RFS, almost all of it reachable by choosing speed and path separately; a binary go / hold symbol is worth little; zero-shot Qwen3-VL-4B delivers none of it

Written 2026-10-08. Open loop, 479 rater frames, no training, nothing submitted. Pre-registration and its addendum B (the coordinator's scope change
to joint path + speed selection; both written and committed before anything was scored or asked):
[plans/2026-10-08-s2-gohold-prereg.md](../plans/2026-10-08-s2-gohold-prereg.md). Code: `scripts/s2_gohold.py` (`fit` / `qwen` / `report` / `figs`).
Tables: [s2_gohold/](s2_gohold/); figures: [../figs/s2_gohold/](../figs/s2_gohold/). Conventions as in [wod_gap.md](wod_gap.md): cluster-mean RFS,
paired bootstrap over sequences (B 4 000), WP2 = per-frame mean of the two seeds' scores; `stopped` = v0 < 0.5 m/s (120 frames), `moving` (359),
`turn` = left / right intent (52). The script reproduces decisions 163 / 164 (WP2 8.111, shipped 8.005, log 8.131, top-rated 9.587; `stopped` 7.598).

**Every "oracle" row uses the val rater trajectories to choose. It is a privileged ceiling, not a method.** Codebook, launch profile and trajectory
anchors are fitted on WOD train logs only (`wod/r2-train`, 394 808 rows); nothing is fitted on val except one threshold, reported out of fold.

## Answer

1. **What a System 2 selector could be worth (privileged ceiling, all frames).** Picking per frame the best of K candidates, WP2 included:

   | candidate set (no val labels in its construction) | K | d RFS vs WP2 [95% CI] | share of the 1.476 to the top-rated trajectory |
   |:--|--:|:--|--:|
   | System 1's own outputs: WP2 + shipped / + other seed + WP1 x 2 | 2 / 5 | +0.357 [+0.262, +0.465] / +0.408 [+0.310, +0.520] | 24 % / 28 % |
   | k-means anchors of train logs, one global set | 8 / 16 / 32 / 64 | +0.591 / +0.842 / +0.951 / +1.083 | 40-73 % |
   | k-means anchors per speed bin | 8 / 16 / 32 / 64 | +0.890 / +1.004 / +1.109 / +1.244 | 60-84 % |
   | **F20: WP2's path family (keep, +/-1.2 m, +/-3.5 m) x speed (own, hold, creep, go)** | 20 | **+1.068 [+0.891, +1.256]** | 72 % |
   | F30: the same paths x (own + 5 train speed buckets) | 30 | +1.223 [+1.040, +1.422] | 83 % |

   The plan head exposes one hypothesis only, so "System 1's own outputs" are the stored arms. A set built from the model's own plan (F20) is as good as
   64 global anchors and needs no anchor that replaces WP2's path.
2. **How much needs a joint decision: little.** On F20, choosing only the speed profile (WP2's path kept) gives +0.856 [+0.71, +1.02], choosing only
   the path (WP2's speed kept) +0.367 [+0.26, +0.50], the better of the two per frame +1.026, the joint choice +1.068: **+0.042 [+0.008, +0.078] (4 %)
   needs both at once** (F30 +0.076, 6 %). On turn frames it is +0.14 [-0.06, +0.36] of +2.44. With the continuous top-rated trajectory as the
   target (decision 164's swaps) the per-frame "needs both" is +0.283 [+0.201, +0.369] of 1.476 (19 %). The oracle keeps WP2's path on 415 of
   479 frames, nudges it by 1.2 m on 53 and shifts a lane on 11; it changes the speed profile on 182. On this board System 2's value is mostly a
   longitudinal choice per frame, plus a lateral nudge on a tenth of the frames, and the two rarely have to be decided together.
3. **A binary go / hold symbol is too coarse.** WP2's path re-timed to a fixed train-log profile of the class of the top-rated trajectory
   (gate: only where WP2's own plan is in another class), standstill frames: V2 (hold / go) +0.206 [-0.085, +0.505], V3 (hold / creep / go) +0.270
   [-0.105, +0.660], V5 +0.444 [+0.028, +0.876], continuous profile +1.068 [+0.586, +1.594]. By the pre-registered rule the binary decision has **no
   demonstrated value** on this board; five buckets do. On all frames V2 is -0.070 [-0.234, +0.094], V5 +0.190 [-0.023, +0.405], continuous +0.722.
   The reason is the decoder, not the label: "go" spans 2-30 m in 5 s and one launch profile (9.3 m) lands outside the trust region of most of
   them. Picking the better of the two fixed profiles per frame would be +0.926 [+0.615, +1.275] (codebook bound, privileged).
4. **Zero-shot Qwen3-VL-4B does not deliver.**
   - Go / hold (primary arm A: decomposed questions then the decision, V2, gate, standstill frames only): **+0.123 [-0.274, +0.515]** vs WP2 on
     `stopped`; applied to moving frames as well it loses **-0.230 [-0.374, -0.098]** there. Verdict by the pre-registered rule: not delivered
     (standstill CI contains 0, moving loses). It agrees with the oracle class on 42.5 % of standstill frames; WP2's own plan agrees on 67.9 %, the log
     on 78.3 %, "always go" on 74.2 %. "Always hold" scores +0.211 [-0.279, +0.693], more than Qwen3; "always go" -0.738 [-1.218, -0.325].
   - Selector (primary arm B: path picked on a set-of-mark image, speed picked in text, F20): **-0.740 [-0.995, -0.489]** on all frames, harmful;
     path choice alone -0.152 [-0.248, -0.075], speed choice alone -0.592 [-0.845, -0.344]. Its pick reaches the oracle's best score on 38.8 % of
     frames; leaving WP2 untouched does on 55.6 %. No scenario cluster has a gain; Cyclist, Foreign Object Debris, Intersections, Pedestrian and
     Single-Lane Maneuvers have CIs below 0.
   - Why: Qwen3 answers the rule state of the scene, the raters reward what the ego does over the next 5 s. It says hold on all 26 red-light and all
     45 stop-sign standstill frames; the top-rated trajectory moves more than 2 m within 5 s on 54 % and 82 % of them. Where the cue is in the frame
     (green light) it is right (90.5 %) and so is WP2 (90.5 %): there is nothing to add.
5. **Cost.** One forward is 0.29 s with one image (1.2 k prompt tokens) and 0.91 s with four (4.3 k); a decomposed go / hold decision (5 + 1
   forwards) 3.1 s, the selector (8 forwards) 4.1 s, a single-prompt decision 0.9 s. Measured on a card shared with two training jobs at 100 %
   utilisation and at the processor's default resolution on 972 x 1079 images: an upper bound for this setup, not comparable with decision 84's
   68-125 ms.
6. **Timing is not measured here.** WOD scores one frame per sequence, so "when" appears only as the shape of the chosen speed profile at that
   frame. A read that measures timing needs consecutive decisions: closed loop on HUGSIM (the frame at which System 2 releases a stop or starts a
   pass, against collisions and route completion) or the multi-frame sequences behind navtest scenes (per-frame decisions along a log: the
   switch time against the logged launch / lane-change time). Not run.

## A. Longitudinal go / hold vocabularies (one row of the answer)

Vocabulary (fitted on train logs, `s2_gohold/codebook.json`): at standstill `hold` = under 2 m in 5 s, else `go`; moving `hold` = under 0.5 m in the
last second, else `go`; V3 / V5 split `go` at train quantiles (standstill: distance at 5 s, edges 7.8 m / 4.3, 7.8, 12.9 m; moving: equivalent mean
acceleration per speed bin). One prototype arc-length profile per class and speed bin (train mean; standstill 0.3 m for hold, 9.3 m for the V2 go =
the fixed launch profile, 4.5 / 14.0 m for V3 creep / go). Fusion: `gate` re-times System 1's own path to the prototype only where its own plan is
in another class; `replace` always re-times. Tables: `oracle.md`, `oracle_increments.md`, `oracle_context.md`, `classes.md`.

### Oracle ceiling by vocabulary size (privileged; d RFS vs System 1)

| System 1 | decision | K | `stopped`, standstill-only scope | `moving`, scope all | all frames, scope all |
|:--|:--|--:|:--|:--|:--|
| WP2 | class of the top-rated trajectory, gate | 2 | +0.206 [-0.085, +0.505] | -0.178 [-0.395, +0.011] | -0.070 [-0.234, +0.094] |
| WP2 | | 3 | +0.270 [-0.105, +0.660] | -0.036 [-0.264, +0.188] | +0.059 [-0.122, +0.255] |
| WP2 | | 5 | **+0.444 [+0.028, +0.876]** | +0.059 [-0.192, +0.292] | +0.190 [-0.023, +0.405] |
| WP2 | continuous top-rated speed profile (decision 164's O1) | inf | +1.068 [+0.586, +1.594] | +0.554 [+0.372, +0.742] | +0.722 [+0.523, +0.932] |
| WP2 | best class per frame (codebook bound), gate | 2 / 3 / 5 | +0.926 / +1.079 / +1.321 | +0.331 / +0.528 / +0.697 | +0.498 / +0.692 / +0.871 |
| WP2 | class of the top-rated trajectory, replace | 2 / 3 / 5 | +0.135 / +0.335 / +0.546 | -0.921 / -0.213 / +0.043 | -0.616 / -0.032 / +0.219 |
| shipped | class of the top-rated trajectory, gate | 2 | +0.078 [-0.253, +0.408] | -0.102 [-0.289, +0.067] | -0.032 [-0.190, +0.114] |
| shipped | | 3 | +0.097 [-0.229, +0.433] | +0.093 [-0.142, +0.324] | +0.118 [-0.063, +0.311] |
| shipped | | 5 | +0.219 [-0.207, +0.631] | +0.362 [+0.112, +0.613] | +0.368 [+0.164, +0.579] |
| shipped | continuous | inf | +0.611 [+0.176, +1.069] | +0.919 [+0.674, +1.184] | +0.874 [+0.667, +1.092] |
| shipped | best class per frame, gate | 2 / 3 / 5 | +0.664 / +0.856 / +1.092 | +0.466 / +0.788 / +1.005 | +0.521 / +0.807 / +1.021 |

- The binary symbol recovers a fifth of the continuous standstill ceiling and its CI contains 0; each refinement adds (V5 - V2 on all frames +0.260
  [+0.105, +0.427] for WP2, +0.400 [+0.234, +0.580] for shipped). System 2's output has to carry how far, not only whether.
- System 1 already makes the binary decision as often as not: WP2's own class equals the oracle's on 67.9 % of standstill frames (shipped 66.7 %,
  log 78.3 %) and 84.3 % of moving frames (log 83.3 %). For V3 / V5 at standstill it drops to 44 % / 28 % (log 62 % / 51 %).
- The gap between "class of the top-rated trajectory" and "best class per frame" (+0.21 vs +0.93 at V2) is the fixed decoder: a frame whose
  top-rated trajectory moves 4 m is labelled go, WP2's own 18 m plan is also go, so the gate keeps it and it stays floored, while the hold
  prototype would have scored 9.9 (figure 00). The bound is what a selector by score gets; a selector by semantic class does not.
- By standstill context (`oracle_context.md`, V2 gate, WP2): red +0.47 [-0.32, +1.38] (n 26), green +0.44 [0.00, +1.20] (21), stop sign +0.10
  [-0.23, +0.37] (45), lead +0.50 (11), open -0.11 (16). The stop-sign block, the largest standstill loss (wod-launch), gets nothing from a binary
  symbol (the oracle is go on 82 % and so is WP2) and +0.66 [-0.00, +1.24] from V3.

### Zero-shot Qwen3-VL-4B: go / hold

Questions (`scripts/s2_gohold.py`, fixed with the pre-registration): light, stop-controlled junction, lead (two frames), crossing (wod-launch's
questions verbatim; answers identical to its labels on all 479 frames), right of way (new, three front cameras); then the decision on four images
(front one second ago, front-left / front / front-right now) with the ego's state in text. `D-decomp` (primary) appends the five answers to the
decision prompt, `D-direct` asks without them, `D-rule` is a fixed rule over the answers, `D-decomp3` asks hold / creep / go.

Agreement with the oracle V2 class on standstill frames (`agreement.md`; go recall / hold recall in brackets where shown):

| group | n | oracle go | Qwen3 D-decomp | D-direct | D-rule | WP2's own plan | shipped | log | always go |
|:--|--:|--:|:--|--:|--:|:--|--:|--:|--:|
| stopped | 120 | 89 | **42.5 %** (go 27 %, hold 87 %) | 25.8 % | 64.2 % | 67.9 % (go 76 %, hold 44 %) | 66.7 % | 78.3 % | 74.2 % |
| red or yellow light | 26 | 14 | 46.2 % (says hold on all 26) | 46.2 % | 46.2 % | 40.4 % | 53.8 % | 57.7 % | 53.8 % |
| green light | 21 | 18 | 90.5 % (says go on 20) | 14.3 % | 81.0 % | 90.5 % | 85.7 % | 90.5 % | 85.7 % |
| stop-controlled junction | 45 | 37 | 17.8 % (says hold on all 45) | 17.8 % | 62.2 % | 75.6 % | 68.9 % | 82.2 % | 82.2 % |
| lead within 20 m | 11 | 4 | 63.6 % | 63.6 % | 63.6 % | 72.7 % | 81.8 % | 90.9 % | 36.4 % |
| open | 16 | 15 | 25.0 % | 6.3 % | 81.3 % | 56.3 % | 43.8 % | 75.0 % | 93.8 % |
| moving (hold = at rest by 5 s) | 359 | 305 | 77.4 % (go 87 %, hold 26 %) | 81.6 % | 70.5 % | 84.3 % | 84.4 % | 83.3 % | 85.0 % |

`D-direct` says hold on all 120 standstill frames (it is the "always hold" arm). `D-decomp3` equals the oracle V3 class on 53.3 % (WP2's own 44.2 %).

RFS of the fused plan (V2 codebook; hold -> the 0.3 m hold prototype, go -> the fixed 9.3 m train launch profile; `fused.md`, `fused_context.md`):

| System 1 | decision | fusion | `stopped`, standstill-only scope | `moving`, scope all | all frames, standstill-only / all | vs the oracle V2 on `stopped` |
|:--|:--|:--|:--|:--|:--|--:|
| WP2 | **Qwen3 D-decomp (primary A)** | gate | **+0.123 [-0.274, +0.515]** | **-0.230 [-0.374, -0.098]** | +0.031 [-0.076, +0.141] / -0.138 [-0.283, +0.010] | -0.083 |
| WP2 | Qwen3 D-decomp | replace | -0.102 [-0.525, +0.324] | -0.936 [-1.209, -0.696] | -0.040 / -0.712 | -0.237 |
| WP2 | Qwen3 D-direct (= always hold at standstill) | gate | +0.211 [-0.279, +0.693] | -0.213 [-0.342, -0.095] | +0.055 / -0.110 | +0.005 |
| WP2 | Qwen3 D-rule | gate | +0.091 [-0.269, +0.462] | -0.464 [-0.646, -0.294] | +0.048 / -0.299 | -0.116 |
| WP2 | Qwen3 D-decomp3 (V3) | gate | -0.669 [-1.228, -0.115] | | -0.174 | -0.939 |
| WP2 | always go | gate | -0.738 [-1.218, -0.325] | | -0.200 | -0.944 |
| WP2 | D-decomp, p(go) threshold chosen out of fold (0.9 on most folds) | gate | +0.191 [-0.224, +0.608] (in sample +0.211) | | | |
| shipped | Qwen3 D-decomp | gate | -0.186 [-0.470, +0.088] | -0.162 [-0.296, -0.031] | -0.043 / -0.157 | -0.264 |
| shipped | Qwen3 D-direct / always go | gate | -0.195 [-0.607, +0.207] / -0.969 [-1.503, -0.454] | | | |

By context (primary arm, WP2): stop sign +0.54 [-0.02, +1.04] (from holding where WP2's creeping plan was floored), lead +0.61 [-0.15, +1.90],
red +0.34 [-0.61, +1.24], green -0.26 [-0.97, +0.04], open -1.05 [-1.80, +0.13]. Where the log moves (`SL`, 95 frames) +0.29 [-0.08, +0.69], where
it stays (`SS`, 25) -0.37 [-1.42, +0.72].

- The positive point estimate is not Qwen3's reading of the scene: forcing hold on every standstill frame gives more. What helps is replacing WP2's
  in-between plans (2-8 m creeps that fall between the raters' modes, wod-launch's finding) by a clean stay; what hurts is every frame where
  Qwen3 says go, because the fixed launch profile is worse than WP2's own launch (always go -0.74).
- On moving frames a "hold" from Qwen3 (16 % of them, 26 % recall of the 54 oracle holds) brakes plans that the raters want to continue.

## B. System 2 as a selector among K candidate trajectories (addendum B)

Candidate sets: `S1x2` / `S1x5` stored System 1 arms; `KM{K}` k-means anchors of train-log futures (300 000 rows, ego frame, 5 s), `KMv{K}` one
set per speed bin; `F20` / `F30` WP2's path family x speed profiles (path: keep, +/-1.2 m ramped over max(10 m, 2 s x v0), +/-3.5 m over max(20 m,
4 s x v0); speed: WP2's own + the V3 / V5 prototypes of part A). For every set and candidate c, with p = WP2's plan: joint = c; longitudinal only =
p's path at c's arc-length profile; lateral only = c's path at p's profile; each maximised per frame with p as a fallback. `either` = the better of
the last two per frame, `needs joint` = joint - either. Table: `select_oracle.md` (every set x all / stopped / moving / turn / 10 clusters).

### Privileged ceilings and the longitudinal / lateral / joint split (d RFS vs WP2)

| set | K | stratum | best of set alone (RFS) | joint, with WP2 [95% CI] | longitudinal only | lateral only | needs joint [95% CI] |
|:--|--:|:--|--:|:--|--:|--:|:--|
| S1x5 | 5 | all | 8.519 | +0.408 [+0.310, +0.520] | +0.319 | +0.208 | -0.058 [-0.161, +0.008] |
| KM8 / KM64 (global) | 8 / 64 | all | 7.651 / 8.882 | +0.591 / +1.083 | +0.622 / +1.129 | +0.162 / +0.358 | -0.114 / -0.149 |
| KMv8 | 8 | all | 8.512 | +0.890 [+0.735, +1.062] | +0.983 | +0.193 | -0.135 [-0.226, -0.054] |
| KMv16 | 16 | all | 8.751 | +1.004 [+0.840, +1.177] | +1.101 | +0.264 | -0.163 [-0.261, -0.075] |
| KMv64 | 64 | all | 9.222 | +1.244 [+1.069, +1.435] | +1.179 | +0.397 | -0.032 [-0.111, +0.041] |
| **F20** | 20 | all | 9.179 | **+1.068 [+0.891, +1.256]** | +0.856 [+0.71, +1.02] | +0.367 [+0.26, +0.50] | **+0.042 [+0.008, +0.078]** |
| F20 | | stopped | 8.974 | +1.376 [+1.015, +1.771] | +1.204 | +0.320 | +0.082 [+0.015, +0.170] |
| F20 | | moving | 9.203 | +0.943 [+0.74, +1.17] | +0.70 | +0.40 | +0.03 [-0.01, +0.07] |
| F20 | | turn | 8.545 | +2.439 [+1.956, +3.199] | +1.622 | +1.500 | +0.137 [-0.057, +0.358] |
| F30 | 30 | all | 9.334 | +1.223 [+1.040, +1.422] | +0.990 | +0.373 | +0.076 [+0.033, +0.126] |
| reference: the top-rated trajectory itself (continuous) | | all | 9.587 | +1.476 [+1.271, +1.694] | +0.722 (O1 swap) | +0.296 (O2 swap) | +0.283 [+0.201, +0.369] |
| reference | | stopped / turn | 9.657 / 9.205 | +2.059 / +3.098 | +1.068 / +0.833 | +0.236 / +1.515 | +0.476 / +0.891 |

F20, joint ceiling by scenario cluster (longitudinal only / lateral only in brackets):

| cluster | n | RFS WP2 | F20 joint [95% CI] | (lon / lat) | needs joint |
|:--|--:|--:|:--|:--|:--|
| Special Vehicles | 25 | 7.31 | +1.45 [+0.68, +2.32] | (+1.20 / +0.55) | -0.00 |
| Multi-Lane Maneuvers | 42 | 7.77 | +1.39 [+0.86, +2.00] | (+1.02 / +0.44) | +0.18 [+0.02, +0.38] |
| Cut_ins | 20 | 8.39 | +1.22 [+0.62, +1.91] | (+1.08 / +0.29) | +0.06 |
| Foreign Object Debris | 78 | 8.05 | +1.22 [+0.86, +1.63] | (+0.95 / +0.62) | -0.04 |
| Interections | 116 | 7.92 | +1.14 [+0.82, +1.46] | (+0.94 / +0.25) | +0.10 [+0.01, +0.22] |
| Cyclist | 71 | 8.45 | +1.01 [+0.68, +1.35] | (+0.81 / +0.13) | +0.15 [+0.03, +0.30] |
| Pedestrian | 52 | 8.25 | +0.92 [+0.45, +1.48] | (+0.67 / +0.41) | -0.01 |
| Construction | 15 | 8.53 | +0.86 [+0.08, +1.90] | (+0.60 / +0.43) | 0.00 |
| Single-Lane Maneuvers | 38 | 8.37 | +0.84 [+0.39, +1.38] | (+0.69 / +0.39) | -0.01 |
| Others | 22 | 8.06 | +0.63 [+0.22, +1.13] | (+0.60 / +0.15) | -0.01 |

- **Every set passes the pre-registered "worth a selector" line** (CI lower bound > 0 on all frames), System 1's own five stored arms included
  (+0.41): the arms disagree usefully, mostly in speed (+0.32) and on turn frames in path (+0.90 of +1.16).
- **Train-log anchors are a speed vocabulary.** For every k-means set, WP2's path at the anchor's speed profile beats the anchor itself (`needs
  joint` negative, e.g. KMv16 +1.10 vs +1.00): the anchors' paths are worse than WP2's path, and their value is the spread of 5 s distances.
  Conditioning on the speed bin is worth a factor of 2-4 in K (KMv8 0.89 against KM16 0.84).
- **The lateral share is concentrated**: turn frames (+1.50 lateral only, +1.62 longitudinal only, +2.44 joint), Foreign Object Debris (+0.62),
  Special Vehicles (+0.55). Lane-change-size shifts are almost never the best candidate (11 of 479 frames, ties resolved toward the plan); the
  1.2 m nudges are (53 frames). The joint remainder has a CI above 0 in Multi-Lane Maneuvers (+0.18), Cyclist (+0.15) and Interections (+0.10)
  and is zero elsewhere.
- Against decision 164 ("speed 49 %, path 20 %, the rest needs both"): that was the continuous top-rated trajectory as the only target, and each
  swap applied to every frame. A selector that may also leave the plan alone gets 80 % of F20's ceiling from speed alone and 34 % from path alone
  (they overlap), and 4 % needs both. With the continuous target and the same per-frame fallback, "needs both" is 19 % (+0.283), 23 % at
  standstill and 29 % on turn frames.

### Zero-shot Qwen3-VL-4B as the selector on F20

Set fixed in the addendum before any score (F20: the only set whose candidates can be drawn and named; no k-means set of K <= 16 beats its
ceiling by 0.2, so the choice stands). Path: the five paths of WP2 (seed 0) projected onto the front image with the camera calibration as thin
coloured lines with letters, green = the plan (nothing filled, no map, no labels; overlay rules of decision 99), prompt with the six scene answers
(the five above + lane obstruction); speed: hold / creep / follow / go in text on the four images. Tables: `selector.md`, `selector_agreement.md`.

| arm | all (479) | stopped (120) | moving (359) | turn (52) | vs the F20 oracle, all |
|:--|:--|:--|:--|:--|:--|
| **Qwen3 joint pick (primary B)** | **-0.740 [-0.995, -0.489]** | +0.020 [-0.462, +0.519] | -0.999 [-1.274, -0.727] | +0.261 [-0.396, +1.017] | -1.808 [-2.042, -1.570] |
| Qwen3 path only (WP2's speed) | -0.152 [-0.248, -0.075] | -0.004 [-0.020, +0.009] | -0.182 [-0.291, -0.093] | -0.375 [-0.803, -0.062] | -1.219 |
| Qwen3 speed only (WP2's path) | -0.592 [-0.845, -0.344] | +0.026 [-0.458, +0.527] | -0.821 [-1.092, -0.565] | +0.520 [-0.114, +1.255] | -1.660 |
| F20 oracle (privileged) | +1.068 | +1.376 | +0.943 | +2.439 | 0 |

| | all | stopped | moving | turn |
|:--|--:|--:|--:|--:|
| top-1: Qwen3's pick reaches the oracle's best score (ties count) | 38.8 % | 41.3 % | 38.0 % | 28.8 % |
| top-1: WP2 untouched (keep / follow) reaches it | 55.6 % | 45.8 % | 58.9 % | 23.1 % |
| Qwen3's path can reach it with some speed / keep can | 82.6 % / 86.4 % | 84.2 % / 85.8 % | 82.0 % / 86.6 % | 50.0 % / 55.8 % |
| Qwen3's speed can reach it with some path / follow can | 47.1 % / 62.5 % | 44.2 % / 51.3 % | 48.1 % / 66.3 % | 50.0 % / 49.0 % |
| paths chosen A / B / C / D / E: Qwen3 | 37 / 1 / 432 / 6 / 3 | 9 / 1 / 104 / 5 / 1 | 28 / 0 / 328 / 1 / 2 | 12 / 0 / 36 / 1 / 3 |
| paths chosen: oracle (seed 0, ties toward the plan) | 6 / 28 / 415 / 25 / 5 | 4 / 5 / 103 / 6 / 2 | 2 / 23 / 312 / 19 / 3 | 1 / 11 / 29 / 8 / 3 |
| speeds follow / hold / creep / go: Qwen3 | 98 / 112 / 102 / 167 | 0 / 96 / 1 / 23 | 98 / 16 / 101 / 144 | 1 / 19 / 12 / 20 |
| speeds: oracle | 297 / 62 / 72 / 48 | 60 / 22 / 13 / 25 | 237 / 40 / 59 / 23 | 23 / 16 / 10 / 3 |

Joint pick by cluster (d RFS vs WP2): Single-Lane Maneuvers -1.18 [-2.06, -0.36], Cyclist -1.15 [-1.67, -0.63], Pedestrian -0.98 [-1.71, -0.30],
Foreign Object Debris -0.84 [-1.42, -0.29], Multi-Lane Maneuvers -0.80 [-1.57, +0.03], Cut_ins -0.71 [-1.69, +0.18], Interections -0.65 [-1.08,
-0.24], Others -0.60 [-1.42, +0.19], Construction -0.44 [-1.53, +0.64], Special Vehicles -0.07 [-1.16, +1.04]. It helps nowhere.

- **Speed is where it hurts.** Qwen3 leaves the speed alone on 98 of 479 frames; the oracle does on 297. Its "creep" and "go" on moving frames
  (245 of 359) replace WP2's profile by a train-mean profile that is worse than WP2's own on most frames (the same decoder problem as in A).
- **Path: it over-selects the lane shift to the left** (37 picks against 6 for the oracle) and never the nudges the oracle uses (1 + 6 against
  53). The lane-obstruction question that feeds it answers "obstructed, pass on the left" on 323 of 479 frames (the first option): unusable
  zero-shot, and it was passed into the path and speed prompts.
- The one direction with a positive point estimate is speed choice on turn frames (+0.52 [-0.11, +1.26], 52 frames; mostly "hold" / "creep"
  where WP2 runs long): undecided.

## Figures

Top left: the front image with the five candidate paths as Qwen3 saw it for the path choice (A magenta / B cyan / C green = WP2's plan / D orange /
E blue). Top middle: Qwen3's answers (A = go / hold, B = selector) next to the privileged oracle's. Bottom: the road and wide frames openpilot is
fed at t0. Right: BEV, ego at the star heading up, lateral axis stretched, markers at 3 s and 5 s: black = top-rated rater trajectory, grey dashed =
the other two, orange = log, red = WP2 (seed 0), blue = WP2 re-timed by Qwen3's go / hold (dashed: by the oracle V2), green = Qwen3's selector
pick (dashed: the oracle's best of F20). Selection fixed in the plan (standstill: per context the largest gain / loss of arm A; moving: the three
largest gains and losses of arm B among frames where Qwen3 left path C). Numbers: `s2_gohold/figures.csv`.

| figure | frame | what to look at |
|:--|:--|:--|
| [00](../figs/s2_gohold/00_standstill_stop_sign_gain_0ff0c31f.png) | stop sign, gain | WP2 plans 18 m (floored, 4.0); the top-rated trajectory creeps 3.9 m. Qwen3 says hold: 9.9. The oracle V2 says "go", WP2 is "go", the gate keeps the 18 m plan: why a binary class is too coarse. |
| [01](../figs/s2_gohold/01_standstill_stop_sign_loss_99dc1a0c.png) | stop sign, loss | WP2 already matches the top-rated 5 m (10.0); Qwen3 holds because of the stop sign: 6.0. The rule state is not the decision. |
| [02](../figs/s2_gohold/02_standstill_green_gain_fa7cf34b.png) | green, gain | WP2 stays (0.9 m), Qwen3 reads the green light and the launch profile scores 10 for go / hold; the selector's "go" overshoots (4.0): same decision, different decoder. |
| [03](../figs/s2_gohold/03_standstill_green_loss_dfce31ed.png) | green, loss | Green light but all three raters stay within 1 m; WP2 stays (10.0); Qwen3 says go and picks the left lane: 4.4 / 4.0. |
| [04](../figs/s2_gohold/04_standstill_red_loss_1bbb0008.png) | red, loss | Qwen3 reads red and holds; the top-rated trajectory and WP2 move 5-7 m within 5 s (10.0 -> 4.0). A red light now is not a hold for 5 s. |
| [05](../figs/s2_gohold/05_standstill_lead_loss_40bc0964.png) | lead, loss | Small loss (8.0 -> 6.8): hold where the raters creep 4 m behind the lead. |
| [06](../figs/s2_gohold/06_standstill_open_gain_1999b9e4.png) | open, gain | WP2 goes 8 m, the top-rated 5 m; hold lands in a lower-rated rater's region (5.6 -> 8.0), the oracle's F20 pick "creep" gets 10. |
| [07](../figs/s2_gohold/07_standstill_open_loss_3522a012.png) | open, loss | The top-rated trajectory does not move; Qwen3 sees nothing to wait for and launches (9.5 -> 4.0). |
| [08](../figs/s2_gohold/08_moving_Foreign_Object_Debris_gain_d851fd8c.png) | moving, selector gain | Debris: Qwen3's lane-left / go scores 8.3 against WP2's 4.7; the oracle takes the 1.2 m nudge at WP2's speed (9.0). |
| [09](../figs/s2_gohold/09_moving_Interections_gain_a9531d2b.png) | moving, selector gain | 0.8 m/s at a stop-controlled junction: WP2 stops (8.0), "creep" matches the top-rated 5.9 m (9.7). |
| [10](../figs/s2_gohold/10_moving_Others_gain_8f8d0a6d.png) | moving, selector gain 3 | The third-largest "gain" is already zero on seed 0 (4.0 -> 4.0): gains are rare among the 31 moving frames where Qwen3 leaves path C. |
| [11](../figs/s2_gohold/11_moving_Foreign_Object_Debris_loss_d6dd84e3.png) | moving, selector loss | 11 m/s in a cone-lined lane, WP2 at 10.0; Qwen3 shifts a lane to the left, across the double yellow line (4.0). At this speed the five lines converge in the image: the set-of-mark is hard to read beyond 30 m. |
| [12](../figs/s2_gohold/12_moving_Pedestrian_loss_ae776e97.png) | moving, selector loss | Pedestrian frame, WP2 at 10.0; nudge right + creep: 4.0. |
| [13](../figs/s2_gohold/13_moving_Single_Lane_Maneuvers_loss_9fcc91df.png) | moving, selector loss | WP2 at 10.0; lane-left + creep: 4.0. |

## Latency (`latency.md`; median over 469 frames, first 10 dropped)

| forward | images | prompt tokens | preprocess ms | model ms (median / p95) |
|:--|--:|--:|--:|:--|
| light / crossing / path choice | 1 | 1 160-1 400 | 40-49 | 241-257 / 343-378 |
| stop-controlled / lead | 2 | 2 220-2 240 | 79-81 | 393-401 / 532-542 |
| right of way / lane obstruction | 3 | 3 250-3 270 | 107-117 | 576-578 / 762-766 |
| go / hold or speed decision | 4 | 4 300-4 440 | 132-133 | 775-800 / 1 024-1 049 |
| D-direct (1 forward) / D-decomp (5 + 1) / selector (8) | | | 132 / 494 / 651 | 775 / 2 605 / 3 439 |

The card (`nvidia-smi` at 100 %) was shared with two training jobs of other lanes for the whole run; these are not clean service times.

## Checks

- Reproduces WP2 8.111, shipped 8.005, log 8.131, top-rated 9.587, WP2 on `stopped` 7.598; the continuous rows reproduce decision 164's O1 / O2
  (+0.722 / +0.296).
- `along` equals `pp_wod_diag.retime` (< 1e-9); a plan re-timed to its own arc length returns itself; F20's keep x follow candidate is WP2's plan.
- Qwen3's light / stop-controlled / lead / crossing answers equal wod-launch's stored labels on 479 / 479 frames (same prompts, same scoring).
- The overlay projection was checked by eye before the run (8 frames rendered, 3 inspected: the 3.5 m offsets land about one lane away).
- Splits from `jevdrive.data.splits`: `wod/val` for evaluation, `wod/r2-train` for the codebook and the anchors.

## Deviations from the pre-registration

- Addendum B (joint selection, clusters, budget 2 card-hours) was added at the coordinator's request before any score existed; part A is unchanged.
- The oracle's best F20 candidate is reported with ties resolved toward the plan (keep / follow first); the first render used the first index. Scores
  are unaffected.
- The continuous reference's `either` / `needs joint` columns were added after the first read (*post hoc*).
- The moving-frame figure rule yielded only two frames with a real gain; the third is kept as selected.
- Qwen3 ran 54 min on a contended card (planned 30-40 min).

## Caveats

- Open loop, 479 frames; 120 at standstill, 52 turn frames, clusters of 15-116. Cluster and context CIs are descriptive.
- **Best-of-K ceilings are optimistic by construction**: the maximum over K candidates of a score with three rater modes and a 4.0 floor grows with
  K for any diverse set (8 global train anchors already give +0.59). They bound what a selector can get; they do not say a learnable signal exists.
  Decision 164's out-of-fold nulls (speed bin, lead) still stand as the only evidence about learnability.
- The "class of the top-rated trajectory" oracle depends on the class edges and on one prototype per class; another decoder (e.g. scaling WP2's own
  profile within the class) could move it. 39 top-rated trajectories end before 5 s and count as "hold" in the moving regime (a label artefact);
  the highest speed bin's hold prototype is fitted on 4 train rows. 82-92 % of decoded prototypes fall in their own class.
- At standstill WP2's path is a point; every "go" candidate drives straight ahead, so standstill rows test the profile, not a path WP2 produced.
- Qwen3 saw the JPEG export of the three front cameras, not openpilot's model frames; candidate paths were drawn for seed 0 and the pick applied to
  both seeds. Zero-shot only: prompts were written once and not tuned (tuning on val would need a held-out split). The lead question is known to be
  unreliable (wod-launch: 52.5 % against hand labels), the new lane-obstruction question is biased to its first option, the right-of-way
  question was not hand-checked.
- Context classes come from the same VLM's zero-shot labels (wod-launch).
- No test submission; val rater labels were used for the oracles and for evaluation only.
