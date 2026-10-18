# LAUNCH-DIAG: what in the JTN stage teaches the late launch (2026-10-18/19)

Question: on wait set v2 variant b (the logged human launches at 3.0 s; lead classes other_lead / red_queue), V2 = RTB1ARO-L18-s0 launches
on time, but JTN-L18-s0 and all three RTJA arms (V2 base + the JTN stage + P0 arrival KD) launch 0.3-0.8 s late and only 74-88 % of the time
(`dose.md`). Which component of the JTN stage (pool on1_JT, mixed mode, on-k 70 = 32 S rows + 38 trainval imitation rows drawn by JT00,
history-speed noise, warm-up 500, lr floor 0.1, 9 500 steps) installs this?

**Verdict: the 38 trainval imitation rows per batch (the JT00 draw) install the late launch.** The S-row hinges (agent with the logged boxes,
road), the side-buffer term, the other lateral terms, the history-speed noise and the training length do not. The effect is a low-speed
bias: the plan from standstill gets shorter. The same bias makes JTN wait better on variant a, so the wait gain and the late launch are one
shift, not two. Confidence: medium-high that the trainval rows are the cause. Three independent lines agree: natural ablations in stored
checkpoints, per-term influence at frozen weights, and one registered 1 000-step removal. Confidence is medium on which trainval classes do
it (starting and braking rows): that comes from the influence measurement only.

Code: `experiments/defect_fix/scripts/launch_diag.{py,sh}`, `rm_train.py --diag-infl / --diag-freeze` (monitoring only; off = unchanged).
Box outputs: `$DATA_DIR/runs/defect_fix/launch_diag/` (lead.json, dump/, replay/, confirm_cmd.diff), the influence runs'
`runs/route_mem/train-LDF-<init>/`, and the new wait-set tables `runs/alpasim_proxy/wait_set/reads/<tag>/table/`.

## 1. Closed loop: who launches late (wait set v2, lead classes; no training)

Wait-set reads made for this diagnosis (`wait_set.sh readc`, priority 11): RB0-F-s0, NR1HV2-L18-s0, JTNS1K / JTNN1K-L18-s0, JTN-L18-s1,
plus LDNTV1K (section 4). Tables were collected for the stored but uncollected reads (NR1HV, NR1H*, RTB1AR-F, NR1FB05 / BALL) with
`df_wscollect.py`; chunked reads whose collect step stopped on the three windows without a logged human launch used the same patched set
(`ws_set_patched.json`). JT10's read did not finish in time and is not used.

| checkpoint | recipe (what differs) | a wait other_lead / red_queue | b launched | b delay med s |
|:--|:--|--:|--:|--:|
| RB0-F-s0 | JTN's start | 0.64 / 0.44 | 0.98 / 0.94 | -0.16 / -0.18 |
| RTB1AR-F-s0 | RTJA's start | 0.83 / 0.74 | 0.96 / 0.97 | -0.18 / -0.18 |
| RSG-L18-s0 | RB0 + L18 (13 S rows, 2 500 steps, no noise / lateral) | 0.50 / 0.38 | 0.98 / 0.94 | -0.23 / -0.13 |
| RTB1ARO-L18-s0 (V2) | RTB1AR + the same L18 | 0.83 / 0.72 | 1.00 / 0.97 | -0.03 / -0.08 |
| NR1HV-L18-s0 / s1 | V2 base, **32 S rows**, 5 000 steps, lateral 0.5v2 **incl. side 0.5**, **noise**, no trainval rows | 0.83 / 0.72, 0.81 / 0.72 | 0.96-0.98 / 0.94 | -0.03 / -0.08 (both) |
| NR1HV2-L18-s0 | V2 base, **64 S rows, 10 000 steps**, lateral 1.0v2 (side 1.0), noise, no trainval rows | 0.83 / 0.69 | 0.96 / 0.91 | -0.06 / -0.13 |
| JTNS1K-L18-s0 | JTN recipe, **1 000 steps** (warm-up 50) | 0.94 / 0.92 | 0.73 / 0.66 | **+0.57 / +0.77** |
| JTNN1K-L18-s0 | JTNS1K with side 0 | 0.94 / 0.92 | 0.73 / 0.66 | **+0.57 / +0.82** |
| JTN-L18-s0 / s1 | JTN, 9 500 steps | 0.89 / 0.92 (both) | 0.81 / 0.80 (both) | +0.37 / +0.32, +0.42 / +0.32 |
| RTJA0 / 05 / 1-L18-s0 | V2 base + JTN stage + KD 0 / 0.5 / 1 | 0.92-0.89 / 0.92-0.87 | 0.83-0.88 / 0.74 | +0.52..+0.37 / +0.77..+0.32 |
| NR1B10-F-s0 | base stage with trainval imitation rows (D10, incl. waits) | 0.92 / 0.87 | 0.96 / 0.89 | +0.24 / +0.02 |

What this rules out:
- **S-row hinge dose and training length.** NR1HV2 has twice JTN's S rows per batch, 10 000 steps and about JTN's integrated lr, and it
  is on time. JTN's lateness is already full at 1 000 steps (JTNS1K, larger than at 9 500).
- **Side buffer.** JTNS1K and JTNN1K are the same window for window: +0.57 / +0.77 against +0.57 / +0.82 s. NR1HV / HV2 carry side 0.5 / 1.0
  and are on time.
- **History-speed noise.** NR1F, NR1HV and NR1HV2 all train with it and are on time.
- **Pool YS / Y.** Its S rows have the same composition as RB0 / RTB1's (ego < 0.5 m/s 4.1-7.1 %, launch kind 3.7-3.8 % in every
  sub-pool).

What remains distinctive of the JT stage: the trainval imitation rows (38 / batch, so navtrain falls from 83 to 45 rows per batch), and the
warm-up / floor schedule.

The lateness does not depend on the lead's distance. Against V2, JTN's per-window delay is +0.30 / +0.50 / +0.50 / +0.70 s by lead-gap
quartile (gap at the window start 3.0 / 4.0 / 5.1 m quartile cuts); RTJA0's is +0.70..+0.80 s in every quartile. A front-margin hinge
against a close logged lead would concentrate the delay at small gaps. A uniform offset is a standstill prior.

## 2. Fixed inputs: where the plan collapses (dump replay, no training)

The 83 lead-class b windows were rerun with `SH30_DUMP` under V2, JTN and RTJA0 (`ld-dump-*`; the model frames and the fed ego row of every
decision). `launch_diag.py replay` then fed each checkpoint the same inputs. The route memory was zeroed for every tag (the `--rm-drop` input,
layout-free, so the anchored RTJA arms are comparable); the dumping run's own memory gives the same ordering. The lead's movement onset comes
from the raw log (`launch_diag.py lead`: first frame more than 0.25 m from its start; median 1.40 s into the window, so the human moves 1.4-2.6 s
later). The table gives the mean plan 4 s arc in m by time since the lead moved:

| checkpoint | JTN's states: < -0.5 s | -0.5..+0.5 | +0.5..+1.5 | V2's states: < -0.5 | -0.5..+0.5 | +0.5..+1.5 |
|:--|--:|--:|--:|--:|--:|--:|
| RB0-F-s0 | 4.10 | 5.46 | 7.58 | 4.16 | 5.95 | 9.19 |
| RTB1AR-F-s0 | 2.79 | 4.94 | 7.52 | 2.91 | 5.35 | 9.00 |
| RSG-L18-s0 | 4.16 | 5.52 | 7.62 | 4.22 | 6.00 | 9.20 |
| V2 | 3.18 | 5.12 | 7.57 | 3.30 | 5.61 | 9.06 |
| NR1HV-L18-s0 | 3.31 | 5.21 | 7.55 | 3.41 | 5.66 | 9.00 |
| NR1HV2-L18-s0 | 3.37 | 5.25 | 7.64 | 3.48 | 5.75 | 9.12 |
| JTNS1K / JTNN1K | 2.66 / 2.65 | 4.27 / 4.26 | 7.02 / 7.00 | 2.80 / 2.80 | 5.08 / 5.06 | 9.04 / 9.03 |
| JTN-L18-s0 / s1 | 2.45 / 2.45 | 4.15 / 4.08 | 7.00 / 6.94 | 2.62 / 2.62 | 4.97 / 4.91 | 8.97 / 8.94 |
| RTJA0 / 05 / 1 | 2.18 / 1.93 / 1.92 | 3.91 / 3.79 / 3.77 | 6.73 / 6.71 / 6.71 | 2.32 / 2.09 / 2.07 | 4.65 / 4.52 / 4.50 | 8.73 / 8.66 / 8.65 |
| P0 | 0.82 | 1.72 | 3.07 | 0.88 | 2.05 | 4.12 |

On identical inputs, at the moment the lead moves, the JTN-stage checkpoints plan 0.85-1.35 m (17-26 %) shorter than V2 / NR1HV / NR1HV2 on JTN's states (0.55-1.1 m on V2's).
The gap is largest at standstill (JTN's own states, where the car has not crept) and shrinks once the car rolls (V2's states at +0.5..+1.5 s:
-0.1 m for JTN, -0.3 m for RTJA). So the plan does not fail to read the lead's motion; it carries a speed-dependent shortening near zero
speed. Closed loop compounds it: a shorter plan gives a slower car and then a shorter plan again. The JTN plan in its own closed loop is
5.0 against V2's 6.4 m at 0-0.5 s after the onset (`launch_diag.py cl`).

The KD arms (RTJA05 / 1) are shorter still at standstill (P0 is the arrival teacher and P0 barely moves: 1.7 m). KD shortens the delay
through the rolling states, not through the launch moment.

## 3. Per-term influence on launch-like rows (frozen weights, no training)

`rm_train --diag --diag-infl 128 --diag-freeze` ran JTN's exact loss (command from `DRY=1 jt.sh train`, only `--init` replaced; eager;
400 steps; every group's lr 0 so the weights stay at the init while Adam's moments accumulate). Every 5 steps, for fixed probe rows, it took
the gradient of the mean plan 4 s arc and dotted it with each loss term's trunk gradient under Adam's current preconditioner at the base lr.
The result, d1k, is the arc change per 1 000 steps that the term alone predicts. It is first-order and nominal: use it for sign and relative
size only. The imitation term is split by the rows that carry it (navtrain, yaw-rate, trainval and each trainval draw group). The split's
residual against the full `imit` is 0. `imit_clean` is the same rows without the history-speed noise.

The probe is launch_tv: 128 trainval rows where the human stands at t0 and launches within 4 s. Values are mean +- s.e. over 15 logged points
(each the mean of 5 samples):

| init | probe arc4 (m) | imit_tv | of which starting / braking | imit_nav | agent_S | road_S | lat_side / lat_onc | noise (imit - imit_clean) |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|
| RB0-F-s0 | 6.63 | **-630 +- 40** | -364 / -208 | +12 +- 56 | +12 +- 14 | +32 +- 15 | +2 / 0 | +115 |
| RTB1AR-F-s0 | 6.01 | **-236 +- 32** | -131 / -123 | -45 +- 35 | +4 +- 10 | +37 +- 14 | -0 / +2 | +41 |
| V2 | 6.24 | **-311 +- 34** | -224 / -120 | -24 +- 41 | -17 +- 8 | +31 +- 12 | +2 / - | +43 |
| NR1HV-L18-s0 | 6.15 | **-357 +- 38** | -226 / -136 | -19 +- 34 | +5 +- 10 | +17 +- 9 | +2 / - | +36 |
| JTN-L18-s0 | 5.52 | **-250 +- 57** | -116 / -188 | +98 +- 51 | +17 +- 14 | +4 +- 7 | +7 / -8 | +62 |

The same ordering holds on the other probes: the standing trainval rows (stand_r), the fitted pool slice's launch / still S rows (launch_S,
still_S). For example, at RB0 imit_tv gives -584 on launch_S and -479 on stand_r, and navtrain imitation gives 0 +- 54 and +8 +- 33.
- The trainval imitation rows are the only term that pulls the standstill / launch plan short, at every checkpoint. Their pull is 6-20 times
  that of any constraint term. Within them the pull comes from the `c_starting` (about 5 rows / batch) and `c_braking` (about 12.5 / batch)
  draw groups; `c_cruising` and `c_turning` are about 0. JT00 draws braking at 33 % and starting at 14 % of the trainval rows, against
  about 10 % / 5 % in the trainval pool's natural mix.
- Navtrain imitation at its natural mix nets to about 0. The S-row agent hinge against the logged non-reactive boxes is about 0 (-17..+17);
  the road hinge and the side term slightly lengthen the plan. The history-speed noise works against the shortening (imit is less negative
  than imit_clean).
- The total is the trainval term: at RB0, -594 against -630 for imit_tv on launch_tv.

## 4. Confirmatory 1 000-step arm (NR1 prereg supplement 13, written before the job)

`LDNTV1K-L18-s0` = JTNS1K-L18-s0's command with the trainval rows removed. The `confirm_cmd.diff` shows only three changes: the tag,
`--on-k 70 -> 32`, and the draw file `JT00 -> JT00S` (trainval weight 0). Navtrain therefore has 83 rows per batch, as in NR1HV. Everything
else is the same: the JT pool's S rows, 1 000 steps, warm-up 50, the lateral terms with side, noise, seed 0. Read: wait set v2 (job ids in
`$O/jobs.txt`).

| | a wait other_lead / red_queue | b launched | b delay med s | a / b mean | a / b zeros | dev ADE @1000 |
|:--|--:|--:|--:|--:|--:|--:|
| JTNS1K-L18-s0 (control) | 0.94 / 0.92 | 0.73 / 0.66 | +0.57 / +0.77 | 0.979 / 0.654 | 3 / 1 | 0.5456 |
| **LDNTV1K-L18-s0** | 0.31 / 0.28 | 0.98 / 0.94 | **-0.33 / -0.28** | 0.945 / 0.949 | 8 / 4 | 0.5336 |
| V2 (reference) | 0.83 / 0.72 | 1.00 / 0.97 | -0.03 / -0.08 | 0.959 / 0.927 | 6 / 3 | - |

Registered reading: "established" required both lead-class delays within V2 +- 0.15 s and launches >= 90 %; "refuted" required both delays
>= +0.40 s. LDNTV1K removes the lateness completely: -0.90 / -1.05 s against the control, launches 98 / 94 % against 73 / 66 %. It
overshoots to the early side, though: 0.30 / 0.20 s earlier than V2, at RB0 / RSG's early-launch level, outside the band's lower edge.
**By the letter of the rule the outcome is inconclusive** (neither established nor refuted). On the mechanism, which was the question, the
arm is decisive: without the trainval rows the same stage from the same base launches early, like its base (RB0 -0.16 / -0.18). Waiting
collapses with it (a 0.94 -> 0.31): the trainval rows supplied both JTN's wait and its late launch.

## 5. What it implies

- The late launch is not a constraint-term or pool problem. It is the imitation target the JT00 trainval draw adds: a dose of low-speed
  human rows (people pulling away gently, braking) about three times their natural share. The base's open-loop launch plan overshoots those
  humans (launch probe 6.63 against the human 6.02 m at RB0), and fitting them pulls the standstill plan to about the human's (5.52 m at JTN).
  A plan calibrated to the human in open loop is late in closed loop, because the car only gets the first part of each 0.5 s re-plan. V2 /
  NR1HV keep a +0.1-0.2 m open-loop margin (6.15-6.24 m) and launch on time.
- Fix, untested: keep the stage but take the trainval rows out, or draw them at the natural class mix without the starting / braking
  oversampling. That brings back the on-time (in fact early) launch, and the variant-a wait goes with it. On these inputs "wait at
  standstill" and "launch on time" sit on one standstill-bias axis; no trainval mix tested here moves one without the other. Getting both
  needs a launch cue the plan conditions on (lead motion), not a different dose of the same rows. Arrival KD (RTJA) does not do it either:
  it shortens the rolling states.
- For the RTJA line: a fourth KD dose cannot fix the delay. Only the trainval part of the stage can.

## 6. Limits

- One seed for every new checkpoint read here (JTN has a second seed, which replicates). Each wait-set class has 35-48 windows.
- The influence numbers are first-order and per term at fixed weights (Adam preconditioner, nominal peak lr, no clip or momentum
  interaction). They give sign and relative size, not a trajectory. The navtrain term was not split by class, so "starting / braking rows"
  is established for the trainval draw only.
- LDNTV1K changes the navtrain share too (45 -> 83 rows per batch). Its result is "the trainval rows, as drawn", not "trainval rows of any
  composition".
- The dump replay uses each dumping model's states. The comparison is exact on inputs but does not cover states the other models would
  reach.
