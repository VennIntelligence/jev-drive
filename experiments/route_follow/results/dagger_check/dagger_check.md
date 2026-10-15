# DAGGER-CHECK: do R / RL move at the policy's own handover states, and what would expert labels on all own states give? (2026-10-16)

Two offline reads behind RECOVERY-ENVELOPE ([../recovery_envelope/pilot.md](../recovery_envelope/pilot.md)). There is no training and no new simulation.
Code: `scripts/rf_dagger.py` (`a`, `b`), plus `rf_recov.py fwd` for the plans. Box dir: `$DATA_DIR/runs/route_follow/dagger_check/`. It holds
`rows.parquet` (one line per on1_L18RSG row: segment, contacts, disagreement, flags), `expert_L18RSG.npz` (the expert plan and info for
every row; it can serve as a label file) and `pool/` (job logs). Numbers: [read_a.json](read_a.json), [read_b.json](read_b.json).

## Read A: imitation error at the recovery rows of on1_RCVall

- **Rows.** 952 rows: 82 at k = h (the policy's own state at the handover decision) and 870 at k > h (states the expert drove to).
- **Splits.** Dev logs (crc32 rule) hold 143 rows (11 h / 132 post, 4 logs). Train logs hold 809 rows (71 / 738, 15 logs).
- **Target.** `fut`, the expert's served plan.
- **Plans.** Each plan is a forward pass on the pool's stored inputs (`rf_recov.py fwd`, one GPU job).
- **Replay check.** At both kinds of row, RSG-L18-s0's replayed plan matches the stored `served_model`: median 0.008-0.013 m, p95 <= 0.022 m.
- **Columns.**
  - ADE: mean pose distance, in m.
  - "4 s": the end-point error.
  - lat4 / dh4: the lateral (m) and heading (deg) parts of the 4 s error, in the expert pose's frame.
  - move: the mean pose distance of the checkpoint's plan from RSG's plan.
  - toward: the ADE reduction divided by move. 1 means every centimetre of change went toward the expert.
  - The CI is log-clustered, on the ADE change.

| split | kind | rows / logs | checkpoint | ADE | change [CI, m] | 4 s | lat4 | dh4 | move | toward |
|:--|:--|:--|:--|--:|:--|--:|--:|--:|--:|--:|
| dev | h | 11 / 4 | RSG-L18-s0 | 3.03 | - | 5.93 | 2.74 | 13.2 | - | - |
| | | | RCVR-L18-s0 | 2.99 | -1.2 % [-0.18, +0.11] | 5.97 | 2.65 | 11.2 | 0.34 | 0.10 |
| | | | RCVR-L18-s1 | 2.98 | -1.5 % [-0.18, +0.08] | 5.95 | 2.57 | 11.4 | 0.38 | 0.12 |
| | | | RCVRL-L18-s0 | 3.02 | -0.2 % [-0.31, +0.20] | 6.08 | 2.64 | 11.4 | 0.40 | 0.02 |
| | | | RCVRL-L18-s1 | 3.01 | -0.6 % [-0.37, +0.24] | 6.06 | 2.63 | 11.5 | 0.47 | 0.04 |
| | | | RCVRLF-L18-s0 | 3.01 | -0.7 % [-0.37, +0.16] | 6.01 | 2.66 | 11.4 | 0.40 | 0.05 |
| | | | RCVRLF-L18-s1 | 3.01 | -0.7 % [-0.48, +0.21] | 6.01 | 2.70 | 11.7 | 0.43 | 0.05 |
| dev | post | 132 / 4 | RSG-L18-s0 | 1.24 | - | 3.24 | 0.73 | 3.6 | - | - |
| | | | RCVR-L18-s{0,1} | 1.01 / 1.02 | -18.3 / -17.8 % (CI excl. 0) | 2.63 / 2.65 | 0.60 / 0.62 | 3.1 | 0.48 | 0.47 |
| | | | RCVRL-L18-s{0,1} | 0.98 / 0.97 | -21.1 / -21.4 % | 2.55 / 2.54 | 0.59 / 0.60 | 2.9-3.0 | 0.53-0.57 | 0.48 |
| | | | RCVRLF-L18-s{0,1} | 0.97 / 0.96 | -21.7 / -22.5 % | 2.55 / 2.52 | 0.59 / 0.57 | 3.0 | 0.53-0.57 | 0.50 |
| train | h | 71 / 15 | RSG-L18-s0 | 3.20 | - | 6.46 | 2.75 | 11.0 | - | - |
| | | | RCVR-L18-s0 | 2.73 | -14.8 % [-0.65, -0.36] | 5.29 | 2.03 | 7.9 | 0.59 | 0.80 |
| | | | RCVR-L18-s1 | 2.68 | -16.2 % [-0.72, -0.39] | 5.15 | 2.01 | 7.4 | 0.63 | 0.83 |
| | | | RCVRL-L18-s0 | 2.65 | -17.3 % [-0.77, -0.40] | 5.08 | 2.02 | 8.3 | 0.68 | 0.81 |
| | | | RCVRL-L18-s1 | 2.57 | -19.8 % [-0.83, -0.49] | 4.89 | 1.95 | 7.8 | 0.77 | 0.83 |
| | | | RCVRLF-L18-s0 | 2.63 | -17.8 % [-0.79, -0.42] | 5.03 | 2.02 | 8.0 | 0.69 | 0.83 |
| | | | RCVRLF-L18-s1 | 2.61 | -18.5 % [-0.80, -0.45] | 4.99 | 2.00 | 8.0 | 0.73 | 0.81 |
| train | post | 738 / 15 | RSG-L18-s0 | 1.28 | - | 3.24 | 0.90 | 6.0 | - | - |
| | | | R / RL / RLF (6 ckpts) | 0.90-0.93 | -26.9 to -29.4 % (CI excl. 0) | 2.33-2.43 | 0.61-0.64 | 4.1-4.3 | 0.56-0.61 | 0.58-0.65 |

Turning instrument (rf_recov gate_turn: 276 decisions 0 / 1 turning rows, 139 held scenes; pilot - RSG-L18-s0, deg):

| checkpoint | 4 s heading | equal-arc heading [CI] | 4 s arc (m) |
|:--|--:|:--|--:|
| RCVR-L18-s0 | +0.35 | +0.23 [-0.11, +0.62] | +0.02 |
| RCVR-L18-s1 | +0.16 | +0.03 [-0.23, +0.30] | +0.02 |
| RCVRL-L18-s0 | -0.77 | +0.21 [+0.01, +0.36] | -0.65 |
| RCVRL-L18-s1 | -0.47 | +0.53 [+0.20, +0.82] | -0.72 |
| RCVRLF-L18-s0 | -1.05 | +0.55 [+0.28, +0.79] | -1.04 |
| RCVRLF-L18-s1 | -0.76 | +0.95 [+0.57, +1.25] | -1.08 |

**Verdict A.**
- **Held-out handover states.** On held-out logs the trained policy does not move toward the expert at its own handover states.
  - At k = h on dev logs, ADE changes by -0.2 to -1.5 % and every CI includes 0.
  - The plans do change, by 0.34-0.47 m, but only 2-12 % of that change goes toward the expert.
  - The policy stays 3.0 m ADE from the expert there: 6.0 m at 4 s, 2.6-2.7 m lateral and 11-12 deg of heading at 4 s.
  - This is the thinnest cell: 11 rows, 4 logs.
- **Training handover states.** On the training logs' own k = h rows it moves, -15 to -20 %, and about 80 % of the movement goes toward the
  expert. It still ends 2.6-2.7 m ADE, 4.9-5.3 m at 4 s and about 2.0 m lateral away.
- **States the expert drove to (k > h).** These move twice as much in relative terms: -27 to -29 % on train logs, -18 to -22 % on dev logs,
  with every CI excluding 0. Expert-driven states sit 1.2-1.3 m ADE from the target to begin with, against 3.0-3.2 m at k = h.
  So what R transfers to held-out logs is the "continue the expert's trajectory" part. What it does not transfer is the recovery at the
  policy's own drifted state.
- **Turning.** The turning gain grows with the L / F terms (equal-arc +0.03 to +0.23 for R, +0.21 to +0.53 for RL, +0.55 to +0.95 for RLF).
  It comes together with shorter 4 s arcs (-0.65 to -1.08 m), i.e. slower plans.

## Read B: the rule expert planned from every row of on1_L18RSG

### Setup

- **Inputs.** XP-L12-E46 (lookahead 1.2 s, turn entry 4.6 m/s) plans from each row's rear-axle anchor, measured speed and simulation time.
  The rows of one scene share one expert session, in k order.
- **Packs.** Decision 319's stage packs (`rewind_onpol_all/{rest,s50,smoke}/pack`), so no packing job was needed. They cover 1 019 of the
  1 020 scenes; one scene has no pack (16 rows).
- **Frame and time check.** On the replay rows (k <= 1), the anchor lies 0.0 m (max 4e-7 m) from the pack's log path at the row's time.
  The expert's lateral offset from its chain has a median of 0.33 m and a p95 of 2.8 m; 1.1 % of rows are more than 3.5 m off.
- **Output.** 17 464 expert plans, no errors. The context-route proxy was not needed.
- **Geometry.** Contact uses rf_rdom's hinge geometry over 0.1-4 s: AgentHinge distance < 0 to a counted box, or a Pacifica corner with
  road sdf < 0. "Contact-free" also requires no contact at t = 0.
- **Stop lines.** The expert ignores stop lines and lights. Its plan was checked against rf_recov's must-stop bound alone (excess > 0.05 m/s
  counts as running the line). This applies to the 1 471 rows with a must-stop line 0-60 m ahead.
- **Units.** Logs are nuPlan log files (97). The CIs are clustered on them.

### Validity (own states, k >= 2, unless marked)

| segment | rows / scenes / logs | expert contact-free 4 s [CI] | expert agent / road | served contact-free 4 s [CI] | served agent / road | expert runs a must-stop line | expert plan shortened by its yield |
|:--|:--|:--|:--|:--|:--|--:|--:|
| own, all | 15 434 / 1 020 / 97 | 0.892 [0.875, 0.908] | 0.083 / 0.025 | 0.887 [0.872, 0.900] | 0.074 / 0.044 | 0.009 | 0.22 |
| replay k <= 1 | 2 040 / 1 020 / 97 | 0.925 [0.903, 0.944] | 0.051 / 0.024 | 0.938 [0.922, 0.953] | 0.011 / 0.051 | 0.006 | 0.28 |
| junction | 1 287 / 170 / 63 | 0.842 [0.795, 0.885] | 0.068 / 0.090 | 0.700 [0.642, 0.752] | 0.124 / 0.197 | 0.001 | 0.17 |
| turn approach | 1 789 / 242 / 68 | 0.918 [0.887, 0.944] | 0.064 / 0.017 | 0.849 [0.808, 0.882] | 0.094 / 0.061 | 0.009 | 0.20 |
| stop approach | 909 / 95 / 51 | 0.801 [0.753, 0.847] | 0.168 / 0.039 | 0.937 [0.898, 0.967] | 0.056 / 0.008 | 0.140 | 0.18 |
| straight | 10 085 / 815 / 96 | 0.900 [0.877, 0.919] | 0.082 / 0.018 | 0.910 [0.894, 0.925] | 0.068 / 0.025 | 0 | 0.24 |
| other (curves, off route) | 1 364 / 206 / 65 | 0.915 [0.885, 0.944] | 0.065 / 0.018 | 0.906 [0.875, 0.932] | 0.053 / 0.043 | 0 | 0.17 |

The expert is cleaner than the policy only in the junction (0.84 vs 0.70) and on the turn approach (0.92 vs 0.85). On straight road it is
equal (0.90 vs 0.91). On stop approaches it is worse (0.80 vs 0.94): it ignores the line, runs it on 14 % of these rows, and touches the
queue ahead (agent contact 0.17 vs 0.06).

### Disagreement (served minus expert, expert pose frame; medians of |.| unless signed)

- **Columns.**
  - d1 / d2 / d4: the distance at 1 / 2 / 4 s, in m.
  - lon4 signed: the mean, with its CI. Negative means the served plan is behind the expert, i.e. slower.
  - turn-signed dh4: rows where the expert turns more than 10 deg in 4 s. Negative means the policy turns less than the expert.
  - share > line: the share of rows with d2 > 0.5 m or |dh4| > 5 deg.

| set | rows | d1 | d2 | d4 | lat 2 s / 4 s | lon 2 s / 4 s | lon4 signed, mean [CI] | dh4 | turn-signed dh4: rows, median | share > line |
|:--|--:|--:|--:|--:|:--|:--|:--|--:|:--|--:|
| own, all | 15 434 | 0.81 | 2.27 | 5.58 | 0.23 / 0.46 | 2.09 / 5.41 | -0.97 [-1.53, -0.42] | 1.3 | 4 243: -2.9 | 0.94 |
| replay k <= 1 | 2 040 | 0.71 | 1.20 | 3.02 | 0.20 / 0.44 | 1.01 / 2.63 | +1.94 [+1.29, +2.65] | 1.7 | 596: -1.1 | 0.84 |
| junction | 1 287 | 0.86 | 2.30 | 5.29 | 0.92 / 1.49 | 1.74 / 4.75 | +1.72 [+0.72, +2.69] | 5.3 | 1 167: -2.5 | 0.97 |
| turn approach | 1 789 | 0.71 | 2.01 | 4.70 | 0.24 / 0.74 | 1.87 / 4.42 | -0.86 [-1.76, +0.20] | 6.2 | 951: -5.1 | 0.96 |
| stop approach | 909 | 0.89 | 2.57 | 5.70 | 0.10 / 0.25 | 2.53 / 5.62 | -4.52 [-5.11, -3.90] | 1.1 | 62: -13.0 | 0.96 |
| straight | 10 085 | 0.82 | 2.36 | 6.11 | 0.20 / 0.36 | 2.22 / 5.96 | -1.13 [-1.91, -0.30] | 0.9 | 1 277: -3.6 | 0.93 |
| other | 1 364 | 0.70 | 1.76 | 3.92 | 0.46 / 1.00 | 1.37 / 3.47 | -0.19 [-1.03, +0.61] | 3.8 | 786: -1.4 | 0.95 |

- **Mostly speed.** The disagreement is mostly longitudinal.
  - The expert's cruise speed is the human's speed at the station, with a 3 m/s floor. Its plans run ahead of the policy's on straight road
    (-1.1 m at 4 s on average) and on stop approaches (-4.5 m, because it ignores the line).
  - In the junction they run behind the policy's (+1.7 m), because the expert holds 4.6 m/s through the turn.
  - Lateral gaps are small outside turns: 0.2 m at 2 s, 0.4 m at 4 s.
- **Turning.** Where the expert turns, the policy turns less: -2.5 deg median in the junction, -5.1 deg on the approach.
- **The 0.5 m / 5 deg line does not select.** It flags 94 % of own rows, almost all of them on speed.

### Failing vs completing turns

- **Windows.**
  - fail_kref: decision 319's event / off units, rows k_ref - 6 .. k_ref - 1. These are the units' identity-run k_ref mapped onto the stored
    18-decision rollouts, rows k <= 17 only.
  - fail_entry / done_entry: the 6 decisions before the rear axle crosses the strict turn's entry, for failed and completed units.
  - rdom_fail / rdom_ctrl: decision 318's states on the same stored rollouts.

| window | units | rows / logs | served contact-free 4 s | expert contact-free 4 s | d2 [CI] | abs dh4 [CI] | lat4 | turn-signed dh4, mean [CI] | expert fixes a contact (rows) |
|:--|--:|:--|:--|:--|:--|:--|--:|:--|--:|
| fail_kref | 73 | 386 / 45 | 0.409 [0.321, 0.502] | 0.811 [0.728, 0.877] | 3.16 [2.88, 3.46] | 13.9 [11.0, 17.0] | 3.10 | -2.4 [-6.7, +2.2] | 168 |
| fail_entry | 60 | 296 / 38 | 0.780 [0.703, 0.850] | 0.926 [0.865, 0.973] | 1.84 [1.60, 2.13] | 19.0 [16.3, 21.9] | 1.46 | +3.6 [-1.5, +9.0] | 53 |
| done_entry | 67 | 336 / 37 | 0.863 [0.820, 0.909] | 0.914 [0.859, 0.971] | 1.67 [1.41, 1.89] | 18.2 [15.4, 21.2] | 1.39 | +4.0 [-0.9, +10.9] | 39 |
| rdom_fail | 62 | 362 / 42 | 0.359 [0.271, 0.449] | 0.820 [0.746, 0.888] | 3.26 [2.89, 3.65] | 16.3 [12.3, 20.6] | 2.97 | -1.0 [-6.5, +4.8] | 182 |
| rdom_ctrl | 54 | 761 / 34 | 0.845 [0.791, 0.903] | 0.879 [0.817, 0.940] | 2.65 [2.43, 2.89] | 9.4 [7.6, 11.1] | 1.17 | +3.0 [+1.0, +5.2] | 74 |

- **In the 3 s before a turn failure:**
  - The policy's own plan is contact-free on 0.36-0.41 of rows. The expert's plan from the same state is contact-free on 0.81-0.82.
  - The two plans differ by 3.2 m at 2 s, 3.0-3.1 m lateral and 14-16 deg of heading at 4 s.
- **At the approach (6 decisions before the entry),** failed and completed turns look alike. Served plans are contact-free 0.78 vs 0.86,
  d2 is 1.8 vs 1.7 m and |dh4| 19 vs 18 deg. The expert's heading differs from the policy's by about 18 deg in both. At the approach, the
  expert's disagreement is a difference of driving style that does not mark the turns that fail. The separation appears only within the
  last 3 s.

### Yield (candidate DAgger labels)

- **Column definitions.**
  - "registered" = the expert plan is contact-free over 4 s, stop-legal, AND d2 > 0.5 m or |dh4| > 5 deg.
  - "lateral" = contact-free and stop-legal, AND |lat2| > 0.5 m or |dh4| > 5 deg.
  - "fixes" = the expert plan is clean where the served plan contacts. The bracket is that row count's share of the served-contact rows.
  - Counts are rows / scenes / logs. "train" counts drop the dev logs.

| set | registered | registered, train logs | registered, plan not yield-shortened | lateral | lateral, train logs | fixes a contact |
|:--|:--|:--|--:|:--|--:|:--|
| own, all | 12 832 / 1 007 / 97 | 11 190 / 883 / 81 | 10 491 | 5 403 / 649 / 95 | 4 818 | 1 156 / 338 / 88 (0.66) |
| replay k <= 1 | 1 555 / 910 / 97 | 1 360 / 794 / 81 | 1 140 | 731 / 431 / 88 | 656 | 97 / 75 / 44 (0.76) |
| junction | 1 054 / 163 / 62 | 962 / 149 / 55 | 902 | 923 / 160 / 62 | 849 | 258 / 85 / 48 (0.67) |
| turn approach | 1 574 / 235 / 68 | 1 438 / 215 / 58 | 1 297 | 1 067 / 210 / 68 | 966 | 208 / 78 / 44 (0.77) |
| stop approach | 617 / 81 / 48 | 574 / 75 / 44 | 484 | 107 / 31 / 24 | 102 | 47 / 13 / 13 (0.82) |
| straight | 8 407 / 787 / 96 | 7 164 / 678 / 81 | 6 801 | 2 488 / 421 / 90 | 2 172 | 569 / 192 / 74 (0.63) |
| other | 1 180 / 187 / 62 | 1 052 / 167 / 51 | 1 007 | 818 / 150 / 57 | 729 | 74 / 29 / 23 (0.58) |
| fail_kref | 311 / 67 / 43 | 293 / 63 / 39 | 262 | 301 / 66 / 42 | 286 | 168 / 50 / 34 |
| rdom_fail | 290 / 58 / 41 | 259 / 52 / 37 | 212 | 267 / 56 / 40 | 242 | 182 / 45 / 34 |

- **Registered yield.** It is 12 832 own rows over 1 007 scenes. That is 156x the 82 k = h rows of R and 12x decision 319's 1 064 label rows. Two thirds of it is the expert's speed
  profile on straight road.
- **Lateral subset.** 5 403 rows; 1 990 of them sit in the junction or on the turn approach (at most 370 scenes).
- **Contact-fix subset.** 1 156 rows; 466 in the junction or on the approach, and 168 in the 3 s before a turn failure.

## Caveats

- **Expert speed.** The expert's longitudinal label is the human's speed with a 3 m/s floor and 4.6 m/s through turns.
  - It ignores lights and stop lines, so stop-approach rows carry a "do not brake yet" label: -4.5 m at 4 s against the policy, which
    brakes early like the human.
  - Its 3 s yield rule shortens 22 % of the plans, which stop behind boxes in the swept path, including moving leads.
  - The registered disagreement line therefore mostly measures speed. A label set needs a speed policy of its own: keep or mask the
    longitudinal part, and exclude the stop approach.
- **Expert contact.** On straight road the expert plan touches agents slightly more often than the policy's (0.082 vs 0.068), because it is faster.
- **Read A sample.** The k = h dev cell has 11 rows from 4 logs.
- **Window mapping.** The fail_kref / fail_entry windows map decision 319's identity-run k_ref onto the stored rollouts. Units whose k_ref is
  past decision 17 contribute fewer rows. rdom_fail is the exact same-rollout cut and agrees with it.

## Cost

- GPU forward job: 1.6 min wall, one card, 12 GB booked; it covered the 7 checkpoints x 3 row sets.
- CPU: Read A, 2 x under 1 min on 4 cores. Read B, 2 x about 1 min on 24 cores (the expert on 17 474 rows took 13 s).
- Total under 0.05 card-h and about 1 core-h. Wall time 07:43-07:52 CST.
