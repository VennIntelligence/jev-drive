# NR1STOPPT: "do not pass the human's stop point" on approach states (2026-10-19)

Prereg: NR1 prereg supplement 18 ([../../plans/2026-10-17-nr1-prereg.md](../../plans/2026-10-17-nr1-prereg.md), 59337035), written before any read.
Code: `lib/nr1_free.py` (`stop_label`, `stoppt_*`), `rm_train --on-stoppt`, `scripts/stoppt.{py,sh}`. Box: `$DATA_DIR/runs/defect_fix/stoppt/`.
One seed throughout. Decision 386.

**Answer.** The offline gate passes, but the 1 000-step pilot fails in both arms. The term barely fires in training: 0.43 % of the drawn
pool rows in arm A and 0.99 % in arm B. The PAI overrun count does not move (11 to 11 in both arms). The probe fails its registered rule
(TOK - LEAD < 0.05), but this is **not an input problem**. The approach stop is already readable from the model's own heads, at AUC 0.92
for the lead head + kinematics and 0.94 for the tokens. A token probe fires in time in 20 of the 23 overrun scenes. So the term can only
learn a prior. On top of that, the training pool holds almost no overrun states.

## Step 1: probe (own states; scene / log clustered AUC, B = 2000)

**Data.**
- A0R is a full rerun of A0 (NTV0-L18-s0 + ACC + GAP) on A0's 8 chunk lists. 280 of 281 scenes have a score identical to A0.
- P160 is the practice pool (RSG-L18-s0, train160).
- nuPlan rows come from on1_JT's S rows.
- Positives: the time to the human's stop point at the ego's current speed is tau in [2, 4] s. Negatives: no stop point within 8 s.

| set (no lead, tau [2, 4]) | pos (clusters) | KIN | LEAD | PLAN | TOK | TOK - KIN | TOK - LEAD |
|---|--:|--:|--:|--:|--:|--:|--:|
| **PAI own (primary)** | 2011 (87) | 0.780 | 0.917 | 0.926 | **0.943** [0.92, 0.96] | +0.163 [+0.12, +0.21] | **+0.026 [+0.00, +0.05]** |
| PAI own, stop line | 677 (34) | 0.724 | 0.942 | 0.940 | 0.950 | +0.225 | +0.007 [-0.03, +0.04] |
| PAI own, other | 1334 (56) | 0.808 | 0.905 | 0.918 | 0.940 | +0.132 | +0.036 [+0.00, +0.08] |
| nuPlan own (on1_JT S) | 1110 (20) | 0.754 | 0.827 | 0.833 | 0.823 | +0.070 | -0.003 [-0.05, +0.04] |

- **Registered verdict: not readable beyond kinematics + lead.** The term can only learn a prior.
- In substance the cue is well present. It sits in the model's own lead and plan heads. This is far above d362's 0.79, which used a
  y5 label and the RSG heads.
- First-fire table on the 23 A0 overrun scenes (10 % FPR, 3 decisions in a row, early enough for 3 m/s²): TOK fires in time in 20 / 23
  scenes, LEAD in 16 / 23, KIN in 5 / 23.
- Full tables: [probe.md](probe.md).

## Step 2: offline gate (passed)

| line | value |
|---|---|
| G1: the human's two 4 s paths fire | 0 / 26 851 nuPlan rows, 0 / 3 138 PAI rows, 0 / 7 578 A0 decisions (by construction) |
| G2: A0 overrun decisions fired (>= 50 %) | **95.3 %** (2 559 / 2 684); stop-line scenes 96.9 %; 21 / 23 scenes; 20 in time at 3 m/s² |
| G3: the human does not stop within 8 s (< 2 %) | A0 0.05 % (11 / 22 917); nuPlan S rows 0.006 % |
| G4: switch off is bit-identical | identical (60 steps eager, 59337035 against 5ebb882e) |

**Qualifying rows.** nuPlan has 26 851 of 87 012 S rows; PAI has 3 138 of 25 634. The plan fires on few of them:

| rows | stored plans fire | during training fire |
|---|--:|--:|
| nuPlan pool | 1.1 % | |
| PAI pool | 3.3 % | |
| arm A pool draw | | 0.43 % |
| arm B pool draw | | 0.99 % |

Files: [gate.json](gate.json), [labels.json](labels.json).

## Step 3: 1 000-step pilot against JTNS1K-L18-s0 (native reads)

| | control | A STOPPT1K (term) | B STOPPTP1K (term + PAI rows) |
|---|--:|--:|--:|
| held 460 zeros (<= +2) | 42 | 42 | 39 |
| b launch delay other_lead / red_queue (s, <= +0.15) | 0.57 / 0.77 | 0.52 / 0.84 | 0.44 / 0.77 |
| b launched (>= 0.95x) | 0.729 / 0.657 | 0.729 / 0.686 | **0.667 / 0.571 FAIL** |
| held front collisions / arena AF-front (<= +2) | 7 / 2 | 5 / 2 | 5 / 3 |
| **PAI half paired (>= +0.01)** | 0.6733 | **-0.0076 [-0.033, +0.013] FAIL** | **-0.0057 [-0.054, +0.043] FAIL** |
| PAI d381 overrun zeros | 11 | 11 (same scenes) | 11 (same scenes) |
| arena subset paired (>= 0) | 0.8618 | +0.0018 [-0.0004, +0.0047] | +0.0058 [-0.010, +0.021] |
| held 460 mean, paired (description) | 0.8268 | -0.0064 [-0.012, -0.003] | -0.0074 [-0.018, +0.001] |

**Both arms: FAIL.** No arm-specific axis leaked except B's launch count. Arm A is a near no-op: the plan rarely reaches past a stop
point on the training rows. Files: [pilot_A.md](pilot_A.md), [pilot_B.md](pilot_B.md).

## Limits
- One seed, 1 000 steps, one dose (W 0.004).
- G2 / G3 use stored plans of other policies.
- The probe's heads are route-off forwards.
- PAI rows come only from the 160 old scenes.
