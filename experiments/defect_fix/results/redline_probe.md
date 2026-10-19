# REDLINE-PROBE: do the frozen tokens read "red lamp at the line / must stand"? (2026-10-19)

Diagnosis, no training. Prereg [plans/2026-10-19-redline-probe-prereg.md](../plans/2026-10-19-redline-probe-prereg.md), committed before any
output. Code: `scripts/redline_probe.{py,sh}` (select / dump / feats / probe / report). Pool jobs: 19 dump reads 1019-101514-8cd1 ..
1019-101522-b012, feats 1019-101522-b4fc, probe 1019-101523-d986 (priority 12, about 15 min wall in total). Box outputs:
`$DATA_DIR/runs/defect_fix/redline_probe/` (windows.csv, feats.npz, probe/{probe.json, scores.npz}, dump/ 32 GB with only the k = 0 / 3 npz kept).
Full tables: [redline_probe.json](redline_probe.json); `redline_probe.py report --src <dir>` prints them.

**Answer.** By the registered rule the lamp is **not readable**, which points to an input problem. In substance: the tokens separate "the
human stands at a red line with no lead" from the negatives at AUC 0.975 [0.94, 1.00]. But KIN plus the policy's own lead head already
reaches 0.944 [0.88, 0.99]. The margin is +0.031 [+0.00, +0.08], below the +0.05 the rule requires. What the tokens carry is "standing, no
vehicle ahead", not the lamp. Three checks show this:
- The probe calls "must stand" on 100 % of no-light / no-lead static windows and on 100 % of unknown-light static windows.
- It cannot tell the red-line windows from the 40 green launches (P vs G 0.531 [0.44, 0.64]).
- The plan reads the scene the wrong way round. Its own decision-0 arc ranks P as the *least* standing class (AUC 0.21).

The information needed to stand in these 239 windows (standing, no lead) is already in the policy's readouts. The lamp colour, which is the
only thing that would separate the 40 green launches, is not readable beyond that.

## Setup (as registered)

- Windows: the 2 228 common arena held windows (`native_gap_report.common`), classes by `native_gap_report.cue` plus census light labels.
  - P: red line, no lead, static. 239 windows, 18 logs.
  - N1: close lead <= 8 m, static. 477 windows, 27 logs.
  - N2: no light, no lead, static. 38 windows, 6 logs.
  - N3: moving (v0 >= 2 m/s) towards a light line 0-60 m ahead that is never red in the window. 65 windows, 25 logs.
  - Reported separately: G, launches at a green with no lead (40 windows, 19 logs); U, unknown-state light, static (75); F, far lead 8-30 m,
    static (49).
- Frames: JTN-L18-s1 native arena closed loop with `SH30_DUMP` (970 new windows; 13 reused from native_gap/dump/JTN1).
  - Decision k = 0 is force-GT, with the ego on the human's pose. One keyframe exists: slot 7 is real, slots 0-6 are its backwarp synthesis.
  - At k = 3 there are 4 keyframes: slots 2 and 7 are real.
  - Tokens = Cinque `view_39` on the 8 slot pairs, the serve function.
- Gate (passed): the k = 0 forward on the dumped frames reproduces the stored native reads' k = 0 plan for both RSG and JTN1. Max error
  0.061 / 0.055 m, median 0.006. RSG's plan on JTN1's frames matches RSG's own read, so the k = 0 frames are model-independent. The route
  memory equals the dump's in 983 / 983 windows.
- Probes: 5 folds by sha1(log). Linear = standardised logistic regression, C = 0.1. TOK = KIN + slot tokens -> LN -> 16 -> MLP 256, batch 64,
  30 epochs, mean of 3 seeds. CIs: log-cluster bootstrap, B = 2000, paired differences on the same draws.

## Primary (P vs N1 + N2 + N3, k = 0, 239 vs 580, 95 % log CI)

| set | AUC | same-log pairs only (3 114 pairs, 15 logs) |
|---|--:|--:|
| KIN | 0.704 [0.57, 0.83] | 0.806 [0.57, 0.95] |
| LEAD-RSG (KIN + lead head) | 0.938 [0.86, 0.98] | 0.944 [0.81, 1.00] |
| LEAD-JTN1 | 0.944 [0.88, 0.99] | 0.949 [0.83, 1.00] |
| ARC0-RSG (- plan 4 s arc at k = 0) | 0.206 [0.14, 0.30] | 0.375 |
| ARC0-JTN1 | 0.207 [0.14, 0.29] | 0.335 |
| TOK-lin (token mean) | 0.976 [0.94, 1.00] | 0.991 |
| **TOK** (slot 7) | **0.975 [0.94, 1.00]** | 0.983 [0.95, 1.00] |
| TOK - KIN | +0.271 [+0.15, +0.39] | |
| **TOK - LEAD-JTN1** | **+0.031 [+0.00, +0.08]** | +0.034 [-0.00, +0.14] |

Verdict rule: TOK >= 0.85 holds. TOK - LEAD >= +0.05 fails: +0.031, with its CI lower bound at 0.00. The prereg's third branch applies,
"only at the heads' level": **the lamp is not readable, so this is an input problem.**

## Per negative class

| contrast | TOK | LEAD (best) | KIN | ARC0-JTN1 | TOK - LEAD |
|---|--:|--:|--:|--:|--:|
| P vs N1 close lead | 0.987 | 0.980 | 0.708 | 0.083 | +0.007 [-0.01, +0.03] |
| P vs N2 no light, no lead (38 windows, 6 logs) | 0.783 [0.49, 0.95] | 0.422 | 0.164 | 0.399 | +0.361 [+0.06, +0.57] |
| P vs N3 moving to a green / unknown light | 0.996 | 0.988 | 0.992 | 1.000 | +0.009 [-0.01, +0.03] |
| P vs N1 + N2 (static only) | 0.972 | 0.939 | 0.668 | 0.106 | +0.033 [+0.00, +0.09] |

How to read it:
- N1 is a lead-detection task. The lead head does it (0.98), and the tokens add nothing.
- N3 is a speed task. KIN does it (0.99).
- N2 is the only lamp-like contrast: both classes stand, neither has a lead. There the tokens rank above the heads (0.78 against 0.42; TOK-lin
  0.83). But the evidence is 38 windows from 6 logs, and the same-log pairs come from only 4 logs (TOK 0.77 [0.10, 1.00]). At the operating
  threshold N2 is not separated at all (next table). Whatever ranks P above N2 is weak. It is as likely the junction or stop-line geometry
  as the lamp.

## Operating point: share called "must stand" at each fold's 10 % FPR threshold (training negatives)

| class | KIN | LEAD-JTN1 | TOK |
|---|--:|--:|--:|
| P red line, no lead | 0.31 | 0.97 | 1.00 |
| N1 close lead | 0.12 | 0.06 | 0.28 |
| N2 no light, no lead | 0.66 | 1.00 | 1.00 |
| N3 moving to a light | 0.00 | 0.06 | 0.08 |
| **G green launch, no lead** | 0.35 | 0.95 | **1.00** |
| U unknown-state light, static | 0.49 | 0.93 | 1.00 |
| F far lead, static | 0.14 | 0.43 | 0.59 |

P vs G: TOK 0.531 [0.44, 0.64], LEAD-JTN1 0.551, ARC0-JTN1 0.639. The red-line windows and the green launches look the same to every
feature set. This is consistent with d340: the rendered lamp colour never changes. It also means no input-side lamp reader can recover the
-13 points of the hold trade (d372).

## Slots and distance

- Per slot (k = 0, primary contrast): TOK 0.963-0.976 for every slot. The real keyframe slot 7 gives 0.975; the synthesised slots 0-6 give
  0.963-0.976. At k = 3 (JTN1's own states, real slots 2 and 7), TOK gives 0.965-0.981 across slots, against LEAD-JTN1@k3 0.958 and KIN@k3
  0.795. Real and synthesised slots do not differ: the separating content is static scene layout, which the warp preserves.
- Distance from the front bumper to the line (P bins against all N):

  | bin | windows / logs | TOK | LEAD | TOK - LEAD |
  |---|---|--:|--:|--:|
  | < -1.5 m | 78 / 6 | 0.951 | 0.926 | +0.025 |
  | -1.5 to -1 m | 41 / 2 | 1.000 | 0.956 | +0.044 |
  | -1 to -0.5 m | 88 / 5 | 0.983 | 0.953 | +0.030 |
  | >= -0.5 m | 32 / 6 | 0.979 | 0.946 | +0.033 |

  No bin reaches +0.05 with a CI above 0.
- N3 by light distance (0-20 / 20-40 / 40-60 m): every set is at 0.96-1.00. Speed alone separates them.

## What it decides

- **Registered: not readable (only at the heads' level), so an input problem for the lamp.** The tokens hold "standing, no vehicle ahead".
  They hold nothing about the lamp beyond the lead head and kinematics. They do not separate red-line stands from green launches (0.53).
  - The lamp lead in the 6-log N2 contrast (+0.36 [+0.06, +0.57]) is weak.
  - At the operating point it vanishes: N2, U and G are all called "must stand".
- **For the +72-point hold (d372) this is not a blocker.** The facts a "stay where you stand when no lead is ahead" rule needs are already
  readable in the policy's own readouts: standing (KIN) and no lead (lead head 0.94-0.98). The model stands on N2 already (1 zero / 38). It
  rolls on P, and its decision-0 plan is *longer* there (ARC0 AUC 0.21). So holding at the red line is a training-signal problem on readable
  features, and it will pay the G windows (-13), because the lamp cannot tell them apart. Reading the lamp itself (green launch) is an input
  problem: the render's lamp is constant, and the tokens show no lamp signal beyond the heads.
- **Evidence: medium for "beyond the heads: no", weak for N2.** There are 18 positive logs, and same-log windows are strongly correlated.
  Each model was read once; the k = 0 frames are deterministic. The light labels come from the log (census), not the render.
