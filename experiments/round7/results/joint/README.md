# R7J: round7 joint arm (keep-pace labels + F2, S-MIX, route-heading command), 2026-10-21 / 22

Pre-registration and amendments 1-4: [../../plans/2026-10-22-joint-arm.md](../../plans/2026-10-22-joint-arm.md) (Chinese). Lane code
[../../scripts/joint.sh](../../scripts/joint.sh), [../../scripts/joint.py](../../scripts/joint.py); trainer switch `rm_train.py --cmd-rule heading`,
serve switch `pai_driver.py JEV_CMD=heading`, rule `experiments/route_mem/lib/rmem.py cmd_heading`. Decision 441.

**Verdict: not adopted (fails on both seeds).** R7J s0 / s1 read 0.787 / 0.786 on PAI 281 (ACC + GAP + JEV_CMD, Harmonizer OFF, 274 common scenes)
against R6DAG 0.779 / 0.816 and NTV0 0.750; with the Harmonizer ON 0.774 / 0.765 against R6DAG-s1 ON 0.780 and R6DAG-s0 ON 0.796.

## Arm

R6DAG's argv (from NTV0-L18-s0, 9 500 steps) with three changes:

1. Longitudinal hygiene. **F2** drops 9 612 PAI human rows, the same count as decision 433. The 90 300 drawable DAgger iteration-1 states are
   relabelled with **teacher b5**: b3's lateral path, with arc length = max(student, min(human from the matched station, v0 t + t^2)). This
   applies only where all of these hold: the human keeps >= 1 m/s for the next 6 s, the clocks agree within 3 s, and no object leads on the
   label path. Labels over the T4 line revert to b3. Of the drawable rows 14 677 are floored (16 %, +5.7 m median at 4 s; 8 660 of the 59 827
   straight rows), and 7 177 more go to weight 0 under F2 on the new labels. Files: [rows.md](rows.md).
2. **S-MIX**. The S draws are re-weighted by R6DAG-s1's own per-row terms (E0): x0 unlearnable, x4 active, x2 zero / partial source. The
   learnable-active share of S draws rises from 11.9 % to 30.5 %. The batch becomes navtrain 25 / yaw-rate 13 / pool 98 (S 58), with
   **ho_w 0.86**: the registered 1.3 gave 1.52x the control's S / imitation gradient ratio and was rescaled once, as registered
   ([smoke_R7JSM-L18-s0.json](smoke_R7JSM-L18-s0.json)).
3. **Command from the route's heading change** (40 m, 20 deg, persistent) on the PAI human, DAgger and S rows, and at serve. Non-turn PAI rows
   with a turn command drop from 25-30 % to 5-6 % ([cmd_rule.md](cmd_rule.md)).

Gates: G1 labels, G2 command, G3 S-MIX and G5 smoke all pass. G4 identity is bit-identical ([identity.json](identity.json)). The serve gate
passes: with the switch off, 7 082 inferences have identical served poses and identical scores; with it on, the logged command matches the
rule 100 % ([serve_gate.json](serve_gate.json)).

## Results (full table [pai281_R7J.md](pai281_R7J.md), ruler [pcs_joint.md](pcs_joint.md))

| | R7J s0 | R7J s1 | R6DAG s0 | R6DAG s1 | NTV0 |
|:--|--:|--:|--:|--:|--:|
| OFF mean | 0.787 | 0.786 | 0.779 | 0.816 | 0.750 |
| - same-seed R6DAG | +0.008 [-0.017, +0.033] | **-0.030 [-0.062, +0.001]** | | | |
| - NTV0 | +0.037 [-0.007, +0.081] | +0.036 [-0.007, +0.080] | | | |
| offroad + zero-other | 19 | 26 | 24 | 16 | 17 |
| rear-hit / slow partials | 17 / 6 | 17 / 3 | 22 / 6 | 20 / 6 | 25 / 11 |
| launched (LAUNCH-37) | 33 | 35 | 35 | 35 | 33 |
| ruler - NTV0 | -153 [-265, -35] | -145 [-246, -6] | -213 | -155 | |
| ON mean (Harmonizer) | 0.774 | 0.765 | 0.796 | 0.780 | 0.748 |
| ON - R6DAG-s1 ON | -0.006 [-0.033, +0.021] | -0.015 [-0.041, +0.009] | | | |
| ON - same-seed R6DAG ON | -0.022 [-0.050, +0.005] | -0.015 [-0.041, +0.009] | | | |
| ON launched | 32 | 32 | 31 | 32 | 33 |

Rule failures: s0 fails the NTV0 CI, launches (33 < 34) and Harmonizer ON. s1 fails the mean vs R6DAG-s1, the NTV0 CI, offroad + zero-other
(26 vs 16) and Harmonizer ON. The ruler rule passes on both seeds.

## Reading

- **The longitudinal change did what it was built for, on the partials.** Rear-hit partials fell to 17 / 17 (R6DAG 22 / 20, NTV0 25) and slow
  partials to 6 / 3. The ruler gap to NTV0 shrank to -153 / -145. On the six ruler-gap scenes: bbc53aa8 is solved by s1 and is 0.93 for s0
  (R6DAG 0.74); 2a0efada is 1.0 on both seeds; d90fe123 stays a rear-hit OFF (0.71 / 0.66) but scores 1.0 ON on both seeds.
- **It did not survive the Harmonizer as a gain.** ON - OFF is -0.012 / -0.021 (R6DAG-s1 -0.035). But R6DAG-s0 *gains* under ON (0.796,
  +0.017), so the same-seed ON comparison is -0.022 / -0.015, and launches ON stay at 32.
- **The lateral side got worse, not better.** s1 has offroad +9 [+3, +16] against R6DAG-s1. a1b166cc and 94c1d722 are offroad on every R7J read.
  The two stuck curve cut-ins are unchanged: 596af0c7 offroad OFF and a rear-hit partial ON, ea468790 offroad everywhere. The joint arm cannot
  say which change costs the lateral: S-MIX moved S pressure and ho_w also scales the lateral and DAgger agent terms, and the command rule
  changed 17 % of served commands.
- **The seeds converged.** s1 - s0 is -0.000 [-0.025, +0.025], against +0.040 for R6DAG. The mean over seeds (0.787) is below R6DAG's
  (0.797); R6DAG-s1 remains the best single checkpoint OFF, and R6DAG-s0 is the best ON read we have (0.796).
- New zeros / partials on the 182 scenes NTV0 aces: 13 / 5 and 13 / 5 (R6DAG 12 / 11 and 5 / 7); offroad is the largest class.

## Cost and disk

Trainings about 45 min each; four 281 reads (two ON); the gates totalled under 1 card-hour. The pools `on1_JTPHDK` / `on1_JTPHDI` add 2 x 0.98 GB
(tabs only; the rest are hard links). Cleanup log: [../../../cl_infra/cleanup/20261021d_joint.md](../../../cl_infra/cleanup/20261021d_joint.md).
