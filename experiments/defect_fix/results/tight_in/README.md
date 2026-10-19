# TIGHT-IN offline gate (NR1 prereg supplement 22): FAIL, lane stopped (2026-10-20)

Pre-registration: [plans/2026-10-20-nr1-supp22-tightin.md](../../plans/2026-10-20-nr1-supp22-tightin.md) (commit 704e9a79). It was registered before
any label, gate output or training. Code: `lib/nr1_tight.py`, `lib/nr1_lat.py` (`tightin`), `rm_train --on-tightin W`,
[scripts/tight_in.py](../../scripts/tight_in.py), [scripts/tight_in.sh](../../scripts/tight_in.sh). See also decision 401.

**The term.** It acts on moving S rows in tight turns:
- the ego at 2 m/s or more;
- the human's logged 4 s arc at least 4 m;
- the logged path turning at least 15 deg over the next 20 m, with a max 1 m curvature of at least 0.108 /m.

It has two parts:
- **Lateral:** smooth L1 on plan poses that sit more than 0.2 m outside the more outward of two lines: the human's line, or the ego's present offset. Inside is free.
- **Arc:** TURN-CAP's 1.05 x human-arc cap.

The switch is off by default.

**Code checks.** `nr1_tight.py` passes its known-answer checks on the Mac and on the box. The bit-identical switch-off check never ran: it was queued behind the gate and failed on the dependency.

## Gate (box CPU, stored reads; [gate.json](gate.json))

| check | measured | line | result |
|:--|--:|:--|:--|
| G1: human paths as the plan, on1_JTX domain rows (3 050) | lateral 345 / 344 fires (`fut` / `fut_reg`), arc 0 / 0 | 0 | **FAIL** |
| G2: B2 s1 sharp-set lateral zeros, any fire 3 s .. 1 s before the exit | 15 / 21 = 71 %; targets 6 / 8 | >= 50 % and >= 5 / 8 | pass |
| G3: B2 s1 held windows scored 1, decisions that fire | 313 / 16 272 = **1.92 %** (lateral 263, arc 265) | <= 1 % | **FAIL** |
| G3: same, windows with any fire | 84 / 2 034 = 4.1 % | <= 5 % | pass |

s0 gives the same picture:
- G3: 1.83 % of decisions and 3.9 % of windows fire;
- G2: 17 / 22 failing windows hit, and 6 / 7 targets.

## Why the gate fails, and why the draft's 0.72-0.97 % estimate was too low

1. **The domain is about 2.5x larger than the draft measured.**
   - On the full log, 2.5 % of B2 s1's passing decisions are in the domain. The draft measured about 1.0 %.
   - The draft took curvature from the 5.5 s human track stored with each window. That track is clamped late in the window, so "the next 20 m" is often cut short. The box computes the full 70 m path at every decision, as the trainer does. The draft's own caveat named this risk.
   - With that track the domain alone was 1.03 % on s1, so any term on that domain could only have passed the 1 % line by firing on a fraction of it.
2. **The term fires on most of the domain: 76 % of the passing domain decisions.**
   - The draft's lateral estimate used one point, the outward offset at 4 m of arc (0.37 %). The registered term checks all 8 poses out to +25 m of path.
   - B2's plans run wide almost everywhere in tight turns. On passing domain decisions the median max-outward is 0.75 m and the 90th percentile 8.5 m.
   - So fire rate ≈ domain share. That is the same base-rate story as decision 397: tightness picks the place, and B2's state does not separate failing turns from passing ones.
   - The fires are not only in turn windows. By stratum: turn 97, other 88, braking 87, launch 38.
3. **G1: the term is wrong on its own human labels (a design error in the finalised rule, change 1 of the registration).**
   - The reference is the more outward of the human line and the ego's present offset.
   - On 11 % of domain rows the human's own reachable path (`fut` / `fut_reg`) still runs more than 0.2 m further out. Median max-outward on those rows is 1.15 m.
   - These are rows where the ego is already outside the line: median outward offset 0.68 m, against 0.0 m on the rows that do not fire. They are also rows where the ego heading points out.
   - There, the imitation path that blends the ego's heading back onto the line swings outward first. 175 of the 344 rows peak at the 1 s pose.
   - So even the human's way back is penalised. A reference built from the present offset alone ignores the heading. The draft's form `-sigma (d_j - d_0)` has the same fault.

The pool description is in [label.json](label.json):
- 3 050 domain rows (1 905 right, 1 145 left, 24 logs), 2.4 % of S and 1.0 % of the JTX00 draw;
- 452 more tight moving-turn rows are dropped by the human-arc limit;
- the stored served plans fire on 79 % of the domain rows (lateral 2 001, arc 1 413).

## Jobs and cost

| item | count | card-hours |
|:--|:--|--:|
| CPU jobs that ran | `ti-label` (done) and `ti-gate` (exit 2, GATE FAIL) | 0 |
| jobs failed by dependency, never started | identity pair, bitcmp, dose, 6 control read jobs, both arms' train / reads, psum | 0 |
| **GPU total** | | **0** |

Box outputs are in `runs/defect_fix/tight_in/` (gate/decisions.parquet, lists/, stages.txt), and the label file is `runs/op_parity/cache/on1_JTX/tight_in.npz`.

## Files
- [gate.json](gate.json), [label.json](label.json);
- [tight_set.csv](tight_set.csv): the 79 held windows whose road has max 1 m curvature >= 0.108 /m;
- [gate_windows.csv](gate_windows.csv): the sharp-set windows of B2 s1 / s0, with exit times and labels;
- [cmd_diff.txt](cmd_diff.txt): the dry arm commands.
