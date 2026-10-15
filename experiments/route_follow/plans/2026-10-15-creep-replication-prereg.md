# Pre-registration: PLAN CREEP replication on held 460 checkpoints decision 308 did not use (no training, no rollouts, CPU only), 2026-10-15

**Why.** Decision 308 found plan creep post hoc (Amendment 1 of `plans/2026-10-15-plan-switching-prereg.md`): in failed PAI turns each
new plan's 2 s point lies outside the previous plan's path, -0.063 vs -0.015 m per metre of travel on PAI RSG-L18-s0 (diff -0.048
[-0.104, -0.016]), late-wide -0.133 m / m; the car turns 2.5 deg less than each plan asks over 2 s (failed - completed -4.1 deg
[-6.6, -2.3]). On held the six d308 sets pointed the same way only in late-wide (6 turns, -0.104) and the pooled CIs reached 0. This lane
tests the finding on data d308 did not see, with the measure frozen. Written and committed before any statistic below is computed on the
new data. Amendments are appended, never edited in place.

## Data (stored reads, read-only)

Held 460 at 18 decisions (decisions every 0.5 s), the 24 strict held turns of every checkpoint below (rf_follow `scenes.csv` rows
`turn == True` of label `<tag>@18`; the same 24 scenes for every tag). d308 used PAI RSG / RV2a / RV2b / APP1X and held RSG-L18-s0 / s1,
RV2a-L18-s0 / s1, RV2b-L18-s0 / s1; none of these enter the claims below.

| family | checkpoints | follow rows | rf_tax extract |
|:--|:--|:--|:--|
| lr | RSGlr{03,10,30}n{1250,2500,5000}-L18-s0 except lr10n2500 (= RSG-L18-s0, used by d308), RSGlr03n5000-L18-s1 (9) | `runs/route_follow/lrsteps/follow` | new (this lane) |
| RV2c | RV2c-L18-s0, RV2c-L18-s1 | `runs/route_follow/v2/report/follow` | `runs/route_follow/corridor/tax` |
| APP0C | APP0C-L18-s0 | `runs/route_follow/approach/w0/report/follow` | new |
| APP2W05 | APP2W05-L18-s0 | `runs/route_follow/approach/v2/report/follow` | new |
| FMH (flow head) | FMH-F-s0, FMH-F-s1 | `runs/flowhead/closed_loop/report/follow` | RECEDING extract exists: `runs/flowhead/closed_loop/held.pkl` |
| RGH (its regression twin) | RGH-F-s0, RGH-F-s1 | same | same |

17 checkpoints x 24 = 408 turns expected. The extracts are built on the box by `scripts/rf_creep_rep.sh` (rf_tax extract of the read
dirs `runs/route_mem/read/<tag>/c*/run`, rf_tax report for the class, rf_receding.py extract-held), copied to the Mac, read by
`scripts/rf_creep_rep.py`. Turns, entry / exit, completion and failure classes exactly as RECEDING `read_turn` with the rf_tax report's
primary label of each set (no d299 over-turn zeros list exists for these sets; over-turns come from the tax label only).

## Measure (frozen: rf_switching.py `turn_switching` / `creep`, unchanged code)

Window: served decisions with the ego station in [entry - 20, entry + 20] m and t0 < t_f (the first zero; no zero: the whole read),
consecutive pairs 0.5 s apart (+-20 ms). Sign: positive = outward (away from the turn), i.e. the negative of d308's signed j.
- **creep per metre** c = - median over the turn's pairs with ego travel ds > 0.05 m of j / ds (j = perpendicular offset of the new
  plan's 2 s point from the previous plan's path carried forward by the measured ego motion); >= 3 pairs.
- **summed creep** C = - sum of j over the window's pairs (m).
- **2 s heading shortfall** x2 = per-turn median over window decisions (t0 + 2 s < t_f) of (the car's yaw change over [t0, t0 + 2 s] -
  the plan's own heading at 2 s), toward the turn, deg; negative = the car turns less than the plan.
- **human log as a plan** (sanity): the same c on RECEDING's `Log.plan_at_station` (j_h / ds), per-turn median.
- Groups: failed = every non-completed turn (as d308), completed = rf_follow `completed`.

## Statistics

Pool all 408 turns. Bootstrap B = 10 000, seed 0, percentile 95 %, clusters = the nuPlan log (scene id without its last field; the
RECEDING `cluster`); every turn of the same scene, across all checkpoints, falls in one cluster (a log may hold several of the 24 scenes,
which is at least as conservative as scene clusters). A replicate resamples clusters and recomputes both group medians (rf_switching
`diff_boot`).

- **R1** (creep): d_c = median c (failed) - median c (completed) with its CI; ratio rho = median c (failed) / median c (completed).
  **Supported**: CI lower > 0 and median c (failed) > 0 and (median c (completed) <= 0 or rho >= 2). **Refuted**: d_c <= 0, or CI upper
  < 0.010 m / m. Else **inconclusive** (a CI above 0 with rho < 2 reads "difference holds, ratio < 2", not supported). Summed C reported
  with its CI, no verdict.
- **R2** (heading shortfall): d_x = median x2 (failed) - median x2 (completed) with its CI. **Supported**: CI upper < 0. **Refuted**:
  d_x >= 0, or CI lower > -0.5 deg. Else inconclusive.
- Both claims also need >= 10 turns with the quantity defined in each group (else inconclusive).
- **R3** (descriptive): c, C and x2 by class (late_wide, straight, over_turn, wrong_exit, other, no_zero; each vs the pooled completed
  turns, difference and CI), by rf_follow outcome (event, off, short = no zero, ran out of the 18 decisions short of the exit), and by
  family (failed vs completed of the same family where both have >= 3 turns; FMH completes ~1 / 48 turns, so its failed creep is set
  against RGH's failed and completed and against the pooled completed: does the flow head's catastrophe show as creep?).
- **R4** (descriptive; d308's open question, where the shortfall starts): per decision, the car's heading change and the plan's own
  heading change over 0-0.5 s, 0-1 s and the 1-2 s segment (car: yaw(t0 + 2) - yaw(t0 + 1); plan: psi(2 s) - psi(1 s)); shortfalls x05,
  x1 and x12 = car - plan; per-turn medians, failed / completed / per class, group medians with cluster-bootstrap CIs of the medians.
  0-0.5 s is the only interval in which the car is driven by plan k alone (the next plan arrives at t0 + 0.5 s); 0-1 s and 1-2 s include
  1 and 2-3 newer plans. Fixed reading on the failed turns: **near segment under-tracked** if the x1 CI lies below 0 and median x1 <=
  0.5 x median x2 (half or more of the 2 s shortfall is in the first second); **appears in 1-2 s** if the x1 CI covers 0 and the x12 CI
  lies below 0, and then the plan's near vs far heading (psi(1 s) vs psi(2 s) - psi(1 s)) says whether the plan's near segment is the
  straighter one; else **both**.
- Sensitivity (descriptive): R1 / R2 on failed turns with a zero only (outcomes event / off) vs completed; R1 / R2 leaving out each family;
  R1 / R2 with scene clusters; each checkpoint alone.

## Gates (before R1 / R2 are read)

- **G-code**: the read run on d308's held extract (`tmp/receding/held.pkl`, the six d308 held sets, same tax files) reproduces
  `results/switching/switching.md` held_all6: j per m failed / completed +0.003 / -0.006, exec - plan 2 s -1.6 / +0.4 (to the printed
  digits). Failing it, the code is fixed before anything else is read.
- **G-frame**: median |anchor - controller| <= 0.3 m and <= 1 deg per checkpoint (as d308).
- **G-human**: pooled median per-turn human creep |c_h| <= 0.010 m / m. Failing it, R1 is inconclusive.
- **G-n**: >= 380 of the 408 turns extracted with plans; class agreement is not checked (no prior classes exist for these sets).

## Outputs

`scripts/rf_creep_rep.sh` (box stages), `scripts/rf_creep_rep.py` (read on the Mac, imports rf_switching / rf_receding),
`results/creep_rep/` (turns.csv per turn, decisions.csv.gz, creep_rep.json, creep_rep.md with the CIs and verdicts),
`figs/creep_rep.png` (creep per metre by outcome and class; the R4 profile), one README paragraph. research/decisions.md is not edited.

## Amendments (appended after the prereg commit; each says whether it came before or after a statistic was seen)
