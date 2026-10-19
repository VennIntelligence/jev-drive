# NR1FREE-M: the one-sided longitudinal term on moving states only (2026-10-19; NR1 prereg supplement 16)

Question: can the speed ratchet of decisions 372 / 375 be fixed on the training side? In closed loop each plan asks 2-4 % less than the
current speed, and the frames lock that in. The fix tried here keeps decision 365's term (penalise the plan only when it is short of the
human and the way ahead is free, no target speed) but moves it off the low-speed wait / launch axis that broke waiting there:

- Rows: S rows where the ego moves (t0 speed >= 3 m/s) and the human neither stops nor brakes (every 0.5 s segment of his logged 4 s
  >= max(3 m/s, 0.8 x his first segment)).
- Band: 0.95 x the reference (decision 365: 0.8). The reference is min(the human's logged arc, the same speed profile reachable from the
  visited state).
- Dose: W 0.004, as in decision 365. The free check (logged boxes, 2 m + 0.5 s x v ahead) is unchanged.

Code: [lib/nr1_free.py](../../lib/nr1_free.py) (`configure`, `moving`), `rm_train --on-free W --on-free-mode move --on-free-frac 0.95`
(defaults reproduce supplement 15), [scripts/free_m.py](../../scripts/free_m.py) (gate, extract, arena, pilot rule),
[scripts/free_m.sh](../../scripts/free_m.sh) (chain). Box outputs: `$DATA_DIR/runs/defect_fix/free_m/`.

## Offline gate: passes ([gate.json](gate.json))

| check | set | result | line |
|:--|:--|--:|:--|
| false fires on the human's own paths | on1_JT S rows on moving states (44 881 of 87 012), plan = `fut` / `fut_reg` | 0 / 0 | <= 1 % |
| fires on ratchet decisions | JTN-L18-s1 native arena held, non-static windows, k >= 2, ego >= 3 m/s, plan 4 s arc < 0.95 x ref | **2 317 / 3 559 (65.1 %)**, 495 windows | >= 50 % |
| fires on AF-front windows | JTN-L18-s1 / s0 native + the four JT reads, 3 s before the first front collision | **0 / 264 windows** | <= 5 % |
| fires on static windows | JTN-L18-s1 native, 878 static windows, every decision | **0 / 8 780** | <= 1 % |
| fires on the decision 365 failure set | launch_diag dumps (JTN-L18-s0, RTJA0), wait set v2 b, lead classes, every decision | **2 / 1 660 (0.12 %)** | <= 1 % |
| switch off = bit-identical | JTN's command, 60 steps eager, 524f6a32 against the new code | identical (275 scalars, weights, dev ADE, draw hash) ([identity.json](identity.json)) | identical |

How to read it:
- Ratchet population: 7 275 moving decisions. Their median plan / reference ratio is 0.953, and 49 % are below the 0.95 band. Of those
  short decisions, 1 114 are blocked because the human brakes or nears a stop, and 128 because the way is not free. The term fires on 32 %
  of all moving decisions.
- The human check passes by construction (the reference is the smaller of the two human readings). With one reference alone the other
  human path fires often: `fut` against the raw log fires 24 %, because a slower ego cannot reach the human's arc within 2 s; `fut_reg`
  against `fut` fires 3.6 %. Without the minimum, the term would ask for more than is reachable.
- The AF-front windows are trivial again: of 186 moving decisions in those windows, none is short of 0.95 x the reference.

## Pilot: fails the registered rule ([pilot.md](pilot.md), [pilot.json](pilot.json), [arena_pilot.json](arena_pilot.json), [pai_half.md](pai_half.md))

FREEM1K-L18-s0 against JTNS1K-L18-s0, the same recipe on disk (JTN command, 1 000 steps, warm-up 50, RB0-F-s0, seed 0). The command diff is
only `--tag` and `--on-free 0.004 --on-free-mode move --on-free-frac 0.95` (checked against JTNS1K's logged argv). Every read is native: no
hold, no stop3.

| | JTNS1K (control) | FREEM1K | line |
|:--|--:|--:|:--|
| held 460 zeros | 42 | 56 | <= control + 2: FAIL |
| a wait other_lead / red_queue | 0.944 / 0.923 | 0.889 / **0.795** | >= control - 0.05: FAIL (both) |
| held 460 front collisions | 7 | 13 | <= control + 2: FAIL |
| arena subset AF-front windows (753 windows) | 2 | 8 | <= control + 2: FAIL |
| arena subset k = 9 speed ratio (median v / v_h) | 0.841 | **1.025** | +0.184 [+0.146, +0.245] >= +0.02: ok |
| PAI half (144 scenes) mean / zeros | 0.6733 / 40 | 0.6677 / 43 | -0.0056 [-0.039, +0.026] >= 0 and zeros <= +2: FAIL |

Description (outside the gate):
- **The ratchet is gone, and the speed overshoots.** Same-station speed ratio by k, control against pilot: k2 0.985 / 0.998, k5 0.912 / 1.018,
  k9 0.841 / 1.025. The pilot ends 2.5 % faster than the human. On the arena subset the "slow" partial scores drop from 50 to 16.
- **Static windows pay for it.** Strata on the arena subset (control -> pilot): static (331 windows) 0.761 -> 0.674, zeros 79 -> 108;
  braking 0.953 -> 0.961; cruise 0.974 -> 1.000; launch 0.883 -> 0.910; other 0.946 -> 0.957; turn 0.950 -> 0.942. Overall mean
  0.8618 -> 0.8290, paired -0.033 [-0.067, -0.001] (log-clustered), zeros 96 -> 125.
- Held 460 mean +0.030 [+0.006, +0.049] (most held windows are moving), while held zeros go 42 -> 56. Wait set b: launched 0.73 / 0.66 ->
  0.77 / 0.74, median delay +0.57 / +0.77 -> +0.32 / +0.52 s. The term never fires below 3 m/s, yet the launch moved too.
- PAI under ACC + GAP: there the speed loop is set mostly by the serve-side JEV_LEAD cut, which ACC + GAP removes. Only a ~0.45 m/s
  per-decision plan shortfall is left for the term to reach (decision 376, [pai_input_loop.md](../../../route_follow/results/pai_input_loop.md)).
  A small PAI effect was expected. It came out slightly negative, with the CI across 0.

Reading: by the rule written before the reads, **fail: the line stops; no full-length arm** (the summary job did not submit it). The term
does what it was built for, and more: it removes the plan's speed bias on moving states, overshooting to 1.03 of the human by k = 9. But
it does not stay on the moving axis, even though it never fires on a row where the ego is below 3 m/s or the human brakes. The
static-window creep, waiting at a lead and front collisions all get worse. That is the same coupling as decisions 352 / 353 / 363 / 365.
"Do not plan short" learnt on moving free-road rows generalises to "go" in standing states, which the model cannot tell apart from its
inputs. The pull is strong: at W 0.004 and band 0.95 the moving ratio overshoots, so the dose was too high for the ratchet alone. A lower
dose or a 0.9 band would shrink both the fix and the leak (untested). Nothing in this pilot shows that the two separate.

Limits: one seed, 1 000 steps, one dose and one band. The arena read covers 6 of 18 chunks. The control's wait set and held 460 reads are
the stored ones from 10-18 (serve unchanged with its new switches off). The pool fire share during training was not extracted.
