# NR1FREE: one-sided longitudinal consequence term (2026-10-18/19; NR1 prereg supplement 15; practice plan item 5)

Question: can a longitudinal term with no target speed fix the late launch without teaching "drive faster"? The term penalises the plan only
when its 4 s arc is below 0.8 x the human's and the way ahead is free. It does nothing when a logged object is in the way (the agent hinge
handles a lead). The two-sided arc term of decisions 352 / 353 failed at exactly this.

Code: [lib/nr1_free.py](../../lib/nr1_free.py) (term + self-test), [scripts/free_gate.py](../../scripts/free_gate.py) (offline gate, pilot
rule), [scripts/free.sh](../../scripts/free.sh) (chain), `rm_train --on-free W` (off by default; through nr1_lat as `lon_free`).
Box outputs: `$DATA_DIR/runs/defect_fix/free/` (gate/gate.json, gate/sim_rows.npz, pool/, jobs.txt, stages.txt).

## Term (fixed before any output)

- Reference per 0.5 s step: s_ref = min(the human's logged travel `fut_reg`, the same speed profile made reachable from the visited state
  `fut`). It is never above what the human did, nor above what the ego could reach.
- Fires when s_ref(4 s) >= 1 m, plan arc(4 s) < 0.8 s_ref(4 s), and the way is free. Value: the sum over the 8 steps of smooth L1 of
  relu(0.8 s_ref - s_plan). The poses move only along their own heading.
- Free: at every step the ego box, with its front edge pushed 2.0 m + 0.5 s x v_ref ahead, is placed at 8 positions between the plan's and
  the human's distance along the plan's path (extended straight past its end). No logged box (vehicle, generic object, pedestrian, bicycle)
  whose centre lies ahead may overlap any placement (separating-axis test).

## Offline gate: passes ([gate.json](gate.json))

| check | set | result | line |
|:--|:--|--:|:--|
| false fires on the human's own paths | on1_JT S rows (87 012), plan = `fut` / `fut_reg` | 0 / 0 (0.0 %) | <= 1 % |
| fires on late-launch decisions | launch_diag dumps, wait set v2 b lead classes, human launched and the ego not yet | JTN 55 / 86 (64 %), RTJA0 78 / 105 (74 %); pooled **69.6 %** | >= 50 % |
| fires on JT front-collision windows | JT10 / JT00 arena held reads (leash3, off), AF-front windows, 3 s before the first front collision | **0 / 251 windows** (0 / 1 624 decisions) | <= 10 % |

How to read it:
- The human check passes by construction: the reference is the smaller of the two human readings. With only one of them as reference, the
  other human path fires on 2.6-2.7 % of the rows (`fut_reg` against `fut`: the human brakes harder than a reachable blend from a faster ego;
  `fut` against `fut_reg`: the ego is slower than the human, so the reachable path is shorter). That is why the reference takes the minimum.
  Without it the term would push the plan past a braking human.
- On the late states the plan is far short of the reference: mean 4 s arc 8.7 / 9.8 m against 15.1 / 15.8 m. The free test blocks 12 of the
  67 slow JTN decisions and 6 of the 84 slow RTJA0 decisions (lead still too close).
- The front-collision line holds trivially. In the 3 s before a front collision the JT plans are never below 0.8 x the human (0 slow
  decisions), so the free test is never needed there. The term cannot make JT's collisions worse, but this set does not test the free
  check either. Only the pilot's closed loop tests that.
- Frame alignment of the closed-loop states: on 400 rollouts that follow the human, the controller trace mapped onto the log is a median
  0.20 m off the logged pose over the first 1.5 s (p90 0.94 m, where the ego leaves the human's path).

## Identity

With `--on-free` off the new code is bit-identical to 5a20dc80: JTN's command, 60 steps eager, 275 scalars, every weight, dev ADE and the
pool draw hash are the same ([../identity_free.json](../identity_free.json)).

## Pilot: fails the registered rule ([pilot.md](pilot.md), [pilot.json](pilot.json))

NR1FREE1K-L18-s0 against its control NTVN1K-L18-s0. Both arms run JTNS1K's command (1 000 steps, RB0-F-s0, warm-up 50) with supplement 14's
natural-share trainval draw JT00N. The pilot adds `--on-free 0.004`. Seed 0, one seed.

| | NTVN1K (control) | NR1FREE1K | line |
|:--|--:|--:|:--|
| b launched other_lead / red_queue | 0.792 / 0.743 | **0.938 / 0.943** | >= control - 0.03: ok |
| b median delay s | +0.34 / +0.32 | **-0.23 / -0.03** | <= control + 0.05: ok (effect: yes) |
| a wait other_lead / red_queue | 0.861 / 0.872 | **0.583 / 0.487** | >= control - 0.05: FAIL |
| held 460 zeros | 37 | **47** | <= control + 3: FAIL |
| held front collisions | 5 | **12** | <= control + 2: FAIL |
| held rear collisions (description) | 35 | 6 | |
| held mean / vs control | 0.8507 | 0.8885, +0.038 [+0.020, +0.062] | description |
| held vs RSG | -0.015 [-0.033, +0.009] | +0.023 [+0.005, +0.054] | description |
| wait-set b mean | 0.731 | 0.929 | description |
| dev ADE @1000 | - | 0.5516 (0.5276 at 0) | description |

Pool share of S rows that fire (held slice): 25 % at step 0, 18 % at step 1 000. The term is met only partly in 1 000 steps.

Reading:
- The term does what it was built for. The late launch is gone: lead-class delay +0.33 s -> -0.23 / -0.03 s, launches 74-79 % -> 94 %.
  Being rear-ended for being slow falls 35 -> 6. That buys the held mean gain: +0.038 over the control, +0.023 over RSG, CI above 0.
- It also costs the guarded lines. Standing and waiting collapse (a wait 0.86 / 0.87 -> 0.58 / 0.49) even though the term never fires
  while the human stands. Front collisions double (5 -> 12) and held zeros rise by 10. As in decisions 352 / 353 and LDNTV1K (decision 363),
  "launch on time" and "wait at standstill" move together on one low-speed axis. A term that only lengthens the plan in free-ahead
  states still generalises to "go" in states where the lead or the light is the reason to stay. The free check sees logged boxes only
  (no lights, non-reactive), and the model cannot tell those states apart from the inputs it uses.
- By the rule written before the reads: **fail, the line stops.** The term does not go into the joint arm at this dose. The held-mean gain
  is real but comes with +10 zeros and +7 front collisions. The guard was written to refuse exactly that trade.

Limits: one seed, 1 000 steps, one dose. A lower W was not tried. Given the coupling above, a lower dose would most likely trade launch
gain for wait loss along the same axis, not escape it (untested).
