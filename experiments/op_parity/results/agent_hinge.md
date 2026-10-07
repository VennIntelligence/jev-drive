# Agent hinge (pilot): the plan-footprint vs logged-agent-box hinge cuts NC + TTC failures by 0.16 pp, half the gate's 0.3 pp; the lane stops

Written 2026-10-07. Pre-registration [../plans/2026-10-07-agent-hinge-prereg.md](../plans/2026-10-07-agent-hinge-prereg.md); its section "执行补记与声明的偏离"
was written before any HA arm was trained or scored. Code: `lib/agent_hinge.py` (labels, loss, pre-training check), `pp_train.py --agent-lam`,
chain `scripts/pp_agent_chain.sh`, report `scripts/pp_agent_report.py`. Tables: [agent_hinge/](agent_hinge/) (`pilot_navtest_{arms,paired,strata}`),
the check and gate JSONs in [agent_hinge/checks/](agent_hinge/checks/). Context: [four_dirs.md](four_dirs.md) direction 3, decisions 148, 153.

## Result

Pilot read, seed 0, pilot scale (navtrain s0of12 + s1of12, W frames, frozen vision, P2 + drivable hinge lambda 10, 3 000 steps x 64, same row stream as the
reference): **HA-F-s0** = that recipe + agent hinge lambda_a 10, margin 0.5 m, against **HP-F-s0** (the turn-train H). navtest, 12 146 tokens, devkit v2;
paired cluster bootstrap over the 136 logs, B 10 000, 95% CI.

| quantity | HP-F-s0 | HA-F-s0 | HA - HP | gate threshold | met |
|---|---|---|---|---|---|
| NC + TTC failures (% tokens) | 2.44 | 2.27 | **-0.16 [-0.26, -0.08]** | <= -0.3 | **no** |
| EPDMS | 87.71 | 87.90 | **+0.18 [+0.05, +0.31]** | >= +0.2 | no |
| EP | 86.95 | 86.85 | -0.10 [-0.13, -0.07] | >= -0.2 | yes |
| NC failures (%) | 1.65 | 1.49 | -0.16 [-0.25, -0.08] | | |
| TTC failures (%) | 2.31 | 2.16 | -0.15 [-0.24, -0.07] | | |
| DAC failures (%) | 4.37 | 4.31 | -0.06 [-0.17, +0.05] | | |

WA-JEPA: EPDMS 91.71, NC + TTC failures 1.17%. HA closes 0.16 of HP's 1.27 pp NC + TTC gap to WA-JEPA (13%) and 0.18 of its 4.00 EPDMS gap.

- **Gate verdict: STOP.** The NC + TTC drop is real (CI excludes 0) but below the registered 0.3 pp; the lambda_a 3 branch only opens when NC + TTC moved
  and EP fell, which did not happen (EP -0.10 is within the guard). By the pre-registration the full stage (P2H10 recipe + agent hinge, 2 seeds,
  navtest / navhard / HUGSIM 64 `spec_plan_smooth`) is **not run**; there are no full-stage, navhard or HUGSIM numbers for this arm.
- **Where it acts** (strata on the P2H10 D3 tokens of four_dirs, fixed before scoring; [pilot_navtest_strata.md](agent_hinge/pilot_navtest_strata.md)):

| stratum (P2H10 D3 tokens) | n | NC + TTC fail % HP | HA - HP (pp) | EPDMS HA - HP | WA fail % |
|---|---|---|---|---|---|
| any D3 | 285 | 80.7 | -1.4 [-3.4, +0.4] | +1.49 [-0.06, +2.86] | 28.8 |
| stopped vehicle ahead | 104 | 82.7 | -1.9 [-5.2, 0.0] | **+2.26 [+0.30, +4.50]** | 20.2 |
| plan > 1.1 x logged speed | 138 | 84.1 | **-2.9 [-5.4, -0.7]** | **+2.25 [+0.50, +3.67]** | 36.2 |
| lead vehicle | 40 | 80.0 | -2.5 [-7.1, 0.0] | +2.81 [0.00, +7.38] | 25.0 |
| static object | 50 | 72.0 | +2.0 [0.0, +5.6] | -1.08 [-4.50, +4.40] | 16.0 |
| cut-in / oncoming / sideswipe | 57 | | 0.0 | 0.00 | |

  Turn bins (all tokens): the drop is on straight and gentle tokens (< 5 deg -0.20 pp [-0.35, -0.08], 5-20 deg -0.19), none on 20-45 deg (0.00) and
  -0.13 on > 45 deg. The registered targets (stopped vehicle ahead, over-speed) move in the right direction and carry the EPDMS gain, but on 104 / 138
  tokens the change is 2-3 of ~83 failing tokens per 100; static objects (cones / barriers are outside the registered classes) get slightly worse.
- **Why it is small** (post-hoc, `checks/posthoc-HA.json`, same geometry as the loss): HA's navtest plans still enter a labelled agent box (hinge > 0)
  on 1.99% of tokens vs 2.21% for HP, and 73% of HA's remaining NC failures are still footprint overlaps with an agent the loss covers (HP 75%). The
  training-row agent loss stays ~0.002 (0.0022 at step 100, 0.0012-0.0020 at the end). So the constraint is not missing from the labels; the plan head
  on frozen comma features does not turn it into held-out behaviour at this scale. That matches decision 147 / four_dirs (collision failures partly
  decided at the representation; HUGSIM lead distance read +1.95 m at < 3 m), but this lane did not test that split.

## Pre-training geometry check (deviation 3)

`lib/agent_hinge.py check`: human log trajectory of 300 random navtest tokens (and all 12 146) through the hinge; share of P2H NC-failure tokens with hinge > 0.
Pass = human < 1% and NC coverage >= 70% (P2H10-F-s0; HP-F-s0 also reported).

| variant | human 300 | human all | P2H10 NC cov. | HP NC cov. | P2H10 NC+TTC cov. | pass |
|---|---|---|---|---|---|---|
| registered literal: K 16, side margin 0.5 | 1.33% | 0.92% | 0.739 | 0.776 | 0.558 | no (human) |
| K 16, side margin 0 | 0.33% | 0.07% | 0.670 | 0.721 | 0.493 | no (coverage) |
| K 32, side margin 0.5 | 1.33% | 0.96% | 0.840 | 0.866 | 0.657 | no (human) |
| **K 32, side margin 0 (used)** | **0.33%** | **0.11%** | **0.766** | **0.816** | 0.580 | **yes** |
| K 32 + cones / barriers / signs, side 0 (not used: unregistered classes) | 0.33% | 0.09% | 0.787 | 0.826 | 0.595 | yes |

Every human violation of the literal geometry is a lateral pass of an adjacent-lane vehicle 1.5-7 m ahead of the ego centre at 2.5-2.9 m lateral offset
(box gap 0.1-0.5 m); none is a front gap. The used variant keeps the registered margin for objects in the ego's lateral corridor (front edge extended
0.5 m + relu(0.5 - d): 1.0 m front clearance, as written) and penalises objects outside it only on contact. K 16 was full on 92% of navtest tokens
(pedestrians), pushing farther leads out of the label; K 32 keeps the same nearest-at-t0 rule.

## Deviations (all declared in the pre-registration before training)

1. Hinge reduction: per (row, 0.1 s step) the max over counted objects of relu(m - d), mean over imitation rows x 41 steps.
2. Front half plane and the 3 m lateral window are taken about the centre of the unextended footprint at the interpolated pose.
3. Geometry fix after the failed check: K 32, margin 0 outside the lateral corridor (table above). lambda_a, margin and gate thresholds unchanged.
4. NC + TTC failure = NC < 1 or TTC < 1 (four_dirs D3); the gate's EP is the devkit ego_progress x 100.
5. Gate gap case (NC + TTC moved, EP fine, EPDMS < +0.2) reads as stop; not reached (NC + TTC did not move enough).
6. D3 strata = P2H10's four_dirs D3 token sets; over-speed = their spd > 1.1.
7-8. Full-stage naming, HUGSIM reference runs and criterion readings were fixed but are moot (full stage not run).

## Caveats

- One seed at pilot scale (17 k rows, 3 000 steps); the four_dirs size estimate was for the full recipe, and a full-scale effect can differ. The gate
  is a point-estimate rule; the observed drop is about half the threshold with a CI ending at -0.26, so even optimistic noise would not reach -0.3.
- The labels are logged and non-reactive (the scorer's view); objects not present at t0 or outside the 32 nearest are not penalised; cones, barriers
  and construction signs are not in the registered classes (they are part of NC's static-object collisions).
- The prereg's SDF approximation saturates for deep overlaps (penetration depth is capped at the lateral overlap, so a deep head-on overlap pushes
  sideways rather than backwards); shallow approaches (d in [-0.15, 0.5] m) give the longitudinal gradient.
- Cost: labels ~10 s CPU per split, pilot training 4 min (13 it/s), navtest plans + scoring ~10 min, all through the pool; well inside the 20 min budget.
