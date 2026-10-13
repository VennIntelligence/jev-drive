# Pre-registration: STOPLINE, the red-light stop-line hinge on the RSG-L18 recipe, 2026-10-13

Why: the zero taxonomy ([research/zero-taxonomy.md](../../../research/zero-taxonomy.md), RSG-L18-s0 on PAI 281 at gains 20) puts 6.0 of the
36.8-point PAI gap on "the human stopped, the car did not" (17 corridor zeros `cor_route_end`: the log stops at a junction / line and ends at
0 m/s, the ego drives past the log's end), and 29 of the 62 PAI scenes where the human waited at a map stop line were reached by the ego
still moving (> 2 m/s while the human was still stopped). This arm gives the policy one signal for "stop here": the B hinge of
self-consequence L. Written, committed and pushed before any training except the 60-step smoke. Amendments are appended, never edited.

## History of B

B (lib `experiments/onpol_nuplan/lib/on_lon.py`, design.md 3.2, plans/2026-10-11-onpol-prereg.md "Self-consequence L"): on a pool row, every
traffic-light stop line of the route ahead (stop-sign lines masked) whose light is red at the log frame nearest each 0.1 s step charges the
plan's front bumper for crossing `max(d_line - 1.0 m, s_f(0))` while red and not yet cleared on green; gradient only along each pose's own
heading (speed profile, no lateral push). It was trained once, only together with A, C and P, in ONPOL-NUPLAN's ON-SELF (decision 259): L
as a whole cost -0.024 [-0.038, -0.011] and +7 zeros on held 460 at 10 decisions, with more progress; the items were never separated. No
B-only (or B + P) arm has been run. LONG-PROBE (decision 256) found B readable from the plan + vision (+0.056) and flagged four PAI
light / stop-line overruns 2.2-6.3 s ahead.

## Arm

| Tag | Recipe |
|:--|:--|
| `RSGB-L18-s{0,1}` | RSG-L18's exact command + `--on-lon on_lon` (pack items B, P; lambda 10 = `--on-lon-lam` default, x `--ho-w 3` as every pool term) |

Command (`$TR experiments/route_mem/scripts/rm_train.py train`): `--tag RSGB-L18-s{k} --init RB0-F-s0 --steps 2500 --warmup 1 --lr-scale 0.1
--eval-every 500 --yr-mass 0.1 --label-bank --agent-lam 10 --agent-on-only --shape --ho-w 3 --rm-lr 3e-4 --on-k 13 --on-split
alpasim_nuplan/onpol-train --on-mode self --on on1_L18RB0 --on-lon on_lon --seed {k}`; pool job `--train --vram 28 --cpu 6 --ram 70`.

**Pool: `on1_L18RB0`, not `on1_L18RSG`** (deviation from the task text, decided here): RSG-L18-s{0,1} were trained on `on1_L18RB0`
(RB0-F-s0's rollouts, 17 429 rows; run meta of `train-RSG-L18-s0`), `on1_L18RSG` is RSG's own round-2 rollouts. With `on1_L18RB0` the arm
differs from its reference in `--on-lon` only. The two pools have the same B statistics (table below).

Pack: `experiments/onpol_nuplan/scripts/on_lon_pack.py pack --name L18RB0 --items B P` (P on strictly open rows against the amended D_k,
the defaults). A and C stay out: one mechanism (stop at red) plus its guard (P: on rows with nothing ahead within 60 m, the plan may not
fall below 0.95 of the reference travel - 0.5 m, so B cannot buy itself a generally slower car). The 0.2 % rule (prereg L (iii)) is read
on the served plans; P was to be dropped if it failed it. It passes.

## Pack activation (done before this file; gates `results/stopline/lon_pool-*.json`)

| rows | rows ok | scenes | light line ahead | red in the 4 s window | red now | B active on served plans (rows, scenes) | B mean when active | P open rows | P active (rows, scenes) | P mean when active |
|:--|--:|--:|--:|--:|--:|:--|--:|--:|:--|--:|
| `L18RB0` (training pool) | 17 416 / 17 429 | 1 020 | 22.3 % | 12.4 % | 9.5 % | **0.48 %** (84, 13) | 0.77 m | 12.6 % | 7.1 % (1 234, 146) | 2.32 m |
| `L18RSG` (RSG-L18-s0's rollouts, train) | 17 461 / 17 474 | 1 020 | 22.3 % | 12.5 % | 9.6 % | 0.47 % (83, 14) | 0.61 m | 12.7 % | 7.5 % (1 305, 155) | 2.19 m |
| `L18RSGallheld` (RSG-L18-s0, held) | 7 837 / 7 837 | 447 | 18.4 % | 9.5 % | 7.5 % | 0.13 % (10, 3) | 0.01 m | 18.6 % | 10.3 % (809, 104) | 2.22 m |

B passes the 0.2 % rule on both train pools, narrowly, and is concentrated: on `L18RB0` the 20 most active rows come from 4 scenes. The
logged driver's own future at replay decisions is active 0.39 % with mean 0.013 m (the 1 m margin is almost log-consistent). The light state
is known at 97 % of dense steps (the context's 24 keyframes end at 11.5 s; late 18-decision rows lose the last steps, which then charge nothing).
P carries roughly 15 x B's rows at 3 x the hinge: by mass this arm is mostly the progress guard. **Expectation, stated before the read: B
is a thin signal on nuPlan (13 scenes in training); the PAI stop-line gains below may well not come.**

## Gates (training)

1. Smoke (60 steps, seed 0, tag `RSGB-L18-smoke60`): every logged loss finite; `on/lonB_ho` and `on/lonP_ho` present in the log; step-0 dev
   ADE equals RB0-F-s0's (0.5276).
2. Full runs: dev ADE (memory on) at step 2500 within +0.02 m of RSG-L18's (0.528 / 0.527); G-rely read true (the registered reliance read,
   rely.json). A seed that fails either is reported, not read.

## Lines

**Public 1491 at 10 decisions** (the RSG-L18 guardrail way, `tmp/rn/guard.sh` / `experiments/route_follow/scripts/rf_allrows.sh`:
`ol_chain.sh readlist` over the 10 `all_n10` chunks, `ol_confirm.py` against the stored 2026-10-09 YR10m10-F-s0 rollouts, RSG-L18's stored
guard reads as the reference; 1 479 scenes with a rollout, 136 logs, log-clustered bootstrap). Seed mean `RSGB-L18` against `RSG-L18`:

- P1 mean score: point estimate of RSGB - RSG >= -0.010.
- P2 zeros: point estimate of RSGB - RSG <= +3 (RSG's own seed spread is 2), and the 95 % CI of the difference includes 0 or lies below it.
- P3 base guard (Amendment 5 rule): RSGB not more than 0.02 below YR10m10-F-s0.
Pass = P1 and P2 and P3. Pairs reported: RSGB - RSG, RSGB - YR10m10-F-s0, each seed against RSG-L18. Descriptive: collision / offroad /
corridor split, at-fault zeros, GT distance.

**PAI exam (281 scenes, Tokyo, run later by the coordinator, same image and gains 20 as RSG-L18-s0's taxonomy read; one rollout per scene;
the better public seed only)**, classified by `experiments/route_follow/scripts/rf_tax.py` with its RULES unchanged and the same waitlines:

- E1 stop-line scenes (the 62 with `n_stopline > 0` in `results/taxonomy/pai281_rsg.csv`): reached the line still moving (`stopline_ran > 0`)
  29 -> <= 15.
- E2 route-end zeros (`primary == cor_route_end`): 17 -> <= 10.
- E3 control half (`experiments/alpasim/results/pai_exam/control.txt`, 75 scenes, RSG-L18 zeros there 0 by construction): mean score
  RSGB - RSG >= -0.01 and RSGB zeros there <= 2.
Pass = E1 and E2 and E3. Descriptive (the cost B could have): PAI speed ratio median and rear-end truncations against RSG-L18-s0's (0.84,
70), all-281 mean score and zeros.

**Offline read (held, `on1_L18RSGallheld`, 7 837 rows of RSG-L18-s0's rollouts on 447 held scenes)**: `rf_lonread.py`: B and P on the plans
of RSG-L18-s{0,1}, RB0-F-s0 and RSGB-L18-s{0,1} at the same states; B active share, B sum on the 9.5 % rows with red in the window. Descriptive
only: B is already near its floor on held (0.13 % on the served plans), so the read can show the hinge moved, not that it matters.

Seed for PAI: fewer public zeros, then higher public mean. It is copied with its sidecar to Tokyo `/data/runs/op_parity/runs/RSGB-L18-s<k>/`.
