# Q3: a drift-free image-command fine-tune? (navtrain + CARLA, 2026-10-03, lane IMG2)

Pre-registration with the dated addenda (course change, sky + green arm, deviations):
[../plans/2026-10-04-img-cmd-ft2-prereg.md](../plans/2026-10-04-img-cmd-ft2-prereg.md). Tables: [q3/band/](q3/band/) (first design) and
[q3/sky/](q3/sky/) (`<model>_vs_O_<set>[_sg].{md,csv}`, `_guards.json`, `select*.json`). Scripts: `img2_{bank,train,eval,chain}`,
`img3_{bank,train,chain,sheet}`, `img3g_chain.sh`.

**Answer: no.** Five design changes and nine configs did not bring the no-overlay drift at junction frames under 0.10 m while keeping
uptake >= 0.4, in either design. The pre-registered stop rule applied in both arms (3 configs each), so no seeds 1-2 and **no
closed-loop smoke** were run.

## Common setup

Port of Cinque (op_adapt_l / L3), stage 4 + plan pathway trained, 3 000 steps (SC / GC 2 000), batch 56, three configs per arm trained in
parallel on one card (~25 min) and chosen only on dev. Changes against Q2 (all arms):

- **Large distillation pool**: L3's nav (lb_h1train minus the eval logs, 1 216 frames) / WOD (2 780) / CARLA p6 (2 588) frames plus
  the no-overlay junction frames, teacher = the shipped model. 22-24 of 56 rows per batch.
- **CARLA junctions in training**: the 179 re-recorded junctions split by route, `b2d/img-carla-{train,dev,test}`: test 60 routes
  (the 13 decision-95 routes present, including the four turn_agree-failure routes 27297 / 27043 / 9196 / 24944, plus random), dev 18,
  train 101 (20 train samples sharing a junction node with test dropped). Samples: train 376 / dev 75 / test 225.
- **Timing from the original**: overlay targets keep the original's own speed profile, because the Q2 drift was mostly
  longitudinal (|dx| 0.22 vs |dy| 0.07 m) and the logged speed made the overlay a "go" signal.
- **Negatives**: the overlay on lane-keeping frames, trained to keep the original's plan.

Lines (pre-registered): `none` drift median (0-5 s mean L2 vs the original) <= 0.10 m on nav junction, nav straight and CARLA
junction frames; gate-family uptake >= 0.4 on navtrain eval and CARLA test (dev selection: >= 0.45 on dev); lane keeping unchanged.

## Arm 1, first design: band + barrier (superseded)

The I-row target was the branch centreline, timed by the original. None of the configs passed dev (band dev uptake / drift: A 0.30 nav /
0.46 CARLA at drift 0.20 / 0.13 / 0.47 m; B 0.05 / 0.22 at 0.17 / 0.10 / 0.35; C 0.02 / 0.08 at 0.14 / 0.10 / 0.27). Test, config A
only (B and C were stopped by the course change): band uptake 0.32 [0.26, 0.39] nav, 0.40 [0.36, 0.45] CARLA; drift 0.22 / 0.13 / 0.44 m.
The user then dropped occluding overlays (barrier, wall, cones, fills: they hide vehicles and pedestrians) and white road paint (it
blends with real markings). The user also ruled out map geometry at test time, so these results stay in the record only.

## Arm 2: sky arrow (primary, no privileged input at test time)

The overlay is a screen-fixed magenta arrow with a black outline above the horizon: road view centred at rows 1-29, wide view top-left.
Its shape is left, straight or right, and its size follows the navigation distance to the junction. Review sheet:
[../figs/sky_sheet.png](../figs/sky_sheet.png). The only test-time inputs are the command and the distance to the turn.

The target is now a **residual**: the original's own plan + (centreline of the commanded branch − centreline of the branch the original
already takes). If the command is the branch the original takes, the target is its plan unchanged. Configs: SA (lam_d 10 / lam_c 1),
SB (30 / 3), SC (SA with lr 5e-5, 2 000 steps).

| model | sky uptake nav eval | correct nav | sky uptake CARLA test | correct CARLA | `none` drift nav junction / straight / CARLA junction (m) | L3 dev pools drift |
|:--|:--|:--|:--|:--|:--|:--|
| original (zero-shot) | -0.00 [-0.01, 0.00] | 0.50 | 0.00 [0.00, 0.01] | 0.45 | 0 | 0 |
| SA | 0.29 [0.23, 0.34] | 0.71 | 0.39 [0.37, 0.42] | 0.78 | 0.13 / 0.07 / 0.27 | 0.07-0.08 |
| SB | 0.13 [0.09, 0.17] | 0.59 | 0.23 [0.20, 0.26] | 0.68 | **0.09 / 0.05** / 0.19 | 0.05 |
| SC | 0.00 | 0.51 | 0.00 | 0.46 | 0.10 / 0.05 / 0.23 | 0.04-0.06 |

Dev (selection): SA 0.32 / 0.43 uptake at 0.11 / 0.10 / 0.24 m; SB 0.14 / 0.27 at 0.08 / 0.07 / 0.20; SC ~0 at 0.09 / 0.05 / 0.22. None passed.
n: nav eval 385 junction / 293 straight frames (247 logs), CARLA test 225 samples (60 routes); CIs paired cluster bootstrap by log / route.

- **Zero-shot the arrow does nothing to the route** (uptake 0.00 in both domains). It shortens the 4 s plan by 1.2 m (nav) / 0.6 m
  (CARLA), and a magenta disc in the same place moves the plan as much as the arrow does.
- **Fine-tuned, the command reads through**: SA moves the plan toward the commanded branch by 1.7 m (nav) / 9.3 m (CARLA) at 4 s.
  Right turns are taken up best (toward L / S / R +0.7 / +0.2 / +1.7 m nav, +1.7 / +3.3 / +9.3 m CARLA). CARLA uptake is just under
  the line (0.39), nav is not (0.29).
- **Lane keeping is unchanged**: the 3 s lateral error on straight frames moves by <= +0.01 m, with or without the arrow. Without an
  overlay the model does not guess the branch on nav (toward-taken -0.03 [-0.08, +0.01] m). On CARLA, SA moves 0.63 m *away* from the
  taken branch [-1.07, -0.22].
- **Controls (not trained)**: after the fine-tune, the uninformative disc and a wrong arrow (an exit the junction does not have) move
  the plan sideways. On CARLA the mean |dy| at 4 s is 3.0 m for the disc and 7.5 m for the wrong arrow under SA, against 0.28 / 0.29 m
  for the original. On nav it is 0.44 / 1.0 m. Part of what SA learned is "magenta in the sky means turn", not the shape. SB halves this
  (CARLA 2.5 / 5.6 m).

## Arm 3: sky arrow + green ground line placed by perception (separate arm)

The green line (0.3 m) follows the centre of the original's own t0 inner lane lines up to the navigation distance. It then turns with a
fixed-radius quarter arc toward the command (left 12 m, right 8 m), or continues straight to 60 m. No map is used. Configs GA / GB / GC
match SA / SB / SC.

| model | sg uptake nav eval | correct nav | sg uptake CARLA test | correct CARLA | `none` drift nav junction / straight / CARLA junction (m) |
|:--|:--|:--|:--|:--|:--|
| original (zero-shot) | -0.01 | 0.51 | 0.05 [0.03, 0.06] | 0.48 | 0 |
| GA | 0.33 [0.27, 0.38] | 0.82 | 0.39 [0.35, 0.43] | 0.84 | 0.15 / 0.09 / 0.37 |
| GB | 0.22 [0.17, 0.27] | 0.66 | 0.30 [0.27, 0.34] | 0.73 | 0.11 / 0.08 / 0.27 |
| GC | 0.04 | 0.55 | 0.15 [0.13, 0.18] | 0.60 | 0.13 / 0.09 / 0.33 |

The green line adds a little: GA vs SA has correct 0.82 vs 0.71 nav and 0.84 vs 0.78 CARLA, and GB vs SB has uptake 0.22 vs 0.13.
It costs drift (CARLA 0.37 vs 0.27 m). Zero-shot the line moves CARLA plans by 5% of a branch switch. The wrong-command control
moves the plan even more (CARLA |dy| 8.1 m under GA).

## Why the drift line is not reachable here

- On ordinary frames the fine-tune is already drift-free: L3's nav / WOD / CARLA dev pools drift 0.04-0.08 m in every sky config.
  Junction frames drift 2-4x more, and CARLA junctions most.
- The 0.10 m line at junctions sits **below the original's own sensitivity** there. A magenta disc in the sky, which carries no
  information and covers no road, moves the shipped model's junction plan by a median 0.17 m (nav) / 0.24 m (CARLA). The band on
  every branch moves it 0.37 / 0.47 m. At a junction the plan sits between branches and small input changes move it. Any weight change
  big enough to read a command moves it by the same order.
- Distillation weight trades uptake against drift on one curve: SA -> SB -> SC and GA -> GB -> GC. Nav junction drift reaches the line
  (SB 0.09 m) only once uptake is 0.13. On CARLA junctions no config gets under 0.19 m.
- Two changes worked as intended. The residual target cut nav junction drift from 0.20-0.29 m (Q2, Q3 band) to 0.09-0.13 m at
  comparable uptake. Original-model timing removed the longitudinal "go" leak.

## Limits

One seed per config; three configs per arm (the plan's stop rule). The CARLA test uses the original's re-recorded P4-rig frames, not
closed-loop frames. Each history frame is drawn from t0 geometry (the green line from t0 perception), while at test time each frame
would use its own. The sky arrow sits above the horizon, where far traffic lights can appear. Nav eval frames are mostly
slow and near the stop line.
