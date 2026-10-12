# Pre-registration: route-night (fresh route-following base RB, on-policy round RBS-L18), 2026-10-13 night

Task book: [research/next-round/route-night.md](../../../research/next-round/route-night.md) (Chinese). Lines copied from its section 6
verbatim before any reading; this file is committed and pushed before the first gate is read. Amendments are appended, never edited in place.

## Arms (all from the shipped weights; no old checkpoint is an init)

| Tag | Init | Recipe |
|:--|:--|:--|
| `RB0-F-s{0,1}` | shipped weights | YR10m10 recipe (10 000 steps, `--yr-mass 0.1 --label-bank`) + route-memory branch (`rm_train.py`, `--rm-lr 3e-4`) |
| `RB-F-s{0,1}` | shipped weights | RB0 + counterfactual pool `on1_CF` 12 rows / batch, `--on-mode self --on-lon on_route` |
| `RBS-L18-s{0,1}` | `RB-F-s0` (or `RB0-F-s0` if RB is not trained / fails G-rely) | d264 full schedule 10 000 steps, `--on on1_L18RB --on-k 13 --on-mode self --agent-lam 10 --agent-on-only --shape --ho-w 3 --on-lon on_route` |
| `RNONE-L18-s{0,1}` | same base as RBS | same steps, no on-policy rows |

References (stored reads, not rerun): `YS-L18-s{0,1}`, `YS2G-L18-s{0,1}` (`runs/onpol_long/read/<tag>/run`), `YR10m10-F-s{0,1}` (`runs/alpasim_long/held/h18/`).

## Counterfactual pool gates (before full-scale build)

- G-cf1: 50 random rows, median lateral gap between the first counterfactual message and the real route at equal arc length >= 2 m.
- G-cf2: >= 95 % of counterfactual polyline samples inside the road raster (sdf_road > 0).
- G-cf3: 6 BEV figures (real route, counterfactual route, logged future, raster) inspected by the coordinator before the full build.

## Offline lines (dev rows)

- **L1 G-rely** (d282's line): junction rows, memory masked, ΔADE >= 0.01 m, and junction − straight difference CI lower bound > 0, both seeds.
- **L2 G-swap** (RB only): under the counterfactual memory the 1.5–2.0 s lateral move >= 50 % of the needed move (median); 4 s endpoint on the
  counterfactual side >= 70 %; G-ST: straight / shifted-by-<= 0.5 m routes move <= 0.10 m.
- **L0 G-real**: real-route dev ADE not above RB0 + 0.01 m (the counterfactual rows did not bias imitation).

## Closed-loop lines (held 452, 18 decisions, log-clustered paired bootstrap, B = 10 000; `ol_report.py` conventions)

- **L3** RBS vs RB0 (or RB): fewer zeros with CI upper bound < 0; lateral GT distance not larger (CI contains 0 or < 0); at-fault zeros CI upper bound <= +2.
- **L4** RBS vs RNONE: fewer zeros (CI upper bound < 0).
- **L5 route following**: >= 60° turn completion, memory-covered bucket RBS − RB0 >= +10 percentage points; junction-segment lateral distance to the route median < 1.0 m.
- Reference (no line): YS-L18, YS2G-L18, YR10m10-F held readings and completion rates in the same table.

"Good base" = L1 passes (dependence established) and L3, L4 pass (rows useful in closed loop). L5 is a mechanism reading, reported either way.

## Pool gates (on-policy rows of the base)

Rows ≈ 17 k (d264 order); `route.npz` ok >= 0.95, cov20 share recorded; G-route `route0_share_005` >= 0.99 (d282's 0.993).

## Serving

nuPlan reads: all serving switches off. PAI (Tokyo): `JEV_VCONT=1.0 JEV_LEAD=1`; controller gains per expert B's pilot verdict (20 if it passes, default otherwise). VHIST, curvature cap untouched.
