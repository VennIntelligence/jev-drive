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

## Amendment 1 (2026-10-13, before any RB0 read is analysed)

L5 is read per decision, not per scene: decisions in the junction stretch (20 m before entry to 10 m after exit) of >= 60° turn scenes, split
by whether that decision's own memory covers +20 m; the statistic is the lateral distance to the route, compared between checkpoints on the
same scene and decision with a log-clustered paired bootstrap (B = 10 000). Line: within the covered half, RBS − RB0 median lateral distance
smaller with CI upper bound < 0 on held 452; the uncovered half and the train-split rollouts (RB0 / RB vs YR10m10-F-s0 / YS-L18 references,
same scenes) are reported alongside; per-scene turn completion stays a description on both splits. Reason: held 452 has 24 turn scenes and
the train split 64, with 1–3 scenes per split whose memory covers +20 m at the entry, so the scene-level bucket of section 6 cannot be read.

## Amendment 2 (2026-10-13 ~03:00 JST, before any RBS / RBSG read)

RNONE-L18 reads 80 / 75 zeros vs RB0's 59 (continuation alone hurts the base, as d264 item 4). A gentle-schedule arm RBSG-L18-s{0,1} is
added: --init RB0-F-s0, --steps 2500 --warmup 1 --lr-scale 0.1 --eval-every 500, same rows / hinges / --on-lon on_route as RBS (d264
addendum 6's schedule), judged on the same lines L3 (vs RB0) and L5; RNONEG-L18-s0 (same gentle schedule, no rows) as its control for L4.
Added 2026-10-13 ~03:00 JST before any RBS / RBSG read.

## Amendment 3 (2026-10-13 ~03:40 JST, before any of these reads)

Two drop-one arms on the gentle schedule (--init RB0-F-s0 --steps 2500 --warmup 1 --lr-scale 0.1 --eval-every 500, REC flags --yr-mass 0.1
--label-bank --agent-lam 10 --agent-on-only --shape --ho-w 3 --rm-lr 3e-4, rows --on <pool> --on-k 13 --on-split alpasim_nuplan/onpol-train
--on-mode self): RSG-L18-s{0,1} = RBSG without the route hinge (no --on-lon; S hinges only, d264's YSG recipe on the new base) to attribute
RBSG's extra offroad zeros to the route hinge or to the rows; RBSGc-L18-s{0,1} = RBSG with the route hinge applied only on rows whose memory
covers +20 m (pool copy on1_L18RB0c whose route.npz has ok &= cov20, everything else identical), testing the hypothesis that the hinge on rows
with an empty near-field memory teaches blind turns. Lines: L3 vs RB0 (zeros fewer, CI upper < 0; at-fault upper <= +2; GT not larger) and
L5 per decision as Amendment 1; RNONEG-L18-s0 (60 zeros) is the gentle control. Added 2026-10-13 ~03:40 JST before any of these reads.
