# ROUTE-ONLY results (2026-10-13)

Prereg: [../plans/2026-10-13-route-only-prereg.md](../plans/2026-10-13-route-only-prereg.md) (commit `0689dd80`; Amendment 1 R-bridge
`dddbf628`, both before any held / public read). Switch: `JEV_ROUTE=off|own|bridge|log` in `experiments/alpasim/lib/route_path.py`, hooked
into `sh30_driver.py` (default off). Checkpoint RSG-L18-s0 in every arm, every other serving switch off (as its stored reads). Box only
(Tokyo dropped by the coordinator; PAI 281 is a follow-up, `pai_driver.py` is not hooked). Box runs `$DATA_DIR/runs/route_only/`
(smoke/, held/<arm>/, pub10/own/, report/, figs/); copies here: [held/](held/), [pub/](pub/), [smoke/](smoke/),
[offset_and_changes.txt](offset_and_changes.txt).

Arms: **R-own** = path from the carried route memory (Hermite blend from the ego over Lb = clip(2 s x v, 8, 30) m), poses at the
plan's own arc lengths; where the memory starts ahead of the ego (the gap) the model plan is kept. **R-bridge** (Amendment 1) = R-own
but the gap is bridged by the Hermite curve from the ego into the route start. **R-log-priv** = R-own's path at the human log's arc
lengths (privileged; reference only).

## Smoke (2 held turn scenes, 18 decisions)

- Switch off reproduces the stored RSG-L18-s0 served poses **bitwise on 36 / 36 decisions**, with identical scores, so the stored read is the paired reference.
- Served arcs match the plan arcs within 0.004 m (own), the log arcs within 0.022 m (log) and the plan arcs within 0.04 m (bridge).
- Coverage: the memory starts 32-34 m ahead of the ego for the first 12-13 decisions. Both junctions lie inside that gap, because
  AlpaSim never sends the near 40 m and the memory only holds points that were once 40-80 m ahead.

## Held 460 / 18 decisions (paired on 452 scenes, 39 logs; log-clustered 95 % CI, B = 10 000)

| arm | mean score | zeros | collision / offroad / corridor | at-fault | GT (m) | progress_rel (460) | travelled / log m (460) | turn completion (24) | route served (decisions) | poses on route (mean of 8) |
|:--|--:|--:|:--|--:|--:|--:|:--|:--|--:|--:|
| RSG-L18-s0 (stored) | 0.8809 | 26 | 4 / 17 / 6 | 21 | 1.135 | 0.788 | 45.9 / 58.5 | 66.7 % [45, 86] (16 / 1 event / 7 short) | - | - |
| **R-own** | 0.8781 | 27 | 5 / 17 / 6 | 22 | 1.148 | 0.788 | 45.7 / 58.5 | 66.7 % [45, 86] (identical outcomes) | 47.6 % | 2.53 |
| R-bridge (Amendment 1) | 0.7404 | **90** | 12 / **74** / 7 | 86 | 1.128 | 0.800 | 41.5 / 58.5 | 45.8 % [27, 68] (11 / 11 event / 2 short) | 99.6 % | 7.97 |
| R-log-priv (privileged) | 0.9328 | 30 | 8 / 18 / 6 | 25 | 0.916 | 0.941 | 54.2 / 58.5 | 91.7 % [77, 100] (22 / 1 / 1 off / 0 short) | 55.8 % | 3.25 |

| a - RSG-L18-s0 | mean score | zeros | at-fault zeros | GT (m) |
|:--|:--|:--|:--|:--|
| R-own | -0.0028 [-0.0127, +0.0058] | +1.0 [-3.2, +5.7] | +1.0 [-3.2, +5.7] | +0.013 [-0.018, +0.043] |
| R-bridge | -0.1405 [-0.1800, -0.0983] | +64.0 [+45.0, +82.0] | +65.0 [+45.5, +84.6] | -0.007 [-0.181, +0.148] |
| R-log-priv | +0.0520 [+0.0298, +0.0737] | +4.0 [-4.7, +12.8] | +4.0 [-3.5, +12.0] | -0.220 [-0.322, -0.111] |

Turn-completion differences (rf_follow, pp): R-own +0.0 [+0.0, +0.0], R-bridge -20.8 [-44.4, +4.5], R-log-priv +25.0 [+0.0, +47.8].
Changed scenes vs the reference (score > 0 vs 0): R-own broke 3 (2 offroad, 1 collision) and fixed 2 (offroad); R-bridge broke 70
(59 offroad, 8 collision, 3 corridor) and fixed 6; R-log-priv broke 14 (7 offroad, 7 collision) and fixed 10.

Coverage of R-own (8 129 decisions): the route path is served on 47.6 % of decisions (gap 52.1 %, no route 0.1 %, behind the
memory's end 0.2 %). The served share rises with the decision index: 9 % at k = 0, 30 % at k = 3, 49 % at k = 9, 78 % at k = 17. The
median gap where the plan is kept is 19.7 m. Of the 24 held turns, 23 are uncovered at 20 m before the entry (rf_follow's cover
flag), so the turn is driven by the model in R-own.

## Public 1491 / 10 decisions, R-own (paired on 1 479 scenes, 136 logs)

| arm | mean score | zeros | collision / offroad / corridor | at-fault | GT (m) | progress_rel (1491) | route served |
|:--|--:|--:|:--|--:|--:|--:|--:|
| RSG-L18-s0 (stored guard read) | 0.9473 | 46 | 8 / 31 / 8 | 39 | 0.792 | 0.850 | - |
| **R-own** | 0.9406 | 56 | 10 / 38 / 9 | 48 | 0.800 | 0.850 | 30.1 % (9 % at k = 0, 45 % at k = 9) |
| R-own - RSG | **-0.0068 [-0.0105, -0.0031]** | +10.0 [+4.6, +15.6] | | **+9.0 [+3.1, +15.0]** | +0.008 [+0.001, +0.017] | | |

R-own broke 10 public scenes (8 offroad, 2 collision) and fixed none.

## Lines (as registered)

| line | R-own | R-bridge |
|:--|:--|:--|
| H1 corridor zeros <= 2 (reference 6) | **fails** (6) | fails (7) |
| H2 verdict | **worse by the letter**: mean CI lower -0.0127 >= -0.02 holds, but the at-fault CI upper is +5.7, above +5. The point estimates are a tie (+1 zero) | worse (-0.14, +64 zeros) |
| H3 GT CI upper < 0 | fails (+0.013 [-0.018, +0.043]) | fails |
| H4 turn completion >= 80 % | fails (66.7 %, identical to RSG) | fails (45.8 %) |
| G1 public: mean >= -0.01 and at-fault CI lower <= 0 | **fails** (mean -0.0068 passes; at-fault +9.0 [+3.1, +15.0]) | not run (conditional on H2) |

**The user's hypothesis is not supported in the form AlpaSim allows.** With the route carried as memory, the route path cannot be served
where it matters, and serving it elsewhere costs zeros.

## What carries it (mechanisms, measured)

1. **Coverage gap (R-own).** The route messages hold only the points 40-80 m ahead, so the memory never contains the first ~40 m of the
   episode's route. Junctions that are near at the start (all 24 held turns but 1) are driven by the model. R-own equals RSG on all 24
   turn outcomes and serves the route on only 48 % (held) / 30 % (public, 10 decisions) of decisions.
2. **Route the ego is not on.** On served decisions the ego's lateral offset from its own carried route has median 0.39 m (held) /
   0.75 m (public), but p90 5.1 / 11.1 m. 9 % / 13 % of served decisions are more than 6 m off. In 7 of the 10 public breaks the max
   offset on a served decision is 17-36 m. The memory then describes another lane or road (route snapping, parking-lot and loop routes;
   figs/held_own_41da8ea7.png: the memory runs behind and left of the ego), and the blend drags the car toward it: offroad. Scenes
   with any served decision more than 6 m off have 14 zeros under R-own against 7 for RSG (public).
3. **Bridging the gap is worse (R-bridge).** A smooth curve from the ego to where the route resumes (~30 m ahead, often on the exit
   road) does not know the lane geometry. It cuts corners and leaves the road: offroad zeros 74 vs 17, 70 broken scenes, turn
   completion 46 %. The model's own junction path is better than a geometric bridge to the route.
4. **Speed is the bigger lever (R-log-priv, privileged).** Same path rule, log arc lengths: score +0.052, progress_rel 0.94 vs 0.79, GT
   -0.22 m. Turn completion goes from 67 % to 92 %, and all 7 of RSG's "short" turns (still on the route when the rollout ended, too
   slow) are completed. In the gap decisions this arm serves the model's path at the log's speed, so most of its gain is speed on the
   model's own path (decision 285's slowness). It costs 4 more collisions (8 vs 4), from log timing on a different position. No
   submission can use it.

## Caveats

- One checkpoint (s0), one read per arm. The reference is the stored read, and S1 shows the switch-off re-run is bitwise identical,
  so differences come from the switch, not from re-run noise.
- The blend rule (Lb = 2 s x v, Hermite) was chosen before the read and not tuned. A plausibility gate (serve only when the ego is
  within ~2-3 m of the route) would remove mechanism 2 but cannot fix mechanism 1; it was not registered or run.
- The "route" is AlpaSim's lane-snapped log path. Where it is wrong (snaps, loops), it is wrong for any route-following driver.
- PAI 281 not run (Tokyo full). PAI sends 10 Hz route messages with the same 40 m offset; mechanism 1 applies unchanged. The route
  snaps to neighbouring lanes there with p95 3.8 m (decision 284), so mechanism 2 is likely larger.
- Box use: 28 read jobs (smoke 4, held 3 x 6, public 10), ~20-35 min each, two per card. GPU utilisation is low because these reads are
  renderer/CPU bound.

## Figures (BEV in the decision's rear-axle frame; grey band = route path served, grey dots = the fed memory window, red = model plan,
teal squares = served poses, lines = ego tracks of each arm over the whole rollout)

- [figs/held_own_8c9e3fd9_k4.png](../figs/held_own_8c9e3fd9_k4.png): right turn, R-own at k = 4. The memory starts 32 m ahead, beyond the turn, so the served
  poses are the model's (0 / 8 on route) and all tracks coincide. This is mechanism 1.
- [figs/held_bridge_94a3b8d4.png](../figs/held_bridge_94a3b8d4.png): right turn, R-bridge at k = 2. The Hermite bridge to the route start on the exit road turns
  inside the model's plan, and the R-bridge track ends at an offroad event while RSG and R-own complete the turn. This is mechanism 3.
- [figs/held_bridge_51552f78.png](../figs/held_bridge_51552f78.png): left turn, R-bridge at k = 2. The bridge runs wider than the model plan and the turn ends in an event
  that RSG did not have.
- [figs/held_log_ffe92084.png](../figs/held_log_ffe92084.png): right turn that RSG left "short". R-log-priv at k = 10 is on the route and finishes the turn at the log's
  speed. This is mechanism 4.
- [figs/held_own_41da8ea7.png](../figs/held_own_41da8ea7.png): a broken scene. The carried memory lies 28 m to the side and behind the ego, and R-own (and R-log) are
  pulled into a hard left loop, where RSG's own path stays on the road. This is mechanism 2.
- [figs/pub_own_d743862c.png](../figs/pub_own_d743862c.png): a public break at k = 0. The route path turns left across the junction while the model goes straight.
- [figs/smoke_bridge_8c9e3fd9_k4.png](../figs/smoke_bridge_8c9e3fd9_k4.png): the smoke junction decision (model plan vs bridged route path).
