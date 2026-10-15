# ROUTE-DOMINANCE: does the route input push RSG-L18-s0's plans into the contacts of its failing turns? (decision 318)

2026-10-16. Open loop, no training, no new closed-loop runs. Prereg [../../plans/2026-10-16-route-dominance-prereg.md](../../plans/2026-10-16-route-dominance-prereg.md)
(Amendment 1 was written after the CPU stage and before any forward pass). Code: `scripts/rf_rdom.py` and `scripts/rf_rdom.sh`. Box dir:
`$DATA_DIR/runs/route_follow/route_dominance`. Files: [read.json](read.json) (every registered number), [route_dominance_states.csv](route_dominance_states.csv)
(one row per state and checkpoint), figure [../../figs/route_dominance.png](../../figs/route_dominance.png).

**Verdict: BLIND.** At RSG-L18-s0's own states 0.5-3 s before a turn zero, the served plan is already in contact in 65 % of states. Zeroing the
route memory frees only 5.6 % [3.0, 8.6] of those plans. That is below the 12.8 % the same zeroing frees on completed turns. Zeroing the memory
moves the 1-2 s plan by a median of 3 cm. The route does not push these plans into curbs or cars. The plan does not respond to the contact
whether the route is present or not.

## Instrument

- **States.** The on-policy pools hold RSG-L18-s0's own rollouts, with cached Cinque tokens, route memory, simulator agent boxes and road raster:
  - held: `on1_L18RSGallheld`, 7 837 rows. Its run has the same 34 zeros, scene for scene, as the held read.
  - train: `on1_L18RSG`, 17 474 rows on 1 020 train scenes.
- **Turn zeros.** These use rf_appzeros' definitions:
  - failure reason offroad, collision_at_fault or left_corridor_laterally; the "route folds back" scenario errors are excluded;
  - zero step k_z;
  - a turn of at least 45 deg with its entry within [s0 - 10, s_zero + 60] m.
- **Failing states** are the pool decisions k_z - 6 .. k_z - 1.
  - held: 19 turn zeros, 108 states, 11 logs.
  - train: 65 turn zeros, 362 states, 42 logs.
- **Control states** lie on completed turns of non-zero rollouts, within ±25 m of the turn entry.
  - held: 40 turns, 508 states.
  - train: 54 turns, 761 states.
- **APP1X-L18-s0.** Its held read has no frame dumps, and there is no APP1X on-policy pool. So its own 52 new zero states cannot be read. APP1X is
  read only as its weights at RSG's states.
- **Inputs.** Frames and the rest of the ego row are fixed.
  - served: the stored memory.
  - zero: memory set to 0, i.e. `ego[:, 20:] = 0`. This is the training's `--rm-drop` masking. The command is kept.
  - zeroS: zero plus command S. Descriptive only.
  - swap: the other exit. rf_cfpool's lane-graph rule is used: an exit at least 45 deg away, reachable from the route lane. The memory is rebuilt
    through `rmem.window`, its valid bits are ANDed with the served bits, and the command is recomputed.
- **Contact** is read over 0.1-4.0 s with the hinges' own geometry: a Pacifica corner with road-raster sdf < 0, or AgentHinge distance < 0 to
  a counted box. States already in contact at t = 0 are left out of the denominator (8 states).
- **Gates.**
  - G-rep: plans against the stored served plans, median 0.008 m and p95 0.022 m.
  - G-det: two passes are bit-identical.
  - The ego and feature rows equal the pool's.
  - G-feat (Amendment 1: one end slot may differ): 0.984 / 0.996. The strict version gives 0.81 / 0.84.
  - G-swap-sanity: on control states the swap moves the 4 s end a median of 0.18 m toward the other exit. This is positive but small.

## Registered line (RSG-L18-s0 at its own states)

F = the share of failing states whose served plan contacts and becomes contact-free with the memory zeroed. The CI is a log-clustered bootstrap
with B = 10 000.

| set | contact states / logs | F | 95 % CI | line |
|:--|--:|--:|:--|:--|
| pooled (primary) | 301 / 45 | **5.6 %** | [3.0, 8.6] | **BLIND** |
| held | 75 / 10 | 6.7 % | [0.0, 13.6] | BLIND |
| train | 226 / 35 | 5.3 % | [2.4, 8.7] | BLIND |
| by class, pooled: offroad | 171 / 25 | 4.7 % | [1.6, 8.3] | BLIND |
| collision | 85 / 16 | 7.1 % | [1.2, 14.1] | BLIND |
| corridor | 45 / 14 | 6.7 % | [0.0, 16.0] | BLIND |

The verdict holds at the 0.1-2.0 s horizon as well: 5.3 % [2.4, 9.2]. With the command also set to S (zeroS), 13.3 % [8.6, 18.4] are freed,
still in the BLIND range.

## Baselines and secondaries (pooled)

| | failing | control |
|:--|--:|--:|
| served plan in contact (0.1-4 s) | 65 % [57, 73] | 11 % [8, 15] |
| contacts freed by zeroing | 5.6 % [3.0, 8.6] | 12.8 % [6.2, 22.0] |
| contacts **induced** by zeroing (contact-free served plans) | 10.6 % [5.0, 17.6] | 5.8 % [3.6, 8.1] |
| swap leads a contact-free plan into contact (S1) | 17 % [0, 44] (n 29, 8 logs) | 3 % [0, 8] (n 62) |
| zeroing displacement, mean abs lateral 1-2 s, median / p90 (m) | 0.033 / 0.15 | 0.028 / 0.14 |
| zeroing displacement, abs lateral 4 s, median / p90 (m) | 0.40 / 1.50 | 0.21 / 1.16 |
| zeroing heading change at 4 s, abs median (deg) | 3.3 | 1.3 |
| swap displacement, abs lateral 4 s, median / p90 (m) | 0.40 / 4.4 | 0.38 / 4.1 |

- On all pool rows (S2), zeroing moves the plan by:
  - turn rows (route turns at least 30 deg within 60 m): 0.012 m at 1-2 s and 0.07 m at 4 s;
  - straight rows: 0.005 m and 0.02 m.
- The swap exists for only 51 / 616 held and 102 / 1 123 train states. Most of the rest are turn-only lanes (`no_other_exit`), so S1 is thin.
- Reading: at the failing states, removing the route takes more plans *into* contact (10.6 %) than out of it (5.6 %). The route's net effect on
  contact is protective or nil, not pushing.

## APP1X-L18-s0 weights at the same states

| | RSG-L18-s0 | APP1X-L18-s0 |
|:--|--:|--:|
| F pooled | 5.6 % [3.0, 8.6] | 9.5 % [4.6, 15.0] (BLIND) |
| failing served contact | 65 % | 52 % |
| induced by zeroing (failing) | 10.6 % | **30.3 % [21.3, 40.2]** |
| zeroing displacement 1-2 s / 4 s, median (m) | 0.033 / 0.40 | 0.16 / 0.85 |
| F corridor | 6.7 % | 28 % [10, 50] (UNDECIDED) |

- APP1X's route branch moves its plans 3-5 times more than RSG's.
- Removing APP1X's route branch creates contact in 30 % of failing states, i.e. its route keeps plans *out* of contact.
- Only the corridor class reaches the undecided band.
- APP1X's own failing states were not readable, as noted under Instrument.

## Figure

[../../figs/route_dominance.png](../../figs/route_dominance.png). Each panel shows the front frame (the dumped CAM_F0) over a BEV. The BEV has
the road raster, agent boxes at t0 (solid) and 2 s (faint), the served (blue), zeroed (orange) and swapped (green) plans, and the route-memory
dots. Lateral is drawn with left on the left.

1. A held corridor state. It is one of the rare cases where zeroing frees the plan: the served arc grazes the island.
2. A train collision state. Served and zeroed are both in contact; zeroing straightens the plan into the box ahead.
3. A train corridor state where the other-exit route turns the plan into a parked box.
4. A completed turn: the swap sends the plan to the other exit, while served and zeroed stay together.

Look at panels 1-2: whether or not the route is present, the plan sweeps through the obstacle.

## Limits

- Open loop at fixed frames. The read measures the route's marginal effect at the failing state. It does not measure what the route did
  seconds earlier in closed loop, such as committing the approach.
- States earlier than 3 s before the zero were not read.
- Zeroing is in-distribution for offline rows, which are trained with a 25 % drop. Pool rows were never zeroed in training
  (`rm_drop_pool false`).
- Contact uses the rollout's simulator boxes. A changed plan would meet reactive agents differently in closed loop.
- The swap covers 13 % of states.
- Evidence strength: medium. It is one checkpoint seed, on its own states; the line is clear in both sets and all three classes.
