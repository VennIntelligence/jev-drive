# LINEAGE-PAI281: P0 -> RB0-F-s0 -> NTV0-L18-s0 on PAI 281, by serve step (2026-10-20)

PAI 281 closed-loop reads that complete the model lineage: shipped Cinque (P0, no adapter), RB0-F-s0, NTV0-L18-s0. The table also holds
the stored RSG / JTN / NTV0 / B / B2 rows for comparison. Code: `scripts/lineage.sh` (reads through `experiments/alpasim/scripts/pai_box.sh`
unchanged, rf_tax extracts, summary) and `scripts/lineage_pai281.py` (table). Box outputs: `$DATA_DIR/runs/defect_fix/lineage/`
(`read/<arm>/`, `tax/<label>.pkl`, `lineage_pai281.{md,json}`).

## Setup

- **Same read as every stored PAI 281 read.** The chunk lists are NTV0-L18-s0 + ACC + GAP's (`CHUNKS_FROM`; the same 8 lists, md5
  0f7f9449, as every row below). CONC 4, g10 (`controller.gains.rel_front_steering_angle_weight=10`), driver `pai`.
- **ALPASIM_FAST=1 for every new arm.** That matches every stored ACC + GAP read and N0. The two RSG g10 rows (`pai_g10/g10`, `/acc`) predate
  the switch and ran on the official camera set; FAST=1 is bit-identical to it (36 / 36 on g10 c001, `experiments/alpasim/results/pai_fast.md`).
  The setting does not differ, so the NTV0 + ACC + GAP anchor re-read was not run.
- **Native = no serve switch** (`SERVE=` empty). `serve_fix.new` then returns None and the driver returns the plan bit for bit: no
  speed-continuous serving, no lead MPC, no ACC. No code change was needed. The driver defaults that every row of this table has stay on:
  the track_ref `gate` reference (decision 309, `JEV_TRACK` unset) and route memory. A read of the plan with track_ref off as well would be
  `SERVE=JEV_TRACK=0`, an env switch only; it was not run.
- **Scenes and classifier.**
  - Scenes: those the read scored without a driver RPC error (decision 327).
  - Zero and partial classes: `pai281.ptype` / `zclass` (`nx_regress.failure_type`).
  - Corridor sub-split: `rf_tax.classify`'s primary label on an rf_tax extract of each read. Overshoot = `cor_route_end`, the ego passed the
    end of the human's log. Route turn = `cor_straight_at_turn` / `cor_turn_under` / `cor_turn_over` / `cor_wrong_exit`. Other = `cor_drift`
    / `cor_lane_change`.
  - The stored rows reproduce `tmp/pai0/inv.json` exactly (means, zeros, rear-hit partials checked on 6 rows).
- **Dropped arms (scope change, 2026-10-20, before launch):** P0 + ACC, RB0-F-s0 + ACC, RB0-F-s0 + ACC + GAP, NTV0-L18-s0 + ACC without
  GAP, NTV0-L18-s0 + LEAD + GAP without ACC, and NTV0-L18-s0 + ACC + GAP at JEV_GAP=1.0,4.0. Stored evidence covers them:
  - decision 330: default gap against 0.7,2.7 on PAI, collisions 9 against 9, rear-ended 40 -> 28;
  - decision 381: where NTV0 + ACC + GAP loses points;
  - the RSG serve steps below.
- **Cost.**
  - 5 new reads, 40 chunk jobs, each on its first try: 13.3 stack-hours, about 4.4 card-hours at 3 stacks per card.
  - RB0-F-s0 native, added later (`lineage.sh add RB0-NAT`): 8 chunks on the first try, 2.8 stack-hours (about 0.9 card-hours).
  - Wall time 1.26 h on cards 0-4. The 18 rf_tax extracts were CPU pool jobs, a few minutes each.

## Short answer

- **The adapter is worth more than any serve step.** On P0, VC+LEAD changes nothing over native (+0.008 [-0.030, +0.044]). ACC + GAP
  then adds +0.10, but that leaves P0 at 0.351. RB0-F-s0 sits +0.31 above P0 at VC+LEAD and is level with RSG-L18-s0 (+0.016 [-0.006,
  +0.039]). NTV0 adds +0.055 [+0.026, +0.086] over RB0-F at VC+LEAD.
- **NTV0 native loses 0.17 to VC+LEAD, and most of the loss is front collisions.** AF-front zeros are 43 native against 2 with the lead
  MPC.
- **The plan's own lead following broke at the adapter step (P0 -> RB0-F), not at NTV0.** Read natively:
  - AF-front: P0 16, RB0-F-s0 50, NTV0-L18-s0 43.
  - 38 of NTV0's 43 are also AF-front for RB0-F; only 6 of RB0-F's 50 are AF-front for P0.
  - On RB0-F's 50 AF-front scenes, P0 native is 7 ok, 13 rear-hit partials (it stops behind the lead and is hit from behind), 6 AF-front, 15
    corridor, 3 offroad, 3 AF-side, 2 slow, 2 no-launch.
  - With the lead MPC (RB0-F VC+LEAD), 26 of the 50 score 1 and 5 stay AF-front.
  - Along the native chain the mean still rises: RB0-F - P0 +0.134 [+0.079, +0.189], NTV0 - RB0-F +0.065 [+0.036, +0.097]. The adapter's
    route-following gain outweighs the lead following it loses.
- **On P0, ACC + GAP moves zeros from offroad and collisions into the corridor.** Offroad 70 -> 47, corridor 75 -> 94 (other corridor
  24 -> 44). Rear-hit partials halve, 54 -> 25, as they do for NTV0 (75 -> 26).

## Table

| read | n | mean | mean (common) | zeros | AF-front | AF-side | offroad | corridor | overshoot / route-turn / other | other zeros | rear-hit partial | slow partial | other partial | RPC excluded |
|---|--:|--:|--:|--:|--:|--:|--:|--:|---|--:|--:|--:|--:|---|
| P0, native | 280 | 0.2419 | 0.2396 | 161 | 16 | 11 | 58 | 76 | 9 / 40 / 27 | 0 | 56 | 12 | 14 | 09528162 |
| P0, VC+LEAD | 281 | 0.2487 | 0.2469 | 172 | 11 | 16 | 70 | 75 | 14 / 37 / 24 | 0 | 54 | 12 | 3 | - |
| P0, +ACC+GAP | 281 | 0.3511 | 0.3480 | 164 | 15 | 8 | 47 | 94 | 18 / 32 / 44 | 0 | 25 | 5 | 2 | - |
| RB0-F-s0, native | 281 | 0.3757 | 0.3711 | 135 | 50 | 3 | 29 | 53 | 22 / 20 / 11 | 0 | 51 | 31 | 1 | - |
| RB0-F-s0, VC+LEAD | 280 | 0.5564 | 0.5527 | 81 | 5 | 5 | 18 | 53 | 18 / 28 / 7 | 0 | 64 | 34 | 1 | 0cc7fa67 |
| NTV0-L18-s0, native | 281 | 0.4412 | 0.4424 | 107 | 43 | 5 | 16 | 43 | 24 / 14 / 5 | 0 | 62 | 40 | 2 | - |
| NTV0-L18-s0, VC+LEAD (stored N0) | 281 | 0.6114 | 0.6091 | 59 | 2 | 3 | 15 | 39 | 17 / 17 / 5 | 0 | 75 | 42 | 0 | - |
| NTV0-L18-s0, +ACC+GAP (stored) | 280 | 0.7443 | 0.7441 | 59 | 3 | 0 | 12 | 44 | 23 / 15 / 6 | 0 | 26 | 11 | 0 | 49bcab6f |
| NTV0-L18-s1, +ACC+GAP | 281 | 0.7393 | 0.7380 | 62 | 3 | 1 | 14 | 44 | 22 / 17 / 5 | 0 | 24 | 11 | 0 | - |
| RSG-L18-s0, VC+LEAD | 280 | 0.5713 | 0.5729 | 75 | 3 | 6 | 16 | 50 | 18 / 25 / 7 | 0 | 69 | 35 | 1 | 13d646af |
| RSG-L18-s0, +ACC | 280 | 0.6760 | 0.6739 | 75 | 5 | 4 | 16 | 50 | 21 / 19 / 10 | 0 | 36 | 6 | 0 | 054b5901 |
| RSG-L18-s0, +ACC+GAP | 281 | 0.6909 | 0.6917 | 76 | 7 | 2 | 14 | 53 | 21 / 18 / 14 | 0 | 26 | 8 | 0 | - |
| JTN-L18-s0, VC+LEAD | 280 | 0.6011 | 0.6001 | 58 | 7 | 2 | 13 | 36 | 18 / 17 / 1 | 0 | 76 | 39 | 3 | 90ab8c24 |
| JTN-L18-s0, +ACC+GAP | 281 | 0.7140 | 0.7119 | 65 | 4 | 1 | 19 | 41 | 21 / 13 / 7 | 0 | 30 | 12 | 0 | - |
| JTN-L18-s1, +ACC+GAP | 280 | 0.7036 | 0.7022 | 68 | 6 | 2 | 18 | 42 | 21 / 13 / 8 | 0 | 28 | 11 | 0 | 288829f8 |
| B STANDFMX-L18-s0, +ACC+GAP | 280 | 0.6973 | 0.6957 | 71 | 9 | 1 | 18 | 43 | 22 / 13 / 8 | 0 | 21 | 6 | 2 | 0caa8f1a |
| B STANDFMX-L18-s1, +ACC+GAP | 281 | 0.7263 | 0.7209 | 63 | 3 | 1 | 16 | 43 | 21 / 14 / 8 | 0 | 22 | 6 | 2 | - |
| B2 STANDFMXH-L18-s0, +ACC+GAP | 281 | 0.7040 | 0.7015 | 66 | 8 | 1 | 16 | 41 | 22 / 11 / 8 | 0 | 30 | 9 | 1 | - |
| B2 STANDFMXH-L18-s1, +ACC+GAP | 279 | 0.7155 | 0.7155 | 64 | 8 | 0 | 16 | 40 | 20 / 13 / 7 | 0 | 28 | 8 | 2 | 0cc7fa67, 229791a5 |

**Mean (common):**: the 272 scenes every row above scored without a driver RPC error.

## Paired steps (scenes both reads scored cleanly)

| step | n | A mean | B mean | A - B, 95 % CI | zeros A - B |
|---|--:|--:|--:|---|--:|
| RB0-F-s0, native - P0, native | 280 | 0.3762 | 0.2419 | +0.1343 [+0.0789, +0.1893] | -26 |
| NTV0-L18-s0, native - RB0-F-s0, native | 281 | 0.4412 | 0.3757 | +0.0654 [+0.0356, +0.0965] | -28 |
| RB0-F-s0, VC+LEAD - RB0-F-s0, native | 280 | 0.5564 | 0.3759 | +0.1805 [+0.1294, +0.2304] | -54 |
| RB0-F-s0, VC+LEAD - P0, VC+LEAD | 280 | 0.5564 | 0.2496 | +0.3068 [+0.2504, +0.3633] | -90 |
| NTV0-L18-s0, VC+LEAD (stored N0) - RB0-F-s0, VC+LEAD | 280 | 0.6116 | 0.5564 | +0.0552 [+0.0263, +0.0863] | -22 |
| NTV0-L18-s0, VC+LEAD (stored N0) - P0, VC+LEAD | 281 | 0.6114 | 0.2487 | +0.3627 [+0.3049, +0.4204] | -113 |
| NTV0-L18-s0, +ACC+GAP (stored) - P0, +ACC+GAP | 280 | 0.7443 | 0.3523 | +0.3919 [+0.3363, +0.4490] | -104 |
| NTV0-L18-s0, native - P0, native | 280 | 0.4418 | 0.2419 | +0.1999 [+0.1459, +0.2537] | -54 |
| RSG-L18-s0, VC+LEAD - RB0-F-s0, VC+LEAD | 279 | 0.5711 | 0.5550 | +0.0161 [-0.0056, +0.0390] | -6 |
| P0, VC+LEAD - P0, native | 280 | 0.2496 | 0.2419 | +0.0077 [-0.0299, +0.0440] | +10 |
| P0, +ACC+GAP - P0, VC+LEAD | 281 | 0.3511 | 0.2487 | +0.1024 [+0.0534, +0.1530] | -8 |
| NTV0-L18-s0, VC+LEAD (stored N0) - NTV0-L18-s0, native | 281 | 0.6114 | 0.4412 | +0.1703 [+0.1263, +0.2165] | -48 |
| NTV0-L18-s0, +ACC+GAP (stored) - NTV0-L18-s0, VC+LEAD (stored N0) | 280 | 0.7443 | 0.6100 | +0.1342 [+0.0940, +0.1742] | +0 |
| RSG-L18-s0, +ACC - RSG-L18-s0, VC+LEAD | 279 | 0.6785 | 0.5733 | +0.1051 [+0.0665, +0.1443] | +0 |
| RSG-L18-s0, +ACC+GAP - RSG-L18-s0, +ACC | 280 | 0.6934 | 0.6760 | +0.0173 [-0.0082, +0.0436] | +0 |

## Read roots and serve flags

| read | root | SERVE | WIZ | ALPASIM_FAST | code |
|---|---|---|---|---|---|
| P0, native | `$DATA_DIR/runs/defect_fix/lineage/read/P0-NAT` | `(none)` | `controller.gains.rel_front_steering_angle_weight=10` | 1 | e1079613 |
| P0, VC+LEAD | `$DATA_DIR/runs/defect_fix/lineage/read/P0-VL` | `JEV_VCONT=1.0 JEV_LEAD=1` | `controller.gains.rel_front_steering_angle_weight=10` | 1 | 80cd4ddd |
| P0, +ACC+GAP | `$DATA_DIR/runs/defect_fix/lineage/read/P0-AG` | `JEV_VCONT=1.0 JEV_LEAD=1 JEV_ACC=1 JEV_GAP=0.7,2.7` | `controller.gains.rel_front_steering_angle_weight=10` | 1 | e1079613 |
| RB0-F-s0, native | `$DATA_DIR/runs/defect_fix/lineage/read/RB0-NAT` | `(none)` | `controller.gains.rel_front_steering_angle_weight=10` | 1 | 82ccd478 |
| RB0-F-s0, VC+LEAD | `$DATA_DIR/runs/defect_fix/lineage/read/RB0-VL` | `JEV_VCONT=1.0 JEV_LEAD=1` | `controller.gains.rel_front_steering_angle_weight=10` | 1 | e1079613 |
| NTV0-L18-s0, native | `$DATA_DIR/runs/defect_fix/lineage/read/NTV0-NAT` | `(none)` | `controller.gains.rel_front_steering_angle_weight=10` | 1 | e1079613 |
| NTV0-L18-s0, VC+LEAD (stored N0) | `$DATA_DIR/runs/route_follow/pai_input_loop/N0` | `JEV_VCONT=1.0 JEV_LEAD=1` | `controller.gains.rel_front_steering_angle_weight=10` | 1 | 96583428 |
| NTV0-L18-s0, +ACC+GAP (stored) | `$DATA_DIR/runs/defect_fix/pai_accgap/NTV0-L18-s0` | `JEV_VCONT=1.0 JEV_LEAD=1 JEV_ACC=1 JEV_GAP=0.7,2.7` | `controller.gains.rel_front_steering_angle_weight=10` | 1 | 2e2d0e46 |
| NTV0-L18-s1, +ACC+GAP | `$DATA_DIR/runs/defect_fix/pai_accgap/NTV0-L18-s1` | `JEV_VCONT=1.0 JEV_LEAD=1 JEV_ACC=1 JEV_GAP=0.7,2.7` | `controller.gains.rel_front_steering_angle_weight=10` | 1 | 2f0f5be9 |
| RSG-L18-s0, VC+LEAD | `$DATA_DIR/runs/route_follow/pai_g10/g10` | `JEV_VCONT=1.0 JEV_LEAD=1` | `controller.gains.rel_front_steering_angle_weight=10` | unset (pre-switch: official set) | c37e7150 |
| RSG-L18-s0, +ACC | `$DATA_DIR/runs/route_follow/pai_g10/acc` | `JEV_VCONT=1.0 JEV_LEAD=1 JEV_ACC=1` | `controller.gains.rel_front_steering_angle_weight=10` | unset (pre-switch: official set) | 8d89dbf7 |
| RSG-L18-s0, +ACC+GAP | `$DATA_DIR/runs/route_follow/acc_gap/read/s2` | `JEV_VCONT=1.0 JEV_LEAD=1 JEV_ACC=1 JEV_GAP=0.7,2.7` | `controller.gains.rel_front_steering_angle_weight=10` | 1 | 94711ca9 |
| JTN-L18-s0, VC+LEAD | `$DATA_DIR/runs/defect_fix/pai_nohold/JTN-L18-s0` | `JEV_VCONT=1.0 JEV_LEAD=1` | `controller.gains.rel_front_steering_angle_weight=10` | 1 | 5ecb907d |
| JTN-L18-s0, +ACC+GAP | `$DATA_DIR/runs/defect_fix/pai_accgap/JTN-L18-s0` | `JEV_VCONT=1.0 JEV_LEAD=1 JEV_ACC=1 JEV_GAP=0.7,2.7` | `controller.gains.rel_front_steering_angle_weight=10` | 1 | d78d5f2b |
| JTN-L18-s1, +ACC+GAP | `$DATA_DIR/runs/defect_fix/pai_accgap/JTN-L18-s1` | `JEV_VCONT=1.0 JEV_LEAD=1 JEV_ACC=1 JEV_GAP=0.7,2.7` | `controller.gains.rel_front_steering_angle_weight=10` | 1 | 5490eb4b |
| B STANDFMX-L18-s0, +ACC+GAP | `$DATA_DIR/runs/defect_fix/stand_b/pai281/STANDFMX-L18-s0` | `JEV_VCONT=1.0 JEV_LEAD=1 JEV_ACC=1 JEV_GAP=0.7,2.7` | `controller.gains.rel_front_steering_angle_weight=10` | 1 | 5d00f4d0 |
| B STANDFMX-L18-s1, +ACC+GAP | `$DATA_DIR/runs/defect_fix/stand_b/pai281/STANDFMX-L18-s1` | `JEV_VCONT=1.0 JEV_LEAD=1 JEV_ACC=1 JEV_GAP=0.7,2.7` | `controller.gains.rel_front_steering_angle_weight=10` | 1 | 7331a09c |
| B2 STANDFMXH-L18-s0, +ACC+GAP | `$DATA_DIR/runs/defect_fix/stand_b/pai281/STANDFMXH-L18-s0` | `JEV_VCONT=1.0 JEV_LEAD=1 JEV_ACC=1 JEV_GAP=0.7,2.7` | `controller.gains.rel_front_steering_angle_weight=10` | 1 | 79fe0e3f |
| B2 STANDFMXH-L18-s1, +ACC+GAP | `$DATA_DIR/runs/defect_fix/stand_b/pai281/STANDFMXH-L18-s1` | `JEV_VCONT=1.0 JEV_LEAD=1 JEV_ACC=1 JEV_GAP=0.7,2.7` | `controller.gains.rel_front_steering_angle_weight=10` | 1 | 7331a09c |

## Commands

```bash
# box, repo root
bash experiments/defect_fix/scripts/lineage.sh first    # staged first arm (P0 VC+LEAD); its first chunk checked before the rest
bash experiments/defect_fix/scripts/lineage.sh submit   # the other 4 arms, 18 extracts (CPU), summary --after all
bash experiments/defect_fix/scripts/lineage.sh add RB0-NAT   # RB0-F-s0 native, its extract, summary
```
