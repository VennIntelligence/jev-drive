# Pre-registration: turning school, demonstration rows with the human's timing (on1_PAIT, on1_CFTv), 2026-10-13

Context. User decisions 2026-10-13: (a) the turning-school demonstrations come from the PAI train160 human logs and carry the longitudinal
profile (arc length vs time from the log), not the shape only: the dominant PAI failure is entering the junction at 14-17 m/s without
slowing; (b) the PAI exam hard half is carved by a stable hash into hard_train (59, training-eligible) and hard_exam (72, never trained,
never used for selection): [pai_exam README](../../alpasim/results/pai_exam/README.md) "v2". rf_pai_follow (commit 1f62b46a,
[results/pai_follow/](../results/pai_follow/)): RSG turn completion 5 / 36 on the real route; 14 of 22 RSG route zeros had a drivable straight
exit at the turn; the ego slowed before the entry in 0 of 20 cases vs the human in 10 of 20; median entry speed ego 9.0 vs human 5.1 m/s.
RCFT-L18 (on1_CFT, commit cb596fb6) failed G-swap (0.31) and the public 1491 guard (+13 zeros vs RSG): the PAI-human arm is therefore the
primary, the CFTv addition secondary. Design note: [research/next-round/turning-school-rows.md](../../../research/next-round/turning-school-rows.md).
Scripts: [rf_pait.py](../scripts/rf_pait.py), [rf_pait_chain.sh](../scripts/rf_pait_chain.sh), [rf_cftv.py](../scripts/rf_cftv.py);
trainer flag `--on-weight` in [rm_train.py](../../route_mem/scripts/rm_train.py). Written before any training on these pools; amendments are
appended, never edited in place.

## Disclosure: what was seen before this file

- on1_PAIT80 (the 80 train160 scenes already rendered along the log, onpol_pai pass 2) built and gated; its 6-row BEV sheet looked at. The
  turn definition was switched from a first-draft heading rule to rf_pai_follow's strict rule and the window extended to the -40 m approach
  on the coordinator's input (before any training); the gate thresholds did not change.
- on1_CFTv: the first gates-only run failed G-v2 (0.928: the 2 m curvature smoothing under-read sharp centreline corners, peak a_lat p99
  3.48); the profile now tightens its lateral limit until the 0.5 s measure is within 3.3 m/s^2 (gate unchanged); second run passed.
- No training, no closed-loop read of any arm below.

## Pools

| pool | rows | content | gates |
|:--|--:|:--|:--|
| `on1_PAIT80` (Tokyo + box) | 1 361 | 12 train160 scenes, 13 strict turns (topo: straight_exit 5 turns / 386 rows, curve_single 6 / 747, no_straight_exit 1 / 158, lane_change 1 / 70); hard_train 0 (not yet rendered) | route / road / speed pass; R1 read pending (CPU) |
| `on1_PAIT` (chain) | 1 361 + the 104 rolled scenes | `rf_pait_chain.sh`: 38 priority (14 train160 log-turn scenes + 24 hard_train turn scenes) + 66 other train160 scenes, rolled along the log on Tokyo card 0 when `/data/runs/route_night/CARD0_FREE` appears, then rows, tokens, gid-ulp, route, pack, merge, rsync | same gates, per part (gates_PAITn.json) |
| `on1_CFTv` (box) | 12 314 | on1_CFT's 11 442 (src human) + 872 synthetic-profile rows (e2-only 942, 62 infeasible at 2.5 m/s^2, 8 actor overlap) | G-v1 1.000, G-v2 1.000, G-v3 equal to on1_CFT |
| `on1_PAITC` / `on1_PAITC80` | PAIT + CFTv | `rf_pait.py merge`; tag `pool` = a (PAIT) / b (CFTv) | contract check passes |

Row target on PAIT: the row's `imit` = the logged path at arc s_ego + d_k, d_k the log's own advance over t0 + 0.5 k s, so the imitation
term (pp_train dist on x, y, yaw; `--shape` only touches the hinges, not the imitation term) carries the human's speed. Gate G-speed: demo
chord arc at 4 s within 10 % of the log's on >= 99 % of rows (on1_PAIT80: 100 %, median ratio 1.000).

## Arms (box, GPU pool; gentle schedule as the CFT prereg)

`REC = --init RB0-F-s0 --steps 2500 --warmup 1 --lr-scale 0.1 --eval-every 500 --yr-mass 0.1 --label-bank --agent-lam 10 --agent-on-only --shape --ho-w 3 --rm-lr 3e-4`

| arm | rows | flags beyond REC | role |
|:--|:--|:--|:--|
| **RPAIT-L18-s{0,1}** | on1_PAIT | `--on on1_PAIT --on-k 12 --on-mode both --on-weight topo=straight_exit:2,topo=straight_road:2` | **primary** |
| RPAITC-L18-s{0,1} | on1_PAITC | `--on on1_PAITC --on-k 12 --on-mode both --on-weight pool=a:9,topo=straight_exit:2,topo=straight_road:2` (PAIT and CFTv drawn about equally; synthetic rows weight 1) | secondary |
| RNONEG-L18-s{0,1} | none | stored (same REC, no rows) | control |

`both` on PAIT rows = imitation + the agent hinge on the log-pass actor boxes (road raster absent on PAI: ok False). If the chain has not
finished when training is wanted, the arms run on `on1_PAIT80` / `on1_PAITC80` with the tag suffix `-80` (e.g. RPAIT80-L18-s0) and the
full pool is a later arm, never mixed into the same comparison.

## Reads

1. **PAI exam**: hard_exam 72 + control 75 (147 scenes), serving gains `rel_front_steering_angle_weight=20`, `JEV_VCONT=1.0 JEV_LEAD=1`,
   with the v6 spin-fix controller once it has passed its own guard (else the current controller); every comparator (RSG-L18-s0, RNONEG-L18)
   read under the identical serving: RSG-L18-s0's stored read counts only if its serving equals, else it is re-read on the same 147 scenes.
2. **held 460** (rm_chain submit-read, 18 decisions) and **public 1491** guard (allrows prereg method, 10 decisions).
3. **Entry speed**: rf_pai_follow `speeds` on the hard_exam turn scenes (strict rule; s48_turn_scenes.txt for the turn-completion list):
   ego speed at route station entry - 40 / - 20 / 0 m vs the human's at the same stations.

## Lines (seed means; an arm failing one is not called a candidate)

- L1 (primary) hard_exam zeros fewer than RSG-L18-s0's on the same 72 scenes and serving; paired scene-level bootstrap CI reported.
- L2 turn completion (rf_pai_follow, strict real-route rule, hard_exam turn scenes) higher than RSG-L18-s0's (point; counts reported).
- L3 held 460 and public 1491 zeros no worse than RSG-L18-s0's beyond seed noise (RSG-L18 s0 / s1 spread; public: not more than RSG + 5).
- L4 entry speed at entry - 20 m and at entry: median ego speed lower than RSG-L18-s0's, reported against the human's (5.1 m/s at entry in
  rf_pai_follow; 4.6 m/s on on1_PAIT80's turns).
- Control 75 zeros: reported, no line (it holds no turn scene).
- Every arm against RNONEG-L18 (same schedule, no rows) to separate the rows from the continuation.
