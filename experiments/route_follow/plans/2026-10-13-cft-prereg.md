# Pre-registration: counterfactual TRAJECTORY rows `on1_CFT` (turning school, demonstration source 2), 2026-10-13

Context: [decision 286](../../../research/decisions/286.md): the counterfactual pool `on1_CF` with the route hinge taught the junction route
swap (G-swap 55 / 56 %) but over-followed (RB-F dev ADE +0.077 over RB0, held 87 zeros); the hinge version is not adopted. This pool keeps
on1_CF's rows (tokens, counterfactual memory, road raster, route.npz) and replaces the hinge with an imitation target shaped like a trajectory
(R-CF as first proposed in [route-follow.md](../../../research/next-round/route-follow.md) 3.3). Design note:
[research/next-round/cft-rows.md](../../../research/next-round/cft-rows.md). Script:
[rf_cftpool.py](../scripts/rf_cftpool.py). Written, committed and pushed before the full 12-shard build and before any training.
Amendments are appended, never edited in place.

## Disclosure: what was seen before this file

Shard s2of12 was built and gated as the smoke (`cf/cft_gates_CFTs2.json`, 6 figures looked at). The lateral merge length was chosen on it:
Lc = 2.5 s of travel clipped to [6, 25] m gave G-cft2 0.78 (lc 0.42); 3.5 s / [8, 40] 0.84; **4.5 s / [10, 60] 0.86 (chosen)**; 6 s / [12, 80]
0.89 (rejected: the lc "stay in lane" demo would still be drifting at 4 s). The exclusion rules e1 - e4 and the gate thresholds below were
fixed after the s2 read; the full build adds 11 unseen shards.

## The demonstration (per on1_CF row, in the token's rear-axle frame)

C = rf_cfpool geom's counterfactual centreline (junction j0 / j1: lane-graph centreline of the alternative exit from the ego's projection; lc:
the t0 lane run). S_k = the log's own arc length at t_k (polyline through the origin and tab `fut`). Path P(u) = C(s0 + u) + d(u) n: starts at
the token's pose with its heading (cubic Hermite lateral offset from (d0, tan heading error) to (0, 0) over Lc = clip(4.5 s x S_8 / 4 s, 10, 60) m).
`fut` = P at arc S_1 .. S_8, yaw = tangent. Everything else in the pool equals on1_CF's row (names `on:cft:<token>:<j0|j1|lc>`).

## Exclusions (rows of on1_CF's selection with a finite demo = candidates)

- e1 continuity: max |demo - log| at 0.5 / 1.0 s > 0.5 m (the counterfactual cannot branch smoothly at this state: the human is already
  committed to the driven exit / mid lane change).
- e2 kinematics: demo peak |v x yaw rate| over the 8 half-second steps > 4 m/s^2 (the human's speed on a sharper counterfactual path).
- e3 actors: the demo's Pacifica box overlaps a valid actor box of `agent_labels/navtrain_all-k32` at the same time (t + 0.5 .. 4 s).
- e4 road: a demo sample inside the raster extent with sdf_road <= 0.

## Gates (full build; on the candidates unless said)

- **G-cft1** demo samples inside the raster with sdf > 0 >= 99 %. (s2: 99.99 %.)
- **G-cft2** per-row rule = e1, so every pool row satisfies it; gate: >= 80 % of candidates satisfy it before exclusion, and lc >= 55 %
  (the exclusion must not gut the pool). (s2: 86.3 %, lc 67.3 %.) The coordinator's literal form (first 1 s within 0.5 m) holds on 100 %
  of pool rows by construction.
- **G-cft3** 6 BEV figures (3 junction, 3 lane change, pool rows, seed 0): `experiments/route_follow/figs/cft_bev_{0..5}.png`, for the coordinator.
- **G-cft4** demo chord arc at 4 s within 10 % of the log's (log arc < 1 m: within 0.1 m) on >= 99 % of candidates. (s2: 100 %.)
- Pool >= 8 000 rows. A failing gate stops the pool from being offered for training (reported, design revisited by amendment).

## Registered training plan (NOT run here; the coordinator decides)

`$DATA_DIR/envs/op-train/bin/python experiments/route_mem/scripts/rm_train.py train --tag RCFT-L18-s{k} --init RB0-F-s0 --steps 2500 --warmup 1
--lr-scale 0.1 --eval-every 500 --yr-mass 0.1 --label-bank --agent-lam 10 --agent-on-only --shape --ho-w 3 --rm-lr 3e-4 --on on1_CFT --on-k 12
--on-mode both --seed {k}` for k = 0, 1. No `--on-lon` (route.npz and its dz are never read). `both` = imitation of the demo + S's road
hinge on the token's P2H10 raster (agent ok False: no agent hinge on these rows). Variant `--on-mode imit` (imitation only) is the same pool;
the mode is the only switch.

Control = the same without rows: `RNONEG-L18-s{k}` (gentle schedule, no `--on`; s0 stored, s1 trained if missing with the identical command
minus the four `--on*` flags).

Lines (seed means; an arm failing one is not called a candidate):
1. **G-swap** (rf_gates.py on `on1_CFdev`, junction rows): 1.5 - 2.0 s lateral move / required (median) >= 0.5, as RB-F had (0.55 / 0.56).
2. **dev ADE** (memory on, final) <= RB0-F's (0.528 / 0.531) + 0.01 m; G-rely reported.
3. **public 1491 at 10 decisions** (guard method of the allrows prereg): zeros not more than RSG-L18's + 5.
4. **PAI exam hard 131** (Tokyo, same serving as RSG-L18-s0's stored read): corridor "straight at junction" zeros fewer than RSG-L18-s0's
   (point), with the hard-131 total zeros and mean score difference CI reported beside it.
