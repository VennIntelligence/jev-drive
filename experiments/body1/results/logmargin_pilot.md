# BODY1: log-consistent margins for the body lesson S (`P2H10SM`): stopped at the pilot's drift line

2026-10-11. One pilot decided by the user, registered in [plans/2026-10-11-logmargin-pilot-prereg.md](../plans/2026-10-11-logmargin-pilot-prereg.md)
(pushed c023694c with the code, before any job). The fifth trained variant of the loss arm 4.3. Tables: [logmargin/](logmargin/) (the generated
report [logmargin/report.md](logmargin/report.md), G0 [logmargin/g0-P2H10SM-P-s0/g0.md](logmargin/g0-P2H10SM-P-s0/g0.md)); per-row clearances and
the plan dumps stay on the box (`$DATA_DIR/runs/body1/logmargin/g0-P2H10SM-P-s0/c.npz`, `runs/body1/prog/ol_lm_*`).

**Result.** By the registered lines the pilot is **not a candidate: line (i) is missed**. On navtest turn tokens over 45 deg the own plan is
more than 2 m outside the logged path in 142 tokens against 110 for the base pilot (limit 110 + 12); the S pilot, read in the same call, has 140,
and 137 of the 142 tokens are S's. The mean outward shift at 4 s is S's (+0.008 m [-0.000, +0.016] against the S pilot, +0.124 m [+0.082, +0.165]
against the base pilot). Lines (ii) and (iii) are met: the lesson is kept whole (hold-log own-plan agent contact -37.9 %, boundary -33.8 %, the
S pilot's -37.1 % / -33.8 %) at no cost (dev ADE 0.5902 m, 4 s arc ratio 0.9992). By the pre-registered reading: **the fixed margin firing on the
log's own path is not the drift mechanism**, at pilot scale. G0.c says why there was little to remove: the logged path violates S's road margin
on 13.3 % of hinge-only rows (10.2 % of those on turns over 45 deg, fewer than elsewhere) and S's agent margin on 0.04 % (none on turns).

## 1. What was run

`bd4_train.py --logmargin` on the S pilot's command (shards s2 + s3, 3 000 steps, batch 128, seed 0, warp frames, `--ho ot1:4,yr1:4,bd4:5
--agent-lam 10 --ho-w 3 --ho-excl navsim/body1-val-logs --shape`), tag `P2H10SM-P-s0`. Per row and new hinge, m = min(m0, c - 0.05 m), c = the
clearance of the row's logged path (logged t0 pose + the 8 logged poses, the plan's 0.1 s densification, the hinge's own geometry; on hinge-only
rows from the logged t0 pose, i.e. the source log row's world path). The agent cap applies to both of its margins (0.3 m in the corridor, 0 m
outside). P2H10's own on-log hinge and everything else unchanged. References: the archived `P2H10S-P-s0` (S pilot) and `P2H10-P-s0` (base pilot).
Everything ran through the pool (`scripts/logmargin_chain.sh`): identity run, pilot (G0.b / G0.c at its setup), four GPU readers, three CPU steps.

## 2. G0

| Check | Result | Line | Verdict |
|:--|:--|:--|:-:|
| G0.a: flag off, 60 steps of the S recipe on shard s2, against the stored `B43A7-IDENT-S1` ([ident_off.json](logmargin/ident_off.json)) | 42 logged scalars and every trained weight bit for bit (max difference 0) | bit for bit | met |
| G0.b (i): the logged path gives zero hinge with the new margins, agent | 39 739 of 39 739 rows (0 exceptions) | >= 99.5 % | met |
| G0.b (i), road | 24 352 of 24 352 rows (0 exceptions) | >= 99.5 % | met |
| G0.b (ii): hinge-only row's c = its source log row's c, agent | 24 352 of 24 352 (largest difference 1.5e-5 m) | >= 99.5 % within 1e-3 m | met |
| G0.b (ii), road | 24 352 of 24 352 (largest difference 7.6e-6 m) | >= 99.5 % within 1e-3 m | met |

So the identity also ties today's checkout to the code path that trained the archived pilots (`B43A7-IDENT-S0` = S code path = `B43A7-IDENT-S1`
= `SDROP-IDENT-S` = `LM-IDENT-OFF`), and the expectation that a hinge-only row's logged path is its source row's world path holds to the float32
rounding of the caches.

**G0.c (diagnostic; rows a pilot batch can carry the hinge on: agent 15 387 imitation rows of the train logs + 24 352 hinge-only rows, road
the 24 352 hinge-only rows).**

| Hinge | rows | n | logged path violates S's margin | c < 0.3 m | margin lowered (c < 0.35 m) | margin < 0 | lowered margin q05 / q25 / q50 / q75 (m) |
|:--|:--|--:|--:|--:|--:|--:|:--|
| agent | all | 39 739 | 0.04 % | 0.33 % | 0.39 % | 0.05 % | -0.09 / +0.07 / +0.15 / +0.24 |
| agent | imitation rows | 15 387 | 0.05 % | 0.42 % | 0.49 % | 0.06 % | -0.08 / +0.07 / +0.15 / +0.23 |
| agent | hinge-only | 24 352 | 0.03 % | 0.28 % | 0.33 % | 0.04 % | -0.09 / +0.08 / +0.14 / +0.24 |
| agent | > 45 deg (5 262; 3 598 hinge-only) | 5 262 | 0.00 % | 0.08 % | 0.11 % | 0.00 % | +0.24 / +0.24 / +0.24 / +0.26 |
| road | hinge-only (all) | 24 352 | 13.31 % | 13.31 % | 16.85 % | 3.61 % | -0.16 / +0.02 / +0.12 / +0.23 |
| road | `ot1` / `yr1` / `bd4` | 9 368 / 9 368 / 5 616 | 13.66 / 13.76 / 11.98 % | same | 17.06 / 17.15 / 16.01 % | 3.80 / 3.80 / 2.96 % | |
| road | > 45 deg | 3 598 | 10.23 % | 10.23 % | 13.23 % | 3.59 % | -0.26 / -0.01 / +0.11 / +0.24 |

What to read: the hypothesis needs the log to sit inside S's margin where the drift happens. For the agent hinge it essentially never does (the
counted objects ahead are almost never within 0.3 m of the logged path's extended box), so log-consistency leaves that hinge as it was. For the
road hinge it does on a sixth of the hinge-only rows, but less often on turns over 45 deg (10.2 %) than elsewhere (13.3 %).

Reported with G0 (no line): the logged future fed as the plan of the training loss, i.e. densified from the row's own t0 (the perturbed pose on
hinge-only rows), has a non-zero road hinge on 18.0 % of hinge-only rows with S's margins and 7.4 % with the new ones (agent 0.33 % / 0.30 %):
the remaining 7.4 % is the perturbation, the recovery part of the lesson, which the definition keeps. The other reading of the definition
(clearance of that path instead of the logged one) would have lowered the road margin on a further 8.7 % of hinge-only rows (agent 0.8 %); it was
not run. (The agent "c - c_op" quantiles in `g0.md` mix in rows whose logged path counts no object, c = +inf capped at 10 m; only the 0.8 % is
meaningful.)

## 3. The pilot against the three lines (one read; hold logs: shards s2 + s3, 5 062 states, 118 logs)

| Line | `P2H10SM-P-s0` | S pilot `P2H10S-P-s0` | base pilot `P2H10-P-s0` | limit | verdict |
|:--|:--|:--|:--|:--|:-:|
| (i) W2, navtest turn tokens > 45 deg (1 517), warp frames, one reader call | **142** | 140 (archived bench-plan read 141) | 110 (archived 110) | <= 110 + 12 = 122 | **NOT met** |
| (ii) own-plan agent-contact rate, hold pooled | 0.0162, fall 37.9 % (difference -0.0099 [-0.0135, -0.0067]) | 0.0164, fall 37.1 % | 0.0261 | fall >= 18.56 % (20.1 % with the task's -40.2 %) | met |
| (ii) own-plan boundary rate (NAVSIM raster), hold pooled | 0.0182, fall 33.8 % (difference -0.0093 [-0.0135, -0.0053]) | 0.0182, fall 33.8 % | 0.0275 | fall >= 16.91 % | met |
| (iii) dev ADE at step 3 000 (m) | 0.5902 | 0.5882 | 0.5913 | <= 0.5932 | met |
| (iii) 4 s arc ratio against the base pilot, hold pooled | 0.9992 [0.9981, 1.0004] | 0.9989 [0.9978, 1.0000] | 1 | in [0.995, 1.005] | met |

Line (ii)'s reference is the S pilot's own fall (results/loss/g3_a6_hold_S.json, -37.1 % / -33.8 %); the task text's -40.2 % is the R pilot's
(route_pilot.md), and the pilot clears either. Line (i)'s tolerance 12 is the base's W2 seed spread on warp frames (decision 234: 84 / 79 / 87 / 91).

**Reported reads (no lines)**

| Read | `P2H10SM-P-s0` | S pilot | base pilot |
|:--|:--|:--|:--|
| navtest > 45 deg: W2 / leaves the 4 m tube / mean lateral at 4 s (+ = outside) | 142 / 26 / +0.432 m | 140 / 27 / +0.425 m | 110 / 25 / +0.309 m |
| navtest > 45 deg right (599): W2 / tube exits / lateral | 54 / 7 / +0.492 m | 53 / 7 / +0.475 m | 39 / 5 / +0.231 m |
| navtest > 45 deg left (918): W2 / tube exits / lateral | 88 / 19 / +0.394 m | 87 / 20 / +0.392 m | 71 / 20 / +0.360 m |
| hold on-log turn rows (207): W2 | 15 | 15 | 10 |
| hold turn rows, four families (686): W2 / tube exits / lateral | 80 / 28 / +0.269 m | 77 / 25 / +0.252 m | 80 / 27 / +0.088 m |
| hold own-plan agent / boundary rate, difference against the S pilot | -0.0002 [-0.0013, +0.0009] / +0.0000 [-0.0011, +0.0012] | | |
| validation part (4 019 states): agent / boundary rate, fall against the base pilot | 0.0251, 29.9 % / 0.0254, 36.3 % | | 0.0358 / 0.0398 |
| hold families `log` / `ot1` / `yr1` / `bd4`: agent fall | 12.5 / 33.3 / 38.8 / 41.7 % | 25 / 27 / 35 / 43 % (shape_pilot.md) | |
| 4 s arc ratio, hold open / lead / `bd4` | 1.0008 / 0.9957 / 0.9976 | 1.0004 / 0.9956 / 0.9964 | 1 |
| 4 s arc ratio, navtest pooled / open / lead | 0.9994 / 1.0002 / 0.9964 | 0.9995 / 1.0005 / 0.9964 | 1 |
| dev_drift_off | 0.048 | 0.048 | 0.042 |
| training, steps 2 701-3 000: hinge-only rows with a non-zero road hinge (`ot1` / `yr1` / `bd4`) | 0.194 (0.141 / 0.177 / 0.251) | 0.228 (0.191 / 0.211 / 0.271) | |
| training: road hinge on hinge-only rows (`road_ho`) / agent hinge positive share on them | 0.0013 / 0.045 | 0.0017 / 0.042 | |

**The pilot against the S pilot directly** (paired by log, same reader call): on navtest turn tokens the mean lateral at 4 s differs by +0.008 m
[-0.000, +0.016] (S against base: +0.116 m [+0.075, +0.156]); 137 of the pilot's 142 W2 tokens are among S's 140; the mean distance between the two
plans is 0.043 m on turn tokens, against 0.166 m between S and the base. On hold turn rows +0.017 m [-0.000, +0.040] (S against base +0.164 m),
every family pointing the same way.

## 4. What it means

The change did reach the loss: the road hinge's mass on hinge-only rows fell by a quarter (0.0017 -> 0.0013) and its positive share from 0.228
to 0.194. It did not move the plan: whatever S's outward shift is made of, removing the road hinge's force on rows where the log itself is within
the margin removes none of it (+0.008 m, if anything further out). Two readings of where the force comes from instead, neither tested here:
(a) the hinge acting where the plan, not the log, comes within the margin, i.e. the plan cutting the inside more than the log does, with nothing
on hinge-only rows naming the route (decision 234's reading; the margin stays 0.3 m on the 87 % of rows where the log is clear of it); (b) the
agent hinge on hinge-only rows, which on its own produces about half of S's shift at full scale (decision 244, noC: +0.059 of +0.123 m) and
which log-consistency cannot touch, since the log almost never violates it. Decision 234 found that a route tube does not separate the inside-edge lesson from the route; this pilot
finds that making the margins log-consistent does not either.

## 5. Costs, deviations, limits

Costs: 0.245 card-hours of GPU (each GPU job counted as a whole card; the budget was 1.5): identity 0.7 min, pilot 7.4 min (7.28 it/s, 15.4 GB),
readers 1.4 to 1.7 min each on four cards; route tables and report on CPU. Wall 12.5 min from the first job (17:40:50 box time) to the report
(17:53:15). Disk: 1.6 MB on the box.
Deviations from the task text, all fixed in the pre-registration before any job: (1) line (ii) uses the S pilot's own fall (-37.1 %), not -40.2 %
(the R pilot's); (2) the hinge-only rows' logged path starts at the logged t0 pose (design.md 3.1; the task's equality check), not at the
perturbed pose; (3) G0.a compares against the stored S-recipe identity run on shard s2 rather than a fresh 60-step pair on s2 + s3, which also
ties today's checkout to the archived pilots' code path; (4) the three checkpoints of each read are forwarded by one reader call (S and base
reproduce their archived counts within one token).
Limits: one seed, pilot scale (2 of 12 shards, 3 000 steps); open loop only, and whether W2 predicts closed-loop corridor zeros has never been
tested; the hold logs have now been read eight times by this lane's variants (five at pilot scale); the 0.05 m slack and the per-row cap were not
varied; the other reading of the definition (8.7 % more hinge-only rows with a lowered road margin) was not run.
