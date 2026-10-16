# WAIT-TRAIN 补记 D: exploratory reads after the pilot STOP (2026-10-17, 02:10-03:45 CST)

These reads are exploratory. The registered pilot verdict stands: STOP, both doses fail ([pilot_gate.md](pilot_gate.md)), on gate 4
(dev ADE <= C + 0.01) only. Every read below uses the 1 000-step pilot checkpoints (seed 0). The control is WTCP-L18-s0 (1 000 steps, RSG-L18
recipe), not the 2 500-step WTC. The lines were stated in the prereg's 补记 D (b154d102) before any of these outputs existed. No formal was
started, and no registered verdict file was touched (pilot_pick.txt = STOP, no WD_PASS / FALLBACK).

Code: `wt.sh explore` / `explore-pub`, `wt.py devall` / `xheld` / `xsum` (6d843ef4), and `scripts/wt_devdecomp.py`. Box outputs are in
`runs/wait_cap/train/explore/`. Files here: [explore_d/dev.md](explore_d/dev.md) / `dev.json` (decomposition),
[explore_d/cl.md](explore_d/cl.md) / `cl.json` / `held.json` (closed loop).

## Verdict

**The dev-ADE rise is a real cost, not a proxy artefact. Closed loop shows the cost, and it shows no waiting.**

- **Open loop.** J learned a broad longitudinal shortfall. Its plans are shorter than the human's on ordinary moving rows. It does not wait
  wrongly on those rows.
- **Closed loop.** That shortfall becomes about 3x larger when the model drives on its own history:
  - held distance is -11 % at J2 and -21 % at J5;
  - zeros are +12 / +17;
  - offroad doubles.
- **Waiting.** It does not appear anywhere in closed loop.

## Read 1: dev-ADE decomposition (the gate's own 1 789 dev rows, 32 logs)

`devall` reproduces every arm's DONE dev_ade exactly (J2 0.60795, J5 0.76235, CP 0.52904). P0 is shipped Cinque's stored teacher plan
on the same rows (ADE 1.576).

| | J2 vs CP | J5 vs CP |
|:--|--:|--:|
| dADE [log-clustered CI] | +0.079 [0.058, 0.104] | +0.233 [0.189, 0.285] |
| along-track share of the rise | **0.95** | **0.96** |
| 2 s arc bias, plan - human (J / CP) | -0.07 / -0.02 m | -0.23 / -0.02 m |
| 4 s arc bias (J / CP / P0) | **-0.49 / -0.03 / -0.36 m** | -1.04 / -0.03 / -0.36 m |
| median 4 s arc ratio, plan / human (CP 0.995, P0 0.974) | 0.972 | 0.944 |
| rows worse than CP | 0.61 | 0.70 |
| top 5 % rows (89, 21 logs): share of the rise | 0.54 | 0.34 |
| plan waits (4 s arc < 1 m) where the human goes >= 3 m | 2 rows (CP 0) | 2 rows |

Share of the rise by context and by the human's behaviour (J2; J5 is within ±0.05 of these):

- **Context:** free road 0.64; lead near 0.23; pedestrian 0.12; low TTC 0.00.
- **Behaviour:** cruising 0.57; starting 0.29; braking 0.15. No dev row has a human who stays stopped.
- **v0:**
  - 0-0.5 m/s: 0.13 (4 s bias -0.80 vs +0.06);
  - 0.5-3 m/s: 0.29;
  - 3-8 m/s: 0.49 (4 s bias -0.41 vs +0.04);
  - >= 8 m/s: 0.09.
- **Top rows:** slow cruising or launching at about 3 m/s on free road. The human covers 11-14 m in 4 s; J2 plans 5-8 m and CP 9-12 m.

**Mechanism.** The wait rows taught a global "go slower at low speed" prior, not a conditional wait:

- J's plans fall short of the human along the path. This happens on ordinary moving rows, mostly on free road, and most strongly when
  launching or cruising below 8 m/s.
- It is not false waits (2 rows) and not ambiguous wait-like labels.

Against the lines in 补记 D:

- line 1 (slow / brake: >= 60 % along-track and a negative 4 s bias): **met**;
- line 2 (proxy artefact: >= 50 % on starting rows or wait-like context): not met, 0.49;
- line 2 (real cost: >= 50 % on free-road cruising rows): not met, 0.41. The rise is diffuse rather than in either bucket.
- line 3 (tail): at J2 the top 5 % carry 0.54. These rows sit in 21 logs and are the same low-speed free-road kind, not outliers.

What is true about the proxy:

- This dev set cannot credit waiting. Only 14 of the 1 789 rows have a human 4 s arc < 1 m (navtrain is filtered, decision 325).
- The tolerance of 0.01 m is 1 % of the P0 -> CP gap (1.05 m).
- J2's speed bias is about the shipped model's own (median arc ratio 0.972 vs P0 0.974).

So the gate is strict, but the closed loop below shows it was pointing at something real.

## Read 2: closed loop (one seed, 1 000 steps; control WTCP-L18-s0)

| tag | held 460 @18: zeros | mean | paired vs CP [session CI] | progress_rel | dist travelled (m) | offroad | at-fault | held turns /24 | static scene @10 | PAI stop17 zeros |
|:--|--:|--:|:--|--:|--:|--:|--:|--:|--:|--:|
| WTCP-L18-s0 | 41 | 0.852 | - | 0.783 | 45.9 | 14 | 6 | 15 | 0 (drives 2.4 m into the lead) | 16 / 17 |
| WTJ2P-L18-s0 | **53** | 0.739 | **-0.112 [-0.146, -0.072]** | 0.704 | 40.9 (median ratio 0.94) | **28** | 7 | 12 | 0 (2.45 m) | 16 / 17 |
| WTJ5P-L18-s0 | **58** | 0.624 | **-0.228 [-0.264, -0.195]** | 0.626 | 36.4 (median ratio 0.83) | **39** | 6 | 13 | 0 (2.3 m) | 17 / 17 |
| RSG-L18-s0 archive (PAI config) | | | | | | | | | | 16 / 17 |

Context: the 2 500-step references read 34 held zeros (WTC-L18-s0, WTC-L18-s1, RSG-L18-s0), so the 1 000-step CP is already 7 zeros worse.
The like-for-like comparison is J vs CP.

Zero flips vs CP:

- **J2:** 21 J-only zeros (15 offroad, 3 at-fault, 3 corridor) against 9 CP-only zeros.
- **J5:** 34 J-only zeros (26 offroad) against 17 CP-only zeros.

Static scene and stop17:

- On the static scene, the three tags drive the same 2.3-2.45 m into a lead 1.6 m ahead, whatever their open-loop wait share (0.007 vs
  0.86 / 0.91). That suggests the motion in that scene is not set by the plan, for example initial speed or controller lag. This is a
  hypothesis, not checked.
- PAI stop17 is identical per scene for CP, J2 and the archive.

Lines (补记 D):

- (a) held, J <= CP + 3 zeros and turns J >= CP - 2: J2 fails both (+12; 12 vs 15). J5 fails zeros (+17) and meets turns (13 vs 15).
- (b) waiting visible in closed loop (static scene > 0, or stop17 zeros < CP): **not met** for either J.
- (c) public 1491 @10: did not run, because no J met (a).

**Reading.** On its own history the open-loop shortfall compounds:

- In open loop the 4 s arc is 2.8 % short at J2. In closed loop J2 travels 11 % less, and J5 21 % less (open loop 5.6 %). This matches
  decision 315 (LON-DIAG): the open-loop longitudinal channel follows the trend of the car's own motion, with loop gain 0.936, so a
  slower history yields a slower plan.
- Offroad doubles. The cause is not diagnosed. Candidates are slower turns that leave the drivable area, and an effect of the
  flat-history S / Y rows on lateral tracking. The lateral open-loop gate (median |dy| at 2 s 0.011 m) did not see it.

## Options for the morning (evidence only; the choice is the user's)

- **(i) Re-register the ADE gate and run the formal at j2: not supported.**
  - The gate is strict and blind to waiting: the dev set has 14 wait rows out of 1 789.
  - But the thing it caught is real. J2 is -0.112 on held, has +12 zeros and twice the offroad, and shows no closed-loop waiting. A
    formal at j2 would probably be a cleaner version of the same trade.
- **(ii) A lower dose: weak.**
  - The cost is monotone in dose (J5 is worse than J2 on every read).
  - The closed-loop benefit is zero even at j5. A lower dose can at best reduce a cost; it cannot add a gain that no dose has shown.
- **(iii) Rebuild the base with the conditional pathway kept: best supported.**
  - Added on top of the RSG-L18 base, wait rows do not restore P0's conditional wait. They install an unconditional low-speed slowdown
    (J's 4 s bias is about P0's, but without P0's discrimination in closed loop).
  - Decision 326 places the loss of waiting at P2-F. Keeping P0's longitudinal behaviour on stationary or launch frames while building
    the base attacks the cause, not the symptom.
- **(iv) Anti-copycat / inverse dynamics: the right companion, not a replacement.**
  - The 3x amplification from open to closed loop is the own-history trend following of decision 315.
  - Any longitudinal training (WAIT or otherwise) will compound through it until the ego-history channel is decorrelated.

## Cost

- 28 pool jobs (devall, 18 held reads, 3 static-scene reads, 3 PAI stop17 chunks, rf_follow, two CPU gates): 4.3 job-hours, about 1.1
  card-hours at 4 reads per card.
- Wall clock 02:16-02:42.
