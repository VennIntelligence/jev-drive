# RTB1 cost diagnosis: stop / relaunch timing on the shared held zeros and the extra public rear contacts (prereg 补记 2, exploratory)

This is a CPU-only diagnosis of decision 337 under the class rule in prereg [2026-10-17-retain-anticopy-draft.md](../../plans/2026-10-17-retain-anticopy-draft.md)
§补记 2. It compares a = RTB1O-L18-s0 against RSG-L18-s0 and the human log; b / c / d are checked against their own references.
It reads stored runs only: the controller CSVs (ego speed every 0.1 s), the metrics.parquet onsets, the scene's rig trajectory (the human,
in the same frame and on the same clock), the scene map, and the rollout.asl where it survived. Code: `nx_atfault.py --relaunch`. Per-scene
data: [explore_f/relaunch.json](explore_f/relaunch.json); the full table for every arm: [explore_f/relaunch.md](explore_f/relaunch.md).

**Definitions** (fixed in code before any output). A stop is v < 0.3 m/s, with motion gaps under 0.5 s merged. The relaunch is the end of
a stop. Accel = (v(relaunch + 2 s) − v(relaunch)) / 2. A hard brake is a drop of ≥ 2.5 m/s within 1 s. "Human stopped there" means
the human has a stop within ±5 m of the ego's station along the human path. A scene zeroes on at-fault collision, offroad or corridor
(> 4 m off the human path; docs/alpasim.md); the zero event is the earliest onset among these. The public event is the first `collision_rear` onset.
Classes are checked in rule order and every match is kept. A rear contact while stopped goes to (i) if the human stopped there and to
(iii) if it did not; this is how "where the human did not stop" is read.

## The seven held 460 @18 zeros that every arm shares

The flips are recomputed from `rtb_cont_<a..d>.json` (J_only against the matched RSG seed, intersected over the four arms). They give
exactly 53d3, de4a, 6d2b, 9d1d, 9a67, ddc1 and 1f54. RSG-L18-s0 and s1 score > 0 on all seven. All four arms get the same class on every
scene, and their event times agree to within 1.5 s.

| scene | log | zero event, a (b / c / d) | class | stop t / dur / relaunch / accel: a · RSG · human | note |
|:--|:--|:--|:-:|:--|:--|
| 53d3 | 06.28 veh-38 | corridor 9.0 s (8.5–9.0) | iv | no stop · no stop · no stop | a levels at 3.4–3.6 m/s against RSG 4.2 and the human 4.4, and ends 4.1 m off the human path |
| de4a | 06.28 veh-26 | at-fault front 6.5 s (c, d: lateral) | v | no stop (min 1.5) · no stop · no stop | a peaks at 2.7 m/s at 3 s, then slows to 1.9 at contact while the human goes 5.7 → 6.9 (RSG 3.4). It hits a same-direction car front-left (x 1.6, y 2.4 m, dpsi −15°). Not a relaunch and not a rear contact |
| 6d2b | 09.09 veh-40 | at-fault front 6.0 s (d 7.0) | v | 0.1 / 1.2 / **1.3** / +1.40 · 0.1 / 1.6 / 1.7 / +1.36 · 0.0 / 1.8 / 1.8 | a relaunches *earliest* and runs hottest: 8.4 m/s at contact against RSG 7.0 and the human 5.9. It hits a same-direction car ahead-right (x 5.1, y −1.7) 4.7 s after the relaunch, just outside the 4 s window of (ii) |
| 9d1d | 09.09 veh-40 | offroad 1.5 s (all 1.5) | iv | no stop · no stop · stops at 8.8 | 6.3 m/s at the event, the same as RSG (6.2) and the human (6.4). The zero is purely lateral |
| 9a67 | 09.16 veh-42 | at-fault front 6.0 s (all 6.0) | iv | no stop (min 0.96) · no stop (min 1.0) · **1.5 / 2.0 / 3.5** | head-on (dpsi 147°) with the ego in the opposite lane. a does **not** stop where the human stops for 2 s; it accelerates to 5.6 m/s against RSG 3.6 and the human 3.0 |
| ddc1 | 10.06 veh-52 | offroad 5.0 s (6.5 / 4.0 / 6.0) | iv | no stop (min 0.45 at 4.6 s) · min 1.4 · min 1.7 | a crawls at 0.5 m/s where the human holds 2.0, and leaves the road while crawling |
| 1f54 | 10.06 veh-52 | offroad 8.5 s (8.5 / 8.0 / 7.5) | iv | no stop · no stop · no stop | a's speed stays close to RSG's (3.0 against 3.9 at the event). Lateral |

**Held classes: (iv) 5, (v) 2, (i) / (ii) / (iii) 0.** Only one of the seven scenes (6d2b) contains a stop at all, and there a relaunches
before RSG and the human, not after them. The guess in decision 337 ("the extra stops bring extra relaunches") does not hold for the shared held
zeros. Five are lateral (offroad 4, corridor 1, plus the 9a67 head-on in the opposite lane). The two longitudinal contacts are a slow
car that never stopped (de4a) and an early, hot launch (6d2b). Over the 9 s horizon, "never launches" (under 1 m driven while the log
drives over 5 m) occurs in 1 / 1 / 2 / 0 held scenes for a / b / c / d and 0 for RSG.

## Public nuPlan 1491 @10: the rear contacts a has and RSG-L18-s0 does not

Recomputed: a is struck from behind in 24 scenes and RSG in 18. **13 are a-only** (7 are RSG-only). The rollouts are 5 s. The rollout.asl
is pruned, so the struck actor is unread; every timing below comes from the controller CSV and the rig trajectory. The ego is in its own
lane at every contact.

| scene | log | a score (RSG) | rear contact | class | stop t / dur / relaunch / accel: a · RSG · human | v at contact a / RSG / human | note |
|:--|:--|--:|--:|:-:|:--|:--|:--|
| ed82 | 06.28 veh-38 | 0.00 (1.00) | 4.5 | **i** | 0.1 / 4.9+ / never · 0.1 / 1.5 / 1.6 / +1.50 · 0.0 / 2.0 / 2.0 | 0.0 / 4.9 / 5.8 | never leaves the start standstill; struck while stopped |
| 4211 | 08.30 veh-40 | 0.01 (0.57) | 5.0 | **i** | 0.1 / 4.9+ / never · 0.1 / 1.5 / 1.6 / +0.61 · 0.0 / 1.8 / 1.8 | 0.0 / 2.2 / 6.1 | same |
| 982b | 09.09 veh-48 | 0.00 (0.62) | 5.0 | **i** | 0.1 / 4.9+ / never · 0.1 / 1.8 / 1.9 / +0.83 · 0.0 / 1.9 / 1.9 | 0.0 / 2.5 / 5.1 | same |
| f351 | 09.16 veh-42 | 0.01 (1.00) | 4.5 | **i** | 0.1 / 4.9+ / never · 0.1 / 1.4 / 1.5 / +1.56 · 0.0 / 2.1 / 2.1 | 0.0 / 5.3 / 5.7 | same |
| f4ee | 09.29 veh-28 | 0.01 (1.00) | 4.5 | **i** | 0.1 / 4.9+ / never · 0.1 / 1.4 / 1.5 / +0.89 · 0.0 / 1.7 / 1.7 | 0.0 / 3.5 / 3.1 | same |
| eeba | 10.06 veh-52 | 0.02 (0.75) | 4.0 | **i** | 0.1 / 4.9+ / never · 0.1 / 1.4 / 1.5 / +0.84 · 0.0 / 1.3 / 1.3 | 0.0 / 2.4 / 3.6 | same |
| e29b | 06.03 veh-35 | 0.10 (1.00) | 4.0 | v | 0.1 / 2.8 / **2.9** / +1.13 · 0.1 / 1.4 / 1.5 / +1.68 · 0.0 / 1.8 / 1.8 | 1.2 / 4.7 / 4.7 | late relaunch (1.1 s after the human), struck 1.1 s later at 1.2 m/s. Not stopped at contact, so not (i) on the strict rule |
| c279 | 09.29 veh-28 | 0.13 (0.65) | 5.0 | v | 0.1 / 3.2 / **3.3** / – · 0.1 / 1.5 / 1.6 / +0.72 · 0.0 / 2.0 / 2.0 | 0.8 / 3.5 / 6.7 | same pattern: relaunches 1.3 s after the human and is struck at 0.8 m/s |
| 83fd | 05.25 veh-35 | 0.84 (1.00) | 5.0 | v | no stop · no stop · no stop | 1.1 / 3.5 / 5.6 | a slows 4.8 → 1.1 m/s over 2–5 s (largest 1 s drop 1.4, under the hard-brake bar) while the human holds 5.6 |
| 3296 | 10.06 veh-52 | 0.82 (1.00) | 4.5 | v | no stop · no stop · no stop | 1.1 / 3.8 / 3.4 | a slows 5.6 → 1.1 (1 s drop 1.8) where the human and RSG ease to 3.4–3.8 |
| e086 | 06.03 veh-35 | 0.96 (1.00) | 5.0 | v | no stop · no stop · no stop | 2.2 / 2.7 / 4.2 | a and RSG both slow more than the human; a slightly more |
| 9551 | 09.16 veh-45 | 0.84 (0.95) | 4.5 | v | no stop · no stop · no stop | 4.7 / 4.9 / 6.7 | a's trace is RSG's (dip to 4.3); contact chance |
| 9d42 | 09.29 veh-28 | 0.94 (0.00) | 5.0 | v | no stop · no stop · no stop | 5.0 / 2.5 / 6.2 | a is accelerating at contact; RSG zeroes here itself (at-fault lateral 4.5 s) |

**Public classes: (i) 6, (v) 7, (ii) / (iii) / (iv) 0.** Two of the (v) scenes (e29b, c279) are late relaunches struck at < 1.5 m/s,
which would be (i) if "struck while stopped" were read loosely (8 / 13). Another two (83fd, 3296) are soft slowdowns the human did not
make, which would be (iii) with a lower brake bar. No stop or hard brake by a happens where the human was moving.

**The public cost is a launch failure.** Across all of public 1491, a moves under 1 m in 5 s while the log moves over 5 m in **12 scenes; RSG
in 0**. In the 8 that were traced (the 6 (i), e29b, c279), the human starts stopped and goes at 1.3–2.1 s, and RSG
goes at 1.4–1.9 s; the other 4 were not traced. 7 of the 12 are rear-ended (the 6 (i)
scenes and e29b). The 12 carry **−9.44 of the −17.15 paired score points**: the public paired mean is −0.0115 with them and −0.0052
without them.

## Does (iii) dominate?

**No.** (iii) is 0 / 7 held and 0 / 13 public, so the registered reading "(iii) majority supports the V2 hypothesis" is not met. The
public set is dominated by (i): a stays at the start standstill after the human (and RSG) have gone. That is the launch delay the V2
hypothesis text describes (the W static-hold rows teach an unconditional hold; pilot v0 < 0.5 launch frames −0.44 m). It is not
the "stops where the human did not stop" mechanism that (iii) encodes. The held shared zeros are mostly lateral (iv) and are not
stop / relaunch events. The one held stop (6d2b) relaunches early, not late. This page reports the rule's counts and does not re-map them.

## Figure

[explore_f/traces.png](explore_f/traces.png) shows the speed against time for all 20 scenes: the 7 held, then the 13 public. Grey is the
human, orange RSG-L18-s0, blue a. The shaded bands are stop episodes in each tag's colour; the dashed line marks a's zero event or rear
contact (orange dashed marks RSG's own event). What to look at:
- The public (i) panels (ed82, 4211, 982b, f351, f4ee, eeba): blue stays flat at 0 for the whole 5 s. Grey leaves the standstill at
  1.3–2.1 s and orange follows within 0.5 s. The rear contact comes 2.5–3 s after the human left.
- e29b and c279: blue leaves 1.1–1.3 s after grey and climbs more slowly. It is struck while still under 1.5 m/s.
- 83fd and 3296: blue slows steadily toward 1 m/s while grey and orange hold speed. This is a soft brake the human never makes.
- Held 6d2b: blue leaves the start stop *first* and stays above orange and grey up to the contact. Held 9a67: grey stops for 2 s, while
  blue and orange never stop and blue accelerates into the opposite lane.

## Caveats

The first pass of the code checked (i)'s "human moving" at the start of the stop instead of at the event. It was fixed before this
write-up; the rule is unchanged. Public reads a only, from one read. The struck actor on public is unknown (the asl is pruned).
"Never launches" is a summary count (distance driven < 1 m, log > 5 m); it is not part of the class rule.
