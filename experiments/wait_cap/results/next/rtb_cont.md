# RTB1 line: retain base + RSG-L18 continuation, closed loop (prereg 2026-10-17 补记 1, exploratory)

The retain base is RTB1-F (the RB0-F recipe, 10 000 steps, plus a P0 longitudinal-only KD term on arrival frames, λ 1). In the
registered sense it is a STOP: the pilot fails line 5, dev ADE 0.5556 against 0.5276 ([rtb_pilot_gate.md](rtb_pilot_gate.md)). The user
overrode that, so everything below is exploratory and makes no registered verdict. Decision: [337](../../../../research/decisions/337.md).

| v | arm | base | continuation pool | cont. seed | matched ref |
|:--|:--|:--|:--|--:|:--|
| a | RTB1O-L18-s0 | RTB1-F-s0 | old on1_L18RB0 (off-policy for the new base) | 0 | RSG-L18-s0 |
| b | RTB1N-L18-s0 | RTB1-F-s0 | new on1_L18RTB1 (RTB1-F-s0's own 18-decision rollouts; faithful) | 0 | RSG-L18-s0 |
| c | RTB1O-L18-s1 | RTB1-F-s0 | old on1_L18RB0 | 1 | RSG-L18-s1 |
| d | RTB1OB1-L18-s1 | RTB1-F-s1 | old on1_L18RB0 | 1 | RSG-L18-s1 |

WTC-L18-s0 / s1 are the 2 500-step controls trained bit-identically to RSG-L18-s0 / s1. Every held number against them matches the
RSG reference, so the table does not repeat them.

## Held 460 @18 and the arrival reads (one read per arm)

| arm | zeros (ref) | paired mean vs ref [session CI] | at-fault (ref) | offroad (ref) | turns /24 (ref) | static scene: dist / contact | PAI stop17 zeros | arrival 4 s arc @k=1 | dev ADE |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| a | 38 (34) | +0.000 [−0.019, +0.021] | 9 (4) | 15 (17) | 16 (16) | **0.53 m / no** | **15 / 17** | 1.34 | 0.5472 |
| b | 41 (34) | −0.005 [−0.028, +0.016] | 9 (4) | 18 (17) | 17 (16) | **0.54 m / no** | 16 / 17 | 1.40 | 0.5448 |
| c | 41 (34) | −0.008 [−0.028, +0.007] | 10 (6) | 17 (13) | 17 (14) | **0.54 m / no** | **15 / 17** | 1.41 | 0.5434 |
| d | 39 (34) | −0.003 [−0.029, +0.015] | 11 (6) | 14 (13) | 17 (14) | **0.63 m / no** | **15 / 17** | 1.51 | 0.5447 |
| RSG-L18-s0 | 34 | – | 4 | 17 | 16 | 2.61 m / yes | 16 / 17 | 2.90 | 0.5284 |
| RSG-L18-s1 | 34 | – | 6 | 13 | 14 | – | – | – | – |
| P0 | 213 | – | – | 98 | – | 0.48 m / no | 14 / 17 | 0.43 | – |

There are 460 held scenes (452 paired). The CI is session-clustered. Each arm is paired against its seed-matched reference (ref);
the static scene and stop17 references are RSG-L18-s0 only. The arrival replay for the base alone is RTB1-F-s0 0.61 m (RB0-F-s0 3.30). For
comparison, WTCP-L18-s0 reads static 2.40 m with contact, stop17 16 / 17, arrival 3.06.

- **Arrival stop is robust.** All four arms stop on the static scene without contact (0.53–0.63 m; the human stops at 0.50 m). Every
  RB0-based stage since P2-F hits the lead there. stop17 reads 15 / 17 in a, c and d and 16 / 17 in b. The continuation lengthens the
  replayed first arrival plan from 0.61 to 1.34–1.51 m, but the closed loop still stops.
- **The cost is also robust.** Zeros run +4 to +7 against the matched reference, and the descriptive line "≤ C + 3" fails in all four
  arms. At-fault runs +4 to +5. Mean score is flat in all four (every CI covers 0). Turns are no worse. Offroad is mixed (−2 to +4).
- **The same scenes break in every arm.** Seven scenes are new zeros in all four arms: 53d3, de4a, 6d2b, 9d1d, 9a67, ddc1 and 1f54.
  They come from logs 2021.06.28 veh-26 / veh-38, 2021.09.09 veh-40, 2021.09.16 veh-42 and 2021.10.06 veh-52. Those zeros do not
  depend on the seed or the pool.
- **Base seed (d) and faithful pool (b).** d differs from a only by seed and shows the same pattern (39 zeros, at-fault 11). The new
  on-policy pool (b) is no better than the old off-policy pool (a): 41 vs 38 zeros, at-fault 9 vs 9, stop17 16 vs 15.

## Wide reads, RTB1O-L18-s0 (a) vs the stored RSG-L18-s0 reads

| read | scenes | zeros (C) | mean (C) | paired [session CI] | at-fault (C) | offroad (C) | front / lateral / rear (C) | stop17 zeros (C) |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|
| public nuPlan @10 | 1491 | 63 (58) | 0.9282 (0.9397) | **−0.011 [−0.023, −0.001]** | 7 (8) | 36 (31) | 5 / 2 / 24 (6 / 2 / 18) | – |
| PAI 281, submission config (g10 + VCONT + LEAD + ACC, fast) | 281 | 76 (76) | 0.6765 (0.6736) | +0.003 [−0.034, +0.038] | 12 (9) | 17 (16) | 6 / 6 / 37 (5 / 4 / 40) | 15 / 17 (16 / 17) |

The C columns use these stored reads: public `route_follow/guard/read10/RSG-L18-s0` and PAI `route_follow/pai_g10/acc` (decision 328 / 330).
On public 1491 the score is slightly worse, with a CI that excludes 0. That set has +5 zeros, +5 offroad and +6 rear contacts: the ego
gets struck from behind more often, which fits a car that stops or brakes where the log does not. At-fault does not rise there
(7 vs 8). On PAI the score, zeros and offroad are unchanged, at-fault is +3, and there is one more stop17 stop.

## At-fault geometry (a vs RSG-L18-s0, held)

The source is [rtb_cont/atfault.md](rtb_cont/atfault.md) (`nx_atfault.py --geo`; traces in [rtb_cont/atfault_traces.png](rtb_cont/atfault_traces.png),
per-contact data in [rtb_cont/atfault_geo.json](rtb_cont/atfault_geo.json)). a has 9 at-fault scenes and RSG has 4. Three are shared:
0722 (same-direction rear-end), daf3 (crossing through the opposite lane) and 47e0 (head-on in the opposite lane). One is RSG-only:
7bbb. Six are a-only:

| scene | class | what happens |
|:--|:--|:--|
| 9a67 | opposite lane | head-on (struck heading 147°) with the ego in the opposing lane, 3.0 m off the human path; RSG stays in its own lane at the same station |
| 365c | opposite lane | cuts the corner through the opposing lane (0.57 share 2 s before), then hits a stationary car ahead in its own lane, 1.77 m off the human path |
| a3c4 | own lane, late stop | rear-ends a stationary queue: 5.1 → 2.3 m/s over the last 2 s, so the brake comes too late |
| 6d2b | own lane, launch | same-direction car at 9.2 m/s; ego accelerates 4.5 → 8.4 m/s into it on launch |
| de4a | own lane, launch | same-direction car at 2.6 m/s, ego 2.4 → 1.9 m/s at contact; a merge on a slow launch |
| 84b3 | own lane, relaunch | crossing car (−98°) hit at 1.5–1.7 m/s after a relaunch, in the veh-52 log |

Four of the six are longitudinal own-lane contacts around stopping and relaunching (a late stop, two launches into a same-direction car,
one relaunch into a crossing car). The extra stops come with extra relaunches, and the relaunches are where the base gets hit.
de4a, 6d2b and 9a67 are among the seven new zeros that every arm shares.

## Files

- Per arm: [rtb_cont_a.md](rtb_cont_a.md), [rtb_cont_b.md](rtb_cont_b.md), [rtb_cont_c.md](rtb_cont_c.md), [rtb_cont_d.md](rtb_cont_d.md) (+ `.json` with zero flips).
- Wide reads: [rtb_wide_RTB1O-L18-s0.md](rtb_wide_RTB1O-L18-s0.md). Pilot and dev ADE decomposition: [rtb_pilot_gate.md](rtb_pilot_gate.md),
  [rtb_devdecomp.md](rtb_devdecomp.md).
- Box: `runs/wait_cap/next/` (`cont/<v>/`, `cont/wide/`, `jobs.txt`, `stages.txt`). Code: `experiments/wait_cap/scripts/nx.sh`, `nx.py` (`cont`, `wide`), `nx_atfault.py`.
