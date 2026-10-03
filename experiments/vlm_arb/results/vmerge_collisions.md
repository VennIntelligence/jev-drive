# vmerge: why vehicle collisions rose (11 vs 6 for drive, seeds 0 and 1)

Source: `scripts/vmerge_collisions.py` and `scripts/vmerge_timeline.py` over the logged attempts (contacts.jsonl = first contact per actor, privileged.jsonl = actors and
bypass flags at the last snapshot before contact, vlm_decisions.jsonl = arbitration state). Official count = scorer's `collisions_vehicle` (27297 s1 is one actor counted
twice, so vmerge has 10 distinct events / 11 official). Time = game seconds, ego v in m/s, `rel` = actor centre in the ego frame (long, lat) in m at the snapshot
about 0.1 s before contact, actor v in m/s. "R1" = junction slow-down active (cap 4.5 m/s inside 25 m of a junction); drive's cap is 8. All times are logged; the
reading in the last column is from the 3-5 s timelines, not from a controlled test.

**Correction (2026-10-04, lane BYP, from [vm3_cross_offline.md](vm3_cross_offline.md)):** the 17280 and 27297 rows below read the contact time on
the world clock of contacts.jsonl, which runs ~1 s ahead of the scenario clock of privileged.jsonl, so they took the ego state ~1 s after the
contact. On the scenario clock (frame numbers) the 17280 contacts are at 17.5-18.2 s with the ego moving at ~4 m/s, i.e. just after it left
the stop sign, not standing at it; 27297 likewise. The class-B reading "hit while stopped" is therefore wrong: both are hits right after a
release (stop-sign dwell end or cusum release), the case a release check addresses. The other rows are unaffected by this note only where
their ego speed is not near a stop; they were not re-read.

## Per collision (vmerge, seeds 0 and 1) with the same route / seed in drive

| route | seed | t | ego v | other actor (v, rel) | arbitration at contact | drive, same route and seed | class |
|:--|--:|--:|--:|:--|:--|:--|:--|
| 27043 | 0 | 18.8 | 4.2 | impala (13.6, +8.6 / +6.0) | R1, no bypass; R2 release in the last 5 s | collided at 24.2 (ford, v 4.9) | A |
| 27043 | 1 | 20.3 | 3.9 | mini (0, -4.2 / -2.5) | R1, no bypass | collided at 24.3 (ford, v 4.8) | A |
| 9196 | 0 | 36.5 | 2.6 | firetruck (14.4, +9.7 / -2.2) | R1, no bypass | collided at 31.3 (firetruck, v 1.9) | A |
| 9196 | 1 | 36.1 | 3.4 | firetruck (1.3, -0.2 / +3.6) | R1, no bypass | collided at 30.6 (firetruck, v 1.7) | A |
| 37969 | 1 | 51.9 | 4.5 | patrol (0, -8.6 / -1.5) | R1, no bypass | collided at 37.3 (ford, v 1.6) | A |
| 27297 | 1 | 34.8 | 3.7 | impala (4.3, +1.9 / -1.9), counted twice | R1, R2 released within 5 s, no bypass | no collision (DS 70) | B |
| 17280 | 1 | 18.8 | 0.0 | patrol (0, +3.4 / -3.3), heading 134 deg off | R1, ego stopped at the stop sign, no bypass | no collision (DS 80, one stop infraction) | B |
| 19324 | 0 | 28.8 | 3.2 | mkz (0, -3.9 / -2.7) | bypass active 5.1-34.5 s, pulling out at 2-3 m/s beside traffic at 8-11 m/s | no collision (DS 33, stuck behind the obstacle) | C |
| 19324 | 1 | 30.1 | 2.4 | mustang (8.3, -5.9 / -4.7) | bypass active 5.1-35.3 s, same pull-out | no collision (DS 33) | C |
| 19832 | 1 | 27.2 | 0.2 | coupe (0, +3.1 / +1.8), standing 27 s | bypass active 7.9-35.5 s, creeping past the obstacle at 0.2 m/s, gap 0.0 m (impulse 229, a scrape) | no collision (DS 33) | C |

Other vmerge runs with a drive collision and none in vmerge: 37969 s0 (drive collided at 41.2; vmerge had a layout collision). Drive's six events: 27043 x2, 37969 x2,
9196 x2, all at ego v 1.6-5.3 with no bypass, no VLM state.

## Reading

- **Class A (5 of 11): not caused by the arbitration.** Same junction conflicts (27043, 9196, 37969) in which drive collides too (6 events in drive at these three routes,
  5 in vmerge). The ego is at 2.6-4.5 m/s in both arms at contact (drive 1.6-5.3), i.e. R1's 4.5 m/s cap did not make it slower than drive was; cross traffic (14 m/s on 27043, the
  firetruck on 9196) hits the car in the junction, mostly its rear quarter (`rel_long` negative). vmerge arrives earlier on 27043 (18.8 / 20.3 s vs 24.2 s).
- **Class B (3 official): R1 / R3 at a junction where drive had no collision.** 17280 s1: the car stood at the stop sign (v 0.0) and a turning patrol car hit it at 18.8 s
  (R3 / stop compliance puts the ego into the path of the scenario vehicle; drive does not stop there and took a stop infraction instead, DS 80 vs 60 for the collision).
  27297 s1: the cusum release (R2) was followed within 5 s by a hit by a crossing car; ego 3.7 m/s under R1. Hypothesis, not tested: the
  release rule has no cross-traffic check.
- **Class C (3 of 11): the bypass (R4 / pbyp3) pull-out.** The ego leaves its lane at 2-4 m/s next to a lane with traffic at 8-11 m/s and is hit from behind / beside (19324 both
  seeds), or scrapes the standing obstacle (19832 s1, gap 0.0 m at 0.2 m/s). The bypass gap flag (`gap_open`) flickers in the last seconds of 19324 s1. Drive never tries this and
  stays stuck (DS 33), so these collisions come with +27 to +67 DS on the same routes (DS 60 with a collision vs 33 without).
- Net: 11 - 6 = +5 = 3 (class B) + 3 (class C) - 1 (37969 s0, drive only).

## vmerge2 (seeds 0-3, 76 runs) and drive (seeds 0-3, 76 runs): the same classes

| arm | official vehicle collisions | A: R1 junction routes shared with drive (27043, 9196, 37969) | B: 17280 / 27297 | C: bypass active at contact (19324, 19832, 2520) |
|:--|--:|--:|--:|--:|
| drive | 12 | 12 | 0 | 0 |
| vmerge | 11 (seeds 0, 1) | 5 | 3 | 3 |
| vmerge2 | 21 | 11 | 3 (17280 seeds 1, 2, 3, ego v 0.1-0.3 at 18.4-19.1 s, the same patrol car) | 7 (19324 s0, 2520 s0 and s2, 19832 s0 (2), s1, s3) |

17280 is a systematic exchange: the stop compliance (no stop infraction in any vmerge / vmerge2 run) trades the 0.8 stop-infraction factor for a 0.6 collision factor in 4 of
6 vmerge-family runs (vmerge2 s0 DS 100, s1 16.8 with a second penalty, s2 and s3 60). Full per-collision rows for vmerge2 and drive (seeds 0-3) come from
`python scripts/vmerge_collisions.py drive vmerge vmerge2` on the box; the bypass collisions grow with more seeds because bypass fires on every obstacle route, the junction
ones are constant.
