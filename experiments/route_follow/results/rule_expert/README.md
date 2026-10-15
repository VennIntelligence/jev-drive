# RULE EXPERT: a map-based rule planner in the closed loop (teacher check, no training)

Prereg `plans/2026-10-15-rule-expert-prereg.md` (Amendment 1 after the smoke). Code: `experiments/alpasim/lib/rule_expert.py` (served
through sh30_driver.py / pai_driver.py with `JEV_EXPERT`), `scripts/rf_expert.py` (pack | fig | report), `scripts/rf_expert.sh`.
Registered variant XP-L12-E46 (lookahead 1.2 s, turn entry speed 4.6 m/s); exploratory XP-L12-E60 (entry 6.0 m/s, pre-declared in
Amendment 1 because the held list had 2 speed failures; it cannot change the verdict).

## Smoke (Amendment 1)

[figs/rule_expert_smoke.png](../../figs/rule_expert_smoke.png): top row the 3 held turn scenes, bottom row the 3 PAI strict turns. Look at
the red expert path against the blue human log and the dashed chain it tracks: it follows the chain through every turn, while the orange
RSG-L18-s0 leaves the road (e14711ef, 9f393c8a), the corridor (3232fdb7) or is hit (22fa5c92). Red dots are decisions with a yield
contact (22fa5c92 waits for the oncoming car turning across it). e9cfc98f (top right) is the held speed failure: the expert takes the
connector's wider line at 4.6 m/s and does not reach exit + 10 m within 18 decisions. Three fixes came out of the smoke (chain start falls
back to the route, parked-box nudge, swept-footprint yield as registered); see Amendment 1.

## Held 460 / 18 decisions (box)

| | XP-L12-E46 (registered) | XP-L12-E60 (exploratory) | RSG-L18-s0 | RSG-L18-s1 |
|:--|--:|--:|--:|--:|
| held turns completed (the 24 of RSG-L18-s0's rf_follow read) | **22 / 24** | 23 / 24 | 16 / 24 | 14 / 24 |
| mean score, 452 scored (ol_confirm) | 0.9178 | 0.9252 | 0.8809 | 0.8830 |
| zeros / at-fault zeros (452) | 22 / 13 | 21 / 12 | 26 / 21 | 26 / 19 |
| collision / offroad / corridor zeros | 4 / 9 / 9 | 3 / 9 / 9 | 4 / 17 / 6 | 6 / 13 / 8 |
| GT distance (m) | 1.388 | 1.438 | 1.135 | 1.144 |
| E3 held straight (306 scenes, no turn before the log's end) | 0.9159 | 0.9159 | 0.8993 | 0.9081 |

- **E1 passes** (22 / 24 >= 20). Paired vs RSG-L18-s0: mean +0.037 [+0.001, +0.071], zeros -4 [-16, +8], at-fault -8 [-18, +3], GT
  distance +0.25 m [+0.08, +0.42] (the expert drives lane centres and keeps its own speed, not the human's line). Completion +25 pp
  [0, +50] vs s0, +33 pp [+10, +58] vs s1 (rf_follow). Junction-stretch distance to the route: median 0.22 m vs 0.52 m.
- Per-turn list ([held_XP-L12-E46/xp_turns.md](held_XP-L12-E46/xp_turns.md)): both failures are **speed** (`short`, no zero):
  e9cfc98f (the human accelerates from rest through the turn at 6-8 m/s) and 6f393be3 (019c508a; RSG completes it). No wrong lane chain, no
  collision, no corridor failure on the 24. At E60, 6f393be3 completes and only e9cfc98f stays short (0.93).
- **E3 passes** (+0.017 vs s0, +0.008 vs s1; line >= -0.05). E46 and E60 drive these scenes identically (V_E only acts at turns).
- Entry state ([entry_entry.md](held_XP-L12-E46/entry_entry.md)): entry speed equal to RSG's (median 4.27 vs 4.30 m/s; the held turns
  start about 5 m before the entry, so the approach is not tested here), lateral offset toward the inside at the entry 0.26 vs 0.34 m.
- Zeros: 13 shared with RSG-L18-s0, 21 of RSG's removed, 17 new (corridor 8, offroad 5, at-fault collision 4). 10 of the 17 new zeros happen
  at or past the human log's end station: the human stopped or slowed (light, queue), the expert, which ignores stop lines and keeps a
  3 m/s floor, drives on and leaves the evaluator's straight extension of the log ([xp_lines.md](held_XP-L12-E46/xp_lines.md)). These
  scenes are outside the 24-turn set.

Pack coverage ([pack_held_index.json](pack_held_index.json)): 460 / 460 packs, 0 fallbacks, 1 scene with a chain break, 15 with ambiguous
samples; 293 route turns, 3 without a matching chain turn and 5 with the route > 2 m from the chain (Amendment 1's route fallback covers
them); 87 scenes nudge around 146 parked boxes.

Box: `$DATA_DIR/runs/route_follow/rule_expert/` (pack_held, smoke/, report/XP-L12-E{46,60}), reads
`$DATA_DIR/runs/route_mem/read/RSG-L18-s0-XP-L12-E{46,60}`.

## PAI 281 (Tokyo, gains 20, the reference's 8 chunk lists; card 1 only, another lane held card 0; 15 381 s)

| | XP-L12-E46 | RSG-L18-s0 (r281/rsg) |
|:--|--:|--:|
| **strict-36 turns completed** | **29 / 36** | 5 / 36 |
| d293 shape gate (completed and within the class gate) | 24 | 1 |
| 22 runway turns completed | 15 | 3 |
| exam turn scenes completed (72, 18 with a strict turn) | 14 | 3 |
| all 281: mean / zeros (collision / offroad / corridor) | 0.7646 / 36 (7 / 8 / 21) | 0.5720 / 70 (9 / 14 / 47) |
| paired mean vs RSG, all 281 | +0.193 [+0.136, +0.250] | |
| exam 72: mean / zeros | 0.6872 / 16 | 0.3348 / 38 |
| control 75 (E3): mean / zeros | 0.8116 / 8 | 0.7779 / 0 |
| rear-ended / handoff spins (all 281) | 49 / 2 | 77 / 2 |

- **E2 passes, at the line** (29 / 36 >= 29). Gained 26 strict turns, lost 2 (382a7247, d90fe123).
- **E3 passes**: control 75 +0.034 [-0.062, +0.129] (line >= -0.05), but with 8 zeros where RSG has none (6 corridor).
- Runway turns (rf_appread, [pai_XP-L12-E46/runway.md](pai_XP-L12-E46/runway.md)): entry speed median 4.57 vs RSG 7.38 m/s (human
  4.99), speed excess over the human from entry - 26 m -1.13 m/s [-1.99, -0.45], inward offset at entry - 20 m +0.29 m [+0.12, +0.51]
  (RSG sat 0.44 m toward the outside). The approach state the earlier lines tried to teach (d298 / d303) is what the expert produces.
- Zeros: 21 shared with RSG, 49 of RSG's removed, 15 new (corridor 8, offroad 4, at-fault collision 3); 9 of the 15 are at or past
  the human log's end station (the same artefact as on held).
- Per-turn failure list ([pai_XP-L12-E46/xp_turns.md](pai_XP-L12-E46/xp_turns.md)), 7 failed:
  - **speed 4** (6fd8cf30, abd45a30, d90fe123, e37aa159): stopped for long stretches by the yield rule (63-174 yield decisions) and
    then rear-ended (`short_rear`) or out of clip (`short_slow`). The swept-footprint yield stops 1.5 m short of where a slow lead is
    now, not of where it will be: correct for crossing / oncoming boxes, too conservative behind same-direction traffic.
  - **corridor 2** (1e8e939e, a174faed): offroad on the chain, on the sweeping bend after the turn / 50 m after the exit. PAI has no road
    area layer and narrow lane polygons; pure pursuit at 1.2 s lookahead cuts the inside of a long bend.
  - **collision 1** (382a7247): at-fault collision at the turn entry while yielding (105 yield decisions).
  - No wrong lane chain (all 36 route turns matched, route-to-chain distance <= 1.6 m).
  - 403d558a completes its turn and then has an at-fault collision (counted completed, as for every label).
- Figure [figs/rule_expert_pai_fails.png](../../figs/rule_expert_pai_fails.png): the 8 strict-turn scenes with a failure or a zero.
  Look at the red yield dots along the approach in the four speed panels (the expert waits, the human did not), and at the red X
  placements in 1e8e939e / a174faed: on the chain, but off the PAI road.

## Verdict (prereg rule)

**E1 and E2 both pass: the expert qualifies as a teacher** (E3 passes on both sets). Held 22 / 24 (RSG 16 / 14), PAI strict 29 / 36
(RSG 5), every failure on both sets is speed / yield or PAI corridor geometry, none is a wrong lane chain. Caveats: E2 sits exactly on
its line; the expert is privileged (map, human speeds, the replayed boxes 3 s ahead); stop lines are ignored, which costs zeros past the
log's end (held 10, PAI 9) and 8 control zeros; one run per set. Weak points to fix before using its labels: same-direction yield
(follow the lead's future box, not its current one) and the PAI off-road margin on long bends.

Tokyo: `/data/runs/route_night/expert/` (pack_pai, smoke/, XP-L12-E46 with read.md, follow.pkl, turns.json, xp/).
