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

## PAI 281 (Tokyo)

Pending: queued behind another lane's spin_v9 reads on both cards (`/data/runs/route_night/expert/wait_run.sh`, tmux jev:xp-wait).
