# Release policy replay on the logged red-light answers (vred, vred2, vred3)

Offline replay of release policies on the answers Qwen3-VL-4B gave while the car was held at a red light; no CARLA, no GPU, no driving run. Scope: 62 hold episodes of the original runs (`vred` 21, `vred2` 20, `vred3` 21; 10 routes with a light, 2 traffic seeds, the two seeds give nearly identical episodes, so there are about 14 distinct light situations, each seen up to 6 times), plus the 7 reruns with a chase camera (`v2-gif*`), used only in the infraction table. Definitions: [../plans/2026-10-02-offline-analyses-definitions.md](../plans/2026-10-02-offline-analyses-definitions.md) section A. Scripts: [release_extract.py](../scripts/release_extract.py), [route_common.py](../scripts/route_common.py), [release_replay.py](../scripts/release_replay.py), [release_replay_report.py](../scripts/release_replay_report.py). Row-level CSVs: `release_replay_policies.csv`, `release_replay_by_episode.csv`, `release_replay_episodes.csv`, `release_replay_infractions.csv`.

**Limit of every number here:** this is a replay on logged answers. A different release decision would have moved the car differently, the later frames (and therefore later answers) would not be the logged ones, and what the car would have done afterwards (start-up time, clearing the line before the light turns red) is not visible. Only the decision, its timing against the truth light, and the rule's behaviour on the answers that exist can be read.

## 1. Policy trade-off, all positions

Each row replays one rule on the answers of every hold episode, starting at the moment the logged hold began. False first release = the first release of the episode happened while the truth light of the governing light was red or yellow. Delay = seconds from the truth light turning green to the release, with a fresh rule state at that moment (so answers that still show red right after the change count against the rule); "missed" = not released before the light left green or the window ended, counted as above 1 s and above 2 s. `K=2 (current)` reproduces the 8 false releases of the logs exactly (verified from logs).

| policy | episodes | episodes with a false first release | false release events (re-armed) | episodes with a green window | released in it | missed | delay median s | delay p90 s | n > 1 s (missed incl.) | n > 2 s (missed incl.) |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| K=1 | 62 | 11 | 17 | 57 | 56 | 1 | 0.52 | 1.30 | 8 | 2 |
| K=2 (current) | 62 | 8 | 8 | 57 | 55 | 2 | 1.05 | 1.80 | 31 | 2 |
| K=3 | 62 | 5 | 5 | 57 | 54 | 3 | 1.55 | 2.30 | 57 | 11 |
| K=4 | 62 | 5 | 5 | 57 | 54 | 3 | 2.05 | 2.80 | 57 | 32 |
| single p>=0.5 | 62 | 11 | 17 | 57 | 56 | 1 | 0.52 | 1.30 | 8 | 2 |
| single p>=0.8 | 62 | 7 | 10 | 57 | 56 | 1 | 0.55 | 1.30 | 8 | 2 |
| single p>=0.9 | 62 | 6 | 9 | 57 | 56 | 1 | 0.55 | 1.30 | 8 | 2 |
| single p>=0.95 | 62 | 5 | 8 | 57 | 56 | 1 | 0.55 | 1.30 | 8 | 3 |
| single p>=0.99 | 62 | 5 | 5 | 57 | 55 | 2 | 0.55 | 2.30 | 11 | 9 |
| K=2 and p>=0.9 | 62 | 5 | 5 | 57 | 55 | 2 | 1.05 | 1.84 | 35 | 7 |
| K=2 and p>=0.99 | 62 | 4 | 4 | 57 | 46 | 11 | 1.05 | 1.22 | 37 | 13 |
| cusum >= 3 | 62 | 6 | 9 | 57 | 56 | 1 | 0.55 | 1.30 | 8 | 2 |
| cusum >= 5 | 62 | 5 | 5 | 57 | 55 | 2 | 0.55 | 1.16 | 8 | 3 |
| cusum >= 7 | 62 | 5 | 5 | 57 | 55 | 2 | 1.00 | 1.80 | 29 | 3 |
| cusum >= 10 | 62 | 5 | 5 | 57 | 55 | 2 | 1.05 | 1.82 | 33 | 6 |
| transition: red p>=0.9, then 1 green p>=0.9 | 62 | 4 | 4 | 57 | 20 | 37 | 0.60 | 1.46 | 40 | 39 |
| transition: red p>=0.9, then 2 greens p>=0.9 | 62 | 3 | 3 | 57 | 18 | 39 | 1.06 | 1.21 | 53 | 39 |
| transition: red p>=0.7, then 2 greens p>=0.9 | 62 | 3 | 3 | 57 | 24 | 33 | 1.05 | 1.19 | 53 | 33 |
| transition: red p>=0.5, then 1 green p>=0.9 | 62 | 6 | 6 | 57 | 31 | 26 | 0.55 | 1.30 | 32 | 27 |
| position-gated: K=2 before the line, K=4 and p>=0.99 past it | 62 | 4 | 4 | 57 | 51 | 6 | 1.05 | 1.83 | 35 | 9 |

Reading it: the rules that release on any single answer (K=1, p>=0.5) release falsely 11 times; K=3 and K=4 cut that to 5 only by costing 0.5 and 1.0 s more on every green; `cusum >= 5` reaches 5 false releases at the speed of K=1 (median 0.55 s). The `transition` rules (a confident red directly before the green run) miss most real greens, because the answer before a real change is rarely at p_red >= 0.9 at distance.

## 2. Same, split by where the car stood

Position at the answer that triggered the release (false column) or when the light turned green (delay and missed columns), from the logged distances: A = front bumper before the stop line (stop-line distance > 0); B = past the stop line, rear axle still before the junction entrance; C = rear axle past the entrance. B+C is "past the line or under the light". `vred` stops at the junction entrance (position B), `vred2` / `vred3` at the stop line (position A).

| policy | A: false first rel. | A: green eps | A: missed | A: delay med s | A: n > 2 s | B+C: false first rel. | B+C: green eps | B+C: missed | B+C: delay med s | B+C: n > 2 s |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| K=1 | 6 | 46 | 0 | 0.55 | 0 | 5 | 11 | 1 | 0.50 | 2 |
| K=2 (current) | 2 | 46 | 0 | 1.05 | 0 | 6 | 11 | 2 | 1.05 | 2 |
| K=3 | 1 | 46 | 0 | 1.55 | 7 | 4 | 11 | 3 | 1.53 | 4 |
| K=4 | 1 | 46 | 0 | 2.05 | 25 | 4 | 11 | 3 | 2.02 | 7 |
| single p>=0.5 | 6 | 46 | 0 | 0.55 | 0 | 5 | 11 | 1 | 0.50 | 2 |
| single p>=0.8 | 3 | 46 | 0 | 0.55 | 0 | 4 | 11 | 1 | 0.53 | 2 |
| single p>=0.9 | 2 | 46 | 0 | 0.55 | 0 | 4 | 11 | 1 | 0.53 | 2 |
| single p>=0.95 | 2 | 46 | 0 | 0.55 | 1 | 3 | 11 | 1 | 0.55 | 2 |
| single p>=0.99 | 0 | 46 | 0 | 0.55 | 5 | 5 | 11 | 2 | 0.55 | 4 |
| K=2 and p>=0.9 | 1 | 46 | 0 | 1.05 | 4 | 4 | 11 | 2 | 1.05 | 3 |
| K=2 and p>=0.99 | 0 | 46 | 7 | 1.05 | 9 | 4 | 11 | 4 | 1.05 | 4 |
| cusum >= 3 | 2 | 46 | 0 | 0.55 | 0 | 4 | 11 | 1 | 0.50 | 2 |
| cusum >= 5 | 1 | 46 | 0 | 0.55 | 1 | 4 | 11 | 2 | 0.50 | 2 |
| cusum >= 7 | 1 | 46 | 0 | 1.00 | 1 | 4 | 11 | 2 | 1.05 | 2 |
| cusum >= 10 | 1 | 46 | 0 | 1.05 | 4 | 4 | 11 | 2 | 1.05 | 2 |
| transition: red p>=0.9, then 1 green p>=0.9 | 1 | 46 | 32 | 0.55 | 32 | 3 | 11 | 5 | 1.02 | 7 |
| transition: red p>=0.9, then 2 greens p>=0.9 | 0 | 46 | 32 | 1.06 | 32 | 3 | 11 | 7 | 1.15 | 7 |
| transition: red p>=0.7, then 2 greens p>=0.9 | 0 | 46 | 26 | 1.05 | 26 | 3 | 11 | 7 | 1.15 | 7 |
| transition: red p>=0.5, then 1 green p>=0.9 | 2 | 46 | 22 | 0.56 | 22 | 4 | 11 | 4 | 0.55 | 5 |
| position-gated: K=2 before the line, K=4 and p>=0.99 past it | 2 | 46 | 0 | 1.05 | 0 | 2 | 11 | 6 | 2.05 | 9 |

Reading it: before the stop line (A, 46 episodes with a green window) K=2 releases falsely twice, K=1 and the p >= 0.5 rule six times, `cusum >= 5` once, and the fast rules release in about 0.55 s. Past the line (B+C, 11 episodes with a green window) every rule releases falsely 2 to 6 times and misses 1 to 7 of those 11 greens; a rule that is strict there (the position-gated row) removes false releases only by missing 6 of 11 real greens and by waiting 2 s.

## 3. How often the green probability was high on a truly red light

Per position, over the answers given while the car was held (first table), and over all logged answers of the three arms with a signalised junction ahead (second table; stop line within 50 m ahead or up to 6 m past the entrance). Truth = the state of the governing light at the query time, reconstructed per run (see the notes). The right-hand columns give the same probabilities on a truly green light (recall of green).

| position | answers on a red or yellow light | arm-route cells | p_green >= 0.5 | p_green >= 0.9 | p_green >= 0.99 | answered green (argmax) | answers on a green light | p_green >= 0.5 | p_green >= 0.9 |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| A before stop line | 2199 | 30 | 0.7% | 0.1% | 0.0% | 0.7% | 258 | 95.3% | 88.0% |
| B between stop line and entrance | 346 | 13 | 6.9% | 4.9% | 3.5% | 6.9% | 216 | 50.9% | 43.5% |
| C past entrance | 15 | 7 | 53.3% | 53.3% | 53.3% | 53.3% | 55 | 80.0% | 74.5% |

| position | answers on a red or yellow light | arm-route cells | p_green >= 0.5 | p_green >= 0.9 | p_green >= 0.99 | answered green (argmax) | answers on a green light | p_green >= 0.5 | p_green >= 0.9 |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| A before stop line | 2447 | 30 | 0.7% | 0.1% | 0.0% | 0.7% | 686 | 96.5% | 91.3% |
| B between stop line and entrance | 363 | 16 | 6.9% | 5.0% | 3.3% | 6.9% | 339 | 64.6% | 59.6% |
| C past entrance | 107 | 17 | 15.9% | 15.9% | 15.9% | 15.9% | 219 | 77.6% | 68.0% |

| arm | position | answers | p_green >= 0.5 | p_green >= 0.9 |
|:--|--:|--:|--:|--:|
| vred | A before stop line | 504 | 0.8% | 0.2% |
| vred | B between stop line and entrance | 318 | 5.3% | 3.5% |
| vred | C past entrance | 8 | 62.5% | 62.5% |
| vred2 | A before stop line | 823 | 0.2% | 0.0% |
| vred2 | B between stop line and entrance | 20 | 25.0% | 20.0% |
| vred2 | C past entrance | 3 | 33.3% | 33.3% |
| vred3 | A before stop line | 872 | 1.0% | 0.2% |
| vred3 | B between stop line and entrance | 8 | 25.0% | 25.0% |
| vred3 | C past entrance | 4 | 50.0% | 50.0% |

Reading it: before the stop line a confident green on a red light is rare (p_green >= 0.9 in 0.1% of 2199 answers, never >= 0.99). Between the stop line and the entrance it is 4.9% (3.5% at >= 0.99, 346 answers, 13 arm-route cells); past the entrance 53% of 15 hold answers and 16% of 107 logged answers. The same positions are where green is recognised worst (51% and 80% at p >= 0.5, against 95% before the line). Verified from logs; the counts in position C are small.

## 4. Red-light infractions of the vred family, re-read with the release definition

False release = a release by K=2 consecutive green answers while the governing light was red or yellow. Infraction time is not in the logs (the logged location is the light's position), so the table uses the time the car's tail passed the stop line / the junction entrance (rear axle + 1.06 m) and the first of the two at which the light was red. Category = what ended the last hold before that crossing.

| arm | seed | route | tail at stop line s | tail at entrance s | truth then (stop line) | truth then (entrance) | category | last R2 segment ended | false K=2 releases in the approach (s, truth) |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| vred | 1 | 15612 | 11.1 | 15.6 | 2 | 0 | false green release, then crossed | ('green', 8.6) | 8.4(truth 2) |
| vred | 0 | 16390 | 8.6 | 9.6 | 2 | 2 | no R2 hold before the crossing | None | - |
| vred | 1 | 16390 | 8.6 | 9.6 | 2 | 2 | no R2 hold before the crossing | None | - |
| vred | 0 | 16529 | 46.0 | 47.5 | 2 | 2 | R5 25 s fallback released a red light | ('r5', 44.0) | 8.9(truth 2) |
| vred | 0 | 27297 | 17.1 | 18.1 | 2 | 2 | false green release, then crossed | ('green', 15.6) | 15.4(truth 2) |
| vred | 1 | 27297 | 17.1 | 18.1 | 2 | 2 | false green release, then crossed | ('green', 15.6) | 15.4(truth 2) |
| vred2 | 0 | 15483 | 33.5 | 38.0 | 2 | 0 | correct green release, light red again before the tail cleared | ('green', 30.6) | - |
| vred2 | 1 | 15483 | 33.5 | 35.0 | 2 | 2 | correct green release, light red again before the tail cleared | ('green', 30.6) | - |
| vred2 | 0 | 15612 | 21.1 | 23.1 | 2 | 0 | correct green release, light red again before the tail cleared | ('green', 15.6) | - |
| vred2 | 1 | 15612 | 18.6 | 23.1 | 2 | 0 | correct green release, light red again before the tail cleared | ('green', 15.6) | - |
| vred2 | 0 | 16390 | 8.6 | 9.6 | 2 | 2 | no R2 hold before the crossing | None | - |
| vred2 | 1 | 16390 | 8.6 | 9.6 | 2 | 2 | no R2 hold before the crossing | None | - |
| vred2 | 1 | 27297 | 18.1 | 19.1 | 2 | 2 | false green release, then crossed | ('green', 15.6) | 15.4(truth 2) |
| vred3 | 0 | 15483 | 33.5 | 35.0 | 2 | 2 | correct green release, light red again before the tail cleared | ('green', 30.6) | - |
| vred3 | 1 | 15483 | 33.5 | 35.0 | 2 | 2 | correct green release, light red again before the tail cleared | ('green', 30.6) | - |
| vred3 | 0 | 15612 | 21.6 | 23.1 | 2 | 0 | correct green release, light red again before the tail cleared | ('green', 15.6) | - |
| vred3 | 1 | 15612 | 18.6 | 23.1 | 2 | 0 | correct green release, light red again before the tail cleared | ('green', 15.6) | - |
| vred3 | 0 | 16390 | 8.6 | 9.6 | 2 | 2 | no R2 hold before the crossing | None | - |
| vred3 | 1 | 16390 | 8.6 | 9.6 | 2 | 2 | no R2 hold before the crossing | None | - |
| vred3 | 1 | 27297 | 18.1 | 18.6 | 2 | 2 | false green release, then crossed | ('green', 15.6) | 15.4(truth 2) |
| gif_vred2_gif | 1 | 15483 | 33.5 | 35.0 | 2 | 2 | correct green release, light red again before the tail cleared | ('green', 30.6) | - |
| gif_vred_gif | 0 | 16390 | 8.6 | 9.6 | 2 | 2 | no R2 hold before the crossing | None | - |
| gif_vred_gif | 0 | 16529 | 46.0 | 47.0 | 2 | 2 | R5 25 s fallback released a red light | ('r5', 44.0) | - |
| gif_vred_gif | 0 | 27297 | 20.1 | 21.6 | 2 | 2 | false green release, then crossed | ('green', 18.6) | 18.4(truth 2) |
| gif_vred_gifr1 | 1 | 15612 | 11.1 | 16.1 | 2 | 0 | false green release, then crossed | ('green', 8.6) | 8.4(truth 2) |
| gif_vred_gifr1 | 0 | 16529 | 63.0 | 64.0 | 2 | 2 | false green release, then crossed | ('green', 61.0) | 8.9(truth 2); 60.9(truth 2) |

Avoidability of the correct-release infractions (same motion after the release, release moved to the green onset): margin s = latest release time that still clears before the light turns red, minus the green onset; negative = impossible.

| arm | seed | route | margin s |
|:--|--:|--:|--:|
| vred2 | 0 | 15483 | -0.45 |
| vred2 | 1 | 15483 | -0.35 |
| vred2 | 0 | 15612 | -2.65 |
| vred2 | 1 | 15612 | -0.25 |
| vred3 | 0 | 15483 | -0.45 |
| vred3 | 1 | 15483 | -0.45 |
| vred3 | 0 | 15612 | -3.1 |
| vred3 | 1 | 15612 | -0.2 |

| windows | min s | median s | p90 s | max s | n <= 3.5 s | n <= 5 s |
|:--|--:|--:|--:|--:|--:|--:|
| 51 | 2.70 | 10.00 | 10.30 | 51.65 | 8 | 8 |

Original `vred` runs, 6 infractions: **3 followed a false release** (15612 seed 1, 27297 seeds 0 and 1: released at red or yellow by two green answers, the tail crossed the line while red), **1 followed the 25 s R5 fallback** (16529 seed 0: the car was falsely released once at 8.9 s and re-held; the light was green from 29.5 to 39.4 s but the model, standing past the stop line, answered red throughout; R5 released the car at 44 s when the light was red again; counted loosely, 4 of 6 had a false release somewhere in the approach), and **2 had no hold at all** (16390 seeds 0 and 1: green to red at 7.6 s with no yellow, the front bumper crossed at 7.5 s). So the written cause in `vred.md` (only one misread light) under-counted: 3 of 6, not 1, followed a false release. In `vred2` and `vred3` (7 infractions each) 1 followed a false release (27297 seed 1), 4 were correct releases after which the light turned red again before the tail cleared (15483, 15612; the avoidability table below shows that releasing at the very moment of the green onset would not have cleared in time, margins -0.2 to -3.1 s, assuming the same motion after the release), 2 were the 16390 no-hold case. The reruns with a chase camera (`v2-gif*`) show false releases in 3 of their 6 red-light runs (15612 seed 1, 16529 seed 0 in one of its two reruns, 27297 seed 0), consistent with decision 86's supplement.

Reading the table: `truth then` columns use 0 green, 1 yellow, 2 red at the moment the tail passed. A row is "false green release, then crossed" only when the last hold ended in a K=2 release that was false at its release time.

## 5. By route and by arm

Per route (cells: false first releases / missed green windows, over the 6 episodes of the route = 2 seeds x 3 arms, 24944 has 9):

Cells: false first releases / missed green windows.

| route | episodes | K=1 | K=2 (current) | K=3 | cusum >= 5 | single p>=0.99 | position-gated: K=2 before the line, K=4 and p>=0.99 past it |
|:--|--:|--:|--:|--:|--:|--:|--:|
| 15102 | 6 | 1 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 |
| 15483 | 6 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 1 |
| 15612 | 6 | 2 / 0 | 2 / 1 | 1 / 1 | 1 / 1 | 1 / 1 | 0 / 1 |
| 16529 | 6 | 4 / 1 | 2 / 1 | 0 / 1 | 0 / 1 | 0 / 1 | 1 / 2 |
| 24944 | 9 | 0 / 0 | 0 / 0 | 0 / 1 | 0 / 0 | 0 / 0 | 0 / 2 |
| 27043 | 6 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 |
| 27297 | 5 | 4 / 0 | 4 / 0 | 4 / 0 | 4 / 0 | 4 / 0 | 3 / 0 |
| 27870 | 6 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 |
| 28147 | 6 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 |
| 9196 | 6 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 |

| arm | policy | episodes | false first releases | missed | delay median s | delay p90 s | n > 2 s (missed incl.) |
|:--|--:|--:|--:|--:|--:|--:|--:|
| vred | K=1 | 21 | 6 | 1 | 0.50 | 1.30 | 2 |
| vred | K=2 (current) | 21 | 5 | 2 | 1.03 | 1.80 | 2 |
| vred | K=3 | 21 | 3 | 3 | 1.50 | 2.30 | 6 |
| vred | cusum >= 5 | 21 | 3 | 2 | 0.50 | 1.25 | 3 |
| vred | single p>=0.99 | 21 | 3 | 2 | 0.60 | 3.10 | 7 |
| vred | position-gated: K=2 before the line, K=4 and p>=0.99 past it | 21 | 2 | 6 | 1.80 | 2.05 | 9 |
| vred2 | K=1 | 20 | 2 | 0 | 0.50 | 0.78 | 0 |
| vred2 | K=2 (current) | 20 | 1 | 0 | 1.00 | 1.28 | 0 |
| vred2 | K=3 | 20 | 1 | 0 | 1.55 | 2.33 | 3 |
| vred2 | cusum >= 5 | 20 | 1 | 0 | 0.55 | 1.12 | 0 |
| vred2 | single p>=0.99 | 20 | 1 | 0 | 0.55 | 1.22 | 2 |
| vred2 | position-gated: K=2 before the line, K=4 and p>=0.99 past it | 20 | 0 | 0 | 1.00 | 1.28 | 0 |
| vred3 | K=1 | 21 | 3 | 0 | 0.55 | 0.98 | 0 |
| vred3 | K=2 (current) | 21 | 2 | 0 | 1.05 | 1.48 | 0 |
| vred3 | K=3 | 21 | 1 | 0 | 1.55 | 1.98 | 2 |
| vred3 | cusum >= 5 | 21 | 1 | 0 | 0.55 | 0.98 | 0 |
| vred3 | single p>=0.99 | 21 | 1 | 0 | 0.55 | 0.98 | 0 |
| vred3 | position-gated: K=2 before the line, K=4 and p>=0.99 past it | 21 | 2 | 0 | 1.05 | 1.48 | 0 |

Reading it: the false releases that no rule removes sit on two routes. 27297 (4 episodes) is a persistent confident green (p_green about 1.0 for 1.5 s and longer) on a red light at or past the stop line; 15612 seed 1 is similar (four green answers, one at p = 1.0). The three episodes in which `cusum >= 5` and K=3 beat K=2 (15612 seed 0, 16529 seed 0, 16529 seed 1 of `vred3`) are two-answer flips with p_green between 0.51 and 0.92. In `vred2` (car at the stop line) K=2 and `cusum >= 5` are at 1 false release and 0 missed, the position-gated rule at 0 and 0.

## Which policy

`cusum >= 5` (S = max(0, S + clip(ln((p_green + 0.001) / (p_red + 0.001)), -7, 7)) per answer, S reset to 0 when p_red >= 0.9, release at S >= 5) dominates `K=2` and `K=3` in the table: fewer false first releases than K=2 (5 against 8, K=3: 5), median delay 0.55 s against 1.05 s (K=3: 1.55 s), 8 of 57 releases slower than 1 s against 31 (K=3: 57), same missed windows as K=2 (2), one more release above 2 s than K=2 (3 against 2, K=3: 11). It also dominates `single p >= 0.99` (same false releases, p90 1.16 s against 2.30 s). `cusum >= 3` is equal on speed with 6 false releases. Nothing in the table removes the remaining 5 (27297, 15612 seed 1): those are confidently wrong answers while the car stands past the stop line, so an answer-only rule cannot separate them from a real change. The position cut (stricter rule past the line) is the only lever that touches them, and in this data it costs 6 of 11 real greens past the line, so it is only worth using together with the stop-line target (`vred2`, where it gave 0 false and 0 missed).

Evidence strength: the differences between `cusum >= 5` and K=2 are 3 episodes on 2 routes; the slower-release savings (0.5 s median) are robust across routes. Releasing earlier does not repair the largest infraction class of `vred2` / `vred3` (green window too short for the start-up, margins all negative).

## Verified and inferred

- Verified from logs: hold episodes and their end causes, the answers and probabilities, the K=2 false releases (replay equals logged), the distances used for the position classes, the infraction messages, the green onsets and window lengths per run.
- Inferred: the truth light per run is reconstructed from the plan-step context (`ctx.tl`, `ctx.tl_id`, stop-line distance) and from the light states logged with each answer (`gt.lights`), mapped to the junction by the stop-line window the agent itself uses ([entrance - 15 m, entrance + 1 m]); the infraction crossing time uses the tail 1.06 m behind the rear axle and takes the first of the stop line / junction entrance at which the light was red, because the scorer's line is not logged; the avoidability margins assume the motion after the release is unchanged; the position classes are computed from the logged route progress (0.5 s resolution of the arbitration steps).
- Not available: infraction timestamps; what the car would have done under another rule.
- Correction to earlier tooling: [bypass_common.py](../scripts/bypass_common.py) `route_lights` assumes the light timers are identical across the runs of a route. They are identical across seeds of one arm but not across arms: the scenario sets the ego light when the car arrives, so on 15102 the light turns green at 8.1 s in the `drive` arms and at 44.5 s in the `vred` arms (15483: 12.9 s against 21.7 s / 29.4 s). The replay therefore uses per-run timelines.
