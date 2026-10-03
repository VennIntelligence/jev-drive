# Loss budget: how many points come back if one failure class were fixed perfectly

2026-10-04. **Every number below is an oracle ceiling, not an achievable gain.** A class is fixed by substituting the reference, a better arm, or a
term value of 1 on its own failing cases, then re-scoring with the board's scorer; nothing is predicted and no new driving was run. Points are the board's own scale
(navtest PDMS, navhard two-stage EPDMS, HUGSIM mean HD x 100, B2D mean DS); they are not comparable across boards. Reference driver = shipped Cinque; "current best" = it_dw3-s0 +
selector (decision 101) on navtest / navhard / HUGSIM, and `vmerge2` on B2D (the adapted model is not in B2D yet). Code: `scripts/loss_budget_*.py`.
"Recovered so far" = reference-driver ceiling minus best-driver ceiling per class (how much of that class's headroom the best driver already removed); it is a
difference of two ceilings and the classes overlap, so it does not add up to the measured total gain.

## Method per board

- **navtest** (official v1 PDMS = NC x DAC x (5 EP + 5 TTC + 2 C) / 12, n 12 146; DDC is reported only and not in the score). Per-token results of the shipped and best arms are the official CSVs; the in-process
  harness (skill_pack/scripts/offroad_lib) re-simulates the plans for the DAC features. A class's tokens take the human trajectory's official token scores
  (`v1_navtest_human`, PDMS 94.55) except that a token is never made worse than its own score (clip; the unclipped variant is in the json, difference < 0.02).
- **navhard** (two-stage EPDMS, n 5 912, the devkit's scorer and aggregation in-process; the harness reproduces 33.33 and 35.76). The official human run does not exist (human trajectory is None on the synthetic stage-2 frames), so the substitute is the
  PDM-Closed reference trajectory scored in-process (its own combined EPDMS is 52.82); clip as above; HC and EC stay each arm's own in the substitution and are
  handled by the comfort row.
- **Failing token** = arm token score below 0.8 x the reference token's score (navtest 1264 / 12 146 shipped, 1200 best; navhard 2484 / 5 912 shipped, 2154 best; 95% of the navhard failing tokens are stage 2).
  Classes (overlapping; one token can be in several): wrong direction = plan end on the other side of the reference end (decision 88 rule: |y| > 1 m both, opposite sign, gap > 2 m);
  road edge wide = stage-2 DAC failure where the model's own road edge at the departing corner is more than 1 m beyond the map boundary (roadedge_table.pkl of decision 88; computed for the shipped model only, the
  best driver's edge output was not rerun); early clip = DAC failure whose first departure is within 1.5 s and the start is more than 0.5 m or 0.1 rad off the route centreline (decision 88 rule, applied to both stages);
  collisions = NC < 1 or TTC < 1; EP / progress = failing and EP < 0.8, oracle sets EP to 1 on every token with EP < 0.8 (the "term if perfect" convention of q4), or raises it only to the reference token's EP (the `EP at reference` row, the more realistic one);
  stopped or too slow = failing, EP < 0.5 and the plan's 4 s end is under half of the reference's (or v0 < 1 m/s and the end under 2 m); comfort = HC / EC set to 1 on all tokens (share = failing tokens with HC or EC < 1);
  no command = failing, the reference turns (|yaw at 4 s| >= 0.3 rad), the navigation command is left or right, and the plan turns less than half as much or the other way (wrong direction is a subset).
- **HUGSIM 64** (PR #57 controller; the runs of `hugsim/results/derot`, `op_adapt_h/results/hugsim64` and `one_driver`, copied under `results/loss_budget/hugsim_inputs/`). Each scenario of the shipped (`cinque-fixed`) and the best (`it_dw3-s0` + sel3) run gets one class: spin
  (the logged spin flag, priority), else the run's end (bg / fg collision, max_steps, off_route), collisions with route completion >= 0.9 are "route-end crash". Ceilings set the class's scenarios to their best HD over our 14 Cinque-family arms (all runs listed in the json; arms
  on only 10 scenarios count where they exist) or additionally over the LTF, cv and Lebowski arms of the exam, or to 1.0. Best-observed is an oracle per scenario (the best arm is chosen after the fact).
- **B2D** (19 diagnostic routes x seeds 0-3, 76 runs per arm, `vlm_arb/results/vmerge2_runs.csv`; low-scoring dev routes, not a random sample). DS = RC x 0.5^ped x 0.6^veh x 0.65^layout x 0.7^red x 0.8^stop x lane-departure factor (recovered per run; reproduces the logged DS exactly).
  Removing a class sets its factor to 1; "blocked / timeout" lets the run complete (RC = 100) either mechanically (other factors unchanged) or exposure-aware (DS of the same route's completed runs, which carries the infractions the longer route exposes).

## navtest (PDMS; shipped 84.18, best 84.70, human 94.55)

| class | share of failing tokens (shipped) | ceiling, shipped (84.18) | ceiling, best (84.70) | recovered so far by best (shipped minus best ceiling) |
|:--|--:|--:|--:|--:|
| wrong direction (decision 88) | 1% (15) | 0.09 | 0.17 | -0.07 |
| off-road, model road edge too wide (decision 88; shipped only) | n/a | n/a | n/a | n/a |
| early clip (decision 88) | 1% (9) | 0.07 | 0.07 | -0.00 |
| context: every DAC failure (superset of the three rows above it) | 43% (539) | 4.18 | 3.82 | +0.36 |
| collisions (NC < 1 or TTC < 1) | 46% (579) | 3.24 | 3.15 | +0.08 |
| EP / progress (EP set to 1) | 87% (1096) | 7.67 | 7.51 | +0.16 |
| context: EP raised only to the reference token's EP | 87% (1096) | 3.30 | 3.15 | +0.15 |
| comfort (HC, EC set to 1) | 0% (4) | 0.00 | 0.00 | -0.00 |
| stopped or too slow | 21% (263) | 0.90 | 0.74 | +0.15 |
| turn missed, no command (desire off) | 13% (160) | 0.58 | 0.69 | -0.11 |
| context: DDC / TLC failures | 0% (0) | 0.00 | 0.00 | +0.00 |
| **all trajectory classes together** (wrong direction, road edge, early clip, collisions, stopped, no command; replaced by the reference) | - | 4.03 | 3.88 | +0.14 |
| **everything together** (the above + EP set to 1 + comfort set to 1) | - | 11.24 | 10.98 | +0.26 |

**Reading.** The headroom on navtest is progress and collisions, not the three navhard failure classes: EP is the largest single term (raised to the human's own EP it is 3.3 points; the 7.7 for EP = 1 exceeds what the human trajectory itself reaches, so the "everything" row is above the human PDMS gap of 10.4),
collisions 3.2 and DAC failures 4.2. Wrong direction (0.09), early clip (0.07) and comfort (0) are nearly empty on navtest: these are navhard (stage-2) phenomena, not a property of the model on real frames.
Missed turns without a command cost 0.58, and standing still or crawling 0.90. The best driver's ceilings are within ~0.3 of the shipped ones on every row, matching its +0.52 total: adaptation plus selector have not moved any class's headroom on navtest. Classes are overlapping membership sets (collision tokens are 46% of the failing tokens, DAC 43%), so rows do not add up; the joint rows do.

## navhard (two-stage EPDMS; shipped 33.33, best 35.76, PDM reference 52.82)

| class | share of failing tokens (shipped) | ceiling, shipped (33.33) | ceiling, best (35.76) | recovered so far by best (shipped minus best ceiling) |
|:--|--:|--:|--:|--:|
| wrong direction (decision 88) | 7% (172) | 1.92 | 0.88 | +1.05 |
| off-road, model road edge too wide (decision 88; shipped only) | 16% (395) | 5.63 | n/a | n/a |
| early clip (decision 88) | 8% (198) | 3.73 | 2.26 | +1.46 |
| context: every DAC failure (superset of the three rows above it) | 30% (742) | 15.10 | 13.00 | +2.10 |
| collisions (NC < 1 or TTC < 1) | 13% (328) | 2.69 | 2.91 | -0.22 |
| EP / progress (EP set to 1) | 78% (1935) | 9.82 | 8.92 | +0.90 |
| context: EP raised only to the reference token's EP | 78% (1935) | 9.75 | 8.86 | +0.89 |
| comfort (HC, EC set to 1) | 67% (1657) | 5.82 | 5.61 | +0.20 |
| stopped or too slow | 39% (980) | 6.97 | 3.15 | +3.82 |
| turn missed, no command (desire off) | 14% (358) | 3.23 | 3.04 | +0.19 |
| context: DDC / TLC failures | 13% (324) | 2.05 | 2.12 | -0.07 |
| **all trajectory classes together** (wrong direction, road edge, early clip, collisions, stopped, no command; replaced by the reference) | - | 13.69 | 8.59 | +5.09 |
| **everything together** (the above + EP set to 1 + comfort set to 1) | - | 30.51 | 24.29 | +6.22 |

**Reading.** On navhard the lost points are in the drivable-area family: replacing every DAC-failing token by the reference is worth 15.1 points for the shipped driver (13.0 for the best), of which the three decision 88 classes are wrong direction 1.9, early clip 3.7 and road edge too wide 5.6
(the road-edge class is the largest of the three and overlaps early clip). Progress is the next block: EP to the reference's is 9.8 and the stopped / too-slow tokens 7.0 (overlapping), then comfort 5.8
(HC or EC is below 1 on 67% of the failing tokens; the q4 table of decision 88 puts nearly all of it on EC), collisions 2.7 and the missed-turn class 3.2 (wrong direction is a subset of it, so giving the model the command is the broader lever, while the wrong-direction class alone is 1.9). All trajectory classes together are 13.7 for the shipped driver; the sum of the single rows is larger because the classes overlap.
What the best driver already took: wrong direction +1.0 (the selector's job: 172 to 82 wrong-direction tokens) and stopped / too slow +3.8, early clip +1.5;
collisions did not move (2.7 to 2.9) and comfort did not move. Caveats: the substitute is the PDM reference, whose own EPDMS is 52.8 (95% of the failing tokens are stage 2, where the reference starts from the displaced pose);
the road-edge class could not be evaluated for the best driver without rerunning the model's edge output.

## HUGSIM 64 (HD x 100; shipped 27.8, best 34.5; best observed over our arms 46.8, over all arms 55.3)

| class | scenarios, shipped (share of failures) | ceiling to best observed (our arms / + LTF, cv, Lebowski), shipped HD 27.8 | ceiling to HD = 1, shipped | scenarios, best | ceiling to best observed (our / all), best HD 34.5 | ceiling to HD = 1, best | recovered so far (HD = 1 ceiling, shipped minus best) |
|:--|--:|--:|--:|--:|--:|--:|--:|
| stopped / max_steps | 14 (23%) | 8.2 / 10.2 | 17.0 | 8 | 2.2 / 4.8 | 7.9 | +9.2 |
| collision: foreground (fg) | 26 (43%) | 4.3 / 7.8 | 37.2 | 30 | 5.8 / 9.2 | 42.2 | -4.9 |
| spin (heading > 45 deg off route) | 10 (16%) | 4.8 / 7.0 | 10.2 | 4 | 1.4 / 1.9 | 4.0 | +6.2 |
| collision: background (bg) | 4 (7%) | 0.9 / 1.4 | 5.1 | 6 | 1.9 / 3.2 | 6.8 | -1.7 |
| route-end crash (collision at RC >= 0.9) | 1 (2%) | 0.2 / 0.4 | 0.7 | 1 | 0.2 / 0.4 | 0.7 | -0.0 |
| off-road / off-route | 0 (0%) | 0.0 / 0.0 | 0.0 | 1 | 0.0 / 0.3 | 1.2 | -1.2 |
| complete (HD < 1) | 6 (10%) | 0.7 / 0.8 | 2.1 | 9 | 0.8 / 0.9 | 2.7 | -0.6 |
| **all failing scenarios together** | 61 | 19.0 / 27.5 | 72.2 | 59 | 12.3 / 20.8 | 65.5 | +6.8 |

**Reading.** HUGSIM is a collision-and-standing-still board. For the shipped driver 26 of 61 failing scenarios end in a foreground collision (43%), 14 stand until max_steps (23%) and 10 spin (16%). Measured against what some arm already achieved on the same scenario, the largest single ceiling is stopped / max_steps (8.2 points), then spin (4.8) and foreground collisions (4.3);
against the pure HD = 1 bound the foreground collisions dominate (37.2), because the arms run so far have never survived those scenarios (best-observed gain is only 4.3), i.e. the foreground collisions are the largest unexplored headroom and the stopped / spin ones are the largest already-demonstrated headroom.
The best driver has already taken most of the spin and stopped headroom (spin 10.2 to 4.0, stopped 17.0 to 7.9 on the HD = 1 scale) and its failures are now 51% foreground collisions.
Caveats: single runs per cell (same-day reruns reproduce 8 of 10 spin scenarios, so small class counts are within run noise); the spin class takes priority over the end type (8 of the 10 shipped spins end in a background collision, so "bg collision" excludes them); "foreground / background" are HUGSIM's own end codes; `complete (HD < 1)` is a run that reached the end with penalties.

## B2D (DS; drive 66.86, vmerge2 76.08; 19 routes x 4 seeds)

| class | share of failing runs, drive (events) | ceiling, drive (DS 66.86) | ceiling, drive, exposure-aware | share, vmerge2 (events) | ceiling, vmerge2 (DS 76.08) | ceiling, vmerge2, exposure-aware | recovered so far (drive minus vmerge2 ceiling) |
|:--|--:|--:|--:|--:|--:|--:|--:|
| red light | 47% (25) | 8.71 | 8.71 | 28% (10) | 3.41 | 3.41 | +5.30 |
| stop sign | 8% (4) | 1.05 | 1.05 | 0% (0) | 0.00 | 0.00 | +1.05 |
| collision with vehicle | 23% (12) | 5.24 | 5.24 | 56% (21) | 8.65 | 8.65 | -3.41 |
| collision with layout (static) | 17% (9) | 1.37 | 1.37 | 22% (8) | 1.57 | 1.57 | -0.20 |
| collision with pedestrian | 0% (0) | 0.00 | 0.00 | 0% (0) | 0.00 | 0.00 | +0.00 |
| route deviation (outside route lanes) | 8% (4) | 0.07 | 0.07 | 28% (10) | 0.20 | 0.20 | -0.13 |
| blocked / timeout (Agent got blocked, TickRuntime cap): run completes | 36% (19) | 12.57 | 12.91 | 33% (12) | 5.83 | 7.67 | +6.74 |
| all collision classes together | 38% (21) | 6.68 | n/a | 67% (29) | 10.70 | n/a | -4.03 |
| **everything removed** (DS = 100) | 53 failing runs of 76 | 33.14 | - | 36 failing runs of 76 | 23.92 | - | +9.22 |

**Reading.** For `drive` the largest ceilings are blocked / timeout (12.6; 12.9 exposure-aware) and red lights (8.7), then vehicle collisions (5.2); stop sign, layout collisions and route deviation are each at most 1.4.
`vmerge2` has taken most of the red-light (5.3 recovered), stop-sign and blocked headroom (6.7; the paired table of vmerge2.md attributes the gain to the obstacle routes) but vehicle collisions went up from 5.2 to 8.7 (collisions are the remaining lever: 10.7 points for all collision classes), and blocked / timeout is still 5.8.
**Caveat, exposure.** Removing one class can expose another: a run that now completes the route meets the lights, junctions and traffic the cut-off run never reached. The exposure-aware column replaces a blocked run by the mean of its route's completed runs and raises the blocked ceiling only slightly for `drive` (12.6 to 12.9) but by a third for `vmerge2` (5.8 to 7.7). The mechanical rows of the five multiplier classes are first-order: those runs' other infractions are held fixed, but fewer red lights mean fewer stops and later exposures to collisions are not modelled. The 19 routes were picked from earlier low scorers, so shares are specific to this set.

## Summary: levers ranked across boards (ceiling points for the reference driver; the current best driver's remaining ceiling in brackets)

| rank | lever | navtest | navhard | HUGSIM 64 | B2D 19 routes | sum over boards (reference driver) | sum over boards (current best) |
|--:|:--|--:|--:|--:|--:|--:|--:|
| 1 | stopped / too slow / blocked | 0.9 (0.7) | 7.0 (3.2) | 8.2 (2.2) | 12.6 (5.8) | 28.6 | 11.9 |
| 2 | off-road (every DAC failure; B2D: route deviation) | 4.2 (3.8) | 15.1 (13.0) | 0.0 (0.0) | 0.1 (0.2) | 19.4 | 17.1 |
| 3 | collisions (HUGSIM fg + bg + route-end; B2D all classes) | 3.2 (3.2) | 2.7 (2.9) | 5.4 (7.9) | 6.7 (10.7) | 18.0 | 24.7 |
| 4 | EP / progress (EP raised to the reference's) | 3.3 (3.1) | 9.8 (8.9) | n/a | n/a | 13.1 | 12.0 |
| 5 | red light / stop sign (B2D); DDC + TLC (navhard) | 0.0 (0.0) | 2.1 (2.1) | n/a | 9.8 (3.4) | 11.8 | 5.5 |
| 6 | spin / wrong direction | 0.1 (0.2) | 1.9 (0.9) | 4.8 (1.4) | n/a | 6.8 | 2.4 |
| 7 | comfort | 0.0 (0.0) | 5.8 (5.6) | n/a | n/a | 5.8 | 5.6 |
| 8 | turn missed, no command (desire off) | 0.6 (0.7) | 3.2 (3.0) | n/a | n/a | 3.8 | 3.7 |

Rows are sorted by the sum over the four boards of the shipped-driver ceilings; the unit differs per board (PDMS, EPDMS, HD x 100, DS), so the sum is a rough index and the per-board cells are what to read. Row definitions: navhard / navtest "off-road" is the whole DAC-failure family (it contains wrong direction, early clip and road edge, which are therefore not listed again);
HUGSIM numbers use the best-observed-on-our-arms ceiling (the HD = 1 bound is in the HUGSIM table and is up to 9x larger for foreground collisions); B2D red light row adds stop sign; the navhard / navtest cells of "red light / stop sign / DDC + TLC" are the DDC and TLC failures, which are tiny.
EP rows use the reference-EP variant (EP = 1 gives 9.8 on navhard and 7.7 on navtest).

**Ranking in words.** (1) Standing still, crawling and being blocked has the largest sum and is the largest single lever on HUGSIM (8.2) and B2D (12.6); on navhard (7.0) it ranks behind the DAC family and EP, and it overlaps EP (stopped tokens are low-progress tokens), so the two are not additive. The current best has taken much of it on navhard and HUGSIM and about half on B2D. (2) The drivable-area family is by far the largest navhard lever (15.1), and it is a navhard-specific problem (navtest 4.2, HUGSIM and B2D about 0).
(3) Collisions are about 3-7 points on every board and the one lever the best drivers did not reduce (B2D vmerge2 went up). (4) Progress (EP) is 3-10 points on the two NAVSIM boards but a different thing from HUGSIM / B2D completion. (5) Red lights are worth 8.7 on B2D drive and are already mostly taken by vmerge2. (6) Spin / wrong direction is worth 2-5 and the selector already took most of it. (7) Comfort (5.8) is a navhard-only lever (EC on stage 2) and nothing on navtest. (8) The missing command (desire off) is bounded by about 3.2 points on navhard and 0.6 on navtest, with the wrong-direction part already counted above.

## Files

`results/loss_budget/{navtest,navhard,hugsim,b2d}.json` (all numbers, per-class counts), `hugsim_classes_*.csv` (class of every scenario), `hugsim_inputs/` (copied run results), code `scripts/loss_budget_{prep,nav,hugsim,b2d,report}.py`
(prep and nav run on the box in the navsim2 env, ~7 min on 40 cores, CPU only; hugsim, b2d and report run on any machine with pandas). Intermediate pickles stay on the box under `runs/leaderboard_audit/loss_budget/`.
