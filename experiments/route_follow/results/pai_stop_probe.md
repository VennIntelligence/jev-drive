# PAI-STOP-PROBE: do the policy's own features carry "the human stops ahead"? (2026-10-18)

**Answer.** Partly, and no better than what the model already reads out. On RSG-L18-s0's own PAI states, a probe on the frozen
tokens separates "the human stops within 5 s of travel" from the other decisions at AUC 0.79. That holds for the no-lead stops too
(stop line / other: 0.786 [0.70, 0.85]), not only for lead stops (0.793). But a linear probe on kinematics plus the model's own lead-head
outputs reaches the same number (0.788). At y5 the image tokens add nothing beyond that (TOK - LEAD -0.002 [-0.08, +0.10]). Only at 3 s, on
stop lines, do the tokens add something (+0.146 [+0.02, +0.28], 6 scenes).

On the 18 d357 class-C scenes (our car drives through the human's rest point), the token probe fires early enough to stop in **8 / 18**.
The threshold sits at 10 % false positives per decision, which would not be usable as a stop trigger. Lead + kinematics gives 7 / 18,
kinematics alone 3 / 18.

**Registered verdict: "readable only for leads"** (prereg `plans/2026-10-18-pai-stop-probe-prereg.md`). The agent stratum is >= 0.75.
The no-lead stratum passes the AUC clause (0.786, lower bound 0.70 > 0.65) but fails the other two: TOK - LEAD has a lower bound below 0, and
8 / 18 is under the 9 required. The label misleads, though. The no-lead stops are as readable as the lead stops. What fails is that neither
is readable well enough, early enough or beyond the existing readouts. Read it as: **a weak cue (AUC ~0.79) exists in the input for both
kinds of stop; it is not a clean signal a curriculum can rely on.**

Code: `scripts/rf_stopprobe.{py,sh}` (feats / probe / report), pool jobs 1018-210602-b303 (rows), -cc4e (cache), 1018-210603-fa13 (feats),
-873a (probe). Box outputs: `$DATA_DIR/runs/route_follow/stop_probe/` (`rows/c18`, `cache/c18`, `feats/`, `probe/{probe.json, scores.npz}`).
Split `physical_ai_av/stop-probe-scenes@v1:d9bdbf9ea4a2`. GPU: about 40 card-minutes (the probe job was mostly CPU bootstrap and logistic fits).

## Setup

**Rows** are 10 Hz decisions from three onp_cache row caches. The frozen Cinque tokens do not depend on the checkpoint, so the three
sources can be pooled.

| source | states | scenes | rows kept | origin |
|---|---|--:|--:|---|
| log | human: the ego follows the log; the serve is not executed | 80 (train160) | 13 571 | `pait/cache_log` (10-11 log-replay pass) |
| p1 | RSG-L18-s0 own, pre-ACC serve | 48 (test281) | 7 675 | `pait/cache/p1rsg` (10-13 Tokyo read) |
| c18 | RSG-L18-s0 own, d330 ACC-GAP serve | 18 (d357 class C) | 1 797 | today's camera rerun `pai_stop_cues/read` (rollout.asl) -> `onp_rows` rows + cold (G-in drive.jsonl half 198/198 per scene; no PAI_DUMP) -> `onp_cache build` |

- **Rows kept:** usable, read by the scorer (before the first event), v0 >= 2 m/s, |e0| <= 3 m. p1 and c18 share 4 scenes, which are
  grouped together.
- **Not available:** no other PAI 281 rows with tokens exist on the box. The rsgacc PAI pools (`on1_PAIA*`) and their row and token caches
  have been deleted. So "own" covers 30 positive scenes and the strata are small.

**Labels.**
- `a_stop` = arc along the human's logged path to the first point where the logged speed is < 0.5 m/s, within 100 m (`path_v`).
- y5 = `a_stop <= 5 v0`; y3 = `<= 3 v0`.
- Cue strata (positives only, in this order):
  - agent = an actor centre within 1.8 m of the path at `[a_stop - 2, a_stop + 20]` m;
  - line = a stop line at `[a_stop - 15, a_stop + 5]` m;
  - other = the rest. "No lead" = line + other.
- Each stratum's AUC is its positives against all negatives.

**Feature sets.** Every set includes KIN = `ego_in` (20).
- LEAD = KIN + RSG's lead-head means (72) and lead_prob (3).
- PLAN = KIN + RSG's plan means (495).
- HID-RSG / HID-JTN = KIN + the checkpoint's `hidden_state` output.
- TOK-lin = KIN + the newest slot's 32 tokens, averaged (512).
- TOK = KIN + the newest slot's 32 x 512 tokens: per-token LN, then a linear map to 16 dims, then an MLP of 256.

**Deviations from the prereg.**
- RSG and JTN were run on cached tokens + `ego_in` without the route memory, which is not in the row caches. So LEAD / PLAN / HID are the
  route-off forward; TOK is unaffected.
- `hidden_state` turned out to be 16 384-d (the queued state), not 512 as registered. It was used as is, and the logistic probe on it
  likely overfits (HID < KIN).
- There are no log ids for PAI clips, so the folds are by scene.

**Probe.**
- Linear sets: standardized logistic regression, C = 0.1, balanced classes.
- TOK: AdamW 1e-3, wd 1e-2, dropout 0.2, 3 epochs, positive reweighting. No tuning.
- 5 folds by sha1(scene); every score comes from a fold model that never saw the scene.
- 95 % CI from a scene-cluster bootstrap, B = 2000 (the paired differences reuse the same draws).

## AUC (y5 = human stop within 5 s of travel; 95 % scene-bootstrap CI)

| source | stratum | pos (scenes) | KIN | LEAD | PLAN | HID-RSG | HID-JTN | TOK-lin | TOK | TOK - KIN | TOK - LEAD |
|---|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| **own** | all | 458 (30) | 0.745 [0.65, 0.81] | 0.787 [0.72, 0.84] | 0.787 | 0.724 | 0.723 | 0.787 | **0.788 [0.71, 0.85]** | +0.042 [-0.05, +0.17] | +0.000 [-0.07, +0.08] |
| own | agent | 86 (10) | 0.849 [0.75, 0.93] | 0.782 [0.62, 0.89] | 0.821 | 0.695 | 0.682 | 0.749 | 0.793 [0.68, 0.89] | -0.057 [-0.17, +0.04] | +0.011 [-0.08, +0.10] |
| **own** | **no lead** | 372 (26) | 0.721 [0.60, 0.80] | 0.788 [0.71, 0.84] | 0.779 | 0.731 | 0.733 | 0.796 | **0.786 [0.70, 0.85]** | +0.065 [-0.04, +0.22] | -0.002 [-0.08, +0.10] |
| own | line | 173 (6) | 0.751 [0.41, 0.85] | 0.802 [0.57, 0.89] | 0.805 | 0.790 | 0.778 | 0.834 | 0.823 [0.71, 0.91] | +0.073 [-0.06, +0.48] | +0.021 [-0.07, +0.27] |
| own | other | 199 (21) | 0.695 [0.56, 0.79] | 0.777 [0.68, 0.85] | 0.757 | 0.680 | 0.693 | 0.763 | 0.754 [0.64, 0.86] | +0.059 [-0.10, +0.26] | -0.022 [-0.14, +0.11] |
| log | all | 220 (14) | 0.899 [0.86, 0.93] | 0.908 | 0.913 | 0.703 | 0.717 | 0.869 | 0.889 [0.83, 0.94] | -0.010 [-0.07, +0.05] | -0.019 [-0.05, +0.02] |
| log | no lead | 210 (14) | 0.908 [0.87, 0.94] | 0.913 | 0.915 | 0.718 | 0.733 | 0.874 | 0.899 [0.85, 0.95] | -0.008 [-0.07, +0.05] | -0.014 [-0.05, +0.03] |
| all | all | 678 (44) | 0.818 [0.76, 0.86] | 0.847 | 0.845 | 0.722 | 0.730 | 0.835 | 0.840 [0.79, 0.88] | +0.022 [-0.04, +0.09] | -0.007 [-0.05, +0.04] |
| all | no lead | 582 (40) | 0.811 | 0.851 | 0.844 | 0.731 | 0.741 | 0.841 | 0.844 [0.79, 0.89] | +0.033 [-0.03, +0.11] | -0.007 [-0.06, +0.05] |

y3 (stop within 3 s of travel), own states:

| stratum | pos (scenes) | KIN | LEAD | PLAN | TOK | TOK - KIN | TOK - LEAD |
|---|--:|--:|--:|--:|--:|--:|--:|
| all | 266 (29) | 0.793 [0.69, 0.86] | 0.793 [0.71, 0.86] | 0.815 | 0.856 [0.77, 0.92] | +0.063 [-0.05, +0.19] | +0.063 [-0.04, +0.16] |
| agent | 47 (8) | 0.930 [0.84, 0.99] | 0.875 | 0.862 | 0.884 [0.83, 0.92] | -0.047 [-0.11, +0.02] | +0.008 [-0.07, +0.21] |
| no lead | 219 (25) | 0.763 [0.65, 0.85] | 0.775 | 0.804 | 0.850 [0.75, 0.92] | +0.087 [-0.04, +0.24] | +0.075 [-0.04, +0.19] |
| line | 79 (6) | 0.784 | 0.766 | 0.811 | **0.912 [0.81, 0.96]** | +0.127 [+0.01, +0.45] | **+0.146 [+0.02, +0.28]** |
| other | 140 (20) | 0.751 | 0.780 | 0.801 | 0.815 [0.69, 0.93] | +0.064 [-0.12, +0.28] | +0.035 [-0.10, +0.19] |

The full table, all sources, both labels and every set with CIs, is in `probe.json` on the box. `python rf_stopprobe.py report` prints it.

What to read:
- **Human states are the easy case and say nothing about the image.** The human brakes before stopping, so kinematics alone gives 0.90.
- **On own states the policy does not brake, and KIN drops to 0.72 on no-lead stops.** Anything image-derived brings that to ~0.79:
  tokens, the lead head, or the plan. The image therefore carries some stop context. But the raw tokens hold no more of it than the
  model's own heads already expose.
- **At 3 s, at stop lines, the tokens are the only set that clearly exceeds the heads** (0.91). That is 79 rows from 6 scenes.
- The lead-like stratum is no easier than the no-lead one for the image sets. KIN is best on agent stops (0.85 / 0.93): the ego is
  already slowing behind the lead under ACC.

## The 18 class-C scenes (d357): first fire before the human's rest point

These are c18 rows with the cross-fitted y5 score.
- Threshold = the fold's training negatives at 10 % FPR.
- Fire = 3 consecutive decisions above it.
- The fire is early enough ("ok") if the distance to the stop point is >= v² / (2 x 3 m/s²).
- "hit" = the share of the y5-window decisions above the threshold.
- Rows before the stop point are few in several scenes (3-16): the v0 >= 2 m/s and before-first-event filters cut most of the approach.

| scene | cue | rows | TOK: d m @ v (need) / hit | LEAD | KIN | HID-RSG |
|---|---|--:|---|---|---|---|
| 054b5901 | other | 7 | never / 0.00 | never / 0.00 | never / 0.00 | never / 0.00 |
| 127c4f85 | other | 3 | never / 0.67 | never / 0.33 | never / 0.00 | never / 0.67 |
| 2466266e | (no y5 row; stop in a turn) | 26 | never | 42 @ 7.5 (9) ok | 60 @ 9.0 (14) ok | 76 @ 10.9 (20) ok |
| 444fca69 | line | 31 | 50 @ 8.9 (13) ok / 1.00 | 26 @ 8.9 (13) ok / 0.27 | never / 0.00 | 43 @ 8.9 (13) ok / 0.87 |
| 44df49fe | other | 10 | 37 @ 10.8 (19) ok / 0.71 | never / 0.29 | never / 0.00 | never / 0.14 |
| 47db2eb6 | other | 12 | never / 0.40 | never / 0.20 | never / 0.20 | never / 0.20 |
| 4cb7f7ac | line | 35 | 50 @ 10.7 (19) ok / 0.50 | never / 0.00 | never / 0.00 | never / 0.11 |
| 4ecc114d | agent | 10 | never / 0.17 | never / 0.00 | never / 0.00 | never / 0.17 |
| 559d4ad6 | other | 16 | 49 @ 11.9 (24) ok / 0.79 | 49 @ 11.9 (24) ok / 0.43 | never / 0.00 | 49 @ 11.9 (24) ok / 0.43 |
| 62cfe9d3 | other | 14 | never / 0.00 | never / 0.00 | never / 0.25 | never / 0.00 |
| 68c3be44 | other | 7 | never / 1.00 | never / 0.00 | never / 0.00 | never / 0.00 |
| 7750265d | agent | 7 | never / 0.60 | never / 0.00 | never / 0.00 | never / 0.20 |
| 7d994ed3 | other | 20 | 42 @ 10.9 (20) ok / 0.75 | 55 @ 10.9 (20) ok / 0.58 | never / 0.00 | never / 0.25 |
| 9b6c68b4 | line | 77 | 42 @ 9.5 (15) ok / 0.86 | 55 @ 10.9 (20) ok / 0.97 | 42 @ 9.5 (15) ok / 0.51 | 63 @ 11.3 (21) ok / 0.73 |
| 9f8acd00 | other | 6 | never / 0.00 | never / 0.00 | never / 0.00 | never / 0.20 |
| ba40be62 | other | 8 | 39 @ 10.6 (19) ok / 1.00 | 39 @ 10.6 (19) ok / 1.00 | never / 0.00 | 55 @ 10.7 (19) ok / 0.67 |
| bc309fd6 | other | 3 | never / 0.00 | never / 0.00 | never / 0.00 | never / 0.67 |
| fc9fde87 | other | 16 | 34 @ 9.3 (15) ok / 0.60 | 65 @ 10.6 (19) ok / 0.40 | 65 @ 10.6 (19) ok / 0.40 | 77 @ 13.0 (28) ok / 0.60 |

Early enough: TOK **8 / 18**, LEAD 7 / 18, KIN 3 / 18, HID-RSG 6 / 18.
- When the token probe fires, it fires 34-50 m before the stop point at 9-12 m/s. That is about 2x the comfortable stopping distance.
- The misses are mostly the "other" stops (no lead, no mapped line), and both "agent" scenes.
- 2466266e has no y5 row: its stop is inside a turn. Its "fires" for LEAD / KIN / HID count toward the totals but are not stop detections.

## Verdict for the curriculum

- **Registered: readable only for leads.** In substance: the cue is weakly present for both kinds of stop. It sits at the level of
  readouts the policy already has (lead head, plan), and it fires in time in under half of the class-C failures at an operating point
  (10 % FPR per decision) that would itself stop the car without cause.
- **What this means for a "stop where the human stops" curriculum.** On PAI, the curriculum would teach a target that the input supports
  only partly: AUC ~0.79, with 10 of 18 failure scenes giving no early signal.
- **The one exception:** stop lines at short range, where the tokens read 0.91 at 3 s. The evidence is thin there (6 scenes). A
  line-only curriculum is the narrow case this result does not rule out.
- **Evidence: weak to medium.** 30 own-state positive scenes, 1 seed, route-off heads, small strata (line 6 scenes, agent 10).
  Positives come from the human log, so some "other" stops may have causes the render does not show.
