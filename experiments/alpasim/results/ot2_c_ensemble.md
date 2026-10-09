# alpasim OT2-C: adapter ensemble on the shared frozen encoder (OT30-F-s0 + s1), 700 public scenes

Written 2026-10-09. Pre-registration: [plans/2026-10-09-ot2-ensemble-prereg.md](../plans/2026-10-09-ot2-ensemble-prereg.md) (pushed before any ensemble score). Decision 211.
Code: `lib/ens_driver.py` (`run.sh <dir> ens`, `ENS_TAGS=a+b`), `scripts/ot2_c_chain.sh`, `scripts/ot2_loop.py`, `scripts/ot2_report.py`, `scripts/ot2_ens_diag.py`. Tables: [ot2/](ot2/). Nothing was submitted to AlpaSim.

## Answer

**Not adopted, and the line of work stops (registered stop rule).** Averaging the plans of the two OT30 seeds scores between its members: 0.9312 against 0.9335 for the better seed.

| read-out (700 scenes, 27 logs) | value | registered line |
|:--|:--|:--|
| ENS-OT30 mean scene score | 0.9312 [0.9017, 0.9532] | |
| members OT30-F-s0 / s1 | 0.9258 / **0.9335** | |
| ensemble - better member (s1) | **-0.0022 [-0.0097, +0.0045]** | >= +0.008 and lower bound > 0: **not met** |
| at-fault events, ensemble / better member | 23 / 21 | not higher: **not met** |
| ensemble - mean of its members | +0.0016 [-0.0055, +0.0071] | |
| per-scene best-of-members (hindsight ceiling) | 0.9468, +0.0133 [+0.0056, +0.0221] over s1 | |
| stop rule: point estimate < +0.004 | -0.0022 | **stop**: E2 / E3 are not run |

Zeros: ensemble 39 (collision 12, offroad 11, corridor 16), members 42 / 38; slow scenes 88 (members 93 / 82). CIs: paired per scene, bootstrap over whole logs, 10 000 draws.

## Why averaging gets none of the ceiling

| scenes by the members' own single-driver result | scenes | ensemble zero | scenes with a fork decision |
|:--|--:|--:|--:|
| exactly one member zero (split) | 20 | 9 | 2 |
| both members zero | 30 | 30 | 0 |
| no member zero | 650 | 0 | 20 |

- On the state the ensemble actually drives, the two seeds almost agree: the lateral spread of their 4 s endpoints is 0.05 m at the median, 0.21 m at p90, 0.64 m at p99; fork decisions (> 1.0 m, the registered definition) are 0.4 % of 7 000 decisions and occur in 22 scenes, 1 of them an ensemble zero. The longitudinal spread is 0.18 m at the median.
- So neither registered reading applies. The seeds' failures are not bifurcations that averaging makes worse (3 % of zero scenes and 3 % of non-zero scenes have a fork decision; the ensemble adds no zero where no member failed), and they are not plan-level disagreements that averaging smooths away (it fails on 9 of the 20 split scenes, i.e. it inherits about half of them, which is what picking a member at random gives: mean of members + 0.0016).
- What the numbers say instead: on a shared state the seeds' plans differ by centimetres; whether a rollout ends as a zero is decided by how such small differences grow over the 10 closed-loop steps. The per-scene best-of-two is the maximum of two noisy outcomes of nearly the same policy, not two different competences. Averaging returns that policy once more. This was not tested by a perturbation experiment; it is the reading the disagreement table supports.

Full table of the 39 ensemble zeros with the members' own results and the largest member spread: [ot2/ENS-OT30_diag.md](ot2/ENS-OT30_diag.md). Scores / lines: [ot2/c_ens_ot30_report.md](ot2/c_ens_ot30_report.md), per scene `ot2/c_ens_ot30_per_scene.csv`.

## Latency

| single step (median) | frames (CPU warp) | encode | policy | total |
|:--|--:|--:|--:|--:|
| OT30-F-s0 alone, same process and card (100 calls, random frames, loaded card) | 47.0 | 17.2 | 22.3 | 89.6 ms |
| ensemble of 2 | 48.7 | 19.0 | 40.9 | 109.7 ms |
| ensemble of 2 inside the 700-scene run (7 000 calls, 8 concurrent rollouts) | 61.0 | 13.5 | 30.2 | 110.6 ms |

One more policy pass costs about 20 ms (the policy, not the encoder, is what repeats), +22 %, and takes the step over the 0.1 s target. "Almost no extra latency" does not hold for this policy.

## Gates, deviations, limits

- Health: 7 000 `drive` calls, 7 000 real inferences, 0 errors; members' frozen weights verified identical at start.
- Identity gate, amended reference: the one-member ensemble reproduces the single OT30-F-s0 driver on 8 of 8 pilot scenes when the reference is the c0b pilot run of the same 8-scene list. Against the c0b 700-scene chunk run of the same driver one scene differs (0.9705 vs 0.9771; plans differ by 1.5 cm from decision 0 on with identical ego inputs), and the c0b pilot differs from the c0b chunk run in exactly the same way. The simulator is deterministic for a fixed scene list (decision 199), not across lists. The registered gate named the d201 run; the reference was changed to the same-list run before the 700-scene scores were read. Size of this list-composition noise on 700 scenes: not measured (one sample: 1 scene in 8, 0.0065).
- Limits: two seeds of one recipe; one simulation each; local rendering; the verdict is about equal-weight plan averaging, not about selection (decision 202) or other fusion rules.
- Cost: about 1.0 job-hour (two 8-scene pilots, three 700-scene chunk jobs of 19 min, the latency job).
- Runs (box): `$DATA_DIR/runs/alpasim/ot2/c-ENS-OT30{,-pilot}/`, manifest `c-ENS-OT30/manifest.json`.
