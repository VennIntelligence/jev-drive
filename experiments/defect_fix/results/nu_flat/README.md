# NU-FLAT: results, 2026-10-20

Plan: [plans/2026-10-20-nu-flat-prereg.md](../../plans/2026-10-20-nu-flat-prereg.md) (registered 21:40 CST, before any output). Code:
`scripts/nu_flat.{sh,py}`; `rm_train` unchanged. There are three continuations of the nuPlan candidate B2 `STANDFMXH-L18-s1`. Each uses its stored
training command, with only the tag, `--init`, `--steps` and `--warmup` changed. The arms are CTRL `B2C-L18-s1`, F `B2F-L18-s1` (+ `--flat-brk flat_onset.txt --flat-p 0.5`)
and FR `B2FR-L18-s1` (F + `--on-stand-roll 3.0`). Each ran 3 000 steps (the registered rule: 101 499 / (0.75 x 45)), with warm-up 150 and seed 1.
All reads are native: the 18 arena held chunks, held 460 @18, and wait set v2. The full gate table is [gate.md](gate.md) (Mac, frozen pool).

**Verdict: F and FR both FAIL, and both fail on the binding launch guard. Neither reaches PCS >= CTRL + 15.**
- F: flattening moves nothing beyond noise on nuPlan. Arena +0.0012 [-0.0016, +0.0039] and PCS +7 against CTRL. Held 460 zeros 39 -> 37 and front collisions 5 -> 4.
  Wait-set b launches 28 -> 26 / 48 (other_lead) and 14 -> 13 / 35 (red_queue). That misses the 0.95 line by one launch, which is within noise.
- FR: the STAND-R red-line fix carries over to the full model. Red-line easy zeros 10 -> 3 (7 fixed, 0 new), arena zeros 98 -> 77 of 2 228.
  Flattening does not offset STAND-R's launch cost, though. Launches fall to 21 / 48 and 9 / 35, against 28 and 14.
  PCS is 1386 -> 1290 (-96), held 460 is -0.033 [-0.056, -0.018], and free-road ratchet / slow easy partials go 5 -> 14.
  The arena mean is level (+0.0009) because the new slow partials cost about what the fewer zeros win.
- CTRL against the stored B2-s1 (the continuation effect itself): arena -0.0003 [-0.0055, +0.0045], PCS 1398 -> 1386, zeros 101 -> 98.
  Wait-set b launches go 31 -> 28 / 48 and 17 -> 14 / 35. By the guard line, re-warming the recipe for 3 000 steps already costs launches.

| read (18 arena chunks unless noted) | B2-s1 (stored) | CTRL | F | FR |
|:--|--:|--:|--:|--:|
| arena held native mean / zeros (2 106 frozen items) | 0.9365 / 96 | 0.9365 / 92 | 0.9377 / 88 | 0.9374 / 74 |
| frozen-pool PCS | 1398 | 1386 | 1392 | 1290 |
| easy misses (red-line easy zeros) | 47 (9) | 48 (10) | 46 (11) | 51 (3) |
| 716-window subset: easy red-line zeros | 7 | 9 | 9 | 3 |
| held 460 mean / zeros / front collisions | 0.8527 / 41 / 4 | 0.8496 / 39 / 5 | 0.8548 / 37 / 4 | 0.8170 / 38 / 5 |
| wait b launched other_lead / red_queue (of 48 / 35) | 31 / 17 | 28 / 14 | 26 / 13 | 21 / 9 |
| wait a held other_lead / red_queue (of 36 / 39) | 34 / 37 | 34 / 37 | 34 / 37 | 35 / 39 |

Paired arena held native (2 228 windows, log-clustered 95 % CI): F - CTRL +0.0012 [-0.0016, +0.0039], F - B2-s1 +0.0009 [-0.0034, +0.0051],
FR - CTRL +0.0009 [-0.0055, +0.0086], FR - B2-s1 +0.0006 [-0.0058, +0.0075]. AF-front windows: B2-s1 2, CTRL 4, F 3, FR 2.

## Reading
- The PAI lead-brake gain (d409, +0.15 on lead_break) does not show on nuPlan's closed-loop reads. Front collisions and AF-front are already
  at 2-5 windows on B2, so there is little for flattening to fix here. The onset shortcut matters on PAI's late, fast approach states, which navtrain rarely has (d407).
- The STAND-R trade carries over at the full-model scale and has the same sign as d399 / d400: the red line is fixed and launches are lost.
  On this model the trade loses PCS, because the launch and slow partials weigh more than the red-line zeros.
- Every arm, CTRL included, launches less than the stored B2-s1 on the wait set. Any later continuation of B2 needs its own CTRL; the stored B2-s1 reads cannot serve as the control.
- Not run: lead_break / inertia. Both scripts run on PAI states only, and there is no nuPlan-state entry point.

## Files
- [gate.md](gate.md) / gate.json: the registered lines per arm (Mac; `tmp/pcs_local/.venv/bin/python scripts/nu_flat.py gate`, decisions from the box's `ext_full`).
- s17table_{C,F,FR}.{md,json}: stand.py tables (subset arena guards, wait set, held 460; C against B2-s1, F / FR against CTRL); arena_full_*.json, arena_sub_*.json,
  drift_*.json, held.json, arena_windows.csv.gz (every window score).
- identity*.json, smoke.json, steps.json, cmd_diff.txt: the pre-checks (all passed); costs.{md,json}.

## Cost
About 19.8 job-hours, or about 5.3 card-hours: 3 trainings of about 8 min each (0.47 h), arena 12.4 h, held 460 3.6 h, wait set 3.1 h (including
B2-s1's reference read), and identity / smoke / summary 0.2 h. Wall time was 21:42-23:10 CST. Pool priority was 4 / 5. From the coordinator's message (about 21:55)
until all 18 ROUND2 trainings and PAI-FIX chunks had started (23:02), the queued jobs were held on cards 2-4. The guard never had to re-narrow a job.
