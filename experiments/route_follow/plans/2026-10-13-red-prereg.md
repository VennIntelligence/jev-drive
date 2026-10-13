# Pre-registration: RED, a red-light stop pool on nuPlan log states with the B hinge alone, 2026-10-13

Why: STOPLINE's arm RSGB failed (decision 286 follow-up, `results/stopline/read.md`): B was lit on 0.5 % of the on-policy pool rows (84 rows
in 13 scenes), P carried ~0.6 of the loss mass, and the arm learned "drive further" (public GT +0.39 m). STOPLINE-PROBE
([research/next-round/traffic-light-probe.md](../../../research/next-round/traffic-light-probe.md)) showed the frozen tokens see the
ego-lane light on nuPlan (AUC 0.936; moving, no lead 0.920), so the eyes are not the bottleneck; the note's recipe for a trainable B is
nuPlan light labels, red rows sampled to a real share, B only, no P. This file is written, committed and pushed before any training
other than the 60-step smoke. Amendments are appended, never edited in place.

## Pool `on1_RED` (built and gated before this file)

Script [rf_redpool.py](../scripts/rf_redpool.py) (docstring = the exact rules), gates `results/red/pool_gates.json`, figures
`figs/red_sheet.png` (3 red + 3 green rows: BEV + speed profile), `figs/red_labels.png` (label sanity).

- Rows: navtrain log states (LONG-PROBE context, stopline_probe's matching: route -> first traffic-light stop line -> lane connector ->
  the log's light) with the front bumper 0-40 m before the line and the light known at t0: 16 146 candidates.
- **red** (red at t0, never green in the 4 s window, logged driver does not cross on red, stop needs <= 3 m/s^2): 8 113 eligible. Dropped:
  1 183 crossed on red (11.4 %), 253 other non-eligible, 794 turn green in the window, 14 need > 3 m/s^2, 4 target-deceleration exclusions.
  Target: the log's path re-timed (lateral = log; longitudinal only): arc = min(log arc, a constant-deceleration stop with the front bumper
  ~1.05 m before the line), so a queue / lead vehicle the log stops behind is kept.
- **green** (non-red at t0, eligible): 5 403; target = log.
- **Matching**: per cell (distance 0/5/10/20/30/40 m x speed 0/0.5/2/5/10/inf m/s) min(n_red, n_green) rows of each class: exactly 1:1 in
  every cell (distance and speed medians red / green 18.1 / 17.9 m, 2.25 / 2.24 m/s; 53.5 % of each class moving >= 2 m/s).
- **Pool: 2 937 red + 2 937 green = 5 874 rows, 539 logs**, body1 hold logs excluded (they contain op-parity-full-dev).
  **Red share: 50 % of the pool; with `--on-k 14` (14 pool rows of 128 per batch) red rows are 7 / 128 = 5.47 % of every batch's rows**
  (the 5-10 % of the task, read as the share of trained rows; RSGB's B-lit rows were 0.5 % of its pool and 0.05 % of its batch rows).
  A literal "red = 5-10 % of the pool" is not possible with "oversampled": red is already 64 % of the light-known candidates.
- Held twin `on1_REDheld` (hold logs, every eligible row, log targets): 895 red + 540 green, 62 logs; never trained.
- Label gates (pass): target B hinge 0 on 99.9 % of red rows (log 97.5 %), target front bumper never past the line (log 1.8 %), target
  lateral offset from the log path <= 7e-7 m, target arc <= log arc on all rows. **Disclosure: on 92.8 % of the red rows the target equals
  the log within 0.5 m at 4 s** (median difference 0.0 m): the logged drivers already decelerate for the red light within 4 s, so the red
  rows mostly add log imitation on red-light states (they are already ~8 % of navtrain); the new signal is the B hinge on the student's own
  plan where it overruns, and the 1:1 green contrast.

Trainer: no flag added. `--on-mode both` = imitation of `fut` (the re-timed path is a full pose target whose lateral part is the log) +
S's road hinge (road raster of the token; `agent.npz` ok False as in on1_CFT); `--on-lon on_lon` reads `on1_RED/lon.npz` with `items = [B]`
(no P, no A / C): lambda 10 x ho_w 3, gradient along each pose's heading only (timing_only).

Gate R1 (log-state version, `rf_redread.py r1`, RB0-F-s0): (i) the pool rows' inputs are byte-equal to their source navtrain rows (tokens,
ego + route memory, teacher, camera; green targets); (a) plans on pool rows equal plans on the source rows within one fp16 ulp on >= 99.9 %
of rows; controls that must fail: (b) shuffled source rows differ > 0.03 m on >= 95 % of rows, (c) the wrong checkpoint (RSG-L18-s0) on
>= 50 %. Plus a 60-step training smoke (`RREDsmk-L18-s0`): the trainer accepts the pool, logs `on/lonB`, finite loss.

Done before this file (`results/red/r1.json`): R1 passes: 478 pool rows from shard s0, inputs byte-equal, plans bit-equal (max 0.0 m),
controls fail as they must (shuffled 100 %, wrong checkpoint 66.5 % of rows > 0.03 m). The first R1 attempt failed on (i) green targets:
the pool then took the LONG-PROBE context's rebuilt future (<= 0.165 m off the cache); the pool now uses the cache's own `fut` (every count
and label gate unchanged). Smoke (on that first build; format identical): dev ADE 0.5276 at step 0 = the init's, 0.5316 at 60, G-rely
read; B active (`on/lonB_pos`) on 0.7-2.3 % of the batch's pool rows, `lonB_ho` 0.0002-0.001 (x 30 in the loss) against `imit` ~0.64:
B is again small by mass; the red imitation rows carry the arm.

## Arms and controls

| Tag | Recipe |
|:--|:--|
| `RRED-L18-s{0,1}` | `rm_train.py train --tag RRED-L18-s{k} --init RB0-F-s0 --steps 2500 --warmup 1 --lr-scale 0.1 --eval-every 500 --yr-mass 0.1 --label-bank --agent-lam 10 --agent-on-only --shape --ho-w 3 --rm-lr 3e-4 --on on1_RED --on-k 14 --on-mode both --on-lon on_lon --seed {k}` (CFT's gentle schedule; pool job `--train --vram 28 --cpu 6 --ram 70`) |
| control `RNONEG-L18-s{0,1}` (stored) | the same without the five `--on*` flags |
| reference `RSG-L18-s{0,1}` (stored) | the candidate of decision 286 (RB0-F-s0 + S-only on-policy rows `on1_L18RB0`) |

Expectation stated before training: RRED carries no on-policy rows, so off the light-line states it should behave like RNONEG (held 460:
60.5 zeros vs RSG's 34); line (b) against RSG is therefore likely to fail for reasons unrelated to red lights. The causal read of the red
rows is RRED - RNONEG; line (b) is kept as the coordinator wrote it (does the arm stand next to the candidate).

## Lines (seed means unless said; an arm failing one is not a candidate)

**(a) offline, held nuPlan light rows** (`rf_redread.py read` on `on1_REDheld`; groups RRED = s0, s1; RNONEG = s0, s1; RSG = s0, s1; the
log's own numbers as reference; CIs 2 000 log resamples, paired):
- (a1, primary) the matched red - green 4 s plan arc gap on moving rows (per distance x speed cell, weighted by red rows) is more negative
  for RRED than for RNONEG by >= 0.5 m, with the 95 % CI of the difference below 0.
- (a2) stop rate before the line (front bumper never past the line within the plan's 4 s) on red rows that would reach the line at their
  t0 speed (v0 x 4 s >= distance): RRED >= RNONEG + 5 pp.
- (a3) no general slowdown: RRED's mean 4 s arc on moving green rows not shorter than RNONEG's by more than 0.3 m.
(a) passes iff a1, a2 and a3 pass. Reported beside: the same against RSG, the red rows' B hinge mean.

**(b) held 460 at 18 decisions** (`rm_chain.sh submit-read`, common scenes, `held.py`): RRED zeros <= RSG-L18's 34 + 3, mean >= RSG's
0.8666 - 0.010, GT distance <= RSG's 1.140 + 0.05 m (seed noise on held 460: seed pairs differ by <= 1 zero and <= 0.004 mean). RSGB's
+0.39 m GT on the public set is the failure to avoid. RRED - RNONEG reported.

**(c) public 1491 at 10 decisions** (guard method of `tmp/rn/guard.sh` / CFT: 10 `ol_chain.sh readlist` chunks, `ol_confirm.py` against
the stored YR10m10-F-s0 rollouts; s0 only, as CFT): zeros <= RSG-L18's 45 + 5, mean >= 0.9480 - 0.005, GT <= 0.796 + 0.05 m.

**(d) closed loop PAI, the 24 test281 red-light scenes** (`experiments/stopline_probe/results/pai_test281_events.csv`, label red; Tokyo,
same serving as RSG-L18-s0's stored PAI 281 read at gains 20, `rf_tax.py`): "drove past the human's stop" zeros = scenes with score 0 and
`ran_stop_line` in `secondary`; RSG-L18-s0's stored read: **7** of 24 (13 zeros, 12 scenes with any stop line run, mean 0.368).
Line: RRED-L18-s0 fewer than 7 (point; n = 24 is descriptive). Queued through the Tokyo chain agent: a request file
`/data/runs/route_night/RED_READ_REQUEST` (checkpoint tag, box path, scene list); no Tokyo GPU job is started by this lane.

Order: R1 + smoke -> this file pushed -> training s0, s1 (pool) -> held reads (6 jobs per seed), guard s0 (10 jobs), line (a) read, report
job; results in `experiments/route_follow/results/red/`.
