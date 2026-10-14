# Pre-registration: turning school P4, self-branch search on the policy's own junction states, 2026-10-14

Context: [P3](2026-10-14-p3-branch-prereg.md) and its read [results/p3_branch](../results/p3_branch/README.md) (decision 290: a 3 s push of
the policy's own plan lets 10 / 12 PAI turn scenes complete, identity 4 / 12, but the push is scene-specific and the handover came from the
log); the shape read [results/p3_branch/shape.md](../results/p3_branch/shape.md) (decision 293); decisions 284-286; the open-loop P1 read
[results/pai_openloop/p1_diag.md](../results/pai_openloop/p1_diag.md) (at human states the plan already turns and slows: what is missing is
"already fast on your own state, the scene says slow and turn"). Written, committed and pushed before any smoke rollout. Amendments are
appended, never edited in place. Design fixed by the main session; deviations are listed with their reason.

**Question.** On the states the policy itself reaches before a junction, is there a simple branch of its own plan (a short curvature push
and / or a human-speed longitudinal profile) that completes the turn at human entry speed, and does one exist from several handover
distances? If yes for most turns, the completing branches are the demonstration rows (realized trajectory with its realized speed, failing
branches of the same state as negatives) of the turning school's phase 2. This read is coverage only: no rows, no training.

## Facts this design rests on (shape read, decision 293)

- P3's completing branches have the human's geometry (peak curvature 1.13x the human on the same turn) but 1.3x the speed and 1.6x the
  lateral acceleration. The vehicle does not smooth the pushed curvature (driven / served peak curvature 1.00).
- P3's speed cuts compound with the speed-history ratchet (decision 285): fed speed 0.1-1.8 m/s at the end of the window; handed over at
  the entry, they pivot on the spot (curvature 0.20-0.29 at 1-2 m/s). Hence the per-decision scale relative to the plan's own speed (no
  compounding), the crawl floor (never below 0.8 v_e before the entry) and a lateral-acceleration gate on the realized track.
- Human: entry-speed median per topo class on the 36 strict PAI turns: straight_exit 4.6, curve_single 3.9, straight_road 5.3,
  no_straight_exit 4.6, other 4.6 m/s (right turns 4.5, left 5.9; the class value is used, not the side). The human starts braking about
  26 m before the entry (n = 12), so D40 / D30 are the natural handovers; 2.5 m/s^2 is enough from 9 m/s.

## Instrument

- Driver `experiments/onpol_pai/lib/onp_log_driver.py`, new mode **`ONP_CELLS`** (a JSON file; default `<driver log dir>/p4cells.json`),
  pure functions in `experiments/onpol_pai/lib/onp_p4.py` (shared with the offline checks of `rf_p4.py`). With ONP_CELLS unset and no such
  file the driver is unchanged (P2 / P3 behaviour). The n-th session of a scene (start order) runs cell n of the scene's list, so one
  simulator launch with `runtime.simulation_config.n_rollouts = len(cells)` runs every cell once. Cell = {tag, D, src, g, lon, switch}.
  Every drive record gains `p4` {tag, D, src, h, sigma (route station), d_route, v_meas, lat, lon, released, and while the profile is on
  arc_prof, arc_plan, s_k, v_ent_prof}, and `poses_branch` (served) whenever a transform is on; `poses` stays the raw plan.
- Serving = the reference `/data/runs/route_night/r281/rsg` exactly: **RSG-L18-s0**, `JEV_VCONT=1.0 JEV_LEAD=1`, holdlead off, wizard
  `controller.gains.rel_front_steering_angle_weight=20`, `+e2e_challenge=dev`, harmonizer off, CONC 4; Tokyo: `pai_run.sh` with driver image
  `jev-alpasim:p2h10-f-s0-6f3d05d1`, `alpasim-base:0.89.0`; box: `pai_native.sh` (the same stack natively). Added for the instrument (no
  effect on the policy, checked by S1): `runtime.simulation_config.send_recording_ground_truth=true` (log fields, log fallback) and
  `PAI_DUMP=100000` (1 Hz model-input dumps, the rows' G-in gate). Full 20 s rollouts. rollout.asl kept (the rows replay it), mp4s dropped.
- **Route and turn** (`rf_p4.py plan`, `results/p4_self/plan.json`, committed with this prereg): per scene AlpaSim's MAP route generated
  along the logged path (`RouteGenerator.create(rig path, map, MAP, 40 m)`, `generate_route` at the log's 0.5 s poses: the route a
  log-following ego gets; on the 14 scenes shared with shape.md its strict turn has the same exit station and class); the **strict turn**
  (rf_pai_follow: >= 60 deg within 30 m, entry / exit at 10 / 90 % of the change, exit + 10 m inside the log); release = exit + 10 m; the
  route on a 0.5 m grid with curvature (heading over a +-2 m chord, smoothed over 2 m); topo class with the human as the ego
  (rf_shape.py's call of rf_pai_follow.topo); v_e = the class value above; P2's completion target on the log (rf_branch.turn_points).
- **Handover h.** src = self: the policy drives from decision 0 (as the reference); h = the first decision k >= 17 (end of the dev
  preset's force-GT) at which the ego's route station (monotone windowed projection of the plan anchor, held while > 10 m off the route)
  is within D m of the entry and the ego within 10 m of the route, D in {40, 30, 20, 10, 0}. Computed by the driver at run time, logged.
- **Branch family, 9 cells per handover**: lateral gain g in {1, 1.25, 1.5} x longitudinal profile in {raw, ve, ve08}.
  - Lateral: P3's validated push (`branch_poses`: yaw and segment course about the ego x g at the same arc length) for 30 decisions
    (3 s) from h; x2 is excluded.
  - Longitudinal (`onp_p4.profile`), from h until the route station reaches exit + 10 m, then the raw plan: at each decision the
    curvature-limited profile from the measured speed v0 at the ego's station: inside [entry, exit + 10) the limit is
    min(v_target, sqrt(3.3 / |kappa|)) (kappa of the route ahead); a backward pass limits braking ahead of every limit to 2.5 m/s^2; before
    the entry the envelope never drops below 0.8 v_e (no crawl); a forward pass from v0 accelerates <= 1.5 and decelerates <= 2.5 m/s^2.
    v_target = v_e (ve) or 0.8 v_e (ve08). The served plan is the plan's own path re-sampled at s_k times its arc length (P3's s-scale,
    per decision) with s_k = clip(profile arc over 2.0 s / plan arc over 2.0 s, 0.4, 1.0): never faster than the plan, scaled relative to
    the plan's own speed each decision, so it cannot compound with the ratchet the way P3's fixed s did.
  - g = 1 with raw = the identity (the policy alone).
- **Per-scene job** (`rf_p4.sh job`): phase A = the identity cell alone (1 rollout); `reach` = per D from phase A's own stations and its
  zero event (rf_branch's rule): reached (h) or **unreached** (the policy's own rollout ended or zeroed, or was > 10 m off the route, before
  getting within D m of the entry); phase B = one launch with, per distinct reached handover, the 8 non-identity cells (src = self), and per
  unreached distance the 9 cells with **src = log** (ONP_SWITCH-style: the log up to its decision k_log(D) = round(10 t_log(entry - D)),
  then the policy, branch from max(k_log, 17), tagged `src=log`); then the scene's read.

### Deviations from the main session's design, with reasons

1. **27 of the 38 listed scenes.** The 38 (= the first 38 of `/data/runs/route_night/pait/roll.txt`: hard_train's 24 turn scenes + train160's
   14 log-turn scenes; asserted disjoint from hard_exam.txt, exam_turn26.txt and control.txt) were chosen by other turn rules. 11 have no
   strict turn whose exit + 10 m lies inside the log (5: no strict turn; 5: exit + 10 m beyond the log; 1: P2's completion target outside the
   log), so neither the handover nor the completion is defined there. They are listed in plan.json `bad` and not run. **N = 27** (14 hard_train,
   13 train160; 18 right, 9 left; curve_single 11, straight_exit 11, straight_road 3, no_straight_exit 2). Human entry speed on these 27:
   median 4.0 m/s.
2. **h at run time, unreached from the scene's own phase A**, not from r281/rsg: the 13 train160 scenes have no RSG-L18-s0 reference
   rollout, and the box is not bit-identical to Tokyo (docs/alpasim.md: 19 / 22 scenes within 0.01). S1 checks on Tokyo that the policy's own
   rollout under this instrument is the reference rollout.
3. **The identity rollout runs once per scene** (phase A), not once per D: with src = self it is the same rollout for every D.
4. **Distances whose handover decision coincides are run once** (`dup` in reach.json; the larger D keeps the cells). On many of these scenes
   the clip starts within 40 m of the entry: from the log's own timing 4 scenes have one distinct handover (all D at decision 17), 10 have 2,
   3 have 3, 10 have 5 (83 distinct handovers in total). Counting a duplicated distance twice would count one rollout twice, so the line
   below counts **distinct handover decisions**. 4 scenes can therefore not reach ">= 2", so the line's maximum is about 23 / 27.
5. **One simulator launch per phase** (n_rollouts = cells, 4 concurrent) instead of one launch per cell: the start-up (25-45 s) and the
   single-rollout wall (147-219 s alone) would otherwise triple the cost. The output layout is therefore `<root>/<scene>/<id8>-A/`,
   `<id8>-B/` (pai_run / pai_native run dirs) with the per-cell index in `cells_A.json`, `cells_B.json`, `reach.json` and the scene's
   `read.json`, not one directory per cell. The cell of each rollout is in its drive records (`p4.tag`).
6. **Rollout count about 27 + 83 x 8 = 690** (plus one per unreached handover), not 1 710: items 1, 3 and 4.

## Measures (`rf_p4.py read`, AlpaSim image / venv)

Per rollout: tag, h, D, src, score, zero kind and time (rf_branch's rule), rear-ended (collision_rear > 0), **turn completed (P2's rule,
rf_branch: log station at or past P2's target within 2.5 m of the route generated at the ego's poses, before the zero event)**, ego-to-route
distance and fed speed at the end of the lateral window (h + 30), and on the plan route with rf_shape.py's operators (results/p3_branch/shape.md:
0.25 m resampling, heading chord +-0.5 m, kappa / dkappa over +-1 m, 5-sample speed box, window entry - 40 m .. exit, cut at the zero event):
realized entry speed (v_0), min speed inside [entry, exit], peak curvature, peak curvature rate, peak lateral acceleration, peak deceleration;
crawl = min speed between h and the entry < 0.5 v_e; where the rollout ended (station relative to the entry, time, reason); min s_k.

**Shape gate** (a completing branch becomes a demonstration row only if it passes; computed here, used by the row builder): peak curvature
<= the human class p95, peak curvature rate <= the human class p95 (class value when the class has >= 5 human turns, else all turns, as
rf_shape.py), peak lateral acceleration <= 4.42 m/s^2 (human p95, all turns; the 3.3 cap rejects 13 of 36 human turns and is not a gate),
realized entry speed <= class v_e + 1 m/s. Thresholds per scene in plan.json `gate`: straight_exit kappa 0.159 / dkappa 0.036 / v 5.6;
curve_single 0.164 / 0.046 / 4.9; straight_road 0.202 / 0.026 / 6.3; no_straight_exit (all turns) 0.184 / 0.043 / 5.6.

## Lines (coverage read, no training)

- A turn **counts** when it has completing branch cells (any of the 8 / 9 non-identity cells; the identity never counts, it is no branch) at
  **>= 2 distinct handover decisions**, each with realized entry speed <= class v_e + 1 m/s.
- **Pass: >= 60 % of the 27 turns (>= 17).** **Fail: <= 35 % (<= 9).** Between: inconclusive.
- Both src = self and src = log cells count (an unreached distance has no own state); the self-only count is reported next to it.
- Secondary, reported, no line: the three longitudinal profiles' completion counts against each other; per-D completion; src = log counted
  apart; rear-ended counted but not disqualifying (the log's following car does not react); crawl count; the full shape gate's yield
  (turns with gate-passing completions at >= 1 and >= 2 distinct handovers); the identity's completions; entry speed of completing cells
  against the human's; turns whose line is unreachable (one distinct handover).

## Not claimed

27 turns, one seed, one checkpoint, deterministic simulator; a completing cell is one draw of a chaotic closed loop. Coverage of the branch
family, not that rows built from it will train the turn.

## Commands

```bash
# Tokyo, the plan (done before this commit; result committed as results/p4_self/plan.json)
BACKEND=tokyo bash experiments/route_follow/scripts/rf_p4.sh plan /data/runs/route_night/p4/plandev
# Tokyo smoke, frozen clone at this commit, tmux jev window p4smoke (both cards: the card-0 reservation was withdrawn 2026-10-14)
git clone -q ~/mycode/jev-drive /data/runs/route_night/p4/code   # checked out at this commit
tmux new-window -d -t jev -n p4smoke "bash /data/runs/route_night/p4/code/experiments/route_follow/scripts/rf_p4.sh smoke /data/runs/route_night/p4/smoke"
# box, staged launch (repo root, after git pull; DATA_DIR set): 1 scene, inspect; 10 scenes, inspect; all (adds the collect read)
bash experiments/route_follow/scripts/rf_p4.sh fanout $DATA_DIR/runs/route_follow/p4/run 1
bash experiments/route_follow/scripts/rf_p4.sh fanout $DATA_DIR/runs/route_follow/p4/run 10
bash experiments/route_follow/scripts/rf_p4.sh fanout $DATA_DIR/runs/route_follow/p4/run all
bash experiments/route_follow/scripts/rf_p4.sh read $DATA_DIR/runs/route_follow/p4/run     # also the collect job's command
```

Box jobs: `python -m jevdrive.cl fanout --name p4 --arms <id8,...> --vram 20 --cpu 6 --ram 40 --timeout-h 4 --tries 2 --env
CL_HIST_KEY=alpasim-read4` (4 per card, the AlpaSim read packing of 2026-10-13), one job per scene (phase A, reach, phase B, read).
The 30 scenes missing from the box's scene cache were started on 2026-10-14 (`pai_fetch.sh` from hf-mirror, tmux `jev:p4fetch`,
list `$DATA_DIR/runs/route_follow/p4/fetch38.tsv`); a job whose scene is not there fails in its stack.

**Staged-launch checklist** (after 1 and after 10 scenes): every job DONE, phase A / B rollouts = cells, no rollout without metrics;
reach.json present with h increasing as D decreases; per rollout `p4.tag` matches its cell file once; transform on exactly from h (the
S2 checks, `rf_p4.py s2`, on the 10); score / zero rates of the identity rollouts in the range of the r281 RSG reads on these scenes
(Tokyo `rsg_read.json`); disk per scene within the estimate.

## Estimate

About 690 rollouts. P3: 192 rollouts in 3 h 19 min on one 3090 at CONC 4 (62 s per rollout). Box: one stack 84 s per scene at CONC 4
(card shared), one rollout alone 219 s. Per scene job: phase A ~4 min, phase B ~25 cells -> 7 rounds of 4 -> ~10-12 min, reach + read
~3 min: ~20 min. 27 jobs at 4 per card on 3 cards: ~2-3 h, depending on how much 4 stacks per card slow each other (not yet measured at 4).
Restated from the smoke's measured per-rollout time in results/p4_self/smoke.md. Disk: ~0.45 GB per rollout (rollout.asl + dumps) -> ~310 GB.

## Smoke gates (Tokyo, before the box; the runner stops on a failure, a failure is reported, not worked around)

- **S1** (card 0): a174faed and eaba3f6a (P3 scenes, in r281), each run twice in one launch as the identity cell (the policy alone, all P4
  additions on): every served plan bit-identical to `/data/runs/route_night/r281/rsg` at the same (scene, k), tracks and scores identical,
  4 / 4 sessions; plus S2's phase A identity rollout against the reference (3rd scene).
- **S2** (card 1): f9e361aa, D20 only, cells g1.25-raw, g1-ve, g1.25-ve08 through the full job (phase A, reach, phase B, read):
  transform first on at exactly h, with h recomputed offline from the logged stations equal to the logged h; lateral window exactly 30
  decisions (g1.25 cells); the served poses recomputed offline from the logged raw plan, station and speed (s_k by `onp_p4.profile`, poses
  by the instrument's `branch_poses`) within 0.05 m; the profile released (raw plan served again) from the first decision at or past
  exit + 10 m; realized speed at the entry within 1 m/s of the profile's planned entry speed at the last decision before the entry (if the
  profile binds there, s_k < 1; otherwise reported as not binding).
- **S3**: `rf_p4.py read` + `table` on S2's rollouts: every record field above present and in range.

## Amendment 1 (2026-10-14, written before the smoke that is read and before any box run): one rollout at a time per launch, row-builder requirements

**What the first two smoke tries found** (Tokyo, `/data/runs/route_night/p4/smoke_try1`, `smoke_try2`; nothing of them is read):
- Try 1 (code 1ad87410): S2 never started (docker compose rejects upper case in project names: phase dirs are now `<id8>-a` / `<id8>-b`);
  S1 3 / 4 sessions bit-identical. The fourth (a174faed, second session of the scene in the same launch) got 0.3 s of smeared frames at
  k 0-6 (a half-built scene) and its plans differ from k 0.
- Try 2 (code 2918449b, driver holds session n >= 1 of a scene in start_session until session 0 has 10 frames): S1 8 / 8 bit-identical
  (4 concurrent sessions per scene), S2 phase A bit-identical to the reference, but in S2's phase B 2 of 3 sessions failed in the renderer
  (`get_available_cameras` -> `Failed to create backend`, `UnpicklingError` in `torch.load(artifact.checkpoint)`); the same failure in a
  one-scene box job (`$DATA_DIR/runs/route_follow/p4/boxsmoke`, 1 of 2). Cause, from NRE's `serve.py get_backend` on the box rootfs:
  the renderer checks out a free backend per request and builds a new one when none is free; concurrent builds of the same scene share
  the artifact's temporary cache and corrupt each other (a failed build, or a backend built from a partly read checkpoint, the smeared
  frames of try 1). Renderer session init happens before the driver session, so the driver cannot order it. The reference reads never
  run a scene twice at once, which is why they never met it.
- **Change:** every P4 launch runs **one rollout at a time** (`P4_CONC=1`, `n_concurrent_rollouts=1`): one backend per launch, built once.
  The driver-side wait stays (harmless at CONC 1). Throughput comes from many launches in parallel instead: one pool job per scene,
  `--vram 12 --cpu 3 --ram 24` (one renderer backend 4.6 GiB + driver 2.5-3 + physics ~1 GiB), history key `alpasim-p4c1`, so about
  6-7 jobs per card by VRAM. S1 is now 2 sessions per scene (sequential) plus S2's phase A.
- A stack now counts only scored rollouts (a renderer-failed rollout has `error` in its metrics), so a failure stops the job.

**Row-builder requirements** (rf_selfpool.py, commit ee88176e), adopted:
1. rollout.asl and metrics.parquet are kept for every session until the scene's read is written; then the .asl of every NON-completing
   rollout is deleted (metrics, drive.jsonl, controller CSV, start.jsonl, configs kept; `KEEP_ASL_ALL=1` keeps all). Disk: 0.4-0.55 GB per
   .asl; per scene job up to ~25 x 0.5 = 13 GB while it runs; afterwards only completing rollouts (if a third complete: ~690 / 3 x 0.5 =
   ~115 GB) plus dumps (~30 MB per rollout, ~20 GB). Peak with all jobs running ~27 x 13 = ~350 GB. `fanout` refuses below 500 GB free.
2. Served code: box jobs run from a clean worktree at the commit of `fanout` (`<root>/code`, `git worktree add --detach`), Tokyo from the
   frozen clone `/data/runs/route_night/p4/code`; `<root>/code.json` records the served tree, commit and dirty count.
3. Per decision in drive.jsonl (pai_driver's record, unchanged): k, t0, now, infer, anchor, hist, ego, cmd, poses (raw), poses_model,
   rmem, route0, plus `p4` and `poses_branch`; `kind: route` records; per session start.jsonl, the controller CSV, results-summary.json
   (rollout_id), eval-config.yaml, generated-user-config-0.yaml (pai_run / pai_native layout).
4. Per root: plan.json (this format) and the merged read.json (`rf_p4.sh read`, also the collect job): rid, tag, D, src, h, done,
   t_zero (s from the rollout start), score, zero.
5. JEV_VHIST off; PAI_DUMP on.
6. Reader: min speed inside [entry, exit] (`v_min_inside`, rf_shape's speeds) and **crawl inside** (< 0.5 v_e) are reported next to the
   approach crawl, as a secondary read.

**Estimate, restated.** About 690 rollouts, one at a time per launch. Tokyo 3090, one rollout alone ~100 s plus ~45 s start-up; box
6000D one rollout alone ~190 s plus ~25 s start-up (docs/alpasim.md, card shared). A scene job (phase A + ~25 cells + reach / read) is
then ~1.2-1.5 h on the box alone; with ~6 jobs per card the card is the limit, not the job. Restated from the smoke's measured
per-rollout times in results/p4_self/smoke.md; the staged launch (1, then 10 scenes) measures the per-card throughput.

## Amendment 2 (2026-10-14, after the smoke, before any box run): smoke as run, job layout, estimate

- **Smoke as run** (results/p4_self/smoke.md): Tokyo, both cards (the card-0 reservation was withdrawn on 2026-10-14): S1 on card 0, S2 on
  card 1, in parallel, code abb350ae, frozen clone `/data/runs/route_night/p4/code`, root `/data/runs/route_night/p4/smoke`, 13:24-13:34
  JST. S1 5 / 5 sessions bit-identical to r281/rsg; S2 all checks pass (h 44 = recomputed, lateral window 30, served poses within
  0.0002 m of the recomputed rule, entry speed 4.85 / 3.83 vs profile 4.91 / 3.86 m/s, release at exit + 10 m); S3 every field present.
  Extra check of the log fallback (card 1, one cell): pass. Tries 1-4 (two instrument faults, Amendment 1; one port clash) are listed in
  smoke.md and not read.
- **Job layout change:** phase B of a scene is split round-robin over `P4_PAR` launches side by side (box 3, Tokyo 1), each one rollout at
  a time with its own renderer process (the race of Amendment 1 is inside one renderer). Measured on the box (one 3-launch job alone on a
  card): 3 one-cell launches in 264 s (one rollout alone ~210 s), peaks 18.8 GB VRAM, 25.1 GB RSS, 3.3 cores; pool declaration
  `--vram 22 --cpu 4 --ram 32 --timeout-h 8`, history key `alpasim-p4c1`. Tokyo stacks use ports 7600+ (try 4 met another stack on 6501).
- **Estimate:** ~690 rollouts; ~40 min per scene job alone (~60 min for 5-handover scenes); 4 jobs per card by VRAM; with a 1.5-2x slowdown
  at 12 rollouts per card (not measured) **~2.5-4.5 h** for the 27 jobs; restated after stage 10. Disk peak ~160 GB, final ~115 GB.
- **Scene cache:** 15 of the 27 scenes are on the box; the other 12 are being copied (hf-mirror fetch on the box + Tokyo rsync);
  `fanout` submits only cached scenes, so `fanout ... all` is repeated after the copies finish.

## Amendment 3 (2026-10-14 evening, after the coverage read, before any row pool of this round is built): one training round under an inconclusive verdict

**Why.** The coverage line read **10 / 27 = 37 %: INCONCLUSIVE** (results/p4_self/README.md). This prereg says "coverage only: no rows, no
training". The **user decided on 2026-10-14 to run one training round anyway**. A later instruction from the user ("take the longitudinal out")
narrows the rows to the plan's own speed. Everything below is fixed before any pool of this round is built, any checkpoint is trained, or any
read is made.

**Rows (pool `on1_SELF4R`, rf_selfpool.py `build --cells g1.25-raw --d D40,D0`, P4 root `$DATA_DIR/runs/route_follow/p4/run`, its read.json,
plan.json and served worktree `<root>/code`).**
- Completing cells (P2's rule) that pass the full shape gate G-shape as registered (decision 293, the plan's `gate`), with **lateral gain
  1.25**, **handover D in {40, 0}** and **longitudinal profile raw only**. Raw is the plan's own speed with s_k = 1, so no imposed deceleration
  profile enters the targets: the longitudinal part of the target is the policy's own realized speed. Then the row gates (G-crawl as written).
- Cells with src = self and src = log both count. They are reported separately.
- The identity (Did) does not count: it is no branch. D30 / D20 / D10 do not count.
- **Masking the longitudinal part: not available.** The imitation term of `both` rows is pp_train's `dist` on the 8-pose target (x, y, yaw).
  Neither on_train.py nor rm_train.py has a per-row option that keeps only the cross-track part. on_lon.timing_only is the opposite: it keeps
  the along-heading part, and it is used only by the `--on-lon` terms. So **no RSGSELF-lat variant**: that would need new trainer code, which
  the user's instruction rules out. Raw-only rows are the way the longitudinal is taken out.
- **Expected count, and the threshold for "too few".** From the per-branch table of the existing on1_SELF (same builder, same gates, all
  cells; results/p4_self/pool_self/pool_gates_SELF.json):
  - The selection gives 763 rows, 12 branches and 8 turns: 741 rows with src = self, 22 with src = log.
  - By D: D40 7 branches / 476 rows (232d240e 78, 268949d2 43, 8a365ac2 55, a4be61d6 77, c6c01d25 60, ddc3e8df 79, ea468790 84).
    D0 5 / 287 (1e8e939e 22 [log], 232d240e 79, 268949d2 39, 8a365ac2 63, ea468790 84).
  - **Threshold: >= 500 rows and >= 6 turns.** Below it, the **registered fallback** is all three profiles of gain 1.25 at D40 / D0
    (expected 2 088 rows, 30 branches, 9 turns). Since no mask exists, the fallback is unmasked and is reported as the fallback.
  - The rebuilt pool's own count decides.
- **Exam scenes are never rows here.** rf_selfpool refuses hard_exam / control scenes. The 27 scenes were asserted disjoint from hard_exam,
  exam_turn26 and control when the plan was made.
- **Known before the build (pre-check on the existing on1_SELF rows of this selection):** G-route is 98.7 % of rows <= 0.05 m (max 0.055 m),
  below its 99 % line. The fallback is 98.85 % (max 0.17 m). The gates are checked on the rebuilt pool. **A gate that fails by the letter
  stops the round before any training, and is reported.** No gate is loosened here. G-route guards only the route-hinge label (`route.npz`
  poly). This recipe has no `--on-lon`, so it never reads that label, but any waiver is the user's decision, not this amendment's.

**Arms (box, `rf_self_train.sh`, pool jobs only).**
- **RSGSELF-L18-s0, -s1:** RSG-L18's recipe (`--init RB0-F-s0 --steps 2500 --warmup 1 --lr-scale 0.1 --eval-every 500 --yr-mass 0.1
  --label-bank --agent-lam 10 --agent-on-only --shape --ho-w 3 --rm-lr 3e-4`, `--on-k 13 --on-split alpasim_nuplan/onpol-train`).
  - The pool is on1_L18RB0 (`self` rows) + on1_SELF4R (`both` rows) in `--on-mode mixed`, merged as `on1_RSGSELF`.
  - The SELF rows get 30 % of the pool draws (W from rf_self_train.sh merge).
  - Per-scene cap inside the SELF rows: **CAP = 0.20**. The registered default 0.10 is infeasible on 8 scenes (8 x 0.10 < 1); 0.20 is 1.6x
    the uniform 0.125, the same ratio as the dry run's 0.25 on 6 scenes. On the fallback (9 scenes) CAP stays 0.20.
  - The anti-lateral-ratchet guard is as written (rf_selfread.py lat, <= +0.02 m).
- **Same-steps no-rows control: RSGCTL-L18-s0.** This is RSG-L18's recipe with no SELF rows (`--on on1_L18RB0 --on-mode self`, seed 0,
  2 500 steps), trained now on the current code.
  - It is compared with the stored RSG-L18-s0 weights (`on_train.py ident`).
  - If bit-identical, its nuPlan reads are the stored RSG-L18-s0 reads and are not rerun.
  - Otherwise it is read on held 460 and public 1491 like the arms, and its difference from RSG-L18-s0 is the run-to-run noise yardstick.
  - Either way, it gets the PAI read below.
- **Staged launch:** RSGSELF-L18-s0 first (mixcheck, then training). Its training gates are checked before s1, the control and any read:
  - final dev ADE <= RSG-L18-s0's 0.528 + 0.010 m;
  - route reliance, junction masked minus on, >= +0.0125 m (half of RSG-L18-s0's +0.025) with CI lower > 0;
  - loss finite and not diverging; the SELF rows' on/ade falling.

**Reads, with the settings of the stored RSG-L18 reads so they pair.**
- Held 460 @18 decisions: rm_chain submit-read.
- Public 1491 @10 decisions guard: 10 ol_chain readlist chunks.
- Turn completion on held 460 (24 turns, rf_follow.py: completion = crossing exit + 10 m within 2.5 m of the route before a zero event).
- The PAI identity read on the 27 P4 train turns: rf_p4.sh fanout with `TAG=<ckpt> P4_ONLY=none`, so phase A only, the same instrument and
  the same box; P2's completion rule. The reference is RSG-L18-s0's phase A in the P4 root (9 / 27).
- Exam scenes are never read here: hard_exam 131, exam_turn26 and control are not part of the 27. PASS1 / PASS2 on Tokyo come later.

**Lines (RSGSELF-L18 = seed mean of s0 and s1; control = RSG-L18-s{0,1} stored, with RSGCTL-L18-s0 as the reproduction check).**
- **L1, turn completion.**
  - L1a, the 27 PAI train turns (in-sample states for the 8 row turns): **pass** if both seeds complete >= control + 3 turns; **fail** if
    the seed mean is <= the control; otherwise inconclusive. Here the control is RSGCTL-L18-s0's PAI read, and the stored RSG-L18-s0 phase A
    is reported next to it. The 8 row turns and the other 19 are reported apart.
  - L1b, held nuPlan turns, not among the 27: seed-mean completion minus RSG-L18's seed mean, paired with a log-clustered CI. **Pass** if
    CI lower > 0, **fail** if CI upper < 0, otherwise inconclusive (expected at n = 24).
  - L1 passes if L1a passes and L1b does not fail.
- **L2, held 460 zeros and mean score.** The tolerance is the larger of the stored seed spread (zeros |34 - 34| = 0, mean |0.8656 - 0.8676|
  = 0.002) and RSGCTL-L18-s0's difference from RSG-L18-s0 (0 if bit-identical).
  - **Pass** if seed-mean zeros <= RSG-L18 + tolerance and seed-mean score >= RSG-L18 - tolerance.
  - Worse beyond the tolerance with a paired CI that excludes 0: **fail**. Worse beyond it with a CI that includes 0: inconclusive.
- **L3, public 1491 guard (the guard RBSG2 failed in d286).** Both parts are needed for a pass; either failing is a **fail**.
  - (a) Seed-mean at-fault zeros and mean score against YR10m10-F-s0 inside, or better than, RSG-L18's stored CI: at-fault zeros
    <= -1.4 (stored -11.5 [-22.1, -1.4]) and mean score >= +0.0085 (stored +0.0158 [+0.0085, +0.0235]).
  - (b) The ladder tie-in: at-fault zeros <= RSG-L18's 38.5 + 5 and zeros <= RSG-L18's 45 + 5.
- **L4, GT lateral.** On held 460, the seed-mean lateral distance to the GT trajectory (held.py `latgt`) is not worse than RSG-L18's by
  more than 0.05 m. On public, where only dist_to_gt_trajectory is stored, the same 0.05 m applies to the GT distance. **Fail** if either
  is exceeded.
- **What refutes the round.** "Self-branch rows of the policy's own junction states teach the turn" is refuted if any of these holds:
  - L1a fails: the trained checkpoints do not complete more of the very turns their rows came from;
  - L1a passes only on the 8 row turns with no change on the other 19 (memorising, not a skill);
  - L2 or L3 fails: the rows cost nuPlan driving, as RBSG2's did.
  The round counts as a candidate only if L1-L4 all pass. Nothing here goes to the Tokyo exam by itself.

**Estimate.**
- Pool build ~10 min (the on1_SELF build took 9 min).
- Merge < 1 min, mixcheck ~1 min.
- Training: 3 x ~14 min (2 at once after stage 1).
- Reads: 2-3 checkpoints x (6 held + 10 public chunks), ~60-90 min with the 3 cards free; PAI 3 x 27 phase-A rollouts, ~30-45 min,
  run in parallel with them.
- Report ~5 min.
- **Total ~1.5-2.5 h, about 8-12 card-hours.**
