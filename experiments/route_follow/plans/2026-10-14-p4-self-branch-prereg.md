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
