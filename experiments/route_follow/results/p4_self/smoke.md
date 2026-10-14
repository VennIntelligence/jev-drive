# Turning school P4: smoke of the self-branch instrument (2026-10-14)

Prereg: [../../plans/2026-10-14-p4-self-branch-prereg.md](../../plans/2026-10-14-p4-self-branch-prereg.md) (Amendment 1 written before this
smoke; Amendment 2 records it). Runner `scripts/rf_p4.sh`, reader `scripts/rf_p4.py`, driver `experiments/onpol_pai/lib/onp_log_driver.py`
(`ONP_CELLS`) with `onp_p4.py`, plan `plan.json` (27 turns). Files here: `smoke_s1.txt`, `smoke_s2.txt`, `smoke_table.md`, `smoke_read.json`,
`smoke_log.txt`, `smoke_code.json` (Tokyo, the gated run), `box_smoke_read.json`, `box_smoke_log.txt` (GPU box, path check).

**All three gates pass on the final run** (Tokyo, code abb350ae, frozen clone `/data/runs/route_night/p4/code`, root
`/data/runs/route_night/p4/smoke`; S1 on card 0, S2 on card 1, in parallel, 13:24-13:34 JST). Four earlier tries found two instrument
faults (both fixed before this run; Amendment 1): see "Tries" at the end.

## S1: the policy alone under the instrument = the reference rollout

a174faed and eaba3f6a, each run twice in one launch (one rollout at a time) as the identity cell with every P4 addition on
(ONP_CELLS, `send_recording_ground_truth=true`, PAI_DUMP), plus S2's phase A (f9e361aa), against `/data/runs/route_night/r281/rsg`:

```
a174faed session 4b8d346a cell id: decisions 198 (ref 198), plan max|diff| 0.000000; track rows 198/198 max pos diff 0.000000; score 0.0 / 0.0: IDENTICAL
a174faed session 9ecbf436 cell id: decisions 198 (ref 198), plan max|diff| 0.000000; track rows 198/198 max pos diff 0.000000; score 0.0 / 0.0: IDENTICAL
eaba3f6a session d204cd0a cell id: decisions 198 (ref 198), plan max|diff| 0.000000; track rows 198/198 max pos diff 0.000000; score 0.0 / 0.0: IDENTICAL
eaba3f6a session 054de7a0 cell id: decisions 198 (ref 198), plan max|diff| 0.000000; track rows 198/198 max pos diff 0.000000; score 0.0 / 0.0: IDENTICAL
S1: 4 / 4 identity sessions bit-identical to the reference
f9e361aa session 5d189c2e cell id: decisions 199 (ref 199), plan max|diff| 0.000000; track rows 199/199 max pos diff 0.000000; score 0.0 / 0.0: IDENTICAL
S1: 1 / 1 identity sessions bit-identical to the reference
```

**Pass: 5 / 5 sessions bit-identical** (every served plan, the controller track, the score). The reader also checks every self branch's
plans before h against its scene's phase A: 3 / 3 identical here (`pre_h_diff` 0.0), and 3 / 3 on the box.

## S2: the transform (f9e361aa, D20, cells g1.25-raw, g1-ve, g1.25-ve08, full job: phase A, reach, phase B, read)

`smoke_s2.txt`, verbatim (the cell's entry speed is the controller's speed when its route station first crosses the strict entry; the
profile's is `v_ent_prof` of the last profile decision before the entry):

```
f9e361aa D20-self-g1-ve: {"D": 20, "s_k_last_before_entry": 0.884587, "src": "self", "h_logged": 44, "h_recomputed": 44, "transform_on": "44..198 (155)", "first_on_equals_h": true, "lateral_decisions": 0, "lateral_contiguous": false, "lon_decisions": 155, "released_at_k": null, "sigma_at_release": null, "s_release": 80.0, "raw_after_release": true, "max_station_recompute_diff_m": 0.0001, "max_served_pose_diff_m": 0.0002, "max_s_k_diff": 6e-06, "v_entry_realized": 4.854706046757005, "v_entry_profile": 4.9093, "score": 0.0}
f9e361aa D20-self-g1.25-raw: {"D": 20, "src": "self", "h_logged": 44, "h_recomputed": 44, "transform_on": "44..73 (30)", "first_on_equals_h": true, "lateral_decisions": 30, "lateral_contiguous": true, "lon_decisions": 0, "released_at_k": null, "sigma_at_release": null, "s_release": 80.0, "raw_after_release": true, "max_station_recompute_diff_m": 0.0001, "max_served_pose_diff_m": 0.0001, "max_s_k_diff": 0.0, "v_entry_realized": 11.396421279379316, "v_entry_profile": null, "score": 0.0}
f9e361aa D20-self-g1.25-ve08: {"D": 20, "s_k_last_before_entry": 0.872941, "src": "self", "h_logged": 44, "h_recomputed": 44, "transform_on": "44..130 (87)", "first_on_equals_h": true, "lateral_decisions": 30, "lateral_contiguous": true, "lon_decisions": 87, "released_at_k": 131, "sigma_at_release": 80.3917, "s_release": 80.0, "raw_after_release": true, "max_station_recompute_diff_m": 0.0001, "max_served_pose_diff_m": 0.00015, "max_s_k_diff": 7e-06, "v_entry_realized": 3.8254422768690612, "v_entry_profile": 3.8551, "score": 1.0}
```

| gate | result |
|:--|:--|
| transform on exactly from h | h = 44 logged = 44 recomputed offline from the logged stations (route station within 20 m of the entry); first transformed decision 44 in all 3 cells |
| lateral window exactly 30 decisions | g1.25 cells: 30 contiguous decisions (44..73); g1-ve: 0 (g = 1) |
| profile arcs match the rule (0.05 m) | served poses recomputed offline (`onp_p4.profile` on the logged station / speed, `branch_poses` from the instrument's source): max 0.0002 m; s_k max diff 7e-06; station recompute 0.0001 m |
| realized speed at the entry within 1 m/s of the profile | g1-ve 4.85 vs 4.91 m/s; g1.25-ve08 3.83 vs 3.86 m/s (profile binding: s_k 0.88 / 0.87 there) |
| raw plan from exit + 10 m | g1.25-ve08 released at k 131 (station 80.39 >= 80.0 = exit + 10), no `poses_branch` after it; g1-ve zeroed (corridor) at 12.5 m past the entry before reaching exit + 10, so the profile stayed on to the end, as specified |

**Pass.** g1.25-raw (lateral push only) enters at 11.4 m/s like the identity and leaves the corridor 2.9 m past the entry; g1-ve (profile
only) enters at 4.9 m/s and leaves the corridor 12.5 m past the entry; the pair g1.25-ve08 completes the turn (score 1.0, entry 3.9 m/s,
peak curvature 0.095 / m, a_lat 1.54 m/s^2, passes the shape gate). One state, three cells: illustration only, no claim.

### Log fallback (src = log), extra check

Not in the registered S2 (no distance of the smoke scenes is unreached), checked the same way on Tokyo card 1 with one hand-made cell
(`/data/runs/route_night/p4/logtest`, f9e361aa, D20-log-g1.25-ve08, switch at the log decision of entry - 20 m = 47; `smoke_logfallback.txt`):
the driver returns the log at k 0..46 (`ret` log, no transform) and the policy's plan from k 47; h = 47 logged = recomputed; lateral push
47..76 (30 decisions); profile released at k 137 (station 80.19 >= 80.0); served poses recomputed within 0.00017 m; entry 3.82 vs profile
3.85 m/s; score 1.0. **Pass.**

## S3: the reader

`rf_p4.py read` + `merge` + `table` on the smoke root: 4 rollouts read (phase A identity + 3 cells), every field present
(`smoke_read.json`): rid, tag, D, src, h, score, zero, t_zero (s), rear, done, t_tgt, s_short, end / end_rel_entry / end_t, v_entry,
v_min_inside, v_min_pre, shape_kmax / alat / dk / dec, gate_parts, gate, crawl, crawl_inside, w_droute, w_v, w_zeroed, s_k_min, reach,
pre_h_n / pre_h_diff, wait_s. Values (tag: h, score, outcome, done, v_entry, v_min_inside, kappa, a_lat, gate, w_droute, w_v, s_k_min):

| tag | h | score | outcome | done | v entry | v min inside | peak kappa | peak a_lat | gate | route dist at h+30 | fed v at h+30 | min s_k |
|:--|--:|--:|:--|:--|--:|--:|--:|--:|:--|--:|--:|--:|
| id | - | 0.0 | corridor at 6.5 s | no | 11.36 | 11.39 | 0.001 | 0.16 | no | - | - | - |
| D20-self-g1.25-raw | 44 | 0.0 | corridor at 6.5 s | no | 11.37 | 11.39 | 0.001 | 0.17 | no | 8.03 | 11.69 | - |
| D20-self-g1-ve | 44 | 0.0 | corridor at 9.9 s | no | 4.94 | 4.57 | 0.068 | 1.56 | no | 0.75 | 4.72 | 0.775 |
| D20-self-g1.25-ve08 | 44 | 1.0 | end | yes | 3.90 | 3.58 | 0.095 | 1.54 | yes | 1.01 | 3.89 | 0.751 |

Pass (values in range; the row builder's fields are there: drive.jsonl records keep k, t0, now, infer, anchor, hist, ego, cmd, poses,
poses_model, rmem, route0, p4, poses_branch and the `kind: route` messages; start.jsonl, controller CSV, results-summary.json, eval-config,
generated-user-config per launch; rollout.asl kept only for the completing rollout after the read; PAI_DUMP dumps present).

## Timing and the box

| | Tokyo 3090 (smoke) | GPU box 6000D (path check) |
|:--|--:|--:|
| phase A (start-up + 1 rollout) | 175 s | 266 s (233 s / 307 s in the earlier box tries) |
| one rollout, one at a time in a launch | ~99 s (3 cells in 343 s incl. ~45 s start-up) | ~210 s (3 cells in 654 s in one launch, box try 2) |
| 3 launches side by side, 1 cell each | - | 264 s for all three (incl. 2 x 20 s stagger): scales ~3x |
| job peaks (3 parallel launches) | - | VRAM 18.8 GB, RSS 25.1 GB, 3.3 cores (one launch alone: 6.5 GB, 8.8 GB, 1.2 cores) |

Box path check: `$DATA_DIR/runs/route_follow/p4/boxsmoke3` (1e8e939e, D30, the same 3 cells, served from a frozen worktree at 94cd14dc,
`code.json`): all launches complete, reach / read written, every self branch's plans before h identical to phase A (3 / 3, `pre_h_diff` 0),
rollout.asl removed for the 4 non-completing rollouts. The identity and the 3 branches all collide 9-10 m before the entry, as on Tokyo's
reference; it is a path check only.

## Estimate for the box run

27 turns, about 83 distinct handovers (log timing; the policy's own may differ), so about 27 + 83 x 8 = ~690 rollouts (+1 per unreached
handover). One job per scene: phase A (~4.5 min), then its ~25 cells over 3 launches side by side (~9 rounds x 210 s = ~32 min), then
reach / read (~1-2 min): ~40 min alone, ~60 min for the 5-handover scenes (41 cells). The pool packs jobs by the measured 19 GB: 4 per card
(12 at once on 3 cards = 36 rollouts at once). How much 12 rollouts on one card slow each other is not measured (the 3-launch check ran
alone on its card); at a 1.5-2x slowdown the 27 jobs take **~2.5-4.5 h**. The staged launch measures it: stage 1 (1 scene) gives the
uncontended job time, stage 10 the contended one; the ETA is restated then.

Disk: ~0.5 GB per rollout while a job runs (rollout.asl ~0.45 GB + dumps ~30 MB), then only the completing rollouts keep their .asl:
peak ~12 jobs x ~13 GB = ~160 GB, final ~115 GB if a third complete. Box free 1 502 GB at 12:35 box time; `fanout` refuses below 500 GB.

Scene cache: 15 of the 27 turn scenes were in the box cache at 12:35 box time; the other 12 are coming in two ways at once: `pai_fetch.sh`
from hf-mirror (box tmux `jev:p4fetch`, list `$DATA_DIR/runs/route_follow/p4/fetch38.tsv`) and a Tokyo -> box rsync of the same files in
reverse order (Tokyo tmux `jev:p4copy`, `/data/runs/route_night/p4/copy_missing.sh`, log `copy_missing.log`). `fanout` submits only scenes
already in the cache; rerun `fanout ... all` once the copies are done (it skips submitted scenes).

## The box launch (main session)

```bash
ssh autodl
cd ~/data/jev-drive && git pull
# stage 1: one scene (uncontended job time), inspect against the prereg's staged-launch checklist
bash experiments/route_follow/scripts/rf_p4.sh fanout $DATA_DIR/runs/route_follow/p4/run 1
# stage 10: nine more, inspect (contended job time, restate the ETA)
bash experiments/route_follow/scripts/rf_p4.sh fanout $DATA_DIR/runs/route_follow/p4/run 10
# all: the rest that is in the scene cache, plus the collect job (rf_p4.sh read -> run/read.json, table.md, summary.json);
# repeat once the scene copies have finished (only unsubmitted scenes are submitted; a second collect is harmless)
bash experiments/route_follow/scripts/rf_p4.sh fanout $DATA_DIR/runs/route_follow/p4/run all
```

The first `fanout` makes the serving worktree `$DATA_DIR/runs/route_follow/p4/run/code` at the box's HEAD and writes `run/code.json`;
the jobs serve from it (rf_selfpool `--served` is that path). Jobs are named `p4-<scene id8>` (pool: `--vram 22 --cpu 4 --ram 32
--timeout-h 8`, history key `alpasim-p4c1`); per job log `run/pool/<id8>/`, per scene `run/<scene>/{reach.json, cells_*.json, read.json,
DONE | ERROR}`, run log `run/log.txt`.

## Tries (not read)

1. 1ad87410: S2 never started (docker compose rejects upper-case project names; phase dirs are now `-a` / `-b<i>`). S1 3 / 4: the second
   a174faed session in the same launch got 0.3 s of smeared frames (a half-built renderer scene) and diverged from k 0.
2. 2918449b (driver holds session n >= 1 of a scene until session 0 has 10 frames): S1 8 / 8 identical, but S2's phase B lost 2 of 3
   sessions to the renderer (`Failed to create backend`, `UnpicklingError` in `torch.load(artifact.checkpoint)`), as did a box job.
   NRE builds one backend per concurrent request of a scene and concurrent builds of one scene corrupt each other. Fix (Amendment 1):
   one rollout at a time per launch; several launches side by side for throughput.
3. f43ad906: all gates passed (the same numbers as the final run); rerun only because the job layout then changed (parallel launches).
4. 94cd14dc: S2's runtime found another stack's renderer on port 6501 (host networking on Tokyo); Tokyo stacks moved to ports 7600+.
