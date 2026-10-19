# alpasim_proxy: a local selection set that orders like private

status: live
decisions: 262, 325, 374
index: PCS-LOCAL: board-mix zoib refit gives JTN-s1 1251 / stop3 1333 vs garage 1553 (board 1325 / 1463 / 1543); slow-progress partials cost stop3 ~145 PCS; ARENA running (7 741 uniform windows over the reconstructions, RSG / NR1F + 6 anchors); no public reweighting orders 5 private anchors (frozen 0/84); public GT distance does (tau 0.6); SCENE-MIX: go-straight public 0.52 vs private 0.80, private is waiting-rich; WAIT-SET: 312 windows where the human waits, any tag in one pool job; it separates APY (28 / 146 static zeros, others 1-7) but public + the zero rate does not order the 6 anchors like private (tau -0.47 public, +0.20 best fit, LOO -0.33)

**Question.** Is there a reweighting or stratum of the 1491 public AlpaSim nuPlan scenes that reproduces the private leaderboard ordering and
gaps of policies whose private numbers are known, so the last official slot can be chosen on it?

**Conclusion.** No stratification or reweighting of the public scenes is a valid stand-in for the private suite: proxies fitted on the two submissions alone order none of the three new anchors right (0 of 84 frozen proxies pass), and refits on all five anchors pass only with 20-70 effective scenes and fail leave-one-anchor-out (at most 3 of 5). The gap is driver-specific, not scene-mix: the shipped LTF and DiffusionDrive samples keep their public numbers on private (score -0.09 / -0.03, at-fault km x0.92 / x1.10), the garage 0014 checkpoint gains (+0.11), our openpilot students lose (-0.13 / -0.30); of all public statistics only GT distance (and the share of clips leaving the path by 4 m) ranks the anchors like the private score (Kendall tau 0.6, one discordant family pair) and the private GT distance (tau 0.6, the one real discordance is APY against DD), while public scene score and at-fault distance rank them backwards (tau -0.6) (decision 262).

**Next.** The last slot is chosen by the rule in research/next-round/plan.md section 2.3 (public GT distance within the openpilot family, a warm-up read by the user, the long-horizon read when it lands). Untested hypothesis for the family-specific loss: the openpilot input path (calibration-driven warp) on private rigs; the samples and the garage read raw images.

**Read more.** Pre-registration and Amendment 1: [plans/2026-10-12-proxy-prereg.md](plans/2026-10-12-proxy-prereg.md). Context: the AUD1
audits in [../alpasim/results/](../alpasim/results/) (`aud1_private_gap.md`, `aud1_drift_by_decision.md`). Scene length is owned by the
long-horizon lane ([../alpasim_long/](../alpasim_long/)); this topic takes its length stratum from there and does not run long scenes.

<!-- files:begin -->
## Files

- `2026-10-12-proxy-prereg.md` (plans): PROXY 预注册：一个行为像 private …
- `proxy_fit.py` (scripts): can a reweighting of the 1491 public …
- `proxy_transfer.py` (scripts): which PUBLIC statistic orders the …
- `proxy_collect.py` (scripts): per-scene and per-decision tables of …
- `scene_mix.py` (scripts): human-state statistics of 5.5 s windows …
- `wait_set.md` (results): a local closed-loop set where the …
- `wait_set.py` (scripts): a local closed-loop set where the …
- `wait_set.sh` (scripts): the local closed-loop set where the …

[results/](results/) 71 result files · [plans/](plans/) 3 live plans · [scripts/](scripts/) 18 entry points
<!-- files:end -->

## SCENE-MIX: the private suite is rich in scenes where the human waits (decision 325)

Pre-registration and amendments 1-3: [plans/2026-10-16-scene-mix-prereg.md](plans/2026-10-16-scene-mix-prereg.md); results
[results/scene_mix/](results/scene_mix/) (`n1.md`, `n2.md`); script `scripts/scene_mix.py` (`extract` on the box, `report` / `sample` / `n1` on the Mac).

- **N1.** The host's go-straight is the AlpaSim starter kit (straight along the heading at the current speed, stays at standstill), run
  unmodified on a 300-scene log-stratified public sample (seed 20261016) plus the 25 near-stationary scenes, one pool job (1824 s): public
  **0.522 [0.465, 0.583]** against private 0.799 (line <= 0.70: supported). Its zeros are geometry (offroad 49, corridor 30, collision 18 of
  99). On the 25 near-stationary scenes it scores 1.000 (no progress penalty within 5.5 s), YR 0.920, APY 0.760 (collisions).
- **N2.** 70 052 uniform 5.5 s windows on the 136 raw test logs against the 1491 public windows (a public scene starts 1.5 s before its
  central token): near-stationary start 0.328 vs 0.016 (x20), human static the whole window 0.301 vs one scene, ends stopped 0.354 vs
  0.017, cruise 0.153 vs 0.111. On the one public static scene every openpilot-family driver (and LTF, DD) scores 0 with a rear-end
  collision; garage, WA-JEPA and go-straight pass.
- **Reading.** Drivers that hold at standstill gain on private (go-straight +0.28, garage +0.11), drivers that launch lose (DD -0.03, LTF
  -0.09, YR -0.125, APY -0.295). Size is not measurable from public (the static stratum has one scene); LTF / DD losing only a little bounds
  how static-rich private can be. This narrows decision 262's "driver-specific, not scene mix" to an interaction of mix and launch behaviour.

## PCS-LOCAL: the leaderboard's Policy Capability Score on the arena held split (2026-10-19)

[results/pcs_local.md](results/pcs_local.md); code `scripts/pcs_local.py`; artifacts `results/pcs_local/`. Drive-irt 82ddd5a `BetaIrt`
(zoib 2PL) with evaluate.py's settings, on 33 complete subjects x 2106 held windows. Fitted on our own reads (29 of 33 are our family),
it inverts the board (tau +0.29 over 8 subjects with a private PCS); a pool with the board's composition (garage x30, LTF x10, DD x5,
go-straight x2 + our 4 board entries) reproduces garage 1553 > JTN-s1 + stop3 1333 > native 1251 (board 1543 / 1463 / 1325). Under zoib, a
strong subject's partial score on a window the strong pool aces costs more than a zero: stop3's 105 slow-progress partials (progress < 0.8 of
the human on moving windows, 0.008 of mean) cost about 145-220 PCS, all its zeros 23-40, launch hold partials 18-27, static windows 0. Selector:
keep the arena mean, add the frozen board-mix PCS (`pcs_local.py score`) as the tie-break and PCS readout.

## EASY-ITEMS: what we lose on the windows the garage aces (2026-10-19, d387)

[results/easy_items.md](results/easy_items.md); code `scripts/easy_items.py` (box `extract` of per-step metrics, Mac `report` / `pai`).
1 564 of the 2 106 frozen items have garage 0014 = 1; JTN-s1 native misses 181 (105 zeros, 76 partials). The scorer has no sub-score but
progress (0.8 saturation) and the hard fails, so near-1s are the ratchet just short of 0.8. Ranked by frozen PCS (board-mix / g15): static
red-line roll-out 82 windows +79 / +112, free-road ratchet incl. near-1 58 windows +36 / +38 (+101 / +139 on stop3), lateral zeros 20
windows +16 / +21, rest <= +3 each. The shortfall against the garage is longitudinal (arc 0.68 vs 0.87 of the human, laterally closer than
the garage). PAI: no strong-model local read; on our family 94 % of A0's lost points are scenes all three candidates fail.

## WAIT-SET: a closed-loop set where the human waits at standstill

[results/wait_set.md](results/wait_set.md); code `scripts/wait_set.py` (definitions in the docstring), `scripts/wait_set.sh`; the set
`results/wait_set/set.json` (split `alpasim_nuplan/waitset@v1`), cache `$DATA_DIR/datasets/alpasim_nuplan_wait/v1` on the box.

- **Set (v2).** The 199 public scenes whose human starts from a standstill, with the trajdata window moved into the human's wait (same
  asset, same scorer and serving): variant **a** = 146 windows where the human never moves (window ends 0.5 s before the movement onset),
  variant **b** = 166 windows where the human starts moving at 3.0 s. Classes: red light first in line / queued, other with a lead / free (75-84
  each); 94 windows (44 a, 50 b) come from held-split scenes (v1: 95). Every window lies inside its reconstruction's camera video; 34 windows were dropped for an
  undrawn vehicle in the ego's lane, 48 for a standstill too short, 4 (amendment 2) because every driver's rollout fails there.
- **Read a tag:** `bash experiments/alpasim_proxy/scripts/wait_set.sh read <tag>` (one pool job, public 10-decision driver path + the cache),
  then `wait_set.sh report`.
- **Validated before any model read:** the rebuilt cache equals the shipped one on all 171 source scenes' public windows; all windows pass
  the cached-track check; go-straight 0.993 on a (one window where its held handoff speed creeps it 3.1 m off the road) and 0.212 on b
  (never launches); human replay 1.000 on a and b, launch within one 10 Hz step of the logged human (median -0.03 s).

**Handoff (2026-10-17 16:49 CST; the agent ended before the reads, the main session collects).** Validation passed (gate on set v2:
go-straight 0.993 on variant a, human replay 1.000 on a and b; `wait_set/gate.txt`). Reads in the pool, submitted by
`bash experiments/alpasim_proxy/scripts/wait_set.sh models` (ids also in `$DATA_DIR/runs/alpasim_proxy/wait_set/{jobs.txt, stages.txt}`):
- gate job 1017-164835-7969 (passed) -> `ws-read-P0` 1017-164835-39ce, `ws-read-RSG-L18-s0` 1017-164836-94a0, `ws-read-RTB1ARO-L18-s0`
  1017-164836-3e95; private anchors `ws-read-APY10m10-AB-s0` 1017-164837-d360 (ap2 driver), `ws-read-YR10m10-F-s0` 1017-164837-933c,
  `ws-read-LTF-sample` 1017-164838-76d2, `ws-read-DD-sample` 1017-164838-08d6, `ws-read-G0014` 1017-164839-a6e1 (third-party drivers
  native, as their public reads) -> `ws-report` 1017-164839-7fc1 over every read.
- History: the first chain (15:23) died when the platform SIGKILLed the 316-window runtimes at the box's memory line (a read now runs a and
  b as two runtimes of a public chunk's size, and a killed worker exits 137 for the pool's kill retry); the second (15:48) validated but the
  gate failed (go-straight on a 0.986): 4 windows fail before scoring for every driver, the replay included (route fold-back). Amendment 2
  dropped them: set v2 = 312 windows (146 a, 166 b), split `alpasim_nuplan/waitset@v2`. Tables of v1 reads (316 rows, e.g. defect-fix's
  first reads) hold those 4 zeros: filter by `results/wait_set/set.txt` (report and gate do).

Collect (box, repo root): `bash experiments/alpasim_proxy/scripts/wait_set.sh report` -> `wait_set/report/{tables.md, all.csv}` (per tag and
variant: score, zeros by type, roll while the human waits, wait success, b: early / never / launch delay; by class; held-only; paired
differences). Anchor check: `$DATA_DIR/third_party/alpasim/.venv/bin/python experiments/alpasim_proxy/scripts/wait_set.py anchors --csv
$DATA_DIR/runs/alpasim_proxy/wait_set/report/all.csv --out $DATA_DIR/runs/alpasim_proxy/wait_set/report/anchors.md`: per anchor the variant-a
zero rate z_a and score s_a = 1 - z_a next to public and private, the fits private = public - k z_a and private = (1 - p) public + p s_a
(p = share of waiting windows a private-like mix would need), in-sample and leave-one-anchor-out predictions, and Kendall tau-b of each
against private (public alone ranks the five non-trivial anchors at tau -0.6, decision 262). Six anchors: the five plus go-straight
(`starter`, private 0.799; its public is N1's 300-scene sample, 0.522). One read per anchor; results go to `results/wait_set.md`.

**Result (2026-10-17 18:05 CST).** The eight reads are collected; tables, the 94-window held-split subset and the anchor check are in
[results/wait_set.md](results/wait_set.md) (per-window outcomes `results/wait_set/reads_all.csv.gz`). The waiting set separates APY10m10-AB-s0 (28 of 146 variant-a
zeros, 25 of them at-fault contacts, against 1-7 for every other model; P0 waits, RSG / V2 roll more than 3 m in half of the windows), but public score plus the variant-a
zero rate does not order the anchors like private: Kendall tau-b against private, six anchors, public alone -0.47, linear fit +0.20, mixture fit -0.07, leave-one-out
-0.33 / +0.07 (the five without go-straight: -0.60, +0.40, -0.40, -0.20 / -0.20); leave-one-out error 0.13 against 0.15 for public alone. Two defects found and fixed on the
way: `wait_set.py collect` stops on the 3 variant-b windows without a human launch time once the ego launches there (RSG, V2, YR, LTF, DD and the APY re-read ended rc 1 after
complete runs and the report job failed on the dependency; the tables came from `experiments/defect_fix/scripts/df_wscollect.py`, the report was run by hand), and the AP2 driver died at every
drive since JEV_ACC gave sh30_driver's session `t_first = None` (the first APY read scored 0 on all 312 windows; `experiments/alpasim/lib/ap2_driver.py`, re-read 1017-172558-8b65).

## ARENA: uniform windows over every public reconstruction (running, 2026-10-17)

Prereg [plans/2026-10-17-arena-prereg.md](plans/2026-10-17-arena-prereg.md) (Chinese); code `scripts/arena.py` (definitions in its header), `scripts/arena.sh`;
the set `results/arena/{set.csv.gz, meta.json}`; box `$DATA_DIR/runs/alpasim_proxy/arena/`, caches `$DATA_DIR/datasets/alpasim_nuplan_arena/v1/s0..s6`.

- **Set.** 5.5 s windows on a 3 s log-time grid inside the camera video of each of the 1491 public reconstructions (time and space), the waiting
  set's undrawn-vehicle exclusion, overlapping reconstructions sampled once: **7 741 windows (2 246 held-split) on 136 logs**, 55 read chunks,
  7 cache shards. Human state: static 3 143 (41 %), other 1 610, braking 1 205, launch 783, turn 570, cruise 430 (select on the 1.0 s grid:
  23 188 kept of 34 209). Windows where a read fails before driving (route-fail) are dropped at readout.
- **Reads** (user, 2026-10-17: our two models and every anchor with a known private score; no NR1FB10 / V2 / P0, no re-reads): RSG-L18-s0,
  NR1F-L18-s0, YR10m10-F-s0, APY10m10-AB-s0 (ap2), DD-sample, LTF-sample, go-straight starter, G0014, in that pool order, priority 9.
  Smoke (12 windows, every stratum, starter / RSG / APY / DD) passed 19:35 CST.
- **Handoff (submitted 19:36-19:40 CST).** Builds `ar-build-s0..s6` (1017-193624-d6f8 .. 1017-193626-46c9); 440 reads `ar-read-<tag>-c<n>`, first / last
  ids per tag in box `runs/alpasim_proxy/arena/stages.txt`, all in `jobs.txt`; summaries `ar-sum-core` 1017-194044-0676 (RSG / NR1F / YR / APY) and
  `ar-sum-all` 1017-194044-007b (each waits for its reads, then collects and reports whatever finished). About 108 card-slot hours; expected:
  core about 23:00-00:00 CST, all about 02:00-04:00 CST. **Collect:** `$DATA_DIR/runs/alpasim_proxy/arena/report/{core,all}/` (`DONE`, `arena.md`,
  `rows.csv.gz`, `rows_valid.csv.gz`, `figs/arena_scores_speed.png`, `not_done.txt`) -> copy `arena.md`, `rows_valid.csv.gz` and the figure to
  `results/arena/` on the Mac. Rerun the readout by hand: `$P experiments/alpasim_proxy/scripts/arena.py collect --reads $DATA_DIR/runs/alpasim_proxy/arena/reads --out X.csv.gz`,
  then `arena.py report --set $DATA_DIR/runs/alpasim_proxy/arena/set/set.csv.gz --rows X.csv.gz --out DIR --step 3`; a failed chunk is resubmitted
  with `bash experiments/alpasim_proxy/scripts/arena.sh read <tag> <chunk>` as a pool job body (idempotent: finished chunks are skipped).

## CORRIDOR RULE and ARENA-HOLD (2026-10-18)

- **Corridor rule** ([results/corridor_rule.md](results/corridor_rule.md), `scripts/corridor_rule.py`). The scorer's corridor is a 4 m
  tube around the human's path inside the scored window. Past the path's end, only the offset perpendicular to the human's last segment
  counts. When the human stands still to the window's end, that segment is sub-mm jitter, and the corridor becomes a 4 m disc around
  the rest point. RSG on held still-end static windows: zeroed on 379 of the 382 that reached 4 m, never below 4 m.
  The waiting set's variant a ends while the human already creeps (8-20 mm per 0.1 s along the heading), so rolling there was free.
  That is the source of decision 344's "small lever".
- **ARENA-HOLD** (prereg supplement 1 in [plans/2026-10-17-arena-prereg.md](plans/2026-10-17-arena-prereg.md); `scripts/arena_hold.sh`,
  `arena_hold.py`). RSG-L18-s0 with the serve-side static hold `JEV_HOLD` (`full` / `lead` / `leash3`;
  `experiments/alpasim/lib/static_hold.py`, served by `lib/sh30_hold.py` around an unchanged `sh30_driver.py`; checks
  `experiments/alpasim/scripts/hold_check.py`) on the 2 246 held arena windows, paired against the stored RSG arena read.
  Readout: `$DATA_DIR/runs/alpasim_proxy/arena/hold/report/hold.md`.
  Supplement 2 ([results/arena_hold/hold2.md](results/arena_hold/hold2.md), diagnosis [diag.md](results/arena_hold/diag.md)): `stop3` = leash3 +
  a stop leash at the human's constant-deceleration stop point + 3 m; +0.0095 [+0.0048, +0.0146] against leash3, 0 new / 26 fixed zeros;
  recommended for the submission image.
- **ARENA-RESID** ([results/arena_hold/residual.md](results/arena_hold/residual.md), `scripts/arena_resid.py`, CPU on stored reads): stop3's
  125-point gap to 1.0 on held = launch hold cost 40, human track itself offroad 26 (no policy can score; ceiling 0.988), policy offroad 17,
  free slow 15, AF 13, corridor 8, stop-latch 6. No serve-side lever reaches +0.005 (best: leash 3.7 m +0.0017), so no read.
- **LEAD-AWARE HOLD** ([results/arena_hold/lead_aware.md](results/arena_hold/lead_aware.md), `scripts/lead_aware.py`, CPU on stored reads):
  latch only without a lead, serve natively behind one. It loses for RSG / NR1F / JTN (oracle flag -0.025 / -0.014 / -0.003; serve flag
  -0.050 / -0.043 / -0.018), because behind a stationary lead the models still roll 3.5-16.5 % of the still-end windows into a zero, and
  52 of the 82 costly launches have no lead. Release and shorter-leash variants reach at most +0.0038 at serve time. No read.

## Anchors: five policies whose private numbers are known

Private numbers from the leaderboard JSON pulled 2026-10-12 ([results/leaderboard_nuplan_20261012.json](results/leaderboard_nuplan_20261012.json)).
Public numbers: all 1491 public scenes, native on the GPU box (pool), metrics as in AUD1 (src/eval @ 0bb4c4b; 12 route-fold scenes count 0 in
the score, GT distance = mean truncated per-clip maximum, at-fault distance = driven km / at-fault events).

| anchor | public score / at-fault km / GT m | private score / at-fault km / GT m | change |
|---|--:|--:|---|
| `APY10m10-AB-s0` (submission 1) | 0.935 / 0.863 / 1.130 | 0.640 / 0.132 / 2.288 | -0.295, km x0.15, GT x2.0 |
| `YR10m10-F-s0` (submission 2) | 0.925 / 0.789 / 0.941 | 0.800 / 0.529 / 1.733 | -0.125, km x0.67, GT x1.84 |
| SimScale LTF sample (`simscale-lft-release-v0`) | 0.824 / 0.294 / 1.876 | 0.738 / 0.270 / 2.266 | -0.086, km x0.92, GT x1.21 |
| SimScale DiffusionDrive sample (`simscale-diffusiondrive-release-v0`) | 0.797 / 0.243 / 1.843 | 0.772 / 0.267 / 2.158 | -0.025, km x1.10, GT x1.17 |
| garage `model_0014` (five identical team submissions) | 0.740 / 0.183 / 0.869 | 0.848 / 0.228 / 0.718 | +0.108, km x1.25, GT x0.83 |

Public score ranks the anchors APY > YR > LTF > DD > garage; private ranks them garage > YR > DD > LTF > APY. The third-party drivers keep
roughly their public level, ours fall: the private suite is not uniformly harder, it is harder for the openpilot-student family.

How they were run: LTF and DD are the shipped samples (`e2e_challenge/sample_submission_simscale_navsim_{transfuser,diffusiondrive}`) run natively
in `envs/alpasim-ltf` (torch 2.8 for sm_120 instead of the image's 2.6; driver file `experiments/alpasim/scripts/drivers/dd.sh`, LTF is
`run.sh ltf`). Fidelity check on the Tokyo box: both shipped images built unmodified and run through `experiments/alpasim/docker/smoke.sh` on
the 48 local scenes ([results/tokyo_samples/](results/tokyo_samples/)): LTF container 0.8736 vs native 0.8735 (no scene differs by > 0.011);
DD container 0.8308 vs native 0.8515 (one scene flips to 0, the rest agree). Garage: `py123d_garage` v0.1.1 (src identical to the 0.1.0 release)
with the public `model_0014.pth` (sha256 d536614c...), native env `envs/garage`, entrypoint and env of its `submission.Dockerfile`, wrapped by
`scripts/garage_driver.py` (an observer that counts plan failures per session; no model or driver logic changed; `scripts/setup_garage.sh`,
`experiments/alpasim/scripts/drivers/garage.sh`). Six public scenes have no route left after the anchor frame; the garage driver raises
`InsufficientRouteError` there and serves its fallback, as its image would (60 of 14 790 decisions; kept, `OT_HEALTH_WARN=1`). The sample drivers
had 0 inference errors and 0 fallbacks. Pool jobs: 15 runs, about 4 card-hours. Box: `$DATA_DIR/runs/alpasim_proxy/anchors/`.

## Reweighting the public scenes (P0-P3)

Strata: city (LV 481 / BOS 467 / PIT 323 / SG 220 = left-hand), start speed, near-stationary start (log path < 5 m and v0 < 1 m/s, 25 scenes),
4-way command, first route waypoint lateral / longitudinal, log path length, start acceleration. Proxies: single factor (P1), factor pairs
with a KL penalty (P2), max-entropy tilt on all of them (P3); loss = squared error of anchor-centred log failure rate / log km / log GT.

- **Two anchors only** ([results/fit_a2/](results/fit_a2/), frozen before any new anchor was read): city, left-hand traffic, command, route
  geometry and acceleration cannot even flip the sign of the score gap (best +0.008 with Pittsburgh only; Singapore alone -0.026). Only the
  near-stationary stratum carries the private gap: on its 25 scenes YR - APY is +0.160 score, -1.0 m GT, at-fault km x2.1 (the private gap is
  +0.16, -0.56 m, x4.0); proxies that put 66-77 % of the weight there pass V1 + V2 with 47-58 effective scenes.
- **Out of sample (V5, the main read):** none of the 84 frozen proxies orders the five anchors right ([results/fit_a5/frozen_eval.csv](results/fit_a5/frozen_eval.csv)).
  The near-stationary proxy gets the private GT order exactly (garage 0.84 < YR 1.94 < DD 2.36 < LTF 2.48 < APY 2.66 m) but puts the garage last
  on score (0.807 against YR 0.924; private: garage first).
- **Refit on all five** ([results/fit_a5/fits.csv](results/fit_a5/fits.csv), [loo.csv](results/fit_a5/loo.csv)): the best fits reach score tau 0.8
  only with 20-70 effective scenes (V4 line 300) and predict a held-out anchor's rank in at most 3 of 5 cases. Verdict: **no valid scene proxy.**

## Which public statistic transfers (P4)

[results/transfer_a5/](results/transfer_a5/) (`proxy_transfer.py`; CIs log-clustered over 44 logs):

| public statistic | Kendall tau vs private score | vs private GT |
|---|--:|--:|
| GT distance (lower better) | **+0.6** | **+0.6** (two discordant pairs, both APY against a sample: LTF / APY 2.27 / 2.29 m is a private near-tie, DD / APY 2.16 / 2.29 m is not) |
| share of clips >= 4 m off the path | +0.6 | +0.6 |
| per-decision drift slope k = 4..10 | +0.2 | +0.2 |
| scene score | -0.6 | -0.6 |
| at-fault distance | -0.6 | -0.6 |
| zeros | -0.2 | -0.2 |

None predicts the private at-fault distance (|tau| <= 0.2). Public GT distance passes the registered score line (tau >= 0.6) with one
discordant family pair: APY's public GT (1.13 m) beats LTF / DD (1.84-1.88 m) but its private score is lower. Within our family it has
the right sign every time it was tested (APY vs YR: public -0.19 m, private -0.56 m and +0.16 score).

Ranking of every stored candidate by public GT distance (1491 scenes, 95 % CI): garage 0.87 [0.77, 0.95]; YR10m10-F s1 / s0 0.93 / 0.94
[0.83, 1.04]; P2H10-F-s1 0.96; P2H10S-F s0-s3 0.99; JAPSnoA 1.00 / 1.02; P2H10-F-s0 1.02; APY10m10-AB s0-s3 1.11-1.16; JAPSnoY 1.14-1.18;
WA-JEPA 1.19; JAPS 1.24 / 1.26; SH30 1.34; AP2 1.51; DD 1.84; LTF 1.88. No stored candidate is separable from `YR10m10-F-s0` on it.

## What the proxy cannot see, and what was not done

- Scene length: every public scene is 10 decisions. The long-horizon lane ([../alpasim_long/](../alpasim_long/)) rebuilds the trajdata
  cache at longer windows; once it has a held read, the anchors (LTF, DD, garage: about 15 min per 400 scenes each) should be rerun on
  it, and that read is tested against the same five private triples.
- The family-specific loss is unexplained. Candidates not testable on public scenes: the openpilot warp's reliance on the reported
  calibration on private rigs (AUD1 found the MTGS renderer ignores calibration overrides, so a rig mismatch cannot be simulated here),
  route content. The warm-up track (32 private scenes, status only, the user's action) is the only private probe left.
- Tokyo was used only for the container fidelity check above. The two submitted images were not rerun: the stored native rollouts cover
  all 1491 public scenes, the image and native runs agree (AUD1: 0.9708 vs 0.9701 on 48 scenes; interpreted vs compiled passes +0.0002 on
  220), and no public stratum is missing except length.

## Files

`scripts/proxy_fit.py` (reweighting fits, V1-V5), `scripts/proxy_transfer.py` (P4), `scripts/proxy_collect.py` (box: anchor tables through
`aud1_drift_collect.py`), `scripts/garage_driver.py`, `scripts/setup_garage.sh`; results: `anchors/*.csv` (per-scene anchor tables),
`fit_a2/`, `fit_a5/`, `transfer_a5/`, `tokyo_samples/`, `scene_city.csv`, the leaderboard snapshot. Mac inputs for stored drivers:
`tmp/aud1_drift/`, `tmp/aud1_drift_x/` (box `$DATA_DIR/runs/alpasim/aud1_drift*`). Commands:

```bash
# box: anchors (pool), then tables
OT_LANE=proxy OT_OUT=$DATA_DIR/runs/alpasim_proxy/anchors OT_OWNER=alpasim-proxy python3 experiments/alpasim/scripts/ot2_loop.py dd \
  "DD-o:dd:::ot3new" "DD-f:dd:::cf1fresh" "DD-c:dd:::chunks"          # ltf / garage likewise (garage with OT_HEALTH_WARN=1)
$DATA_DIR/third_party/alpasim/.venv/bin/python experiments/alpasim_proxy/scripts/proxy_collect.py --out <dir> "DD-sample=<anchors>/dd/runs/DD-*/2*"
# Mac
.venv/bin/python experiments/alpasim_proxy/scripts/proxy_fit.py tmp/aud1_drift/per_scene.csv experiments/alpasim_proxy/results/anchors/{ltf,dd,g0014}.csv \
  --anchors APY10m10-AB-s0,YR10m10-F-s0,LTF-sample,DD-sample,G0014 --out experiments/alpasim_proxy/results/fit_a5 --frozen experiments/alpasim_proxy/results/fit_a2/weights.npz
```

## PAI-CORRIDOR: the still-end corridor lever does not carry to PAI (2026-10-18)

Same scorer and 4 m corridor on PAI, but only 2 of 281 PAI scenes have the human still from the handoff to the end, so a handoff hold
(leash3) nets -0.04 (g10) to -0.08 (ACC+GAP) offline, ceiling +0.004; PAI's 14-16 still-end corridor zeros per read are humans stopping
mid-scene. [results/pai_corridor.md](results/pai_corridor.md), script `scripts/pai_corridor.py`.
