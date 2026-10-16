# SCENE-MIX pre-registration: is the private nuPlan suite an unfiltered (static / cruise-rich) scene mix? (2026-10-16 16:40 JST, before any read)

Written before any N1 or N2 number exists. After the first commit only amendments are appended; the text above them is not edited.

## Question and hypothesis

Our openpilot students lose far more on the private AlpaSim nuPlan suite than on the 1491 public scenes (YR10m10-F-s0 0.925 -> 0.80,
APY10m10-AB-s0 0.935 -> 0.640), the third-party samples roughly keep their level (LTF -0.09, DD -0.03), garage 0014 rises (+0.11), and the
host's `go_straight_with_delay_v2` scores 0.799 on private with PCS ~1295-1316 where we have PCS 1007 at the same mean: our private
zeros fall on scenes most entries pass.

**H (scene mix).** The public scenes are navtest tokens (`MTGS_asset/navtest`); NAVSIM's navtest filtering removed trivial samples
(static ego, samples a constant-velocity agent already solves). Decision 218 point 5: stationary navtest tokens launch (log moving 90 %),
and our adapter launches from standstill 93 % of the time even with a stationary lead within 15 m (shipped Cinque 15-33 %). The organiser
(NVlabs/alpasim #204) says the private set was not difficulty-upsampled and DriveIRT was not used in selection. If the private set is NOT
navtest-filtered, it holds many easy "human waits / cruises" scenes absent from public, which a trivial policy passes and where our driver
launches past the human's stop (corridor zero, evaluator bug #198) or rear-ends a stationary lead.

Two reads. N1 tests the consequence H must have (the private suite is far easier than public for a trivial policy); N2 measures how
different an unfiltered mix would be, and how much of our loss such a mix could carry.

## N1: a go-straight driver on the public scenes

**Driver.** The AlpaSim repo has the host's go-straight policy as the starter kit: `e2e_challenge/starter_kit/driver.py` at the deployed
commit 0bb4c4b (since ab57fef, 2026-08-17, "Base Starter Kit Command on Current State"): a straight line along the latest pose's heading
at the latest observed speed (reported dynamic state, pose-delta fallback; 5 m/s only when no speed was ever observed), 5 s at 10 Hz;
at standstill it stays. No variant named "with delay" exists in the repo (all branches, history, issues searched). The host's
`go_straight_with_delay_v2` (2026-09-01) postdates ab57fef, so "v2" plausibly is this current-speed version; "delay" is unpublished. If it
means a latency added to `drive`, it changes nothing locally (the runtime is synchronous, no RPC deadline). The starter kit is run
**unmodified** (`run.sh <dir> starter`, AlpaSim's own venv); this is the mirror.

**Run.** All 1491 public scenes, native AlpaSim stack, `+e2e_challenge_nuplan=full` at 0bb4c4b with `c0b_chain.OVERRIDES` (the settings of
every stored anchor), through `ot2_loop.py` on the GPU-box pool (Tokyo holds the assets of only 48 scenes; 1443 more scenes are ~590 GB at
3.4 MB/s), lists `ot3new` + `cf1fresh` + the three `chunks` (= the 1491, as for the LTF / DD / garage anchors), low priority, at most 3
stacks. Out: `$DATA_DIR/runs/alpasim_proxy/scene_mix/gs/`, table via `proxy_collect.py`, metrics as the anchors (12 route-fold scenes count
0 in the score, GT distance = mean truncated per-clip maximum, at-fault km = driven km / at-fault events).

**Primary line (public mean scene score of go-straight, against its private 0.799):**
- <= 0.70: **supported** (the private suite is >= 0.10 easier than public for a trivial policy; the non-family anchors move by -0.09 to
  +0.11 from public to private, so a gap below ~0.10 cannot be told apart from driver-specific transfer);
- >= 0.78: **refuted** (the private suite is not easier for a trivial policy: its mix is public-like as far as a policy without perception
  can tell);
- in between: undetermined.

Also reported (not gated): at-fault km and GT distance against private 0.258 km / 1.15 m; zeros by type (collision at fault, offroad,
left corridor, route fold); the near-stationary stratum (log path < 5 m and v0 < 1 m/s, the d262 definition, 25 scenes) and the
human-static stratum (N2 definition) with go-straight's and every anchor's score there.

## N2: scene statistics of an unfiltered mix (CPU, box, the raw logs)

**Data.** The 136 raw nuPlan v1.1 test logs rebuilt for experiments/alpasim_long (decision 265; `$DATA_DIR/datasets/nuplan_raw/test`):
exactly the logs the public scenes come from, so "uniform in time over these logs" is the unfiltered version of the same drives. Ego state
from `ego_pose` (position, vx / vy), taken at the `lidar_pc` instants (20 Hz).

**Windows.** 5.5 s (the public scene: 10 decisions at 2 Hz plus the last 0.5 s query). Uniform: every start on a 0.5 s grid whose window
fits in the log (overlapping windows; CIs bootstrap logs, 10 000 resamples, `jevdrive.stats` groups=log). Public: the window starting at
each of the 1491 scene ids' central token (`<log>-<token>`). Alignment check (not a read): the public windows' path length and v0 must
reproduce AUD1's `gt_dist_traveled_m` / `v0` (`experiments/alpasim/results/aud1/per_scene.csv`) to within 1 m / 0.5 m/s on >= 95 % of
scenes and the near-stationary count 25; otherwise the window offset is fixed by that table (amendment) before any share is read.

**Classes (per window, may overlap):**
- **S static:** max displacement from the start position < 1 m over the whole window;
- **E ends stopped:** speed at the window end < 0.5 m/s;
- **N near-stationary start:** v0 < 1 m/s and path < 5 m (the d262 definition; primary);
- also reported: C cruise (v0 >= 5 m/s, |v_end - v0| < 1 m/s, heading change < 5 deg), and the share of windows with nuPlan's own
  `stationary` scenario tag.

**N2a primary line:** share(N) on uniform windows / share(N) on public >= 3 (public d262 count: 25 / 1491 = 1.7 %) = **supported**; <= 1.5 =
**refuted**; in between undetermined. S and E reported with the same ratio, not gated.

**N2b reach (secondary, not gated alone).** Reweight the stored public per-scene scores by stratum (S; N not S; E not S / N; rest) to the
uniform-window shares and predict each anchor's private score (APY, YR, LTF, DD, garage, go-straight). Reported: predicted change against
the observed public -> private change. H "carries" the family loss if the predicted YR and APY drops are each >= half of the observed
(-0.06 / -0.15) and the predicted APY < YR order holds, while go-straight's predicted private score moves toward 0.799. Strata with < 5
public scenes are reported as too thin, not extrapolated.

**N2c (descriptive).** On public scenes where the human is static (S) and the near-stationary ones (N), from the stored rollouts
(`tmp/aud1_drift/per_scene.csv`: APY / YR / P2H10 / SH30 / AP2 / WA-JEPA; anchors `results/anchors/{ltf,dd,g0014}.csv`; go-straight from
N1): score, zeros by type, distance driven.

## Verdict

- **Supported**: N1 supported and N2a not refuted.
- **Refuted**: N1 refuted (a trivial policy finds private no easier than public; whatever the private mix holds, it is not the
  easy-static mix H needs). N2 then only says what an unfiltered sample would look like.
- Otherwise undetermined. N2b decides only how much of the family loss a supported H explains.

**Implication to state either way:** a serve-side "standstill hold" (hold zero speed while ego is near-stationary unless a launch condition
holds) pays where (a) the human stays static and we launch: count those zeros on public S / N scenes from N2c and scale by the private
share N2a implies; and costs where the human launches (public N scenes: 90 % launch in navtest). Report the expected net per 1000 private
scenes under each mix, not as a read.

## Cost and rules

N1: 5 pool jobs, ~1.3 card-hours expected (the LTF / DD / garage anchors took ~4 card-hours for three drivers). N2: CPU only, minutes. No
submission, no login, no Tokyo card used (assets absent there). Results: `experiments/alpasim_proxy/results/scene_mix/`, then one decision
entry. No polling loops: one blocking background wait per phase.

## Amendment 1 (2026-10-16 16:50 JST, before N1 has any result; N2 alignment fix)

The N2 alignment check failed on the first extract: with the public window starting at the central token, |path - gt_dist_traveled| < 1 m
and |v0 - v0_aud1| < 0.5 m/s held on 13.9 % of scenes (line 95 %). Deviation from the plan, stated plainly: `report` printed the class
shares in the same output as the failed check, so the shares on the misaligned public windows were seen (uniform N share 0.33, public
misaligned 0.007). The uniform windows do not depend on the offset. Fix as pre-registered, from the AUD1 table only: a grid over start
offset (-3..+3 s, 0.5 s) and length (4.5 / 5 / 5.5 / 8 s) on 308 public scenes of 40 logs gives offset **-1.5 s**, length 5.5 s (median
|dpath| 0.06 m, 97.7 % within 1 m, median |dv0| 0.006 m/s; every other cell <= 36 %): an AlpaSim nuPlan scene starts at the beginning of
the navtest token's 1.5 s history. Public windows now start at token - 1.5 s (`OFFSET` in `scene_mix.py`); the check is rerun on all
scenes and must pass before the shares are used. No line, class or verdict rule changes.

## Amendment 2 (2026-10-16 17:00 JST, N1 still queued; the rerun alignment check)

With the -1.5 s offset the check holds on 94.4 % of the 1491 scenes (line 95 %): it misses narrowly. All 71 misses are fast scenes
(logged path 36-80 m, v0 3.4-15 m/s) whose 10 Hz path is 1.0-1.4 m (about 2 %) shorter than AUD1's `gt_dist_traveled_m`; v0 agrees on all
1491 (|dv0| < 0.5 m/s), every scene with a logged path < 10 m agrees (157 / 157), and class N counts 24 against d262's 25. The classes
S / E / N live entirely in the slow regime that agrees, so the shares are used as they are; the window is not retuned to pass the line.

## Amendment 3 (2026-10-16 17:15 JST, user decision, before any N1 number exists)

N1 no longer runs all 1491 scenes. The five full-set jobs were cancelled while queued (none had started; ids 1016-152719-20f4,
1016-152719-3509, 1016-152720-4c0c; the other two were never submitted). New N1:
- **Primary set:** a log-stratified random sample of 300 public scenes: largest-remainder allocation proportional to each of AUD1's 44
  logs' scene count, at least one per log, `numpy.random.default_rng(20261016)` (`scene_mix.py sample`; lists
  `results/scene_mix/n1_sample.txt`). **Near-stationary set:** the 25 d262 scenes (AUD1 logged path < 5 m and v0 < 1 m/s;
  `n1_nearstat.txt`), reported separately; 7 overlap with the sample; the run list is the union, 318 scenes (`n1_run.txt`), one pool job
  at priority 25 (above the queued PAI reads at 20 / 12).
- **Lines unchanged** (<= 0.70 supported, >= 0.78 refuted) but now applied with the 95 % CI of the sample mean (bootstrap over logs,
  `jevdrive.stats`, 10 000 resamples): supported if the upper bound <= 0.70, refuted if the lower bound >= 0.78. If the CI straddles a
  line, the result is reported as such (point side plus "CI straddles"), and the sample is not extended.
- Every stored driver is reported on the same 300 / 25 scenes next to its 1491-scene number (a check that the sample is representative,
  not a read). Zero types and N2c strata as before.
