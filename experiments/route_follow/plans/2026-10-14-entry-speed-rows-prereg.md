# Pre-registration: ENTRY-SPEED ROWS, are hinge rows at fast turn entries waste (no training, no GPU), 2026-10-14

Context: the body lesson trains the policy on hinge rows computed on its own rollout states (route hinge `on_route.py`, S's road and agent
hinges; decisions 286, 290, 293). Hypothesis to test, not to confirm: **at high turn-entry speed a lateral error becomes a terminal zero
(offroad / collision / corridor) within 1-2 s, so the hinge rows around it sit in waste states (just before or after the termination, or far
off the route); at lower entry speed the same error stays a small continuous deviation and the rows are recoverable.** If true, the usable
share of hinge-active rows rises clearly as entry speed falls, and a speed-capped practice curriculum is worth building. If flat, the
speed-cap idea is dropped. Written and committed before any statistic below is computed. Amendments are appended, never edited in place.

## Data (stored, read-only)

**Source N (nuPlan, primary).** Closed-loop @18 rollouts (18 decisions at 2 Hz) whose on-policy row pools exist with route labels
(`$DATA_DIR/runs/onpol_nuplan/rows/<P>` meta / active / all, `runs/op_parity/cache/on1_<P>/route.npz`):

| pool P | policy | scenes | turn read (rf_follow, unchanged) |
|:--|:--|:--|:--|
| L18RB0 | RB0-F-s0 | train 1021 | `runs/route_follow/report/follow_train_rb` (label RB0-F-s0@18, same roll dirs) |
| L18RSG | RSG-L18-s0 | train 1021 | `runs/route_follow/entry_speed/follow` (run for this read, same roll dirs as the rows) |
| L18RBSG | RBSG-L18-s0 | train 1021 | same |
| L18RBSG2 | RBSG2-L18-s0 | train 1021 | same |
| L18RSGallheld | RSG-L18-s0 | held 452 | same (label @held) |
| L18RBSG2allheld | RBSG2-L18-s0 | held 452 | same (label @held) |

Pools without `route.npz` (L18Y, L18YS) and P2H10S (roll dirs deleted) are left out. Different policies on the same scene give the
natural within-scene variation in entry speed; scenes are the clusters.

**Source P (PAI, P3 archive on Tokyo, secondary + within-scene contrast).** `/data/runs/route_night/p3/arms` (192 rollouts = 12 hard PAI turn
scenes x handover d10 / d0 x 8 transforms), `read.json` (done, zero, t_zero), the route grid and strict turn of
`results/p4_self/pool_dev/plan_p3.json` (rf_p4 plan: `s_entry`, `s_exit`, `v_e`, `v_log_entry`). The arc-length transforms (s 0.8 / 0.6)
are the designed speed variation. Known before registering (results/p3_branch README / shape.md): with the d0 handover the ego reaches the
entry at the log's speed (the push starts there), so v_entry barely moves within a d0 scene; the s < 1 cells slow the car inside the
junction to 0.1-1.8 m/s (a near stop), and many end rear-ended. Hence P enters analysis 3 by design arm (s < 1 vs s = 1), with the realized
speed reported as the manipulation check, and the bins of analysis 1 are computed on N only.

## Unit, covariate, labels

**Cell** = scene x rollout x strict turn. N: rf_follow's turn (route net heading change >= 60 deg in 30 m inside the log's travel to decision
17, exit + 10 m inside it); P: plan_p3's strict turn. One turn per rollout.

**Decisions of a cell.** N: the pool rows (the rows the trainer draws; the pool already excludes states overlapping an actor or > 4 m off the
logged path, which are counted separately as "excluded") of that rollout whose anchor station (rf_follow `st`, joined on k) lies in
[entry - 40, exit + 10] m. P: every 10 Hz decision with k >= the handover decision k_eff (earlier decisions are log-driven) whose anchor station
on the plan route lies in [entry - 40, exit + 10] m.

**Covariate.** v_entry = the ego speed where its route station crosses the entry (N: rf_follow's fed speed `v`, linear in station between
the two decisions around the crossing; P: controller vx at the crossing, `rf_shape.speeds` v_0 without the zero cut). If the ego never crosses,
the speed at the decision nearest the entry within 10 m, else the cell has no v_entry (counted, left out of analyses 1-2).
v_h = the human's speed at the same entry station (N: the log's position at each decision time (`meta.logfut[0]`) projected on that row's
label polyline gives the log's station, `meta.v_log` its speed, linear in station at the entry; P: plan_p3 `v_log_entry`). **v_ratio =
v_entry / v_h**; cells with v_h < 2 m/s have no ratio (counted, reported in the absolute-speed table only). P also reports v_entry / v_e
(plan class target).

**Labels per decision** (primary T = 3 s, L = 1.0 m, H = 20 deg):
- terminated: at or after the first zero-score event (N: rf_follow `kz` = first decision with offroad / front / lateral collision or
  dist_to_gt >= 4 m; P: read.json `t_zero`, i.e. corridor / offroad / collision; a rear-end has no stored time and is not a zero event here,
  rear-ended cells are flagged).
- lat = distance of the anchor to the row's route label polyline (`on_route.near` with a pose at the anchor; N: the pool's route.npz;
  P: `rm_build.label(plan route grid, anchor)`, the same label builder); head = |route tangent at the anchor's projection - ego yaw|.
- **recoverable** = not terminated, no zero event within T s ahead (N: kz - k > T / 0.5 s; P: t_zero - t > T), lat <= L, head <= H.
  **waste** otherwise; waste is decomposed into after-termination / within T before it / off L or H only.
- **hinge-active** = the row's own stored plan has a nonzero hinge, computed by the trainers' own code, not re-implemented:
  N: `active.npz` agent > 0 or road > 0 (on_rows' S values on the served plan) or `on_route.loss` > 0 on the served plan (`meta.poses`) with
  the pool's route.npz (dead zone 1.0); primary = any of the three, per-term rows reported. P: `on_route.loss` on the adapter plan
  (`poses_model`, the plan the PAI trainers hinge; `poses` where absent) with dz 1.0; road / agent hinges need the simulator replay and are
  not measured on P.
- **usable** = hinge-active and recoverable.

**Outcomes per cell:** usable share = usable / hinge-active rows (undefined with no hinge-active row: counted, left out of the share);
usable count; turn completion (N: rf_follow `completed`; P: read.json `done`).

## Analyses

1. **Bins (primary, N).** Cells binned by v_ratio with edges 0.9 / 1.1 / 1.3 (bins < 0.9, 0.9-1.1, 1.1-1.3, >= 1.3) if every bin then holds
   >= 15 cells and >= 8 scenes; otherwise the pooled N quartiles of v_ratio (computed on the covariate alone, printed). Per bin: usable share
   as the ratio of sums over hinge-active rows, usable count per cell (mean), completion; 95 % CIs by scene-cluster bootstrap (resample
   scenes, B = 10 000, `default_rng(0)`, percentile). Contrast D = share(lowest bin) - share(highest bin), one draw of scenes for both bins
   (a scene can sit in both through different policies; jevdrive.stats has no two-group cluster contrast, hence the local bootstrap).
   Also by absolute v_entry (< 3, 3-5, 5-7, >= 7 m/s) and per policy.
2. **Regression (N).** Cell-level binomial GLM (usable of hinge-active rows) on v_ratio, (a) pooled, (b) with scene fixed effects (the
   within-scene slope); IRLS in numpy; slope CI by scene-cluster bootstrap (B = 2000). The hypothesis predicts a negative slope.
3. **Within-scene P3 contrast.** Per scene x handover x g in {1, 1.5}: share(s < 1 cell) - share(s = 1 cell) (s 0.8 and 0.6 separately and
   pooled), mean over the pairs with both shares defined, scene-cluster bootstrap. Positive = slower is more usable (the hypothesis' sign).
   Reported with the realized speed difference (mean speed over [entry, exit] and v_entry) as the manipulation check.
4. **Sensitivity.** D of analysis 1 and the analysis 3 contrast over T in {1.5, 3, 5} s x L in {0.5, 1, 2} m x H in {10, 20, 30} deg.

**Floor.** Completion per v_ratio bin (and absolute bin) with CI; floor = the lower edge of the lowest bin whose completion CI upper bound
reaches the pooled completion rate (the slowest band not clearly worse at finishing the turn).

## Decision rule (registered)

- **"Speed cap worth building"**: D >= 0.20 absolute with its 95 % CI above 0, **and** the pooled P3 within-scene contrast (analysis 3,
  s < 1 vs s = 1) has the same (positive) sign.
- **"Flat"** otherwise (the speed-cap idea is dropped).
- Counts **against** the cap, stated next to the verdict whatever it is: the lowest bin has fewer usable rows per cell than the highest
  (count difference CI below 0), or its completion is lower (CI clear of 0), or the slow P3 cells lose hinge-active / usable rows (e.g. the
  crawl hides the curvature from the MPC window and the plan stops bending, d284 / d285).
- Analyses 2, 4, the per-term and per-policy rows and the absolute-speed table are reported and interpreted but do not change the verdict.

Threats named in advance: entry speed is not assigned on N (a slow entry can be the consequence of an earlier error, or of a policy that
fails differently), so N is association; P's design contrast is the only assigned variation and has 12 scenes, one seed, and its slowing
happens inside the junction rather than before it. rf_follow decisions are 2 Hz (T resolves to 0.5 s). Pool exclusions remove the worst
post-event states from N, so N's waste share is a lower bound of the decision-level waste. Six policies but all one family (one checkpoint
lineage, seed 0): evidence grade low.

Cost (est.): CPU only. rf_follow on 3 x 1021 + 2 x 452 rollouts at 8 workers in tmux on the box (~15 min, outside the GPU pool, inside the
CPU margin of `cl top`), a numpy extract per pool on the box, a plain-python3 extract of 192 P3 drive.jsonl on Tokyo, the report on the Mac.
