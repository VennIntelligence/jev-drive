# JOINT-DAGGER pilots (2026-10-16)

Prereg: [plans/2026-10-16-joint-dagger-prereg.md](../../plans/2026-10-16-joint-dagger-prereg.md). Code: `experiments/alpasim/lib/rule_expert.py` (stop rule,
`JEV_EXPERT_STOP`), `scripts/rf_jdagger.{py,sh}`, `rf_expert.sh --expert-stop`, `rf_recov_post.{sh,py}` (`submit-held` / `stage2`, kind J). Box run dir
`$DATA_DIR/runs/route_follow/joint_dagger/` (`stops/{held,train}`, `labels.{npz,json}`, `pack.json`, `flags_*.txt`, `gate/`, `xpgate/`, `jobs.txt`).
JSON copies here: [labels.json](labels.json), [pack.json](pack.json), [gates.json](gates.json), `gate_<tag>.json`, [xpgate.json](xpgate.json).

**Verdict: every pilot fails the offline gates (dev ADE, G-rely, equal-arc turning); no held read, no full run.** The policy does learn the labels
(-42 to -49 % label ADE on held-out logs, the same on its training rows), but what the labels teach on the turn approach is a faster plan that turns
later per metre, and the dose costs human imitation (dev ADE 0.60-0.74) and route reliance.

## Teacher fix: the expert's stop rule (step 1)

- **Data.** LONG-PROBE's stop lines (onpol_nuplan ctx.npz) mapped onto each pack's path. The simulation frame is the map frame shifted by O, and
  simulation time is log time - T0, both from the pool metas. The ctx keyframe poses land on the pack's log path within 1.2 cm at p95 (max 6 cm).
- **Coverage.** Held: 447 / 460 packs (13 scenes have no on-policy rows to give O / T0), 96 eligible lines. Train: 1 025 / 1 030 packs, 270 eligible lines.
  56 + 112 lines lie > 3 m from the expert's path and are not eligible.
- **Bit check.** With the switch off, the expert reproduces DAGGER-CHECK's stored plans on all 17 474 rows exactly (max |diff| 0).
- **Expert held read with the rule on** (`RSG-L18-s0-XP-L12-E46-ST` vs the stored `-XP-L12-E46`, 452 paired scenes):
  - The rule stopped the plan on 452 of 8 129 expert decisions (5.6 %).
  - Completed turns 22 / 24 vs 22 / 24. Zeros 22 vs 22 (+0 [+0, +0]); at-fault +0.
  - Mean score -0.0012 [-0.0038, +0.0000]; GT -0.007 m [-0.016, -0.001].
  - In the 44 scenes where the human stopped at an eligible line: zeros 2 vs 2.
  - **Reading: neutral, no gross regression.** The stop rule does not touch the expert's held zeros. Decision 314's "expert drives on after the human
    stopped" zeros are queue / lead stops, not stop lines.

## Labels (step 2)

- **Rows.** on1_L18RSG own rows (k >= 2) with a pack: 15 426.
- **Dropped** because the target's 0.1-4 s footprint touches an agent box (1 058) or leaves the road raster (368), or was already in contact at
  t = 0 (144).
- **Valid.** 13 992 rows / 1 011 scenes / 97 logs: 12 178 train-log rows (pool on1_JDGd) and 1 814 dev-log rows / 16 logs (on1_JDGdev, never trained).
- **hi rows (x 3 draw weight).** 2 585 train rows: 398 in the 3 s before a failed turn, the rest junction / turn approach. They take 45 % of the
  DAgger draws.
- **Re-timing.**
  - With no bound binding, the target equals the expert plan exactly (max 4e-6 m).
  - The upper bounds bind on 3.7 % of valid rows: lead 3.1 %, turn 0.4 %, stop line 0.3 %. Lead binds on 24 % of stop-approach rows.
  - So the target is essentially the expert with the stop rule.
- **Contact-free over 4 s, own rows with labels.** Target 0.907, expert with the stop rule 0.894, served 0.887.
- **Target vs served (valid rows).** d2 median 2.2 m, lat4 0.43 m, lon4 mean -1.6 m (served behind, i.e. slower), dh4 median 1.3 deg.
- **Batch (128 rows).** 13 S rows (self) and 13 yaw-rate rows stay. DAgger rows 32 / 64 (25 / 50 %; weighted draw, expected) replace human
  imitation rows (102 -> 70 / 38).

## Bit identity (gates.json)

RSG-L18 recipe, 60 steps (JDGG-off) vs pool on1_JDG with the DAgger rows at weight 0 (`--on-mode mixed --on-k 13 --on-weight pool=d:0`, JDGG-w0):
80 scalars and every weight identical, on_hash_60 ea8cacc3e451c577 both = RSG-L18-s0's stored hash. **PASS.**

## Pilot gates (1 000 steps, seed 0, vs RSG-L18-s0; init RB0-F-s0 dev ADE 0.5276)

Lines: dev ADE <= 0.56; G-rely true; equal-arc turning >= +0.3 deg; DAgger-label ADE on dev-log rows (1 814 rows) drop >= 30 %. Label ADE of
RSG-L18-s0 on the dev rows: 3.091 m. The "window" column is the 34 dev rows in the 3 s before a failed turn. The "train rows" column is the same
read on the 12 178 training rows (descriptive).

| pilot | share / lr-scale | dev ADE | ade no-mem | G-rely (d_junction) | equal-arc turn [CI] (deg) | 4 s heading / 4 s arc | label ADE dev | window rows | train rows | result |
|:--|:--|--:|--:|:--|:--|:--|:--|:--|:--|:--|
| JDGP25L03 | 25 % / 0.3 | 0.6107 | 0.5876 | False (-0.020) | -0.98 [-1.42, -0.45] | +1.45 / +0.76 m | 1.793 (-42.0 %) | -41.9 % | -40.7 % | FAIL: ADE, rely, turn |
| JDGP25L10 | 25 % / 1.0 | 0.5988 | 0.5958 | False (+0.002) | -1.41 [-1.95, -0.79] | +0.70 / +0.61 m | 1.727 (-44.1 %) | -45.9 % | -44.9 % | FAIL: ADE, rely, turn |
| JDGP50L03 | 50 % / 0.3 | 0.7391 | 0.6668 | False (-0.072) | -1.14 [-1.79, -0.44] | +1.20 / +0.66 m | 1.686 (-45.4 %) | -48.0 % | -47.2 % | FAIL: ADE, rely, turn |
| JDGP50L10 | 50 % / 1.0 | 0.7023 | 0.6659 | False (-0.030) | -1.50 [-2.15, -0.71] | +0.63 / +0.51 m | 1.575 (-49.0 %) | -53.1 % | -52.6 % | FAIL: ADE, rely, turn |

## Reading

- **The labels are learnable and transfer.** Unlike R / RL (decision 319 rows: -0.2 to -1.5 % at held-out handover states), the policy moves 42-49 %
  toward the expert on held-out logs, and the drop equals the training-row drop. The window rows move as much as the rest.
- **Turning goes the wrong way, and the teacher is why.** At equal arc the pilots turn 1.0-1.5 deg less toward the turn, while their 4 s heading
  rises (+0.6 to +1.5 deg) because their 4 s plans run 0.5-0.8 m further.
  - The same offline read of the labels against the served plans (equal-arc heading toward the turn, rows whose target turns > 10 deg):
    - turn approach: -0.93 mean / -1.39 median, arc +1.41 m;
    - junction: +3.16 / +2.56, arc -1.23 m;
    - failure window: +2.28, arc -1.23 m.
  - On the approach the expert (pure pursuit on the lane centreline, lookahead max(6 m, 1.2 v), human speed) is faster and turns later than the policy.
  - It turns more only once inside the junction. The turning instrument's rows (held decisions 0 / 1, [entry - 40, entry + 10) m) are mostly approach
    rows, and the pilots copied the approach.
- **Dose.** Dev ADE is set by the share (25 %: 0.60-0.61; 50 %: 0.70-0.74), not by the lr. At a given share, lr 1.0 is slightly better than 0.3.
  - Even 25 % costs +0.07 against the 0.56 allowance.
  - Route reliance collapses at every dose (`ade_nomem` approaches `ade`; d_junction <= 0). The expert's lane-chain path disagrees with the human path
    the route memory was trained to explain.
- **Not launched** (prereg 5: all pilots fail -> report). Options for the main session / user, not registered:
  - restrict the DAgger rows to the junction and the failure window, where the teacher turns more and is cleaner (1 085 junction + 398 window train
    rows), and drop the approach and straight rows;
  - a lower share (<= 10 %) to keep dev ADE and route reliance. The turn sign would likely stay, because it comes from the approach labels.

## Cost

- GPU:
  - 4 pilots x ~7 min plus 2 identity runs: ~0.55 GPU job-h.
  - 6 expert held chunks: ~1.2 job-h.
  - 6 forward jobs: ~0.2 job-h.
- CPU: stops, labels, pack, gates and xpgate: ~1.5 core-h.
- Wall 09:23-09:40 JST (box 08:23-08:40 CST); one rerun of the base forward job after the dev-tag fix.

## Amendment 1 pilots: junction + failure-window labels only, 10 / 15 % (2026-10-16 ~10:50 JST)

Prereg Amendment 1. Pool on1_JDGJ = on1_L18RB0 (S, self) + on1_JDGjd. on1_JDGjd holds the on1_JDGd rows in the junction segment or in the 3 s
before a failed turn: 1 162 rows / 26 logs (junction 986, window 398, both 222). Dropped: 1 423 approach rows and 9 593 others. The draw inside
the subset is uniform. Copy of the pack record: [pack_j.json](pack_j.json).

Realised batch (expected values; weighted draw with replacement):
- j10: `--on-k 26`, 13 S + 13 DAgger rows (10.2 %, sd 2.5), 89 human rows.
- j15: `--on-k 32`, 13 S + 19 DAgger rows (14.8 %, sd 2.8), 83 human rows.
- Each subset row is drawn about 11 / 16 times in 1 000 steps.
- lr-scale 0.3, seed 0, 1 000 steps.

Dev label rows: 114 rows / 6 logs (junction 99, window 34). Copies: `gate_JDGPJ{10,15}L03-L18-s0.json`.

| pilot | dev ADE | G-rely d_junction [diff CI] | equal-arc turn, window [-40, 10) m | approach [-40, 0) | junction [0, 10) | label ADE dev: trained classes (gate) | junction / window / approach / rest | result |
|:--|--:|:--|:--|:--|:--|:--|:--|:--|
| RSG-L18-s0 (ref) | 0.528 | +0.025 [+0.012, +0.038] true | - | - | - | 2.819 m | - | - |
| JDGPJ10L03 | 0.5530 | +0.005 [-0.016, +0.030] **false** | **-0.36 [-0.63, -0.08]** (n 276, arc +0.33 m) | -0.34 [-0.62, -0.06] (n 271) | -1.10 [-2.46, +0.36] (n 5, 3 scenes) | 1.715 (-39.2 %) | -43.4 / -31.3 / -27.1 / -8.7 % | FAIL: rely, turn |
| JDGPJ15L03 | 0.5599 | +0.001 [-0.018, +0.022] **false** | **-1.32 [-1.82, -0.77]** (arc +0.37 m) | -1.27 [-1.77, -0.73] | -3.64 [-8.15, +0.40] (n 5) | 1.629 (-42.2 %) | -44.8 / -37.6 / -30.1 / -10.6 % | FAIL: rely, turn |

The 4 s heading is +0.96 / +0.01 deg. The label read on the training rows (descriptive) gives:
- trained classes: -43.0 / -48.4 %;
- approach rows (not trained): -18.8 / -21.2 %.

**Verdict: both fail. Not launched** (Amendment 1: both fail -> stop and report).

### Reading, by gate and segment

- **Dev ADE passes now, and the label gate passes.** Dev ADE is 0.553 / 0.560 against the 0.56 line; at 10 % it is +0.025 over the reference.
  The trained classes move 39-42 % toward the teacher on held-out logs.
- **Route reliance still collapses, even at 13 rows per batch.**
  - d_junction falls from +0.025 [+0.012, +0.038] (RSG-L18-s0; RB0-F-s0 +0.025) to +0.005 / +0.001, with CIs that contain 0.
  - Masking the route memory now costs the pilots almost nothing (0.557 vs 0.553).
  - The teacher's lane-chain junction paths are exactly where the route memory used to pay. A small share of them is enough to make it redundant.
- **Turning still goes the wrong way, and the sign comes from the approach.**
  - The registered window has 276 rows, of which 271 are approach rows (held decisions 0 / 1 sit before the entry). Only 5 rows / 3 scenes are
    inside the junction, so the turning instrument cannot see the junction at all.
  - On the approach rows, the pilots turn less at equal arc (-0.34 / -1.27 deg) and run further (+0.33 / +0.37 m in 4 s). Raising the share
    from 10 to 15 % makes this about 4x worse.
  - The approach rows were not trained. Even so, the dev approach rows moved 27-30 % toward the teacher's approach labels, which are faster and
    turn less per metre.
  - So the junction / window labels generalise onto the approach. Dropping the approach rows removes the direct signal, but not the teacher's
    speed and late-turn style that the junction rows carry too: the junction rows' re-timed expert speed sits on the human speed and the
    3 m/s floor.
  - The 5 junction rows point the same way (-1.1 / -3.6 deg, CI contains 0).
- **What this leaves (not registered; for the main session):**
  - The dose line is now clear: below 10 % the label gain shrinks, and at 10 % reliance is already gone.
  - Making this teacher useful would need its approach behaviour to match the policy's. One way: re-time the junction labels at the policy's
    own speed (geometry only). Another: freeze or regularise the route branch's use against the DAgger rows.
  - Either is a new design, not a dose. A junction-row turning instrument (held rows inside [0, exit + 10) m with k > 1) would be needed to see
    any junction gain.

Cost:
- GPU: 2 pilots + 2 forward jobs, about 0.25 GPU job-h.
- CPU: packj and the gates, under 0.2 core-h.
- Wall: 6 min (box 09:46-09:52 CST).
