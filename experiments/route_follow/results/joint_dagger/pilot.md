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
