# STACK cheap checks (prereg `plans/2026-10-16-stack-prereg.md` section 1)

## C1: can the Cinque plan head be sampled?

- No. `net.slices["plan"]` = outputs 917 .. 1907, 990 numbers = ONE MDN component: 495 means (33 x 15, `jevdrive.op_adapt.plan_index`) + 495 spreads.
  No hypotheses, no hypothesis probabilities (the 5 x 991 plan with a probability column in `jevdrive/openpilot/model.py` is the old supercombo).
- The spread is never trained by our recipe: the distillation index keeps 495 positions inside the plan slice, all of them the means
  (`op_adapt.distill_index` takes the first half of every MDN head), and the imitation term reads the means only. Decision 220 read it once
  (2 s lateral std, AUC 0.786 for collisions): a weak uncertainty signal, not a source of candidates.
- Candidates are therefore structured perturbations of the model's own plan, `lib/stack_cand.py` (K = 8): lateral offset d in
  {-1.0, -0.5, +0.5, +1.0} m along each pose's normal, ramped 0 -> d over 0-2 s; curvature gain g in {0.8, 1.2} and arc scale s in {0.75, 0.9}
  (`perturb.branch_poses`, copied as `branch_np`). Box selftest: torch vs numpy max |diff| 3.2e-14 m over 300 plans x 8 candidates.

## Reflex: self-state data and serving

Code: `lib/reflex_mem.py` (the efference memory, one rule for rows and drivers), `experiments/route_follow/scripts/rf_stack_self.{py,sh}`,
`rmem.load_pmodel` / `rmem.reflex_of`, `sh30_core.Core.plan(selfstate=)`, `pai_core.plan(selfstate=)`, `sh30_driver.py`, `pai_driver.py`.
`lib/reflex.py` unchanged. Box outputs `$DATA_DIR/runs/route_follow/stack/self/` (summary.json, replay_checks.json, smoke_checks*.json).

**Rule.** The driver stores every served decision (t0, the plan frame = rear-axle anchor in the local frame, the poses actually served:
after every serve-side switch, the JEV_TRACK reference when on). At a decision the previous one is the newest stored decision at least
0.45 s back (LAG 0.5 s): decision k - 1 on the nuPlan track (2 Hz); on the PAI track (10 Hz inference) the decision 0.5 s back, so dt
matches the rows, and the first 0.5 s has valid 0. rel = the anchor now in that decision's frame, dt from the t0s, v = the measured speed;
`reflex.selfstate`. Under JEV_TRACK with a tilted pose the frame and the pose now both take `track_ref.anchor`'s planar yaw.

**Offline rows** (`self/<P>.npz`: names = the tab's, same order; f (N, 14) float32; source pools also k, dt, k_prev). Each session's
drive.jsonl replayed through `reflex_mem.Memory`; previous served poses from its SH30_DUMP (unrounded), anchors / t0 from drive.jsonl,
v = the tab's `speed`. RCVL maps back by prefix (s -> on1_L18RB0, e0 -> on1_L18RSG; served, ego, speed identical on every row).

| pool | rows | valid | coverage | \|y_0\| median / p90 (m) | \|yaw_0\| median / p90 (rad) | \|x_0\| median / p90 | validation max \|diff\| |
|:--|--:|--:|--:|--:|--:|--:|--:|
| on1_L18RB0 | 17 429 | 16 409 | 0.9415 | 0.0052 / 0.0187 | 0.0025 / 0.0081 | 0.0186 / 0.0523 | 6.6e-5 |
| on1_L18RSG | 17 474 | 16 454 | 0.9416 | 0.0052 / 0.0186 | 0.0024 / 0.0076 | 0.0187 / 0.0525 | 6.3e-5 |
| on1_L18RSGallheld | 7 837 | 7 390 | 0.9430 | 0.0050 / 0.0183 | 0.0021 / 0.0072 | 0.0185 / 0.0530 | 6.2e-5 |
| on1_RCVL (all; src = human on every row) | 29 922 | 28 152 | 0.9408 | 0.0051 / 0.0183 | 0.0025 / 0.0079 | 0.0188 / 0.0529 | (mapped) |
| on1_RCVL pool s / e0 | 17 429 / 12 493 | 16 409 / 11 743 | 0.9415 / 0.9400 | 0.0052 / 0.0187, 0.0050 / 0.0178 | 0.0025 / 0.0081, 0.0024 / 0.0075 | | |

- Invalid rows = exactly the k = 0 rows (1 020 / 1 020 / 447); no k >= 1 row lacks its previous decision, the previous decision is
  always k - 1, dt = 0.500 s on every valid row, no session unmatched, no feature clipped.
- Validation (every session of the three source pools, not just 2): the features recomputed from drive.jsonl alone, as a served driver's
  log gives them (1e-4 m poses, the logged ego's speed), against the rows: max |diff| 6.6e-5 over all 14 features (limit 1e-4), valid
  flag identical on every row.
- The tab's `pose` (n, 4, 3) is the fed history: rear-axle poses at t0 - 1.5 / 1.0 / 0.5 / 0 s in the t0 frame (1e-4 m). pose[:, -2] is the
  previous decision's anchor in the current frame: it equals the inverse of the replayed rel within 5.0e-5 m / 5.0e-5 rad (rounding).
- Scale: on the nuPlan track the ego tracks the previous intent closely (|y_0| 5 mm median); the tau > 0 numbers carry the shape of the
  previous intent (RCVL |y_1.0| median 0.035 / p90 0.32 m). One PAI scene (STKRX1-test): |y_0| median 0.031 / p90 0.088 m, |x_0| 0.022 /
  0.091: the PAI body deviates ~6x more.

**Serving checks** (test checkpoints, new tags, sidecar = RSG-L18-s0's + `reflex: {"sf": 14, "d": 128}`: STKRX0-test = RSG-L18-s0 +
module at its zero init; STKRX1-test = the same with gate 1 and out weights N(0, 1e-3), seed 7).

- Offline replay (`rf_stack_self.sh replays`): `Core.plan` on 45 stored decisions of 3 held RSG-L18-s0 rollouts (the dumped JPEGs of
  decisions k-3 .. k, logged history / ego / route memory; eager, cpu synth), the self-state from reflex_mem over the stored rollout.
  This is the bitwise instrument (same inputs in every process).
  - (1) RSG-L18-s0 through the old code (worktree at 88a392cc) vs the new code: mu and poses bit-identical (max |diff| 0); new vs new repeat
    bit-identical.
  - (2) STKRX0-test vs RSG-L18-s0: bit-identical; fed row 234 = 20 + 200 + 14 (base 220); valid = 1 on all 45 replayed decisions (k >= 3).
  - (3) STKRX1-test moves the plan: max 0.077 / 0.087 / 0.103 m at 0.5 / 1.0 / 1.5 s (median 0.081 m), 0.14 m beyond; <= 0.5 m.
    STKRX1-test with JEV_REFLEX_OFF=1: bit-identical to RSG-L18-s0 (valid 0 everywhere).
- Closed-loop smoke (`rm_chain.sh readlist`, 3 held scenes x 18 decisions, SH30_COMPILE=0, pool jobs): RSG-L18-s0 repeated = itself
  exactly (54 / 54); STKRX1-test + JEV_REFLEX_OFF=1 = RSG-L18-s0 exactly (54 / 54); STKRX0-test = RSG-L18-s0 exactly in a repeat (54 / 54).
  Its first run differed in one scene from k = 4 (4.8 mm at k = 4 with identical logged ego / history / route / anchor, then the
  rollout drifted, 0.83 m by k 17). The repeat matched and the offline replay of the module is bit-identical, so this is
  closed-loop non-determinism outside the logged inputs (most likely the frames), not the module; it bounds what a "bit-identical"
  closed-loop read can claim. Logged `rx`: valid 0 at k = 0, 1 from k = 1 on (STKRX0 / STKRX1), 0 everywhere with JEV_REFLEX_OFF=1; fed
  row 234 on every decision; rx_dt 0.5 s; logged rx against reflex_mem replayed over the same drive.jsonl max 5.3e-5.
  STKRX1-test live: plans differ from k = 1 on as intended (the rollouts then diverge).
- PAI (pai_box.sh, 1 scene, STKRX1-test, default serving = JEV_TRACK gate + JEV_VCONT / JEV_LEAD): runs without error (199 inferences); fed
  row 234; valid from k = 5 on (the 0.5 s lag); rx_dt 0.5 s. A first run used the full-quaternion yaw for the pose now while the stored
  frame took track_ref's planar yaw on tilted poses (63 / 199 here): logged rx vs replay up to 0.20. Fixed (73dc4649: the pose now in the
  same convention); after the fix the replay matches to 0.018, limited by the log (anchor 1e-4 m, tilt 1e-3 rad). The scene ended
  offroad at 13.5 % of its duration: this is a test checkpoint with random weights, so that says nothing about the module.

Cost: about 1 GPU-hour (ckpt 1 min; replays 2 x ~6 min; 6 three-scene reads ~5 min each; 2 one-scene PAI runs ~4.5 min each) + ~2 CPU-min.
