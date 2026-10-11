# ONPOL-NUPLAN round 2: long-horizon read, on-policy S rows on the YR10m10-F base, two more seeds (pre-registration, 2026-10-12)

Lane ONPOL-NUPLAN, main GPU box. Follows decision 259, `tmp/night-handoff-2026-10-12.md` (items 2, 3, 4; written by the coordinating
session for the user, who is asleep) and experiments/alpasim/results/aud1_*.md (official submission #1 scored 0.640 on the private
suite against 0.935 public). Written and pushed before any score, pilot read or row statistic of this round is read. Local closed loop
only; nothing is submitted; the last official slot is not touched. Encoder frozen. **A failed gate stops its item and is recorded; no
deviation of the Amendment 5 kind.** Choices the handoff left open are the main session's proposals, marked (P). Budget: about 12
card-hours. The split, the row pipeline, the label definitions, gates R1 (as amended) / R2, zero classes, fixed / broken and CIs are
those of [2026-10-11-onpol-prereg.md](2026-10-11-onpol-prereg.md) unless said otherwise. No `-ON` checkpoint is reported on navtest /
navhard.

## A. Long-horizon read (handoff item 2; descriptive, no pass line)

Question: does the ordering of candidates hold or change as scenes lengthen? The public scenes are 10 decisions (5.5 s) because of the
trajdata window (`num_timesteps_before 30 / after 80`); `n_sim_steps` caps at 200.

- Smoke: 10 scenes of the `train` split, `P2H10S-F-s0`, the window widened by a wizard override passed as the existing chains pass
  overrides (the shipped config is not edited): `num_timesteps_after` 160, or the largest value for which the 10 scenes run; it is
  recorded whether the recordings, the reconstructions and the route extend, how many decisions a scene has, wall time and memory per
  scene. The smoke's result is recorded whichever way it comes out. If no scene runs longer than the shipped 10 decisions, item A stops.
- Gate A1 (mechanics, on the smoke): at the shipped window the override path reproduces the shipped run on the same 10 scenes (no zero
  flip, scores within 0.01); with the wider window every rollout ends without a driver error and decisions 0..9 are served the same
  inputs as in the shipped-window run up to the rounding of Amendment 1 (plan points within one fp16 ulp or 0.03 m on 99 % of
  decisions). A1 failing stops item A.
- Read: `YR10m10-F-s0 / s1`, `P2H10S-F-s0 / s1`, `SSELFnoL-ON-s0 / s1` on the 460 `held` scenes at the smoke's length (held scenes that
  cannot run at that length are listed and dropped for every checkpoint alike). Reported per checkpoint family at the long horizon and,
  from this lane's and the stored 10-decision reads, at the short one: mean scene score, zeros by kind, at-fault events, slow scenes,
  progress, mean distance to the logged trajectory (GT distance), its growth per decision over decisions 4..20, and the same quantities
  restricted to the first 10 decisions of the long rollouts. Paired differences between the three families with scene and log CIs.
- What is said afterwards: the ordering of the three families by zeros, at-fault events, mean score and GT distance at each length, and
  whether a paired difference changes sign with its CI excluding 0. Descriptive: two seeds, development scenes, a simulator setting the
  official suite may not share.

## B. On-policy S rows on the YR10m10-F base (handoff item 3)

`YR10m10-F` = the P2H10 recipe + yaw-rate rows (`joint_train.py --yr-mass 0.1`, NAVSIM standard; 115 normal incl. 13 yaw-rate rows per
batch of 128; no lesson S). Question: does the one result of decision 259 (S hinges on on-policy rows) hold on a base without S?

- Rollouts: `YR10m10-F-s0 / -s1` on `train` with the frame dump; pool `on1_trainY` by `on_rows.py`; gates R1 (amended) and R2 on it. A
  failed gate stops item B.
- Arms (seeds 0, 1; from the archived `YR10m10-F-s<seed>`; the base's own recipe continued, 10 000 steps, fresh optimiser):
  `YSELFnoL-ON` = 13 on-policy rows per batch drawn from `on1_trainY` (stream `[seed, 0, 53]`), hinge-only, carrying S's two hinges on
  the own plan exactly as `SSELFnoL-ON` (agent hinge against the simulator's actor boxes and road-and-lane raster, lambda 10 each,
  shape-only, x 3, no imitation target); the offline rows stay the YR recipe's (89 normal / 13 yaw-rate / 13 on-policy, and no agent
  hinge on offline rows: the base has none). `YNONE-ON` (P) = the same continuation without on-policy rows. Control: `YR10m10-F-s0 / s1`.
  No L, no imitation arm (decision 259).
- Trainer gates before any pilot: identity (60 steps, no `--init`, no on-policy rows: bit for bit `joint_train.py --yr-mass 0.1`); the
  init loads every tensor and step-0 dev ADE equals the archived run's; `YSELFnoL` rows carry both hinges and no imitation term.
- Pilot (3 000 steps, seed 0, both arms): finite losses, composition, `dev_drift_off` <= 0.30, dev ADE of `YSELFnoL` <= the `YNONE`
  pilot's + 0.010 m, then the 48-train-scene mechanics smoke. **A failed gate stops item B; the pilot table is then the result.**
- Read on `held` (460 scenes): `YSELFnoL-ON-s0 / s1`, `YNONE-ON-s0 / s1`, control `YR10m10-F-s0 / s1`, one driver path, this lane's runs.
  Lines, as round 1: (i) `YSELFnoL` has fewer seed-mean zeros in total and fewer at-fault zeros than the control; (ii) at-fault zeros not
  above the control's (hard gate); (iii) fixed - broken against the control exceeds the null. The control has two seeds, so the null
  (P) is the maximum of |fixed - broken| over the control's s0-against-s1 single-seed comparison and round 1's four-seed null (+2).
  "Holds on the YR base" = (i), (ii) and (iii). Beside it: `YSELFnoL` against `YNONE` (rows against steps), mean score, slow scenes,
  progress, GT distance, corridor zeros and the > 45 deg bucket (the base has no S, so S's corridor cost may appear here).

## C. Seeds 2 and 3 of `SSELFnoL-ON` on P2H10S-F (handoff item 4; queued after B)

`SSELFnoL-ON-s2 / -s3` from `P2H10S-F-s2 / -s3`, the round-1 recipe and pool `on1_train` unchanged, read on `held`. Reported: the
four-seed arm against the four-seed control with round 1's lines (zeros, at-fault zeros, fixed - broken of arm seeds 2, 3 against
control seeds 2, 3 and of all four against all four, the null as registered there), and each new seed alone. "Decision 259 firms up" =
the four-seed paired difference in zeros keeps its sign with a scene CI excluding 0 and seeds 2, 3 alone have fewer zeros than their
control seeds; otherwise 259 is demoted in place. Item C starts only after the official submission #2 result is recorded and the
handoff's branch does not stop it (the Tokyo status command is not run by this lane; it waits for the recorded result).

## Budget (card-hours, estimates)

| Item | Estimate | Basis (round 1) |
|:--|--:|:--|
| A: smoke + 6 checkpoints x 460 scenes at about twice the length | 2.5 - 4 | 0.15 - 0.22 per checkpoint at 10 decisions |
| B: rollouts 2 x 1031 scenes, pool, gates, 2 pilots, 4 trainings, 6 checkpoints x 460 scenes | 5 - 6 | 0.7 + 0.2 + 0.4 + 2.5 + 1.2 |
| C: 2 trainings + 2 x 460 scenes | 1.5 | |
| **Total** | **9 - 11.5** | cut order above 12: C, then `YNONE`'s read |
