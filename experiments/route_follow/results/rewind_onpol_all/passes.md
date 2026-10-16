# Perturbed recovery passes over onpol-train (2026-10-16): identity yield only

Prereg: [RECOVERY-ENVELOPE Amendment 7](../../plans/2026-10-16-recovery-envelope-prereg.md) (data-scaled recovery pool, arm RLFD).
**Stopped by the user.** The main session cancelled the handover stages, the merge, and the RLFD pack, training and reads at
09:10 JST, together with the rest of the old-recipe scaling arms. What exists is the two identity sweeps and a full-pipeline smoke.
No pool on1_RCVallD was built and RLFD was never trained.

## What a pass is

Pass 0 is decision 319 ([rewind_onpol_all.md](rewind_onpol_all.md)): RSG-L18-s0 alone, 30 decisions on the after280_op cache, the
held-read serving.

Pass A and pass B use the same instrument (`rf_rewind.sh train64 ostage`) with one change. The identity run is pushed once, by
JEV_PERTURB (perturb.py, decision 317): on decisions 2 and 3 the served plan is `branch_poses(plan, g, s)`, and after that the
policy drives on its own.
- Pass A: g 1.15, s 1.04.
- Pass B: g 0.85, s 0.96.

The planned handovers would have reused the same perturbation before h, with the expert from h. They required h ≥ 3, so every
handover state lies on the perturbed trajectory. Labels would have followed the d319 rule: completed, zero-free after h, and
within_tol. Units whose state before h fell within the identity tolerance of pass 0's state at the same h would have been
dropped as duplicates.

## Identity yield (all 1 031 scenes, 1 014 read; `passes/fails.json`)

| pass | strict turns | completed / event / off / short | failed (event + off) | also failed in pass 0 | **new failed vs pass 0** | new vs earlier passes | pass-0 failures not failing here | rewindable units (h ≥ 3) | new rewindable vs earlier |
|:--|--:|:--|--:|--:|--:|--:|--:|--:|--:|
| 0 (d319) | 166 | 71 / 65 / 17 / 13 | 82 | - | - | - | - | 155 (h ≥ 1) | - |
| A (g 1.15, s 1.04) | 136 | 46 / 64 / 17 / 9 | 81 | 57 | **24** | 24 | 25 | 131 | 38 |
| B (g 0.85, s 0.96) | 159 | 60 / 75 / 12 / 12 | 87 | 72 | **15** | 8 | 10 | 162 | 18 |
| union | | | **114** | | | | | 211 units | |

- **New failures.** Two perturbed passes raise the failed turns from 82 to 114 (+39 %). Pass A adds 24 new failed turns and pass B
  adds 8 more.
- **Rewindable units.** The (turn, Delta) units the expert could take over grow from 155 to 211 (+36 %).
- **Same turns, other outcomes.** Every strict turn of a pass is also a strict turn of pass 0 (turns_not_in_p0 = 0). The new
  failures are therefore the same junctions failing in other ways or at other times.
- **Turns reached.** Pass A reaches fewer strict turns within 15 s (136 against 166) and completes fewer (46 against 71). This is
  descriptive and not read further.

## How far the perturbation moves the policy (identity against pass 0's identity, same scene; 1 014 scenes)

| | k 4 (window end) | k 10 (4 s later) | k 20 | k 29 | speed ratio at k 4 |
|:--|:--|:--|:--|:--|:--|
| A: anchor distance, median / p90 (share > 0.5 m) | 0.16 / 0.26 m (0 %) | 2.39 / 4.59 m (90 %) | 2.87 / 10.98 m (87 %) | 3.24 / 14.88 m (87 %) | 1.073 [1.066, 1.082] |
| B | 0.16 / 0.25 m (0 %) | 2.49 / 5.23 m (90 %) | 2.75 / 13.41 m (88 %) | 3.39 / 16.64 m (90 %) | 0.929 [0.921, 0.935] |

- **Size of the push.** At the end of the window the push is small and plausible: 16 cm and ±7 % speed, as in d317.
- **Amplification.** The policy's own continuation amplifies it to metres within 4 s.
- **Divergence.** No scene's identity stays bit-identical to pass 0 (identical_to_p0 = 0 in both passes).
- **Most of the gap is longitudinal.** d315: the policy follows its own speed trend, so a speed change persists. The arrival
  timing against the replayed traffic changes, and the failure moves with it.

## Mechanism checks (smoke, 8 scenes, pass A settings; `passes/`)

1. **Switch change.** `sh30_driver.py` now accepts JEV_PERTURB together with JEV_EXPERT_FROM; the perturbation is never applied on
   an expert decision.
   - Perturb-only check: a stored d317 session (rewind_self smoke D3-g1.15s1.1, pushes at k 2 and k 11) re-served on the new code
     is bit-identical on both scenes (`bit_pt317.json`).
   - Both-unset check: plain identity on 2 scenes against d319's stored identity. One scene is bit-identical. The other first
     differs by 2 mm in the plan at k 4 and drifts from there (`bit_plain.json`). No serving code changed between the two commits
     except the inert gate, so this is the run-to-run spread that d319's identity tolerance exists for (d319: 9 / 78 handovers
     outside it).
2. **Handover detection.** n-read takes the handover as the expert's first decision when the session has an expert. Before this
   fix it took the first perturb record, which marked every unit invalid. d316 / d317 / d319 sessions read as before.
3. **Smoke yield and tolerance.**
   - The handover runs dumped frames directly, with no re-run: 8 / 8 valid, 7 / 8 within tolerance, 6 label units, 79 rows. No
     duplicate units (2 had the same h as pass 0, and their states differed by 1.3-2.4 m).
   - Merge into pass 0 (952 + 79 = 1 031 rows), pack and a 20-step RLF training on the merged pool all ran.
   - The pack's R weight equals RLF's (`pool=r:4.97166`; the 79 smoke rows are all dev rows), so the dose arithmetic holds.

## Are the identity rollouts usable as own-state rows?

- **Not as they are.** The identity launches ran with SH30_DUMP = 0, and nsim deletes `rollout.asl`.
- **What exists.** On the box, `$DATA_DIR/runs/route_follow/rewind_passes/p{A,B}/id/c*/run` hold only drive.jsonl (inputs,
  served plans, anchors, ego), the simulator results and the aggregate summary. There are no camera frames, no dumps and no
  tokens.
- **What it would take.** Using them as rows needs a re-run with the dump (`rf_recov_roll.sh nsim ... 1`, identity mode plus
  JEV_PERTURB = `p{A,B}/perturb.txt`). The sessions are deterministic up to the run-to-run spread above, so a re-run reproduces
  them within tolerance. That costs about 1 card-h per pass.
- **What is stored.** Pass A also has tfail.json, from_D2 / D3 and the expert packs. Pass B has tfail.json and from_D2 / D3 (no
  packs). One D2 chunk of pass A ran.

## Cost

- **GPU:** 76 jobs, 9.0 job-h ≈ 2.3 card-h (4 per card). That covers the identity sweeps A and B, the smoke, the bit checks and
  one D2 chunk.
- **CPU:** 0.26 job-h.
- **Timing:** about 4 h of wall time were spent queued behind the RECOVERY-ENVELOPE reads.

## Code

- `experiments/alpasim/lib/sh30_driver.py`: JEV_PERTURB with JEV_EXPERT_FROM.
- `experiments/route_follow/scripts/rf_rewind.sh`, ostage env vars:
  - RW_PERT: per-scene perturb file, handovers from k + 1.
  - RW_DUMP: the D launches dump frames and write onpol_sim.pkl.
  - RW_BASE: the dups job.
  - RW_ROWS: rowsR / tok / rowsB.
- `experiments/route_follow/scripts/rf_rewind.py`:
  - `n-tfail --hmin`.
  - The n-read handover fix.
- `experiments/route_follow/scripts/rf_passes.py`: dups, bitid, tally, fails.
- `experiments/route_follow/scripts/rf_recov_roll.sh rmerge`.
- `experiments/route_follow/scripts/rf_recov.sh` arm RLFD (pool on1_RCVRLFD from on1_RCVallD).

Test artefacts left on the box:
- Pools on1_RCVpS, on1_RCVallS, on1_RCVRLFS.
- Checkpoint RCVRLFS-smk.
- `recovery_envelope/flags_RCVRLFS.txt`.
