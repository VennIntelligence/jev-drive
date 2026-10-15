# RECOVERY-ENVELOPE pilots (2026-10-16)

Prereg: [plans/2026-10-16-recovery-envelope-prereg.md](../../plans/2026-10-16-recovery-envelope-prereg.md) (Amendments 1-3). Code: `scripts/rf_recov.{py,sh}`,
`experiments/onpol_nuplan/lib/on_envelope.py`, `rm_train.py --env-lam / --env-lo-scale / --on-lon-all`. Box run dir `$DATA_DIR/runs/route_follow/recovery_envelope/`
(`gate/<tag>.json` per pilot, `lab_L18RSG.{npz,json}`, `pack_*.json`, `flags_*.txt`, `fwd/`, `gates.json`, `jobs.txt`). Per-pilot JSON copies: `gate_<tag>.json` here (box repo).

## Bit identity (gate 1; `gates.json`)

RSG-L18 recipe, 60 steps, old code = the prereg commit 6a2fdac6 (before any trainer change):

| check | scalars compared | max abs diff (scalars / weights) | on_hash_60 | identical |
|:--|--:|:--|:--|:--|
| G0 every new flag off, old vs new code | 89 | 0 / 0 | ea8cacc3e451c577 = RSG-L18-s0's stored hash | yes |
| G0 new code twice | 89 | 0 / 0 | same | yes |
| G2 `--env-lam 1` on on1_RCVL: `--on-mode self` vs `--on-mode mixed --on-lon-all` (every row self) | 38 (loss / dev) | 0 / 0 | e478c6a76d3d5294 both | yes |

`--env-lo-scale 1` adds no operation (the multiply is skipped).

## Rows

- **L envelope rows** (`rf_recov.py label`, RSG-L18-s0's own pool on1_L18RSG, 17 474 rows): labelled 11 430 rows / 927 scenes / 96 logs. Train-log rows by class
  (first reason wins: stop > turn > lead > free): stop 1 344 (dev 127), turn 1 988 (155), lead 4 739 (470), free 4 422 (929). Stop-line rows: 1 471 rows / 130 scenes in all,
  fed speed p50 2.9 / p90 6.7 / max 10.6 m/s; 718 rows at >= 3 m/s, 229 at >= 6 m/s (the inventory's 451 / 77 counted only the first eligible line within 40 m at the
  decision; the envelope takes 0-60 m and both the human-stopped and red-throughout-4-s rule).
- The policy's own served plans against the bounds (share of rows violating, mean excess): stop 10.5 % / 0.10 m/s, turn 14.5 % / 0.09, lead 6.3 % / 0.08 (time form,
  Amendment 1; the registered any-time sweep: 90 % / 2.95 m/s, not trained on), free lower bound 54 % / 0.59 m/s short.
- **Pool on1_RCVL** = on1_L18RB0 (17 429 S rows, self) + 9 980 train-log envelope rows (self, envelope.npz); on-k 20 = 13 S + 7 envelope rows per batch
  (stop 1, turn 2, lead 2, free 2): `--on-weight env=stop:0.99754,env=turn:1.34879,env=lead:0.56581,env=free:0.60637`.
- **R0 rows** (pool on1_RCVd316, built by `rf_recov_roll`: d316's D2 / D3 handovers re-run with frame dumps; units whose expert continuation completes with no zero
  after h): 361 rows / 38 units / 13 logs: 38 rows at k = h (the policy's exact state), 323 at k > h (expert-driven states the policy is fed); D2 185, D3 176.
  Dev logs (crc32 rule) hold 115 rows / 12 units / 3 logs; 246 rows / 26 units train. Pool on1_RCVR0 = S rows + 246 R rows (both), on-k 16 (13 + 3), per-log cap 0.10
  (10 logs: the cap makes the R draw uniform over logs).
- **Fast approaches** (pool on1_RCVfast, `rf_recov_roll.sh fast`, results/recov_roll/): JEV_PERTURB g 1, arc scale s 1.15 / 1.20 for 2 decisions (s 1.10 gave only +19 %
  in the smoke and was dropped), at the decision k >= 2 where the policy is 40-60 m (else 30-70, else 20-30 m, tagged) before a qualifying stop line (84 of the 85
  inventory scenes) or a strict-turn entry. The 64 train64 turn scenes could not be used (their entry is 2-28 m ahead already at decision 0); 64 other onpol-train
  turn scenes with the entry 40-60 m ahead were taken instead. Speed at the end of the push window vs the unpushed policy at the same decision: L115 stop 7.25 vs
  5.66 m/s (x 1.28), turn 7.78 vs 6.10 (x 1.28); L120 stop 7.80 vs 5.66 (x 1.38), turn 8.32 vs 6.10 (x 1.37). The push mostly wears off before the line (speed at the
  line 3.6-3.7 vs 3.5-3.7 m/s; at turn entries +0.5 m/s). 6 498 rows (L115 3 261, L120 3 237; stop 3 905, turn 2 593; dev 1 366); envelope-labelled 4 352 rows /
  142 scenes (stop 2 302 train + 163 dev, turn 1 334 + 94, lead 365, free 46); stop-line rows at >= 6 m/s 709 (on1_L18RSG: 229), max 13.1 m/s; served plans violate
  the stop bound on 33 % of stop rows (0.39 m/s mean excess). Pool on1_RCVLF = on1_RCVL's rows + 4 047 train-log fast rows (1 draw / batch).
  **LF pilot fails** (dev ADE 0.5383 against 0.538, free ratio flat): not launched.

## Pilot gates (1 000 steps, seed 0, vs RSG-L18-s0)

Lines: dev ADE <= 0.538; G-rely true; turning = equal-arc 4 s heading toward the turn, pilot - RSG mean >= -0.3 deg (Amendment 3; the 4 s heading is reported);
L: stop-line excess drop >= 20 %, A-speed (turn-entry excess) CI high < 0, free-road plan / fed speed ratio CI low > 0; R: imitation ADE to the expert target on
held-out-log R rows drop >= 20 %. Dev rows: stop 127 (7 logs), turn 155 (9), lead 470 (11), free 929 (15); turning rows: on1_L18RSGallheld decisions 0 / 1 within
[entry - 40, entry + 10) m of a >= 45 deg turn, 276 rows / 139 scenes.

| pilot | dev ADE | G-rely d_junction | G-rely | 4 s heading | equal-arc heading [CI] | stop excess (m/s) | A-speed | lead excess | free deficit | free ratio (median), diff [CI] | R imitation ADE | result |
|:--|--:|:--|:--|--:|:--|:--|:--|:--|:--|:--|:--|:--|
| RCVLP1 (W 1) | 0.5713 | +0.0027 | False | -0.20 | -0.09 [-0.29, +0.12] | 0.129 -> 0.103 (-20 %) | 0.0124 -> 0.0045, CI hi -0.0020 | 0.049 -> 0.063 (+30 %) | 0.615 -> 0.089 | 0.961 -> 1.060, +0.177 [+0.134, +0.227] | - | fail: dev ADE, rely |
| RCVLP3 (W 3) | 0.5976 | -0.0073 | False | -0.50 | +0.17 [-0.11, +0.43] | 0.129 -> 0.089 (-31 %) | 0.0124 -> 0.0034, CI hi -0.0023 | 0.049 -> 0.059 (+21 %) | 0.615 -> 0.066 | 0.961 -> 1.076, +0.196 [+0.148, +0.253] | - | fail: dev ADE, rely |
| RCVLP03 (W 0.3) | 0.5506 | +0.0145 | True | +0.05 | -0.26 [-0.43, -0.08] | 0.129 -> 0.123 (-4 %) | 0.0124 -> 0.0083, CI hi -0.0010 | 0.049 -> 0.060 (+24 %) | 0.615 -> 0.250 | 0.961 -> 1.022, +0.108 [+0.082, +0.138] | - | fail: dev ADE, stop |
| RCVLP01 (W 0.1) | 0.5318 | +0.0227 | True | -0.19 | -0.38 [-0.54, -0.19] | 0.129 -> 0.133 (+3 %) | 0.0124 -> 0.0107, CI hi -0.0004 | 0.049 -> 0.056 (+15 %) | 0.615 -> 0.463 | 0.961 -> 0.988, +0.043 [+0.033, +0.054] | - | fail: turn, stop |
| RCVLP1u (W 1, upper only; diagnostic) | 0.5364 | +0.0238 | True | -0.90 | **+0.63** [+0.44, +0.80] | 0.129 -> 0.075 (-42 %) | 0.0124 -> 0.0033, CI hi -0.0025 | 0.049 -> 0.029 (-41 %) | 0.615 -> 0.710 | 0.961 -> 0.936, -0.029 [-0.038, -0.021] | - | fail: free (by construction) |
| RCVLP05l01 (upper 0.5, lower 0.1) | 0.5316 | +0.0257 | True | -0.28 | +0.18 [+0.04, +0.31] | 0.129 -> 0.110 (-14 %) | 0.0124 -> 0.0063, CI hi -0.0018 | 0.049 -> 0.042 (-13 %) | 0.615 -> 0.496 | 0.961 -> 0.980, +0.030 [+0.023, +0.037] | - | fail: stop |
| **RCVLP1l01 (upper 1, lower 0.1)** | 0.5335 | +0.0251 | True | -0.42 | **+0.42** [+0.25, +0.56] | 0.129 -> 0.097 (-25 %) | 0.0124 -> 0.0043, CI hi -0.0023 | 0.049 -> 0.036 (-25 %) | 0.615 -> 0.522 | 0.961 -> 0.973, +0.020 [+0.016, +0.026] | - | **PASS** |
| RCVLFP1l01 (LF: L + fast rows, upper 1, lower 0.1) | 0.5383 | +0.0247 | True | -0.69 | +0.71 [+0.51, +0.87] | 0.129 -> 0.059 (-54 %) | 0.0124 -> 0.0047, CI hi -0.0020 | 0.049 -> 0.029 (-39 %) | 0.615 -> 0.603 | 0.961 -> 0.955, -0.001 [-0.005, +0.005] | - | fail: dev ADE (0.5383), free |
| RCVR0P (R0, 3 rows / batch) | 0.5285 | +0.0238 | True | +0.40 | +0.34 [+0.15, +0.51] | - | - | - | - | - | 1.548 -> 1.386 (-10.5 %) | fail: R imitation |
| RCVR0Pk6 (R0, 6 rows / batch) | 0.5310 | +0.0167 | False | +0.99 | +0.43 [+0.16, +0.68] | - | - | - | - | - | 1.548 -> 1.305 (-15.7 %) | fail: R imitation, rely |
| RCVRP (R, all-scene rows, 3 / batch) | 0.5284 | +0.0246 | True | +0.04 | +0.17 [+0.05, +0.28] | - | - | - | - | - | 1.377 -> 1.259 (-8.6 %) | fail: R imitation |
| RCVRPk6 (R, 6 / batch) | 0.5309 | +0.0194 | True | +0.69 | +0.42 [+0.25, +0.57] | - | - | - | - | - | 1.377 -> 1.207 (-12.3 %) | fail: R imitation |
| RCVRLP1l01 (RL, R 3 / batch, upper 1, lower 0.1) | 0.5366 | +0.0227 | True | -0.25 | +0.75 [+0.51, +0.96] | 0.129 -> 0.093 (-28 %) | 0.0124 -> 0.0031, CI hi -0.0027 | -> 0.039 (-20 %) | 0.615 -> 0.520 | 0.961 -> 0.972, +0.021 [+0.016, +0.028] | 1.377 -> 1.213 (-11.9 %) | fail: R imitation |
| RCVRLPk6_1l01 (RL, R 6 / batch) | 0.5336 | +0.0219 | True | -0.03 | +0.73 [+0.37, +1.07] | 0.129 -> 0.094 (-27 %) | 0.0124 -> 0.0028, CI hi -0.0029 | -> 0.037 (-24 %) | 0.615 -> 0.509 | 0.961 -> 0.975, +0.025 [+0.020, +0.032] | 1.377 -> 1.147 (-16.7 %) | fail: R imitation |

R / RL on the all-scene rewind rows (Amendment 4: on1_RCVall, 952 rows / 84 units / 35 logs, label == True, within_tol only, no shape gate; 809 train rows,
143 dev rows / 4 logs). The same imitation read on the 809 TRAINING rows: RSG 1.448 m, R k3 -10.3 %, R k6 -16.5 %, RL k3 -11.3 %, RL k6 -17.5 %: the held-out
drop equals the training-row drop, so the R line fails on how far 1 000 steps move the targets at all, not on transfer. The 20 % line is the main session's to
revisit; by the registered rule R and RL stop and were not launched.

Turning instrument check (same offline read, pilot - RSG-L18-s0, 4 s / equal arc): APP1X-L18-s0 -1.70 / -1.64 [-2.16, -1.00] (the closed-loop read was -1.0 to -1.5;
by bin -1.48 at 0-10 m and -3.05 at 10-20 m before the entry against -1.5 / -3.1 in decision 303 Amendment 5), APP0C-L18-s0 (W 0 control) -0.31 / -0.21,
RSG-L18-s1 -0.12 / -0.11. The -0.3 deg line sits at the checkpoint-to-checkpoint spread. The 4 s heading falls mechanically when a plan slows before a turn
(RCVLP1u: 4 s arc -0.84 m, 4 s heading -0.90, equal-arc +0.63), hence Amendment 3.

Reading:
- The envelope's harm sits in the free-road lower bound: at W >= 0.3 it pushes the plans faster (free ratio 0.96 -> 1.02-1.08) and costs dev ADE (0.551-0.598) and
  route reliance; upper bounds alone at W 1 keep dev ADE (0.536) and G-rely and make the plan turn MORE per metre (+0.63 deg). Upper W 1 + lower W 0.1 passes every gate.
- R0 learns the expert's recoveries on the training units but transfers weakly to the 3 held-out logs (-10.5 / -15.7 %); its plans turn more (+0.34 / +0.43 deg at equal
  arc). 246 training rows from 26 units (10 logs) are thin; the all-scene rewind (decision 319) is the material R and RL wait for.

## Decisions taken in this session (main session to check)

- Amendment 1 (lead bound in time), Amendment 2 (dose extension W 0.3 / 0.1 + upper-only diagnostic), Amendment 3 (equal-arc turning read decisive; split upper /
  lower weights; R dose k 6), each committed before the pilots it concerns. All three were made after seeing earlier pilots of this round.
- **L launched** (`rf_recov.sh full L 1l0.1`, 01:06 CST): RCVL-L18-s{0,1}, 2 500 steps, `--env-lam 1 --env-lo-scale 0.1`, pool on1_RCVL (no fast-approach rows: the pushed
  rows were not built when L passed; the user's instruction was to launch on pass). Then held 460 @18 (6 chunks / seed), public 1491 @10 (10 chunks / seed), PAI 281
  seed 0 on the box (8 chunks; waits for a 118-scene download and the box RSG-L18-s0-gate reference read), extracts, report -> `RCVL/lines.md`.
- **R0 stopped** (fails its imitation gate at both doses). **RL0 not piloted** (Amendment 3: RL0 combines L's weight with R's chosen dose; R has none).
- **LF not launched** (fails dev ADE by 0.0003 and the free-ratio gate). **R / RL (all-scene rows) not launched**: imitation -8.6 / -12.3 % and -11.9 / -16.7 % at k 3 / 6.
- Launch commands if the main session overrides the R line: `PRIO=7 bash experiments/route_follow/scripts/rf_recov.sh full R k6` and
  `... full RL k6_1l0.1` (each: 2 trainings + 42 read / PAI / report jobs, ~13 GPU job-h).

## L full round (RCVL-L18-s{0,1}, 2 500 steps; partial report `RCVL/lines_partial.md`, PAI pending)

Checkpoints: dev ADE 0.5362 / 0.5356, G-rely true. Held 452 (seed means vs RSG-L18): zeros **+4.0 [-2.9, +10.7] (line <= +3: FAIL)**, at-fault +0.5, GT -0.027 m,
score +0.004; held turns 12 + 14 vs 16 + 14 (-2.0 [-4.5, +0.5]). Public guard: score -0.0016 [-0.0079, +0.0049], zeros -5.0: PASS. **Not adoptable (H1)**. The PAI 281
seed-0 read (8 chunks) and the box reference RSG-L18-s0 read wait for the 118-scene download (job 1016-002200-fb8a); they answer the mechanism question only
(cor_route_end, rear-ended), and the full report replaces lines_partial.md when they land.

## Cost

Pilots and gates: 14 training jobs (9 pilots x ~4.4 min + 5 gate smokes) 0.73 GPU job-h, 21 forward reads 0.16 job-h, 14 CPU jobs 0.09 h; ~19.6 GB per training job
(about 0.2 card-h of VRAM-weighted card time). Wall 00:19-01:06 CST, mostly queueing behind the all-scene rewind.
