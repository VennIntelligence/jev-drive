# Log-consistent margins for the body lesson S: one pilot (pre-registration, 2026-10-11)

Lane BODY1 (topic body1), loss arm 4.3. **One pilot, decided by the user on 2026-10-11** after decision 254; it is the fifth trained variant
of 4.3 (Amendment 4 pilot; Amendment 5; Amendment 6 = S; Amendment 7 = R, stopped at its pilot gate) and is reported as such. A separate file
rather than an Amendment 8: Amendment 7 closed the arm "without a new decision by the user", and this pilot is that decision; the S-DROP lane
set the precedent of a separate pre-registration on the same trainer. Pushed together with the code before any job of it runs; at this commit
no checkpoint, G0 number or read of it exists. Only the code check on synthetic rows of section 3 has been run.

Read before writing: `lib/loss43.py`, `lib/agent_hinge.py`, `lib/drivable_hinge.py`, `ot_rows.off_hinge`, `lib/route.py`, `scripts/bd4_train.py`,
`experiments/alpasim/scripts/joint_train.py`; [2026-10-10-body1-prereg.md](2026-10-10-body1-prereg.md) Amendments 4 (item 10), 6 and 7;
decisions 230, 233, 234, 244, 254; [results/route_pilot.md](../results/route_pilot.md), [results/navtest_warp.md](../results/navtest_warp.md),
[results/shape_pilot.md](../results/shape_pilot.md); `research/self-consequence/design.md` section 3.

## 1. Question

S (`P2H10S`) cuts own-plan contacts but pushes the plan outward on turns over 45 deg, which in the closed loop shows as "left corridor" zeros
(decision 230 point 5, 233, 234, 244 question 2, 254: corridor +14.0 [+3.5, +24.5] on 1 491 scenes). Its two new hinges use fixed margins:
road 0.3 m (road-and-lane raster, hinge-only rows), agent 0.3 m inside the ego corridor and 0 m outside (imitation rows of the train logs and
hinge-only rows). **Hypothesis**: where the logged driver passed closer to the inner edge or to an object than the fixed margin, the hinge fires
on the log's own path and pushes the plan outward; on hinge-only rows no imitation target pulls it back (decision 244: the shift lives on those
rows, about 0.10 m from the road hinge and 0.06 m from the agent hinge). Making the margins log-consistent should remove the drift and keep the
lesson. This is the "the log never violates" rule of design.md section 3 applied to S.

Not re-tested: that S's shift exists (234, 244), that a dead-band tube to the log does not reach it (234), that the shape-only switch restores the
speed profile (232).

## 2. The change: one flag, `bd4_train.py --logmargin` (default off), tags `P2H10SM-*`

For every row r and new hinge X: **m_X,r = min(m_X0, c_X,r - 0.05 m)**, no lower clip (may be negative). c_X,r is the minimum clearance of the
row's logged path to the hinge's labels, measured with the hinge's own geometry (`lib/logmargin.py`):

- **Logged path**: the logged t0 pose followed by the 8 logged future poses the row already carries (`S.fut`), densified to 0.1 s by the
  plan's own interpolation (`drivable_hinge.interp_matrix`, 41 steps, t0 included). On a log row the logged t0 pose is the origin. On an
  off-track row (`ot1` / `yr1` / `bd4`) it is (0, 0, 0) of the logged frame moved by the row's `off` (`lib/route.log_t0`), the convention of
  design.md 3.1 and of Amendment 7's route term. So the logged path of a hinge-only row is the same world path as its source log row's, and
  its c must equal the source row's up to the cache rounding (checked, G0.b (ii)).
- **Agent**: min over the 41 steps and the counted objects of `AgentHinge.distances` (the extended ego box, boxes with motion, the counted mask
  of the logged pose), in the row's own frame (`OffAgentHinge`'s moved boxes); +inf where the logged path counts no object. The cap applies to
  both of the hinge's current margins (0.3 m in the corridor, 0 m outside): m = min(m0, c - 0.05).
- **Road**: min over the 41 steps x 4 footprint corners of the row's road raster SDF (after note (x)'s fallback to the NAVSIM raster), the
  path mapped into the raster's logged frame (`ot_rows.off_hinge`'s mapping) and densified from the logged t0 pose, the raster's origin.
- **Why the logged t0 and not the row's own t0 on hinge-only rows.** The plan of a hinge-only row is densified from the perturbed pose, because
  that is where the car is; the logged driver never was there. The rule concerns the log, so the margin is a property of the logged path. If
  the plan of a hinge-only row reproduced the logged future from the perturbed pose, the first 0.5 s can still touch the hinge: that part of
  the lesson (recover from the perturbation) is kept. How often the other reading (clearance of the path from the row's own t0) would give a
  lower margin is reported in G0 (no line).
- Unchanged: P2H10's own on-log drivable hinge (NAVSIM raster, 0.3 m), every weight (agent 10, road 10, ho_w 3), the rows (4 / 4 / 5 hinge-only
  per batch, `--ho-excl navsim/body1-val-logs`), shape-only gradients, the extended ego box (front edge + 0.3 m), the counted and corridor
  rules. The margins are computed once before the first step (no gradient). Code: `lib/logmargin.py` (new), `lib/loss43.py` (`lm_agent`,
  `lm_road`, None = the code path of S), `scripts/bd4_train.py` (`--logmargin`, `--lm-out`), `lib/agent_hinge.py` (an optional start pose `p0`
  of the densification in `distances`; None = unchanged).

## 3. Checks before the pilot (G0)

- **Already run, code only (synthetic rows, `tmp/`, not data):** a log row and its off-track twin with the same token give the same c to
  3e-6 m (agent) and 4e-6 m (road); the logged path gives zero hinge with the new margins; with no cap and no start pose the capped hinge is
  `per_step` bit for bit; the road clearance of a log row through the logged-frame path equals `OffHinge.margins` exactly.
- **G0.a identity (gate).** The S recipe with the flag off, 60 steps on shard s2 (`LM-IDENT-OFF`), against the stored `B43A7-IDENT-S1`
  (`bd4_train.py ident`: every logged scalar and every trained weight bit for bit). Why shard s2 and a stored run rather than a fresh
  before / after pair on s2 + s3: the stored run is the lane's S-recipe identity reference (Amendment 7: `B43A7-IDENT-S0` (S code path) =
  `B43A7-IDENT-S1`; S-DROP: = `SDROP-IDENT-S`), so a match shows both that the flag-off path is unchanged and that today's checkout reproduces
  the code path that trained the archived pilots, which is what a comparison with the archived `P2H10S-P-s0` needs; the flag-off path runs no
  new operation whatever the data. Missed: the lane stops before the pilot, the cause is diagnosed (a run of the unedited checkout), reported.
- **G0.b (gate), computed by the pilot's own trainer before its first step on every row a batch can carry the hinge on** (agent: imitation
  rows of the train logs with a label + hinge-only pool rows with a label; road: hinge-only pool rows with a label):
  (i) the logged path, through the same functions the loss calls, with the new margins, gives zero hinge on >= 99.5 % of the rows, for each
  hinge; (ii) a hinge-only row's c equals its source log row's c (both +inf, or within 1e-3 m) on >= 99.5 % of the hinge-only rows, for each
  hinge (agent: the source row's own labels in the logged frame; road: the source row's logged future on the same token's raster). Exceptions
  are listed (row, family, values, the step of each minimum). Missed: the trainer exits 2 before the first step; nothing is trained.
- **G0.c (diagnostic, no line; reported whatever it says).** For each hinge, on all rows, on rows with |logged 4 s heading change| > 45 deg,
  on hinge-only rows (and per family): the share whose logged path violates S's fixed margin; the share with c < m0 (0.3 m); the share whose
  margin is lowered (c < m0 + 0.05); the share with a negative margin; quantiles (5, 25, 50, 75 %) of the lowered margins. Reported with it:
  the logged future fed as the plan of the training loss (densified from the row's own t0), share with a non-zero hinge under S's and under the
  new margins; and the other reading of section 2 (share of hinge-only rows whose own-t0 path has a lower clearance than c).

## 4. The pilot

Exactly the S pilot's protocol (`P2H10S-P-s0`, meta.json): `bd4_train.py train --seed 0 --data navtrain_full.s2of12 navtrain_full.s3of12
--steps 3000 --eval-every 1000 --ho ot1:4,yr1:4,bd4:5 --agent-lam 10 --ho-w 3 --ho-excl navsim/body1-val-logs --shape`, plus `--logmargin`;
batch 128, warmup 300, split `navsim/op-parity-full`, warp frames. Tag `P2H10SM-P-s0`. Compared with the archived S pilot `P2H10S-P-s0` and the
archived base pilot `P2H10-P-s0` (switch-off, same trainer, data, steps, seed). Nothing else is trained: no second seed, no full run.

## 5. Reads (one each, after the pilot; tables under `$DATA_DIR/runs/body1/logmargin/`, copied to `results/logmargin/`)

| Read | Tool (unchanged) | States |
|:--|:--|:--|
| W2 = turn rows > 45 deg whose own plan has a pose more than 2 m outside the logged path within 4 s; leaves the 4 m tube; mean lateral at 4 s | `prog_ol.py states` + `route_ol.py` | navtest on-log tokens, warp frames (1 517 > 45 deg); hold states of shards s2 + s3 (on-log 207, four families 686) |
| own-plan agent-contact rate and boundary rate (NAVSIM raster, margin < -0.20 m), pooled, per family, > 45 deg | `bd4_g3.py --shards 2 3` | hold logs (5 062 states, 118 logs); validation part of the train logs (reported) |
| 4 s arc ratio against the base pilot | `prog_ol.py` + `shape_gate.arcs` | hold states (pooled, open, families); navtest (reported) |
| dev ADE at step 3 000 | the trainer's events (`shape_gate.dev_of`) | dev rows of the split |

The three checkpoints are forwarded by the same reader call (no `--bench-plans`), so the base and S counts of a line come from the same
reader as the pilot's; the archived counts (bench plans: W2 141 / 110, tube exits 26 / 25) are reported next to them. Hold logs have been read
by this lane's variants seven times before (decision 244); this is the eighth read and the fifth at pilot scale.
`scripts/logmargin_report.py` (CPU, no model) applies the lines below; `scripts/logmargin_chain.sh` runs everything through the pool.

## 6. Lines (fixed now; point estimates, no rounding in the pilot's favour)

- **(i) Drift removed**: W2(`P2H10SM-P-s0`) <= W2(`P2H10-P-s0`) + **12**, navtest > 45 deg, warp frames. 12 is the base's seed spread of W2 on
  warp frames (decision 234's limitation note: four full-scale base seeds 84 / 79 / 87 / 91); the archived S pilot is +31 over the base pilot.
- **(ii) Lesson kept**: on hold states pooled, the own-plan agent-contact rate falls by >= **18.56 %** and the boundary rate by >= **16.91 %**
  relative to the base pilot, i.e. >= 50 % of the S pilot's own falls (-37.12 % / -33.81 %, `results/loss/g3_a6_hold_S.json`). The task text
  gave -40.2 % / -33.8 %; -40.2 % is the R pilot's (`P2H10R-Pb25-s0`, route_pilot.md), so the S pilot's number is used. The agent fall is
  also reported against the 20.1 % that -40.2 % would give.
- **(iii) Cost**: dev ADE <= the S pilot's + 0.005 m (0.5882 + 0.005 = 0.5932 m), and the 4 s arc ratio against the base pilot, hold states
  pooled (Amendment 6's pilot-gate set), in **[0.995, 1.005]**.

Verdicts: all three met = **candidate S' for the next joint run** (a separate decision; no joint run, no full run, no servable change here).
(i) missed = the fixed margin is not the drift mechanism (at pilot scale; G0.c says how often the margin was lowered at all). (ii) missed =
log-consistent margins remove the lesson. (iii) missed with (i) and (ii) met = not a candidate (imitation accuracy or speed profile). Reported,
not lines: tube exits and the right / left split on navtest, the hold on-log and four-family W2, validation-part rates, navtest arcs, the
per-term training scalars against the S pilot's.

## 7. Budget and conduct

At most 1.5 card-hours of GPU in total; the pilot job is capped at one try and 1 h (if it would exceed 1 card-hour it stops and is reported).
Estimate from the archived runs: identity 0.05, pilot 0.15 to 0.2 (the S pilot: 411 s inside its run), readers about 0.3 (four GPU readers of
6 to 12 GB, history of S-DROP / navtest-warp), total about 0.5. Every job through the pool (`jevdrive.cl submit`), the chain in tmux `jev`.
No full run, no change to any servable or base recipe, no edit of `research/decisions.md` (a draft paragraph goes to the main session).

## 8. Known limits

One seed, pilot scale (2 of 12 shards, 3 000 steps); open loop only; whether W2 predicts closed-loop corridor zeros was never tested (234); the
tolerance 12 comes from full-scale seeds; the hold logs have been read many times by this lane; the 0.05 m slack and the per-row (not per-object)
cap are fixed here without a sweep; a per-row cap also lowers the margin for objects the logged path did not pass closely, when another
object was close.

**Status after the read (2026-10-11 17:53 box time, added after the reads; nothing above was changed).** G0.a met: `LM-IDENT-OFF` against
`B43A7-IDENT-S1`, 42 scalars and every weight bit for bit. G0.b met: the logged path gives zero hinge on 39 739 / 39 739 agent rows and 24 352 /
24 352 road rows; hinge-only rows equal their source log rows on all 24 352 (largest difference 1.5e-5 m agent, 7.6e-6 m road). G0.c: the logged
path violates S's road margin on 13.3 % of hinge-only rows (10.2 % on turns over 45 deg) and S's agent margin on 0.04 % (none on turns). Pilot
`P2H10SM-P-s0`, one read: **(i) not met**, W2 142 against 110 (limit 122; S pilot 140 in the same call); (ii) met, agent -37.9 %, boundary -33.8 %;
(iii) met, dev ADE 0.5902 m, arc ratio 0.9992. By section 6: the fixed margin is not the drift mechanism (pilot scale); not a candidate; no further
run. 0.245 card-hours. Details: [results/logmargin_pilot.md](../results/logmargin_pilot.md).
