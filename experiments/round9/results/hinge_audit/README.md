# HINGE-AUDIT: are the S consequence hinges satisfied by the expert, and what hinge design is (2026-10-22)

The question: the S course's consequence hinges are violated by the human's own log future and by the DAgger teacher's labels far too
often. Is the constraint itself ill-posed? If so, why, and what design does the expert satisfy while the learnable signal is kept?
Pre-registered before any number: [../../plans/2026-10-22-hinge-audit.md](../../plans/2026-10-22-hinge-audit.md). Code:
[../../scripts/hinge_audit.py](../../scripts/hinge_audit.py) (`fwd`, `steps`, `report`, `g5`, `retime`), chains
[hinge_audit.sh](../../scripts/hinge_audit.sh) and [hinge_audit_rest.sh](../../scripts/hinge_audit_rest.sh).

Files here: [audit.json](audit.json) (every number), [strata.csv](strata.csv), [g5_S.json](g5_S.json) / [g5_PAID.json](g5_PAID.json)
(designs fixed / gate / ec_log / ec_wit / ctx), [g5_S_spec.json](g5_S_spec.json) / [g5_PAID_spec.json](g5_PAID_spec.json) (fixed, spec),
[g5_PAIH.json](g5_PAIH.json), [retime_PAID.json](retime_PAID.json). Box outputs: `$DATA_DIR/runs/round9/hinge_audit/` (per-step clearance
files `steps_<pool>.npz`, 0.4-1 GB each). The run was CPU only: three R6DAG-L18-s1 forwards (about 16 min) and the clearances (about 10 min).

## Material

- Pools. nuPlan S = the 87 012 self rows of on1_JTX (rb0 / rtb1 / y / ys), of which 68 939 are live in R6DAG.npz. The onc / soft rasters
  cover only rb0 / rtb1 (27 647 live rows). PAID = the DAgger states, 90 300 teacher-labelled candidates. PAIH = the PAI human-state rows,
  135 793. The navtrain drivable hinge was measured on 172 860 labelled navtrain rows.
- Terms are the trainer's, reimplemented per step and checked against the stored lat_diag values (max difference 7e-8 for agent, 5e-6 for
  road): agent 30 x (front-extended ego box, 0.3 m in the corridor, 0 beside), road 30 x relu(0.3 - corner SDF), onc 15 x and soft 3 x
  region depth (depth 0), side 15 x relu(0.4 - side distance). All terms run at 41 steps of 0.1 s, and the boxes are time-indexed (below).
- Paths:
  - `log` is the human's logged future at the same timestamp, densified from the human's own t0 pose.
  - `fut` is the row's imitation target. On nuPlan S rows it is the blended recovery path, on PAID the DAgger teacher label, and on PAIH
    the human log itself.
  - `plan` is the eval-mode plan of R6DAG-L18-s1, from a CPU forward through the trainer's build: 30 % of the S rows, all 33 294 stored
    PAID rows from pai_s_gate, and 6 % of PAIH.
- Unlearnable (the audit definition) = t0 off the raster, or the plan is active on a term that `fut` also violates. LA = active and not
  unlearnable. Doses per batch: S 32 rows, PAID 16, PAIH 32.

## 1. Audit: unconditional versus active-conditional violation

Share of candidate rows on which the path violates the term. Unconditional means over all rows. "| plan active" means over the rows where
R6DAG-s1's plan has the term > 0, which is the selection the earlier audits read.

| pool | term | human log, unconditional | human log, plan active | recovery label / teacher `fut`, unconditional | `fut`, plan active | plan active |
|:--|:--|--:|--:|--:|--:|--:|
| nuPlan S | agent | **0.11 %** | 1.2 % | 2.0 % | 19 % | 6.5 % |
| nuPlan S | road | **16.2 %** | 55 % | 14.2 % | 73 % | 12.5 % |
| nuPlan S | onc | 1.0 % | 15 % | 1.1 % | 50 % | 2.0 % |
| nuPlan S | soft | 6.0 % | 51 % | 6.6 % | 65 % | 4.1 % |
| nuPlan S | side | 0.9 % | 9.5 % | 2.8 % | 44 % | 4.3 % |
| PAID | agent | **0.19 %** | 1.8 % | 6.4 % | **72 %** | 3.5 % |
| PAID | road (road_pai) | **12.5 %** | 48 % | 11.1 % | 65 % | 11.9 % |
| PAID | side | 1.2 % | 18 % | 5.5 % | 58 % | 3.0 % |
| PAIH | agent | 0.45 % | 12 % | (= log) | | 3.3 % |
| PAIH | road | 0 % (the d418 fallback (d) already removed the 6.9 % human-triggered rows) | | | | 4.1 % |
| PAIH | side | 1.4 % | 53 % | | | 2.4 % |
| navtrain | drivable hinge (0.3 m) | **14.3 %** (footprint really outside 3.6 %) | 76 % | | | 14.1 % |

Reading.
1. **The selection bias is large.** Conditioning on "the plan is active" multiplies the human's violation rate by 5-50x. The earlier
   numbers (51 % road, 12.6 % agent on PAID; 70.5 % road on the S pool) are conditional rates, and they overstate how ill-posed the terms
   are. Unconditionally, the human satisfies agent (0.1-0.5 %), side (0.9-1.4 %) and onc (1.0 %): these terms are well posed for the
   expert. **Road (12-16 %), the navtrain drivable hinge (14 %) and soft (6 %) are not.**
2. **Correction to pai_s_gate (G2, "human log future also violates agent" 12.6 %).** That gate densified the human's future from the
   student's t0 origin, not from the human's own t0 pose. The path therefore began with a fake segment from the student to the human. Read
   from the human's pose, the rate is 1.8 % of agent-active rows. The road reading (51 % there) moves less, to 48 %. pai_s_gate's verdict
   is unaffected because its failing lines were the teacher and reach lines. Its README is not mine to edit, so the main session should
   correct it in place.
3. **Agent is a teacher problem, not a constraint problem.** On PAID, the teacher label violates on 72 % of agent-active rows, and 86 %
   of those violations are real overlaps. Re-timing the label to the human's arc lengths leaves 52 % still violating. Station-matching it
   (the label at the human's station at each step) leaves 44 % ([retime_PAID.json](retime_PAID.json)). So in about half of the cases the
   label's lateral path itself runs through a box: 55 % of the PAID human agent-violation mass and most label overlaps sit in "parked
   beside" scenes, where the map teacher keeps the lane while the human swerved. The rest is speed: the label is longer than the human's
   on 57 % of rows, with median arc ratio 1.09.

Strata ([strata.csv](strata.csv)). For road, the human's violations are spread over speed (13-17 % in every band on nuPlan; on PAID they
rise from 7 % at 0.5-3 m/s to 19 % above 17 m/s) and over scene type (nuPlan junction 17.7 %, straight free 17.7 %, turn 14 %, parked
beside 10 %; PAID turn 24 %, straight 13 %). By clearance, 76 % (nuPlan) and 99 % (PAID) of the human's road mass sits on rows whose
footprint reads < 0 somewhere. For side, the human's violations concentrate in parked-beside and lead scenes at < 8 m/s (3.5-4.7 %).

## 2. Root causes of the human's violations

Each violating row is attributed to the first cause that matches, in the clearance-first reading (the registered order with "initial"
moved last, since a t0 state is a context, not a cause). [audit.json](audit.json) `root_causes` also holds the registered first-match
order and the non-exclusive incidences. 100 % of the human violations got a named cause, against a line of >= 90 %; the residual is
0-0.3 %, except on onc / soft.

| pool / term (human log) | rows | margin larger than the human's clearance (inside, < m) | ego / box size (shallow overlap < 0.25 m) | path beyond the raster window (border clamp) | footprint overhang (axle and centre inside, corner out) | raster narrower than the drivable surface | legitimate manoeuvre | residual |
|:--|--:|:--|:--|:--|:--|:--|:--|:--|
| S road | 11 178 | **81 % rows / 24 % mass** | | 5 % / **50 % mass** | 13 % / 25 % mass | 1.2 % | | 0.1 % |
| PAID road | 10 005 | 42 % / 1 % | | **54 % / 98 % mass** | 3 % | n/a | | 0 |
| S agent | 74 | 39 % (front double margin only) | 61 % / 99 % mass | | | | | 0 |
| PAID agent | 154 | 23 % front double margin + 58 % margin | 19 % / 45 % mass | | | | | 0 |
| PAIH agent | 612 | 35 % front double margin + 36 % margin | 18 % | | | | | (12 % overlap at t0) |
| S / PAID / PAIH side | 644 / 946 / 1 944 | **100 %** (clearance 0-0.4 m) | 0 | | | | | 0 |
| S onc | 287 | 19 % (depth < 0.3 m) | | | | | 44 % turn or passing a static object | 27 % (lane label) |
| S soft | 1 652 | 7 % | | | | | 38 % lane change | 29 % human already in another lane at t0, 26 % other |

Per category:
- **Margin larger than the human's clearance.** This is the dominant cause by row count. On road, 81 % of the human's nuPlan violations
  keep the footprint inside the raster but closer than 0.3 m to its edge. A Pacifica at 2.30 m wide in a 3-3.5 m lane has 0.35-0.6 m per
  side when centred, and less in turns. On side, the 0.4 m buffer exceeds what humans keep from parked cars and queued traffic in every
  violating case.
- **Front double margin (a code finding).** The agent hinge moves the ego box's front edge forward by the margin and then also penalises
  relu(margin - distance), so ahead in the corridor the effective clearance is 0.6 m, not 0.3. This alone explains 23-39 % of the
  human's (rare) agent violations.
- **Static t0 boxes versus moving agents: not a cause.** Boxes are time-indexed along the 4 s on every pool: valid boxes per row at 0 / 4
  s are 17.6 / 16.0 (S), 22.7 / 15.9 (PAID) and 22.8 / 18.5 (PAIH). Freezing them at t0 would remove 45-82 % of the human's agent
  violations, but it would also hide moving cut-ins, so it is not a fix.
- **Time misalignment: not a cause.** Shifting the box time by -0.5 .. +0.5 s gives a flat or minimal human agent rate at 0 to +0.3 s
  (S 0.085 % at 0, 0.05 % at +0.1; PAID 0.41 % at 0, 0.36 % at +0.3; PAIH 0.45 % at 0, 0.40 % at +0.3). A ±0.5 s shift resolves 0 %
  of the real human overlaps on S and PAID.
- **Box inflation / heading or ego geometry.** These explain the remaining human agent overlaps (19-61 % of a 0.1-0.5 % rate). All of
  them are shallower than 0.25 m: a 0.25 m smaller ego box or object box removes every one, so the two causes cannot be told apart at
  this resolution. The hinge also uses the Pacifica footprint (2.297 m wide) on PAI, whose ego is 2.11-2.16 m wide (row files `veh`).
  That is 0.07 m per side of the 0.3 m road margin.
- **Raster coverage (the largest by mass on PAI).** The raster window is x -8 .. +56 m. Corners beyond it read the border cell
  (grid_sample border clamp). This gives 50 % of the human's road mass on nuPlan S and 98 % on PAID. It also hits **the plan: 33 % of
  R6DAG-s1's PAID plans, and 99 % of those above 17 m/s, have out-of-window corners, which carry 58 % of the plan's PAID road-hinge mass
  (PAIH 30 %, nuPlan S 0.1 %).** The road_pai hinge's high-speed "drift" reach (pai_s_gate G4) is therefore mostly a border artefact.
- **Raster gaps or junctions narrower than the drivable surface.** This is small on nuPlan (1.2 % of rows) and unmeasurable on PAI,
  where no second surface exists. **Footprint overhang** (axle and box centre inside, a corner out by < 0.5 m) is 13 % of rows and 25 %
  of mass on nuPlan, mostly in turns.
- **Route / lane-label errors.** These matter only for onc / soft. 27 % of the human's onc violations are not turns or passing
  manoeuvres and point to lane-direction labels. For soft, the design itself is ill-posed: the human changes lanes (38 %) or is already
  in a different same-direction lane from the ego's own chain (29 %; the chain is defined at the ego's position).

## 3. Design method: expert-calibrated consequence constraints

**Principle.** A consequence hinge used as a loss must be *expert-feasible*: in the context where it is applied, the demonstrator's own
behaviour, and the recovery label the row also imitates, satisfy it. A margin is therefore not a free hyperparameter. It is a statistic
of the expert's clearance in that context. Where the expert's clearance is below the nominal margin, the constraint becomes relative to
the expert: the plan may not be worse than the expert, and it is not pushed past the expert's line. A term that the expert violates in
a context carries no gradient there.

Formulas, per row s and term X, with d_X,k(τ) the term's signed clearance at step k (+ = room):
- Witness clearance c_X(s) = min over witnesses w of min_k d_X,k(τ_w; s). The witnesses are the human's time-aligned log, densified
  from the human's own t0 pose, and the row's recovery / teacher label.
- Calibrated margin m_X(s) = min(m_X0, c_X(s) - ε) with ε = 0.05 m. Depth-type terms (onc / soft) get an allowed depth
  δ_X(s) = max(0, -c_X(s) + ε).
- Feasibility floor: if a witness overlaps an object (c_agent < 0 or c_side < 0), the agent / side term is off on that row. This avoids
  teaching that an overlap the label has is acceptable. Such rows are label defects and go to the teacher fix.
- Hinge h_X(τ; s) = mean_k relu(m_X(s) - d_X,k(τ; s)), with one margin only (plain ego box, no front extension), time-indexed agent
  occupancy (boxes at t_k from the replay agents' true future; where unknown, constant-velocity extrapolation), and steps whose
  footprint leaves the raster window masked (or the window enlarged to cover 4 s x v_max).
- Context-quantile alternative (conformal-style, out-of-sample): m_X(b) = min(m_X0, Q_α{c_X(τ_human) : s ∈ calibration, bin(s) = b} -
  ε), so that the expert violates with probability <= α in each context bin.
- Hierarchy: kerb (road), vehicles (agent, side) and the opposite lane (onc) are hard and calibrated. The empty same-direction lane
  (soft) stays soft (lat_soft 0.1, a fifth of onc's 0.5) and calibrated, because lane changes are legitimate.

**Literature (arXiv ids verified on the abstract pages, 2026-10-22).** Constraint inference from demonstrations rests on the same premise
(demonstrations are feasible, so the constraint must contain them): Scobee and Sastry, ICLR 2020, arXiv:1909.05477; Anwar et al., Inverse
Constrained RL, ICML 2021, arXiv:2011.09999; Chou, Berenson, Ozay, WAFR 2018, arXiv:1812.07084; Lindner et
al., CoCoRL, AISTATS 2024, arXiv:2305.16147; Gaurav et al., soft constraints satisfied on average, ICLR 2023, arXiv:2206.01311; Liu et al.,
ICRL benchmark incl. driving, arXiv:2206.09670 (venue not stated on the page). Driving-specific soft constraints learned while imitating:
Mobarakeh et al., arXiv:2412.05717; DRIVE (situation-dependent rules from demonstrations), arXiv:2508.04066. For barrier-function safety
learned from expert data (expert-safe set as the certificate's support): Robey et al., arXiv:2004.03315; Lindemann et al., CoRL 2020,
arXiv:2011.04112; survey, Dawson, Gao, Fan, arXiv:2202.11762. Rule-based scorers with fixed margins against log replay are PDM
(Dauner et al., CoRL 2023, arXiv:2306.07962) and NAVSIM (NeurIPS 2024 D&B, arXiv:2406.15349). They score, they do not train on the
margin. Context-quantile margins follow split conformal calibration (Angelopoulos and Bates, arXiv:2107.07511). Keeping only learnable
rows is RHO-LOSS (Mindermann et al., ICML 2022, arXiv:2206.07137). What differs here: we do not infer the constraint. We keep an
analytic hinge and calibrate its margin per row to the expert witnesses. That makes it a label-free constraint loss that the expert
provably satisfies.

## 4. Candidates on both pools (R6DAG-L18-s1 plans)

Designs: **fixed** (current); **gate** (term kept only where both witnesses satisfy the fixed term); **ec_log** (margin calibrated to the
human log only, d257 generalised to five terms); **ec_wit** (calibrated to min over log and label); **ctx** (context-quantile q05 per
speed band x scene, calibrated on even-hash scenes and tested on odd); **spec** (ec_wit + single front margin + out-of-window mask +
overlap floor; the recommendation).

| pool | design | human violation, hard terms (max) | label also violates, of active | active | LA | **LA rows / batch** | **learnable mass** | mean hinge mass | turn-inside / straight-off / drift / collision / > 13 m/s LA per batch |
|:--|:--|:--|:--|--:|--:|--:|--:|--:|:--|
| S | fixed | road 16.2 % | road 73 %, agent 19 % | 22.3 % | 9.3 % | 2.98 | 25 % (32 % agent / road definition) | 0.87 | 0.39 / 0.59 / 0 / 0.75 / 0.016 |
| S | gate | 0 | 0 | 10.6 % | 10.3 % | 3.31 | 96 % | 0.26 | 0.39 / 0.63 / 0 / 0.89 / 0.016 |
| S | ec_log | 0 | road 48 % | 17.1 % | 10.4 % | 3.33 | 32 % | 0.71 | 0.40 / 0.73 / 0 / 0.81 / 0.019 |
| S | ec_wit | 0 | 0 | 13.5 % | 13.0 % | 4.15 | 88 % | 0.35 | 0.51 / 0.93 / 0 / 1.01 / 0.028 |
| S | ctx | road 3.7 % (test half 4.1 %) | road 55 % | 15.1 % | 8.6 % | 2.73 | 29 % | 0.78 | 0.30 / 0.52 / 0 / 0.80 / 0.016 |
| S | **spec** | 0 | 0 | 13.1 % | 12.5 % | **4.00** | **87 %** | 0.33 | 0.52 / 0.93 / 0 / 0.89 / 0.019 |
| PAID | fixed | road 12.5 % | agent 72 %, road 65 % | 17.0 % | 5.5 % | 0.89 | 16 % | 0.37 | 0.24 / 0.23 / 0.063 / 0.11 / 0.22 |
| PAID | gate | 0 | 0 | 5.3 % | 5.3 % | 0.84 | 100 % | 0.06 | 0.23 / 0.20 / 0.060 / 0.13 / 0.20 |
| PAID | ec_log | 0 | agent 72 %, road 50 % | 13.6 % | 5.8 % | 0.92 | 23 % | 0.25 | 0.24 / 0.25 / 0.071 / 0.11 / 0.22 |
| PAID | ec_wit | 0 | 0 | 8.8 % | 8.8 % | 1.40 | 99.7 % | 0.09 | 0.33 / 0.31 / 0.119 / 0.28 / 0.35 |
| PAID | ctx | road 4.8 % (test half 5.5 %) | agent 72 % | 8.4 % | 3.0 % | 0.48 | 23 % | 0.19 | 0.09 / 0.14 / 0.011 / 0.12 / 0.08 |
| PAID | **spec** | 0 | 0 | 7.5 % | 7.4 % | **1.18** | **99.6 %** | 0.08 | 0.34 / 0.29 / 0.048 / 0.13 / 0.23 |
| PAIH | fixed | side 1.4 % | (= log) | 8.6 % | 7.1 % | 2.26 | 79 % | 0.16 | 0.41 / 0.32 / 0.059 / 0.22 / 0.23 |
| PAIH | **spec** | 0 | | 6.8 % | 6.8 % | 2.17 | 100 % | 0.11 | 0.40 / 0.33 / 0.016 / 0.20 / 0.15 |

G5 sanity: one shape-only step (max 0.2 m per pose) of the summed terms on LA rows.

| pool | design | term reduced | median term cut | median 4 s arc change | end-speed change | step toward the label `fut` / toward the human log | farther from the human line after the step |
|:--|:--|--:|--:|--:|--:|:--|--:|
| S | fixed | 92.7 % | 30 % | +0.07 % | +0.001 m/s | 70.5 % / 70.3 % | 34 % |
| S | ec_wit | 93.5 % | 40 % | +0.08 % | 0.000 | 71.1 % / 72.1 % | 32 % |
| S | **spec** | 94.5 % | 45 % | +0.07 % | 0.000 | 73.5 % / 72.9 % | 32 % |
| PAID | fixed | 97.7 % | 77 % | +0.01 % | 0.000 | 83.9 % / 77.6 % | 25 % |
| PAID | **spec** | 97.1 % | 85 % | +0.02 % | 0.000 | 84.1 % / 78.4 % | 24 % |
| PAIH | fixed / spec | 92.2 % / 89.9 % | 40 % / 48 % | +0.09 % / +0.11 % | +0.007 m/s | (label = log) 70.6 % / 70.0 % | 45 % / 43 % |

Pre-registered lines for spec:

| line | S | PAID | PAIH |
|:--|:--|:--|:--|
| H1 human ≤ 5 % per hard term | pass, by construction | pass, by construction | pass |
| H2 label violates ≤ 30 % of active rows | pass, by construction | pass, by construction | n/a |
| H3 learnable mass ≥ 50 % | pass (87 %) | pass (99.6 %) | pass (100 %) |
| H4 LA per batch ≥ 80 % of fixed | pass (134 %) | pass (133 %) | pass (96 %) |
| H5 bands ≥ 80 % of fixed | pass (turn 1.35, straight 1.58, collision 1.19) | **fail on drift (0.76)** | **fail on drift (0.27) and > 13 m/s (0.67)** |
| H6 term ↓ ≥ 90 %, arc within 2 %, toward ≥ 60 % | pass | pass | **narrow fail on term ↓ (89.9 %)** |

The only out-of-sample H1 reading is ctx, which passes on nuPlan (road 4.1 % on the test half) and narrowly fails on PAID (5.5 %). It
also costs reach (PAID LA per batch 0.48, turn-inside coverage 0.36 of fixed). H1 and H2 hold for spec, ec_wit and gate by
construction, so they are circular. The informative lines are H3-H6.

Reading.
1. **The calibrated designs keep the signal and drop the noise.** spec removes 62 % (S) and 79 % (PAID) of the hinge mass. Nearly all of
   that removed mass was unlearnable: learnable mass rises from 25 % to 87 % and from 16 % to 99.6 %. The absolute learnable mass per row
   rises by 33 % (S) and 31 % (PAID), because rows whose label also violated become learnable relative to the label. LA rows per batch go
   from 2.98 to 4.00 (S) and 0.89 to 1.18 (PAID).
2. **The H5 drift failures on PAI are the artefact, not lost signal.** The "drift" and "> 13 m/s" reach of the fixed design on PAI was
   largely road mass from out-of-window corners (58 % of PAID plan road mass, 99 % of rows above 17 m/s). Masking them removes fake reach.
   Genuine high-speed reach needs a longer raster window.
3. **The gradient is no worse, and slightly better.** The step stays purely lateral (arc and speed unchanged). It cuts the term more (S
   median 45 % vs 30 %) and points toward the human and label line slightly more often. About a third of the S steps (a quarter on PAID)
   still move the plan farther from the human's own line, against 34 % / 25 % for fixed. The d420 kerb-push risk is therefore reduced on
   the rows, not removed by the step direction.
4. **gate is the safe minimum.** It is perfect on H1-H3 but adds no rows on PAID (0.84 per batch). **ctx is dominated:** per-bin margins
   lower the margin for everyone in a bin that contains tight humans.

## 5. Verdict and recommended spec

**Verdict.** The hinges are not uniformly ill-posed. Agent, side and onc are expert-feasible (human 0.1-1.4 %); the high "human also
violates" rates were selection bias plus, on PAID, a path-start artefact. Three things are ill-posed:
- the road and navtrain drivable margins: a 0.3 m margin against a Pacifica footprint in nuPlan / PAI lanes; 12-16 % of humans violate,
  81 % of them by margin only;
- the road raster window: border clamp beyond +56 m, which carries most of PAI's road mass, for human and plan alike;
- the soft term: lane changes.

The agent unlearnability on PAID is a teacher-label defect (72 % of active rows; about half is the label's lateral path, the rest its
speed), not a constraint defect.

**Spec** (a later training arm could adopt it; numbers are the S / PAID values above):
- **Witnesses.** The human log at the same timestamp (densified from the human's own pose) and the row's imitation / teacher label.
- **agent.** Plain ego box. Margin min(0.3, c_w - 0.05) in the corridor and min(0, c_w - 0.05) beside. Off on rows where a witness
  overlaps. Time-indexed boxes, as now.
- **road.** Margin min(0.3, c_w - 0.05), with c_w taken over in-window corners. Steps outside the raster window are masked until the
  PAI / nuPlan rasters are rebuilt with x up to 4 s x v + 8 m (about 130 m on PAI; for example 1 m cells at 256 x 96).
- **onc / soft.** Allowed depth max(0, witness depth + 0.05). soft keeps its weight 0.1 (soft).
- **side.** Buffer min(0.4, c_w - 0.05), off when a witness overlaps.
- **navtrain drivable hinge.** The same per-row rule against the navtrain human label (human minimum SDF q05 is 0.06 m; 14.3 % of the
  human labels violate 0.3 m).
- **Dose.** ho_w 3, the same rows and the same 32 S + 8 DAgger rows per batch. Only the hinge definition changes, which is one change
  per arm. The learnable mass per row is about +32 %, so the effective learnable pressure is about 1.3x R6DAG's. A dose bracket, if one
  is ever run, is ho_w 2-4 (the inverted U of d311 / d347 / d348 was measured with 75 % unlearnable mass and does not transfer one to one).
- **Separately, not a hinge change.** Send the PAID rows whose teacher label overlaps an agent (6.4 % of candidates, 72 % of agent-active)
  to the DAgger teacher: station-capped arc lengths, and a lateral path that respects boxes.

**Expected effect.** The offline gains are certain: learnable mass 3.5-6x, LA rows +33 %, the kerb push from rows the human cannot
satisfy removed, and the PAI border artefact removed. Closed-loop, d257 is the warning: calibrating the road margin on P2H10S cut road
mass by 24 % and the plan did not move. Expected PAI-281 effect at ho_w 3: about 0 to +2 points per seed (range -3 to +4), with fewer
d420-type offroad regressions. nuPlan held: lateral zeros flat to slightly down.

**What would falsify it.**
1. In training (`--diag` activity on the fixed 2 048-row slice), the share of initially active pool rows that the trained model
   satisfies by the end does not rise above the fixed design's (about 20 %).
2. Offroad or collision zeros on PAI 281 or held 452 rise against R6DAG at matched seeds.
3. With the window enlarged, the genuine (in-window) high-speed road reach stays below 0.1 LA rows per batch, which would mean S cannot
   teach the drift class at all.

**Paper (S-course method section).** Replace "fixed-margin consequence hinges" with "expert-calibrated consequence hinges". The section
should define the witness set, the calibrated margin m_X(s) = min(m_X0, c_X(witnesses) - ε) and the feasibility floor, and state the
property: the expert and the recovery label satisfy every term by construction, so every hinge gradient is learnable and never points
past the human's own line. Report the selection-bias analysis (unconditional versus active-conditional expert violation) as the reason.
Report the time-indexed occupancy and the raster-window mask as implementation requirements. This also gives the related-work
positioning against ICRL / constraint inference: we calibrate analytic constraints to demonstrations rather than infer them.
