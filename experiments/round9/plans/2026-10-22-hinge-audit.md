# HINGE-AUDIT pre-registration (2026-10-22, written before any number below is computed)

Question: the S course's consequence hinges (agent, road, lat_onc, lat_soft, lat_side on S rows; the navtrain drivable hinge on imitation
rows) are violated by the human's own log future and by the DAgger teacher's labels far too often (pai_s_gate: label violates on 76 % of
agent-active PAID rows, human log on 51 % of road-active rows; s-rows audit: 63.5 % of the nuPlan S hinge mass on unlearnable rows). Is the
constraint itself ill-posed, why, and what hinge design is satisfied by the expert almost always while keeping the learnable signal?
CPU only: stored files plus a CPU forward of R6DAG-L18-s1. No training, no closed loop, no pool jobs. Precedent: d234, d257 (log-consistent
margin P2H10SM: plan did not move), d264 (lane-centre hinge excluded, log violates 46 %), d418 fallback (d), d420, dose 311 / 347 / 348.
Code: experiments/round9/scripts/hinge_audit.py (+ .sh); results: experiments/round9/results/hinge_audit/.

## Material and definitions

- Pools: nuPlan S pool = the 87 012 self rows of on1_JTX parts rb0 / rtb1 / y / ys (onc / soft rasters exist for rb0 / rtb1 only);
  PAI DAgger states on1_PAID (110 981 rows; candidates = dagger `draw`); PAI human-state rows on1_PAIH (135 793). nuPlan navtrain imitation
  rows for the navtrain drivable hinge (human label only, plus the plan on a sample).
- Terms exactly as R6DAG's trainer sets them up: agent 30 x AgentHinge (0.3 m ahead in the corridor, 0 beside; time-indexed boxes),
  road 30 x relu(0.3 - footprint-corner SDF) on road.npz (drivable.npz fallback for rows off the road raster at t0; road_pai on PAI),
  lat_onc 15 x and lat_soft 3 x nr1_lat region (depth 0), lat_side 15 x nr1_lat side (buffer 0.4). 41 dense steps (0.1 s).
- Paths: `log` = the human's logged future at the same timestamp (time-aligned), placed in the row frame, densified from the human's own
  t0 pose; `fut` = the row's imitation target (nuPlan S: blended recovery path from the ego's offset; PAID: the DAgger teacher label;
  PAIH: the human log itself); `plan` = R6DAG-L18-s1 eval-mode plan (CPU forward, trainer build, seeded sample).
- Violation = term > 0 on the path. Unconditional rate = over all rows with the label; active-conditional = over rows where the plan's term
  is > 0 (the selection the earlier audits used). Hinge mass = sum of the term on the plan. Unlearnable (audit definition, comparable to
  s-rows-and-paper.md and pai_s_gate): t0 footprint off the raster, or the plan is active on a term and `fut` also violates it. LA =
  active and not unlearnable. LA rows per batch at the current dose (S 32 rows of 136; PAI-S 16).

## Deliverables and target lines

1. Audit: per term, per pool, unconditional human (`log`) violation rate and the hinge mass the human path would carry; teacher (`fut`)
   rates; active-conditional vs unconditional; strata: speed, scene (standing / lead < 20 m / parked cars beside / turn > 30 deg /
   junction (lat raster junction share > 0.1, nuPlan rb0 / rtb1) / straight free), clearance band.
2. Root causes of every human violation, mutually exclusive, first match in this order: (i) the human was already inside the margin at
   t0 (initial condition); (ii) margin larger than the human's clearance (human clearance in [0, margin)); (iii) raster gap: road raster
   < 0 where the evaluator drivable surface (nuPlan drivable.npz) is >= 0; (iv) raster coverage: path outside the raster window; (v) ego
   footprint geometry: the overlap disappears with the rear-axle / centre points only or a 0.3 m shorter footprint; (vi) box inflation /
   heading: overlap disappears with object boxes shrunk 0.25 m per side; (vii) time alignment: the overlap disappears under a +-0.5 s box
   time shift; plus a lag sweep (box time shifted -0.5 .. +0.5 s) whose minimum must sit at 0 if aligned; (viii) static-box
   counterfactual (boxes frozen at t0) reported, not a cause unless boxes are static in the trainer; (ix) legitimate manoeuvre (onc: passing a
   static obstacle or turning; soft: lane change); (x) residual. Line: >= 90 % of human violations attributed to a named cause (i)-(ix).
3. Design method: expert-calibrated constraints (written in the result with formulas and verified arXiv ids).
4. Candidates (2-4, from: per-row expert-calibrated margin min(m0, c_ref - eps); per-step / station-matched relative-to-expert margin;
   expert-feasibility gate; context-quantile margins; hard / soft split). Lines, per hard term (agent, road, onc, side; navtrain drivable):
   - H1 unconditional human `log` violation <= 5 % on every pool (soft term reported only);
   - H2 PAID teacher label violates on <= 30 % of agent-active and road-active rows;
   - H3 learnable hinge mass share >= 50 % (current: nuPlan 36.5 %, PAID 16 %);
   - H4 LA rows per batch >= 80 % of the current design's (no large loss of signal) on both pools;
   - H5 loss-class coverage (loss_map bands: turn-inside, straight offroad, drift, collision) of LA rows not below 80 % of current per band;
   - H6 G5 sanity on LA rows: the shape-only step reduces the term on >= 90 %, |median 4 s arc change| <= 2 %, moves toward the human /
     teacher line (cos > 0) on >= 60 % (road) and reported for agent.
   The recommended spec is the candidate passing the most lines with H1 mandatory; if none passes H1 and H3, the verdict says so.
5. Verdict, spec, expected effect and its falsifier; implication for the paper's S-course method section.
