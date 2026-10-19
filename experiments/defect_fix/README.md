# defect_fix: NR1 joint arm against the user's defects

status: live
decisions: pending
index: base doses W 0.84-0.97 but dev ADE +0.028-0.041 (all fail); lateral v2 arm held zeros 42 vs 47; full chain running

**Question.** On top of V2 (RTB1AR base -> RTB1ARO-L18), does one joint arm aimed at the defects the user saw in the zero-score videos (slow
launches and waiting: trainval rows with a visible reason to stand, balanced by launch rows; creep: history-speed noise; lateral failures: a
hard / soft constraint hinge on kerbs, other vehicles and opposite-direction lanes) beat RSG-L18-s0 clearly (prereg section 5)?

**Conclusion.** Pilots (one seed, open loop for the bases, held 460 for the L18 arms): the trainval base doses teach waiting
(W 0.836 / 0.860 / 0.971 against V2's 0.326, G 0.973 / 0.977 / 0.885) but all three miss the dev-ADE line (0.580 / 0.580 / 0.593 against
0.562), so by the prereg the full chain starts from V2's base. Of the lateral terms, the side buffer + soft term barely move held (47 / 46
zeros against 47); the amended opposite-lane region (amendment 1) does: NR1H15O 42 zeros, lateral zeros 28 against 34, paired +0.010
[+0.003, +0.019] against its control. Full chain = RTB1AR-F-s0 -> L18 with NR1H15O's terms + history-speed noise, seeds 0 / 1.
Update 17:45: the user accepted the dev-ADE cost of the trainval-wait base for its waiting gain, so NR1FB10 (NR1B10-F-s0 -> the same L18 recipe, seeds 0 / 1)
is the candidate chain, judged by the same bar, with NR1F as the no-waiting comparison (prereg supplement 4).

**Result 19:00 (d346).** Neither full chain meets the bar: NR1F ~ V2 (held zeros 39 / 41 vs RSG 34, PAI -0.013); NR1FB10 is worse (held 47 / 48,
PAI -0.028); L18 washes out the base's waiting (W 0.86 -> 0.17 / 0.19). **Next** (the user's call): keep the base's conditional pathways through L18, or
serve a base without L18; closed-loop reads of the bases NR1B10-F-s0 / RTB1AR-F-s0 (held 460 + static, `df-baseheld-*`, `df-basestatic-*`) submitted 19:02.
Waiting-set collect for the four chain tags re-submitted 19:00 (1017-185934-9be6). Exploratory NR1FB05 / NR1FBALL still at priority -5 (supplement 5).

**Zero overlap (20:20, [results/zero_overlap.md](results/zero_overlap.md), figure `figs/zo_speed_hist.png`).** Reads are deterministic (RSG held read 3x: 0 / 460 flips).
NR1F's zeros are old ones (held 28-29 shared with RSG, 8-9 V2-lineage offroad / corridor, 1 seed-consistent new); NR1FB10 adds 5 seed-consistent new held zeros
(AF-front, corridor, offroad). Paired on non-1.0 scenes the new chains drive faster than RSG (held +0.4-0.7 m/s), still about 1.5-2 m/s below the human.

**JT and JTB / JTN (10-18; [results/jt/](results/jt/README.md), [results/jtb/](results/jtb/README.md), prereg supplements 9-10).** JT: worse than RSG.
JTN, the same pool and schedule with no arc term, ties RSG + stop3 (-0.001) and held 460 (34 zeros), and is +0.089 natively (static windows).
JTB (arc term with the braking / stopping rows, W 0.0012) also ties under stop3 (-0.003) but fails the crash guard (48 zeros, front collisions 4 -> 14).
Offline it overshoots the human's stops more (+3.5 -> +5.5 m). So the arc term caused JT's loss, and labelling the braking rows did not repair it.

**RTJA and LAUNCH-DIAG (10-18/19; [results/rtja/dose.md](results/rtja/dose.md), [results/rtja/launch_diag.md](results/rtja/launch_diag.md), supplements 12-13).**
No RTJA arm passes the dose rule: lead-class launches are late. The diagnosis traces the late launch to the 38 trainval imitation rows per batch of the JT00 draw
(starting / braking rows oversampled). S-row hinges, the side term, noise and length do not cause it. Removing the rows (LDNTV1K, 1 000 steps) turns +0.57 / +0.77 s late
into -0.33 / -0.28 s early and takes JTN's variant-a wait away with it (decision 363).

**NR1FREE (10-19; [results/free/](results/free/README.md), supplement 15).** A one-sided longitudinal consequence term (plan below 0.8 x the
human's 4 s arc with the way ahead free; no target speed). It passes the offline gate: 0 % false fires on human paths, 70 % on late-launch
states, 0 / 251 JT front-collision windows. In the 1 000-step pilot it fixes the late launch (+0.33 -> -0.23 / -0.03 s) and raises the
held mean +0.038 over its control. It fails the guards: the wait collapses (0.86 -> 0.58 / 0.49), held zeros 37 -> 47, front collisions
5 -> 12. The line stops (decision 365).

## Handoff (2026-10-17 17:05 CST; NR1FB10 chain added 17:45; dose-response arms NR1FB05 / NR1FBALL added 18:17)

- Full chain (launched 17:03 CST): trainings `NR1F-L18-s0` 1017-170326-7b4a, `NR1F-L18-s1` 1017-170335-9107 (RTB1AR-F-s0, 2 500 steps,
  pool on1_L18NR1, `--ego-scale-speed 0.5 --ego-scale-max 0.2`, opposite lane 0.15 + side 0.15 + soft 0.03 on the amendment-1 raster
  runs/defect_fix/lat/L18NR1v2). Every read job id (held 6, public 10, static, wc, waiting set, PAI submitter) is in box
  `runs/defect_fix/stages.txt` (lines `full NR1F-L18-s*`) and `jobs.txt`; the PAI submitter `df-paisub-NR1F-L18-s0` puts stop17 and PAI 281
  (submission config) under `runs/defect_fix/{paisub,pai}/NR1F-L18-s0` (seed 1: stop17 only).
- Candidate chain `NR1FB10-L18-s0 / s1` (prereg supplements 3-4; the user accepted the dev-ADE cost of the trainval-wait base NR1B10-F-s0, 0.5801 against
  the 0.5619 line, for its waiting gain W 0.860 / G 0.977; launched 17:42 CST at pool priority 12, which runs first: `--priority` is higher-first, NR1F's chain is 9):
  `PRIO=12 TAGP=NR1FB10 bash experiments/defect_fix/scripts/df.sh full NR1B10-F-s0 0.15v2` = NR1F's chain and recipe with only the base changed. Trainings
  `NR1FB10-L18-s0` 1017-174202-d892, `NR1FB10-L18-s1` 1017-174213-db2e; the other 20 jobs per seed (held 6, public 10, static, wc, waiting set, PAI submitter) are
  in box `runs/defect_fix/stages.txt` (lines `full NR1FB10-L18-s*`) and `jobs.txt`; PAI submitters `df-paisub-NR1FB10-L18-s0` 1017-174212-ef6a (stop17 + PAI 281 under
  `runs/defect_fix/{paisub,pai}/NR1FB10-L18-s0`, PRIO 12) and `df-paisub-NR1FB10-L18-s1` 1017-174223-56e6 (stop17 only). Judged by the section 5 bar, NR1F is the
  comparison. The first seed-0-only submission of 17:18 (priority 9, `1017-171808-b239` .. `1017-171813-856f`, nothing had started) was cancelled and resubmitted this way.
  Switches `full` gained for it (not needed by the chain above): `SEEDS` (default "0 1") and `NOPAI=1`, next to `NOPUB` and `TAGP`.
- Dose-response look (prereg supplement 5; exploratory, seed 0, cheap reads; not a candidate unless the user promotes it): `NR1FB05-L18-s0` (from NR1B05-F-s0) and `NR1FBALL-L18-s0`
  (from NR1BALL-F-s0) = NR1FB10's L18 recipe with only the base changed (compiled, merged pool, history noise, 0.15v2 lateral terms). Reads: held 460 @18 (6), static, wc (W / G), waiting set
  v2 (312 windows); no public 1491, no stop17 / PAI 281, no seed 1. Submitted 18:16 CST at **pool priority -5** (higher starts first: NR1FB10 chain 12, NR1F chain and df.sh default 9, pool
  default 0, lowest other lane -1), so they only take capacity nothing else wants and may wait a long time; every declaration is df.sh's, as NR1FB10's seed-0 jobs (train 44 GB / 90 GB RAM,
  held 20 / 40, static 20 / 30, wc 10 / 40, waiting set 20 / 30). 10 jobs each, ~2 card-hours, ids also in box `runs/defect_fix/stages.txt` (lines `full NR1FB05-L18-s0`, `full NR1FBALL-L18-s0`)
  and `jobs.txt`:
  - `SEEDS=0 NOPUB=1 NOPAI=1 TAGP=NR1FB05 PRIO=-5 bash experiments/defect_fix/scripts/df.sh full NR1B05-F-s0 0.15v2`: training 1017-181639-a4b3; held 1017-181640-b209, 1017-181640-ca0e,
    1017-181641-a035, 1017-181641-cf06, 1017-181642-d29a, 1017-181642-23b3; static 1017-181643-788a; wc 1017-181643-57c1; waiting set 1017-181644-e757.
  - the same with `TAGP=NR1FBALL` and `full NR1BALL-F-s0 0.15v2`: training 1017-181648-afc8; held 1017-181649-1924, 1017-181649-9f8e, 1017-181650-989c, 1017-181650-ca53, 1017-181651-5a7a,
    1017-181651-3ec4; static 1017-181652-678a; wc 1017-181652-d9a9; waiting set 1017-181653-cf43.
  - When they end: the waiting-set job exits rc 1 at its collect step (the 3 variant-b windows without a logged human launch), so run `bash experiments/defect_fix/scripts/df.sh wscollect
    NR1FB05-L18-s0 NR1FBALL-L18-s0` (apart from step 2 below: these two may finish long after the chains); then NR1F / NR1FB05 / NR1FB10 / NR1FBALL side by side, W and a-variant zeros against
    b-variant launch delay and held zeros (descriptive; no bar line, prereg supplement 5).
- Summaries: `experiments/defect_fix/results/` (written into the box checkout, copied to this repo by hand): `base_pilot`, `l18_pilot`
  (+ `_regress`: the 28 named + 12 no-launch scenes), `lat_gate`, `lat_gate_v2`, `identity_base` / `identity_rsg`, `candgate_*`, `w_*`, `pack_NR1TV`.
- Final table against the bar (box, repo root), in order, each once:
  1. `WAIT_PAI="$DATA_DIR/runs/defect_fix/pai/NR1F-L18-s0 $DATA_DIR/runs/defect_fix/pai/NR1FB10-L18-s0" CAP=7000 bash experiments/defect_fix/scripts/df.sh wait
     df-paisub-NR1F-L18-s0 df-paisub-NR1F-L18-s1 df-paisub-NR1FB10-L18-s0 df-paisub-NR1FB10-L18-s1`
     (one blocking wait, run in the background; since 86209efb it no longer counts a "cancelled: cancelled while queued" STATUS line as open, which
     matched `*queued*` and kept the earlier wait running to its cap)
  2. `bash experiments/defect_fix/scripts/df.sh wscollect NR1F-L18-s0 NR1F-L18-s1 NR1FB10-L18-s0 NR1FB10-L18-s1 NR1H15O-L18-s0 NR1B05-F-s0 NR1B10-F-s0 NR1BALL-F-s0`
     (waiting-set tables: wait_set.py collect stops on the 3 variant-b windows of waitset@v2 without a logged human launch; this reruns it with
     them as NaN; its run-directory path was wrong until a30b33a2 and printed "no finished run" for every tag)
  3. `bash experiments/defect_fix/scripts/df.sh fullsum` -> `results/full_chain.{md,json}` and `full_chain_regress.{md,json}` (bar lines 1-6;
     line 6, the waiting set, is proposed to the user); the candidate chain: `TAGP=NR1FB10 bash experiments/defect_fix/scripts/df.sh fullsum` ->
     `results/full_chain_NR1FB10.{md,json}` and `_regress` (the file names carry the chain, so the second run does not overwrite the first)
- Not done here: the route-memory anchoring (C) runs as a separate agent (results to experiments/route_mem/results/anchor_check.md);
  NR1 does not use it.

**Read more.** [plans/2026-10-17-nr1-prereg.md](plans/2026-10-17-nr1-prereg.md) (Chinese; design, gates, bar, branch rules).

<!-- files:begin -->
## Files

- `2026-10-17-nr1-prereg.md` (plans): NR1 预登记：下一轮联合训练臂（对准用户看出的缺陷）
- `df.sh` (scripts): Box, repo root
- `df.py` (scripts): NR1, the next-round joint arm against …
- `df_lat.py` (scripts): NR1 lateral constraint labels
- `nr1_lat.py` (lib): NR1 hard / soft lateral constraint …
- `rm_train.py` (experiments/route_mem/scripts): onpol_nuplan's on_train.py `train` (its …
- `handoff-2026-10-17-losses-and-plan.md` (tmp): 交接：2026-10-17 失分诊断汇总、下一轮训练方案、在跑任务
- `census_red_rear.md` (experiments/wait_cap/results/next): Red-light and rear-end census
- `explore_g.md` (experiments/wait_cap/results/next): does the retain line track turns worse …
- `wait_set.md` (experiments/alpasim_proxy/results): a local closed-loop set where the …

[results/](results/) 19 result files · [plans/](plans/) 1 live plans · [lib/](lib/) 1 library · [scripts/](scripts/) 4 entry points
<!-- files:end -->

Layout: `scripts/` entry points (live), `lib/` code other topics import, `archive/` one-off code of a concluded
experiment, `results/` small result files, `figs/` figures, `plans/` live plan notes. Refresh the file list and
INDEX.md with `python tools/topic_index.py --topic <name>` (your topic only; a bare run or `--all` would rewrite
other lanes' READMEs, so `--all` is a dry run unless `--write`; `--check` is read-only).

**NATIVE-GAP (10-19; [results/native_gap.md](results/native_gap.md), d372).** Diagnosis of the native arena gap to ~0.93 (no hold). JTN-L18-s1 is 87 points short, 85 of them in static windows at a red light's line with no lead (the model holds behind a close lead, not at a red light; the decision-0 plan arc decides, creep amplifies). Holding there is worth +72 net, an on-time launch +14 (0.9296). The speed ratchet is 90-102 % plan-side but worth <= 13 points on 5 s windows; the human's speed history on the same frames recovers 26-44 % of the plan shortfall.

**REDLINE-PROBE (10-19; [results/redline_probe.md](results/redline_probe.md), prereg [plans/2026-10-19-redline-probe-prereg.md](plans/2026-10-19-redline-probe-prereg.md), d373).** Frozen-token probe on the 239 red-line / no-lead static windows (k = 0, JTN1 native dumps, log folds). Tokens 0.975 against KIN + lead head 0.944: margin +0.031 [+0.00, +0.08], below +0.05, so by the registered rule the lamp is not readable. The tokens read "standing, no lead": they call the 40 green launches "must stand" too (red vs green 0.53). Holding at the red line is a training signal on features the policy already reads; reading the lamp is not available in the render.

**WARP-LOOP (10-19; [results/warp_loop.md](results/warp_loop.md), prereg [plans/2026-10-19-warp-loop-prereg.md](plans/2026-10-19-warp-loop-prereg.md), d375).** Open-loop re-synthesis of the 6 synthesised slots on the native dumps (`scripts/warp_loop.{py,sh}`): the plan follows the warp track's speed with gain 0.69 (JTN1) against 0.07 for the row's speed history; frames + row recover 88 % of the human-speed shortfall. The loop follows the speed level, not its trend: a constant-current-speed warp (`SH30_WARP=const`, diagnostic switch in sh30_core, default off) leaves the ratchet (k = 9 ratio 0.912 to 0.922) and costs launches (0.856 to 0.790), mean -0.009. Static creep does not run through the warp (83-95 % of the growth kept).
