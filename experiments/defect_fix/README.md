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

**Next.** Read the full chain against the bar (handoff below). For the user: whether to accept the base doses' dev-ADE cost for their
waiting gain (a seed-0 look from NR1B10-F-s0, `NR1FB10-L18-s0`, is submitted: prereg supplement 3, handoff below); the opposite-lane region's label definition (amendment 1) for later rounds.

## Handoff (2026-10-17 17:05 CST)

- Full chain (launched 17:03 CST): trainings `NR1F-L18-s0` 1017-170326-7b4a, `NR1F-L18-s1` 1017-170335-9107 (RTB1AR-F-s0, 2 500 steps,
  pool on1_L18NR1, `--ego-scale-speed 0.5 --ego-scale-max 0.2`, opposite lane 0.15 + side 0.15 + soft 0.03 on the amendment-1 raster
  runs/defect_fix/lat/L18NR1v2). Every read job id (held 6, public 10, static, wc, waiting set, PAI submitter) is in box
  `runs/defect_fix/stages.txt` (lines `full NR1F-L18-s*`) and `jobs.txt`; the PAI submitter `df-paisub-NR1F-L18-s0` puts stop17 and PAI 281
  (submission config) under `runs/defect_fix/{paisub,pai}/NR1F-L18-s0` (seed 1: stop17 only).
- Exploratory arm `NR1FB10-L18-s0` (prereg supplement 3, submitted 17:18 CST, seed 0 only, no public / PAI / seed 1; descriptive, no bar line; the user
  decides on expansion): NR1B10-F-s0 (trainval-wait base, W 0.860 / G 0.977, dev ADE 0.5801 > 0.5619) -> the NR1F L18 recipe unchanged (compiled, merged pool,
  history noise, 0.15v2 terms), via `SEEDS=0 NOPUB=1 NOPAI=1 TAGP=NR1FB10 bash experiments/defect_fix/scripts/df.sh full NR1B10-F-s0 0.15v2` (SEEDS / NOPAI are
  new switches of `full`; nothing was submitted and cancelled). Jobs (box `runs/defect_fix/jobs.txt`, `stages.txt`): training 1017-171808-b239; held 460 @18
  `df-held-NR1FB10-L18-s0-0..5` 1017-171809-b141, 1017-171809-969b, 1017-171810-a39a, 1017-171810-31df, 1017-171811-7feb, 1017-171811-2176; static scene
  1017-171812-005b; W / G (wc plans) 1017-171812-0eec; waiting set (v2, 312 windows) 1017-171813-856f. Its waiting-set job may end rc 1 at the collect step, as
  the `df-ws-NR1H*` did (3 b windows without a logged human launch): then add `NR1FB10-L18-s0` to the `wscollect` call below. Expansion if wanted: seed 1 (+ public,
  stop17) `SEEDS=1 TAGP=NR1FB10 bash experiments/defect_fix/scripts/df.sh full NR1B10-F-s0 0.15v2`; seed 0 public `df.sh repub NR1FB10-L18-s0 0 1 2 3 4 5 6 7 8 9`;
  seed 0 PAI 281 `df.sh paisub NR1FB10-L18-s0 pai` (the body `full` submits). `df.sh wait` below also waits for these jobs (it watches every job in `pool/`).
- Summaries: `experiments/defect_fix/results/` (written into the box checkout, copied to this repo by hand): `base_pilot`, `l18_pilot`
  (+ `_regress`: the 28 named + 12 no-launch scenes), `lat_gate`, `lat_gate_v2`, `identity_base` / `identity_rsg`, `candgate_*`, `w_*`, `pack_NR1TV`.
- Final table against the bar (box, repo root), in order, each once:
  1. `WAIT_PAI=$DATA_DIR/runs/defect_fix/pai/NR1F-L18-s0 CAP=7000 bash experiments/defect_fix/scripts/df.sh wait df-paisub-NR1F-L18-s0 df-paisub-NR1F-L18-s1`
     (one blocking wait, run in the background; since 86209efb it no longer counts a "cancelled: cancelled while queued" STATUS line as open, which
     matched `*queued*` and kept the earlier wait running to its cap)
  2. `bash experiments/defect_fix/scripts/df.sh wscollect NR1F-L18-s0 NR1F-L18-s1 NR1H15O-L18-s0 NR1B05-F-s0 NR1B10-F-s0 NR1BALL-F-s0`
     (waiting-set tables: wait_set.py collect stops on the 3 variant-b windows of waitset@v2 without a logged human launch; this reruns it with
     them as NaN)
  3. `bash experiments/defect_fix/scripts/df.sh fullsum` -> `results/full_chain.{md,json}` and `full_chain_regress.{md,json}` (bar lines 1-6;
     line 6, the waiting set, is proposed to the user)
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
