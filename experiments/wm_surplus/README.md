# wm_surplus: do driving policies need grafted world models

status: live
decisions: PD1, PD2, PD4, PD5, PD6, PD7, PD8
index: P2 profile: WA-JEPA completes 31/64 HUGSIM, right turns 1/7, >100 deg turns through 2/8

**Question.** The paper (PD1) claims grafted foundation world models are surplus for end-to-end driving today. P1 measures how
much of the grafted model the driving head actually uses; P2 (scope PD7) profiles what third-party drivers, grafted foundation
models included, can actually do on HUGSIM per manoeuvre (turns, straight, lane-shift proxy) and how they fail (C1).

**Conclusion.** P2 (PD4): WA-JEPA, a grafted foundation world model on its authors' HUGSIM client, completes 31 / 64 scenarios
(HD 0.484 [0.382, 0.591]; repeat 0.458); foreground collisions end 24; right turns complete 1 / 7 (3 leave the route), turns over
100 deg get through 2 / 8, straight 11 / 18 complete. Traffic lights are not judged by HUGSIM (not measurable there). The camera-only
non-foundation references SimScale LTF / DiffusionDrive (same harness, a one-file client feeding HUGSIM's real command) complete
17 / 64 and 15 / 64 (HD 0.342 [0.251, 0.441], 0.307 [0.228, 0.390]): left turns 2 / 16 and 1 / 16, turns over 100 deg 0 / 8;
right turns are 1 / 7 for all three models.

**Next.** P2: the same profile for the remaining camera-only subjects as they register (ReCogDrive next; SimScale LTF / DiffusionDrive done); P1 level (a) on WA-JEPA running (section P1).

**Read more.** [results/p2_match/README.md](results/p2_match/README.md) (P2 profile, HUGSIM limits, per-subject feasibility);
[plans/2026-10-22-p2-match-prereg.md](plans/2026-10-22-p2-match-prereg.md) (P2 pre-registration; its cross-model lines superseded by PD7);
[research/paper-outline](../../research/paper-outline/index.html) sections 3.2 and 5.

## P2: HUGSIM capability profile of third-party drivers (PD4, PD7)

### The recipe (reusable for every subject)

One command reads a list of drivers on several boards. Every allowed driver x board is its own `jevdrive.bench` run in the GPU
pool (HUGSIM: one worker job per card per run, slots per job clamped to what fits a card, so N drivers fill all free cards at
once; navtest scoring shards sized from the box's cores), and one CPU report job, chained `after` all of them, writes the tables:

```bash
cd ~/data/jev-drive; B=".venv/bin/python -m jevdrive.bench"
$B match --drivers <subject> [<subject> ...] --ref WA-JEPA --also P0 --boards hugsim:exam --repeat <label> \
         --out $DATA_DIR/runs/wm_surplus/p2_match/<label> --drop-ply
# returns at once; <out>/DONE appears when the tables are written (wait on that one file, do not poll)
```

- `<out>/hugsim_exam_profile.md`: **the capability profile** of every arm (drivers, reference and `--also`): per manoeuvre class
  from the recorded route (turn left / right, by sharpness; bend; lateral shift = lane-change proxy; straight) the HD, the share of
  each end (complete / fg / bg collision / off route / stuck / spin) with CIs clustered by scene, where turns ended (before /
  inside / through the turn), and example scenario ids (`jevdrive/bench/profile.py`; every HUGSIM `report` writes it).
- `<out>/match.md` and `<board>_{arms,paired,strata}.md`: whole-board numbers and driver - reference pairs (kept for later; under
  PD7 no cross-model score comparison is read now).
- A cell the registry does not allow is not run and is listed with its reason (`Model.benches`); stored references (WA-JEPA on
  navtest / navhard) are read, not re-run. `--repeat L` makes a fresh same-day read; `--drop-ply` deletes the 38 MB / episode `.ply`
  copies after the tables (opt-in, because a match may reuse shared run dirs).
- A new subject needs a bench registry entry (`jevdrive/bench/models.py`) and, for HUGSIM, a client in HUGSIM's FIFO protocol
  (`results/p2_match/README.md` lists what each camera-only subject needs). VRAM per slot is measured (docs/bench.md "VRAM sizing").

### Read so far

HUGSIM 64, preset `exam`: WA-JEPA twice (`p2`, `p2b`), shipped Cinque `P0` as the non-foundation reference, SimScale LTF and
DiffusionDrive (camera-only NAVSIM references, bench drivers `SimScale-LTF` / `SimScale-DD`) twice each. Our checkpoints were
also read before PD7 (B2, PAI checkpoints, RB0-F, SH30); they stay as an input-mismatched reference only
(`results/p2_match/reference_ours/`), as does a stored AlpaSim held-460 pairing (`scripts/held_pair.py`).
Not read: navtest / navhard (our S-trained lineage saw navtest logs), WOD (data deleted), B2D (closed lane).
Results and limits: [results/p2_match/README.md](results/p2_match/README.md).

## P1: how much of the grafted model does the driving head read (usage probes)

Pre-registration: [plans/2026-10-22-p1-usage-prereg.md](plans/2026-10-22-p1-usage-prereg.md). Level (a): project one named
feature tensor onto its top-k navtrain principal directions, run the unchanged planner on navtest, score each k with the
standard devkit through `jevdrive.bench`; read off the smallest k that keeps 95 % / 99 % of EPDMS (paired CIs over logs).

### Plugging in a new model (interface)

`lib/usage_probe.py` is torch-only (no repo imports), so it runs inside the model's own env. A model needs four things:

1. **A feature site**: an object and attribute whose call returns the feature tensor `[..., D]` (a method such as
   `model._encode_scene_context`, or an `nn.Module` attribute). `with Probe(owner, "attr") as p:` wraps it on the instance;
   off by default and bit-identical when off.
2. **An input stream** of the bench's tokens and a `call()` that runs the unchanged model on one sample and returns its
   (8, 3) rear-axle poses (NAVSIM convention). `call` must be deterministic for fixed inputs (diffusion / flow heads: fixed
   or per-call re-seeded noise; check with the determinism gate).
3. **Steps** (all in `usage_probe`):
   - fit: `p.capture(Moments(), stop=True)` around the model call on navtrain samples (catch `Captured`; shard over cards,
     merge with `Moments.merge`, then `Moments.fit(...) -> PCA`, `PCA.save / load`).
   - identity gate: `identity_gate(plain, hooked)` on a few tokens: no hook vs hook off vs `sweep` arm `None`, plus two plain
     calls (determinism). Every line must be bit-identical before a sweep runs.
   - sweep: `p.use_pca(pca, device)`; `p.sweep(call, [None, 8, 16, ...])` returns `{k: poses}`; the producer of the feature
     runs once per sample (cached), the rest of the model once per k; `p.energy` accumulates held-out explained variance
     (`explained_heldout`).
   - level (b): `with skip_blocks(model.encoder.blocks, idx):` drops residual blocks (identity).
4. **Scoring**: write `np.savez(pred_file, tokens=..., poses=(n, 8, 3))` for each arm at
   `jevdrive.bench.navsim.pred_file(resolve("PROBE-<tag>-<arm>"), "navtest")` (bench family `pred`: supplied poses, scored
   only), then `scripts/p1_readout.py --tag <tag> --arms off k8 ... --ev ev.json --out <dir>` (one CPU pool job: submits
   the navtest scoring of every arm, sharded over the box's cores, waits inside the pool, writes the bench paired tables vs
   `off` and `p1_readout.{md,csv,json}` with k95 / k99).

`scripts/p1_wajepa.py` is the worked example (WA-JEPA: commands `gate / fit / pca / sweep / merge`; about 150 model-specific
lines: agent construction, the NAVSIM loader, the feature builder) and `scripts/p1_lane.sh` submits the whole chain at once
(fit shards and sweep shards = one pool job per card; `--when-exists` / `--after` between stages). Copy both for a new model
and change the agent, loader and `SITE`.

### WA-JEPA (level (a))

Site: `MultiViewCausalFutureMaskedJEPA._encode_scene_context` output `context_scene` [1, 4096, 512] (V-JEPA 2.1 ViT-L history
tokens after the planner's own LayerNorm + Linear 1024 -> 512 projector; 4 cameras x 2 tubelets x 16 x 32 patches), the only
image input of the flow planner. PCA fit on navtrain logs of the `navtrain_{current,history}_3` archives (re-downloaded for
this, cameras L0 / F0 / R0 / B0 only), arms off, k 8 .. 512. Outputs: `$DATA_DIR/runs/wm_surplus/p1_usage/` (pca, gate,
sweep chunks, ev.json; bench runs `runs/bench/navtest/PROBE-wajepa-ctx-*`), small results in `results/p1_usage/`.

Results (PD5; tables in [results/p1_usage/](results/p1_usage/)): identity gate bit-identical on 24 tokens (hook off, loader
path, cached sweep, determinism); arm `off` EPDMS 91.71 (= the September read). Retained EPDMS per k (8 .. 256): 0.80 / 0.85 /
0.95 / 0.97 / 0.99 / 0.998; k95 = 64 (CI-backed; point 32), k99 = 256 (point 128) of 512; navtest explained variance at k 64
is 0.63. Losses concentrate in turns > 45 deg and right turns ([navtest_strata.md](results/p1_usage/results/navtest_strata.md)).
Pre-registered reading: partial (between 32 and 256). Level (b): running.

### Knows vs does (any subject)

`lib/know_do.py` + `scripts/p1_knowdo.py`: a subject writes one npz (`tokens`, `X` = training-free pooled planner-input
features, `B` = the planner's non-image inputs) and has its navtest plans as a bench pred file; the script classes GT and plan
with the bench's manoeuvre strata and fits log-fold logistic probes (`blind`, `feat`, `feat+blind`) -> per-class knows / does
table with log-bootstrap CIs and the pre-registered verdicts ([plans/2026-10-23-p1-knowdo-prereg.md](plans/2026-10-23-p1-knowdo-prereg.md)).
WA-JEPA features: `scripts/p1_wajepa_feats.py`. WA-JEPA result: no "sees but does not do" class; the plan's per-class recall
beats the linear probe in every class (balanced accuracy 0.875 vs 0.685)
([knowdo_wajepa.md](results/p1_usage/results_kd/knowdo_wajepa.md)).

Throughput note: the WA-JEPA flow planner is compute-bound in fp32 (0.9 s per call; encoder 0.08 s, loading 0.1 s);
batching 4 tokens x 8 arms gains ~5 %, TF32 nothing, bf16 4x but moves plans up to 0.27 m, so fp32 was kept
([profile.json](results/p1_usage/profile.json)). Declare >= 45 GB per sweep shard so the pool never stacks two on one card.

## P2 swap: WA-JEPA trained with the NTV0 recipe (PD8; feasibility only)

Question: freeze WA-JEPA's V-JEPA 2.1 encoder, train its planner (`predictor` 174 M + `scene_projector` 2 M) with NTV0's batch
composition (83 navtrain / 13 yaw-rate / 32 S rows), losses and 9 500 steps, read on PAI 281 (ACC + GAP) beside NTV0-L18-s0
(0.744, 59 zeros). **Not built (2026-10-23 01:00):** the navtrain sensor blobs were deleted 10-21 (76 of 1 310 logs on disk), the S
pool holds Cinque tokens and labels only (no 4-camera frames; must be re-rendered from stored ego states), PAI has no rear-centre
camera for WA-JEPA's B0, and no WA-JEPA PAI driver or as-shipped PAI read exists. Estimate to do it properly: ~25–80 card-h for one
seed, peak ~60 GB disk, ~5 agent-days of code, 2–3 calendar days (the 282 GB navtrain re-download at 2–4 MB/s sets the pace). Step
table, risks and every unavoidable difference from NTV0: [results/p2_swap/estimate.md](results/p2_swap/estimate.md).

<!-- files:begin -->
## P0: third-party subjects on navtest (reproduction rows, P1 subjects; PD2, PD6)

Camera-only subjects only (PD2). Shortlist: [results/p0_subjects/shortlist.md](results/p0_subjects/shortlist.md); setup log:
[results/p0_subjects/setup.md](results/p0_subjects/setup.md). Subjects: ReCogDrive-2B-RL (LLM: InternVL3-2B / Qwen2.5 hidden state ->
DiT planner, published 90.8 PDMS), AutoVLA (Qwen2.5-VL-3B action tokens, 89.1), DriveVLA-W0 (Emu3 world model + VLA, 87.2,
stretch); camera-only non-foundation references SimScale-LTF / SimScale-DD (no published navtest number for these checkpoints:
execution check only). The LiDAR subjects (hustvl DiffusionDrive, WoTE) are disabled in the registry.

- **Bench**: one JSON per subject in `jevdrive/bench/agents/` (family `agent`; SimScale models keep their HUGSIM family and gain
  navtest through their JSON). Runner `agent` = the fork's own NAVSIM agent via `jevdrive/bench/agent_poses.py` (hydra config +
  eval-script overrides, `compute_trajectory`; per-model sensor-config overrides, e.g. ReCogDrive `lidar_pc=False`); runner `cmd`
  = a thin input adapter (`scripts/simscale_navtest.py`). Poses in chunks of whole logs, one GPU job each -> v2 EPDMS + v1.1 PDMS
  from the same poses; `summary.json` carries `PDMS_v1`, `published`, `delta_vs_published`.
- **Gate** (`scripts/p0_repro.py submit --model M`): smoke on `navsim/p0-smoke-logs` (8 logs, 540 tokens; >= published - 3) ->
  full navtest; `verdict` -> `results/p0_subjects/<M>_full.json`, reproduced iff |PDMS - published| <= 0.5. Not tuned on failure.
- **P1 on the subjects** (`scripts/p1_agent.py`, P1 hook `lib/usage_probe.Probe`, outputs `$DATA_DIR/runs/wm_surplus/p1_agents/<M>/`):
  ReCogDrive site = the LM's last hidden state handed to the DiT. DriveVLA-W0 (not a NAVSIM agent; built in-process as their
  inference script, `W0` / `W0Site`) sites f0..f31 = the input-normed Emu3 hidden state of every layer at the positions its
  Pi0-style action expert attends to (K / V are linear in it), all 32 projected at once; flow noise seeded per token from our
  side; its hooked path re-implements their sampling loop (VLM once, their `forward_with_cache`), so its gate allows 0.01 m mean
  |xy| for the off arm (measured 0.0027) while G1 / G4 / G5 stay bit-identical. Lane: `DIM=4096 VRAM=45 PCPU=8 V1=1
  scripts/p1_agent_lane.sh DriveVLA-W0 w0-vlmkv` (`V1=1`: readout also scores v1.1 PDMS per arm). **PCA fit differs from WA-JEPA's**: navtrain cameras are gone,
  so the PCA is cross-fitted within navtest, 2 folds of whole logs (sorted navtest log names, `i % 2`; a token is projected
  with the other fold's PCA). The `off` pass also stores pooled features + 20 blind inputs (`knowdo_feats.npz`) for
  `scripts/p1_knowdo.py`. **Request to the P1 lane**: for comparability, also refit WA-JEPA with the same 2 log folds
  (`p1_agent.fold_of_logs`) next to its navtrain fit; the subjects' rows will be labelled "navtest 2-fold cross-fit" either way.

Results (PD6): reproduced within 0.5 v1 PDMS on all 12 146 navtest tokens: ReCogDrive-2B-RL 90.52 (published 90.8; EPDMS 82.70),
AutoVLA 89.53 (89.11; 89.24), DriveVLA-W0 87.04 (87.2; 87.00); SimScale-LTF / -DD 87.28 / 87.35 (EPDMS 87.84 / 86.28; no published
number). P1 on ReCogDrive (cross-fitted PCA of the LM hidden state, 1536-d): k95 = 512, k99 = 1024; knows-vs-does: stop = sees but
does not do (0.442 vs 0.152), turns / straight = does what it sees. P1 on DriveVLA-W0 (32 Emu3 layer states x 4096-d the action
expert reads): k95 = k99 = 4096 (full width; k2048 keeps 93.7 % EPDMS, -5.45 [-6.50, -4.31]; v1 PDMS the same); knows-vs-does: the
plan matches or beats the probe in every class, no sees-but-does-not-do (stop / launch / lane change unclear),
[results/p1_usage/DriveVLA-W0/](results/p1_usage/DriveVLA-W0/results/p1_readout.md). Projection-artifact controls (PD10, prereg addendum A;
`scripts/p1_w0ctl_lane.sh`, fixed stratified 3 000-token subset `navsim/p1-w0-ctl`): restoring the row norm leaves k64 unchanged and
recovers part of k512 / k2048 (k2048 -2.74 vs -6.51); projecting a single layer (16 or 31) at k64 costs nothing (+0.14 / +0.02,
CIs span 0), so W0's full-width reading comes from projecting all 32 layers at once, not from any one layer reading a wide subspace
([controls/p1_readout.md](results/p1_usage/DriveVLA-W0/controls/p1_readout.md)). ReCogDrive variants: running. Tables: `results/p0_subjects/*_full.json`,
[results/p0_subjects/p1_recogdrive/p1_readout.md](results/p0_subjects/p1_recogdrive/p1_readout.md),
[knowdo_recog-lmh.md](results/p0_subjects/p1_recogdrive/knowdo_recog-lmh.md).

## Files

- `README.md` (README.md): do driving policies need grafted world …
- `2026-10-22-p2-match-prereg.md` (plans): P2 追平读数（WA-JEPA vs 当前谱系）预登记
- `match.py` (jevdrive/bench): a list of drivers against one reference …
- `index.html` (research/paper-outline): 

[results/](results/) 15 result files · [plans/](plans/) 4 live plans · [lib/](lib/) 2 library · [scripts/](scripts/) 10 entry points
<!-- files:end -->

Layout: `scripts/` entry points (live), `lib/` code other topics import, `archive/` one-off code of a concluded
experiment, `results/` small result files, `figs/` figures, `plans/` live plan notes.
