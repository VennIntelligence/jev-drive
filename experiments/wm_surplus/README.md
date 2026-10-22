# wm_surplus: do driving policies need grafted world models

status: live
decisions: 452, 453, 454
index: P2 match: current lineage vs WA-JEPA, HUGSIM 64 + AlpaSim held 460 (pending)

**Question.** The paper (d452) claims grafted foundation world models are surplus for end-to-end driving today. P1 measures how
much of the grafted model the driving head actually uses; P2 asks whether our lineage (Cinque 382M + body-schema / S-row training,
no grafted model) drives as well as WA-JEPA on the boards we may report, under the same runner and controller.

**Conclusion.** P2: pending (decision 453).

**Next.** P2 reads running (2026-10-22 evening); P1 level (a) on WA-JEPA running (section P1).

**Read more.** [plans/2026-10-22-p2-match-prereg.md](plans/2026-10-22-p2-match-prereg.md) (P2 pre-registration, read-out lines);
[research/paper-outline](../../research/paper-outline/index.html) sections 3.2 and 5.

## P2: paired match against WA-JEPA

### The recipe (reusable for any grafted driver)

One command reads a list of drivers against one reference on several boards. Every allowed driver x board is its own
`jevdrive.bench` run in the GPU pool (HUGSIM: one worker job per card per run, so N drivers fill all free cards at once; navtest
scoring shards sized from the box's cores), and one CPU report job, chained `after` all of them, writes the paired tables:

```bash
cd ~/data/jev-drive; B=".venv/bin/python -m jevdrive.bench"
$B match --drivers STANDFMXH-L18-s0 STANDFMXH-L18-s1 RB0-F-s0 RB0-F-s1 --ref WA-JEPA --also P0 \
         --boards hugsim:exam navtest navhard --repeat p2 --out $DATA_DIR/runs/wm_surplus/p2_match/hugsim --drop-ply
# returns at once; <out>/DONE appears when the tables are written (wait on that one file, do not poll)
```

- Output: `<out>/match.md` (one row per driver x board: driver mean, reference mean, difference [95 % CI], HUGSIM HD parts
  dRC / dNC / dDAC / dTTC / dC) and per board `<board>_{arms,paired,strata}.md` (bench `report`; HUGSIM CIs clustered by scene).
- A cell the registry does not allow is not run and is listed with its reason (`Model.benches`): route-memory checkpoints are
  HUGSIM-only (the navsim plans path does not serve the branch, and lineages with AlpaSim nuPlan pool rows saw navtest logs).
- `--repeat L` makes a fresh, same-day paired read of every HUGSIM cell (the same label resumes); `--drop-ply` deletes the
  38 MB / episode `.ply` scene copies after the tables (opt-in, because a match may reuse shared run dirs).
- A new driver only needs a bench registry entry (`jevdrive/bench/models.py`, docs/bench.md "Adding things"); nothing here is
  driver-specific. WA-JEPA is served on HUGSIM by its shipped client (preset `exam` only), so pair on `hugsim:exam`.

### Drivers and boards read

See the pre-registration. HUGSIM 64, preset `exam` (PR #57 controller tree), repeat `p2`: WA-JEPA (reference), nuPlan
candidate B2 `STANDFMXH-L18-s0 / s1`, PAI checkpoints `R6DAG-L18-s1`, `R8K-L18-s2`, `NTV0-L18-s0`, no-S base `RB0-F-s0 / s1`,
anchors `P0`, `SH30-F-s0 / s1`; repeat `p2b` for WA-JEPA and `STANDFMXH-L18-s1`. Route-memory checkpoints get an empty route memory
on HUGSIM (no route source; in the training distribution). AlpaSim: stored held 460 @18 reads only (WA-JEPA 2026-10-12,
`alpasim_long/fam/wajepa0-3`, linked as read key `WA-JEPA-fam1012`), paired with `jevdrive.facts table --bench held460`.
Not reported: navtest / navhard (contaminated and not served), WOD (data deleted), B2D (closed lane).

### Results

Pending: `results/p2_match/`.

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

Results: pending (decision 454).

<!-- files:begin -->
## P0: third-party subjects on navtest (reference rows and probe subjects)

Decision row 455. Shortlist with repos, checkpoints, data and env recipes: [results/p0_subjects/shortlist.md](results/p0_subjects/shortlist.md);
per-model setup status (written by the setup lane): `results/p0_subjects/setup.md`. Subjects: DiffusionDrive (non-foundation,
88.1 PDMS), ReCogDrive-2B-RL (LLM: InternVL3-2B / Qwen2.5 hidden state -> DiT planner, 90.8), WoTE (BEV world model, 88.3);
DriveVLA-W0 (Emu3 world model + VLA, 87.2) only if its missing artifacts exist publicly. Measurement subjects only.

- **Bench family `agent`** (`jevdrive/bench/agents/<model>.json`, docs in `jevdrive/bench/models.py`): the fork's own NAVSIM agent
  as its eval script runs it (`jevdrive/bench/agent_poses.py`, the model's env), poses in chunks of whole logs, one GPU job each;
  then v2 EPDMS (our board) and v1.1 PDMS (the published metric) from the same poses; `summary.json` carries `PDMS_v1`,
  `published`, `delta_vs_published`. `python -m jevdrive.bench run --model DiffusionDrive --bench navtest [--subset navsim/p0-smoke-logs]`.
- **Reproduction gate** (`scripts/p0_repro.py submit --model M`): smoke on `navsim/p0-smoke-logs` (8 logs, 540 tokens) -> CPU
  gate job (released per-token CSV: within 1.0 PDMS on the same tokens; otherwise >= published - 3) -> full navtest; `verdict`
  writes `results/p0_subjects/<M>_full.json` (reproduced iff |PDMS - published| <= 0.5). A failing model is reported, not tuned.
- **P1 on the subjects** (`scripts/p1_agent.py`, same commands and outputs as `p1_wajepa.py`, through `lib/usage_probe.Probe`):
  sites ReCogDrive = the LM's last hidden state handed to the DiT; DiffusionDrive / WoTE = the two BEV maps of the TransFuser
  backbone that the planner reads. Pred files `PROBE-{recog-lmh,dd-bev,wote-bev}-<arm>`, read out with `scripts/p1_readout.py`.
  Outputs `$DATA_DIR/runs/wm_surplus/p1_agents/<model>/`.

Results: pending.

## Files

- `README.md` (README.md): do driving policies need grafted world …
- `2026-10-22-p2-match-prereg.md` (plans): P2 追平读数（WA-JEPA vs 当前谱系）预登记
- `match.py` (jevdrive/bench): a list of drivers against one reference …
- `index.html` (research/paper-outline): 

[results/](results/) 1 result files · [plans/](plans/) 2 live plans · [lib/](lib/) 1 library · [scripts/](scripts/) 1 entry points
<!-- files:end -->

Layout: `scripts/` entry points (live), `lib/` code other topics import, `archive/` one-off code of a concluded
experiment, `results/` small result files, `figs/` figures, `plans/` live plan notes.
