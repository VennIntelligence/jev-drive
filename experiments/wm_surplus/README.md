# wm_surplus: do driving policies need grafted world models

status: live
decisions: 452, 453, 454
index: P2 match: current lineage vs WA-JEPA, HUGSIM 64 + AlpaSim held 460 (pending)

**Question.** The paper (d452) claims grafted foundation world models are surplus for end-to-end driving today. P1 measures how
much of the grafted model the driving head actually uses; P2 asks whether our lineage (Cinque 382M + body-schema / S-row training,
no grafted model) drives as well as WA-JEPA on the boards we may report, under the same runner and controller.

**Conclusion.** P2: pending (decision 453).

**Next.** P2 reads running (2026-10-22 evening); P1 section to be added by its lane.

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

<!-- files:begin -->
## Files

- `README.md` (README.md): do driving policies need grafted world …
- `2026-10-22-p2-match-prereg.md` (plans): P2 追平读数（WA-JEPA vs 当前谱系）预登记
- `match.py` (jevdrive/bench): a list of drivers against one reference …
- `index.html` (research/paper-outline): 

[results/](results/) 1 result files · [plans/](plans/) 2 live plans · [lib/](lib/) 1 library · [scripts/](scripts/) 1 entry points
<!-- files:end -->

Layout: `scripts/` entry points (live), `lib/` code other topics import, `archive/` one-off code of a concluded
experiment, `results/` small result files, `figs/` figures, `plans/` live plan notes.
