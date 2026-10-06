# Shared libraries for new experiments

Five small modules that every new experiment builds on, so that run dirs, splits, caches, parallel loops and
confidence intervals mean the same thing everywhere. Tests: `python -m unittest tests.test_lib -v` (no GPU, no
network). A complete example in 40 lines: [scripts/lib_demo.py](../scripts/lib_demo.py).

**Benchmarks.** Running a model on navtest / navhard / HUGSIM / Bench2Drive and reporting it (arms, paired CIs, Shapley,
strata) is `jevdrive.bench` ([bench.md](bench.md)): one command, all GPUs through the pool; do not write a lane runner.

**Rule.** Every new experiment runs inside `jevdrive.run.Run` and reads its train / val / test membership from
`jevdrive.data.splits` (and calls `run.use_split` for each). `cache`, `par` and `stats` are the recommended defaults;
deviate only with a stated reason. Old scripts are not migrated.

## `jevdrive.run`: the run dir

```python
from jevdrive.run import Run, cli_args            # cli_args(parser): --seed, --resume DIR, --force
with Run("nq5", "pilot", seed=a.seed, config=vars(a), resume=a.resume) as run:
    for x in run.tqdm(units, desc="units"): ...    # terminal bar; STATUS + a progress event every 60 s
    run.scalar("loss", v, step); run.event("probe_result", auc=0.71); run.info("...")
    run.summary["acc"] = acc                       # lands in DONE
```

Layout `$DATA_DIR/runs/<experiment>/<tag>/<YYYYmmdd-HHMMSS>/` (same as `jevdrive.runlog.RunLog`, which keeps working):

| File | Content |
|---|---|
| `log.txt` | the log lines (every logger), no progress bars |
| `events.jsonl` | `{"t", "kind", ...}` per line, flushed per line: `start`, `scalar`, `progress`, `split`, `pmap`, `error`, `end` |
| `tb/` | TensorBoard scalars; created on the first `run.scalar` (needs torch; otherwise events only) |
| `meta.json` | git sha / branch / dirty, argv, cwd, host, python, pid, seed, config, env of interest, `splits` (name -> split_id), box probe (cores, memory, cards via `jevdrive.cl.box`), start / end / wall_s / ok |
| `STATUS` | one current sentence, `<date time> <name>: <text>`, overwritten (not a log) |
| `DONE` | written on a clean exit: JSON `{t, wall_s, **run.summary}` |
| `ERROR` | written on any exception (incl. Ctrl-C): header line + traceback; the exception is re-raised |

These are the `jevdrive.cl` pool job semantics: a watcher waits for `DONE` or `ERROR`. `resume=DIR` reopens a run dir and
renames a stale `DONE` / `ERROR` to `DONE.<ts>` / `ERROR.<ts>`. `seed_everything(seed)` seeds random, numpy, torch and
`PYTHONHASHSEED` (children) and returns a numpy Generator (`run.rng` when `Run(seed=...)`).

## `jevdrive.data.splits`: one registry of splits

```python
from jevdrive.data import splits
val = splits.load("nuscenes/val")                  # latest version; "nuscenes/val@v1" pins it
run.use_split(val)                                 # meta.json: {"nuscenes/val": "nuscenes/val@v1:d93d05f11081"}
df = df[val.mask(df.scene)]; splits.check_disjoint(splits.load("nuscenes/train"), val)
splits.available("navsim")                         # what exists
```

- A split is `jevdrive/data/splits/defs/<dataset>/<name>.v<k>.json` (members inline, or `<name>.v<k>.txt.gz` beyond
  5000) with `unit` (scene / sequence / token / log / route), `n`, `sha256`, `origin` (rule + frozen source file and its
  hash), `used_by`, `status` (official / frozen / consumed), `notes`.
- **split_id** = `<dataset>/<name>@v<k>:<sha256[:12]>`, the hash over the sorted members only. `load` refuses a file
  whose members no longer match its hash: a version is never edited. New membership = `splits.define(...)`, which
  writes the next version (or use a new name); old ids stay valid forever.
- Two different definitions never share a name (e.g. `navsim/e6-hold-logs` vs `navsim/nq3-hold-logs`). Registered
  v1 (2026-10-01): nuScenes train / val / op-adapt-{train,dev}; WOD train / val / test / val-half{0,1}-s0 /
  r2-{train,dev}; NAVSIM navtrain / navtest / navhard_two_stage / e6-sub / e6-hold-logs / lane-navtrain / n0-T /
  nq3-{hold-logs,heldout-eval,constraint} / r2-{train,dev}-logs; Bench2Drive bench2drive220 / bench2drive-0.0.4-val /
  dev10 / controller-* / l1-v2-heldout / l23-v2-heldout / nq4k-R{1,2} / wl-eval. Each file says where it came from.
- Dataset readers (index, frame iteration) are not here yet; they follow the repo restructure.

## `jevdrive.cache`: idempotent artifacts

```python
from jevdrive import cache
k = cache.key(params=dict(layer=18), inputs=[ckpt], code=extract, version=val.id)
feats = cache.cached(out / "feats.npy", k, lambda: extract(...), force=a.force)
if cache.done(path, k): ...                        # skip-if-done check for loops / pmap(skip=...)
```

- Key = sha256 over params (canonical JSON), input files (path + size + mtime; `content=True` hashes bytes), the
  source of `code` functions, and a free `version` string. Any change recomputes instead of silently reusing.
- Valid iff the file exists and `<file>.key` holds the key. Writes are tmp + rename, the `.key` last, so a crash leaves
  a recomputable entry, never a half file. Each write appends to `<dir>/MANIFEST.jsonl`. Formats by suffix: `.npy
  .npz .pt .json .parquet .csv`, else pickle. `--force` recomputes; resuming is the default.
- Feature files: `cache.feature_path(root, model, version, split_id, unit)` ->
  `<root>/<model>/<version>/<split_id slug>/<unit>.npy`; `version` names the extraction recipe, the split hash makes
  the dir content-addressed. Old feature dirs need not move.

## `jevdrive.par`: shards and parallel map

```python
from jevdrive import par
mine = par.shards(sorted(units), n, i)             # items[i::n], deterministic; name: par.shard_name(i, n)
res = par.pmap(work, units, run=run, skip=lambda u: cache.done(out / f"{u}.npy", key_of(u)))
res.raise_if_failed()                              # after the batch; res.errors = {index: traceback}
```

- Workers default to `jevdrive.common.n_cpus()` (cgroup quota), one task per unit; `threads=True` for I/O-bound
  work, `workers=0` runs inline (debugging). A failing or crashed unit never stops the batch; the summary and the
  first 20 tracebacks go to the run's log and a `pmap` event. `work` must be a module-level function.
- Units that `skip` marks done are not rerun, so rerunning the script resumes. Never `pkill -f`; processes that a
  unit spawns go through `jevdrive.cl.procs` (owned by pid + start time).

## `jevdrive.stats`: CIs and result tables

```python
from jevdrive import stats
r = stats.paired(a, b, groups=log_ids)             # {n, units, mean, lo, hi, n_boot, seed, alpha, mean_a, mean_b}
r = stats.bootstrap(per_route_scores)              # one value per independent unit
stats.write_table([dict(arm="A-B", **r)], run.path("results"))   # results.csv + results.md
```

- Default = the repo's dominant convention, bit-compatible (tests): percentile CI of the mean, B = 10000 resamples,
  `default_rng(0).integers(0, n, (B, n))`, linear interpolation (`nq3_cl_report.boot_mean`, `tfv6_rules.boot_paired`).
  `groups=` gives the cluster bootstrap of `traj.boot_ci` (resample clusters, ratio of sums) for frame-level data.
- Resample independent units: scenes, logs, sequences, routes. Frames of one sequence are not independent; pass
  `groups`. Paired = one resample applied to both arms (pairs with a NaN side are dropped).
- Tables: csv with every column at full precision; md with `.3f` and the constant `n_boot / seed / alpha` moved to a
  footer. CI columns are `lo` / `hi` (the repo's usual names); `stats.fmt(r)` gives `0.054 [0.031, 0.077]`.
