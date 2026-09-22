# jev-drive

Autonomous driving project (VennIntelligence).
Everything in this repo is English: code, comments, logs, docs, commit messages.
Exception: `research/`, `todos/` and `tmp/` are working notes for people and are written in Chinese.
Once a conclusion is settled, write the final English version in `docs/` or in code.

## Workflow
- Edit and commit locally, push to GitHub. The GPU box only pulls from GitHub.
- After every change: commit, push, then pull on the box (`ssh autodl 'cd ~/data/jev-drive && git pull'`).
  Scripts run on the box, so it must never lag behind `main`.
- The box is a tool for running experiments and nothing else. Research notes, figures, literature material and
  anything else unrelated to running an experiment live on this Mac, which is their source of truth. Never back
  them up or copy them to the box (no scp/rsync): a second copy drifts and leaves the two sides inconsistent.
  A change that touches only `research/`, `todos/` or `tmp/` needs no pull on the box; tracked files there
  arrive with the next ordinary `git pull` and that is all.
- There is a second box, the Tokyo box (`ssh ujs@100.108.238.8`), and it has a monitor. It is for *looking*:
  CARLA in a window, a manual drive, a screenshot or recording of a route. Experiments still run on the GPU box.
  **Its GPU 0 is broken — use GPU 1 only, and CARLA's `-graphicsadapter` rank is inverted there**
  (`-graphicsadapter=0` is the good card). See [docs/tokyo-box.md](docs/tokyo-box.md).
- Data, checkpoints and envs stay on the remote data disk, never in git.
- Never commit secrets: passwords, keys, proxy configs, subscription URLs.
- Anything longer than ~1 min runs in the box's tmux session `jev` (`scripts/tmux_run.sh`), with tqdm progress,
  and writes `log.txt` (human), `events.jsonl` (machine) and `tb/` (TensorBoard curves) to its run dir.
  See [docs/long-runs.md](docs/long-runs.md).
- While a long job runs cleanly, report every 3-5 hours, not per file or step.
  Report at once only for an error, a stall, a decision, or completion.

## Code
- Write expert-level code: efficient, compact, readable, with clear logic.
- The box container gets 25 cores, 120 GB RAM and one RTX PRO 6000 Blackwell, 96 GB (the host shows 208 cores;
  size pools with `jevdrive.common.n_cpus()`, see docs/remote-box.md). Use all of it:
  run independent work in parallel across cores (one job per file/archive/shard),
  keep hot data in RAM, batch on the GPU and overlap I/O with compute.

## Research notes (`research/`, `todos/`)
Written for people to read and discuss. Full rules: [research/README.md](research/README.md).
- Results land in [research/decisions.md](research/decisions.md) as they arrive, not when the experiment
  is finally over. A wrong entry is corrected in place - say what it used to claim and why it changed -
  never left standing with a new paragraph appended under it. Evidence getting weaker demotes an entry.
- Chinese natural-language prose, not keyword dumps. Technical terms stay in English.
- Gloss every term on its first use in each doc, in one short clause; use it bare after that.
- Show comparisons and results as tables. Use figures where a trend or distribution matters.
- Figures are publication quality (CVPR paper style): English labels, shared style and palette,
  each followed by 1-3 sentences saying what to look at.

## Before a long run
Applies to OUR code only. Third-party libraries and other people's reproduction or baseline code are run
as they ship: measure them, do not rewrite them.
- Estimate wall time first. Anything above ~3 h gets a profiling pass before it starts: measure on a
  subset, find the actual bottleneck (disk, RAM, CPU, GPU compute, GPU memory, network), then rewrite the
  hot path as an expert would - parallel, vectorised, overlapping I/O with compute - until it is gone.
- Tune the defaults for our box, not for a generic machine: batch size, DataLoader workers, prefetch,
  dtype, chunk sizes. Derive limits at runtime (`jevdrive.common.n_cpus()`, free VRAM, free disk) so the
  code still runs elsewhere, but leave OUR best values as the defaults.
- Verify the optimized code gives the same results (numerical equivalence on a subset), then record the
  before/after numbers and the bottleneck in the run's todo.

## Results, figures and what lives where
- Figures, docs and small result files (results.csv/md, metrics, timings) are pulled to this Mac and
  committed. Checkpoints, features, raw data and everything large stay on the box, see docs/storage.md.
- Every figure is collected by a doc that shows it and says what to look at: a figure with no doc and a
  doc pointing at a missing figure are both bugs. Moving, renaming or deleting one means fixing the other.
- [README.md](README.md) is the top-level index: every doc is reachable from it. Keep it current.

## Git
- Remote: `git@github-venn:VennIntelligence/jev-drive.git` (use the `github-venn` alias, not github.com).
- Branch: `main`. Commits are authored as Gaochengzhi (GitHub noreply email), set per repo, not globally.

## Docs
Start at [README.md](README.md), the index of every doc; how-to docs are under [docs/README.md](docs/README.md).
- GPU box (login, disk, tools): [docs/remote-box.md](docs/remote-box.md)
- Tokyo box (the one with a monitor, for looking at CARLA): [docs/tokyo-box.md](docs/tokyo-box.md)
- A download fails on the box: [docs/network-proxy.md](docs/network-proxy.md)
  (try `source /etc/network_turbo` first, then `proxy_on`)

Adding a doc: one topic per file, add a line to `docs/README.md`.
