# Long runs

Read this when you start anything on the box that takes more than about a minute.

## Where it runs

Lane jobs (GPU or CPU) run in the pool ([closed-loop-runbook.md](closed-loop-runbook.md#the-gpu-pool)); tmux is for
what runs outside it (the dispatcher, viewers, downloads, a hand-started server declared with `cl hold`):

- Always inside the box's tmux session `jev`, one window per job, named after the job.
  Never in a bare `ssh autodl '...'`: the job dies with the connection.
- Start it with `scripts/tmux_run.sh <name> <command ...>`. It uses a bash login shell (a plain tmux
  shell does not have `$DATA_DIR`), runs from the repo root, and keeps the window open afterwards
  with the exit code.
- Do not touch windows you did not start.
- Close your window when the job is done. The window stays open after exit only so you can read the exit
  code; once you have checked it and the logs, `tmux kill-window -t jev:<name>`. This applies above all
  to one-off downloads, smoke tests, probes and experiment runs. Keep a finished window only when there is
  a stated reason (a viewer on the desktop, a result someone still has to read), and say so in the run's
  todo. Before starting a batch of runs, and after finishing one, list your windows
  (`tmux list-windows -t jev`) and close the finished ones; a window whose pane has no child process
  (`pgrep -P <pane_pid>` is empty) is finished.

## What it writes

Every run gets its own directory `$DATA_DIR/runs/<experiment>/<tag>/<YYYYmmdd-HHMMSS>/`
(in Python: `jevdrive.runlog.RunLog("<experiment>", "<tag>")`) with three outputs:

| Output | For | Content |
|---|---|---|
| terminal (tmux window) | people watching live | log lines plus `tqdm` progress bars with rate and ETA |
| `log.txt` | people reading later | the same log lines, without progress bars |
| `events.jsonl` | scripts and agents | one JSON object per line, flushed per line: `{"t": <unix time>, "kind": ..., ...}`. Kinds used so far: `start`, `step_start`, `step_end`, `scalar`, `probe_result`, `end` |
| `tb/` | curves | TensorBoard scalars, only for values that mean something (loss, metrics, metric vs layer, throughput), not bare progress |

Results (`results.csv`, `timings.json`, ...) go in the same directory.
To follow a run from a script: `tail -f .../events.jsonl`, or poll for the `end` event.

## Stopping a job

Kill by PID, or kill the tmux window you started. Never `pkill -f <script name>`:
`-f` matches the whole command line, so it also kills every wrapper whose argv mentions
that script, including other people's supervisors and your own `ssh` command.
Several jobs share this box, so a stray `pkill` takes out work that is not yours.

## Committing from a shared tree

Several sessions and agents edit this working tree at once, so a commit must name its paths:
`git add <path> ...` or `git commit -- <path> ...`. Never `git add -A` and never `git commit -a`:
they sweep in whatever another session happens to have open, and the change lands on `main`
under a message that does not describe it. Nothing is lost when it happens, but the history lies.
Check `git status` before committing, and `git pull --rebase` before pushing.

## Reporting while it runs

Report every 3-5 hours while a job runs cleanly, not per file or step; at once only for an error, a stall, a decision
or completion. Agents submit their jobs and stop; they wait only on their own DONE / ERROR with one `run_in_background`
waiter (`P.wait`, or the lane script's `wait` stage). No polling loops in the agent.

## TensorBoard

- Ours: `scripts/tensorboard.sh` starts it in window `jev:tb`, port 6006, logdir `$DATA_DIR/runs`
  (all experiments at once; filter runs by path in the UI).
- Open it through AutoDL's "custom service" (port 6006) in the console, or
  `ssh -N -L 6006:localhost:6006 autodl` and then http://localhost:6006.
- AutoDL's default TensorBoard (port 6007, `/root/tf-logs`) runs as root under supervisord.
  Our user has no root, so it cannot be killed or pointed elsewhere. Ignore it.

## Before a long run (moved from CLAUDE.md)

Applies to OUR code only. Third-party libraries and other people's reproduction or baseline code are run
as they ship: measure them, do not rewrite them.
- Estimate wall time first. Anything above ~3 h gets a profiling pass before it starts: measure on a
  subset, find the actual bottleneck (disk, RAM, CPU, GPU compute, GPU memory, network), then rewrite the
  hot path as an expert would - parallel, vectorised, overlapping I/O with compute - until it is gone.
- Recipes we already use, their gains and identity status, and the measured headroom: [speed.md](speed.md).
- Tune the defaults for our box, not for a generic machine: batch size, DataLoader workers, prefetch,
  dtype, chunk sizes. Derive limits at runtime (`jevdrive.common.n_cpus()`, free VRAM, free disk) so the
  code still runs elsewhere, but leave OUR best values as the defaults.
- Verify the optimized code gives the same results (numerical equivalence on a subset), then record the
  before/after numbers and the bottleneck in the experiment's README or plan.
- Staged launch for every job of more than ~1 h (closed-loop B2D, CARLA generation, long training, big feature runs):
  run 1 unit, inspect it; then ~10 units (routes, worlds, folds, shards), inspect them against a written sanity
  checklist (completion / blocked / crash rates, value ranges against a known reference, outputs non-degenerate);
  only then the full batch. A pilot that fails the checklist stops the batch; never find out after the full run.
- Equality gates on fp16 plan points (identity, leakage, "same model through another path"): compare only the points
  inside the horizon the experiment uses (<= 4 s, the first ~15 plan points), and judge by the share of rows over the
  tolerance plus a wrong-model control, not by the maximum over all rows and points (far points reach ~190 m, where
  one fp16 ulp is 0.125 m; values in `experiments/op_parity/scripts/tsn_extract.py`: 0.03 m = two ulps at 32 m).
- Size everything from the live box (`python -m jevdrive.cl probe`, `jevdrive.common.n_cpus()`), never hardcoded; run
  independent work in parallel (one job per file / shard / chunk), keep hot data in RAM, overlap I/O with compute.

Last verified: 2026-10-22

## Sharing the box between several agents

- Every job, GPU or CPU, goes through the GPU pool, and a lane submits its whole chain at once, one pipeline per chunk:
  [closed-loop-runbook.md](closed-loop-runbook.md#pipelines-parallel-by-default) (the one place for pool commands,
  declarations, fan-out and chain shape). Work started outside the pool declares itself with `cl hold ... --pid PID`.
- `$DATA_DIR/runs/zeroshot-exam/gpu-plan.md` is the append-only log (`>>` only, never rewrite).
- `scripts/boxwatch.sh` is the box-wide 5 s memory / process sampler (`$DATA_DIR/runs/boxwatch/`); read it for a job
  that died by a signal ([sigkill.md](../experiments/cl_infra/results/closed-loop-infra-acceptance/sigkill.md)).
- A waiter that looks for a process must match a sentinel file, not `pgrep -f <name>` (it matches its own command line
  and waits forever).
