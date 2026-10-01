# Long runs

Read this when you start anything on the box that takes more than about a minute.

## Where it runs

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

An agent babysitting a long job reports on a **3-5 hour** cadence, not per file or per step.
Report immediately only when something needs a decision: an error, a stall, a route or budget
change, or the job finishing. A silent job that is making progress needs no message.

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
- Tune the defaults for our box, not for a generic machine: batch size, DataLoader workers, prefetch,
  dtype, chunk sizes. Derive limits at runtime (`jevdrive.common.n_cpus()`, free VRAM, free disk) so the
  code still runs elsewhere, but leave OUR best values as the defaults.
- Verify the optimized code gives the same results (numerical equivalence on a subset), then record the
  before/after numbers and the bottleneck in the experiment's README or plan.
- Staged launch for every job of more than ~1 h (closed-loop B2D, CARLA generation, long training, big feature runs):
  run 1 unit, inspect it; then ~10 units (routes, worlds, folds, shards), inspect them against a written sanity
  checklist (completion / blocked / crash rates, value ranges against a known reference, outputs non-degenerate);
  only then the full batch. A pilot that fails the checklist stops the batch; never find out after the full run.
- While a long job runs cleanly, report every 3-5 hours, not per file or step. Report at once only for an error,
  a stall, a decision, or completion.
- The box is elastic: cards, cgroup CPU quota, RAM and pids.max change between instances (7 cards / 175 cores /
  644 GiB on 2026-09-28, 3 cards / 75 cores / 276 GiB on 2026-10-01; RTX 6000D, 83.6 GiB each; the host always shows
  208 CPUs). Never hardcode them: read `python -m jevdrive.cl probe`, size pools with `jevdrive.common.n_cpus()`.
  Run independent work in parallel across cores (one job per file/archive/shard), keep hot data in RAM, batch on
  the GPU and overlap I/O with compute.

Last verified: 2026-09-20

## Sharing the box between several agents

When more than one agent (or person) runs jobs on the box at the same time:
- The box's schedule is `$DATA_DIR/runs/sched/table.tsv`, one row per lane (GPUs, CARLA workers per GPU, server
  index block, core list, status, GO file), kept by `scripts/sch_table.py` (`show`, `check`, `grant`, `revoke`, `finish`). The table holds live rows only: a lane
  updates its own row in place, its status column is one current sentence (not a log), and when the lane finishes or is
  revoked `sch_table.py finish <lane>` moves the row to `runs/sched/archive.tsv` (append-only history). Never append a row
  per state change.
  A grant writes the lane's shell-sourceable GO file, which the lane re-reads at its step boundaries; the G lane
  (`experiments/night_queue_4/lib/nq4_g_lane.py`) re-reads its row every round and yields cards to lanes that write
  `runs/sched/demand/<lane>.json`. (Was: `$DATA_DIR/runs/schedule.md`, a hand-edited timetable for the old
  five-GPU box; it went out of use on 2026-09-26/27 when the table and GO grants replaced it.)
  CARLA sizing and closed-loop lanes: [closed-loop-runbook.md](closed-loop-runbook.md) (`jevdrive.cl`). Before it: about 6 servers per card is still the GPU knee on the RTX 6000D box (docs/remote-box.md). The old
  thread-cap limit (pids.max 20480, ~650 threads per worker) no longer binds with reduced thread pools
  (docs/carla.md, "Budget with the reduced pools"); docs/bench2drive-cost.md has the 2026-09-25 measurements from the previous
  instance.
- Launch every scheduled job through `scripts/slot_run.sh <slot> [--after a,b] [--gpu N --vram-gb G] -- cmd`
  inside tmux. It waits with plain `sleep` until the dependency sentinels `$DATA_DIR/runs/sched/<slot>.done`
  exist (and the GPU has room), runs the command, then writes `<slot>.done` or `<slot>.failed`. A failed
  dependency fails the dependent slot instead of starting it on bad inputs.
- `$DATA_DIR/runs/zeroshot-exam/gpu-plan.md` is the append-only log (`>>` only, never rewrite).
- A job that dies by a signal (rc > 128) gets `$DATA_DIR/runs/sched/<slot>.death-<HHMMSS>.txt` from `slot_run.sh`:
  container memory, the largest processes and the last 2 min of `scripts/boxwatch.sh`, the box-wide 5 s
  memory/process sampler that `slot_run.sh` starts (one per box, `$DATA_DIR/runs/boxwatch/`). Unexplained SIGKILLs:
  experiments/cl_infra/results/closed-loop-infra-acceptance/sigkill.md.
- Agents arm their slots and then stop; they watch only their own final sentinel. No polling loops in the agent
  and no interim status messages: waiting costs nothing when bash does it, and tokens when an agent does it.
- Waiters that look for a process with `pgrep -f <name>` must not carry `<name>` on their own command line
  (e.g. inside `bash -c "... pgrep -f name ..."`): they match themselves and wait forever. Put the check in a
  script file, or match on a sentinel file instead. (2026-09-25: a HUGSIM waiter matched itself and stalled
  the WOD test download for ~7.5 h.)
