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
- Equality gates on fp16 plan points (identity, leakage, "same model through another path"): compare only the points
  inside the horizon the experiment uses (<= 4 s, the first ~15 plan points), and judge by the share of rows over the
  tolerance plus a wrong-model control, not by the maximum over all rows and points. The far points reach ~190 m,
  where one fp16 ulp is 0.125 m, so a max-over-everything rule fails on rounding once the run is large; three
  op_parity lanes amended a registered gate for this on 2026-10-08 (margin-critic, turn-selnt, turn-selhug; values
  in `experiments/op_parity/scripts/tsn_extract.py`: 0.03 m = two ulps at 32 m, wrong-fold median difference 0.07 m).
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
- Every GPU job goes through the GPU pool ([closed-loop-runbook.md](closed-loop-runbook.md#the-gpu-pool)):
  `python -m jevdrive.cl submit --name N --vram GB [--carla N] [--cpu N] [--train] [--after ID] -- cmd`. One
  dispatcher (tmux `jev:pool`) starts each job on whichever card has room, sets `CUDA_VISIBLE_DEVICES`, allocates
  CARLA server indices and cores, and writes `log.txt`, `STATUS`, `DONE` / `ERROR` into the job's log dir. Nobody picks
  cards by hand; `queue`, `top`, `show ID`, `cancel ID` show and steer it. Work started outside the pool declares what
  it uses with `python -m jevdrive.cl hold ... --pid PID` (ends by itself when that process exits).
  CARLA sizing: about 6 servers per card is the GPU knee on the RTX 6000D box (docs/remote-box.md); the old
  thread-cap limit (pids.max 20480, ~650 threads per worker) no longer binds with reduced thread pools
  (docs/carla.md, "Budget with the reduced pools"); docs/bench2drive-cost.md has the 2026-09-25 measurements from the previous
  instance.
- **Fan-out rule.** Work that splits into independent runs (seeds, arms, variants, checkpoints, shards) is submitted
  as one pool job per run, all at once, with the next stage chained by `--after`:
  `python -m jevdrive.cl fanout --name N --arms a,b,c --vram GB --cpu N [--collect "cmd {arms}"] -- cmd ... {arm}`
  (Python: `P.fanout`). Never one job that loops over the runs, and never "submit seed 1 when seed 0 is done": the
  pool can only fill cards with jobs it has. A lane's whole remaining chain belongs in the queue, gated by `--after`
  / `--when-exists`, not in an agent that submits the next piece when it next looks.
- **Declare what the job uses.** `--vram`, `--cpu`, `--ram` are upper bounds, not safety margins on top of safety
  margins: the pool books a job at what finished runs of the same name measured once it has three of them, and a new
  name is booked at its declaration, so a first run declared at 24 GB that peaks at 6 keeps three more jobs off its
  card for ten minutes. Keep job names stable across reruns (`<what>-<arm>-s<seed>`) so history applies.
- **Look before you wait.** `python -m jevdrive.cl top` ends with the idle cards, the queued demand and an
  `UNDER-USED` line; `python -m jevdrive.cl usage --hours 24` splits the idle card-hours by cause. An idle card with
  work pending anywhere in a lane is a bug in how that lane submitted, or in the pool: fix it or report it.
- `$DATA_DIR/runs/zeroshot-exam/gpu-plan.md` is the append-only log (`>>` only, never rewrite).
- `scripts/boxwatch.sh` is the box-wide 5 s memory/process sampler (one per box, `$DATA_DIR/runs/boxwatch/`); read it
  for a job that died by a signal. Unexplained SIGKILLs: experiments/cl_infra/results/closed-loop-infra-acceptance/sigkill.md.
- Agents submit their jobs and then stop; they watch only their own final DONE / ERROR (`P.wait` or a
  `run_in_background` until-loop). No polling loops in the agent
  and no interim status messages: waiting costs nothing when bash does it, and tokens when an agent does it.
- Waiters that look for a process with `pgrep -f <name>` must not carry `<name>` on their own command line
  (e.g. inside `bash -c "... pgrep -f name ..."`): they match themselves and wait forever. Put the check in a
  script file, or match on a sentinel file instead. (2026-09-25: a HUGSIM waiter matched itself and stalled
  the WOD test download for ~7.5 h.)
