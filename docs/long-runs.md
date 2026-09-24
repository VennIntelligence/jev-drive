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

Last verified: 2026-09-20
