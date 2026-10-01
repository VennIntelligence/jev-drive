# jev-drive

Autonomous driving project (VennIntelligence). Everything is English (code, comments, logs, docs, commits), except
`research/` and a live lane's `experiments/<topic>/plans/`: Chinese working notes. Settled conclusions get an
English version in `docs/` or code. `tmp/` is gitignored scratch.

## Where to look
| Task | Go to |
|---|---|
| Any experiment: question, status, conclusion, code, results, figures | [experiments/INDEX.md](experiments/INDEX.md) (grep a name or alias) -> `<topic>/README.md` |
| What did we decide, is it still open | [research/decisions.md](research/decisions.md) -> `research/decisions/<NNN>.md` |
| Start a new experiment or result | rules below, [docs/lib.md](docs/lib.md), [experiments/TEMPLATE.md](experiments/TEMPLATE.md) |
| A CARLA / Bench2Drive run | [docs/closed-loop-runbook.md](docs/closed-loop-runbook.md) |
| Box, Tokyo box, env, data, proxy, long runs | [README.md](README.md) -> docs/ |

## Workflow
- Commit locally, push, then pull on the box (`ssh autodl 'cd ~/data/jev-drive && git pull'`); the box only pulls
  from GitHub and must never lag `main`. Changes only to `research/` or READMEs need no pull.
- The box runs experiments only. Notes, figures and literature live on this Mac; never copy them to the box.
- Tokyo box (`ssh ujs@100.108.238.8`, one RTX 3090) is for looking at CARLA: [docs/tokyo-box.md](docs/tokyo-box.md).
- Data, checkpoints and envs stay on the box data disk. Never commit secrets (passwords, keys, proxy configs, URLs).
- No Artifact web pages: every report is Markdown in the repo (topic README, `research/` or `docs/`). Showing docs
  without a push: [docs/web-reader.md](docs/web-reader.md).
- Jobs over ~1 min run in tmux `jev` via `scripts/tmux_run.sh` with log.txt / events.jsonl / tb/; before a job over
  ~1 h follow [docs/long-runs.md](docs/long-runs.md) (estimate, profile >3 h, staged 1 -> ~10 -> all launch).
  Report every 3-5 h, at once on error, stall, decision or completion.

## Code
- Expert-level: efficient, compact, readable. Use the whole box in parallel; never hardcode its size
  (`python -m jevdrive.cl probe`, `jevdrive.common.n_cpus()`, docs/remote-box.md).
- New experiments run inside `jevdrive.run.Run` and take train / val / test membership from `jevdrive.data.splits`
  (`run.use_split`); `jevdrive.cache`, `par`, `stats` are the defaults, deviate only with a stated reason
  ([docs/lib.md](docs/lib.md)). Old scripts are not migrated.
- An experiment is `experiments/<topic>/` (README from experiments/TEMPLATE.md + one INDEX.md line). Code a second
  topic imports goes to `lib/`, code three topics import to `jevdrive/`; finished one-off code sits in `archive/`.
- Third-party code is run as it ships: measure it, do not rewrite it.

## Closed loop (CARLA / Bench2Drive)
- Read docs/closed-loop-runbook.md first. Lanes are job lists run by `python -m jevdrive.cl run <lanefile>` on a
  lease from `python -m jevdrive.cl lease`; no other scheduler. Thread flags / env live only in `jevdrive/cl/profiles.py`.

## Results and research notes
- Results land in research/decisions.md as they arrive; a wrong entry is corrected in place (say what it claimed
  and why it changed); weaker evidence demotes it. Writing rules: [research/README.md](research/README.md).
- Small results, figures and docs are committed under the topic; large outputs stay on the box (docs/storage.md).
  Every figure has a doc that shows it and says what to look at; every doc is reachable from README.md.

## Git
- Remote `git@github-venn:VennIntelligence/jev-drive.git` (alias `github-venn`), branch `main`, author Gaochengzhi
  (GitHub noreply email), set per repo.
