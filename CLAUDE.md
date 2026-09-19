# jev-drive

Autonomous driving project (VennIntelligence).
Everything in this repo is English: code, comments, logs, docs, commit messages.
Exception: `research/`, `todos/` and `tmp/` are working notes for people and are written in Chinese.
Once a conclusion is settled, write the final English version in `docs/` or in code.

## Workflow
- Edit and commit locally, push to GitHub. The GPU box only pulls from GitHub.
- After every change: commit, push, then pull on the box (`ssh autodl 'cd ~/data/jev-drive && git pull'`).
  Scripts run on the box, so it must never lag behind `main`.
- Data, checkpoints and envs stay on the remote data disk, never in git.
- Never commit secrets: passwords, keys, proxy configs, subscription URLs.

## Code
- Write expert-level code: efficient, compact, readable, with clear logic.
- The box has 128 cores, ~500 GB RAM and one RTX 4090 D. Use all of it:
  run independent work in parallel across cores (one job per file/archive/shard),
  keep hot data in RAM, batch on the GPU and overlap I/O with compute.

## Git
- Remote: `git@github-venn:VennIntelligence/jev-drive.git` (use the `github-venn` alias, not github.com).
- Branch: `main`. Commits are authored as Gaochengzhi (GitHub noreply email), set per repo, not globally.

## Docs
Start at [docs/README.md](docs/README.md).
- GPU box (login, disk, tools): [docs/remote-box.md](docs/remote-box.md)
- A download fails on the box: [docs/network-proxy.md](docs/network-proxy.md)
  (try `source /etc/network_turbo` first, then `proxy_on`)

Adding a doc: one topic per file, add a line to `docs/README.md`.
