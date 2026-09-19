# Remote box

Read this when you need to log in to or use the GPU box.

- Login: `ssh autodl` (alias in local `~/.ssh/config`, key auth). User `ujs`, one RTX 4090 D (24 GB).
- Shell: interactive login is fish. Scripts and `ssh autodl '<cmd>'` run bash.
- Disk: the system disk is 30 GB, keep it empty. Put repos, datasets, checkpoints and envs in `~/data`.
  Caches (HF, torch, pip, uv) already point there.
- Public datasets: `/autodl-pub` (read-only). Check it before downloading big data.
- Python: use `uv`, and create venvs under `~/data`.
- Tools: tmux, ranger, btop, nvtop, fish, uv, opencode (opencode has no API key yet).
- A re-created instance loses users, keys and packages, and the host and port change. Re-check this page then.

Last verified: 2026-09-19
