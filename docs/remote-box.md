# Remote box

Read this when you need to log in to or use the GPU box.

- Login: `ssh autodl` (alias in local `~/.ssh/config`, key auth). User `ujs`, one RTX 4090 D (24 GB), 128 cores, ~500 GB RAM.
- Shell: interactive login is fish. Scripts and `ssh autodl '<cmd>'` run bash.
- Disk: the system disk is 30 GB, keep it empty. Put everything in `~/data`, see [storage.md](storage.md).
  Caches (HF, torch, pip, uv, modelscope) already point there.
- Public datasets: `/autodl-pub` (read-only). Check it before downloading big data.
- Code: `~/data/jev-drive`. Update with `git pull`. It uses a read-only deploy key (ssh alias `github-jev-drive`),
  so do not commit or push from the box. Delete `~/.ssh/id_ed25519_jev_drive` before saving an image.
- Python: use `uv`, and create venvs under `~/data`.
- Tools: tmux, ranger, btop, nvtop, fish, uv, opencode (opencode has no API key yet).
- A re-created instance loses users, keys and packages, and the host and port change. Re-check this page then.

Last verified: 2026-09-19
