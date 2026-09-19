# Remote box

Read this when you need to log in to or use the GPU box.

- Login: `ssh autodl` (alias in local `~/.ssh/config`, key auth). User `ujs`, one RTX 4090 D (24 GB).
- CPU and RAM: the host has 128 cores and ~500 GB, but our container's cgroup allows **16 cores**
  (`/sys/fs/cgroup/cpu.max` = `1600000 100000`) and **62 GB RAM** (`memory.max`). `nproc` and `os.cpu_count()`
  still report 128, so size thread and process pools with `jevdrive.common.n_cpus()`, not the host count.
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
