# Remote box

Read this when you need to log in to or use the GPU box.
There are two boxes now: this one runs the experiments, and [tokyo-box.md](tokyo-box.md) is the one
with a monitor, for looking at CARLA with your own eyes.

- Login: `ssh autodl` (alias in local `~/.ssh/config`, key auth; now `connect.westd.seetacloud.com`, port 41708).
  User `ujs`, two RTX PRO 6000 Blackwell Server Edition (96 GB each, sm_120) since 2026-09-24 (one before), driver 595.71, CUDA 13.2. Pin jobs with `CUDA_VISIBLE_DEVICES`.
- Region: West-D (`$AutoDLRegion` = `west-D`). See [storage.md](storage.md) for why it matters.
- CPU and RAM: the host has 208 cores and ~1 TB, but our container's cgroup allows **50 cores**
  (`/sys/fs/cgroup/cpu.max` = `5000000 100000`) and **240 GB RAM** (`memory.max`). `os.cpu_count()` can report
  the host count, so size thread and process pools with `jevdrive.common.n_cpus()`, not the host count.
- Speed: Qwen3-VL-4B features at 800 px run at 21.1 ms/frame (8.8 GB peak VRAM); the old 4090 D did 30.8.
- Shell: interactive login is fish. Scripts and `ssh autodl '<cmd>'` run bash.
- Disk: the system disk is 30 GB, keep it empty. The data disk is 2.5 TB. Put everything in `~/data`, see [storage.md](storage.md).
  Caches (HF, torch, pip, uv, modelscope) already point there.
- Public datasets: `/autodl-pub` (read-only). Check it before downloading big data.
- Code: `~/data/jev-drive`. Update with `git pull`. It uses a read-only deploy key (ssh alias `github-jev-drive`),
  so do not commit or push from the box. Delete `~/.ssh/id_ed25519_jev_drive` before saving an image.
- Python: use `uv`, and create venvs under `~/data`.
- Tools: tmux, ranger, btop, nvtop, fish, uv, opencode (opencode has no API key yet).
- A re-created instance loses users, keys and packages, and the host and port change. Re-check this page then.

Last verified: 2026-09-20
