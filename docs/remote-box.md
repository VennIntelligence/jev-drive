# Remote box

Read this when you need to log in to or use the GPU box.
There are two boxes now: this one runs the experiments, and [tokyo-box.md](tokyo-box.md) is the one
with a monitor, for looking at CARLA with your own eyes.

- Login: `ssh autodl` (alias in local `~/.ssh/config`, key auth as `ujs`; `connect.weste.seetacloud.com`, port 25199).
  On 2026-09-28 the box moved to a new AutoDL instance, a full copy of the old one (same disks, users, keys, envs); the
  old host (`connect.westd.seetacloud.com:41708`) is off and has no GPUs. The root password is only in the Mac's
  gitignored `.env`; key login as `ujs` is all our work needs. sshd on the port is hammered by outside root login attempts
  and answers `Exceeded MaxStartups` / `Connection reset` to bursts of new connections: keep ControlMaster on and
  retry after a few seconds.
- **The box is elastic.** Cards, CPU quota, RAM and pids.max change with the instance: 7 cards / 175 cores / 644 GiB
  on 2026-09-28, 3 cards / 75 cores / 276 GiB on 2026-10-01. Do not trust a count on this page; run
  `.venv/bin/python -m jevdrive.cl probe` (cards, NUMA, quota, pids, memory, pool jobs per card); `python -m jevdrive.cl top` for the GPU pool.
  Closed-loop sizing: [closed-loop-runbook.md](closed-loop-runbook.md).
- GPUs: **NVIDIA RTX 6000D** (Blackwell, sm_120, 156 SMs, 450 W cap) since 2026-09-28, 83.6 GiB each
  (85651 MiB; the old RTX PRO 6000 Server Edition had 96 GB and 188 SMs), driver 595.91.07, CUDA 13.2, PCIe gen5 x8.
  Measured dense matmul: **144 TFLOPS bf16/fp16** per card, 60 fp32 (no TF32), about 55% of the 253-268 bf16 the old card
  reached on real models, so GPU-bound jobs take roughly 1.8x longer; memory copy 1.28 TB/s, H2D 26 GB/s.
  Pin jobs with `CUDA_VISIBLE_DEVICES`. CARLA's `-graphicsadapter=k` lands on CUDA card k (identity; checked with 7 cards and with 3).
  CARLA on one 6000D (2026-09-28, idle box, `experiments/cl_infra/archive/carla_threads.py probe`, 16 CPUs, reduced pools): one server 15.0 FPS,
  six servers 45.8 FPS aggregate at 8.0 GB each, 0 of 12 starts crashed; the six-per-card layout still fits.
- Region: West-E since 2026-09-28 (`$AutoDLRegion` = `west-E` in a login shell; West-D before). See [storage.md](storage.md) for why it matters.
- CPU and RAM: the host (2 x Xeon Platinum 8470Q, Sapphire Rapids) shows 208 CPUs (NUMA 0 = 0-51,104-155,
  NUMA 1 = 52-103,156-207; CPU i and i + 104 are hyperthread siblings), but the container's cgroup quota is what we get:
  25 cores per card so far (175 on the 7-card instance, 75 on the 3-card one; `/sys/fs/cgroup/cpu.max`), RAM from
  `memory.max`, pids.max 20480. `os.cpu_count()` reports the host count, so size thread and process pools with
  `jevdrive.common.n_cpus()`, not the host count. On 2026-10-01 card 0 sits on NUMA 0, cards 1-2 on NUMA 1.
- Speed: Qwen3-VL-4B features at 800 px ran at 21.1 ms/frame (8.8 GB peak VRAM) on the old RTX PRO 6000; the 4090 D did 30.8.
  Not re-measured on the 6000D; expect it to be slower by up to the matmul ratio above.
- **Never shut the instance down to add cards.** On AutoDL a stopped instance releases its GPUs to the pool and
  anyone can rent them; on 2026-09-26 a restart for a 6th card briefly lost every card. A restart also kills every
  tmux job: before one, have every job checkpoint and stop cleanly, and record its resume command.
- uv cannot hardlink from its cache into envs on this disk (every env is a full copy, ~8-10 GB), so
  `uv cache clean` is safe and frees space without breaking any env.
- Shell: interactive login is fish. Scripts and `ssh autodl '<cmd>'` run bash.
- Disk: the system disk is 30 GB, keep it empty. The data disk is 4.2 TB since 2026-09-28 (3.1 TB before; 1.4 TB free after the move). Put everything in `~/data`, see [storage.md](storage.md).
  Caches (HF, torch, pip, uv, modelscope) already point there.
- Public datasets: `/autodl-pub` (read-only). Check it before downloading big data.
- Code: `~/data/jev-drive`. Update with `git pull`. It uses a read-only deploy key (ssh alias `github-jev-drive`),
  so do not commit or push from the box. Delete `~/.ssh/id_ed25519_jev_drive` before saving an image.
- Python: use `uv`, and create venvs under `~/data`.
- A re-created instance loses users, keys and packages, and the host and port change. Re-check this page then.

Last verified: 2026-10-01 (remote box configuration)
