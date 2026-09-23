# Tokyo box 任务书（机械类）：驱动收尾、运维规则更新、TFv6 官方复现

状态: open，2026-09-23 写。执行者：Tokyo box 上的本地 agent。
性质：**纯执行**，步骤和验收都写死了。遇到不在本页范围内、需要判断的事，停下来，把问题写进文末「执行记录」，不要自行扩展。
需要科学判断的 CARLA 闭环研究另有任务书 [remote_carla_research.md](remote_carla_research.md)，不在本页范围。

## 规矩（先读）

- 仓库规则见 `CLAUDE.md`：代码、注释、日志、commit message 全英文；`todos/` 下的笔记用中文。
- **这台机器没有 GitHub 凭据，不要尝试 push，也不要配置凭据。** 在本地 `main` 上正常 commit，Mac 那边会通过 ssh 取走。不 rebase、不 amend 已有 commit、不 force。
- 大文件（raw 日志、逐帧 CSV/jsonl、checkpoint、视频）只放 `/data`，不进 git。`.gitignore` 已经挡住 `todos/**/results/` 下的逐帧文件。
- 超过 1 分钟的任务放 tmux 会话 `jev`，产出 `log.txt` / `events.jsonl`（见 `docs/long-runs.md`）。
- 系统级改动（内核、驱动、GRUB、apt）只做本页写明的那几步，每一步记录命令和输出。

## 0. 同步代码

Mac 会把最新的 `main` 推到这台机器的 `refs/heads/from-mac`。执行：

```sh
cd ~/mycode/jev-drive && git status --short && git merge --ff-only from-mac
```

必须是 fast-forward（本机 `main` 是它的祖先）。不是 fast-forward，或工作区有未提交改动，停下来报告。

## 1. 驱动收尾

2026-09-23 下午的只读检查结论（Mac 侧做的，供参考）：
- 当前启动的是 `7.0.0-31-generic`，**NVIDIA 驱动没有加载**：`linux-modules-nvidia-580-7.0.0-31-generic` 在 06:10 自动更新时停在 half-configured（`dpkg -l` 显示 `iF`），`/lib/modules/7.0.0-31-generic/kernel/nvidia-580/bits/` 下只有 `.sig`，没有 `.ko`；也没装 7.0 的 headers。
- `6.17.0-35-generic` 上 DKMS 已编好 `nvidia/580.173.02`，与用户态库 `580.173.02` 一致。
- 坏卡已拆，剩一张 **RTX 3090（PCI 02:00.0）**；另有 AMD 核显（7a:00.0）。

要做的：
1. 用 `6.17.0-35-generic` 启动（`sudo grub-reboot` 指定该菜单项，或开机时手选），重启。
2. 验证：`nvidia-smi` 正常，只列出一张 RTX 3090、index 0；`cat /proc/driver/nvidia/version` 与 `nvidia-smi` 的 driver 版本都是 580.173.02。
3. 防止复发：把 GRUB 默认固定到 6.17.0-35 这一项（`GRUB_DEFAULT` 写菜单项全名，`sudo update-grub`），并 `sudo apt-mark hold` 住 `linux-image-generic-hwe-24.04 linux-headers-generic-hwe-24.04 linux-modules-nvidia-580-generic-hwe-24.04 nvidia-driver-580`（按实际安装的包名调整）。或者在 `/etc/apt/apt.conf.d/50unattended-upgrades` 里把 `linux-*`、`nvidia-*` 列入 blacklist。选一种，记录下来。
4. 旧的私有用户态库 `/data/tools/nvidia-userspace-580.159.03/` 是给 580.159.03 内核准备的，现在不再适用。不删除，但任何启动脚本都不许再 source 它。

## 2. 实测 GPU 编号

1. PyTorch：`/data/envs/b2d-tcp/bin/python -c "import torch;print(torch.cuda.device_count(), torch.cuda.get_device_name(0))"`，预期 `1 NVIDIA GeForce RTX 3090`，**不设 `CUDA_VISIBLE_DEVICES`**。
2. CARLA：机器上有 AMD 核显，Vulkan 的枚举顺序可能把它排进来。分别用 `-graphicsadapter=0` 和 `-graphicsadapter=1` 启动 CARLA（offscreen 即可），用 `nvidia-smi --query-compute-apps=gpu_uuid,used_memory --format=csv` 看 CARLA 进程是否落在 3090 上；落不到 NVIDIA 上就说明选到了核显。记录一张表：rank → 实际设备。
3. 再带窗口（`DISPLAY=:0`）启动一次，确认显示器上能看到画面，且渲染仍在 3090 上。

## 3. 更新运维规则

以第 2 步的实测为准，把「GPU 0 坏了、只用 GPU 1、rank 反着」这套规则全部改成新现实（预期是：单卡 3090，CUDA index 0，直接用，CARLA 用实测出来的那个 rank）。要改的地方：

- `CLAUDE.md` 里 Tokyo box 那段加粗的规则。
- `docs/tokyo-box.md`：硬件表（第 20 行附近）、「GPU 0 is broken」一节（第 26–48 行附近，改写成实测表，旧事实挪到一句历史说明里）、所有示例命令里的 `CUDA_VISIBLE_DEVICES=1`。
- `scripts/b2d_tcp_campaign.py:32` 的报错文案（"Tokyo: 1, its GPU 0 is broken"）。
- `grep -rn "CUDA_VISIBLE_DEVICES=1\|GPU 0 is broken\|nvidia-userspace-580.159" docs scripts todos/*.md` 找到的其余位置。`todos/` 里已完成实验的历史记录**不改**，只改仍会被照着执行的文档和代码。
- `todos/2026-09-23-lateral-followup/diagnostics/driver-recovery/README.md` 顶部加一句：该方案仅适用于 580.159.03 内核，已被 2026-09-23 的重启取代。

改完跑一遍 controller 和 TCP 的测试（命令见 `docs/b2d-controller.md`、`docs/b2d-tcp-controller.md`，controller 125 项、TCP 17 项、验收分析 15 项），全部通过后 commit。

## 4. 清理

`tmux list-windows -t jev` 列出所有 window。里面没有进程在跑（`pane_current_command` 是 shell）的旧 window 可以关掉；正在跑东西的一个都不动，列在执行记录里。不删除 `/data` 下任何数据。

## 5. 装 TFv6（LEAD `cvpr2026`）

完全照 [planner-scout.md](2026-09-23-controller-next/planner-scout.md) 第 3、6 节在 GPU box 上验证过的配方，放到本机路径：

- 代码：`/data/third_party/lead-cvpr2026`，branch `cvpr2026` @ `730bc1a2f44d5f28312dd55f0ca958e94a24c038`。
- 环境：`/data/envs/tfv6`，uv venv，CPython 3.10，依赖按作者 `pyproject.toml`，torch 换成 2.8.0 / torchvision 0.23.0（cu128 build 在 3090 上也能用）；carla 0.9.15 cp310 wheel。推理用不到的 jupyter、open3d、wandb、ray、pyqt5 可以不装，但前提是 import 和 forward 仍然通过。
- 权重：HF `ln2697/tfv6` 的 `tfv6_resnet34` **三个 seed** 全下，放 `/data/checkpoints/tfv6/tfv6_resnet34/`，逐个核对 sha256 与 HF LFS oid。
- 预先缓存 `timm/resnet34.a1_in1k`，之后运行一律 `HF_HUB_OFFLINE=1`。
- 装 `ffmpeg`（`SensorAgent.setup` 找不到会直接 raise）。
- Bench2Drive：用我们锁定的 `/data/third_party/Bench2Drive`（v0.0.4，`7ec25d1`）。拉下作者自带的 `3rd_party/Bench2Drive`，只做 diff，把差异列进执行记录，**不替换**。
- 验收：跑一次 dummy forward（脚本思路同 planner-scout.md 第 6 节），三个 seed 都 `strict=True` 加载、输出有限。

## 6. 官方 TFv6 在 Dev10 上的复现

- 配置：官方 agent（`lead/inference/sensor_agent.py`），三 seed ensemble，官方 PID（`steer_modality=route`，`throttle/brake_modality=target_speed`），作者默认的 stop-sign/creeping 等启发式全开。不调任何参数。
- 路线：`/data/third_party/Bench2Drive/leaderboard/data/drivetransformer_bench2drive_dev10.xml`，TM seed 0 和 1 各跑一遍。
- 运行：Bench2Drive 的 `leaderboard_evaluator.py`，offscreen，在 tmux `jev` 里跑；raw 输出放 `/data/runs/b2d/tfv6-repro/seed{0,1}/`。基础设施失败（CARLA 崩溃、加载失败）可以重跑并记录次数；**驾驶失败不重跑**。
- 产出（小文件，进 git）：`todos/2026-09-23-controller-next/tfv6-repro/`
  - `results.csv`：每条路线 × seed 的 DS、RC、infraction 分项、是否完成、墙钟。
  - `summary.md`：两 seed 的 DS/RC/SR 均值，与作者公布的 94.72（这是 220 条上的数字，Dev10 只能作为量级参照）并列，注明不可直接比较。
  - 运行开销：每条路线墙钟、模型推理延迟 p50/p95、GPU 显存峰值、CPU 负载。
- 另外挑一条 Dev10 路线，带窗口（`DISPLAY=:0`）跑一次，截 3–4 张屏，放进 `tfv6-repro/`（PNG，每张小于 500 KB）。

## 完成后

在下面「执行记录」逐步写：做了什么、命令、结果、偏离了什么。commit 到本地 `main`，然后告诉用户“remote_carla 完成”。

## 执行记录

（执行者在此追加）
