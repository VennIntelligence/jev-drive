# 2026-09-28 box 数据盘二次盘点：能腾出什么

只列，不删。范围是今天早上 `tmp/2026-09-28-cleanup.md` 清理之后新出现或没细看的东西；那份清理已经做过一轮 trash/git rm，
这里不重复它的结论，只在需要时引用。盘点时间 17:58–18:40 CST，`du` 全部 `nice -n19 ionice -c3`，逐目录跑，没有并发，
没有用到任何删除 / 移动命令。

## 盘的现状

`/dev/md0` 4.2T，盘点开始时 3.7T used / 515G avail（89%），结束时 3.8T used / 481G avail —— 40 分钟内少了 34G，
这不是我动了什么，是 G lane、WL、P3、cosmos、op_adapt 五条在跑的 lane 同时在写。下面每一项都在收集数据那一刻查过
`/proc/*/cwd`、tmux 面板输出和进程命令行，没有猜。

## 在跑什么（不能动）

| lane | tmux 窗口 | 在写什么 |
|---|---|---|
| G lane | `nq4-g-r15`、`g-lane-watch` | `runs/nq4/gk`、`runs/nq4/g-lane`、`runs/sched` |
| WL | `wl-full` | `runs/wl/gen/ba`（62G，`b2d_run.py` 两个 worker 在跑 forks-ba 路线） |
| P3 | `p3-expand`、`p3-rel010` 等 8 个窗口 | `processed/waymo_ds*`、`ckpt/nq4_p3*`（drivestudio 场景 11/12 在 GPU 6 训练；`p3/expand.py` 在插行人） |
| op-adapt | `op-next-cache` | `processed/op_adapt`（op_adapt_next_cache 正在跑到 780/1192，ETA 8 min） |
| Cosmos | `cosmos-seg3` | `runs/cosmos`、读 `models/cosmos/*`（seg base 变体补第 3 对，pilot 判据已达标但 todo 里 `RESULTS_PLACEHOLDER` 还没填） |
| 闭环 | `op-arb-p2`（`op_arb_server.py cinque`） | 常驻进程，磁盘占用不大 |
| 开环插帧 | `opi-*` 五个窗口 | 今天 14:22–17:59 跑完，已退出（`exited with 0`），不是"在跑"，但产物是新的，见下 |

## 分类结果表（≥5 GB，按大小排）

标记：**用**=在用/在跑，**要**=登记结果或有文档/待办依赖，**建**=可重建（给了重建代价），**旧**=像是过时/已有结论不再需要原始体量。

| 路径 | 大小 | 类 | 是什么 | 依据 | 删除风险 |
|---|--:|:--:|---|---|---|
| `datasets/waymo_e2e/front3` | 676G | 要 | Waymo E2E 原始 slim TFRecord（`docs/waymo-e2e.md`） | `jevdrive/waymo.py`、`scripts/wod_zeroshot_cases.py` 直接读原始帧；`processed/waymo_e2e` 的 89G 特征只是 Qwen3-VL 一种 backbone 的产物 | **高**：删了以后任何换 backbone 重抽特征（P3 backbone 阶梯这类实验刚做过一轮）、`wod_zeroshot_cases.py` 要看原图，都要重下；Waymo 官方桶，重下要重新走 32-stream 下载管线，量大 |
| `datasets/navsim/sensor_blobs` | 367G | 要 | NAVSIM 官方 sensor blob | `navsim_zs`、`op_interp` 的 NAVSIM 子集都读它 | 高，公开数据但体量大，重下要时间 |
| `datasets/nuscenes` | 153G | 建 | nuScenes trainval（sweeps 121G + samples 30G） | `docs/storage.md`：本来就该从 `/autodl-pub/data` 的只读归档解压出来，不是下载的 | 低：本地解压，见 `scripts/extract_nuscenes.sh trainval`，不占网络，但要花时间重新解压全量 |
| `datasets/hugsim/scenes` | 93G | 要 | HugSIM 闭环场景资源 | `scripts/hugsim/*` 全套脚本依赖 | 中，公开发布数据，重下看对方链路 |
| `runs/op_interp/nav` | 170G | **旧** | NAVSIM 子集开环插帧诊断的中间帧缓存（`blend.npy`/`gimm_g0.2.npy`/`hold.npy`/`rife*.npy`/`warp*.npy`，每个 3–38 GB 的裸数组） | `research/openpilot-openloop-integration.md` 已经把最终 RFS/PDMS 数表和结论写完；`results.csv`、`report.log` 已经在同目录里；`opi-nav`/`opi-gimm`/`opi-wod` 三个窗口都已 `exited with 0` | **低**：这些是喂 openpilot 用的中间插帧结果，不是原始数据，缺了就是重新跑一遍 `scripts/op_interp_*.sh`（文档写的算力是 3 GPU·h 量级）；小结果表已经落地在仓库里 |
| `runs/op_interp/wod` | 34G | **旧** | 同上，WOD-E2E 子集版本 | 同上；`opi-wod` 窗口 15:53 就退出了 | 低，理由同上 |
| `runs/nq3/a/v1` | 134G | 要 | nq3 lane A 的 v1 世界帧集（openpilot 流 + backbone 特征） | `todos/2026-09-26-night-queue-3.md` 多处点名；是 `research/results/nq3/q3/` 的数据来源 | 高，是当前 nq3 分析在用的原始帧，重跑要重新起 CARLA 世界 |
| `runs/nq3/a/v0rr` | 13G | 要 | nq3 lane A 的 v0 重跑，墙钟时间的实测来源 | night-queue-3 第 718 行：T7 墙钟对比数字直接引用它 | 中，删了这条时间对比就没法复核 |
| `runs/nq3/b` | 22G | 要 | nq3 lane B 闭环臂运行目录 | 是 `research/results/nq3/cl/*.csv`（这次 git status 里正在改动）的数据源 | 高，当前正在用的闭环读数来源 |
| `runs/openpilot_rigs/frames` | 138G | 要 | openpilot rig study 帧集 | 今天早上的清理已经明确点名保留 | 高，按今天清理结论就是要留的 |
| `runs/wl/gen`（`ba` 62G + `p6` 4.1G） | 66G | 用 | WL 在跑 lane 的生成产物 | `wl-full` 窗口活跃写入 | 不能动，job 还在写 |
| `runs/p5v1/gen-pdm` | 50G | 要 | P5 v1 配对生成（PDM 分支） | 今天清理已点名"大数据全部保留" | 高 |
| `runs/p5v1/gen-ba` | 23G | 要 | 同上（BehaviorAgent 分支） | 同上 | 高 |
| `runs/hugsim-exam/scored-base` | 17G | 要 | HugSIM 考卷打分产物（base 组） | `research/results/hugsim-exam/` 引用；今天清理点名保留 | 高 |
| `runs/hugsim-exam/scored-op` | 15G | 要 | 同上（openpilot 组） | 同上 | 高 |
| `runs/hugsim-exam/check-op` | 6.4G | 要 | R2 清单的证据（今天清理原话） | `tmp/2026-09-28-cleanup.md` | 高，是留证据用的 |
| `runs/p6/gen/attempts` | 36G | 要 | P6 生成尝试记录 | 今天清理点名保留 | 高 |
| `runs/navsim_zs/openpilot` | 28G | 要 | openpilot 在 NAVSIM 零样本考卷的产物 | 今天清理点名保留；`jevdrive/navsim_qwen.py` 等脚本在用 | 高 |
| `runs/top10_t3/gen` | 21G | 要 | T3 生成产物 | `scripts/top10_t3_*.sh` 全套依赖；smoke 已在今天清理里单独清过，`gen` 没动 | 高 |
| `runs/cosmos`（`out` 16G + `gen` 3.8G + `clips` 1.2G） | 21G | 用 | Cosmos pilot 的渲染与生成产物 | `cosmos-seg3` 窗口还在写；`todos/2026-09-28-cosmos-pilot.md` 结果段还是 `RESULTS_PLACEHOLDER`，没写完 | 不能动 |
| `runs/nq4/gk` | 13G | 用 | G lane 运行目录（今天早上已经清过里面的错误文件） | `nq4-g-r15` 窗口活跃 | 不能动 |
| `runs/nq4/opl` | 11G | 要 | openpilot 臂 CARLA 验收，撤下决定的证据 | 今天清理原话保留 | 高 |
| `runs/navsim/metric_cache` | 15G | 建 | NAVSIM 官方 PDM 指标缓存 | 官方 devkit 在场景数据上重算得到 | 中：本地重算，不用下载，但要花 CPU 时间，具体多久没测 |
| `runs/nq4/k` | 7.8G | 要（待核实） | K 判据 lane 运行目录 | Mac 上已有 `research/results/nq4/k/{READY,verdict.json,checks/}`（这次 git status 里的未跟踪文件），像是结论已经提出来了，但没法确认 box 上原始目录是否还被复核逻辑引用 | 中，建议跑这条 lane 的人确认一遍再动，不放进"安全" |
| `runs/infra-accept` | 5.7G | 要 | 闭环验收基础设施（`docs/closed-loop-acceptance.md` 等多处引用） | 多篇 docs 引用 | 中 |
| `processed/op_adapt` | 181G | 用 | op-adapt 训练数据（navtrain 68G / wodtrain 51G / nusc 41G / p5 19G） | `op-next-cache` 窗口正在写 next cache | 不能动 |
| `processed/waymo_ds_veh/training` | 168G | 用 | P3-veh lane 的 3DGS 训练数据 | P3 相关 8 个 tmux 窗口在读写 | 不能动 |
| `processed/waymo_ds/training` | 137G | 用 | P3 lane 的 3DGS 训练数据 | drivestudio 训练进程直接指着这个路径 | 不能动 |
| `processed/waymo_e2e/features` | 88G | 要 | Qwen3-VL 冻结特征（`front3` 的一种 backbone 产物） | `jevdrive/waymo.py` | 高，重算要读回 676G 原始数据再跑一遍特征抽取 |
| `processed/drive_backbones`（`op_trainval` 45G 等） | 52G | 要 | backbone 阶梯实验的训练特征 | `todos/2026-09-22-p3-backbone-ladder.md` | 中，实验已出结论（decisions.md 第 1966/2166 行），但没人确认以后不会再加一档 |
| `processed/wod_zeroshot` | 39G | 要 | WOD 零样本考卷数据 | `jevdrive/elicit_e1.py` 等多处 | 高 |
| `processed/top10_exam` | 37G | 要 | Top-10 考卷数据 | `scripts/top10_t3_*` | 高 |
| `processed/nuscenes` | 25G | 建（待核实） | `jevdrive/nuscenes_index.py`/`features.py` 的输出目录，但 `v1.0-trainval` 子目录名看着像是原始 nuScenes 结构，可能与 `datasets/nuscenes` 有重叠 | 代码引用存在，但没查清是否真重复 | 未核实，建议核实后再归类 |
| `processed/statepol` | 13G | 要 | statepol lane 数据 | 多个 `statepol*` 脚本引用 | 中 |
| `processed/carla_p5v1_ba` + `carla_p5v1_pdm` + `carla_p5` + `carla_p6` | ~31G 合计 | 要 | P5/P5v1/P6 的 CARLA 特征缓存 | 对应 `p5v1.py`/`fusion_q4.py` 等 | 中高 |
| `ckpt/nq4_p3` | 18G | 用 | P3 lane checkpoint | 训练进程实时写入 | 不能动 |
| `ckpt/nq4_p3_veh` | 9.3G | 用 | P3-veh checkpoint | 同上 | 不能动 |
| `models/vace`（`Wan2.1-VACE-14B` 70G + `hf` 22G） | 92G | **旧** | Wan2.1-VACE-14B，插入实验里试过的一种局部重画方案 | `research/insertion-options.md`：VACE 输不出影子、还带色偏，比选中的方案 (c) 差一个数量级，"行人那一跑中途停掉"——已经是被放弃的分支 | **低**（决策已定，选中的是别的方案），但重下代价高：文档写 ModelScope 单连接 1–3 MB/s，约 2.5 h |
| `envs/vace` | 7.1G | **旧** | 配 VACE 用的 venv | 同上 | 低，`uv`/`pip` 重建即可（网络代价见上） |
| `models/Qwen3-VL-32B-Instruct-FP8` | 34G | 建 | backbone 阶梯实验测过的一档 | decisions.md 第 1966/2212 行：结果是"参数翻 8 倍、pre-onset 上一动不动"，scaling null，实验已有结论 | 低，HF 模型可重下，`scripts/download_models.sh` 有 id |
| `cache/huggingface/hub/models--Wan-AI--Wan2.2-TI2V-5B-Diffusers` | 32G | 建 | 同一实验测的另一档 | decisions.md 第 2166/2213 行："四个 tap 全是零"，已有结论 | 低，同上 |
| `cache/huggingface/hub/models--nvidia--diffusiongemma-26B-A4B-it-NVFP4` | 18G | 建 | openjev 用的模型，decisions.md 第 689 行已有读数（471 ms） | 已完成的对照实验 | 低，可重下 |
| `cache/huggingface/hub/models--nvidia--Alpamayo-1.5-10B` | 21G | 要 | Alpamayo baseline，多处零样本考卷引用 | `research/results/nq3/q1/alpamayo_cot.csv` 等 | 高，在用的对照模型 |
| `cache/huggingface/hub/models--facebook--vjepa2-vitg-fpc64-256` | 20G | 要 | V-JEPA 2 世界模型（论文主线：openpilot + V-JEPA2 世界模型闭环） | `CLAUDE.md`、`research/decisions.md` | 高，主线在用 |
| `cache/huggingface/hub/models--Qwen--Qwen3-VL-8B-Instruct` | 17G | 建 | 常规 HF 模型缓存 | 多处引用 | 低，可重下，但如果哪个 lane 正在用就先别删 |
| `cache/huggingface/hub/models--Zewei-Zhou--AutoVLA` | 16G | 要 | AutoVLA 对照模型 | `research/results/hack-audit/audits/wod_e2e__autovla.md` | 高 |
| `cache/huggingface/hub/models--Qwen--Qwen3-VL-4B-Instruct` | 8.3G | 建 | Qwen-Drive 基座 | `models/Qwen-Drive-1.0-4B` 的上游 | 低，可重下 |
| `cache/huggingface/hub/models--stabilityai--sd-turbo` | 5.2G | 建 | 常规 HF 缓存 | 未查到直接引用 | 低 |
| `cache/uv/archive-v0` | 34G | **建** | `uv` 包管理器的解包缓存 | 纯工具缓存，不含研究产物 | **极低**：下次 `uv sync` 会自动重新解压，本地已有的 wheel 不用重下 |
| `envs/*`（34 个 venv，每个 5–12G，合计约 200G 属于 ≥5G 档） | ~200G | 建 | 每条 lane 一个独立 Python venv（cosmos-transfer 12G、scout-tfv6/bridgedrive/drivor/drivestudio/hugsim/qwen-drive/hugsim-ltf/gtrs/sparsedrivev2/openjev/vace/simlingo/openpilot/blue/wajepa/statepol/r3d2/autovla/jevdrive/comma-wm/depth/ultralytics/b2d-tcp/gdino/sam3/efficientsam3 等） | `docs/remote-box.md`；逐个查了脚本，没发现明显重复/废弃的 venv（`hugsim` 和 `hugsim-ltf`、`navsim1` 和 `navsim2` 是两套不同配置，都在被脚本引用，不是重复） | 建议整体归为"可重建"而不逐个列：`cache/pip`/`cache/uv` 里已经有包缓存，重建主要是本地解包+装包，不太需要重新联网下载 | 每个 env 单独删风险都是低，但这次没找到"确定没人用"的那一个，不建议整批动 |
| `models/openpilot`（`trt_cache` 14G + `taps` 6.2G + `fp32` 1.5G） | 22G | 用 | 多条 lane（op_interp、op_adapt、op-arb、G lane）共用的 openpilot 权重与派生缓存 | 进程表里 `op_arb_server.py`、G lane 都在读 | 不能动 |
| `models/Qwen-Drive-1.0-4B` | 13G | 要 | 自训 planner（sft/rl/perception 三个 checkpoint） | 项目自己的模型 | 高，自研产物，删了要重训 |

## 类总计（按上表条目粗算，不是全盘精确加总）

| 类 | 合计 |
|---|--:|
| 用（在跑，不能动） | 约 700 GB（WL 66 + P3 三项 310 + op_adapt 181 + cosmos 21 + G lane 13 + ckpt 27 + models/openpilot 22 等） |
| 要（登记结果/当前依赖） | 约 1.9 TB（大头是 datasets 原始数据约 1.3 TB + runs 里各 lane 的"大数据全部保留"项约 500 GB + processed 里的特征/考卷数据） |
| 建（可重建，标了重建代价） | 约 380 GB（envs 全部 ~200G + cache 里 HF/uv 缓存部分 ~140G + models 里已有结论的实验权重 ~34G + datasets/nuscenes 153G 的本地重解压部分单独算，不叠进这个数字里） |
| 旧（像是过时，最值得先处理） | 约 305 GB（`runs/op_interp` 204G + `models/vace`+`envs/vace` 99G） |

（用 + 要 + 建 + 旧 没有精确等于 3.8T，因为很多 < 5 GB 的小文件、`third_party`（46G）、`datasets` 里没细列的部分没有逐项计入；这张表覆盖了 89% 的已用空间里能叫上名字的大头。）

## 最值得先看的安全候选（按省空间排序）

1. **`runs/op_interp/nav` + `runs/op_interp/wod`，共约 204G。** 今天下午到晚上跑完的开环插帧诊断，最终数表和结论已经写进
   `research/openpilot-openloop-integration.md` 和 `results.csv`/`report.log`，产 npy 的三个窗口都已退出。留着的话，
   代价只是"以后想换个插帧方法对比要重新渲染"，不是丢失任何结论。这是目前单项最大、最安全的一块。
2. **`models/vace` + `envs/vace`，共约 99G。** `research/insertion-options.md` 里已经写明 VACE 方案效果不如选中的 (c)，
   这条实验分支已经放弃。**唯一要注意的是重下代价**：文档记录 ModelScope 单连接只有 1–3 MB/s，重下约 2.5 h，所以只在确定不会回头对比时再删。
3. **`cache/uv/archive-v0`，约 34G。** 纯包管理器缓存，跟研究结果毫无关系，删了下次 `uv sync` 自动重建，风险最低。
4. **backbone 阶梯实验已经测完的三档权重**（`models/Qwen3-VL-32B-Instruct-FP8` 34G + HF 缓存里的 `Wan2.2-TI2V-5B` 32G + `diffusiongemma-26B-A4B-NVFP4` 18G，共约 84G）：
   decisions.md 里三档都已经有读数和结论（scaling null / 全零 / 471 ms 基线），HF 模型 id 都在 `scripts/download_models.sh` 里，要用随时能重下。
5. **`processed/nuscenes`（25G）和 `runs/nq4/k`（7.8G）需要先核实再决定**：前者可能和 `datasets/nuscenes` 有重叠，
   后者的结论似乎已经提到 Mac 上的 `research/results/nq4/k/`，但没法确认 box 上原始目录是否还被复核脚本引用 —— 这两个不放进"安全"名单，标成待核实。

## 没有查清楚、留给人判断的

- `envs/` 整体 208G 里，34 个 venv 逐个查了引用脚本，都能对上某条 lane，没找到"确定废弃"的那一个；如果要省这块空间，
  更合理的方向是把公共依赖（torch/cuda 那几 GB）抽出来共享，而不是删掉某个 env，这个不是"列出来删"能解决的，需要工程改动。
- `cache/huggingface/hub` 里除了上面点名的几个大条目，还有十几个 1–5 GB 的模型缓存没有逐个核实是否在用；152G 里能叫上名字的部分大约占了一半。
- `third_party`（46G，未细看）、`datasets/womd`、`datasets/comma1M`、`datasets/bench2drive-mini` 等中等大小目录没有展开核实。

## 已执行（用户批准）

2026-09-28 用户批准删除本文件点名的两项（`cache/uv/archive-v0`，`models/vace` + `envs/vace`），其余全部保留，包括第 3 项的模型权重。
删前查过 `/proc/*/cwd`、`lsof +D`、`ps aux | grep envs/vace`，三项均无进程在用，也没有 tmux 窗口在写。用 `rm -rf --` 删除：

- `cache/uv/archive-v0`（34G）
- `models/vace` + `envs/vace`（92G + 7.1G = 99.1G）

`df -h /root/autodl-tmp`：删除前 3.6T used / 669G avail（85%），删除后 3.5T used / 800G avail（82%），腾出约 131G
（与 34 + 99.1 ≈ 133G 基本对上，差额是同一时段其他 lane 并发写入/释放）。
