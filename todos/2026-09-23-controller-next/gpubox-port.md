# 控制器实验栈迁到 GPU box

2026-09-23。Tokyo box 上的 B2D 轨迹控制器、真实 TCP 纵向对照和横向后续实验（40 个 commit，无 GitHub 凭据未推送）已合并进 `main`，并在无头 GPU box（RTX PRO 6000，25 核）上跑通：三套测试全过，冻结的 pose-g2 六例协议复现了 Tokyo 的四窗数字，8 路并行吞吐约 20 case/min。

## 合并与仓库清理

- `tokyo/main`（54e4406）经 ssh 拉到 Mac 后 merge 进 `main`（5f1ecc6），无冲突。`research/decisions.md` 两边都改过：`main` 写到第 22–25 条，Tokyo 在文末追加 26–29 条，自动合并后编号不冲突。
- merge 带进约 99 MB。其中 48 个逐帧文件（`*frames.csv`、`*samples.csv`、`*.jsonl`、688 KB 的 `failure-analysis.json`，共 41.3 MB，很多是同一数据的多个 edition 副本）已移出 git（4103c37），没有改写历史。原字节放在 GPU box 的 `$DATA_DIR/runs/b2d/controller/git-offload-v1/<仓库相对路径>`，与 git blob 逐一核对（`git hash-object`）一致，校验和在同目录 `SHA256SUMS`。指向这些文件的 markdown 链接改成了写明 box 路径的纯文本；`.gitignore` 挡住 `todos/**/results/` 下的这几类文件再次被加入。

## 代码改动（1cfa729、de19846）

改动只限于路径和显示方式，控制器本身（`b2d_controller.py`、adapter、agent）一个字节没动。

| 位置 | 改动 |
|---|---|
| `b2d_run.py` | 新增 `B2D_ZOO`（`B2D_ZOO_ROOT`，默认 `$DATA_DIR/third_party/Bench2DriveZoo`）和 `WINDOWED`：有 `DISPLAY` 就开窗口，否则 `-RenderOffScreen`，`CARLA_WINDOWED=0/1` 可以覆盖。`Server(windowed=None)` 跟随 `WINDOWED` |
| `b2d_controller_validate.py`、`b2d_calibrate.py`、`b2d_controller_campaign.py`、`b2d_tcp_campaign.py` | 去掉写死的 `windowed=True` 和 `/data/...`；`--windowed` 只在 server 真开窗口时才传 |
| `b2d_tcp_campaign.py` | 原来强制 `CUDA_VISIBLE_DEVICES=1`（Tokyo 的 GPU 0 坏了）。现在只在可见 GPU 多于一张、又没钉卡时报错，Tokyo 行为不变 |
| `b2d_controller_archive.py`、`test_b2d_controller.py`、`test_b2d_tcp_comparison_agent.py`、`benchmark_b2d_preprocess.py` | vendor 源码路径从 `DATA_DIR`/`CARLA_ROOT`/`B2D_ZOO_ROOT` 推出，缺省仍是 Tokyo 的 `/data` |
| `test_b2d_controller_campaign.py` | 假 server 补上 `windowed` 属性 |

Tokyo 照旧能跑：它的 launcher 本来就设 `DATA_DIR=/data DISPLAY=:0 CUDA_VISIBLE_DEVICES=1`。Tokyo 的 NVIDIA 私有用户态库（driver-recovery 的 `env.sh`）只写在 Tokyo run 目录的 `launch.sh` 里，仓库代码不引用，GPU box 也不用它。

GPU box 上补了两样东西：Bench2DriveZoo `tcp/admlp`@8a08b07（从 Tokyo 直接 tar 流过去，GitHub clone 在 box 上断流），以及 `envs/b2d-tcp`（Python 3.8、torch 2.2.2+cu121、numpy 1.23.5，版本对齐 Tokyo）。TCP checkpoint 没拷，测试用不到。

## 测试

| 套件 | 环境 | 结果 |
|---|---|---|
| controller `test_b2d_controller*.py` | `envs/carla` | 125/125，0 skip（vendor PID 源码对照现在也跑了） |
| TCP `test_b2d_tcp*.py` | `envs/b2d-tcp` | 17/17 |
| pose 验收分析 `test_pose_analysis.py` | `envs/carla` | 15/15 |

**坑：** 不设环境变量时有 4 个 controller golden 测试失败，差在最后一位（`-0.189595543119153` 对 `-0.18959554311915308`）。原因是 NumPy 自带的 OpenBLAS 按 CPU 选 matmul kernel：Tokyo 的 Zen 5 不认识，退回 `Barcelona`；Xeon 8470Q 选 `Cooperlake`，两者舍入不同。numpy 版本、glibc、libm 都查过，不是原因。设 `OPENBLAS_CORETYPE=Barcelona` 后逐位一致，125 全过。闭环 smoke 也设了这个变量，这样控制器的算术与 Tokyo 逐位相同。

## 闭环 smoke：冻结 pose-g2 六例

协议、routes、configs 原样使用：两臂 `baseline-zero`/`candidate-fixed-k`，路线 24240/26966/17563，单 server，headless。变的只有三处：variants 文件换成本机绝对路径，内容还是那两个 config 文件，SHA 没变；Tokyo 的 3 个参考 `route_reference.json` 放在 `$DATA_DIR/runs/b2d/controller/tokyo-reference/`，分析时用 `--reference-index` 指过去；冻结分析脚本本身一字未改。

- 结果：6/6 completed，0 collision，G2 全过。必要条件 155/156，唯一失败的 `turn/26966/1/heading_p95_delta` 和 Tokyo 是同一项。
- 墙钟：start→end 117 s，Tokyo 51 s。单 case 比 Tokyo 慢约 2.5 倍，比如 17563 是 16–19 s 对 6–7 s，因为 server tick 是单线程，Xeon 主频低。

| 窗口 CTE RMS (m) | baseline GPU box | baseline Tokyo | fixed-k GPU box | fixed-k Tokyo |
|---|---:|---:|---:|---:|
| 26966 右急弯 | .558701 | .558694 | .429015 | .429021 |
| 24240 左弯 | .107807 | .107807 | .078434 | .078434 |
| 17563 S1 | .223507 | .225250 | .208811 | .209774 |
| 17563 S2 | .271382 | .273410 | .261474 | .262014 |

Town10HD 逐位一致，Town05 差 1e-5 m，只有 Town12 大地图差到约 2 mm，符号和结论都不变。

原始数据：`$DATA_DIR/runs/b2d/controller/gpubox-port-v1`，分析在 `gpubox-port-v1-analysis`，launcher 和 variants 在 `gpubox-port-inputs/`。

## 并行吞吐

N 个 CARLA server，`--server-index 108+3k`（RPC 7400+150k，TM 13400+150k，间距 150），每个跑完整六例，启动间隔 15–20 s。

| N | case | 失败 | 总墙钟 s | 实测 case/min | 单 worker 墙钟 s | 稳态 case/min | 显存峰值 | load 峰值 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 6 | 0 | 117 | 3.1 | 117 | 3.1 | 5 GB | — |
| 4 | 24 | 0 | 184 | 7.9 | 128 | 11.3 | 24 GB | 12 |
| 6 | 36 | 0 | 215 | 10.1 | 136 | 15.9 | 35 GB | 17 |
| 8 | 48 | 0 | 247 | 11.7 | 142 | 20.3 | 47 GB | 20 |

“稳态”按 N×6/平均单 worker 墙钟算，不含启动错开的时间。结果没有漂：所有并行 case 的全程 CTE RMS 与串行 smoke 相比最多差 0.19 mm，全部 G2 通过。

**建议 8 路并发。** 单 worker 从 117 s 只涨到 142 s，扩展接近线性；25 核上 load 20，再加就要跟 CPU 抢了。40–80 个 case 按 8 路切，每路 5–10 个 case、共用一个 server，预计 4–8 min 跑完（含约 25 s 起服和换图）。本次探针脚本：`gpubox-port-inputs/parallel.sh`、`throughput.py`（在 box 的 run 目录，不进 git），输出在 `$DATA_DIR/runs/b2d/controller/gpubox-parallel-v1/n{1,4,6,8}`。

## 在 GPU box 上跑一个 arm × 一条路线

```bash
ssh autodl
cd ~/data/jev-drive
P=todos/2026-09-23-lateral-followup/pose-followup
IN=$DATA_DIR/runs/b2d/controller/gpubox-port-inputs     # route-26966.xml = routes.xml 里只留 26966
scripts/tmux_run.sh ctrl-one env OPENBLAS_CORETYPE=Barcelona PYTHONDONTWRITEBYTECODE=1 \
  $DATA_DIR/envs/carla/bin/python scripts/b2d_controller_validate.py \
  --routes $IN/route-26966.xml --controller-config $PWD/$P/configs/candidate-fixed-k.json --presets pursuit \
  --route-cruises $P/configs/route-cruises.json --out $DATA_DIR/runs/b2d/controller/<tag> --server-index 120
```

实测可用（`gpubox-one-case-v1`，1 case，G2 通过）。整套六例把 `--controller-config/--presets` 换成 `--variants $IN/variants.json`。

## 其他注意

- 真实 TCP 推理还不能直接在这台 box 上跑：torch 2.2.2+cu121 不支持 Blackwell（sm_120），而 Python 3.8 能装的 torch 最高 2.4.1。只有 CPU 测试能过。要跑真实 TCP 得另开环境，这件事需要单独决定。
- `envs/carla` 缺 `tqdm` 和 `tensorboard`。`b2d_tcp_campaign.py` 要用这两个；`b2d_controller_validate.py` 不要。
- 冻结分析的 `routes.csv` 和部分 todos 脚本仍写着 Tokyo 的 `/data/...` 默认路径。它们都是冻结产物，没改；在 GPU box 上用参数指定路径即可。
