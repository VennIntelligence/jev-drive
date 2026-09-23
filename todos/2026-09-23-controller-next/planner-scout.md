# Planner scout：在 Blackwell 上跑一个真实的 learned planner 来测轨迹控制器

状态: done（survey 完成，top-1 在 GPU box 上通过 feasibility check，还没接 CARLA）
日期: 2026-09-23

## 目标

轨迹控制器需要一个真实 learned model 在 CARLA 闭环（Bench2Drive 协议）里给它喂轨迹。TCP 在 Tokyo box 上能跑，但它的环境是 Python 3.8 + torch 2.2 cu121，而 GPU box 的 RTX PRO 6000 是 sm_120（Blackwell 的 compute capability），需要 torch ≥2.7 + CUDA ≥12.8，这又要求 Python ≥3.9。已经决定不重建 TCP 的环境，改为挑一个能在 Blackwell 上跑的模型。本页只回答"选哪个、能不能跑起来"，不跑 CARLA，不做评测。

## 1. 候选一览

DS（Driving Score，route completion 乘以违规惩罚后的均值，0–100）和 SR（Success Rate，零违规跑完的 route 比例）都指 Bench2Drive 官方 220 条 route。标 * 的方法训练时蒸馏了 expert 特征。"sm_120 风险"指把环境搬到 torch ≥2.7 + cu128 时要改多少东西：没有自定义 CUDA op、作者 pin 的 torch 离 2.7 很近的算低风险。

| 方法 | B2D DS / SR | 来源 | 输出 | Horizon / 频率 | 传感器 | 权重、license | 作者环境 | 自定义 CUDA op | sm_120 风险 |
|---|---|---|---|---|---|---|---|---|---|
| **TFv6 / LEAD `cvpr2026`**（TransFuser v6，CVPR26） | **95.28**（regnety032）/ 94.72（resnet34）/ 91.60（vision only）；SR 未公布 | LEAD README 表格；arXiv 2512.20563 | 10 个 route checkpoint（2.5 m 起、间隔 1 m 的空间路径）+ 8 个 temporal waypoint + 8 类 target speed | waypoint 覆盖 +0.25…+2.0 s（4 Hz）；route 覆盖 2.5…11.5 m | 3 个前向相机拼成 1152×384 + LiDAR + 4 个 radar（有 no-radar、vision-only 变体） | HF `ln2697/tfv6`，每个 seed 约 276 MB，3 个 seed，MIT，不 gated | Py 3.10，torch 2.5 cu124，`carla==0.9.15` | 无（timm ResNet/RegNet + `nn.Transformer`） | **低**：作者 README 写明 Blackwell 用 torch 2.7 + cu128；本页实测 torch 2.8 cu128 可用 |
| LEAD `main` v1.5.x（同一作者重写的新 pipeline） | 93.6 ± 1.0（resnet34，3 相机 + LiDAR + radar，3 个 seed）；87.6（6 相机 vision only） | HF `ln2697/transfuser-carla-123d` 的 model card | 同上：route + waypoint + target speed | 同上 | 6 相机 384×384 + LiDAR + radar | 同一 HF 账号，MIT | Py 3.10–3.12，**torch 2.8 cu128**，但 pin 的是 `carla==0.9.16` | 无 | 低，但 client 是 0.9.16，而我们的 server 是 0.9.15（README 说 0.9.15 上也能 eval，未验证）；依赖 py123d |
| TF++（TFv5，carla_garage `leaderboard_2`） | 84.2 | LEAD README 的对比表 | path checkpoint + target speed | 同一家族 | 相机 + LiDAR | S3 zip，代码 MIT，权重 CC BY 4.0 | 老环境（conda） | 无 | 低，但被 TFv6 全面取代 |
| SimLingo（原名 CarLLaVA，CVPR25） | 85.07 ± 0.95 / 67.27 ± 2.11；CarLLaVA-base 85.94 / 66.82 | arXiv 2503.09594 表 2 | 20 个 path 点（间隔 1 m）+ 11 个 speed waypoint（间隔 0.25 s） | 约 2.5 s | 只用 1 个前向相机 | HF `RenzKa/simlingo`，Apache-2.0，`pytorch_model.pt` 2.6 GB | Py 3.8，torch 2.2.0，flash-attn 2.7.0，InternVL2-1B + LoRA，DeepSpeed | flash-attn（可以换 sdpa/eager） | 中：要换 Py 3.10+，重装 torch 2.7+，flash-attn 在 sm_120 上要自己编译或者去掉 |
| HiP-AD（ICCV25） | 86.77 / 69.09 | repo README | multi-granularity waypoint query | — | 6 相机 | GitHub release，Apache-2.0 | Py 3.8，torch 1.13 + cu117，mmcv 1.x | `deformable_aggregation`（要 `setup.py develop`），mmcv ops | **高** |
| ORION（ICCV25） | 77.74 / 54.62 | repo README；LinkVLA 表 1 | VAE 生成的 trajectory | — | 6 相机 | HF `poleyzdk/Orion`，Apache-2.0 | Py 3.8，torch 2.4.1 cu118；FP32 需要 32 GB | mmcv/mmdet3d（StreamPETR/OmniDrive 系） | 高 |
| DriveTransformer-Large（ICLR25） | 63.46 / 35.01 | repo README | planning query 输出 trajectory | — | 6 相机 | Google Drive / 百度云 | mmcv 系 | 有 | 高 |
| UniAD-Base / VAD（Bench2DriveZoo） | 45.81 / 16.36；42.35 / 15.00 | Bench2DriveZoo README | trajectory | 3 s，2 Hz | 6 相机 | HF + 百度云 | mmcv 1.7 系 | deformable attention 等 | 高，而且分数低 |
| ThinkTwice* / DriveAdapter* | 62.44 / 31.23；64.22 / 33.08 | SimLingo 表 2 | trajectory + control | — | 相机 + LiDAR | B2D 版权重是否公开未核实 | mmcv 系 | 有 | 高 |
| TCP-traj*（现在 Tokyo 上跑的） | 59.90 / 30.00 | 同上 | 4 个 waypoint | +0.5…+2.0 s | 前向相机 | 已有 | Py 3.8，torch 2.2 | 无 | 已决定不迁移 |
| LinkVLA | 91.01 / 74.55 | arXiv 2603.01441 表 1 | action token（粗到细） | — | — | **没找到代码或权重** | — | — | 不可用 |
| CaRL | 在 B2D 上几乎是 zero-shot | LEAD README | **直接输出 control** | — | privileged BEV | HF `ln2697/tfv6` 里附带 | — | — | 不适用：没有轨迹，而且输入是 privileged 的 |

carla 0.9.15 的 wheel：PyPI 上只有 cp37、cp38、cp39、cp310 的 manylinux wheel。0.9.16 有 cp310、cp311、cp312。所以要配 0.9.15 server 又要 torch ≥2.7，**Python 3.10 是唯一合适的版本**，TFv6 `cvpr2026` 正好 pin 在 3.10 + 0.9.15。

## 2. 按我们的需求排序

| 排名 | 方法 | (a) sm_120 porting 风险 | (b) 能不能喂我们的控制器 | (c) B2D 分数 | (d) CARLA client |
|---|---|---|---|---|---|
| 1 | **TFv6 `cvpr2026`，`tfv6_resnet34`**（3 个 seed 做 ensemble） | 低，已实测通过 | 能：有带时间戳的 waypoint（+0.25…+2 s），另有空间 route 和 target speed | 94.72（regnety032 是 95.28） | 0.9.15 cp310，已实测 |
| 2 | LEAD `main` v1.5.0 resnet34 | 低（torch 2.8 原生） | 能，输出和 1 一样 | 93.6 | pin 的是 0.9.16，要换成 0.9.15 wheel，没验证 |
| 3 | SimLingo | 中 | 能：path 加上 0.25 s 间隔的 speed waypoint | 85.07 | 作者用 0.9.15 cp38，要改到 cp310 |
| 4 | HiP-AD | 高 | 能 | 86.77 | 要 port 老的 mmcv 栈 |

**推荐 TFv6 `cvpr2026` 的 `tfv6_resnet34`。** 理由：B2D DS 是目前能拿到权重的方法里最高的，比 SimLingo 高约 10 个点；它是纯 PyTorch，没有需要为 sm_120 重编译的东西；作者 pin 的 Python 和 CARLA 版本跟我们的 0.9.15 server 完全对得上；eval 入口本身就是 Bench2Drive 的 `leaderboard_evaluator.py --agent=lead/inference/sensor_agent.py`，可以直接接进我们已有的 Bench2Drive 协议。模型和 control 是分开的：`ClosedLoopInference` 先出 route、waypoint 和 target speed，再由 PID 转成 control，所以替换成我们的控制器边界清楚。选 resnet34 而不选 regnety032，是因为两者只差 0.56 DS，resnet34 的权重小一半，而且 LEAD 的消融都是在 resnet34 上做的。第 2 名只在以后要换 CARLA 0.9.16 时才值得考虑。

## 3. Feasibility check（GPU box）

只做 import、加载权重和一次 dummy forward，没有启动 CARLA。

| 项 | 结果 |
|---|---|
| 代码 | `~/data/third_party/scout/lead-cvpr2026`，branch `cvpr2026` @ `730bc1a2f44d5f28312dd55f0ca958e94a24c038`，sparse checkout 了 `lead/`（不含 `*.h5` 地图）、`scripts/`、`pyproject.toml`、`uv.lock`，共 6.6 MB。`3rd_party/CARLA_0915` 软链到 `~/data/third_party/carla/CARLA_0.9.15` |
| env | `~/data/envs/scout-tfv6`：uv venv，CPython 3.10.21；依赖按作者 `pyproject.toml` 的 pin 安装，只把 torch/torchvision/torchaudio 换成 **2.8.0 / 0.23.0 / 2.8.0**（PyPI 默认的 cu128 build），index 用 tuna |
| 关键版本 | torch 2.8.0+cu128，arch list 里有 `sm_120`；carla 0.9.15（cp310 wheel）；timm 1.0.19；numpy 1.26.0；transformers 4.46.3 |
| 权重 | `~/data/checkpoints/scout/tfv6/tfv6_resnet34/{config.json, model_0030_0.pth}`，只下了 seed0。sha256 `7ce71a2e…c780ce`，与 HF LFS oid 一致 |
| 加载 | `create_model(TrainingConfig(config.json))` 之后 `load_state_dict(strict=True)` 返回 `All keys matched`；参数量 68.9 M；推理用 bf16 autocast（沿用训练 config） |
| forward | dummy 输入（rgb 1×3×384×1152，LiDAR BEV 1×1×320×384，radar 1×300×5，target point、speed、command）：输出 `pred_future_waypoints` (1,8,2)、`pred_route` (1,10,2)、`pred_target_speed_distribution` (1,8)，全部是有限值。直行输入下 route 从 x=2.45 m 开始、间隔 1 m，和源码一致 |
| 延迟 / 显存 | 1 个模型 46.7 / 48.5 / 51.0 ms（mean / p50 / p95），峰值显存 0.9 GB；3 个 seed 串行 ensemble 136.7 / 140.0 / 145.5 ms，1.8 GB。**测的时候 GPU 被 P5 VLM 任务占到 100%**，所以这是在争用下的上限，不是干净的数字 |
| GPU 用量 | 每次不到 1 分钟、不到 2 GB，所以没有写 `RESOURCE_LEDGER.md` |
| 结论 | **能跑，没有 sm_120 blocker** |

**要进闭环还有三件小事（不是 blocker，都还没做）：**

1. 模型构造时，timm 会从 HF 下载 ImageNet 预训练权重（`timm/resnet34.a1_in1k`），随后又被 checkpoint 覆盖。box 上已经走 `proxy_on` 缓存到 HF cache（84 MB），以后跑的时候设 `HF_HUB_OFFLINE=1` 就行。
2. `SensorAgent.setup` 找不到 `ffmpeg` 会直接 `raise`（用来压缩视频），box 上没装 `ffmpeg`。解决办法是装一个静态二进制，或者在我们的 wrapper 里绕开。
3. 作者自带一份 `3rd_party/Bench2Drive`（leaderboard + scenario_runner），这次 sparse checkout 没拉。要先跟我们锁定的 `Bench2Drive@7ec25d1`（v0.0.4）做 diff，确认用哪一份；原则上 eval 应该用我们锁定的那份。

## 4. 接到我们控制器之前要处理的契约差异

我们的控制器是 `update(traj_xy, t_frame)`，输入要求 rear-axle 原点、x 向前、y 向左、+0.25…+5.0 s 共 20 个点。TFv6 的 waypoint 标签来自 `lead/expert/expert_data.py:1248-1260`：`inv(ego_matrix_t0) @ ego_matrix_future[:3,3]`，其中 `ego_matrix` 取自 `ego_vehicle.get_transform()`。

| 项 | TFv6 | 我们的控制器 | 需要的转换 |
|---|---|---|---|
| 原点 | CARLA actor origin（MKZ 的 actor 原点，rear axle 在其后 1.389 m） | rear axle | 平移 x −1.389 m，外加 yaw 变化带来的那一项（同 TCP contract 第 1 节里的 d·sin Δψ） |
| 坐标轴 | CARLA 车体系：x 向前、**y 向右** | y 向左 | y 取反 |
| 时间 | 8 点，+0.25…+2.0 s | 20 点，+0.25…+5.0 s | **horizon 不够**：要么用 route（到 11.5 m）+ target speed 把轨迹外推到 5 s，要么让控制器接受 2 s 的 horizon |
| 频率 | 模型每个 20 Hz tick 都跑一次 | 20 Hz 控制 | 一致 |
| 官方 control 路径 | 横向用 route + lateral PID，纵向用 target speed + PID（`steer_modality=route`，`throttle/brake_modality=target_speed`），waypoint 本身默认**不用于控制** | — | 用 waypoint 做对照时要注意：官方的 DS 不是 waypoint 跑出来的 |
| 后处理 | 3 个 seed 平均；brake_threshold 0.9；另有 stop-sign 和 creeping 等启发式（README 说关掉只影响约 1 DS） | — | 对照实验要固定这些开关 |

最后一行的含义是：TFv6 的 95 DS 是 route + target speed 的解耦表示配上它自己的 PID 跑出来的。把 waypoint 喂给我们的控制器，比较的对象应该是"TFv6 + 官方 PID"这条基线，而不是论文里的数字。

## 5. 磁盘

| 位置 | 大小 |
|---|---:|
| `~/data/envs/scout-tfv6` | 9.5 GB（装全了作者的依赖，包括 jupyter、open3d、wandb、ray、pyqt5；只做推理的话可以裁掉很多） |
| `~/data/checkpoints/scout/tfv6` | 264 MB（只有 seed0；三个 seed 共 828 MB） |
| `~/data/third_party/scout/lead-cvpr2026` | 6.6 MB |
| uv cache 新增 | 约 2.0 GB（torch cp310 847 MiB、open3d 382 MiB、triton 148 MiB 等；nvidia-* wheel 复用了 tuna index 的已有缓存） |
| CPython 3.10.21（uv managed） | 87 MB |
| HF cache `timm/resnet34.a1_in1k` | 84 MB |

安装踩的坑：`uv pip install` 不读 `pip.conf`，不指定 `--index-url` 时直接走 pypi.org，当时链路被一个 ModelScope 下载占满，50 分钟只下了 1.7 GB。改用 `--index-url https://pypi.tuna.tsinghua.edu.cn/simple` 后 8.6 分钟装完，因为 qwen-drive 环境装 torch 2.8 时在 tuna 下过的 nvidia-* wheel 直接命中了缓存。

## 6. 复现

```bash
cd ~/data/third_party/scout/lead-cvpr2026
export PYTHONPATH=$PWD:$PWD/3rd_party/CARLA_0915/PythonAPI/carla OMP_NUM_THREADS=4 HF_HUB_OFFLINE=1
~/data/envs/scout-tfv6/bin/python ~/data/runs/scout/tfv6/dummy_forward.py ~/data/checkpoints/scout/tfv6/tfv6_resnet34 1   # 最后一个参数是 ensemble 的模型数
```

日志和脚本在 `~/data/runs/scout/tfv6/`：`install2.log`、`forward_1model.txt`、`forward_3model.txt`、`dummy_forward.py`。这是一次性脚本，不进 git。

## 下一步（待决定）

- [ ] 下载 seed1/2，装 `ffmpeg`，用我们锁定的 Bench2Drive 跑一条 route（例如 23687），在不修改的情况下确认 TFv6 + 官方 PID 能开起来
- [ ] 写 wrapper：把 TFv6 的 waypoint（或 route + speed）转成 rear-axle、y 向左、5 s 的轨迹，交给 `update(traj_xy, t_frame)`；先离线对齐坐标
- [ ] 显存 1–2 GB/agent，可以和 8 路 CARLA 并行；延迟要在 GPU 空闲时重测一次
