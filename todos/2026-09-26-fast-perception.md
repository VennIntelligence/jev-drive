# 快通道感知：SAM 3.1 的提速选项与更快的替代检测器（延迟 × 召回，同一批帧）

状态: running（预登记写于 2026-09-26 00:10 CST，任何候选的延迟或召回输出之前）
主题: [decisions 第 43 条](../research/decisions.md)、[融合前诊断](2026-09-25-fusion-diagnostics.md) 的 Q4 / Q4d / Q8
卡: GPU 0（独占）；CPU ≤ 20 核；slot 记在 `$DATA_DIR/runs/zeroshot-exam/gpu-plan.md`，标 `[FASTPERC]`

## 目的

融合前诊断里 SAM 3.1（Meta 的开放词表分割模型，文本 prompt）是唯一在 P5 行人 family 上给出非零读出的结构化状态来源
（规则门翻转 31–40%），但有两个问题：

1. **延迟**：Q4d 在我们的卡上 batch 1、6 个 prompt，1 路相机 p50 188 ms，3 路 558 ms；Q8 说 ParkingCrossingPedestrian 的反应窗口 p25 只有 0.53 s，
   558 ms 的感知进不了快通道（20 Hz policy，一拍 50 ms）。
2. **召回**：P5 hazard 行人全部 0.49（≤ 20 m 0.88），nuScenes 行人 ≤ 40 m 0.36（0–10 m 0.68）。远处的「漏检」一半以上是 BEV 放错：
   平地假设把站在人行道上的行人推远 2–7 m，oracle 地面高度把 P5 行人 20–40 m 从 0.12 提到 0.50。

用户问：更快的「segment anything」变体（量化、蒸馏、只做 image 而不做 video）能不能同时解决延迟和读出。
本文件把两件事分开量：**延迟**是模型选择的问题；**BEV 放置**是深度 / 地面高度的问题，换一个更快的 2D 分割器原则上不改变它。
每个候选都报平地召回和 oracle 高度召回，两者之差就是「放错」那一份。

## 调研（2026-09-26，网页与模型仓库，权重逐个核实过是否真的可下载）

| 候选 | 类型 | 接受文本概念？ | 权重（核实日期 2026-09-26） | 备注 |
|:--|:--|:--|:--|:--|
| SAM 3.1 detector（现役） | 开放词表检测 + 分割 | 是 | ModelScope `facebook/sam3.1`，盘上 | 3.1 的改动（2026-03-27 release note）全在 video：Object Multiplex、torch.compile 融合、减少 CPU-GPU 同步；image 检测器没有新的提速项，也没有官方 fp8 / int8 / TensorRT |
| SAM 3.1 + 官方开关 | 同上 | 是 | 同上 | 仓库自带：`Sam3Processor(resolution=…)`、backbone `compile_mode`、`use_rope_real`；flash-attn-3 只支持 Hopper，Blackwell 用不上 |
| SAM 3 / 3.1 video 模式 | 检测 + 跟踪 | 是 | 同上 | 每帧仍跑检测器再加跟踪器，单帧只会更慢；不作为提速选项 |
| EfficientSAM3（Zeng 等，arXiv 2511.15833） | SAM 3 的蒸馏学生：EfficientViT / RepViT / TinyViT 视觉编码器 + MobileCLIP 文本编码器，decoder 冻结沿用 SAM 3 | 是（同一 `Sam3Processor` 接口） | HF `Simon7108528/EfficientSAM3`（Apache-2.0，未 gated）：Stage 3 全模型 `efficientsam3_ft/efficientsam3_{efficientvit,repvit,tinyvit}.pt` 各约 0.47–0.49 GB，2026-06-11 发布 | 参数 89–95 M，对 SAM 3 的 862 M；作者没报延迟和 SA-Co 精度 |
| SAM3-LiteText（同一作者，arXiv 2602.12173） | 只换文本编码器 | 是 | 同上 `sam3_litetext/`、`sam3p1_litetext/`，各 2.2–2.6 GB | 视觉编码器不变；固定 prompt 时文本特征本来就可以缓存，对我们的延迟没有帮助，不测 |
| YOLOE-26（Ultralytics，YOLOE 为 ICCV 2025） | 实时开放词表检测 + 实例分割 | 是（`set_classes`，MobileCLIP2 文本编码器，prompt 嵌入可预算） | Ultralytics assets `v8.4.0`：`yoloe-26{n,s,m,l,x}-seg.pt`，2026-04-16；x 172 MB | 文本 prompt 一次前向覆盖全部类别，延迟与 prompt 数无关 |
| YOLO26-seg（Ultralytics，arXiv 2606.03748） | COCO 闭集检测 + 实例分割 | 否（80 类固定） | 同上 `yolo26{n..x}-seg.pt`，2026-01-13；x 142 MB | 论文报 COCO 40.9–57.5 mAP，T4 TensorRT 1.7–11.8 ms；COCO 没有 cone / debris / 特种车 |
| Grounding DINO tiny | 开放词表检测（只出框） | 是 | HF `IDEA-Research/grounding-dino-tiny`（Apache-2.0） | 只出框，接地点只能用框底中点；公开报告一般在 50–100 ms 量级，排第四优先 |
| YOLO-World v2 | 开放词表检测 | 是 | Ultralytics / AILab-CVC | 已被同一路线的 YOLOE 取代，不测 |
| MobileSAM、EfficientSAM、EdgeSAM、SAM 2 tiny / small、EdgeTAM | 可提示分割 | **否**，要点 / 框 prompt | 都在 HF / GitHub | 需要上游检测器给框，本身解决不了「这帧里有哪些行人」；不测 |
| FastSAM | 无 prompt 全图分割 + CLIP 文本筛选 | 间接 | GitHub | 文本是后接 CLIP 打分，行人这种小目标上不可靠；不测 |
| nuImages 上训练的 2D 检测器（mmdet3d 的 Mask R-CNN 等） | 闭集 | 否 | mmdet3d model zoo | 与 nuScenes 同城同相机，nuScenes 上的召回会偏乐观；不测 |
| YOLO26-depth（Ultralytics，2026-07-16） | 单目度量深度 | — | `yolo26{n..x}-depth.pt` | **不是分割器**，但它直接针对 BEV 放置：接地点沿射线用预测深度定位，不再依赖平地；见下「可选 D」 |

来源：[facebookresearch/sam3 RELEASE_SAM3p1.md](https://github.com/facebookresearch/sam3/blob/main/RELEASE_SAM3p1.md)、
[EfficientSAM3 GitHub](https://github.com/SimonZeng7108/efficientsam3) 与 [HF 仓库](https://huggingface.co/Simon7108528/EfficientSAM3)、
[Ultralytics YOLOE 文档](https://docs.ultralytics.com/models/yoloe)、[YOLO26 论文 arXiv 2606.03748](https://arxiv.org/abs/2606.03748)、
[Ultralytics depth 任务文档](https://docs.ultralytics.com/tasks/depth)、[ultralytics/assets releases](https://github.com/ultralytics/assets/releases)。

## 候选（按优先级）

- **A. SAM 3.1，同一个模型的提速**（`envs/sam3` 只读使用，不改）：
  A0 原样路径（Q4d 的复测，作为同一天的基线）；
  A1 `batched`：6 个 prompt 一次 grounding，3 路相机一个 batch（`sam_detect.Detector(mode="batched")`，Q4 日志 (10) 里它与原样路径的差是 mask IoU ≥ 0.988、分数差 ≤ 4e-3）；
  A2 只留 pedestrian + vehicle 两个 prompt（batched）；A3 只留 pedestrian；
  A4 = A1 + 官方 `compile_mode`（backbone 编译）；A5 = A1 + 输入分辨率 672（`Sam3Processor(resolution=…)` 这个官方参数；ViT 窗口 24 × patch 14 = 336 的整数倍才合法，若模型在 672 上跑不起来就记下原因、不改模型）；
  A6 fp8（torchao 的 float8 动态量化，只作用于 Linear）：只有在 torchao 与 torch 2.13 能直接装上时才测，不另建一套量化流程。
- **B. EfficientSAM3 Stage 3 全模型**：EV-M、RV-M、TV-M 三个里先量延迟，取召回测试的是 TV-M（作者 quick start 的默认）与最快的一个；新建 `envs/efficientsam3`，仓库原样（作者 fork 的 `sam3` 包，不能装进 `envs/sam3`）。
- **C. Ultralytics**（新建 `envs/ultralytics`）：C1 YOLOE-26x-seg，文本 prompt = 同样 6 个词；C2 YOLO26x-seg，COCO 闭集；各跑 `imgsz` 640（默认）与 1280 两档，
  小尺寸（l / m）只量延迟。映射（C2）：person → pedestrian；bicycle → cyclist；car、motorcycle、bus、truck → vehicle；COCO 没有 cone / debris / emergency vehicle，这三类记「不适用」。
- **D. Grounding DINO tiny**（HF transformers，新建 `envs/gdino`）：时间允许才做；只出框。
- **可选 D-depth（只描述，时间允许才做）**：YOLO26x-depth 在同一批图上出度量深度，接地点按「射线上深度 = 预测深度」定位，替代平地；只对胜出的检测器的接地点做，
  报 P5 hazard 行人与 nuScenes 行人的召回，回答「BEV 放置能不能靠一个快的深度头修」。不进判据。

每个第三方模型按作者发布的方式跑：作者的预处理、默认阈值、默认 NMS；我们只写外面的循环、计时、存盘。

## 帧、GT 与投影（全部复用 Q4，不改一行评测代码）

- **帧**：融合诊断 Q4 的两个列表的固定子集 S（写在看任何候选输出之前）：
  - P5：所有「至少一个 hazard actor 在前视里自己可见（像素 ≥ 20）的 x⁺ 帧」× 3 路 = 3 306 帧 × 3 = **9 918 张**。主读数 (i)（hazard 行人 / 车辆，前视）在 S 上与全量完全相同（它本来就只看这些帧）。
  - nuScenes val：150 个 scene 按名字排序取第 0、3、6… 个（50 个 scene）× 全部 keyframe × 前三路 = **6 024 张**。
  - SAM 3.1 在 S 上的基线直接取 Q4 已有的检测（`processed/fusion_diag/sam/{p5,nusc}`）按 key 过滤，不重跑。
  - 快的候选（全量 < 20 min）另在全量 47 814 张上跑一次，作为 side 读数，确认 S 上的差与全量一致。
- **GT、匹配、抬升**：`jevdrive.fusion_q4` 原样（`p5_gt.parquet`、`nusc_gt.parquet`、`lift_dets`、`match`、`evaluate`、`hazard_reading`），匹配门 max(2 m, 0.1 d)，参考点 = GT footprint 离相机最近的点，平地抬升为主读数、oracle 高度为分解读数。
- **接地点**：出 mask 的候选用同一个 `sam_detect.contact`（mask 最低 3 行的平均列）；只出框的（D）用框底中点。为了隔离「mask 接地 vs 框接地」本身的差，SAM 3.1 的已有检测另报一次「框底中点」版本（side）。
- **类别与阈值**：文本类候选用同样的 6 个 prompt；阈值用各自发布的默认（SAM 系 0.5，Ultralytics `conf` 0.25，Grounding DINO box 0.35 / text 0.25）作主读数。存盘阈值放低（SAM 系 0.3，其余 0.05），
  另报 side 读数「同 precision 下的召回」：把阈值调到 nuScenes 行人 precision 等于 SAM 3.1 的 0.36（S 上重算的值）时的行人召回，排除「阈值松紧」造成的假差异。

## 指标

- **延迟**（Q4d 的口径）：GPU 0 独占，batch 1，列表 `lists/p5_prof200.parquet`（Q4d 同一个），20 次 warm-up 后计 n ≥ 150（1 路）/ ≥ 50（3 路）；
  「1 路」= 每次一张图，「3 路」= 每次 3 张图一次调用（模型支持 batch 就 batch）；计时从 host 上已解码的 uint8 图（SAM 系为 pinned 的 CHW tensor，Ultralytics 为 HWC numpy，这是它们各自的输入形式）
  到 GPU 上拿到每个实例的分数、框、原图分辨率的 mask（`torch.cuda.synchronize` 之后）。包括上传、预处理、模型、后处理（NMS、mask 上采样）；不含 JPEG 解码与接地点提取。报 p50 / p95、峰值显存（`max_memory_allocated`）。
  compile 的候选先编译、再 warm-up。
- **召回**：P5 主读数 (i) hazard 行人（全部、≤ 20 m、≤ 30 m 白天）与 hazard 车辆（全部）；P5 (ii) 背景行人 / 车辆按距离档（0–10、10–20、20–40 m）；nuScenes 行人 / 车辆 ≤ 40 m（visibility token ≥ 3）按距离档；
  每项同时报平地召回与 oracle 高度召回；nuScenes precision；≤ 20 m 的 hazard 行人 BEV 误差中位。

## 判据（写死）

- **延迟过关**（主）：3 路 p95 ≤ **50 ms**，即一拍 20 Hz 之内出完三路状态。另报严格档：3 路 p50 ≤ **30 ms**，这是 Q8 的规则
  （窗口 p25 < 延迟 + 0.5 s 则必须留在快通道）下 ParkingCrossingPedestrian（p25 0.53 s）不再被判「来不及」的上限。
- **召回不劣**：在子集 S 上、与 SAM 3.1（S 上重算）比，下面三项的平地召回都不低于 SAM 3.1 − 0.03：P5 hazard 行人（全部）、P5 hazard 行人（≤ 20 m）、nuScenes 行人（≤ 40 m）；
  且 ≤ 20 m hazard 行人 BEV 误差中位 ≤ 1.0 m（Q4 的登记门槛）。
- **候选通过** = 延迟过关（主）且召回不劣。通过的候选里取 3 路 p95 最低者为推荐；没有候选通过时，报延迟过关的里召回最好的、以及召回不劣的里延迟最低的，两者都写出差距。
- Q4 原来的绝对门槛（hazard 行人 ≤ 30 m 白天 ≥ 0.9、全部 ≥ 0.8；nuScenes 行人 ≤ 30 m ≥ 0.8）照报，预计平地抬升下没有候选能过；过不过都不影响本文件的判定。
- 召回的 CI：P5 hazard 行人按 base 路线 bootstrap（`fusion_q4.boot_recall`，2000 次），对 SAM 3.1 的配对差也按路线 bootstrap。

## 资源估计

| 步骤 | GPU·h | 显存 | CPU 核 | 墙钟 | 盘 |
|:--|--:|:--|--:|:--|:--|
| 三个 env + 权重（EfficientSAM3 ~1.5 GB、Ultralytics ~0.6 GB、GDINO 0.7 GB、MobileCLIP ~0.3 GB） | 0 | — | 4 | 30–60 min | ~15 GB |
| 延迟：A0–A6、B×3、C 各档、D | 0.3 | ≤ 20 GB | 4 | 20–30 min | — |
| 召回：S 上 A1、A2、A5、B×2、C1×2、C2×2（~16k 张 × 9 个配置） | 0.8–1.5 | ≤ 30 GB | 12（解码） | 1–1.5 h | < 2 GB |
| 评测（CPU，`fusion_q4` 原样） | 0 | — | 12 | 每个配置 ~2 min | — |

超过估计 2 倍就停下来重估。下载与 WOD test 共用链路，只下上面列的文件，单文件都 < 0.5 GB（GDINO 0.7 GB）。

## 偏离与澄清日志

（执行时追加。）

## 结果

（跑完再填。）
