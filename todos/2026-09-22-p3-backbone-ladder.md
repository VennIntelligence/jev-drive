# P3：backbone 阶梯（换表征，同一个子集、同一套 head、同一个 judge）

状态: done
主题: ../research/survey-counterfactual-video-gen.md

## 目标

第 20 条判到**分支 2**：线性读出这批冻结 Qwen3-VL-4B 特征，在 pre_onset 上已经榨干，
下一步要动表征。P3 就是那一步的开环版本：**换 backbone，别的一律不动**——
同一个分层子集、同一套 head 家族、同一个 judge、同一个 −0.05 m 门槛。

**P3 回答两个问题**：
1. 换表征能不能把 pre-onset 买回来（门槛见下），或者改变 s_ego decile 曲线的形状；
2. **H3 这个 DiT 在 Qwen3-VL-32B 之外还加不加东西**。这一问是 P3 独有的：
   第 21 条记的文献事实是 H3 的理解侧**就是 Qwen3-VL-32B 的第 50 层**，
   所以 (a) 和 (b) 并排跑，两者之差就是「生成侧的 DiT 本身值多少」，
   而这正是 `survey-counterfactual-video-gen.md` 第 4 节里空着的那一格。

## Arms

| arm | 模型 | 许可 | 取什么 | 在盘上了吗 |
|:--|:--|:--|:--|:--|
| **a** | Qwen3-VL-32B-Instruct | Apache-2.0 | 第 50 层（H3 的 conditioner 层）和第 32 层（我们 4B 设定 L18/36 的中层类比）的 mean / last | **是**，76 GB 已在 HF cache |
| **b** | MiniMax H3，fl2va DiT | Community License（排除美/欧/英/韩自托管） | 约 1/2 和 2/3 深度两个 block 的 token mean | 否，要过代理下 |
| **c** | Wan2.2-TI2V-5B | Apache-2.0 | 同上配方，便宜的生成式对照 | 否 |
| **d** | V-JEPA 2 ViT-L | — | 4 帧 clip 到当前帧，pooled | **是**，1.3 GB 已在 cache |

(a) 是判别式的 scaling 对照（同一族、同一套预处理，只是大 8 倍）；
(b)(c) 是生成式；(d) 是第 12 条钉下来的视频自监督对照。

### 每个 arm 的配方

**(a) Qwen3-VL-32B**：**和 `qwen_front3` 同一条预处理路径**——三个相机、同样的
`smart_resize` 到 960×1088、同样的 chat template、batch 钉死 4
（`todos/2026-09-21-waymo-train-features.md` 证明过 batch size 是特征定义的一部分）。
64 层，取第 32 层和第 50 层，各存 `_mean` 和 `_last`，hidden 5120。

**(b) MiniMax H3**（按 `research/lit/2026-09-22-round4-h3-deployment.md` 第 6.1 节）：
- 只要 `transformer/` 的 fl2va 分区，**pruned bf16 约 40 GB**（不含可预计算的 AdaLN 分支）常驻；
  32B 不同时在显存里就够，紧张就换 `pruned_fp8_scaled`（21 GB）
- text conditioner 对一个固定的中性 prompt **预先编码一次存盘**，之后复用
- VAE 编码当前帧。**三个相机是拼成一张 16:9 画布还是一个一个过，跑之前定下来并写进本节**
- 按 DriveLaW 的 t=1 / 2502.07001 的 timestep ≈ 200/1000 加**高噪声**，**只走一次 forward**；
  绕开 diffusers 那套把 `num_frames` 卡到 17n+5、一次 5–15 s 的 blocks，直接调 transformer
- 在约 1/2 和 2/3 深度 hook，token mean-pool 成向量；便宜的话扫两个噪声水平

#### H3 的工程量：比 lit 估的小，diffusers 0.40 已经把整条路铺好了（2026-09-22 查证）

lit 6.1 说「绕开 blocks 自己拼 `denoiser_input_fields` 是本方案唯一真正的工程量」。
实际查 `diffusers==0.40.0` 的源码，**接口是完全公开且带文档的**，不需要猜：

- `MiniMaxH3Transformer3DModel.__init__` 的默认值就是官方配置：**50 层、hidden 5376、
  `text_dim = 5120`**。`text_dim` 正好是 Qwen3-VL-32B 的 hidden size，
  **这是第 21 条那条「理解侧就是 Qwen3-VL-32B 第 50 层」的代码级佐证**。tap 取第 25 和第 33 层。
- `forward` 要 `hidden_states, audio_hidden_states, encoder_hidden_states, timestep, timestep_indices,
  token_tags, position_ids, video_indices, audio_indices, text_indices`。
- 这些索引不用自己推导：
  `diffusers.modular_pipelines.minimax_h3.before_denoise` 里的
  **`build_packed_sequence` 是一个 `@staticmethod`**，签名是
  `(text_token_tags, num_latent_frames, latent_height, latent_width, num_audio_latents,
  patch_size, audio_channels, audio_tag, video_tag, keyframe_anchors)`，
  直接返回 `position_ids, token_tags, video_indices, audio_indices, text_indices, ...`。
  **单帧、无音频、无 keyframe 条件**就是
  `num_latent_frames=1, num_audio_latents=0, keyframe_anchors=()`——
  `17n+5` 那条限制在 blocks 里，不在 transformer 里，这样绕开就没有了。

所以 (b) 的风险**不在工程，在下载**：71 GB。

#### (c) 跑起来之前卡了两次，两次都不是模型的问题（2026-09-23）

**第一次 00:46**：`from_pretrained` 去 HEAD `huggingface.co` 要一个**盘上已经有**的 VAE index，
而那个窗口没有 proxy，于是 `[Errno 101] Network is unreachable` 重试五次、gate 退出 1、**队列停了 55 min**。
`jevdrive/features.py` 一直有 `os.environ.setdefault("HF_HUB_OFFLINE", "1")`，`dit_features.py` 没有。

**第二次 01:41**：加了 `HF_HUB_OFFLINE=1` 和 `local_files_only=True` 之后仍然失败，
这次是 **`AutoTokenizer`**：离线状态下 `repo + subfolder=` 还是走 hub resolver，
而 resolver 会去探 `added_tokens.json` 这类**从来没下载过、因而 cache 里也没有 `.no_exist` 标记**的可选文件；
它没法在不联网的情况下证明这些文件不存在，就报
`couldn't connect ... couldn't find them in the cached files`。
**text_encoder 的 config 用同样的方式却加载正常**，所以第一眼看上去像「文件缺失」，其实是「查找方式不对」。

**修法**：`snapshot_dir(repo)` 用 `snapshot_download(repo, local_files_only=True)` 解析出快照目录，
**四个 `from_pretrained` 全部改成传本地目录路径**（`f"{snap}/tokenizer"` 等），去掉 `subfolder=`，
整个 resolver 就绕过去了。另外 gate 脚本**同时**导出 `HF_HUB_OFFLINE=1` **并** source proxy——
两个都要：前者让不必要的探测不发生，后者让真正需要下载的东西能下得到。

**还真缺了一个文件**：最早那次 Wan 下载的 `allow_patterns` 是
`transformer/* vae/* *.json tokenizer/*`，**不含 text_encoder 的权重**。
`text_encoder/` 底下只有 config 和 index，三个 `model-0000{1,2,3}-of-00003.safetensors`（11.4 GB）
是真的没有，`prompt_cache` 要它们来编码那个固定的中性 prompt。已补下，三个 shard 齐了。

**实测（200 帧 probe，2026-09-23 02:43）**：**743.7 ms/帧、峰值显存 24.32 GB、72.0 KB/帧**，
30 个 block 里 tap 第 15 和第 20 层，σ ∈ {0.2, 0.8}，latent `z_dim` 48，
prompt state 形状 (1, 512, 4096) 已缓存复用。
每帧是 **3 相机 × 2 个噪声水平 = 6 次 transformer forward + 3 次 VAE encode**，
折合约 **124 ms/forward**，比 lit 对 H3 的推算（30–100 ms）高一些。
**全量 20 237 帧约 4.2 h、约 1.5 GB**；这个数是在 d''' 的 ViT-g 抽取同时在跑时测的，所以是上界。

#### (c) 改成**只跑 front 一路相机**，三相机版延后（2026-09-23，用户决定）

**理由是可比性，不只是省时间**：真正带信号的 V-JEPA arm——(d)——**本来就只看 front 一路**。
如果 Wan 用三相机跑，它和 (d) 就同时差两件事（表征 + 看到多少场景），
**那个差值就不能归因给「生成式 vs JEPA」了**。front-only 让这一格是 like-for-like 的。

顺带代价也降到三分之一：每帧从 **6 次 transformer forward + 3 次 VAE encode** 降到 **2 + 1**，
按 probe 实测的 743.7 ms/帧折算，front-only 估 **约 250 ms/帧、全量约 1.4 h**（而不是 4.2 h）。

**三相机那版不是取消，是延后**：只有在 front-only 在 pre-onset 上**显示出东西**时才值得补跑。
如果 front-only 也是空的，再花 3 倍的时间去跑三相机不会改变结论。

#### 队列顺序 2026-09-23 02:45 调整：**d'' 排到 Wan 前面**

d'' 回答的是「起作用的因子是时间还是 JEPA」，这一问决定下一步做什么；
Wan 提供的是生成式那一列的一个数据点。**前者比后者更值钱，所以先跑。**

**(c) Wan2.2-TI2V-5B**：同 (b) 的配方，作为便宜的生成式对照（DriveWAM 的基座，24 GB 可跑）。
实测配置（写死在 `jevdrive/dit_features.py`）：`WanTransformer3DModel` **30 层、hidden 3072、
`patch_size=(1,2,2)`、VAE `z_dim=48`**，tap 取第 15 和第 20 层；
三个相机**一个一个过**（拼成 16:9 会把 3.3:1 的长条横向压掉近一半），
三个 pooled 向量拼成一行；噪声**两档都抽**（σ=0.2 和 0.8），
因为 DriveLaW 的「t=1」和 2502.07001 的「timestep≈200/1000」在 flow matching 的参数化下
指的是**相反的两端**，与其挑一个读法不如两个都测。

**(d) V-JEPA 2**：第一次跑**崩了，原因值得单独记**：clip reader 假定「一个 sequence 待在一个 shard 里」，
于是拿目标帧那个文件去读历史帧的字节区间，读出来的根本不是 JPEG。
**实际上 Waymo 的一个 sequence 会横跨多个 shard**——同一条 sequence 的第 29 帧在 shard 58、
第 31 帧在 shard 91。修法是每个 slot 自带自己的路径，`Shards` 按需开多个 fd。
子集里 **19 663 / 20 237** 帧有严格完整的 4 帧窗口。

`jevdrive/features.py` 里已有 `VJepaFeatures`，但它是 64 帧的 `fpc64` 权重。
这里喂 **4 帧 clip**（tubelet 是 2 帧，4 帧 = 2 个 tubelet），
**这是一个已知的 distribution shift，读结果时必须带上**；另外它只吃 FRONT 一路相机，
而主线是 front3，这两个变量都要写进表里（第 12 条的原话：这一行不能读成「谁是更好的 backbone」）。

## 共用协议（和 P2 完全相同，见 [p2 的 todo](2026-09-22-p2-readout-ladder.md)）

同一个 `p2p3_v1.parquet` 分层子集（约 2 万帧：pre_onset / rater / turn_yaw 全要，
其余按 sequence 分层抽样），半/半 sequence 切分与 L0 逐字相同，
`ridge_late` over ego + feature，λ 由 fit 半内 4 折 grouped CV 选，两个方向，paired scene bootstrap。
**arm A（Qwen3-VL-4B pooled）在同一子集上重算**，所以每一个对比都是 like-for-like。
每个 arm 的逐帧预测都存盘，P1 那边定下来的 judge 以后可以直接套用，不用重新拟合。

## 时间与存储预算（跑之前估；每个 arm 先用约 200 帧实测再改这张表）

| arm | ms/帧（估） | ms/帧（实测，200 帧） | 2 万帧墙钟 | 特征体积 | 依据 |
|:--|--:|--:|--:|--:|:--|
| a Qwen3-VL-32B | 约 1000 | **621.2**（峰值显存 63.8 GB，3060 token/帧，batch 4，权重用 `device_map="cuda"` 直接流进卡里） | **3.5 h** | 53 504 B/帧 → **1.08 GB**（`L32/L50` 的 `_mean`/`_last` 各 5120 维，外加 `vis_mean`/`vit_mean`） | 估计偏保守一倍：只跑到第 50 层是真正的 early exit，省掉 64 层里的 14 层 |
| b H3 DiT | 30–100 / forward | 20–60 min（每个噪声水平） | 2 层 × d × 2 B，约 **0.2 GB** | h3-deployment 6.1 的推算，**未实测** |
| c Wan2.2-5B | 同量级 | 20–60 min | 约 0.2 GB | — |
| d V-JEPA 2 ViT-L | 约 25 | **8.1 实测** | **2.7 min**（19 663 帧，峰值显存 0.75 GB） | 4.0 KB/帧 → **80 MB** | ViT-L、4 帧、256²，比 Qwen 的 3060 token 小一个量级；比估计还快 3 倍 |
| 下载 | — | 见下 | — | — |

### 下载这一项估错了，就地更正（2026-09-22 18:20）

三件事和估计不一样：

1. **(a) 不用下**。`Qwen/Qwen3-VL-32B-Instruct` 已经在 box 的 HF cache 里，76 GB、14 个 shard 齐全。
   P3 最贵的那个 arm 因此没有下载成本。
2. **链路只有 3–5 MB/s，不是 10**。turbo（`source /etc/network_turbo`）对 Wan 的 CDN 一直 SSL handshake
   超时，hf-mirror.com 只有 0.1 MB/s，最后走 Clash（`proxy_on`）拿到 3 MB/s 独占时 5 MB/s。
   **`tmux_run.sh` 跑的脚本是非交互 bash，不读 `~/.bashrc`，所以 `proxy_on` 是 command not found** ——
   脚本里必须自己 `source ~/.bashrc`。
3. **H3 没有 pruned 版可下**。官方 repo 里 `FL2VA/transformer` 是 **61.7 GB**（13 个 shard），
   `FL2VA/video_vae` 9.7 GB，合计约 **71 GB**；lit 里说的「pruned bf16 40.2 GB」只在
   `Comfy-Org/MiniMax-H3` 有（`minimax_h3_fl2va_pruned_bf16.safetensors` 37.5 GB），
   但那是 ComfyUI 的单文件排布，要自己做 key 映射才能进 diffusers 的 `MiniMaxH3Transformer3DModel`。
   **H3 的 text encoder 不用下**：它就是 Qwen3-VL-32B，我们已经有了，而固定中性 prompt 的 conditioner
   state 正好可以用本地这份 32B 预先编码一次。

于是下载预算变成：**Wan 约 23 GB（transformer + vae），H3 约 71 GB**，按 5 MB/s 串行是 **1.3 h + 4 h**。
两个一起下会平分链路（docs/network-proxy.md：链路是零和的），所以按顺序下，Wan 先——
它和 H3 共用同一套抽特征配方，先把便宜的那条调通再上贵的。

**只有 (a) 超过 3 h**，所以只有它按 CLAUDE.md 必须先做 profiling pass；
其余每个 arm 也一律先测 200 帧，把实测写回这张表再开全量。
**任何时候只排一个抽取，不并排两个**，GPU 不空转也不打架。

**失败就记录、不阻塞**：一个模型下不下来、跑不起来，**最多调两小时**，
然后把「具体哪一步失败、报什么错」写进结果一节，换下一个 arm。

## (b) 延后（2026-09-22），已关闭（2026-09-23）

**2026-09-23 用户决定关闭 (b)**：重开条件「Wan 测出信号」没有触发（(c) Wan 处处为零），不再做。下面保留延后时的记录。

**(b) MiniMax H3：**延后，下载已停、半截文件已删（用户 2026-09-22 决定）**。**

理由是 (a) 的结果直接推出来的，不是嫌它贵：**H3 的理解侧就是 Qwen3-VL-32B 的第 50 层，
而 (a) 已经把那一层在 pre-onset 上测成了零**（四个 tap 八个数字全在 −0.036 到 +0.006 之间）。
所以 H3 唯一还能提供的是**「生成式 DiT 值多少」这一个数据点**，
而 **Wan2.2-TI2V-5B 已经在盘上**，用同一套配方回答的是同一个问题，代价小一个数量级
（23 GB 对 71 GB，5B 对 33B）。先用 Wan 把生成式那一列填上。

**已做的清理**：`p3-dl-h3` 窗口关掉，`$DATA_DIR/models/MiniMax-H3`（2.3 GB，直连 curl 那份）
和 `$HF_HOME/hub/models--MiniMaxAI--MiniMax-H3`（1.4 GB，早先 `snapshot_download` 那份）都删了，
**不留半截下载在盘上**，共腾出 3.7 GB。

**如果 Wan 测出信号，再回来做 H3 时走这条路**：不要下 71 GB 的全权重，下
`Comfy-Org/MiniMax-H3` 的 **`minimax_h3_fl2va_pruned_fp8_scaled.safetensors`（19.52 GB）
加 `vae/minimax_h3_video_vae_fp16.safetensors`（4.85 GB），合计约 26 GB**。
代价是它是 **ComfyUI 的单文件排布**，要自己写 key 映射才能进 diffusers 的
`MiniMaxH3Transformer3DModel`；接口那一侧已经查清楚（见本 todo 上面「H3 的工程量」一段），
缺的只是这个 loader。按实测 3 MB/s，26 GB 约 2.4 h。

## 预写的读表（和 P2 共用，门槛来自第 20 条，一字不改）

**一个 arm「买回了 pre-onset」，当且仅当它的 pre-onset ΔADE ≤ −0.05 m
且 CI 不跨零，在两个方向上都成立。** [−0.05, +0.05] 之内一律读成没有实用增益。

| 结果 | 判定 |
|:--|:--|
| 某个 arm 达标 | 表征层面**确实有救**。按 5b 的预登记读法，这是 **confirm**，值得往更大的基座投 |
| 全集 ADE 改善但 pre_onset 仍在 [−0.05, +0.05]，decile 曲线还是同一条倒 U | **kill**：它买到的和 Qwen3-VL-4B 买到的是同一批中等偏离帧，换基座是换汤不换药。**这个负面结果本身可发表** |
| 全集 ADE 变差而 pre_onset 也没改善 | kill，并且顺带证实 continuation bias 的预测 |
| (b) 达标而 (a) 不达标 | **生成侧的 DiT 确实加了东西**——这正是第 4 节空着的那一格，是 P3 最有价值的一种结局 |
| (a) 和 (b) 同样好 | H3 的增量全在它的理解侧（= Qwen3-VL-32B），**DiT 不加东西**，第 21 条「新东西只在 33B DiT 里」要就地改掉 |

**decile 曲线按形状读**：相对增益在第 4–9 档还升不升、见顶是不是还在第 8 档。
**第 10 档不参与判定**（第 20 条已用 rater 证明那一档的 logged future 本身就有争议）。

**零和检查照第 20 条做**：pre-onset 上赚的必须和 straight_yaw 上亏的并排报。

## P3(d') V-JEPA 2 阶梯（2026-09-22 追加，用户批准；排在 (a) 的抽取和 head 之后、Wan/H3 之前）

### 为什么追加

(d) 那一行给出了全项目第一个**形状**变化（第 10 档的掉落被压平、见顶右移），
但**两个方向不一致**：方向 1 的 pre-onset Δ 是 −0.115，方向 0 是 −0.018，
而且方向 0 的 CI [−0.059, +0.023] **排除了** −0.115，所以这不是「一个测得动一个测不动」，
是两边真的不一样。(d) 同时还背着两个偏离主线的变量：**只看一路相机**、**只喂 4 帧**。
d' 就是把这些变量一个一个拆开，**回答两个问题**：

1. **方向不一致会不会随着「多给相机 / 多给帧 / 换大模型」收敛？**
2. **decile 的形状变化（第 10 档不再掉回去）在这些变体上还成不成立？**

### Arms（子集、半/半切分、head 家族、judge、两个方向、逐帧预测存盘，全部同第 23/24 条）

| arm | 变的是什么 | 配置 |
|:--|:--|:--|
| **d** | —（基线，已测） | ViT-L、FRONT 一路、4 帧、stride 2（跨 0.6 s） |
| **d'1 三相机** | 相机数 1 → 3 | 三路各自抽一遍，**把三个 pooled 向量拼接**（3 × 1024 = 3072 维），不做平均 |
| **d'2 clip 8** | 帧数 4 → 8 | 同 stride 2，跨 1.4 s |
| **d'3 clip 16** | 帧数 4 → 16 | 同 stride 2，跨 3.0 s |
| **d'4 ViT-g** | 模型 ViT-L → ViT-g | `facebook/vjepa2-vitg-fpc64-256`，**同样是 256 分辨率**，所以只换了模型大小 |
| **d'5 late fusion** | 表征数 1 → 2 | `ridge_late` over ego + [V-JEPA 2 ‖ Qwen3-VL-4B `L18_mean`]，测两者互不互补 |
| ~~d'6 V-JEPA 2.1~~ | — | **不做，理由见下** |

**三相机为什么拼接而不平均**：平均会把左右两路糊在一起，而 pre-onset 的整个问题就是
「往哪边转」——左右是这件事的信号本身，不是要被平均掉的噪声。拼接保留它，代价只是维度从 1024 到 3072，
对 ridge 可以忽略。

**帧数变体要重新核对跨 shard 那个修法**：`clip_items` 原来假定「一个 sequence 待在一个 shard 里」，
已经证伪并修好（每个 slot 自带路径）。16 帧 × stride 2 要往回取 30 帧，跨 shard 的概率更高，
所以每个变体都要打出**严格完整窗口的帧数**，并把整张表限制在同一批帧上（第 13 条）。

**(4) V-JEPA 2.1 不做，这是合规判断不是技术判断**：HF 上 `facebook` 组织**没有** V-JEPA 2.1 的官方 repo
（`list_models(author="facebook", search="jepa")` 只有 `3d-jepa` 和 `jepa-wms`），
能找到的全是第三方转存（`Dev-Jahn/vjepa2.1-vitl-fpc64-384`、`apiantonio/vjepa2.1-vit-large-384` 等）。
**第 12 条已经就 DINOv3 定过这条规矩**：「绕过许可在合规上站不住，论文里也没法写来源」，
对第三方转存一视同仁。附带的技术理由：它们的 model type 是 `vjepa21`，
不是当前 transformers 里的 `vjepa2`，还要额外对齐版本。**一小时的预算花在这上面不划算，记录并跳过。**

### 成本

抽特征 **8.1 ms/帧**（(d) 实测），所以 (d'1) 三路各一遍、(d'2)(d'3) 各一遍，
合计约 5 × 2.7 min ≈ **15 min GPU**；head 是 ridge，分钟级。**(d'5) 不需要任何新抽取。**
**唯一贵的是 (d'4)**：`vjepa2-vitg-fpc64-256` 是 **19.19 GB**，按实测 3–5 MB/s 要 **1.1–1.8 h 下载**，
所以它**排在 Wan 下载完之后**，抽取本身仍是分钟级。
**任何时候只排一个抽取。**

### 预写的读法（跑之前写死；门槛和第 20 条一字不改）

**口径按第 22 条（2026-09-22 用户采纳）**：主判是
**「对着 log 算 ADE，限制在 s_ego 第 1–9 档」**，DiD 在同一限制上算；
**第 10 档单独一行，只报 RFS 和「对 rater_best 的 ADE」**；**RFS 永远并排**；
全集 ADE 降级成侧栏。引用时必须带上**「s_ego 分档是循环的」**这句限定。
门槛和形状判据**一字不改**，只是判据一里的 ΔADE 换成第 1–9 档上的那个。

**判据一（买不买得回 pre-onset）**：**pre-onset ΔADE（第 1–9 档）≤ −0.05 m 且 CI 不跨零，两个方向都成立。**
每个 arm 都要按方向分开报 Δ 和 CI，和 (d) 并排。

**判据二（方向是否收敛）**——(d) 的核心遗留问题，跑之前定死怎么算「收敛」：

| 结果 | 判定 |
|:--|:--|
| 某个变体两个方向都过 −0.05 门槛 | **方向不一致是「变量不够」造成的**，(d) 的 −0.115 是真的，按判据一算 confirm |
| 两个方向的 Δ 仍然**互相排除**（一个方向的 CI 不含另一个方向的点估计） | 方向不一致**不随这些变量收敛**，那它多半是半 val 切分本身的性质（239 个 sequence 一半），要等 train split，不要在这批数据上再加变体 |
| 两个方向的 CI 开始互相包含，但都没过门槛 | 不一致收敛了，效应本身不够大。诚实读成「V-JEPA 2 买到的是一个小而均匀的增量」 |

**判据三（形状）**——把「形状变了」定义成一个数，不靠肉眼：

> **shape change 成立 ⟺ 第 10 档的相对增益与第 4–10 档里最好那一档的相对增益相差 ≤ 2 个百分点。**

也就是问「顶档还掉不掉回去」。（decile 曲线本身仍按原样画：它描述形状，不是判「买没买回来」的那个量。）
按这个定义回头算已有的数（方向 0 / 方向 1）：
arm A pooled 是 −5.8 对 −1.9（差 **3.9**）和 −6.5 对 −2.2（差 **4.3**）→ **不成立**；
(d) V-JEPA 2 是 −7.1 对 −5.0（差 **2.1**）和 −8.4 对 −6.7（差 **1.7**）→ **方向 1 成立、方向 0 差一点**。
所以这个判据在现有数据上是能分辨的，不是事后挑的。
**第 10 档参与本判据，但不参与判据一**（第 20 条：那一档的 logged future 本身有争议，
所以它可以用来描述形状，不能用来判断「买没买回来」）。

**零和检查照旧**：pre-onset 上赚的必须和 straight_yaw 上亏的并排报。

### P3(d'') Qwen3-VL-4B 的**原生 video 输入**（2026-09-22 追加，用户批准；排在 Wan 和 H3 **之后**，是整条阶梯的最后一个 arm）

**它要拆的混淆**：(d) 里 V-JEPA 2 和所有单帧 arm 同时差两件事——
它**用 JEPA 目标训**，而且它**被喂了时间**。(d') 的三相机 / 更长 clip / 更大模型都只动 V-JEPA 这一侧，
拆不开这两个因子。**把同一段 clip 塞进 Qwen3-VL 自己的 video 通路，目标就被钉住了，只有输入在变。**

> **如果 Qwen-video 也把 pre-onset 挪动了，那么起作用的因子是「时间」，不是「JEPA」。**
> 如果 Qwen-video 不动而 V-JEPA 动，那就是 JEPA（或者 V-JEPA 的预训练数据），不是时间本身。
> 这一条和第 23 条的 (b) 形成三点一线：**时间拼在读出端（b）、时间长在同一个表征里（d''）、
> 时间长在另一个目标训出来的表征里（d）**。

**配方**：和 (d) 给 V-JEPA 2 的**完全同一段 clip**——4 帧、stride 2（跨 0.6 s）、oldest first，
**三个相机**（token 预算够，见下），front 在前；走 `videos=` 而不是 `images=`，
即 Qwen3-VL 的 temporal patching 和 video token type；取和 `qwen_front3` 同一层的
`L18_mean` / `L18_last`。除了「帧是以 video 还是以独立图像进去的」，其余一切不变。

**为什么 video 通路不是「把 4 张图拼一块」**：Qwen3-VL 的 ViT 把 `temporal_patch_size` 帧折进 patch embedding，
所以 T 帧的 clip 花 **T 倍的 patch**，但只产生 **T/2 倍的 LLM token**。
单帧三相机是 12 240 个 patch / 3 060 个 image token；**4 帧三相机是 48 960 个 patch / 6 120 个 video token**。

**成本估计（跑之前；200 帧 profiling 之后就地更正）**：单帧三相机跑到 L18 实测 **90.86 ms/帧**，
其中 ViT 那一段按 4B 的分解约占 45 ms、LLM 约占 36 ms、其余约 10 ms。
ViT ×4、LLM ×2 → 估 **约 260–360 ms/帧**，也就是用户说的「每相机约 4 倍」那个量级。
2 万帧估 **1.5–2 h**；特征只有 pooled，2 × 2560 × 2 B = 10 KB/帧 → **约 0.2 GB**。
**实测（200 帧 probe，2026-09-23 02:48）**：**552.6 ms/帧、峰值显存 10.52 GB、17 408 B/帧**
（`L18_mean`/`L18_last`/`vis_mean`/`vit_mean` 四个数组）。
估的是 260–360 ms，**实测高约 60%**；全量 20 237 帧约 **3.1 h**、**约 0.35 GB**。
注意这个数是在 d''' 的 ViT-g trainval 抽取同时占着 GPU 时测的，所以是上界。
200 帧里 **189 帧**有严格完整的 4 帧 × 3 相机窗口。

**可比性已经核过，不是假定的**：transformers 在 video 通路上抛了两个警告，
一个说它**不施加 qwen-vl-utils 那个 per-frame pixel cap**（所以某些视频的 token 会比参考实现多得多），
一个说**没有 `video_metadata` 就按 fps=24 重采样**。这两条都可能悄悄改掉我们喂进去的东西，
让 d'' 和 arm A 不再可比。**实测 `tokens_per_forward = 6120`，正好是 3 相机 × 2040
（= 2 个时间位 × 30 × 34），和这一节上面推的数字逐个对上**，
也就是说：分辨率仍然是 `smart_resize` 的 960×1088（我们喂进去的图本来就是那个尺寸，没触到 cap），
4 帧也没有被重采样成别的帧数。**所以 d'' 和 `qwen_front3` 看的是同一批像素，只是以 video 的身份进去。**

**读法（跑之前写死，和 (d') 同一套）**：第 22 条的口径
（pre-onset ΔADE 对着 log、限制在 s_ego 第 1–9 档；第 10 档单独用 RFS 和对 rater_best 的 ADE 报；
RFS 并排；全集 ADE 是侧栏；循环性限定随引用），**−0.05 m 且 CI 不跨零、两个方向都成立**的门槛不变，
形状判据（第 10 档相对增益与第 4–10 档最好那档相差 ≤ 2 个百分点）不变。追加一条二选一的归因：

| 结果 | 归因 |
|:--|:--|
| d'' 的 pre-onset Δ 和 (d) 同向同量级 | **因子是时间**。那么 V-JEPA 的 JEPA 目标不是关键，Stage B 应该直接在 Qwen 上喂时间，省掉换 backbone |
| d'' 基本不动而 (d) 动 | **因子不是时间**，是 JEPA 目标或 V-JEPA 的预训练数据。那么「给 Qwen 喂时间」这条路已经被 (b) 和 d'' 两次否掉，要换的是表征本身 |
| 两个都不动 | (d) 方向 1 的 −0.101 多半是半 val 切分的性质，按 (d') 判据二的第二行处理 |

## P3(d''') train split 上的单方向判决（2026-09-23 追加，用户批准；排在 Wan 和 d'' 之后）

### 为什么做这个而不是再加变体

(d') 已经把「方向不一致是不是变量不够造成的」这个问题答完了：
三相机、clip 8、clip 16、ViT-g **四个变量一个都没让它收敛**，
方向 0 的六个变体全在 [−0.031, −0.001]、方向 1 的全在 [−0.114, −0.061]，DiD 按方向变号。
按跑之前写死的判据二，这意味着**不一致是半 val 切分本身的性质**（239 个 sequence 对半分，
每个方向只有 500–750 个 pre-onset eval 帧）。
**唯一剩下的办法是把这个变量去掉**：在 train split 上训、完整 val 上评。
那只有**一个方向**，所以「方向不一致」在定义上就不存在了，而且拟合数据多一个数量级。

### 配置

- **协议用 `jevdrive/waymo_p0.py` 的那一套**（`load_all` 的 row set 和 join、`split_of` 的 fit/eval），
  这样数字和 P0 的并排放，不是放在另一个略有出入的 row set 上。
  实现是 `waymo_ladder.train_context` + `p3train`，`half` = 0 训练 / 1 评估，`directions=(0,)`。
- **s_ego 直接取 P0 那次 run 的**（`preds.npz` 里按 frame_name join），不重算，
  所以 decile 轴就是 P0 的轴。train 那一侧没有 P0 的 s_ego（P0 只评 val），那些行不参与分层。
- **只抽两个配置**：**(d) ViT-L、front、4 帧、stride 2** 和 **(d4) ViT-g、同样设置**。
  不再扫相机数和 clip 长度——(d') 已经证明它们不改变结论。
- **arm**：`ridge ego`（base）、A（Qwen4B pooled L18）、V-JEPA ViT-L、V-JEPA ViT-g、
  **late fusion V-JEPA + Qwen**。
- 抽取范围是 **train + 完整 val**（约 52 万帧有 future），8.1 ms/帧 → **每个配置约 70 min**，
  两个约 2.4 h；特征 4 KB/帧 → 每个约 2.1 GB。**跑之前打出严格完整窗口的帧数**，
  整张表限制在两个配置都有窗口的帧上（第 13 条）。

### 预写的读法（跑之前写死）

口径是第 22 条的（pre-onset ΔADE 对着 log、限制在 s_ego 第 1–9 档；第 10 档单独用 RFS 和
对 rater_best 的 ADE 报；RFS 并排；循环性限定随引用）。**门槛和形状判据一字不改**，
只是没有「两个方向」这一说了——**这是一个单方向的测量**：

| 结果 | 判定 |
|:--|:--|
| pre-onset Δ ≤ **−0.05 m 且 CI 不跨零**，**并且**形状判据成立（第 10 档相对增益与第 4–10 档最好那档相差 ≤ 2 个百分点） | **V-JEPA 的效应是真的**，(d) 方向 1 那个 −0.10 不是切分造成的。那么「把时间长进表征里」是有效的一格，Stage B 立项，并且要和 d''（Qwen 原生 video）并排读来定因子是时间还是 JEPA |
| pre-onset Δ ≤ −0.05 但形状判据不成立 | 效应是真的，但它买的不是「顶档不再掉回去」那件事。按第 22 条的口径老实写成「一个均匀的小增量」 |
| pre-onset Δ 落在 [−0.05, +0.05]，CI 半宽明显小于 0.05 | **(d) 方向 1 的 −0.10 是半 val 切分的 artefact**，V-JEPA 这条线到此为止。这是一个干净的负面结果，要写进第 24 条，不要再加变体 |
| pre-onset Δ 落在带内但 CI 仍然很宽 | 又一次测不动。那说明连 train split 都不够，问题在 pre-onset 只有 1 510 帧这个构造上（第 3b 条），要换的是评测设计不是模型 |

**无论哪一行，结果都写进第 24 条。** 第 10 档对 rater_best 的 ADE 那一列要一起报——
它是 (d') 里唯一两个方向都稳的量（V-JEPA 全族 −0.23 到 −0.48），
**train split 上它站不站得住，和主判据一样重要。**

## 步骤

现在的队列顺序：**(d'') Qwen 原生 video → (c) Wan（front-only）→ (d''') train split 判决**；
(b) 延后，d' 和 ViT-g 已完成。每个 stage 的起止和退出码追加到 `$DATA_DIR/runs/waymo_ladder/QUEUE_STATUS`，
一次读取就能看出哪个 gate 非零退出——这是 2026-09-23 两次「gate 退出 1 但没人看见」各损失约一小时之后加的。

- [x] (a) 200 帧 profiling：**621.2 ms/帧**，峰值显存 63.8 GB，52.2 KB/帧 → 全量 3.5 h
- [x] (a) 全量抽取（20 237 帧、633.0 ms/帧、3 h 33 min、1.08 GB）→ head：**四个 tap 全不过门槛，也不比 4B 好；scaling null**
- [~] (b) **延后**，下载已停、半截文件已删；理由和 fallback 见下面「(b) 延后」一段
- [ ] (c) 同上
- [x] (d) 抽取 + head —— **pre-onset 上方向 1 过门槛（−0.115）、方向 0 没过（−0.018），所以规则不触发；
      但 decile 曲线的形状第一次变了（第 10 档的掉落被压平、见顶右移）**，见 decisions 第 24 条
- [x] (d) 的数字已写进 decisions 第 24 条
- [x] (a) 的数字已补进第 24 条
- [x] (c) 的数字已补进第 24 条；(b) 延后
- [x] d' 阶梯：(d'1) 三相机、(d'2) clip 8、(d'3) clip 16、(d'5) late fusion —— 全部完成，全部不过门槛
- [x] d' 阶梯：(d'4) ViT-g —— 已完成；方向不一致未收敛，见 decisions 第 24 条
- [x] (d'') Qwen3-VL-4B 原生 video —— **唯一两个方向 CI 都不跨零的 arm**，2 h 11 min
- [x] (d''') train split 判决 —— 点估计塌到 −0.030、CI 跨零，**测不动**；瓶颈是 val 只有 479 个 sequence
- [x] (d'6) V-JEPA 2.1 —— **跳过**：没有官方权重，只有第三方转存，按第 12 条的规矩不用

## 结果

跑完再填。run dir 见下。
