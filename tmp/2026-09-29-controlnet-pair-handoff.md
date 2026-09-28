# ControlNet 配对 pilot：交接给下一个 Sonnet（2026-09-28 23:20 CST 写）

任务：等 GPU 2 上的队列跑完 → 渲染 WebP → 拉回 Mac → 生成审阅页 `research/controlnet-pair-pilot.md` → **08:30 CST 前** push → 给 main 发一行 URL，然后停。
用户改过的 brief：**不跑自动检查套件**（检测器 / 次要物体 / seed 噪声地板 / pale-flat），用户明早自己看片段；**先不写 decisions.md**；自己只看到能剔除明显垃圾的程度。

## 代码（都已 push，box 已 pull）

- `jevdrive/cn_pair.py`：prep / specs / anchor / blend / depth / insert / link / webp / summary 子命令
- `scripts/cn_pair.sh`：一个 stage 的链式脚本；`scripts/cn_pair_infer.py`：包 Cosmos v2 的 `scripts/cosmos_infer.py`（只读导入，日志写到 `runs/cn_pair/infer/`）
- `scripts/cn_pair_md.py`（**Mac 上跑**）：读 `research/results/cn_pair/pairs.json`，把审阅区写到 note 里 `<!-- REVIEW ... -->` 标记之下，保留已填的结论
- 不要改 Cosmos v2 的文件（`jevdrive/cosmos_*.py`、`scripts/cosmos_*`），它们属于另一个 agent。

## box 上正在跑的队列（GPU 2，和 GIMM 插帧 8 个进程共享，每个 distilled clip 约 200 s）

- tmux `jev` 窗口 `cn-s2`；命令 `env SCENES_BV=p3_000 SCENES_BVI=p3_000 SCENES_EIG=p3_000,p3_003,p3_009,p3_015 scripts/cn_pair.sh s2 all E,EI,EIG,BV,BVI`
- PID：链 bash 738625（子 bash 739188，当前 Cosmos python 739226，每个 arm 换一个 python）
- 信号文件：`$DATA_DIR/runs/cn_pair/s2.STATUS`（每步一行）、`s2.DONE`、`s2.ERROR`（`DATA_DIR=/root/autodl-tmp/ujs`）
- 顺序与预计：E 全 19 场景（23:15 起，约 2 h）→ EI（depth + insert + 19 个 x⁺，约 1.1 h）→ EIG 4 场景（约 15 min）→ BV p3_000（base 35 步 edge+vis，2 个 clip，估 1–2 h，**没实测过**）→ BVI p3_000（1 clip）。估计 05:30 前 DONE。
- s1（p3_000 上的 E、Ed、EG）已完成；s1 的 BV 我在 23:15 按 PID 手动停了（挪到 s2 最后），`s1.STATUS` 末行记了这件事，不是错误。
- 每个 Cosmos 进程受 `runs/cn_pair/gpu.lock`（flock）保护，一次只跑一个。

**输出路径**（box，`$DATA_DIR/runs/cn_pair/`）：
- `clips/<p3_xxx>/`：`rgb.npy` 真实帧、`clean.npy` LaMa clean plate、`clip.npz`（目标掩膜 / 移除区域）、`edge_plus|minus|drop.mp4`、`prep.json`；插入的 `ins.npz`、`ins.json`、`edge_ins.mp4`、`ins_video.mp4`、`depth_clean.npy`
- `gen/<arm>/<p3_xxx>_{plus,minus}_<arm>.npy`：生成帧（T×704×1280×3 uint8）。arm 目录：`E`、`Ed`、`EG`→贴回后是 `EGb`、`BV`、`EI`、`EIG`→`EIGb`、`BVI`；插入 arm 的 minus 是指向 E（BVI 指向 BV）x⁻ 的软链
- `figs/<p3_xxx>_<arm>.webp`：审阅片段；`infer/*/*/events.jsonl`：每个 sample 的秒数

## 等待（一个阻塞 wait，后台跑，超时 ≥ 3 h）

```bash
ssh -o ControlMaster=no -o ServerAliveInterval=60 autodl 'R=/root/autodl-tmp/ujs/runs/cn_pair; timeout 21600 bash -c "until [ -e $R/s2.DONE ] || [ -e $R/s2.ERROR ]; do sleep 60; done"; date; cat $R/s2.STATUS; cat $R/s2.ERROR 2>/dev/null'
```

若 06:30 还没 DONE（BV 太慢）：不等了，按 PID 停链（先链 bash 738625，再当前 cosmos python），用已有的出页。若 ERROR：看 `tmux capture-pane -p -J -S -200 -t jev:cn-s2`，能修就用同一命令重跑（已生成的 npy 会自动跳过），修不了就用已有的出页并在页里写明。

## 渲染、汇总、拉回

s2 结束后**统一重渲所有 WebP**（s1 的片段是旧尺寸），然后汇总：

```bash
ssh autodl 'cd ~/data/jev-drive && export DATA_DIR=/root/autodl-tmp/ujs PYTHONPATH=$PWD && P=$DATA_DIR/envs/jevdrive/bin/python;
  $P -m jevdrive.cn_pair webp --arm E --scenes all; $P -m jevdrive.cn_pair webp --arm EI --scenes all;
  for a in Ed EGb BV BVI; do $P -m jevdrive.cn_pair webp --arm $a --scenes p3_000; done;
  $P -m jevdrive.cn_pair webp --arm EIGb --scenes p3_000,p3_003,p3_009,p3_015;
  $P -m jevdrive.cn_pair summary --scenes all'
# Mac:
mkdir -p research/figs/cn_pair research/results/cn_pair
scp autodl:/root/autodl-tmp/ujs/runs/cn_pair/figs/p3_*.webp research/figs/cn_pair/
scp autodl:/root/autodl-tmp/ujs/runs/cn_pair/pairs.json research/results/cn_pair/
python3 scripts/cn_pair_md.py
```

WebP 版式：real（目标黄色轮廓；插入对是行人路径轮廓）| x⁺ | x⁻，每面板 384 px 宽，每 2 帧取 1（5 fps），约 0.8–1 MB。只拉 `p3_*.webp`，不要拉 `sheet_*.jpg` / `chk*.jpg`（我的检查图）。
`summary` 只收 webp 已存在的 (场景, arm)；剔除某项 = 删掉 `research/figs/cn_pair/` 里那个 webp 并从 `pairs.json` 删那一行，再跑 `cn_pair_md.py`。

## 什么算明显垃圾（剔除，并在页里「剔除」小节列出 id 和一句原因）

- x⁻ 里目标行人还在（肉眼一眼可见），或 x⁺ 里目标完全没有；
- 画面出现挡风玻璃 / 仪表台 / 车内、整段崩成噪声或卡通；
- 插入对里行人不可见（`ins.json` 的 visible 很少）或明显飘在空中 / 穿过车身。
拿不准的不剔除，交给用户。

## 页面（`research/controlnet-pair-pilot.md`，已写好上半部分）

（注意：这个文件目前只在 Mac 工作区里，还没提交。）上半部分已写：为什么、做法（片段、掩膜、移除区域怎么补、插入的行人、arm 一览表）。标记 `<!-- REVIEW: ... -->` 以下由 `cn_pair_md.py` 生成：「审阅：移除对」「审阅：插入对」两节，每项 `### p3_xxx · <arm 名> \`arm\``、整行 WebP、信息表（场景 / 设置 / 目标行人或插入的行人 / 区域外平均像素差 / GPU 时间）、空的「**结论（用户填）：**」行。每个 arm 的设置文字在 `scripts/cn_pair_md.py` 的 `ARMS` 字典里，与页上 arm 一览表一致：

- E：Edge Distilled 4 步；x⁺ = 真实帧 Canny；x⁻ = 区域外同 x⁺、区域内 LaMa clean plate 的 Canny；同 prompt、seed 2025，各自独立生成
- Ed：同 E，x⁻ 区域内边缘清空
- EG(b)：x⁻ = E 的 x⁻；x⁺ 区域外 latent 钉在 x⁻ 上 + 羽化贴回（区域外逐像素相同，不对称）
- BV：base 35 步 CFG 3，edge 1.0 + vis 0.5（vis 由 Cosmos 对输入视频现算：x⁺ 真实、x⁻ clean plate），带 negative prompt
- EI：x⁻ = E 的 x⁻；x⁺ = x⁻ 边缘 + CARLA `27515-s0` 0 号行人步态循环的 Canny，1.5 m/s 从右路缘横穿，第 80 帧到车道中心，DA3 深度遮挡
- EIG(b)：EI 的 x⁺ 锚定在 E 的 x⁻ 上；BVI：BV 版插入

**还要在标记之上补三小节**（中文，按 research/README.md）：
1. 「p3_000 上的观察」（我已看过）：
   - E 的两边都很真实，x⁻ 里目标已去掉，旁边的非目标行人两边一致；区域外 x⁺ / x⁻ 平均像素差 1.5–2.5（0–255）。
   - **Ed（区域内不给边）更差**：x⁻ 在原来人的位置长出一个偏红的模糊「鬼影」人形；所以区域内用 clean plate 的边补（E）是选定的做法。
   - **颜色漂移**：distilled 只有 edge、没有外观输入，布局跟真实片段一致，但颜色会变（例：红白相间的公交车被画成白 / 灰色，楼的颜色也变了）。针对它排了 BV / BVI（base + vis 控制，vis 带真实颜色），只在 p3_000 上，因为 base 每 clip 估计 30–60 min。
   - EG 区域外逐像素等于 x⁻（按构造），代价是两边不再对称生成。
2. 「成本」：表格按 arm 列每 clip 秒数（`pairs.json` 的 s_plus / s_minus）和合计 GPU·h；注明 GPU 2 与 GIMM 共享，约为独占卡的 2.2 倍（E 在独占卡上 Cosmos v2 测过约 91 s / clip）。
3. 「没做 / 已知问题」：自动检查按用户指示没跑；seg、depth 没进生成（理由已在做法里）；WOD 10 Hz 喂给 16 fps 训练的模型；凸包帧多的场景（p3_001、006、017）移除区域是方框；插入行人外观来自 CARLA（只用了它的边缘和步态）；插入对 x⁻ 与移除对共用。

## 收尾

1. `README.md` 顶层索引加一行（仿照 p3-review-sheet 那行）：`research/controlnet-pair-pilot.md` —— ControlNet（Cosmos-Transfer2.5）同源生成的行人配对 pilot，真实 WOD 片段上的移除 / 插入对，real | x⁺ | x⁻ 审阅片段 + 空结论行。
2. 只提交自己的文件：note、`research/figs/cn_pair/*.webp`、`research/results/cn_pair/pairs.json`、README.md 那一行。不要 `git commit -a` / stash / reset / clean（工作区里有别人的未提交改动）。push。这些只动 research/，box 不用 pull。
3. 撤 scheduler 行：`ssh autodl 'cd ~/data/jev-drive && DATA_DIR=/root/autodl-tmp/ujs /root/autodl-tmp/ujs/envs/jevdrive/bin/python -c "import sys; sys.path.insert(0,\"scripts\"); import sch_table as S; r=S.load(); [x.update(status=\"done 2026-09-29: ControlNet pair pilot finished, GPU 2 released\") for x in r if x[\"lane\"]==\"cn-pair\"]; S.save(r)"'`
4. 关掉 tmux 窗口 `cn-s1`、`cn-s2`、`cn-prep2`（任务都结束之后）。
5. 给 main 一行：页面 URL（`https://github.com/VennIntelligence/jev-drive/blob/main/research/controlnet-pair-pilot.md`），然后停。

规则提醒：box 进程只按精确 PID 停，不用 pkill -f / pgrep -f；不发 Artifact 页面；不打印 / 提交任何密钥。
