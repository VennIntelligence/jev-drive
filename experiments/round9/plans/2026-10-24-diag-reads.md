# 2026-10-24 夜间诊断读数：P1L 开 Harmonizer、INTP75 重读

描述性，不设判读线。用户已授权在空闲卡上跑只读诊断；不训练、不构建镜像、不提交、不删除。

## 1. P1L（PCL1L-F-s0）开 Harmonizer

- 为什么：决定 435 / 439 / 442 / 443 里 NTV0-s0 和 R8K-s2 都有开 Harmonizer 的存档读数，P1L（决定 461）只有关的。想知道 P1L 的「开 − 关」差和这两条线是否同量级，以及零分类别怎么移动。
- 读法：PAI 281 ACC + GAP，serve 同 P1L 存档读数（`JEV_VCONT=1.0 JEV_LEAD=1 JEV_ACC=1 JEV_GAP=0.7,2.7`，g10，ALPASIM_FAST=1，CONC 4），只加 `HARMONIZER=1`（VRAM 30 / RAM 40，同 agH 读数）；8 个分块列表与 P1L 关读数相同。
- 报：新评分器为主、旧评分器并列；对 P1L 存档关读数的配对差；与 NTV0-s0、R8K-s2 的存档开读数在共同场景上并排；零分类别。渲染日志里核对 harmonizer 权重已加载。

## 2. INTP75-NR 重读（同检查点噪声下限）

- 为什么：决定 463 里 INTP75 的负翻转 8（R8K-s2 为 14）只有一次读数，没有同一检查点的基线，闭环不是逐位确定的。重读一次给出同检查点的场景一致率与负翻转波动。
- 读法：与第一次读数完全相同的 serve 与 8 个分块列表，Harmonizer 关；唯一的不同是再读一次（标签 `INTP75-NR-ag-rr`）。
- 报：与第一次读数逐场景一致的场景数（得分完全相同）、平均 |差|、均值差与 CI、在 NTV0-s0 满分场景上的负翻转数（重读 vs 第一次）。

## 做法

`experiments/round9/scripts/diag.sh`（复用 p1l.sh / interp.sh 的读作业体与分块列表，一次提交整条链：两条各 8 个读块 + tax → scorer_v2 new → 报告 `diag.py report`）。产物在盒上 `$DATA_DIR/runs/round9/diag_reads/`，小结果落 `experiments/round9/results/diag_reads/`。
