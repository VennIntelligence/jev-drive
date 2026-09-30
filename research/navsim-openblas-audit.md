# NAVSIM 分数的 OpenBLAS 环境 bug 审计（2026-09-30，CPU only）

**结论先行**：没有任何一个我们产出的 NAVSIM 数字受这个 bug 影响，不需要改任何分数，也不需要重建任何 metric cache。
这个 bug 不是 09-28 换机后才有的：同型号 CPU（Xeon Platinum 8470Q）上 09-24 就撞到过，当时已经修掉（decisions 第 37 条附带发现、[docs/navsim.md](../docs/navsim.md)），
之后所有打分都经 `scripts/navsim_zs_score.sh`（强制 `OPENBLAS_CORETYPE=Haswell`，启动时做 40×40 求逆自检）或自带该变量的脚本。
另一个 agent 在 09-30 13:5x 看到的发散，是它在未设变量的 shell 里做诊断探测（navtest 前 40 个 token）时撞到的，不是某个已交付的 run。
现在 navsim1 / navsim2 两个 env 已经自带该变量（第 4 节），以后没人需要记得。

## 1. bug 与修复的确认（tiny input）

box 上 `/tmp/s_blas.py`（批量 pinv 对 lstsq 的最大差）与一个 40×40 逆、一个 80×41 的 SVD 重构，numpy 1.23.4（`envs/navsim1`，`envs/navsim2` 同结果），未设变量时 OpenBLAS 自动选 Cooperlake：

| 核类型 | pinv 对 lstsq 最大差（s_blas.py） | 40×40 逆误差 | pinv·B − I | SVD 重构误差 |
|:--|--:|--:|--:|--:|
| 自动（Cooperlake）/ 显式 Cooperlake / Sapphirerapids | **3.54** | 132.5 | 9.66 | 42.9 |
| Haswell / SkylakeX | 4.5e-16 | 5.5e-15 | 1.8e-15 | 1.5e-14 |

bug 与修复都成立，并且不止 pinv：inv、SVD 一样坏。所以 `navsim_zs_score.sh` 里的 inv 自检足以抓到它。SkylakeX 与 Haswell 都对。

**灵敏度对照**（证明「设了变量的 run 与没设的 run 分得开」）：`navsim_zs_score.sh` 的副本去掉自检、核类型改成 Cooperlake，打 navtest `none` 的前 300 个 token：

| 300 token | 有 flag（存档 run） | 无 flag（对照） | 逐 token 不同的个数 |
|:--|--:|--:|--:|
| PDMS | 85.95 | **0.00** | 286 |
| DAC | 96.0 | 0.0 | 288 |
| EP | 75.5 | 0.0 | 286 |
| NC | 98.5 | 79.3 | 72 |

坏掉的签名是 DAC = 0、PDMS ≈ 0，不是「差一点」。这一条是第 2 节的判据：任何一个没设 flag 的 run 会整批崩到零。

## 2. 哪些 run 设了 flag（09-28 之后的全部 NAVSIM 打分与建 cache）

判据有三层：(a) 启动脚本里有没有；(b) 还活着的进程的 `/proc/<pid>/environ`；(c) 结果本身有没有第 1 节的签名（这一层不依赖记忆，覆盖所有 run，包括从 tmux 手敲的）。
09-28 之后 box 上 `runs/navsim/eval/` 里新出现 85 个 run（`research/results/navsim-openblas-audit/runs_since_0928.csv`，不含审计自己的 4 个复打）：**无一个 DAC 或 PDMS 接近 0**，最低的 46.3 是 hold 输入变体（op-lb 的 hold 插帧对照，DAC 78% 是输入本身差，不是环境）。

| run 组 | 计算了什么 | flag | 证据 |
|:--|:--|:--:|:--|
| op-lb 全部 `opi_*`（navtest 12 146 / 2 000 子集、navhard、navtrain 3 000） | openpilot Cinque / Lebowski / small 各插帧变体的 PDMS / EPDMS | 是 | 经 `navsim_zs_score.sh`（第 23 行 `export OPENBLAS_CORETYPE=${OPENBLAS_CORETYPE:-Haswell}` + 自检）；`none` 复现 84.18 / 33.33 |
| skill pack N0 / N1 / N1b / N2 / S / N3（`sp_*`，navtest 与 navhard） | 各臂选出的 pose 的官方分 | 是 | 同上，经 `navsim_zs_score.sh`；标签抽取走 `elicit_e6.sh`、`n1_score_native.py`、`skill_pack_n1.sh`、`navsim_raise_{scale,n3,n4}.sh`，脚本内都写死该变量 |
| N0 的 navtrain T 选参集（`sp_n0_T_*`） | 选参用的 PDMS | 是 | 同上 |
| navtrain 全量 metric cache（`v1_navtrain`，106 630 个 pkl） | PDM-Closed 参考轨迹与地图几何 | 是（间接） | `navtrain_mcache.sh` 第 19 行设该变量并自检；该脚本 09-29 14:51（box 时区）提交，pilot1 14:52 启动；活进程环境无法回溯（进程已结束）。**逐个 pkl 扫描（第 3 节）全部正常，此项不依赖启动环境** |
| `v1_navtrain_oplb`（3 000） | 同上 | 是 | 由 `op_lb_lane.sh` 调 `navsim_zs_score.sh cache`，自带变量；扫描正常 |
| N4 全 navtrain 打分（进程 490674 及其 16 个 worker，仍在跑） | anchor 子分标签 | 是 | `/proc/<pid>/environ` 17 个 navsim1 进程全部 `OPENBLAS_CORETYPE=Haswell` |
| op-adapt r2 S 包：V5a、几何抽取（`op_adapt_score_nav.sh`） | devkit 的 DDC / DAC 对照、地图 WKB | 是 | 脚本第 12 行 `export OPENBLAS_CORETYPE=Haswell`；几何来自 cache pkl，不跑仿真 |
| op-adapt r2 S 包里 13:5x 之前的探测（navtest 前 40 token 发散） | 诊断，不是结果 | 否（故意） | 就是这次发现 bug 的那次探测，没有任何交付物依赖它 |
| nq3 / nq4 / real_g* / top10 / seeds 等更早的 lane | 各自的 navtest 打分 | 是 | 全部经 `navsim_zs_score.sh` 或自带变量的 `nq3_d/navscore.sh`；均在 09-24 修复之后；这些 run 多数在 09-28 之前，不在上面 96 个里，09-24 起的修复见 navsim.md |

grep 全仓：`scripts/` 与 `jevdrive/` 里提到 navsim env 或 devkit 的文件中，不含变量的都是不打分的（索引、报告、安装、agent 定义）或者经 `navsim_zs_score.sh` 间接调用的封装（`navsim_raise_n1b.sh`、`navsim_raise_n2.sh`、`skill_pack_n0.sh`、`op_interp_score.sh` 等）；没有一个直接跑 devkit 而不设变量。

## 3. 打分与 cache 有没有被污染

### 3.1 重打分（bug 检查，不是新的选择性读数）

用存档的 pose 文件，用同一 devkit 与同一 cache，在 `OPENBLAS_CORETYPE=Haswell` 下重新打分，逐 token 与原 run 对比。navtest 与 navhard 各两个配置，40 核，约 10 分钟。这是 bug 检查，不算对 navtest / navhard 的一次新读数：没有任何配置或超参是据此选的，也不计入 navsim-raise 的 navtest 次数（6 次）。

| 配置 | 数据 | 原分数 | 重打分 | 逐 token 不同的个数 / 最大差 | 34 列（子分、两阶段）不同的个数 |
|:--|:--|--:|--:|:--|--:|
| `none`（op-lb Cinque 原生） | navtest 12 146，PDMS | 84.1804 | 84.1804 | 0 / 0 | 0 |
| N3 | navtest 12 146，PDMS | 91.5928 | 91.5928 | 0 / 0 | 0 |
| `none` | navhard 5 912，官方 EPDMS / 均匀权重 | 33.33 / 33.43 | 33.33 / 33.43 | 0 / 1e-16 | 0 |
| N3 | navhard 5 912，官方 EPDMS / 均匀权重 | 33.28 / 31.20 | 33.28 / 31.20 | 0 / 1e-16 | 0 |

四个配置逐 token 逐列**逐位相同**（navhard 的 1e-16 是求和顺序）。navhard 的均匀权重对 `none` 的配对差 N3 −2.23 [−4.19, −0.29] 也逐位不变。

### 3.2 metric cache

全部 cache 的 PDM-Closed 参考轨迹逐个扫描（读每个 pkl，5 s 内 11 个点算最大速度与总长；坏掉的签名是 10³–10⁴ m/s）：

| cache | pkl 数 | 建成时间（UTC，由文件 mtime） | 最大速度 m/s | 5 s 最大长度 m | > 40 m/s 的个数 |
|:--|--:|:--|--:|--:|--:|
| `v1_navtrain` | 106 630 | 09-29 06:53 至 19:06 | 19.4 | 87.4 | 0 |
| `v1_navtrain_oplb` | 3 000 | 09-29 06:04 至 06:17 | 15.6 | 75.9 | 0 |
| `v1_e6sub` | 20 000 | 09-25 | 19.4 | 87.4 | 0 |
| `v1_navtest` | 12 146 | 09-24 | 15.6 | 76.4 | 0 |
| `v2_navtest` | 12 146 | 09-24 | 15.6 | 76.4 | 0 |
| `v2_navhard_two_stage` | 5 912 | 09-24 | 15.0 | 72.0 | 0 |

navtest、navhard、e6sub 三个 cache 都在 09-24 与 09-25 建成，早于换机，且在 09-24 的修复之后；navtrain 两个 cache 在 09-29，flag 在脚本里，扫描证明没有一个条目发散。所以「如果缺 flag，估计受影响条目数」这一问的答案是 0，不需要抽样重建来估计。

### 3.3 那 4 个不一致的 token 不是这个 bug

`v1_navtrain` 与 `v1_navtrain_oplb` 的 3 000 个共同 token 中，PDM-Closed 参考轨迹有 4 个不同（复算：最大逐点差 0.089、1.60、1.68、17.65 m）。用 flag 重建这 4 个 token 两次（2 线程与 4 线程），并在同一次里重建 296 个随机共同 token：

| 比较 | token 数 | 不同的个数 | 最大差 m |
|:--|--:|--:|--:|
| 重建 A 对 重建 B（2 线程 vs 4 线程） | 4 | 0 | 0 |
| 重建 对 `v1_navtrain` | 4 / 300 | 1 / 1 | 0.089 |
| 重建 对 `v1_navtrain_oplb` | 4 / 300 | 3 / 3 | 17.65 |
| 上面 296 个随机共同 token，重建 对 两个存档 | 296 | 0 | 0 |

读法：重建（有 flag）与 `v1_navtrain` 在其中 3 个 token 上逐位一致，与 `oplb` 不一致；在另 1 个 token（差 0.089 m）上与 `oplb` 一致、与 `v1_navtrain` 不一致。也就是说这 4 个 token 是「两次建库结果各有胜负」的近平局，flag 缺失会让整批崩掉，不会只坏 4 个。重建对重建逐位一致，说明同一次构建内是确定的，差异来自建库之间（最可能是 PDM-Closed 在提案近似并列时的选择，与批次、进程分配或 ray 调度有关，**推测，未验证**，验证办法是逐 token 记录提案得分再比较）。量级：3 000 中 4 个（0.13%），只影响 navtrain 上这些 token 的 EP 参考。不需要重建。

## 4. 持久修复

选择的方案：每个 env 的 site-packages 里放 `sitecustomize.py`，在 numpy 被 import 之前 `os.environ.setdefault("OPENBLAS_CORETYPE", "Haswell")`；再在 `etc/conda/activate.d/` 放同样的 export（`conda activate` 时用）。
选它是因为：box 上没有 `conda` 命令，`conda env config vars set` 用不了；`sitecustomize` 对每一个由该 env 的 python 启动的进程生效，不管有没有 activate，也不管从 tmux、ray worker、`env -i` 还是 subprocess 启动；`setdefault` 让调用方显式设的值仍然优先；只影响新进程，不动任何运行中的进程。

| 检查 | 结果 |
|:--|:--|
| `env -i PATH=/usr/bin:/bin` 干净 shell 下 `envs/navsim1` 与 `envs/navsim2` 的 python 读到的变量 | `Haswell`（两个 env） |
| 同一干净 shell 下 `s_blas.py` | pinv 差 4.5e-16（原 3.54） |
| 调用方显式设 `SkylakeX` | 保持 `SkylakeX` |
| `python -I`（隔离模式） | 仍然正确 |
| 新脚本 `navsim_zs_score.sh` 的自检 | 不变，照旧保留（多一层保险） |

文件在 box 的 `envs/` 里（不在 git，envs 按规矩留在数据盘）；为了 env 重建时不丢，`scripts/setup_navsim_devkit.sh` 末尾加了同样的写入步骤，`docs/navsim.md` 的说明改为「env 已自带」。已核对：没有 `python -S` 的调用（`-S` 会跳过 sitecustomize）。

## 5. 对其他 lane 的影响

- op-adapt r2、N4 的所有 NAVSIM 数字不需要改；r2 prereg 里「09-28 之后的 NAVSIM 分数需要核对」一句可以标为已核对，指向本文。
- 「自 09-28 起」这个说法本身不对，应为「自 09-24 起在这个 CPU 型号上一直存在，09-24 已修」，新 env 补丁之后不需要再靠脚本记忆。
- 本次审计的产物：`research/results/navsim-openblas-audit/`（逐 run 汇总、逐 token 前后对比汇总 `before_after.json`、cache 对比）；box 上的中间物在 `$DATA_DIR/runs/navsim_audit/`（含重建的 300 个 token 的 cache，`caches/`），没有删任何东西。
