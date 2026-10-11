# FAMILY 预注册：openpilot 家族在 private 上的掉分，是不是相机标定 / 几何的问题（2026-10-12，读数之前写）

## 要解释的现象（decision 262）

private nuPlan 上只有我们的 openpilot 家族掉：APY10m10-AB-s0 0.935 -> 0.640（GT 1.13 -> 2.29 m），YR10m10-F-s0 0.925 -> 0.800
（GT 0.94 -> 1.73 m，at-fault km x0.67）；LTF / DD 样例基本持平，garage 0014 反而涨。重加权 public 场景复现不了；掉分是均匀地离开 log 轨迹，
不集中在某类场景。我们独有的东西：把 CAM_F0 按 session 告诉的标定 warp 到 openpilot 的标准相机（`navsim_zs.OpenpilotMaps`，
`sh30_core.pack_fast`），外加 openpilot 自己的 ego / rig 假设。

## 写预注册之前已经知道的（代码阅读 + 一张图，非 sensitivity 读数）

1. 标定来源：driver 收到的 `available_cameras` 来自 MTGS asset 的 `video_scene_dict.pkl`（`frame_infos[0].cams`，优先 `colmap_param`）
   （`plugins/mtgs/server/artifact_adapter.py:get_available_cameras_from_data_source`）。渲染用同一份 K 和 R，**但渲染还用 asset 的畸变
   （colmap k1 中位数 -0.338，k2 0.113）做 `initInverseRectificationMap`，输出的是带桶形畸变的图；上报给 driver 的标定里没有畸变系数**。
   1491 个 public asset 全部如此（`results/asset_calib_public.csv`）。一张渲染帧和同一时刻的真实 nuPlan 帧对比：与"畸变的真实帧"重合，
   与去畸变后的不重合（边缘 NCC 0.37 vs 0.31，周边 0.40 vs 0.33）。所以我们的 warp 在 public 上就一直把畸变图当针孔图采样；
   训练时（navtrain 真实图）是带 D 去畸变的。LTF / DD / garage 都不读标定，直接裁原图，与它们训练时的原始畸变图一致。
2. 上报与渲染的代码路径对 pickle 的拆包方式不同：渲染路径接受"扁平"或"按场景嵌套"的 dict，上报路径只接受嵌套的；扁平格式的
   asset 会让 driver 收到默认相机（fx 1920, fy 1080, cx 960, cy 560, 单位旋转 = 光轴朝上, t = 0），而渲染照常。public 1491 个全是嵌套。
3. AUD1 的替换实验（4 个场景）已经显示：标定 2 deg yaw / 1 deg pitch 不匹配会让规划偏 0.5-1.5 m，分数 0.74-1.00。

这些是"可能的机制"，不是结论；本预注册只决定怎么量。

## 实验（task 2）

- 场景：96 个 public 场景（`results/s96.txt`），16 辆车每辆 6 个（veh 不足 6 的全取，余数随机补齐，种子 20261012），全部是有指标的场景。
  存储的 YR10m10-F-s0 在这 96 个上：0.953 / GT 1.00 m / 2 次 at-fault。
- 驱动：YR10m10-F-s0（第二次提交的 checkpoint），`sh30` driver，`SH30_COMPILE=0`（与提交镜像一致）。扰动由
  `scripts/fam_driver.py` 包装注入（驱动代码本身不改；`scripts/fam_perturb.py` 定义每个 arm）。
- 对照样例：shipped LTF，`scripts/fam_ltf.py` 包装，同一 96 个场景。
- 运行：GPU pool，`ot2_loop.py`（OT_LANE=family），一个 arm 一个 job。先跑 pilot（A0、Dasset、P+2、L0），过 G0 / G1 再放其余。
- 估计：一个 96 场景 job 约 10-15 min（LTF 400 场景约 15 min；我们的解释执行更慢），25 个 job，约 3-5 卡时，同时最多 3-4 个 job，
  墙钟约 1.5-2 h。

### Arm 列表（标定 arm：图像不动，只改告诉 driver 的标定 = "上报和渲染不一致"）

| arm | FAM_ARM | 含义 | 相对 public 范围 |
|---|---|---|---|
| A0 | (空) | 对照 | |
| P+1 / P+2 / P-2 | pitch=±d | 光轴被认为再向下 d deg | public pitch 0.2-2.4 deg，跨车差 2 deg |
| Y+2 / Y-2 / Y+4 | yaw=d | 光轴被认为再向左 d deg | public yaw -1.8..+2.2 |
| R+2 | roll=2 | | |
| F0.95 / F1.05 | f=s | fx, fy 同乘 | public fx 1435-1623（±5 %） |
| FY0.9 | fy=0.9 | 只改 fy（各向异性） | public fy/fx 0.53-1.07 |
| CX+40 / CY+40 | cx / cy +40 px | 主点 | public 只有 veh-08 偏 10 / 18 px |
| TZ+0.3 / TZ-0.3 | tz=±0.3 | 相机高度 | public 1.49-1.58 m |
| Dnom | D=nom | 告诉 driver public asset 的中位畸变 | **修复候选（纯 serving）** |
| Dasset | D=asset | 告诉 driver 该场景 asset 自己的畸变 | 修复上限（oracle） |
| DEF | default | MTGS adapter 的默认相机 | 扁平 asset 的情形 |

图像 arm（改图像本身，所有 CAM_L0/F0/R0，JPEG 质量 95 重编码）：
| arm | FAM_ARM | 含义 |
|---|---|---|
| RE | reenc=1 | 只重编码（图像 arm 的对照） |
| Z0.9c | zoom=0.9+kfollow=1 | 换成窄 10 % 的镜头且 K 跟着改（一致的不同 rig） |
| Z0.9m | zoom=0.9 | 同样的图，K 不改（不一致） |

LTF：L0（对照）、Lcal（pitch=2+yaw=2+f=1.05，改它收到的标定）、LRE（reenc）、LZ（zoom=0.9）。

### 指标（与 AUD1 / PROXY 同口径，`aud1_drift_collect.py` 读）

每个 arm 对 A0（LTF 对 L0）配对：平均 scene score、GT distance（截断后的每 clip 最大值的均值）、at-fault 次数和 km/次、零分数、
每个决策步 k 的平均离轨距离（k = 4 / 7 / 10）、k=4..10 斜率。CI：场景 bootstrap 10 000 次（只有 8 个 log，log 聚类 bootstrap 作参考）。

### 门（不过就停，写原因）

- G0（管线）：A0 与存储的 YR10m10-F-s0（编译执行）在 96 个场景上平均分差 <= 0.02、GT 差 <= 0.10 m；所有 arm 的 drive.jsonl start 记录里
  的标定等于该 arm 的设定；除 DEF 外 inference / input error = 0。
- G1（LTF）：L0 与存储的 LTF anchor 在这 96 个上平均分差 <= 0.01。

### 判据：哪个扰动"复现了 private 的形状"

一个 arm 复现 YR 的 private 读数，当且仅当同时满足：
(a) Δscore（arm - A0）在 [-0.20, -0.06]（private -0.125）；
(b) GT 比值 arm / A0 在 [1.4, 2.4]（private x1.84）；
(c) 均匀：k=10 平均离轨距离上升的车 >= 12 / 16，而不是几个场景归零贡献全部；
(d) 对照 LTF 在同类扰动下 |Δscore| <= 0.02（标定 arm 对 LTF 按代码应完全不变）。
只满足 (a)(b) 的记为"量级相符、形状不明"。at-fault 只报告不作判据（96 个场景里存储基线只有 2 次）。

如果某 arm 复现：写清是哪个、修复是什么样（serving 侧优先，不重训）、验证要花多少（public 全量 1491 场景约 1 卡时 + 一次 warm-up）。
如果 Dnom / Dasset 在 public 上明显变好（Δscore 下界 > 0），它本身就是一个 serving bug 修复，单独记录，需要 1491 场景确认才可进镜像。

## 不做的事

不跑 challenge CLI submit / warm-up；不改第三方代码（MTGS、LTF、garage 只读）；不改提交镜像的 driver 文件（扰动只在包装里）。

## 补充 1（读完 YR 的 24 个 arm、读 APY 之前写，2026-10-12）

YR 的结果：P+2 / P+3 / Y+4 满足判据 (a)(b)(c)，LTF 的标定 arm 与对照逐 scene 相同。追加两个探索性检验（不改原判据）：
1. **APY 的排序预测**：private 上 APY 掉得比 YR 多（-0.295 对 -0.125，GT x2.0 对 x1.84）。如果"上报的俯仰比渲染偏下约 2 deg"是机制，
   同一扰动下 APY10m10-AB-s0（ap2 driver）应比 YR 掉得更多。预测：APY 在 P+2 下 Δscore < YR 的 -0.148。不满足则机制的证据减弱。
   跑 APY 的 A0 与 P+2，同 96 个场景。
2. **Rnom**：不信上报的安装旋转，改用固定的名义旋转（水平，俯 1.45 deg）：在 public 上要付多少（serving 修复候选的代价）。
