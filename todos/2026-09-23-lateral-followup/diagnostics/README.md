# 同记录重放：分解轨迹更新与运动重投影

2026-09-23。本诊断不启动 CARLA、不修改既有冻结文件，不把反事实控制输出解释为新的整车轨迹。

## 重放先决条件已满足

使用 turns-v1 归档的 Controller 源码、原配置、`trajectories.jsonl` 的完整20×2点和来源时刻，以及 `control.jsonl` 的原 `sim_time/speed_mps/yaw_rate_rps`。按原顺序先 update 后 step；只剥离与运行 agent 一致的 adapter/metadata 配置项，不重建 GPS、定位、route adapter 或模型点。

六例 **2,761帧、692次更新**全部逐值复现：throttle/steer/brake、全部 Controller diagnostics（含 raw_steer、aim、reason、PI等）和 update acceptance 均完全相同，最大数值误差0。使用原 `/data/envs/carla/bin/python`、NumPy1.23.5；系统NumPy2.4.6探查出现约1e-16舍入差，因此正式结果采用原环境。详见 [重放核对](replay-v1/replay-verification.json)。

## 单次更新的反事实定义

每个5Hz更新之前复制同一 Controller 状态：

- A：接受本次记录的新轨迹，再按记录的当前motion step。
- B：只跳过这一次 update，保留旧轨迹，按完全相同的当前motion step。

B只用于这一tick，随后丢弃。下一tick继续真实A状态，不把B输出反馈到车辆，也不假设未来motion仍会相同。两分支当前积分位姿最大差0。

令上一真实raw为P、当前A为N、当前B为O，则：`N−P = (N−O) + (O−P)`。其中 N−O 是接受新轨迹的即时效应；O−P 是保留旧轨迹时运动、参考年龄以及速度相关lookahead的推进效应，并非纯几何平移。代数分解残差最大1.11e-16。emitted steer也分解，但受原限制状态影响，不能据此把线性贡献直接加成车辆效果。

首次无旧轨迹、停车或safe分支的raw缺失保持缺失，不拿emitted替代raw，也不补0。truth位置只用于标记原来四个几何窗口，不传给Controller。

## 主要发现

以下为固定窗口内更新时刻的raw steer单tick变化RMS，单位为归一化steer；这是完整20Hz序列中的更新点子集，不是全窗口steer-rate P95：

| 窗口/配置 | 总变化 | 旧轨迹推进 | 接受新轨迹 |
| --- | ---: | ---: | ---: |
| 26966 baseline | .02896 | .00987 | .02128 |
| 26966 short | .04441 | .02286 | .02588 |
| 24240 baseline | .00923 | .00240 | .00878 |
| 24240 short | .01342 | .00357 | .01131 |
| 17563 S1 baseline | .07480 | .02416 | .06243 |
| 17563 S1 short | .07298 | .02360 | .06079 |
| 17563 S2 baseline | .09726 | .02455 | .07925 |
| 17563 S2 short | .08841 | .02359 | .07323 |

RMS不能直接相加：更新项和推进项有时同向，有时抵消。逐更新有符号分解在外部CSV中完整保留。基线四窗口中，更新项幅度大于推进项的次数分别为26966的16/18、24240的24/30、S1的14/18、S2的15/17。

因此证据已超出“更新帧与变化峰值同期出现”：在固定同一历史状态与当前motion的单tick干预中，**接受新轨迹本身确实改变了raw需求，并贡献较大的瞬时跳变**。这仍不能分清新轨迹变化来自 rejoin 几何、定位噪声或其他规划因素，更不能证明减少更新频率、长期沿用旧轨迹或平滑输出能改善闭环。

short组右弯的推进项也明显增大；不能把候选的全部动作代价都归咎于更新。右弯aim更新距离RMS反而从.0712m降到.0494m，而raw更新变化RMS从.02128升到.02588；这与更短前视的转向敏感性一致，但两次真实轨迹不同，不能视为同状态下只改变lookahead的独立因果估计。

## 文件与复现

- [代码](replay_controller_updates.py)
- [八个窗口摘要](replay-v1/window-summary.csv)
- [每窗最大的五个更新效应](replay-v1/top5-update-effects.csv)：只作定位，不以这些帧重新选择评价窗口。
- [源与输出哈希](replay-v1/manifest.json)
- 完整逐帧/逐更新：`/data/runs/b2d/controller/lateral-followup/controller-update-replay-v1/{full-replay.jsonl,per-update-counterfactual.csv}`；源码副本在该目录`source/`。不覆盖旧目录。

```sh
/data/envs/carla/bin/python todos/2026-09-23-lateral-followup/diagnostics/replay_controller_updates.py \
  --run-root /data/runs/b2d/controller/turns-v1 \
  --windows todos/2026-09-23-tcp-controller/agents/lateral-evidence-v1/turn-windows.csv \
  --out /fresh/path/controller-update-replay
```

根代理另选的Hermite aim只读shadow将另存版本，不修改上述精确重放证据。正式实现/guard及新的闭环验收与此理论prototype分开。

## Hermite aim 的同记录理论 shadow：没有观察到普遍下降

按根代理要求，进一步使用 `circle_sampling.py` 的 `hermite_aim` 原函数，对 **baseline .5** 的三例历史状态/路径只读计算。为避免引入当前可变运行代码，通过 AST 仅提取该函数，并保留其源码副本；原 Controller 仍加载 turns-v1 归档版本。

这次原线性分支的1,380帧输出和全部diagnostics仍逐值一致。每帧保持原参考线、原运动、原历史状态、原lookahead，Hermite只改变理论aim/raw需求；不让shadow控制改变下一帧，也不改变纵向计算/输出。每次更新仍做“新轨迹 vs 继续旧轨迹”的同状态对照，两分支都使用同一种插值来计算其效应。

| 固定窗口 | 原linear raw-rate P95，1/s | Hermite raw-rate P95，1/s | 变化 | 原更新效应RMS→Hermite |
| --- | ---: | ---: | ---: | ---: |
| 26966 | .80011 | .98059 | **+22.56%** | .02128→.02250 |
| 24240 | .21838 | .25174 | **+15.27%** | .00878→.00951 |
| 17563 S1 | 2.19666 | 2.34962 | **+6.96%** | .06243→.06420 |
| 17563 S2 | 2.67829 | 2.80125 | **+4.59%** | .07925→.08052 |

这里是**限幅前raw需求的变化率**，不是已施加的emitted变化率；S的数值超过2/s并不等于修改了实际2/s限制。四窗口共70/117/70/71帧全部有有效raw变化率和向前Hermite aim。

26966 raw-rate RMS从.37638略降为.37055，但P95上升；其余三窗RMS也上升。四窗更新效应RMS均未下降。不能只引用一个RMS改善，把结果描述为普遍更平滑。

这份反例说明：**理想无噪声圆弧上的插值改进，未在历史rejoin轨迹的固定state/path shadow中转化为普遍raw纹波下降。** 理想圆弧结论与此并不矛盾，两者参考路径和状态变化条件不同。它仍只是理论prototype，未包含正式实现的全部guard，也未产生新的闭环运动，因此不构成整车通过/失败判决；但后续判断必须保留这份负证据。

- [shadow代码](interpolation_shadow.py)
- [四窗口摘要](interpolation-shadow-v1/window-summary.csv)、[线性重放验证](interpolation-shadow-v1/verification.json)、[哈希清单](interpolation-shadow-v1/manifest.json)
- 大逐帧/逐更新CSV和source副本：`/data/runs/b2d/controller/lateral-followup/interpolation-shadow-v1/`

```sh
/data/envs/carla/bin/python todos/2026-09-23-lateral-followup/diagnostics/interpolation_shadow.py \
  --run-root /data/runs/b2d/controller/turns-v1 \
  --windows todos/2026-09-23-tcp-controller/agents/lateral-evidence-v1/turn-windows.csv \
  --out /fresh/path/interpolation-shadow
```

## 固定进度的理想位置/航向输入诊断

[pose_shadow.py](pose_shadow.py) 先用每次记录的 `route_progress_m`、`route_terminal_hold`、来源 `pose_xy/pose_yaw` 和原始reference/adapter配置重建20点。六例**692次轨迹全部逐值相同，最大误差0**；随后 Controller 的2,761帧输出及全部diagnostics也保持逐值相同。没有缺失truth来源帧，各反事实分支的当前Controller积分位姿差为0，详见 [核对结果](pose-shadow-v1/verification.json)。

每个更新时刻，单独替换①truth后轴位置，②truth世界yaw，③两者；其他参数、记录的进度与terminal hold、当前运动和Controller历史状态固定。不会用truth重新投影进度，不会让shadow改变后续状态。新局部轨迹仍以所选来源位置/航向生成；truth只用于离线理想定位输入诊断，实际控制没有收到truth。

下表是baseline .5四窗在更新时刻的差值RMS，单位为归一化raw steer：

| 窗口 | 原位置误差RMS，m | 原航向误差RMS，° | 仅理想位置的raw差 | 仅理想航向的raw差 | 两者替换的raw差 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 26966 | .349 | .245 | **.04154** | .00365 | .04049 |
| 24240 | .096 | .035 | **.01096** | .00056 | .01100 |
| 17563 S1 | .186 | .327 | **.04479** | .00564 | .04190 |
| 17563 S2 | .133 | .368 | **.04069** | .00644 | .03892 |

在这份实际记录和固定进度条件下，位置替换对当次raw需求的影响明显大于航向替换。short .375组也有相同量级关系；全部24组摘要见 [window-summary.csv](pose-shadow-v1/window-summary.csv)。

这里的“差值”不是跟踪误差改善。它衡量把实际位置输入换成同帧理想位置的敏感性，包含位置偏差和噪声的综合作用；不能拿它直接当作“定位噪声造成更新跳变的百分比”，也不能声称修改定位过滤器会改善闭环。固定旧进度时的新位置还可能与该进度不完全一致，这正是被限定的单输入干预范围。

虽然没有改参数，内部join-length搜索会响应新输入：baseline S1/S2的18/17次更新中，位置替换分别有1/2次改变所选join length；航向替换没有改变。26966/24240也没有改变。该离散选择已保留在逐更新表，不能把所有差值当成固定线性增益。

### tangent correction 的解析边界

没有实现删除tangent correction的候选。仅在**局部直线、小位置/航向误差、固定join length L**的线性化下，可以写出其纠偏削弱系数。设参考offset为s、u=s/L（0≤u≤1），位置权重 `w=1−10u³+15u⁴−6u⁵`，切向权重 `L*h=L*(u−6u³+8u⁴−3u⁵)`。生成路径减去ego位置并旋转到ego坐标后：

- 对横向位置误差的相对纠偏幅度是 `1−w = 10u³−15u⁴+6u⁵`。
- 对小航向误差、相对于无tangent correction时的纠偏幅度是 `1−h/u = 6u²−8u³+3u⁴`，u=0取连续极限0。

例如u=.5时，两系数为.5和.6875。该推导只说明当前数学构造在重接起点强制贴合ego位置/切向，会削弱近处纠偏。**真实弯道的reference offset s不等于controller的aim路径弧长**，还涉及路径重采样与join-length选择，因此没有把 `lookahead/L` 直接冒充实际窗口的精确反馈系数。本轮优先交付已逐值验证的位姿作用大小，未将该局部模型升级为控制实现或新验收依据。

大数据与源码副本：`/data/runs/b2d/controller/lateral-followup/pose-shadow-v1/`，包含全部2076个位置/航向/联合替换分支；以下文件完整保留所有生成分支：`per-update-pose-shadow.csv`、`shadow-trajectories.jsonl`。小摘要、验证和[源哈希](pose-shadow-v1/manifest.json)留在仓库，旧replay/shadow目录不覆盖。

```sh
/data/envs/carla/bin/python todos/2026-09-23-lateral-followup/diagnostics/pose_shadow.py \
  --run-root /data/runs/b2d/controller/turns-v1 \
  --windows todos/2026-09-23-tcp-controller/agents/lateral-evidence-v1/turn-windows.csv \
  --out /fresh/path/pose-shadow
```
