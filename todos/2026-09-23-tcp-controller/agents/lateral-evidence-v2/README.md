# 转弯窗口 telemetry 对照

`analyze.py` 读取已固定的 v1 几何窗口和原始 telemetry。`results-v1` 是首份保留计算：36 cases、45 windows、90 条全窗口/移动辅助指标。本次未启动仿真或修改运行源码。

依赖 NumPy，运行 `python3 analyze.py --out /fresh/output`。已有输出目录会被拒绝。v1 文件按本目录相对位置解析，运行数据默认为 `/data/runs/b2d/controller`；manifest 记录源文件和结果的 SHA-256。已封存 CSV/manifest 不因本说明翻译而修改。

主表为 `per-window.csv`；`pooled-descriptive.csv` 只是按时间样本合并的辅助表，可能被长时间停滞主导。`cases.csv` 记录实际配置、完成和碰撞；`all-formal-attempts.csv` 保留全部尝试，包括没有被选作路线结果的重试。选择规则为第一个 harness-finished attempt，否则最后一个。未完成转弯不会被删除，缺失输入显式记录。

- CTE 单位 m：将 truth 后轴 world XY 投影到共同原始中心线的最近线段，计算垂直有符号偏差，旧 v2 采用 CARLA 右正叉积符号；RMS/p95/max 反映误差幅度。这是独立诊断，不进入控制器。端点投影可能隐藏沿程误差，自交路线也可能存在歧义；本次局部窗口未发现路线自交歧义。
- 航向单位度：truth yaw 减去投影位置附近 5 m 中心弦方向，并折返到 ±180°。这是平滑道路切线误差，不是原生 TCP 按 90°归一化的 aim 角。
- steer 是发出的归一化控制量，不是实测车轮角。变化率用相邻连续帧差除以实际 `sim_time` 差，名义采样 20 Hz；首样本和跨缺帧记为缺失。窗口首帧可引用窗口外上一帧。统计只排除无法计算的变化率，并保留有效样本数。安全回退跳变可能超过普通控制的 2/s 限制，分析不裁剪。
- 幅值饱和按配置的 `max_steer`（默认 .8）、容差 1e-6 计算。`limited_fraction` 使用日志中的 `steer_limited`，也可能代表变化率限制；不能当作原生 TCP ±1 饱和比例。
- `all` 保留停车、反向、碰撞停留在内的全部窗口帧。`moving_ge_2mps` 只是正向速度 ≥2 m/s 的辅助子集，不是验收门槛，不能代替全窗口。
- entry 是加边界窗口前 5 m；post 是窗口后 10 m。恢复仅作诊断：首次离开核心后，在窗口末端后 10 m 以内，寻找连续 ≥.5 s、|CTE|≤.25 m 且 |航向误差|≤5° 的最早区间。未观测到记为删失，不算通过。首次恢复也不代表永久收敛，仍需查看 post RMS；这些阈值不改变正式验收规则。

G2 各配置共享 oracle 和 PI 纵向；正式 CARLA/TCP/pursuit 的纵向配置和交互 actor realization 可能不同。两种数据都使用 `policy none`，正式 TCP 是 oracle preset，不是模型的完整原生控制。不能据此推断模型优劣、官方得分或正式运行中的独立横向因果。

已有信号：max-PP 比 additive PP 改善两段 S 弯和 26966 p95，但 CARLA-PI 的 26966 RMS 仍更低；PP 的该弯误差没有触发幅值或变化率限制。Max-PP 在部分 S 弯触发变化率限制，因此缩短 lookahead 必须同时检验这项权衡。

下一轮根代理实际选择 `max(3 m, .375 s × speed)` 对照原 `max(3 m, .5 s × speed)`，只改时间系数，不改 floor 或提高 steer_rate。先前 `max(2.5 m, .4 s × speed)` 只是已被收窄的建议，不是执行参数。三路线为 26966@8、17563@6、24240@8，共四个既定窗口、两个配置、六 case。新工具 `../analyze_turns.py` 用 `--run-root` 读取新运行根目录；旧 `analyze.py` 和封存结果继续保留。详细证据见 [横向审计](../lateral-audit.md)。

新 turns 工具显式改用与 validator 一致的左正 CTE，并输出有符号均值/峰值，旧表不回写。26966 右弯的 validator 正残差在弯外侧，下一轮假设是弯外残差/出弯滞后，不是内侧切弯。新 helper 的 SHA-256 与离线验证记录见 `../analyze-turns-verification.json`。
