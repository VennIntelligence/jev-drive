# D3：查清 W2b 剩下的两个失败模式

状态：open，2026-09-24。上游：[report-w2b.md](report-w2b.md)、`research/decisions.md` 第 31 条。这一轮仍然**只诊断，不改控制器**。

W2b 修掉静止切向之后，C 相对 B 的损失主要来自两处：保留集 2084、27529 的 route deviation（车离开了指定路线分支），以及 Dev10 2091 的起步停滞，一直停到 4000 tick 上限。两处都要回答同一个问题：**是 planner 的问题（TFv6 规划错了），是 controller 的问题（规划没错、没跟住），还是两者通过闭环耦合出来的？**

## 已知的线索

- D2 录像（27529/0）：C 比 B 更早、更快地穿过路口，这时 C 的局部 route 和 waypoint 已经是直线，steer 接近 0；而 B 到同一个路口时，route 是朝右弯的。
- D1、W2 的机制表（2091）：TFv6 在静止时给出的首段隐含速度只有 0.07–0.16 m/s，8 点平均约 2 m/s。C 按首段判定期望速度，低于阈值就一直 hold 刹车；B 用 `2·‖wp1 − wp3‖` 算期望速度，同一时刻已经在给油门。
- 作者 agent 的输入链：GPS 经过 Kalman filter 得到滤波位置（运动预测里**用到了实际执行的 control**，`lead/common/base_agent.py` 的 `kalman_filter.step(..., control=self.control)`）；滤波位置驱动 RoutePlanner 的 target point pop（`tp_distances`、自适应 pop 距离、远点跳过，`lead/inference/config_closed_loop.py:45–55`）；target point 和 command 再输入 TFv6。所以换控制器可能通过两条路径改变模型看到的导航目标：一是到达路口时的时机和速度不同，二是 Kalman 预测里的 control 不同。

## 候选机制（事先列出，由证据裁决，不是结论）

Route deviation：
- **R1 目标点提前 pop**：C 更快，或者 Kalman 预测把位置推前了，target point 在路口前就跳到了转弯之后，TFv6 看到的目标变成直行。
- **R2 规划本身错了**：target point 和 command 都正确，TFv6 仍然规划直行，而且在 B 的轨迹上同一情境也会这样，只是 B 没有走到那个情境。
- **R3 跟踪失败**：TFv6 规划了转弯，C 没有转过去，例如速度太快、横向跟不上、steer 限速。
- **R4 其他**：交通、停车规则、creep 之类的交互。

起步停滞：
- **S1 起步律**：TFv6 的中远端 waypoint 要求车走，但首段速度低于 C 的 hold 阈值。
- **S2 真实停车理由**：前车、红灯、行人、路口让行，模型让停是对的，只是 B 走了（B 反而更危险，或者更幸运）。
- **S3 后处理交互**：creep、force-move、stop sign 和 C 的输出发生冲突。

## D3a：只用现有日志（零 CARLA）

1. **全量扫描**：在 W2b 的 96 个 C/D case 和 W2 的 96 个 A/B case 里，列出所有 route deviation 和所有 ≥ 5 s 的停滞段。给出位置（地图坐标、离最近路口的距离）、各臂各 seed 的情况，确认这两个模式是否只出现在这几条路线上。
2. **route deviation 的时间线**（2084、27529，所有出事的 seed，以及同 seed 的 B）：在事件前 10 s 内，逐 tick 把 TFv6 的 route 和 waypoint 放回世界坐标，分别算它们到**正确分支**和**错误分支**（用 evaluator 的 global plan）的距离，找出规划第一次偏向错误分支的 tick。同时记录车速、到路口的距离、steer 和 steer 限幅。只要 target point 和 command 在现有日志里能还原，也一起给出；还原不了就写明，由 D3b 补记录。
3. **起步停滞的时间线**（2091 的所有 C/D seed，以及同 seed 的 B、A）：逐段给出 TFv6 的首段速度、8 点平均速度、target speed、C 的期望速度和 reason、B shadow 的期望速度和 control、后处理有没有实际改写 control。
4. **开环反事实**：在 C 的停滞段上，用同样的输入离线重算 C 的纵向输出，比较几种期望速度定义（首段、`2·‖wp1 − wp3‖`、8 点平均），看在哪种定义下 C 会起步。这只说明控制律对同一输入的响应，不代表闭环结果。

## D3b：带补充记录的定点重跑与因子实验（D3a 完成后自动进行，约 2.5 小时预算）

对 (2084, 27529, 2091) × seed {0, 1, 2} 这 9 组：

1. **补充记录重跑**：C 和 B 各跑一次，另外逐 tick 记录 target point（所有 `tp_distances`）和 command、Kalman 滤波位置与 GPS 原始位置、RoutePlanner 剩余路点与 pop 事件、交通灯状态、最近 8 个 actor。每个 case 都开 recorder。
2. **2×2 因子实验**：新增两个仅用于诊断的混合臂，E = C 的 steer + B 的 throttle/brake，F = B 的 steer + C 的 throttle/brake；混合之后，作者后处理和 brake 互锁按原规则执行。和上面的 B、C 一起，构成横向 {B, C} × 纵向 {B, C}。这样可以判断失败是跟着横向走，还是跟着纵向走。
3. **Kalman 隔离**（只对出现 R1 迹象的组）：C 臂跑一次，把 Kalman 预测里的 control 换成 B 在同一 tick 的 shadow control，只改滤波器的输入，不改执行的 control。如果 deviation 消失，就说明是 Kalman 路径导致的。这一项会改动第三方代码的输入，只能以包装的方式注入，不修改 LEAD 源码；做不到就跳过并写明原因。
4. 每组对关键时间窗渲染追车视频，叠加 TFv6 的规划、target point 和 command，截 contact sheet。

混合臂和 Kalman 隔离都只用于诊断，不进入任何正式对照。每个 case 都检查 W2b 的不变量 I1–I3，外加一条：混合臂的实际 control 必须等于按混合规则拼出来的 shadow 结果。

## 产出

`diagnosis-d3.md`（中文，表格加图，视频所见和推断分开写），以及 `results/diagnosis/d3/` 下的 CSV 和 contact sheet。结尾对每个模式给出候选机制的裁决：支持、排除、或者不确定，并说明依据；最后列出下一步最值得做的一件事，只提建议，不去实施。
