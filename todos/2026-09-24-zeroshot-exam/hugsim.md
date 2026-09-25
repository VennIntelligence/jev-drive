# Zero-shot 考试：openpilot 在 HUGSIM 上的 HD-Score（已停，只记数字）

状态: 停（2026-09-25 用户决定：闭环考试先不投入，等 closed-loop infra 验收；Alpamayo 部分未跑）
相关: [closed-loop infra 验收](../2026-09-25-closed-loop-infra-acceptance.md)、其中的 [HUGSIM 控制器验收](../2026-09-25-closed-loop-infra-acceptance/hugsim-controllers.md)

HUGSIM 是基于 3DGS（3D Gaussian Splatting）重建场景的闭环 benchmark，主指标 HD-Score（NC × DAC × 加权 TTC/Comfort，再乘 RC 即 route completion）。
64 个场景（nuScenes / Waymo / PandaSet / KITTI-360 各 16，easy–extreme 四档），每个 agent 各跑两种 controller：HUGSIM 官方 controller（official）
和我们的 fixed 跟踪器。openpilot 用 4 Hz dilated clock（rate study 选定），adapter 的倒车 bug 已修。
原先 owner 写的 pre-registration、rate study 和 adapter checklist 在 `research/results/hugsim-exam/`；bootstrap CI 和 failure analysis 没做，考试已停。

## 结果（逐场景均值，n = 64；逐场景表 `research/results/hugsim-exam/scored_{op,base}.csv`）

| agent | controller | HD-Score | RC | NC | DAC |
|---|---|---:|---:|---:|---:|
| openpilot Cinque | fixed | 0.278 | 0.349 | 0.791 | 0.914 |
| openpilot Lebowski | fixed | 0.251 | 0.290 | 0.819 | 0.918 |
| constant velocity | fixed | 0.292 | 0.340 | 0.734 | 0.935 |
| LTF | fixed | 0.279 | 0.433 | 0.495 | 0.941 |
| openpilot Cinque | official | 0.033 | 0.097 | 0.521 | 0.563 |
| openpilot Lebowski | official | 0.036 | 0.094 | 0.495 | 0.480 |
| constant velocity | official | 0.039 | 0.083 | 0.621 | 0.622 |
| LTF | official | 0.250 | 0.388 | 0.465 | 0.886 |

读法：controller 决定了几乎全部差异。official controller 下 openpilot 和 constant velocity 都接近 0（原地打转，见 checklist），
只有 HUGSIM 自带的 LTF 能用它；换成 fixed 跟踪器后四个 agent 挤在 0.25–0.29，openpilot 与 constant velocity 分不开。
所以这组数字不能用来比较模型，只说明 HUGSIM 闭环分数目前由 controller 和基础设施主导，这正是暂停闭环考试的理由。
infra 验收里的 fixed2（lqr-tracker-v2）通过了 held-out 验证，闭环恢复时应以它重跑。
