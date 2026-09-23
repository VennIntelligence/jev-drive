# 首例前负面证据：真实历史状态下的Hermite shadow

该附录不改变冻结门槛。独立agent在原baseline .5三例1380帧上复现旧控制逐值0误差，保持历史motion/新轨迹原样，仅改变aim插值的shadow显示：

| 固定窗口 | linear raw-rate P95 /s | Hermite raw-rate P95 /s | 变化 |
|---|---:|---:|---:|
| 26966/1 | 0.800106 | 0.980592 | +22.56% |
| 24240/1 | 0.218380 | 0.251735 | +15.27% |
| 17563/1 | 2.196655 | 2.349625 | +6.96% |
| 17563/2 | 2.678289 | 2.801251 | +4.59% |

四窗均上升，更新项RMS亦小升。理想无噪声圆轨迹扫描验证了几何采样纹波，但不代表它是历史真实更新跳变的全部来源，更不能预报Hermite闭环获益。shadow保留旧motion，不反馈新控制至plant，与将要执行的闭环含义不同。根代理仍执行已批准的唯一六例检验反馈下行为；若失败，不调阈值或换弱基线。

[具体方法与边界](diagnostics/README.md)、[原摘要](diagnostics/interpolation-shadow-v1/window-summary.csv)、[原manifest](diagnostics/interpolation-shadow-v1/manifest.json)。完整逐帧/源码在`/data/runs/b2d/controller/lateral-followup/interpolation-shadow-v1`。本附录据已落盘CSV核对数值，未重跑任何控制或CARLA。
