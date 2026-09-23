# 固定 k 位姿补偿：独立绘图

`plot_pose.py` 仅读取冻结 `analyze_pose.py` 产物，不重算或修改验收。使用共享 `research/plot_style.py`，脚本与样式实际字节随每次图归档；运行前冻结 SHA 另见最终 `plot-freeze-v2.json`（v1 为首例前术语审查历史，原字节在 plot-frozen-v1）。

```sh
/data/envs/carla/bin/python todos/2026-09-23-lateral-followup/pose-followup/analysis/plot_pose.py \
  --analysis /absolute/path/to/closed-analysis \
  --out /absolute/path/to/new-figures-edition
```

输出目录必须不存在。三条路线 26966/24240/17563 的全程图与四个固定 pad 窗口图全部生成，每图包含 CTE、实际/参考速度、emitted steer-rate、横向 acceleration/physical jerk、pose position norm、pose−truth 左向有符号差、真实航向跟踪误差。另存六种量的四窗汇总图。每图 6.875in、STIX 8–9pt、灰色基线/蓝色候选、300dpi PNG 和矢量 PDF；无大标题，面板文字英文。

`plot-data.csv` 保留输入 `frames.csv` 全部原始列/行次序；`window-plot-data.csv` 保留四窗原样本及固定界限；`summary-plot-data.csv` 直接复制 `metrics.csv` 数值而不创造新阈值。启动、瞬态、停车、故障、低速帧均保留。时间轴为各臂首个选中样本的相对时间；不插值对齐两臂，也不重采样。不连续帧/时间用 NaN 断线，缺值不填0；缺失情况写入 `plot-notes.json`。全部结果只用于展示，最终判定来自原 `required-conditions.json`，即使图看似更好也不能代替判定。

图注必须明确：pose−truth 的左向量使用 truth yaw 的车体左向投影，独立于路径 CTE；物理 jerk 先对 world acceleration 求导再投当前 body，不能用 steer-rate 替代；速度点线为独立参考。四窗汇总不加权抵消失败，亦不把原协议未规定的逐窗 pose/jerk 非恶化悄悄加入必要条件。

夹具图仅检验数据保留与版式，不具备闭环候选资格。实际六例闭合后另建目录输出，不覆盖夹具或旧阶段图。
