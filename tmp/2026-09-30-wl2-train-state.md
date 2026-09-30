# WL-2 训练与读数：状态（2026-09-30 18:2x，box 缩到 2 卡前写）

**训练已全部完成，结果都在 box 盘上**（`~/data/runs/wl2/model/<arm>/seed<s>/<stamp>/{model.pt,preds.npz,forks.parquet,curve.json}`）：
A、B、Bh、W 各 seed 0-4，Bhc / Bs / T 各 seed 0-2，Ax / Bx 各 fold 0-4，vrep（WL-1 冻结预测器）seed 0-2。
lane 的 DONE 文件：b、d、e、f、g、h 都有；lane a、c 是我手动按 PID 停掉后拆给别的 lane 的（a 的 Ax/Bx 因与 T 同卡太慢，c 的第二阶段与 e 重复），
它们的 arm x seed 全部有 model.pt，不缺任何一个。没有训练需要 GPU。

**唯一没跑完的是读数（`wl2_report`，2 000 次 bootstrap，约 30-40 min，GPU 只用来拟合 critic 的小 MLP）**。box 重启后一行命令重跑（任意空闲卡）：

    ssh autodl 'cd ~/data/jev-drive && scripts/tmux_run.sh wl2-report bash -c "CUDA_VISIBLE_DEVICES=<gpu> WL2_RESULTS=\$HOME/data/runs/wl2/results taskset -c <cores> .venv/bin/python -m jevdrive.wl2_report --n-boot 2000"'

输出在 `~/data/runs/wl2/results/`（`c1_main.csv`、`c1_wl1eval.csv`、`c2_c3.csv`、`slow_shift.csv`、`c4_flips.csv`、`ego_error_*.csv`、`xfit.json`、`results.json`、`curves.csv`），
之后 `scp` 到 `research/results/wl2/results/`，跑 `python scripts/make_wl2_figs.py`。
读数代码已在 WL-1 干跑数据上复现（`--validate-wl1`，最大差 5e-5，只是 csv 四位舍入）。

## 更新（18:3x，box 无 GPU 后）

- 读数进程已随 box 变动死掉，tmux server 也没了；`~/data/runs/wl2/results/` 里只有 `c1_main.csv`（主评测集的 C1 / C1c 已算完）和 `curves.csv`、`ego_error_main.csv`、`stage2_sanity.json`，其余（`c1_wl1eval.csv`、`c2_c3.csv`、`slow_shift.csv`、`c4_flips.csv`、`xfit.json`）要重跑。
- 重跑需要 1 张 GPU（critic 的 MLP 拟合，约 200 个小拟合，几分钟；C4 另要一次前向），bootstrap 是 CPU，1 张卡总共约 30-40 min。没有别的事需要 GPU。
- 还没有任何 C1-C4 判格写进 prereg 执行记录、decisions.md 或 research/：这些全部等这次重跑的输出（判据与代码已固定，没有看结果改动）。
- 已知的一处偏离要写进执行记录：主评测 x+ 分叉点 247 个里只有 210 个能取到 8 步历史（fork tick 在 run 开始后不足 1.4 s，或被 render / pose gate 剔除），其余无法读数，全部读数只在这 210 个上；不是看结果后的选择。
- 冷启动后 GPU 不在时不用等：`--validate-wl1` 和 T 之外的训练都已经结束。
