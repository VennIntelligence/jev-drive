# Task 10 中途交接（2026-09-25 东京时间）

按 `/data/runs/b2d/tfv6-w2/bus/STOP-TASK-10.md` 停止。已中断我启动的 expert L1 恢复进程和 7200 端口 CARLA；没有 Task 10 的 tmux 窗口。保留了所有已经完成的 case 和被中断的原始目录。**Task 10 尚未完成；不要把此交接当作最终 controller-scorecard。**

## 已完成、冻结与数据位置

- 工作树 `/home/ujs/mycode/jev-drive-w2`，分支 `w2-tfv6`。规则是 `controller-eval-rules.md`，操作定义在冻结提交 `21cb250` 的 `controller-eval-protocol.md`；评判阈值、8 条 held-out 路线、5 个扰动种子均不得再根据结果修改。
- 实现和恢复脚本已提交：`eb48a31`、`8576851`、`2b5d702`、`a792b80`。核心脚本在 `scripts/b2d_controller_eval_{l1_score,l1_summary,l1_canonical,l1_recover,l23_score,campaign,interface_score}.py`，TCP 实际驾驶封装在 `scripts/b2d_tcp_eval_agent.py`。A/B 与 LEAD 作者控制公式做过合成输入对照；冻结前的 dev pilot 和测试见操作协议。
- 8 条路线的特权真值位姿 expert 参考全部有效，映射文件 `/data/runs/b2d/controller-eval/references/reference-traces.json`。
- Route-oracle L1 **160/160 有效**，规范化目录 `/data/runs/b2d/controller-eval/l1-oracle-canonical/`，内有 `manifest.json` 指向原始尝试。两个 Town13 零 tick CARLA 超时原始记录被保留。只对 oracle 作过临时评分：`/data/runs/b2d/controller-eval/l1-oracle-provisional.csv`；primary 中位数 A 3.701、B 3.265、C 1.039、D 1.028。C/D 对 B 的路线整组 bootstrap 95% CI 分别约为 [-2.286,-1.674]、[-2.285,-1.675]。这些不是最终判定；expert、L2、L3 仍待测。
- Expert L1 目前 **149/160 有效**：原始批 `/data/runs/b2d/controller-eval/l1-expert/` 中 142 例，恢复批 `/data/runs/b2d/controller-eval/l1-expert-recovery/` 中 7 例（p01 C/D、p02 A/B/C/D、p03 A）。原始批在 3364/C/p01 零 tick CARLA render 崩溃；该例第二次尝试有效。收到停止指令时 p03 B 正在启动；`l1-expert-recovery/p03-B-attempt-1` **没有 route_start、validation.json 或受控 tick**，仅有启动文件，因人为停止而未完成。没有重跑任何有效驾驶结果。

## 继续运行（先检查无其他 CARLA 占用）

在工作树执行，优先完成 L1 expert 剩余 11 例。新输出目录可使被中断的 p03 B 启动尝试原样保留；恢复脚本只跳过有完整 telemetry 的有效案例。Town13 每例单独启动 CARLA，基础设施故障每例最多三次：

```bash
/data/envs/tfv6/bin/python scripts/b2d_controller_eval_l1_recover.py \
  --reference expert \
  --sources /data/runs/b2d/controller-eval/l1-expert \
    /data/runs/b2d/controller-eval/l1-expert-recovery/p0*-*-attempt-* \
  --out /data/runs/b2d/controller-eval/l1-expert-recovery2 --server-index 104
```

完成后核对 160 个**唯一有效** case，再构建 symlink-only 规范视图和计分：

```bash
python3 scripts/b2d_controller_eval_l1_canonical.py \
  --out /data/runs/b2d/controller-eval/l1-expert-canonical \
  --sources /data/runs/b2d/controller-eval/l1-expert \
    /data/runs/b2d/controller-eval/l1-expert-recovery/p0*-*-attempt-* \
    /data/runs/b2d/controller-eval/l1-expert-recovery2/p0*-*-attempt-*
/data/envs/tfv6/bin/python scripts/b2d_controller_eval_l1_score.py \
  --oracle /data/runs/b2d/controller-eval/l1-oracle-canonical \
  --expert /data/runs/b2d/controller-eval/l1-expert-canonical \
  --references /data/runs/b2d/controller-eval/references/reference-traces.json \
  --cruises todos/2026-09-23-tfv6-controller/controller-eval/l1-cruises.json \
  --out todos/2026-09-23-tfv6-controller/controller-eval/results/l1-cases.csv
/data/envs/tfv6/bin/python scripts/b2d_controller_eval_l1_summary.py \
  --cases todos/2026-09-23-tfv6-controller/controller-eval/results/l1-cases.csv \
  --out todos/2026-09-23-tfv6-controller/controller-eval/results/l1-summary.json
```

L2/L3 驾驶尚未开始。先用 dev 路线 24240 对 `b2d_tcp_eval_agent.py` 做 TCP N/C 冒烟检查，再分 planner 启动正式活动；每例运行后立即检查不变量，驾驶/runner bug 就停批，基础设施最多重试三次，不能重跑有效驾驶结果：

```bash
/data/envs/tfv6/bin/python scripts/b2d_controller_eval_campaign.py \
  --planner tfv6 --out /data/runs/b2d/controller-eval/l23-tfv6 --server-index 110
/data/envs/tfv6/bin/python scripts/b2d_controller_eval_campaign.py \
  --planner tcp --out /data/runs/b2d/controller-eval/l23-tcp --server-index 110
/data/envs/tfv6/bin/python scripts/b2d_controller_eval_l23_score.py \
  --tfv6 /data/runs/b2d/controller-eval/l23-tfv6 \
  --tcp /data/runs/b2d/controller-eval/l23-tcp \
  --out todos/2026-09-23-tfv6-controller/controller-eval/results
```

接口扰动 L1 也尚未跑。冻结子集是 `controller-eval/l1-interface.xml` 的 24240、17563 与四条 held-out 路线；模式 `short_2s`、`sparse_5s`、`stop_jitter`，每个 5 seed × 4 controller。还需生成两个 dev expert 参考、dev nominal run；`scripts/b2d_controller_eval_references.py --expected 2` 支持参考映射，`scripts/b2d_controller_eval_interface_score.py` 聚合固定子集。具体参数在冻结协议，不得改阈值。

## 已知问题与操作注意

- Town13 在同一 CARLA 进程连续多例时曾两次出现 render-thread 卡死、进程消失及 90 秒 RPC timeout；都是零 tick，未形成驾驶结果。单例新 CARLA 进程已稳定完成 oracle 其余 18 例及 expert 已恢复的 7 例。`l1_recover.py` 仅把零 tick 的 90 秒 timeout / setup error 认作可重试基础设施；其他异常立即停止。
- TCP Task 10 wrapper 和完整 L2/L3 campaign 尚无 CARLA 实测，不能声称有效。TCP native `only_traj` 与 A/B/C/D 的同帧 shadow 都由 wrapper 记录；正式驾驶前需用 dev smoke 检查 model forward、native PID、frame 对齐、实际选中 control。
- Expert 轨迹由特权真值位姿的 C 类 `prod-truth` 控制器驾驶生成，非 LEAD 自带 expert；这可能偏向 C/D，最终报告须明确，并与独立 route-oracle 结果分开报告。
- L2 scorer 已支持碰撞过早导致的缺失 plan 指标：L3 官方结果仍保留，L2 指标记为缺失，不能据此判通过。最终判定按冻结协议的路线整组 bootstrap 和两个 planner 的 L3 护栏执行。
