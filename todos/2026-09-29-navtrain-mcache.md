# navtrain 全量 PDM metric cache（2026-09-29 至 09-30）

## 做了什么

给 openpilot adaptation round 2 的 S_jev 打分器和 leaderboard lane 的重打分预先算好 NAVSIM v1.1 navtrain 的全量 metric cache。CPU only，没有用 GPU。

- 位置：`$DATA_DIR/runs/navsim/metric_cache/v1_navtrain`（box 上 `/root/autodl-tmp/ujs/runs/navsim/metric_cache/v1_navtrain`）
- 版本与配置：和 op-lb 的 `v1_navtrain_oplb` 完全一致，navsim-v1.1 devkit 原样运行（`third_party/navsim-v1.1`，env `navsim1`），`train_test_split=navtrain`（1192 个 log、103 288 个 token），`OPENBLAS_CORETYPE=Haswell`，环境变量同 `scripts/navsim_zs_score.sh`。
- 脚本：`scripts/navtrain_mcache.sh pilot1|pilot10|full`（分阶段，跑在 tmux `jev`，run dir `$DATA_DIR/runs/navtrain_mcache/<stage>-<ts>/` 有 log.txt、events.jsonl、DONE/ERROR），检查脚本 `scripts/navtrain_mcache_check.py`。
- 开工前盘点：已有 v1_navtest 12 146、v2_navtest 12 146、v2_navhard_two_stage 5 912、v1_e6sub 20 000、v1_navtrain_oplb 3 000，都只是子集，没有完整 navtrain，所以新建目录，旧目录一个文件都没动、没删。

## 大小与数量

| 项 | 值 |
|:--|:--|
| navtrain token | 103 288 / 103 288，缺 0 |
| 目录内 pkl 总数 | 106 630 |
| 磁盘 | 36 GB（约 0.35 MB 每个 scenario） |
| 多出的 3 342 个 | pilot10 的 log 过滤在 tokens=null 时把这 10 个 log 里的全部 scenario 都算了，其中不属于 navtrain token 表的 3 342 个留在目录里；无害（scorer 按 token 过滤），按规则没有删 |

## 用时

| 阶段 | 结果 |
|:--|:--|
| pilot1 | 1 个 log，154 scenario，单 worker 151 s（约 1 s / scenario） |
| pilot10 | 10 个 log，3 933 scenario，1 568 s（最大的 log 1 105 scenario 单 worker 顶到最后） |
| full 第一段 | 56 workers，97 min 做到 88 259 个，之后被 main 因 PID 上限叫停（按精确 PID 停，已有文件全保留，最后 4 分钟写的 523 个文件解压检查全好） |
| full 第二段 | 16 workers + 各种线程环境变量设 1，续跑 36 379 s（10.1 h），补完剩下约 15 k 个 |
| 合计 | 约 11.8 h |

第二段比 pilot 的单 worker 速度慢 20 多倍（约 2 k 个 / 小时），原因是 nice 19 加上这些核被 Cosmos / op-lb 的 CARLA 与打分抢满；估计的 1 h 没有兑现。没有做 profiling，因为瓶颈是别的 lane 占核，不是代码热点。如果以后要重建，需要更多核或者不用 nice 19。

## Sanity

| 检查 | 结果 |
|:--|:--|
| 文件完整性 | 106 630 个 lzma 全部解压成功，0 个空文件 |
| token 覆盖 | navtrain 表 103 288 个全在 |
| PDM-Closed 参考轨迹 | 300 个抽样：每条 51 个点，最大逐点步长 0 至 1.55 m（0.1 s，最快约 15 m/s，合理），5 s 长度 0 至 74 m，起始车速 0 至 13.8 m/s，route lane 4 至 207 条 |
| 与 `v1_navtrain_oplb` 对比 | 共同的 3 000 个 token：2 996 个逐位一致；4 个 PDM-Closed 轨迹不同（最大偏差 0.11 m 和 18 m 各有一个），route lane 集合一致。PDM-Closed 自身在近平局的 proposal 选择上有不确定性，不是 bug，属于已知量级很小的一类差异 |

## 其它

- 调度表行 `navtrain-mcache` 已 finish；tmux 窗口已关。
- box 已 pull 到最新 main。
