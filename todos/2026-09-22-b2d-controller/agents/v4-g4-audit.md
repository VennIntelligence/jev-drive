# v4开发封存与正式G4汇总准备

2026-09-23 JST；没有操作CARLA、git或控制器核心，没有覆盖任何结果。

## v4闭合证据

`/data/runs/b2d/controller/development-v4`仅一个end事件，18例completed、17例全部gate通过。原server94记录PID188112已不存在后才执行索引。
[完整索引](../results/development-v4-file-index.json)：173文件、31,171,941字节。[summary与GPU逐字节副本](../results/development-v4-closed/README.md)只保留事实，不声明正式路线资格。

最终PI图表[附加源码审计](../results/pi-v4-figures-source-audit/README.md)核对248个source与14个声明output哈希；两份Python绘图源码延迟归档，但字节匹配原绘图report中已记录的哈希。完整figure目录含补充分析另有全文件索引。原图目录未写入或覆盖。

## G4 helper

实现：`scripts/b2d_controller_g4.py`；验证：8项`test_b2d_controller_g4.py`测试通过，结果和源文件hash在[验证记录](../results/g4-helper-validation/verification.json)，完整[test log](../results/g4-helper-validation/tests.log)。将只有一组smoke的输入故意指定完整Dev10，CLI正确返回incomplete_comparison，未生成整组验收条件。

正式运行示例（必须使用新输出目录）：

```bash
/data/envs/carla/bin/python scripts/b2d_controller_g4.py \
  --campaign /data/runs/b2d/controller/formal-v4 \
  --routes /data/third_party/Bench2Drive/leaderboard/data/drivetransformer_bench2drive_dev10.xml \
  --preset-configs todos/2026-09-22-b2d-controller/results/v4-freeze/preset-configs.json \
  --out todos/2026-09-22-b2d-controller/results/formal-v4-g4-01
```

每个seed必须carla/tcp/pursuit三组都有全部所需route的最终官方record及保存report，两个seed均齐后才有整体条件。错误/基础设施缺记录、人为cap保持不完整；官方TickRuntime是真实失败，保留分母。选择继承既有first-harness-finished attempt规则，完整失败尝试仍在JSON，不挑最好一次。

- 固定参考CARLA横向+共享PI(.5,.25)+additive，在G2预先选定；TCP vendor另外报告，不能根据Dev10结果重选最有利参考。
- 实际campaign归档配置字节必须匹配冻结preset配置，记录原source commit、report及输入/source hash，错误配置不得标成预定对照。
- driving_completed使用completion≥100。subset_diagnostic_sr对所选官方record要求Completed/Perfect且除了min_speed外无非空infraction列表，分母为所需子集大小，不使用官方脚本写死的220；不是官方full220成绩。
- completion按两个seed分别不降；横向20%、速度增加≤.1等主数值按两个seed等权路线均值检查。每seed相同条件也完整报告为诊断，避免暗增“每seed横向都改善20%”门槛。
- 全程真值CTE用于主横向比较；首次碰撞前独立报告，缺失不替代成全程或0。速度使用report的trajectory-reference误差，明确它不是独立固定巡航/官方舒适性。
- 非零blocked/deviation需要已有证据人工归因。可用`--adjudications`传入`{"pursuit-seed0/ROUTE":{"vehicle_blocked":"controller","route_dev":"external"}}`；只有controller/external可信标签参与控制导致事件的比较，其余为unknown。不自动由原始事件数推断因果。
- 即使这些G4条件满足，helper仍不宣布new default，其他既有G1–G3、复跑波动、确认和最终采纳由根代理按完整协议审查。

当前真实正式run可能尚未结束，本文件不包含正式G4成绩，也不根据部分组宣布赢家。外部失败归因、指标口径或配置缺项会显式使结果待核实。
