# FLOW1 state (2026-10-11 ~01:30 CST)

lane 已收尾。训练链和读链都 DONE（box: `$DATA_DIR/runs/flowhead/flow1/{chain,read,report,div,nc,geom.npz}`）。
- 结果：`results/flow1.md`（结论）、`results/tables.md`、`results/summary.json`（从 box `flow1/report/` 拷来）。决策 246。
- 读法：FM 出界 -4.98 pp 但 EPDMS -1.10、EP/LK/EC 降，不是候选，支路关闭；机制读数方向相反。
- 追加：用户批准的 flip 表（fixed/broken，每 seed 与合计）在 `flow1.py report` 的 4c 节，`flow1.md` 里有摘要；描述性，不进判定。
- 没做：补记 A.3 的 HUGSIM 64（条件成立：FM 对 RG navhard combined / stage 2 CI > 0），因为 head 没有 HUGSIM 服务路径，建它要改 bench（select_4 之后的 head 服务选项 + 等价闸门）；A.4 已不过，HUGSIM 只能再添负格。
- 重跑读表：box 上 `flow_read_chain.sh` 可续跑；只重做表：`flow1-report` 池作业（`flow1.py report`，op-train env）。
- 若恢复：不要再加训练/新臂；除非有人要闭环读数，才建 head 服务路径。
