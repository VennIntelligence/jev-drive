# navtest warp 帧更正：状态（2026-10-11）

lane：把读错帧的 navtest own-plan 读数（`bd4_g3.plans` 在 GIMM 帧上读 `lb_navtest`，checkpoint 是 warp 帧训练的）逐条更正。
只重读、不训练、不做新实验。总表（逐条 old -> new、判定）：[results/navtest_warp.md](../results/navtest_warp.md)。

## 注意：今晚有另一个 session 在做同一件事

00:57–00:59 JST 另一个 session 改了并提交了 `research/decisions.md` 与 `decisions/{229,230,232,233,234,244}.md`（4c26af7d），数字与本
lane 的重读表一致（我逐项对过 232 / 234 / 244 的主要数字）。本 lane 因此没有再动 decisions 文件，只写结果文档与表。继续之前先
`git fetch && git log --oneline -10 -- experiments/body1 research/decisions research/body1`，看对方又改了什么，避免重复改。

## 已完成

- box 上两条重读链都 DONE（`$DATA_DIR/runs/body1/navtest_warp/chain/DONE` 23:16，`chain2/DONE` 23:53 CST），没有 ERROR，没有重跑。
- 小表拷到 `results/navtest_warp/`（95 个文件，rsync 排除 `chain*` 与 > 400 KB），已提交。dump（parquet / npy）留在 box。
- `results/navtest_warp.md`：出错原因、reader 的 identity check（CPU forward 对 bench 存档 plan：W2 141 / 97 / 111 / 83 对
  141 / 97 / 110 / 84）、第 229 / 230 / 232 / 233 / 234 / 236 / 244 条逐条 old -> new -> 判定、其他 consumer 列表。
- `results/sdrop.md`、`results/route_pilot.md` 原位更正（旧数写在旁边，表里留了 superseded 行）。提交 2eaabf05，已 push。
- decisions 229 / 230 / 232 / 233 / 234 / 244：另一个 session 完成（4c26af7d）。
- 第 245 条（`research/corridor/index.html` 第七节、`experiments/corridor/results/head1b/c_tables.md` 的 W2 99 / 120 对 141 / 150）：
  **不受影响**。`head1_pilot_report.py --widening` 的 plan 来自 `jevdrive.bench` 的存档 plan（`turn_oracle.pf(spec)`，checkpoint 自己的帧），
  不经过 `bd4_g3.plans`；它的基线 141 与 warp 重读的 `P2H10S-P-s0` 141 相同。没有改这三个文件。

## 关键结论（报告用）

- 244：两个登记问题的判定都不变；W2 约翻倍（base 84 / 79，S 116 / 118，noA 111 / 114，noB 85 / 82，noC 97 / 98），连续外移约减半
  （S +0.123 [+0.085, +0.163]）；noC 的通用规则标签 not needed -> carries（48%）；三句旁支不成立（noB、noC 的 navtest 出界率下降，
  navtest lead 后变短 0.3–0.4%）。
- 234：gate 判定不变（138 对 110，原 67 对 54）；navtest 外移 0.09–0.15 m（原 0.18–0.27）；navtest 弧长 0.9993（原 0.9915），
  「全量还有第二条线」撤回；base 的 W2 seed spread 是 12 个 token（原 4），+5 的容差在 spread 之内。
- 232：**G3 (b) seed 0 翻转为过**（135 对 140，原 149 对 148）；seed 3 的 (e) 也过（1.0001，原 0.9949）；上一配方 navtest 弧长
  0.9985 / 0.9980（原 0.9905）。当时停臂所依据的读数是 reader 的错。是否补登记闭环由 main / 用户定。

## 2026-10-11 01:30 JST 更新：未完成项已做完

下面「未完成」第 1 项里的文档都已原位更正并 push（两个 session 合起来）：`shape_pilot.md`、`loss_g3.md`（d3d307e6），
`progress_diagnosis.md`、`shape_closed_loop.md`（41aa5c99；`loss_closed_loop.md` 里没有按 base plan 分组的表，不受影响），
lane README、`experiments/INDEX.md` 的 body1 行、`research/body1/index.html`、`navtest_warp.md` 的「已更正页面」与「遗留问题」两节（1c04f85b），
预登记的更正行与第 236 条（8d374db6），S-DROP 预登记末尾的更正行。第 3 项的 grep 残留只剩：标明「原 / first read / superseded」的旧数，
`results/{shape,route,sdrop}/` 下保留为旧读数的生成表。三张旧图（`figs/shape/shape_gate.png`、`figs/prog/ol_arc.png` 左图、
`figs/prog/cl_progress.png`）已在同一夜用 warp 表重画，各页的「未重画」备注已删。
遗留给 main 的两点：(1) 第 232 条的登记闭环读数是否补做；(2) `prog_cl.py` 的 decision-0 对齐检查在 warp 帧 plan 上是 0.909（原 1.010），
即 AlpaSim 里第一个 served plan 比 bench 评分的 plan 短 9%，原因未查（`navtest_warp.md` 末节）。

## 未完成（按优先级；写于 01:01，已由上面的更新取代）

1. 其余引用旧 navtest own-plan 数字的文档还没有原位更正（只在 `navtest_warp.md` 里给了新数）：
   - `results/shape_pilot.md`（第 232 条的结果页：G3 (b)、(e) 的表与「(b) on seed 0 is the miss」一段、pilot 的 navtest 弧长行）
   - `results/loss_g3.md`（第 229 条：G3 (b)）、`results/progress_diagnosis.md`（第 230 条：navtest 开环弧长、分组份额）
   - `results/shape_closed_loop.md`、`results/loss_closed_loop.md`（按 base plan 分组的表；新表 `navtest_warp/{shape_cl,prog_cl}/`）
   - `research/body1/index.html`（中文页）、`experiments/body1/README.md`（加一行指向 `results/navtest_warp.md`；现在只能从
     `sdrop.md` / `route_pilot.md` / decisions 走到它）、`experiments/INDEX.md` 的 body1 行、预登记里结果后补的 status 段。
   - 第 236 条里「SH30 的 own-plan agent 接触率 0.0327 对 0.0328，`P2H10S` 0.0234」：没查它读的是哪个集合（`results/sh30s/` 有
     val 与 navtest 两套；val 不受影响）。新表 `navtest_warp/sh30s/`。
   做法：对照 `navtest_warp.md` 的表与 `results/<dir>/` 对 `results/navtest_warp/<dir>/` 的同名文件，原位换数、旧数写在旁边。
2. `results/sdrop/summary.md` 等生成文件保持为旧读数（superseded 记录，`sdrop.md` 页首已说明），没有覆盖。
3. grep 旧数字的残留：`grep -rn "58 / 65\|45 / 46\|149 对 148\|0\.9905\|67 对 54" research experiments --include=*.md --include=*.html`。

## 怎么继续

- box：`ssh -o ControlPath=none autodl`，`$DATA_DIR/runs/body1/navtest_warp/`；本 lane 没有在 box checkout 留文件，没有提交新 job。
- Mac 上对比新旧表：`.venv/bin/python`（有 pandas）；新旧 csv schema 相同，`sdrop/contrasts.csv` 可按
  board / subset / metric / arm / vs merge。
- git：只按路径 stage，`git fetch && git merge --ff-only origin/main` 后提交并立刻 push。
