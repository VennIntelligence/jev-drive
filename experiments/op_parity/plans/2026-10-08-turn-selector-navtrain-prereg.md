# op_parity turn selector navtrain 预登记：用 navtrain held-out 标签（28 323 token）训练的非特权 selector，在 navtest > 20° 转弯 token 上有没有 out-of-sample 正增益（2026-10-08，任何本线 selector 读数之前写定）

## 问题

第 186、187 条：navtest > 20° 的 3 154 个 token 上，SH30 plan 附近 F19 的可信特权上限是 F19 × pc +9.55（L9 × epcap +9.20）。在 navtest 内部按 log cross-fit，
非特权输入（ego + 指令 + plan、SH30 隐状态、未池化冻结视觉 token、自路沿 margin）全部学不出这个 pick（最好点估计 +0.32 [−0.65, +1.26]），特权地图 margin 回收 34–44%
（+3.2 ~ +4.2）；非特权神经头的学习曲线在 100% 的 3 k token 处仍在升。第 190 条给出 9 倍规模的标签：28 323 个 navtrain 转弯 token 的 held-out（5 折按 log）SH30 plan 与 F19 全部候选的 simulator 分数。
本线只问一个问题：**在 navtrain held-out 标签上训练、只用非特权输入的 selector，在 navtest 转弯 token（真实的 SH30-F-s0 / s1 plan 和已有候选分数）上有没有正的 out-of-sample 增益。**
不新打任何 simulator 分，不训基座；navtest 的标签不参与拟合和选模，超参只在 navtrain 上按 log 选。

## 已有的、不重测

第 178、186、187、190 条的全部数（上限 +9.55 / +9.20，E 臂 −0.27，187 的 P1–P3 / N1–N7 读数与学习曲线，190 的 DAC 失败率 5.13% 对 7.38%）。本线把 187 的 navtest 内部读数并排放进表里作对照，不重算（E 臂除外，见闸门）。
已看过、所以不算盲的数：上述全部；**没看过的**：任何用 navtrain 标签训练的 selector 在 navtest 上的读数、navtrain 上的 out-of-fold 增益、fold 模型的隐状态 / 路沿输出。

## 数据与特征

**训练集**：navtrain 转弯 token 28 323 个（900 个 log），一个 token 一行（fold 模型 seed 0 的 held-out plan）。每个 token 的全部特征来自**没见过它那条 log 的 fold 模型 `CF5f{j}-F-s0`**（j = `sha256("cf5|" + log) % 5`）：
- 标签：`cscore_all_F19.csv`（F19 = c00..c18，non-reactive，v2_navtrain）的 8 个子分，口径按 `turn_dewater.conv`：F19 × pc（主）与 L9 × epcap（一致性；L9 = 速度恒等的 9 个点，全部在 F19 内），从 navtrain 的 F19 子分重算；目标 = 候选相对恒等的 no-EC EPDMS 增益。
- E 流：`ego`（`tab["ego"]`，20 维，含指令）+ `plan_desc`（15 维，held-out plan 的描述）。
- 隐状态 H：fold 模型 `select_4` 与 `mean`（各 512，拼成 1 024 维）；同一次前向顺带取 road_edges。
- 未池化视觉 token：shard cache `front.npy` 最后一帧 32 × 512（冻结 shipped 视觉编码器，与 fold 无关）。
- 自路沿 margin C：fold 模型 road_edges 经 `calibration.json` 里 SH30-F-s0 的 (s, b)（对所有 fold 模型同一组常数，不重拟合）；定义同 turn_selinput（4 角点足迹，累积到 1 / 2 / 3 / 4 s，截 [−2, 4]）。
- 特权地图 margin M：`op_probe/labels/navtrain_all.npz` 的 SDF（全部转弯 token `ok`），定义同 turn_selinput。标 PRIV。

**测试集**：navtest > 20° 转弯 token 3 154 个 × SH30-F-s0 / s1，输入特征、候选、标签、行定义与第 186、187 条完全相同（`turn_selinput.prepare`）。拟合阶段只读测试特征（含 margins.npz），**不读 navtest 的分数**；分数只在 `report` 阶段读。
读数 = 被选候选在族口径下的分数 − 恒等，对 seed 取均值，按 log 聚类配对 bootstrap（`jevdrive.stats`，B = 10 000，seed 0，108 个 log）；回收率 = 增益 / 同族同口径 navtest 上限（`TC.ratio_ci`）；分桶 20–45° / > 45°；选择后的 DAC 失败率（pc 口径下被选候选 DAC < 1 的比例，对 seed 平均）。

## 臂

头的写法：L = 多输出 ridge 回归各候选增益后 argmax（λ 网格 / k / margin m 同 `turn_dewater`，内层 4 折按 log 在训练集内选）；G = HistGB 树，行 = (token, 候选)，超参固定同 187（200 iter、lr 0.05、15 叶、min leaf 50、l2 1，不调）；R = 一参数规则（τ 网格同 187，按训练集实现增益选）；NN = 小神经头（见下）。标准化、PCA（隐状态白化，k ∈ {8, 32, 128}）只在训练集上拟合，不看标签。

| 臂 | 头 | 输入 | |
|:--|:--|:--|:--|
| E | L | ego + plan | 对照 |
| N1 | L | E + 全部 18 条候选的 C | |
| N2 | G | E + 候选 C 行 | |
| N3 | R | C 规则 | |
| N4 | L | E + 隐状态 PCA | |
| N5 | NN | E + 未池化视觉 token（注意力池化） | |
| N6 | NN | E + 隐状态（MLP） | |
| **N7** | **NN** | **E + 视觉 token + 隐状态 + 全部候选 C** | **主臂，事先指定** |
| N8 | NN | E + 视觉 token + 全部候选 C（无隐状态） | 隐状态闸门失败时的替补主臂 |
| P1 | L | E + 全部候选 M | PRIV 上限 |
| P2 | G | E + 候选 M 行 | PRIV |
| P3 | R | M 规则 | PRIV |
| P4 | NN | E + 全部候选 M | PRIV，与主臂同一种头 |

主臂选 N7 的理由：它包含全部非特权信息，187 里神经头的学习曲线仍在升，9 倍标签是对这一点最直接的检验；不为后面的读数改。

**神经头**：输入流各自标准化；token 流 LayerNorm → Linear 512→64 → 4 个可学习 query 的 softmax 注意力池化（256 维）；ego / 隐状态 / C 流各 Linear→GELU→宽度 w；拼接 → Linear → GELU → Dropout(p) → Linear→ 18 个增益；MSE，AdamW lr 1e-3，30 epoch，batch 512，
按训练集内 20% 的 log（hold-out）上实现增益取最好 epoch。**配置网格**（w, p, weight decay）：(64, 0.1, 0.05)、(128, 0.2, 0.05)、(128, 0.3, 0.3)、(256, 0.3, 0.3)，**每个 NN 臂分别**在 navtrain 上按 log 3 折 CV 的 out-of-fold 实现增益（F19 × pc）选一个，ties 取更小的 w。
最终读数 = 5 个初始化（不同种子与不同 early-stop hold-out log）预测增益取平均后 argmax（一个预测，不是 5 个读数的平均）。恒等始终是候选 0（增益 0），沿用 `picks_of` 的 m = 0。

## 判定与停止线（固定，看到 navtest 读数后不改）

**主读数**：N7 在 F19 × pc、> 20° 的 navtest 增益。多重性：两个登记的族 × 口径（F19 × pc 与 L9 × epcap）按 Bonferroni m = 2，读数的 CI 取 97.5%（也报 95%）；其余非特权臂（E、N1–N6、N8，m = 7）与特权臂（P1–P4，m = 4）的对照只在表里报 Bonferroni CI，**不能用来救主臂**。
- **停止线**：N7 在 F19 × pc 的 97.5% CI 下界 ≤ 0 → 「从 simulator 分数在冻结特征上学转弯 pick」这条线在 navtest 上结束（隐状态闸门失败时以 N8 代替 N7，同一规则）。若 N7 失败但别的非特权臂的 Bonferroni 下界 > 0，写成线索，不改判，要继续须新的预登记且不能再用 navtest 选臂。
- **一致性**：N7 的 L9 × epcap 点估计必须 > 0，否则 F19 × pc 的正值判为不一致（疑为放慢 / 视野效应）。
- **值得做全量方法的增益**：F19 × pc 点估计 ≥ **+2.0**（> 20°，EPDMS × 100）且上述下界 > 0。理由：+2.0 = 同族上限 +9.55 的 21% 回收，等于 187 里「几乎没学到」的 20% 线；是 SH30 对 WA-JEPA 在这些 token 上 6.52 差距的 31%；
  折到 navtest 全量约 +0.5 EPDMS（转弯 token 占 26%），与 hinge 各变体之间的差（P2H10 对 SH30 约 0.9）同量级；同时只有 187 中特权地图 margin 臂（+3.2 ~ +4.2）的一半，是对非特权输入的合理要求。
  下界 > 0 但点估计 < +2.0：「有但不值」，线按 navtest 读数结束；点估计 ≥ +2.0 但下界 ≤ 0：不判，按停止线结束。
- **学习曲线**（N7、P4、P2）：训练 log 的 5 / 10 / 25 / 50 / 100%（900 个 log 里随机子集，3 次重复，树 2 次），每点超参取 100% 选出的；读数 = 重复间平均预测增益；与 187 的 navtest 内曲线（3 k token ≈ 本线的 10%）并排。曲线在 75 → 100% 仍升 ≥ 0.3 判「仍在升」（沿用 187）。

## 闸门（任何一项不过就停下报告，不读 selector 读数）

- **G-leak**（每个 navtrain token）：(a) 抽取用的模型 j = `fold_of_token.csv` 的折 = `sha256("cf5|" + log) % 5`，且该 log 不在 `navsim/op-parity-cf5f{j}-train` 里；(b) 抽取得到的 plan_pos 与同一模型存档的 plan_pos（`bench/ol/<shard>/plans/`）在速度 ≥ 0.5 m/s 的行上最大差 < 0.05 m（特征与标签的 plan 来自同一次模型输出）；
  (c) 反例对照：用错误折的模型在 300 个 token 上抽取，plan_pos 差的中位数必须明显大于 (b) 的差（证明检验能区分）；(d) `navtrain` 与 `navtest` split 不相交；(e) c00 的 8 个子分在全部 28 323 token 上等于 `score.csv`。
- **G-E**：navtest 内 cross-fit 的 E 臂（`turn_selinput.unit2`，F19 × pc，10 次重复）复现第 186 条 −0.27 [−0.52, −0.06]（均值差 ≤ 0.01）。
- **G-margin**：navtrain 上恒等候选的 4 s 地图 margin 对 DAC 失败的 AUC ≥ 0.85（同 187 的 G1），自路沿 margin 的 AUC 落在 0.58–0.72（G2；若落在外面只标注，不停）。
- **G-hidden**：5 个 fold 模型在 navtest 转弯 token 上的隐状态与 SH30-F-s0 隐状态的逐坐标 Pearson 相关中位数 ≥ 0.8（fold 模型与 SH30 同配方同初始化，隐状态空间是否对齐是训练 / 测试特征可比的前提）；不过则 N4 / N6 / N7 标「隐状态空间不可比」，主臂换 N8。
- **冒烟**（约 300 个 navtrain token，5 折各约 60，跨 shard）：整条链（抽取 → build → 头 → navtest 预测）跑通，G-leak 全过；按冒烟实测外推全量成本，若超过预算（约 4 h 墙钟、3 card-h、60 core-h）的 1.5 倍停下报告。

## 不做

不新打 simulator 分，不碰 sh30_crossfit 的标签 / 检查点和 metric cache；不用 logged future、dyaw 标签、score 列作输入；车道线输出不做；不训练第二个 fold 的 seed；不在 navtest 上调任何超参、选臂或选 epoch；
L9 × epcap 外不加新的族 / 口径；不做闭环。
