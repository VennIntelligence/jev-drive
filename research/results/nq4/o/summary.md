# O：训练数据出处与元数据缺口

**CX 产出，待 main 复核。状态：已完成 main 授权的描述性 fallback；原 `< 30 m` 近邻计数不可得。**

产物：[overlap_fallback.csv](overlap_fallback.csv)，每条评测路线 × 每个数据集一行。仅 B2D base / full 可计数；TFv6 / BridgeDrive / SimLingo / BLUE 标 `not determinable`。全部行的 `trigger_distance_checked=false`。不作与 G 的相关。

以下保留前期出处审计与停止原因。 2026-09-26 17:29 CST 按任务书第 5 条停下：现有元数据没有建立官方 Bench2Drive 训练 clip（一次实际记录）到 scenario trigger（场景触发点）的可靠对应关系，不能把未知近邻数写成 0。未生成 `overlap.csv`，未做与 G 幽灵率的相关分析。

## 已登记的口径

按 todo 中 17:15 CST 的 [CX] 条目：评测单位是 `bench2drive220.xml` 的一个 route。训练 clip 与评测 route 的 town 标识、原始 scenario 类字符串必须完全相同，CARLA 世界平面坐标的触发点欧氏距离必须严格小于 30 m；多触发点按 clip ID 取并集。没有实际训练成员关系或触发坐标的样本不推断，也不以数据采集路线模板代替实际 clip。原估时 2 h，超 4 h 停；此次因元数据缺口先停。

## 数据来源核查

下表中的行号来自 box 上指定 commit 的 README；链接固定到相同 commit。已查清数据族不等于已查清用于某个公开 checkpoint 的全部 clip 成员。

| 项 | 已核实的来源 | 实际训练成员与触发点状态 |
|:--|:--|:--|
| 官方 B2D base / full | [Bench2Drive README，commit `7ec25d1`，第 60、69–75 行](https://github.com/Thinklab-SJTU/Bench2Drive/blob/7ec25d1c9f7522d923ce5f3420986cef1cb2d956/README.md#L60)：base 指向 `docs/bench2drive_base_1000.json`；full 指向 `docs/bench2drive_full+sup_13638.json`。 | 已找到官方成员清单，记录 archive 文件名、SHA-256 与大小；清单没有触发坐标。clip ID 与训练 XML 的可靠对应关系**未查清**。 |
| TFv6 / LEAD | [TFv6 使用的 cvpr2026 README，commit `730bc1a`，第 12、342–346 行](https://github.com/kesai-labs/lead/blob/730bc1a2f44d5f28312dd55f0ca958e94a24c038/README.md#L342)：下载地址是 `ln2697/lead`。 | 公开数据族已查清。原论文 / checkpoint 实际使用的完整 clip 成员与触发坐标**未查清**；没有用后来的主分支数据版本替代。 |
| BridgeDrive / LEAD | [BridgeDrive README，commit `85aa089`，第 111–112、357、379–383 行](https://github.com/shuliu-ethz/BridgeDrive/blob/85aa089a321a7b7cbc42aa58b3be508eb223d9e2/README.md#L379)：榜单版本标 LEAD，并要求下载官方 LEAD 数据；其适配指定 LEAD commit `a41d116`。 | [指定 LEAD README 第 64–65 行](https://github.com/kesai-labs/lead/blob/a41d11616d06843ba89388a278e6e025b6a47878/README.md#L64) 明确说公开数据与原实验数据不同。该版本第 214–218 行指向 `ln2697/lead_carla`。实际训练版本与完整成员**未查清**。 |
| SimLingo | [SimLingo README，commit `743b243`，第 95–96、136、143、209 行](https://github.com/RenzKa/simlingo/blob/743b243afd6cf5ff51b9fa1f8cac86f22d569684/README.md#L94)：使用 `RenzKa/simlingo`，PDM-Lite expert；采集路线在 `data/simlingo/`。 | 公开数据族已查清。[数据卡](https://huggingface.co/datasets/RenzKa/simlingo/blob/main/README.md) 明确提示样本不来自唯一的路线。采集 XML 本身不能证明实际成功、清洗后保留的 clip 成员；完整成员与触发点**未查清**。 |
| BLUE | [BLUE README，commit `6970cb6`，第 48–51 行](https://github.com/George-Ling3/BLUE/blob/6970cb69e05ef904b37f4264207ea0e1dab35ef2/README.md#L48)：backbone 使用官方 SimLingo checkpoint。第 6、61 行另指向 gate（逐帧语言开关）训练数据 `George-Ling/blue_data`。 | backbone 继承 SimLingo 的上述缺口；额外 gate 训练 clip **未查清**。[官方训练数据卡](https://huggingface.co/datasets/George-Ling/blue_data/blob/main/README.md) 的 “This Release Includes” 写明训练数据尚待发布。不能把 BLUE 全部训练数据等同于 SimLingo。 |

HF（Hugging Face，公开数据仓库）数据卡与 API 清单查于 2026-09-26 17:29 CST；它们是查询时状态，不是固定版本的数据成员证明。

## box 上的元数据盘点

| 元数据 | 位置或核查方式 | 核实内容 | 对本项计数的限制 |
|:--|:--|:--|:--|
| 官方评测路线 | `$DATA_DIR/third_party/Bench2Drive/leaderboard/data/bench2drive220.xml` | XML 有 220 个 route；包含 town、scenario type、trigger_point 的 x / y。 | 评测侧字段齐全。 |
| 官方 base 清单 | 同树 `docs/bench2drive_base_1000.json` | 约 179 KiB；JSON 字典的 key 是 clip archive 名。 | 没有 trigger_point；未遍历或下载传感器 archive。 |
| 官方 full 清单 | 同树 `docs/bench2drive_full+sup_13638.json` | 约 2.5 MiB；与 base 相同清单类型。 | 同上。 |
| 官方训练路线 XML | 同树 `leaderboard/data/routes_training.xml` | 90 个长 route，示例 route ID `0`，一个 route 包含 scenario 触发点。 | 未建立它与清单中 `Route1102` 等 clip ID 的映射，不能按同名或近似位置猜。 |
| 本地 B2D mini | `$DATA_DIR/datasets/bench2drive-mini/` | 存在三个 clip 目录。 | 不代表 base / full 全体，不据此报告完整分布。 |
| SimLingo 采集路线 | `$DATA_DIR/third_party/simlingo/data/simlingo.zip` | 约 18.5 MB，ZIP 目录含多个 training 配置下的 XML。 | 未解压或计数；模板与实际 clip 成员尚未对应。 |
| SimLingo 公开成员辅助文件 | [HF 根目录 API](https://huggingface.co/api/datasets/RenzKa/simlingo/tree/main?limit=1000) | `buckets_paths.pkl` 为 619,707,258 bytes；其余可见 driving 数据是混合传感器 `.tar.gz` archive。 | 未下载 bucket 或 archive；其是否足以恢复 clip 成员与触发点尚未查清。 |
| LEAD 公开 archive 名 | [HF `ln2697/lead/Accident` API](https://huggingface.co/api/datasets/ln2697/lead/tree/main/Accident?limit=10) | 目录名可给 scenario，ZIP 名可给 town 与 route ID。 | 名称没有触发坐标；未下载 ZIP。 |
| BLUE gate 数据 | [HF 根目录 API](https://huggingface.co/api/datasets/George-Ling/blue_data/tree/main) | 查询结果仅 `.gitattributes` 与 README。 | 没有可核查的 clip 成员或触发点。 |

以上是前期出处和文件盘点数字，**不是近邻结果**。原 `< 30 m` 近邻分布仍不可得；后续获准的同 town、同 scenario fallback 分布见文末。

## 停止点与交接

需要 main 提供能关联实际 clip ID、town、原始 scenario 类与触发点坐标的元数据清单，或明确新的部分覆盖范围。当前规格未允许只数路线模板、只数 mini 子集，或把公开但与原实验不同的数据当成同一个训练集，因此这些替代均未执行。

检查使用 CPU 198–199，未使用 GPU，未下载传感器数据。没有启动计数长任务或触碰 night-queue-3 进程。实际本轮审计约 5 min（box 17:25–17:29 CST）；现有 17:15 CST 登记之前的实际操作时长未查清。

## 17:38 CST 恢复委派后的补核

再次核查的结果是：可以把缺口说得更具体，但尚不能开始近邻计数。没有改变匹配口径，也没有把采集路线 XML 当成实际训练 clip。

| 核查 | 一手证据 | 对恢复计数的影响 |
|:--|:--|:--|
| 原始 B2D 每帧 annotation 是否能直接提供 scenario trigger | 固定版本 [tools/data_collect.py 第 792–821 行](https://github.com/Thinklab-SJTU/Bench2Drive/blob/7ec25d1c9f7522d923ce5f3420986cef1cb2d956/tools/data_collect.py#L792) 的 `anno_data` 字段包含 ego pose、导航 target、weather、bounding_boxes、sensors 等，没有 scenario 类、route ID 或 scenario trigger 坐标。box 已有 mini clip 的一个 annotation 文件字段核对相符。 | 仅下载每帧 annotation 并不能直接补齐本项要求的 clip→trigger 对应关系。ego 位置和交通灯 trigger volume 不能替代 scenario trigger。 |
| 官方是否声称评测路线与训练 clip 相同 | [官方仓库 issue #193 的维护者回复](https://github.com/Thinklab-SJTU/Bench2Drive/issues/193#issuecomment-3270122438) 说 220 条路线不在 train / val 数据中。 | 这是路线成员声明，不是距离证据：不同路线仍可能共享邻近触发点，不能据此把近邻数置 0，也不能把评测 trigger 直接赋给训练 clip。 |
| 是否发现新的独立 clip→trigger manifest | 官方仓库 tracked 文件清单和相关公开 issue 中，本次没有找到可靠映射。 | 仍为「未查清」，不声称不存在。此次补核没有下载传感器、bucket 或其他大型数据。 |

能恢复计数的最小输入是一份逐实际 clip 的元数据表，列为 `dataset_id, dataset_version, clip_id, town, scenario_type, trigger_x_m, trigger_y_m, source_uri`。一 clip 有多个 scenario trigger 时可重复列行，计数按 clip ID 并集。还需要每个榜单族 / checkpoint 对该版本训练成员的出处；已公开但与原实验不同的 LEAD 数据，应明确作为哪个版本报告。B2D 也可提供原始采集 XML 加经过出处核实的 clip ID→XML route ID 映射。以上输入均不需要传感器文件。

在这些输入到位前，`route × dataset` 的数值 CSV 仍无法完成；本次未生成全为 unknown 的占位表，避免把占位状态误读成计数产物。恢复补核约 3 min（17:36–17:39 CST），低于此次 15 min 限时。


## 18:27 CST main 授权后的描述性 fallback（已完成）

main 明确将本次交付范围改为：按每条评测路线，统计训练 clip 的 **town 和原始 scenario 类完全相同**的数量，**不检查触发点距离**。这是类别与城市覆盖计数，不能称为 `< 30 m` 近邻，也不能据此判断位置重叠或背题。原任务要求的距离统计不可得。

输入为官方 commit `7ec25d1` 的两个 JSON 清单与 `bench2drive220.xml`，总元数据 3,417,420 bytes，无传感器下载。完整成员数分别为 base 1,000、full 13,638；每个清单 key 是一个实际 clip archive，Weather 后缀不影响 clip 唯一性。full 清单有一个 `...Route523_Weathe.tar.gz` 截短后缀，town / scenario 字段仍完整，直接按原始 key 计一个 clip，没有补写或猜测 Weather。Town10HD 保留原名，没有折叠到 Town10。全部 key 的 town / scenario 均可解析，排除数为 0。

| 数据集 | 训练 clip n | 评测 route n | 可计数 route n | 同 town、同类 clip 数均值 | min | p25 | median | p75 | p95 | max | 计数为 0 的 route n |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| B2D base | 1,000 | 220 | 220 | 5.97 | 0 | 1.00 | 4.00 | 7.00 | 26.00 | 35 | 53 |
| B2D full | 13,638 | 220 | 220 | 122.04 | 0 | 6.00 | 18.00 | 123.75 | 697.30 | 980 | 11 |
| TFv6 / LEAD | 未查清 | 220 | 0 | not determinable | — | — | — | — | — | — | — |
| BridgeDrive / LEAD | 未查清 | 220 | 0 | not determinable | — | — | — | — | — | — | — |
| SimLingo | 未查清 | 220 | 0 | not determinable | — | — | — | — | — | — | — |
| BLUE | 未查清 | 220 | 0 | not determinable | — | — | — | — | — | — | — |

读法：这两个 B2D 列仅表示评测路线所属 town / scenario 在公开训练清单里的覆盖量；不是附近有多少 clip。分位数用排序后线性插值，单位均为 clip。其他族缺完整、明确版本的训练成员清单，不把未知值记为 0。`overlap_fallback.csv` 共 1,320 行（220 route × 6 数据集），每条评测路线原 XML 恰有一个 scenario。若以后出现多 scenario 路线，脚本按 scenario 类集合匹配，同一 clip 每 route 仅计一次。

可复现脚本：`scripts/nq4_o_fallback.py --metadata-dir <三个元数据文件所在目录> --output-dir research/results/nq4/o`。固定来源链接见上表；输入 SHA-256、清单大小与解析排除数保存在 [fallback_sources.json](fallback_sources.json)。脚本检查评测 route 数和 ID 唯一性、全部 clip 名字段解析及计数总和。输出再次核对 220 route × 6 dataset 唯一键、880 个未知值与 440 个可计数值；同一 town / scenario 的路线计数一致。

fallback 本轮于 18:27 CST 登记后开始，约 4 min 完成。仅本机单线程 CPU，不使用 GPU、CARLA 或传感器 archive。
