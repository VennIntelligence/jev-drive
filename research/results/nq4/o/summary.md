# O：训练数据出处与元数据缺口

**CX 产出，待 main 复核。状态：暂停，尚未计数。** 2026-09-26 17:29 CST 按任务书第 5 条停下：现有元数据没有建立官方 Bench2Drive 训练 clip（一次实际记录）到 scenario trigger（场景触发点）的可靠对应关系，不能把未知近邻数写成 0。未生成 `overlap.csv`，未做与 G 幽灵率的相关分析。

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

这些是出处和文件盘点数字，**不是近邻结果**。当前每个训练集的近邻分布均未计算；没有均值、分位数或零近邻路线数可报告。

## 停止点与交接

需要 main 提供能关联实际 clip ID、town、原始 scenario 类与触发点坐标的元数据清单，或明确新的部分覆盖范围。当前规格未允许只数路线模板、只数 mini 子集，或把公开但与原实验不同的数据当成同一个训练集，因此这些替代均未执行。

检查使用 CPU 198–199，未使用 GPU，未下载传感器数据。没有启动计数长任务或触碰 night-queue-3 进程。实际本轮审计约 5 min（box 17:25–17:29 CST）；现有 17:15 CST 登记之前的实际操作时长未查清。
