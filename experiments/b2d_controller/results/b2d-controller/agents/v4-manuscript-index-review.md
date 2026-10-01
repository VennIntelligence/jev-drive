# 稿件与归档复核

复核日期：2026-09-23 JST。范围仅iteration-v2.md、docs/b2d-controller.md、相关现有raw与测试记录；未改上述两份根代理文件、核心或原始run，未执行git/CARLA。

## 稿件结论与链接

两份文档所有本地Markdown链接均可解析。v2 20例/10043tick/15例gate通过、v3 18例/13例gate通过与原summary一致。初版和修正版图各自保留且正文链接指向选定版本。102项回归测试有对应v3-freeze原始日志支撑，作为历史版本描述正确；最终版本应另报v4 corrected的111项，不倒改历史计数。

现有DS/舒适性段落方向正确：DS=completion×infraction penalty；本项目driving_completed只以completion≥100计数，不能称官方SR。固定版本merge_route_json.py还要求status为Completed或Perfect，且除min_speed_infractions外所有infraction列表为空；其全榜聚合分母写死220，因此不能把开发子集直接交给该脚本得到的值标成合法全榜SR/DS。开发集使用自己的分母时必须命名为子集统计，不冒充官方full220结果。

Driving Smoothness由独立脚本计算；速度RMS是本地G2跟踪门槛，并非官方SR、DS或舒适性。正文保留这一边界即可。yaw导数/deg-rad等版本源码疑点留在[纵向附录](v2-longitudinal.md)和[source observations](../results/scoring-source-observations/README.md)，不作为标题或主要收益结论。

根代理收尾时需做的少量文字修正：

1. docs的Last verified更新为2026-09-23；日期改变不代表未完成试验已经验收。
2. “pinned”三份官方链接宜将可移动分支0.0.4改成已记录commit `7ec25d1c9f7522d923ce5f3420986cef1cb2d956`；SR来源为`tools/merge_route_json.py`。
3. iteration原“下一步”PI小节可标为“首个候选的预声明设计”，减少已完成v3与未来时态混淆。
4. 根代理取得完整v4后只更新实际通过数、失败gate、配置/source及结果链接；当前未完成时不从前三例推断整组。

## 可直接采用的最终验收措辞

若v4仍有某配置列失败：

> v4使用固定Kp=.5、Ki=.25完成全部18例开发对照，结果为〔X〕例通过全部原门槛，失败见逐例表。〔具体配置〕仍未通过〔具体gate〕，本轮不宣布新的合格默认。保留现有vendor默认与全部实验配置、原始证据和失败结果；本轮不继续增益搜索。

若某列六组全部通过、其他列仍有失败：

> 〔精确配置〕通过预声明的五条开发路线及额外1773@6探针，达到本轮G2开发门槛；完整矩阵其余失败仍保留。这支持冻结该配置进入原计划的复验与集成/正式路线对照，尚不构成全榜收益或替换默认的验收结论。

若18例全部通过：

> 固定Kp=.5、Ki=.25的18例开发矩阵均通过原门槛。该结果验证本轮开发范围，尚不能替代原计划的冻结复验、G3集成及G4对照，也不能证明官方DS/SR或Driving Smoothness改善。

v1历史两个seed与holdout验证的是旧配置，不能自动继承给改变adapter或PI参数后的新版本。是否推进已有后续验收由根代理按原plan执行；本复核不新增搜索、路线集或舒适性必过gate。

## 已完成的归档修补

[机器可读审计](../results/post-verification-archive-audit.json)逐项给出旧索引哈希与新索引路径。

- development-v2旧176文件逐一哈希未变；仅新增解释器pycache，完整新索引177文件/25,477,098字节。
- development-v3旧170文件逐一哈希未变；同样仅新增pycache，完整新索引171文件/31,601,570字节。
- 两份旧索引完整保留。新路径为`development-v2-post-verification-file-index.json`与`development-v3-post-verification-file-index.json`。
- [PI G1 Kp1全索引](../results/raw-file-index/pi-g1-v1.json)：21文件/8,487,529字节；[Kp.5全索引](../results/raw-file-index/pi-g1-kp05.json)：21文件/8,494,421字节；各14case、28条start/end事件，不是28case。
- [rejoin replay全索引](../results/raw-file-index/rejoin-replay-v1.json)：37文件/3,747,691字节。
- [验证日志归档](../results/verification-v2-v4/README.md)：9份现有日志逐字节复制并记录raw路径/hash。包含v2模块名错误和v4缺DATA_DIR的失败命令及corrected版；未删除失败记录。未来日志须新文件名/新版索引，不覆盖此捕获。

尚余原任务内收尾：根代理在v4闭合后归档其完整raw/source/summary/最终图与索引，链接本轮新增索引及日志，按完整结果写最终状态。当前所有已记录旧raw哈希已复核不变；源缓存新增不是控制器源代码更改。
