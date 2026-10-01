# paired-v2最终报告文本冻结与路径重定位

2026-09-23。原 `../paired-v2-final-audit/manifest.json` 将项目主 README.md 和 TCP-final-report.md 记录为 source hashes。本目录在主README继续整合转弯结果前保存这两份文本的准确字节；两份SHA256均已与旧审计逐项核对一致。旧报告数值、旧审计及其hash不改写。

此后主README可以更新。复核旧审计时，对这两条原文档路径使用manifest.json中的relocations映射，读取本目录冻结副本并验证原SHA256。其余原始日志、分析、图表及协议来源仍按旧审计路径验证。冻结副本中的相对链接沿用原文档文字，不经重写，阅读原路径语境请参考original_path。

TCP-final-report中的“转弯阶段正在进行”是当时写作状态；后续完成状态由根代理综合报告说明。其成本段沿用当时字段名称；attempt.wall_s实际为evaluator覆盖字段，统一成本解释应以根代理新成本报告为准。这些后续澄清不改变本冻结文档的原始字节或实验数值。
