# 原始证据索引

每个JSON列出对应已结束阶段的全部文件、字节数与SHA256；包括失败和部分日志，不修复或覆盖原始内容。
原始根目录写在各JSON的`root`字段。索引是捕获时刻的快照，后续新增文件不能被当作已由旧索引覆盖。

| 索引 | 数据 |
|---|---|
| [baseline](baseline.json) | 开发前diff和基线记录 |
| [calibration](calibration.json) | 车辆physics及响应，包括失效采样 |
| [calibration-units](calibration-units.json) | 转向单位与IMU正负号探针 |
| [development](development.json) | 第一次失败开发尝试 |
| [development2](development2.json) | 包含验证器误判的第二轮完整trace |
| [development3](development3.json) | 最终12例无交互开发验证 |
| [smoke](smoke.json) | 完整官方2390 smoke |
| [offline-traced](offline-traced.json) | 49例逐tick理想plant记录及等价性核验 |
| [Dev10 seed0](../dev10-seed0-file-index.json) | 首轮3preset全部attempt与崩溃重试 |
| [confirmation v1](confirmation-v1.json) | 完整seed1、6条保留集与坡道保持，含两次额外失败attempt |
| [development S v1](development-s-v1.json) | 新增17563真实S弯的3例完整原始记录，全部速度gate失败 |

用于轨迹/控制曲线的数值数据已保留；本轮未录制相机RGB流或视频。早期开发源代码追溯的限制见[文章素材](../../article-notes.md)。
