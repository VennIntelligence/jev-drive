# v2–v4验证日志快照

2026-09-23 JST复核。完整原始路径、字节数、SHA256与结果见[index.json](index.json)。所有日志逐字节复制，包含失败命令和更正版本；没有覆盖原始日志，未来新日志需新文件名及新版索引。

- v2 freeze：91 tests OK；pre-matrix：86 tests OK。
- v2 speed diagnostic首次命令引用不存在的test_b2d_controller_metrics模块而失败；corrected：19 tests OK。
- v3 pre-freeze：97 tests OK；freeze：102 tests OK。
- v4首次freeze未设置DATA_DIR而测试模块导入失败；corrected：111 tests OK。不要把这两次命令环境/模块问题误记为仿真失败，也不能删掉失败记录。
- pi-g1-kp05.log保留合成低Kp运行输出；合成14例与真实CARLA18例使用不同plant，不能互换验收。

这份索引仅覆盖捕获时已存在的文件，不预先代表后续v4运行或新增验证。
