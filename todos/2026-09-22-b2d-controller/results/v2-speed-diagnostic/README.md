# v2匹配速度诊断：闭合归档

原始目录 `/data/runs/b2d/controller/development-v2-speed-diagnostic`。6/6 completed、无collision/exception/cleanup error；6/6 cruise_speed失败，17563/TCP额外full_lateral_rms与full_lateral_p95失败。不能把完成等同验收通过。

[summary.json](summary.json)为原文件逐字节复制；[cases.csv](cases.csv)为完整6例表；[metric-verification.json](metric-verification.json)记录固定巡航真值3D速度与另外两种诊断口径，前者逐例复算差<1e-12。
[完整raw索引](../raw-file-index/development-v2-speed-diagnostic.json)：87 files / 10,216,815 bytes，逐文件SHA256。归档前事件以end/completed/cases6闭合，server PID178521已不存在；索引逐文件检查读取前后mtime/size。未改写raw。

[纵向机制与候选](../../agents/v2-longitudinal.md)；此前v1/v2结果保留。速度RMS为自建G2验收，非B2D官方Driving Smoothness。
