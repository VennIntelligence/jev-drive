# 正式v4成本与复现

63次attempt均保留，60条正式结果包含11条驾驶超时；3次基础设施失败额外88.1s。
campaign起点至结束事件2247.413s，全部attempt2194.8s；结束事件后的server.stop不在此计时。
已选attempt合计2106.7s，剔除每条首20tick的profiled loop合计860.017s；1246.683s余量包含setup、cleanup、warmup及未计时工作，不伪称已测得独立启动成本。

[逐组摘要](artifacts/summary.json)、[63次attempt CSV](artifacts/attempts.csv)、[全部输入/输出哈希](artifacts/manifest.json)。
重算：`PYTHONDONTWRITEBYTECODE=1 /data/envs/carla/bin/python analyze.py --campaign /data/runs/b2d/controller/formal-v4 --out /data/runs/b2d/controller/formal-v4-cost-reproduction`，在本目录执行，输出须不存在。
所有数据均为route oracle/policy=none；不包括真实模型推理，不作Dev10乘22的full220外推。
