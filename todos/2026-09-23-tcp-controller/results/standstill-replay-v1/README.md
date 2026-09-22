# 真实静止负速度的反馈修复

paired-v1在第5例1773发现微小负速度被新helper误当倒车，根代理主动停止（原harness字段仍保留cancelled_by_user，实为agent维护，不是用户取消）。前4完整+第五部分数据均保留，不与v2拼接。

捕获1299个有预测的输入重放中，旧reverse fault456、新3；235帧原生启动油门不再被保护强改全制动。仍有真实≤−.01m/s反向故障，未删除。输入重放只证明规则变化，完整六例需在paired-v2重跑；不能从此证明障碍绕行已恢复。

完整输入前缀位于/data/runs/b2d/tcp-controller/standstill-replay-v1/captured-motion-prefix.jsonl，SHA见summary/intervention；小型逐帧输出、旧新代码和17项测试日志收录于此。旧paired-v1闭合原始目录162文件/15007021字节，索引另列。

replay.py复现脚本的输出目录是脚本所在目录，必须将脚本与captured输入放入新目录运行，拒绝覆盖replay.csv；源码基线明确为d8bfffb。原始版本都保留。
