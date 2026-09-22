# 共享进展

## 2026-09-22：实施启动

- 旧工作树已备份 diff 至 `/data/runs/b2d/controller/baseline/pre-controller.patch`。
- 既有 runtime/sensor/preview 测试：Python 3.8 下 18 项通过。
- 评测使用 `/data/third_party/Bench2Drive` 的 `0.0.4`，仅物理 GPU 1，一个 server。
- 分工：controller 子代理负责纯 NumPy 数学和离线验证；agent 子代理负责定位/route/双频率集成；
  report 子代理负责 CLI 与结果汇总。主代理负责基线、车辆标定、无交互实车验证和实验执行。

当时尚未产生结果；以下记录按阶段保留，当前实测总表见 [article-notes.md](article-notes.md)。

## 基线、标定与首轮离线结果

此前的 smoke/cancellation 和 TCP/runtime 工作分别保存为 `0636c06`、`b1a436a`；共享目录为 `e2f61a9`。
发现远端 main 已有独立 Waymo 工作，已无冲突合并，保留双方历史，当前回到 main。

车辆参数来自 `/data/runs/b2d/controller/calibration/physics.json`：stock CARLA 0.9.15，Town10HD，
server port 5000，GPU UUID `GPU-b90dd90e-394b-7800-f23f-5892a8e3d0f1`。
轴距 2.86047149 m，后轴相对 actor 原点 x=-1.38863322 m，前轮最大转角 69.99999237°；
steering curve 为 (0,1)、(20,.9)、(60,.8)、(120,.7)。[参数与完整 physics](results/calibration-physics.json)。

再次标定 `/data/runs/b2d/controller/calibration-units/` 独立验证了单位和正负号：临时诊断曲线
(0,1)、(10,.1)，实际 2.00147 m/s、steer=.05，前轮平均角 1.22293°，与 km/h 横轴一致。
IMU gyro z 与 actor 的右正 yaw rate 相关系数 .999946，比例中位 .9999977。
诊断后恢复 stock physics；该临时曲线不进入评测。[单位探针](results/steering_unit_probe.json)、
[传感器符号](results/sensor-sign-check.json)。10 m/s 右转响应采样发生速度坍塌，未作为转向拟合数据。

12 项 controller 契约测试、8 项 agent 测试、10 项 report fixture 与既有 18 项测试，全套共 48 项通过。
[离线结果](results/offline-selftest.json) 中 pursuit/TCP/CARLA 圆弧 RMS 分别约 .0091/.1546/.4671 m；
CARLA 参考未通过原 .2 m 门槛，未修改增益隐藏失败。pursuit 近时段速度窗口停点误差约 -.0284 m，
减速 RMS .1712 m/s；动力学为 synthetic，不能替代实车结论。

## 首次 G2 实车开发

`/data/runs/b2d/controller/development/` 的 1773/carla 实际到达终点附近。汇总脚本因 nullable
target speed 比较失败而退出，全部 control/trajectory/validation trace 已保存；此轮标为开发失败。
发现静止时极小负速度导致 invalid_motion，终点 GNSS 抖动导致虚假再起步。agent 子代理正在修复
近零速度边界与 route 停车状态；controller 子代理正在独立修订 G2 指标和失败保存逻辑。
下一轮使用新目录，禁止覆盖第一次失败证据。

## 完成G2、smoke与数据保留工具

上述问题已修复。development2另暴露验证器在4.95s抢先判blocked的问题；修复后在development3完整重跑12例，
CARLA和pursuit各4/4通过G2，TCP在26966横向RMS=.7028m、p95=1.5254m、巡航速度RMS=.6962m/s，未通过。
三个preset都完成全部开发路线；完整5s停车位移均小于.001m。坡道保持仍须另测。

官方smoke2390、pursuit、无人工cap：completion100、composed100，216tick，attempt29.3s。
这是policy=none的route诊断，不代表真实模型闭环成绩。

提交18571c0冻结真实控制与验证代码；Dev10首轮运行中核验9个核心文件与此commit逐字节一致。
提交5f7cb0c加入源码/输入快照、原始文件SHA256目录、比较和绘图工具、caller-owned坡道诊断。
archive/compare/plot拒绝覆盖非空结果目录。原始图figures和figures-v2均保留，后续结果使用新目录。
早期失败版本缺少完整逐文件快照的限制已写进文章素材，不补造追溯证据。

Dev10 seed0当前：CARLA9/10完成、均值94.726%；TCP9/10、94.914%。TCP组的27494首尝试遭遇server rc139，
原日志与部分轨迹保留；自动换端口后第二尝试成功。三组复用server，只有故障触发重启。
由于两参考的completion与横向误差排序不同，下一轮扩为三preset完整seed1与6条保留集，不以微小completion差挑基线。
相对初计划增加16个route实验，预计增加约10min，仍在原server预算内；在pursuit首轮完成前固定此扩展。

发现route adapter的起点连接段可将输入轨迹导数推高于配置8m/s；全组冻结相同实现，原速度记录不篡改，
明确区分该轨迹导数与G2独立真值巡航速度。真实planner接入前需单独解决空间路线到可行定时轨迹的边界。
用户将稍后自行GH登录；本地提交继续，push权限阻塞不重复请求或尝试。


## v1完整证据封存；v2独立开发继续

2026-09-22：两轮Dev10+6条保留集均结束，完整[78行正式结果/81次attempt](results/v1-full/README.md)已生成到新目录。
Dev10三组两seed均9/10完成；保留集三组5/6、均值completion89.2817%，pursuit全程CTE0.56793m仍差于CARLA0.44881m。
保留集pursuit/2084首attempt在telemetry前server rc139，重试成功；v1总计3次额外基础设施attempt全部保留。
新[保留集图](figures/v1-holdout/README.md)保留19次attempt和缺失提示，先前图表不覆盖。

实际Town04坡度6.357°/11.14%，pursuit全零轨迹静止保持5s通过，101帧位移/速度记录均0；仅验证settle后制动保持，
不冒称坡道接近停车或sensor-score验收。[坡道summary](results/v1-full/slope-summary.json)。

新增真实S弯Town12/17563@6m/s三组到终点且无碰撞，但CARLA/TCP/pursuit独立巡航RMS分别0.66327/0.92157/0.63632m/s，
均高于0.5门槛；TCP横向RMS0.90514m/p951.89419m也失败。重算旧validator口径与summary精确一致，不能归咎报告速度定义错误。
[S弯summary与速度诊断](results/v1-full/README.md)。

确认两阶段结束事件和旧server PID消失后，新增[confirmation](results/raw-file-index/confirmation-v1.json)与
[S弯](results/raw-file-index/development-s-v1.json)逐文件SHA256索引，分别679/62个文件；无原始证据改写。

v1现在明确作为冻结基线，无合格新默认。v2在`/data/worktrees/jev-drive-controller-v2`独立开发，处理轨迹桥接和后续有限lookahead对照；
v1的失败与保留集不因v2工作重标为成功，也不用于逐路线调参。本次仅补文章、索引和图表，不改控制代码、不开CARLA、不做git操作。

## 2026-09-23：闭环反馈迭代与异常输入修复

v2轨迹起点定时修复、6m/s隔离诊断、v3 PI与v4唯一降Kp复验均完成；
完整过程与图表见[第二轮记录](iteration-v2.md)。v4候选pursuit max + PI .5/.25通过六项G2条件，
CARLA横向配相同PI参考也6/6通过，additive pursuit为5/6。Dev10前已冻结候选与两个参考。

原2390 smoke两次在86.11%出现Agent crashed。增加motion.jsonl后确认第182帧仅罗盘NaN，
修复1ee2eb2提供最长.2s陀螺仪航向预测与超时制动/定位复位。正常181帧及46条轨迹重放与旧版完全一致；
126项测试通过。新的smoke-v4-recovery正在完整闭环验证，正式60条尚未启动。
两次失败目录与输入重放证据均保留：[罗盘异常重放](results/smoke-v4-motion-replay/README.md)。

另完成18对v3/v4全程物理舒适性诊断，纵向加速度RMS下降21.5%、纵向jerk下降25.9%，横向加速度上升2.0%。
这不是官方Smoothness或DS；[全部样本、图与口径](results/comfort-v3-v4-v1/README.md)已保留。


00:27起：恢复版smoke官方100/100通过后，冻结60条formal-v4启动。第一组CARLA横向+PI参考seed0于00:33完成，
9/10驾驶完成，346.8s全部attempt耗时；25424施工碰撞后达到4000tick的官方TickRuntime。
完整三组/双seed未齐，不提前宣布候选优势或新默认；实际模型对照边界见[迭代记录](iteration-v2.md)。
