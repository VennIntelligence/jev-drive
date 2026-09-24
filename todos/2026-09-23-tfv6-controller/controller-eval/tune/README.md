# C 纵向迭代（dev L1 调参）

目标：保住 C/D 的跟踪精度，降低额外 jerk。只用 dev 路线（`../l1-tune-dev.xml`，15 条，均为控制器开发中用过的路线，不在 held-out 内），ramp + profile 两类参考，p01。held-out 40 条只在最后对冻结的一个候选跑一次。

机制（见 `scripts/b2d_controller.py` 的 `longitudinal_mode='accel'`）：C 的 PI effort 为负就直接刹车，而 MKZ 滑行本身约 −2.8 m/s²、刹车最轻一点就约 −3.2 m/s²，plan 要 −0.5 m/s² 时 C 也会点刹，油门刹车来回切。`accel` 模式输出期望加速度 = plan 前馈 (v[0.5,1]−v[0,0.5])/0.5 + PI(速度误差)，做 jerk 限幅，再用实测油门/刹车→加速度表反解，能靠松油门实现的减速不碰刹车。

**选优规则（跑之前写定）**：合格 = dev 上 primary 中位数 ≤ 1.05 × D 的中位数，且 collision+blocked 次数 ≤ D；合格者中额外 jerk 中位数最小者胜，平手取 kp 小者。横向一律用 D 的设置（rear_slip + ackermann）。

第一轮网格：E1 kp1/jerk4、E2 kp0.5/jerk4、E3 kp2/jerk4、E4 kp1/jerk2、E5 kp1/jerk8（ki 0.3，刹车方向 jerk 限 8），对照 B/C/D。
