# DESIRE-ROUTE 方案草图（未启动；依据第 332 条）：路线派生 desire，与路线记忆共用一条通路

第 332 条：P0 的 laneChange 通路是**部分、分级**的（开环 4 s 1.04 m、8 s 1.54 m，方向 100 %，desire_state 读到 1.000），P2-F 砍掉 69 %、左变道几乎全砍（1.09 → 0.03 m），
YR / RSG 恢复到 65 % 但左边仍死（0.10–0.20）；路线记忆清空不改 desire 响应（两者今天互不干扰，也互不合作）。turn desire 在路口转弯行上不是转弯通路（同侧 −3.3°，反侧 +19.5° 把弯掰直），
只在慢速直路上有 10.6°。所以：**变道走 desire，转弯不走 desire**（转弯继续由路线记忆分支负责）；desire 与路线记忆从同一个路线派生，一个 builder，train / serve 同一函数。

## 1. driver：由收到的路线派生 desire（`lib/route_desire.py`，driver 与离线行共用）

```python
LC_MIN_V, LANE_W = 8.9, 3.5            # DesireHelper LANE_CHANGE_SPEED_MIN (20 mph); nominal lane width
T_MAX, ON_M, OFF_M = 10.0, 1.75, 0.5   # LANE_CHANGE_TIME_MAX; start when the route sits >= half a lane aside; done within 0.5 m

def lc_intent(mem):                    # mem: rmem point block (40, 5) of the decision (ordered layout), rear-axle frame, left +
    v, x, y, c, s = mem.T; y = 20 * y
    ok = lambda a: v[rmem_slot(a)] > .5
    if not all(ok(a) for a in (0, 20, 40, 60)) or rmem.turn_dist(mem) < 60: return 0, 0.0      # a turn ahead: no lane change, the branch turns
    y40, y60 = y[rmem_slot(40)], y[rmem_slot(60)]
    head = np.degrees(np.abs(np.arctan2(s, c)[rmem_slot(30):rmem_slot(60) + 1])).max()
    if abs(y40) < ON_M or abs(y60 - y40) > 1.0 or head > 10: return 0, abs(y[rmem_slot(10)])
    return int(np.sign(y40)), abs(y[rmem_slot(10)])                                              # side, near offset still to cover

class RouteDesire:                     # openpilot desire_helper.py state machine, the blinker = the route's lane-change intent
    def __init__(self): self.state, self.side, self.t_on, self.t_start = "off", 0, None, None
    def step(self, t, v, mem, ds):     # ds: the model's desire_state of the last decision (softmax 8)
        side, near = lc_intent(mem)
        if self.state == "off" and side and v >= LC_MIN_V:
            self.state, self.side, self.t_start = "starting", side, t; self.t_on = t                 # rising edge -> one pulse
        elif self.state == "starting" and (near < OFF_M or ds[3 if self.side > 0 else 4] < 0.01):
            self.state = "finishing"                                                                # route lane reached / model done
        elif self.state == "finishing" and near < OFF_M:
            self.state, self.side, self.t_on = "off", 0, None
        if self.state != "off" and (t - self.t_start > T_MAX or v < LC_MIN_V - 1.0 or side == -self.side):
            self.state, self.side, self.t_on = "off", 0, None                                       # abort (timeout / slowed / route flipped)
        return {1: 3, -1: 4}.get(self.side, 0) if self.state != "off" else 0

def desire_queue(d, t_on, t):          # modeld's queue as the stateless policy reads it: (1, 33, 8), block 32 = newest 0.2 s
    q = np.zeros((1, 33, 8), np.float32)
    if d and t_on is not None and t - t_on < 6.6: q[0, 32 - int((t - t_on) / 0.2), d] = 1.0
    return q
```

- driver（`pai_driver.drive` / `sh30_driver`）：每个决策 `d = rd.step(t, v, mem_o, last_ds)`，`pai_core.plan(..., desire=desire_queue(d, rd.t_on, t))`；
  `PModel.forward` 加一个可选参数 `desire`（None = 今天的 0，逐位不变），在 `policy_feeds` 之后写 `_to_copy`。决策日志记 state / side / t_on / 喂入块。
- 不用 turn desire（第 332 条：路口上它掰直弯）；路线三值命令 one-hot 保持（与 d284 相同）。interface：`command.channel = desire-route`（semi，路线派生，同 B2D 的声明）。
- 时序与 openpilot 一致：一次上升沿、保持到完成 / 超时 10 s；中途路线翻向即中止（第 304 条的「先偏后直」由状态机而不是逐帧命令决定）。

## 2. 训练：保住 / 恢复 desire 响应，而不是另开一条并行通路

同一 `lc_intent` 规则作用在训练行自己的路线上（NAVSIM 行：事后路线 / 车道匹配；on-policy 行：喂入的路线记忆），所以训练里 desire 出现的时刻与 serve 一样。

- **D1 变道行带 desire（模仿）**：navtrain 的 `lc` 行（rf_cfpool，ok 2 615 行）+ on-policy pool 里规则触发的行：喂 `desire_queue(d, t_on, t0)`，`t_on` = 规则在该 log 上首次触发的时刻
  （年龄 0–2 s 的行都有），目标 = 人的真实变道轨迹。adapter 学到「desire 在 → 让它过去」，人做的就是 desire 要的。
- **D2 desire 增量蒸馏（保通路）**：每个 batch 的锚行（adapter 关）与 1/8 普通行随机喂一个 LC desire（年龄 ∈ {0, 0.4, 1, 2} s），损失只在 plan 的 desire 增量上：
  `|| (plan_θ(x, d) − plan_θ(x, 0)) − (plan_P0(x, d) − plan_P0(x, 0)) ||`（教师 = shipped Cinque 同一输入，离线存 `teacher_desire.npz`）；none 的 plan 不受这一项影响。
  这让被训练的 plan pathway 不再把 desire 权重洗掉（P2-F 的 69 % / 左边 97 % 损失就是在这里发生的）。
- **D3 守卫**：desire 全 0 的行逐位与无 D1 / D2 的配方相同（开关关 = 原配方）；dev ADE 线 +0.01 m；本实验的读数（C_fast 4 s ≥ 0.9 × P0、左右各 ≥ 0.8 m、方向 ≥ 0.95）当离线闸；
  P_lca 路线一侧响应 ≥ 1.5 m；闭环 held 452 与 PAI 的地图核过的变道点（d299 / d302 的 6 / 23 个）完成数当主读。
- 一个联合臂（RSG-L18 配方 + D1 + D2）+ weight 0 对照，2 seed；1 000 步两剂量 pilot（D2 权重 1 / 3）先过离线闸再续训。

## 3. 风险

- P0 本身在开环只给 1 m（帧不动）；闭环里帧跟着横移，响应是否累积成一条车道没有测过：先在 PAI 上只开 driver 侧（D0 = 只喂 desire，不训练）看 RSG-L18 右变道（保留 1.27 m）能否完成，
  0 卡时训练成本，便宜的第一步。
- 左变道在所有训练阶段都没了：D0 只对右变道有意义，左边要 D2 才能回来。
