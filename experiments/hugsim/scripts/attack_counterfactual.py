"""Offline counterfactuals against HUGSIM's AttackPlanner, with the upstream planner code and the simulator's ego model.

usage (box, hugsim env, CPU): attack_counterfactual.py <run_dir> <scenario.yaml> <out_prefix>

What it does
  1. Rebuilds hug_sim.step()'s world loop without rendering: ego bicycle model (Lr+Lf = 2.7 m, dt 0.25 s) and
     planner.plan_traj() for the scripted actors (AttackPlanner / ConstantPlanner code imported from the HUGSIM checkout).
  2. Validates it: replays the recorded (acc, steer_rate) of <run_dir>/infos.pkl and compares the actor tracks with
     the recorded ones (max position error is printed).
  3. Runs scripted ego manoeuvres (brake-and-hold, lane-change swerve with speed profiles) and reports which of them
     avoid contact with every actor and stay on ground points.
The recorded run itself is the model's behaviour; the manoeuvres are what a driver with perfect knowledge of the
simulator could have done, they are not model outputs.
"""
import itertools, json, math, pickle, sys
import numpy as np, torch, yaml
sys.path.insert(0, "/root/autodl-tmp/ujs/third_party/HUGSIM")
from sim.utils.agent_controller import AttackPlanner, ConstantPlanner, constant_headaway  # noqa: E402
from sim.utils.score_calculator import create_rectangle  # noqa: E402  (the simulator's own box -> polygon)

DT, L = 0.25, 2.7
run, yml, outp = sys.argv[1], sys.argv[2], sys.argv[3]
infos = pickle.load(open(f"{run}/infos.pkl", "rb"))
cfg = yaml.safe_load(open(yml))
EGO_W, EGO_L = 1.6, 3.0
dims = [(o[3], o[4]) for o in infos[0]["obj_boxes"]]  # (w, l) per actor, recorded


def build():
    stats, ctl = [], []
    for args in cfg["plan_list"]:
        stats.append(torch.tensor([float(v) for v in args[:5]]))  # a, b, height, yaw, v
        kind, kw = args[6], args[7]
        ctl.append(AttackPlanner(**{k: v for k, v in kw.items() if k != "ATTACK_FREQ"}) if kind == "AttackPlanner" else ConstantPlanner())
    freq = next((a[7]["ATTACK_FREQ"] for a in cfg["plan_list"] if a[6] == "AttackPlanner"), 3)
    return stats, ctl, freq


def plan_step(t, ego, stats, ctl, freq):
    """upstream planner.plan_traj(), single call"""
    all_stats = torch.stack([ego] + [s[[0, 1, 3, 4]] for s in stats])
    fut = constant_headaway(all_stats, 20, DT)
    for i, c in enumerate(ctl):
        s = stats[i]
        d = torch.norm(all_stats[:, :2] - s[:2], dim=-1)
        nb = fut[torch.argsort(d)[1:4]]
        if isinstance(c, AttackPlanner):
            nxt = c.update(state=s[[0, 1, 3, 4]], unified_map=None, dt=0.1, neighbors=nb[1:], attacked_states=fut[0],
                           new_plan=((t // DT) % freq == 0))
        else:
            nxt = c.update(state=s[[0, 1, 3, 4]], dt=DT)
        n = torch.zeros_like(s); n[[0, 1, 3, 4]] = nxt.float(); n[2] = s[2]; stats[i] = n


def ego_poly(a, b, th): return create_rectangle(b, -a, EGO_W, EGO_L, -th)  # hug_sim.ego_box: (x=b, y=-a, yaw=-theta)
def act_poly(a, b, yaw, w, l): return create_rectangle(b, -a, w, l, yaw)       # hug_sim.objs_list: yaw = stat[3]


# ground cells (scene frame: a = world x, b = world z), 0.5 m grid
raw = open(f"{run}/ground.ply", "rb").read(); hdr = raw.index(b"end_header\n") + 11
pts = np.frombuffer(raw[hdr:], dtype="<f8").reshape(-1, 3)
cells = set(map(tuple, np.floor(pts[:, [0, 2]] / 0.5).astype(int)))


def on_ground(a, b):
    i, j = int(math.floor(a / 0.5)), int(math.floor(b / 0.5))
    return any((i + di, j + dj) in cells for di in (-1, 0, 1) for dj in (-1, 0, 1))


def simulate(policy, T=14.0, record=False):
    stats, ctl, freq = build()
    ab, eu = cfg.get("start_ab", [0, 0]), cfg.get("start_euler", [0, 0, 0])
    ego = dict(a=float(ab[0]), b=float(ab[1]), th=math.radians(eu[1]), v=float(cfg.get("start_velo", 1)), st=float(cfg.get("start_steer", 0)))
    t = 0.0
    for _ in range(2):  # closed_loop.py calls env.reset() twice before the first step; the planner state survives it
        plan_step(t, torch.tensor([ego["a"], ego["b"], ego["th"], ego["v"]]), stats, ctl, freq)
    log = []; hit = None; off = None
    for k in range(1, int(T / DT) + 1):
        t += DT
        plan_step(t, torch.tensor([ego["a"], ego["b"], ego["th"], ego["v"]]), stats, ctl, freq)
        acc, sr = policy(k, t, ego, stats)
        ego["v"] += acc * DT; ego["st"] += sr * DT
        th = ego["th"]; ego["a"] += ego["v"] * math.sin(th) * DT; ego["b"] += ego["v"] * math.cos(th) * DT
        ego["th"] = th + ego["v"] * math.tan(ego["st"]) / L * DT
        ep = ego_poly(ego["a"], ego["b"], ego["th"])
        if hit is None:
            for i, s in enumerate(stats):
                if ep.intersects(act_poly(float(s[0]), float(s[1]), float(s[3]), *dims[i])): hit = (k, i)
        if off is None and not all(on_ground(-y, x) for x, y in ep.exterior.coords): off = k  # polygon (x=b, y=-a) -> (a, b)
        if record: log.append((k, ego["a"], ego["b"], ego["v"], [(float(s[0]), float(s[1])) for s in stats]))
        if hit: break
    return hit, off, log


# 1. validation by replay
rec_actions = [(s["accelerate"], s["steer_rate"]) for s in infos]
def replay(k, t, ego, stats): return rec_actions[min(k, len(rec_actions) - 1)]
hit, off, log = simulate(replay, T=DT * (len(infos) + 1), record=True)
err = 0.0
for k, a, b, v, acts in log:
    if k >= len(infos): break
    for i, (aa, bb) in enumerate(acts):
        o = infos[k]["obj_boxes"][i]; err = max(err, math.hypot(bb - o[0], -aa - o[1]))
import os
if os.environ.get("DEBUG_REPLAY"):
    for k, a, b, v, acts in log[:34:3]:
        s_ = infos[k]; e = s_["ego_box"]; o = s_["obj_boxes"][0]
        print(k, "ego mine (%.2f %.2f v %.2f) rec (%.2f %.2f v %.2f)" % (b, -a, v, e[0], e[1], s_["ego_velo"]), "| atk mine (%.2f %.2f) rec (%.2f %.2f)" % (acts[0][1], -acts[0][0], o[0], o[1]))
    sys.exit()
print(f"replay: first overlap (step, actor) = {hit}; recorded run ended at step {len(infos) - 1}; max actor position error {err:.3f} m")

# 2. manoeuvres
def brake_hold(T0, dec):
    def p(k, t, ego, stats):
        if t < T0: return (0.0, 0.0)
        return (-dec if ego["v"] > 0.05 else (-ego["v"] / DT), -ego["st"] / DT)
    return p

def reverse(T0, vmin):
    def p(k, t, ego, stats):
        if t < T0: return (0.0, 0.0)
        return ((-3.0 if ego["v"] > vmin else 0.0), -ego["st"] / DT)
    return p

def swerve(T0, side, width, Tlc, acc_cmd, vmax):
    def p(k, t, ego, stats):
        acc = float(np.clip(acc_cmd if t >= T0 else 0.0, -3, 3))
        if acc > 0 and ego["v"] >= vmax: acc = 0.0
        if ego["v"] < 0: acc = max(acc, 0.0)
        s = min(max((t - T0) / Tlc, 0.0), 1.0); target = side * width * (3 * s * s - 2 * s ** 3)  # smoothstep lateral offset (a axis)
        la = max(4.0, 1.5 * abs(ego["v"]))  # pure-pursuit look-ahead
        tb = ego["b"] + la; ta = target
        alpha = math.atan2(ta - ego["a"], tb - ego["b"]) - ego["th"]
        des = math.atan2(2 * L * math.sin(alpha), la); des = float(np.clip(des, -0.6, 0.6))
        sr = float(np.clip((des - ego["st"]) / DT, -0.4 * 2.5, 0.4 * 2.5))
        return (acc, sr)
    return p

res = []
for T0 in np.arange(0, 9.01, 0.5):
    for dec in (1.5, 3.0):
        h, o, _ = simulate(brake_hold(T0, dec)); res.append(("brake_hold", T0, dec, h, o))
for T0, vmin in itertools.product(np.arange(0, 9.01, 0.5), (-2.0, -5.0)):
    h, o, _ = simulate(reverse(T0, vmin)); res.append(("reverse", T0, vmin, h, o))
for T0, side, width, Tlc, acc, vmax in itertools.product(np.arange(0, 8.01, 0.5), (-1, 1), (2.0, 3.0), (1.5, 3.0), (-3, 0, 3), (4.0, 8.0)):
    h, o, _ = simulate(swerve(T0, side, width, Tlc, acc, vmax)); res.append(("swerve", T0, (side, width, Tlc, acc, vmax), h, o))
free = [r for r in res if r[3] is None]; free_g = [r for r in free if r[4] is None]
print(f"manoeuvres tried {len(res)}; no overlap with any actor in 14 s: {len(free)}; and also on ground points throughout: {len(free_g)}")
for r in free_g[:12]: print("  ", r)
kinds = {k: [sum(1 for r in res if r[0] == k), sum(1 for r in res if r[0] == k and r[3] is None), sum(1 for r in res if r[0] == k and r[3] is None and r[4] is None)] for k in ("brake_hold", "reverse", "swerve")}
print("SUMMARY", json.dumps({"run": run, "replay_hit": hit, "replay_err_m": err, "tried_free_free_on_ground_by_kind": kinds}))
json.dump([[r[0], float(r[1]), str(r[2]), r[3], r[4]] for r in res], open(outp + "_grid.json", "w"))
