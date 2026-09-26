"""Night queue 3, lane A: CARLA generation for P6 (todos/2026-09-26-night-queue-3.md, Q1 / Q3 and the [A] entries).

  need-v0    the v0 re-record: every world lane C's exam reads (processed/carla_p6/nq3_exam_frames.parquet) and the last
             tick k it reads -> runs/nq3/a/v0rr/{need.json, ids.txt}, research/results/nq3/q1/v0rr_worlds.csv
  build-v1   P6 v1: route pool (Bench2Drive 0.0.4 val clips + clips cut from the Leaderboard 2.0 long routes,
             scripts/nq3_clips.py), per-class selection, town hold-out, variant XML incl. the recovery worlds
             -> runs/nq3/a/v1/{pairs.xml, need.json, ids_*.txt}, research/results/nq3/q3/{routes,cases}.csv
  ids        worlds of an id file not yet done in a generation dir (comma list, for scripts/nq3_a.sh)
  check-det  E1: a re-recorded world's expert against its P6 v0 recording, tick by tick up to need_k
  recovery   the recovery smoke's gate: share of shifted worlds whose expert is back within |d| < 0.3 m by 3 s
  blue-plan  scripts/top10_t3_blue.py's plan over a generation dir (world, attempt dir, referenced camera ticks)
  status     progress of a generation dir (done / total, failures, rate, ETA) as markdown
"""
import json
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd

from . import p5_pairs as P
from . import p6
from .common import data_dir, get_logger

log = get_logger(__name__)
REPO = Path(__file__).resolve().parents[1]
RES = REPO / "research" / "results" / "nq3"
B2D = "third_party/Bench2Drive/leaderboard/data/"
OBST = ("Accident", "ConstructionObstacle", "ParkedObstacle", "HazardAtSideLane", "AccidentTwoWays",
        "ConstructionObstacleTwoWays", "ParkedObstacleTwoWays", "HazardAtSideLaneTwoWays", "VehicleOpensDoorTwoWays")
TEST_TOWNS = ("Town13",)
NEW_PER_CLASS, NEW_TEST_PER_CLASS = 16, 4
# recovery worlds: own id space (base + 5e6), TM seed 0 like the seed-0 main worlds; shift in m, left positive
REC_BASE = 5_000_000
REC = {"center": (1, 0.0, False), "center_wnull": (2, 0.0, True), "L15": (3, 1.5, False), "L10": (4, 1.0, False),
       "R10": (5, -1.0, False), "R15": (6, -1.5, False)}
REC_TICKS = 200                   # record 10 s after birth (the reading is 1-3 s after birth)
# worlds with the oncoming flow populated from the first tick: not reproducible run to run even under the P6 v0
# recorder itself (smoke, [A] 17:35), so E1 exclusions there are per frame and they do not count for the E1 breaker
FLOW = ("x11", "x01", "mirror")


def root(*p) -> Path:
    d = data_dir() / "runs" / "nq3" / "a" / Path(*p)
    d.mkdir(parents=True, exist_ok=True)
    return d


def _res(sec: str) -> Path:
    d = RES / sec
    d.mkdir(parents=True, exist_ok=True)
    return d


# ---------------------------------------------------------------- recorder configs

def configs():
    """scripts/nq3_recorder.py configs: P6 v0's PDM-Lite recorder settings plus the lane's rigs and shadows.
    v0rr   lean rig (v0's cameras spawned, not rendered), BridgeDrive shadow, BLUE camera at 5 Hz, need_k stop
    full   P6 v0's rig (Waymo cameras 5 Hz, visibility), TFv6 + BridgeDrive shadows, BLUE camera (the smoke vs v0)
    v1     full, recorded to 15 s after the ego has passed the obstacle; recovery worlds stop at their need_k
    e1     lean rig without shadows or BLUE: the v1 determinism re-drive"""
    d = data_dir()
    bl = str(d / "third_party/bridgedrive/lead")
    produce = " ".join("produce_%s=False" % k for k in ("demo_image", "demo_video", "debug_image", "debug_video",
                                                        "input_image", "input_video", "grid_image", "grid_video", "input_log"))
    bd = {"model": "bridgedrive", "python": str(d / "envs/bridgedrive/bin/python"), "lead": bl,
          "model_dir": str(d / "models/bridgedrive"),       # model_BridgeDrive_m2_k60_0030.pth, T3's
          "env": {"LEAD_CLOSED_LOOP_CONFIG": "steer_modality=route throttle_modality=target_speed "
                  "brake_modality=target_speed step_num=20 diffusion_speed=False " + produce,
                  "LEAD_TRAINING_CONFIG": "diffusion_speed=False plan_anchor_path=%s/anchor_utils/anchor_data/"
                  "lead_cp_kmeans_60_10.npy" % bl, "SAVE_PATH": str(d / "runs/top10_t3/lead_save")}}
    tf = {"model": "tfv6", "python": str(d / "envs/p5v1-pdm/bin/python"), "lead": str(d / "third_party/scout/lead-cvpr2026"),
          "model_dir": json.loads((d / "runs/p5_pairs/agent_config.json").read_text())["tfv6_model_dir"],
          "env": {"SAVE_PATH": str(d / "runs/p6/lead_save")}}
    p6v0 = {"save_threads": 3, "driver": "pdm_lite", "after_trigger_s": 40.0, "stuck_s": 40.0, "max_sim_s": 70.0,
            "record_props": True, "pass_stop_s": 8.0}       # runs/p6/agent-p6.json, P6 v0
    a = root()
    cfgs = {"v0rr": dict(p6v0, rig="lean", cam_period=1, blue=1, shadows=[bd], need=str(a / "v0rr/need.json")),
            "full": dict(p6v0, rig="p5", cam_period=4, blue=1, shadows=[tf, bd]),   # smoke only: decimated cameras slip
            "full_c1": dict(p6v0, rig="p5", cam_period=1, blue=1, shadows=[tf, bd]),
            "v1": dict(p6v0, rig="p5", cam_period=1, blue=1, shadows=[tf, bd], pass_stop_s=15.0,
                       need=str(a / "v1/need.json")),
            "e1": dict(p6v0, rig="lean", cam_period=1, blue=0, shadows=[], pass_stop_s=15.0, need=str(a / "v1/need.json"))}
    for k, v in cfgs.items():
        (a / ("agent_%s.json" % k)).write_text(json.dumps(v, indent=1))
    log.info("configs: %s", sorted(cfgs))


# ---------------------------------------------------------------- v0 re-record

def need_v0():
    fr = pd.read_parquet(data_dir() / "processed" / "carla_p6" / "nq3_exam_frames.parquet")
    fr["rid"] = fr.frame_name.str.split("-").str[0]
    need = fr.groupby("rid").k.max()
    order = [v for v in p6.variants(p6.cases()) if v in need.index]
    d = root("v0rr")
    (d / "need.json").write_text(json.dumps({r: int(need[r]) for r in order}))
    (d / "ids.txt").write_text(",".join(order))
    rows = fr.groupby(["rid", "base_id", "seed", "world"]).agg(frames=("k", "size"), k_min=("k", "min"),
                                                              k_max=("k", "max"),
                                                              readings=("reading", lambda r: ",".join(sorted(set(r)))))
    rows.reset_index().to_csv(_res("q1") / "v0rr_worlds.csv", index=False)
    log.info("v0 re-record: %d worlds %s, %d ticks to record", len(order),
             rows.reset_index().groupby("world").size().to_dict(), int(sum(need[r] + 3 for r in order)))


def attempt(gen: Path, rid: str) -> Path | None:
    f = gen / "done" / (rid + ".json")
    if not f.exists():
        return None
    return gen / "attempts" / rid / str(json.loads(f.read_text())["attempt"])


def _pose(adir: Path) -> pd.DataFrame:
    p = pd.read_json(adir / "pose.jsonl", lines=True).drop_duplicates("frame")
    p["k"] = (p.t / P.TICK).round().astype(int)
    return p.set_index("k")


def check_det(gen: Path, need_file: Path | None, ids=None, ref: Path | None = None) -> pd.DataFrame:
    """E1 (T3's, on P6): per world, ticks compared up to need_k (default: the shorter recording), max position /
    heading / speed difference, first tick past P5's divergence threshold (1 cm or 0.1 deg), and whether the
    camera-tick grids agree. Reference: the P6 v0 recording (default) or another generation dir (v1's re-drive)."""
    need_k = json.loads(Path(need_file).read_text()) if need_file else {}
    g0, rows = ref or p6.root("gen"), []
    for rid in ids or sorted(need_k):
        a, o = attempt(gen, rid), attempt(g0, rid)
        if a is None or o is None:
            continue
        A, B = _pose(a), _pose(o)
        lim = need_k.get(rid, min(A.index.max(), B.index.max()))
        m = A[["x", "y", "yaw", "vx", "vy"]].join(B[["x", "y", "yaw", "vx", "vy"]], rsuffix="_o", how="inner")
        m = m[m.index <= lim]
        d = np.hypot(m.x - m.x_o, m.y - m.y_o)
        dy = np.abs((m.yaw - m.yaw_o + 180) % 360 - 180)
        dv = np.hypot(m.vx - m.vx_o, m.vy - m.vy_o)
        bad = np.flatnonzero((d >= P.DIV_M) | (dy >= P.DIV_DEG))
        ka, ko = (set((pd.read_json(x / "frames.jsonl", lines=True).t / P.TICK).round().astype(int)) for x in (a, o))
        try:
            world = p6.parse_id(rid)[1]
        except KeyError:
            world = "rec"
        rows.append({"rid": rid, "world": world, "flow": world in FLOW, "ticks": len(m), "need_k": lim, "max_pos_m": float(d.max()), "max_yaw_deg": float(dy.max()),
                     "max_dv_mps": float(dv.max()), "first_div_k": int(m.index[bad[0]]) if len(bad) else None,
                     "cam_grid_same": {k for k in ka if k <= lim} == {k for k in ko if k <= lim}})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- v1

def _single(route) -> ET.Element | None:
    sc = list(route.iter("scenario"))
    return sc[0] if len(sc) == 1 and sc[0].get("type") in OBST else None


def pool() -> pd.DataFrame:
    """Every candidate route: Bench2Drive 0.0.4 val clips (not in the 220 set, not a known crasher) and the long-route
    clips of scripts/nq3_clips.py; the v0 routes (220 set) are listed with source v0. One route per scenario instance:
    a candidate whose trigger point lies within 5 m of an earlier one in the same town (v0, then val, then long-route
    clips; Bench2Drive cut its own clips from the same long routes) is dropped."""
    rows = []

    def add(r, source, **kw):
        sc = _single(r)
        if sc is None:
            return
        tp = sc.find("trigger_point")
        rows.append({"base_id": r.get("id"), "town": r.get("town"), "scenario": sc.get("type"), "source": source,
                     "tx": float(tp.get("x")), "ty": float(tp.get("y")), **kw})
    v0 = ET.parse(data_dir() / B2D / "bench2drive220.xml").getroot().findall("route")
    v0_ids = {r.get("id") for r in v0}
    for r in v0:
        if r.get("id") not in P.CRASHERS:
            add(r, "v0")
    for r in ET.parse(data_dir() / B2D / "bench2drive_0.0.4_val.xml").getroot().findall("route"):
        if r.get("id") not in v0_ids and r.get("id") not in P.CRASHERS:
            add(r, "b2d_val")
    for r in ET.parse(root("q3") / "clips.xml").getroot().findall("route"):
        add(r, "lb2_clip", origin=r.get("source"))
    df = pd.DataFrame(rows)
    keep = np.ones(len(df), bool)
    for town, g in df.groupby("town"):
        xy = g[["tx", "ty"]].to_numpy()
        for j, i in enumerate(g.index):
            if j and df.source[i] != "v0" and (np.hypot(*(xy[:j] - xy[j]).T) < 5.0)[keep[g.index[:j]]].any():
                keep[i] = False
    log.info("route pool: %d candidates, %d dropped as the same scenario instance", len(df), int((~keep).sum()))
    return df[keep].reset_index(drop=True)


def select(pl: pd.DataFrame, seed: int = 0) -> pd.DataFrame:
    """Per class NEW_PER_CLASS new routes: NEW_TEST_PER_CLASS in the test towns, the rest elsewhere; Bench2Drive's own
    val clips before long-route clips, random order (seed 0) within a source; the v0 routes all kept. A class short on
    one side is filled from the other (reported by the split counts)."""
    rng = np.random.default_rng(seed)
    pl = pl.assign(u=rng.random(len(pl)), pri=pl.source.map({"v0": 0, "b2d_val": 1, "lb2_clip": 2}))
    pl["test"] = pl.town.isin(TEST_TOWNS)
    keep = [pl[pl.source == "v0"]]
    for c in OBST:
        cand = pl[(pl.scenario == c) & (pl.source != "v0")].sort_values(["pri", "u"])
        t = cand[cand.test].head(NEW_TEST_PER_CLASS)
        o = cand[~cand.test].head(NEW_PER_CLASS - len(t))
        if len(t) + len(o) < NEW_PER_CLASS:
            t = cand[cand.test].head(NEW_PER_CLASS - len(o))
        keep.append(pd.concat([t, o]))
    s = pd.concat(keep, ignore_index=True).drop(columns=["u", "pri"])
    s["cls"] = s.scenario.map(p6.CLASS)
    s["split"] = np.where(s.test, "test", "train")
    return s.drop(columns=["test"])


def rec_id(base: str, name: str) -> str:
    return str((int(base) + REC_BASE) * 100 + REC[name][0] * 10)


def build_v1():
    """Variant XML of P6 v1 (seed 0): the new routes' main worlds as P6 v0 builds them, and the recovery worlds on every
    1W route (new and v0): obstacle hidden (x00's hook), ego born shifted, the centre twin and its weather null."""
    pl = pool()
    sel = select(pl)
    src = {}
    for f in ("bench2drive220.xml", "bench2drive_0.0.4_val.xml"):
        src.update({r.get("id"): r for r in ET.parse(data_dir() / B2D / f).getroot().findall("route")})
    src.update({r.get("id"): r for r in ET.parse(root("q3") / "clips.xml").getroot().findall("route")})
    out, cases, need = ET.Element("routes"), [], {}
    for _, s in sel.sort_values(["town", "base_id"]).iterrows():
        r0 = src[s.base_id]
        if s.source != "v0":
            ws = p6.worlds_of(s.cls, 0)
            for w in ws:
                out.append(p6._variant(r0, s.cls, w, 0))
            cases.append({"base_id": s.base_id, "town": s.town, "scenario": s.scenario, "cls": s.cls, "seed": 0,
                          "split": s.split, "source": s.source,
                          **{w: p6.variant_id(s.base_id, w, 0) if w in ws else "" for w in p6.WORLDS}})
        if s.cls == "1W":
            row = {}
            for name, (code, shift, wnull) in REC.items():
                v = p6._variant(r0, s.cls, "wnull" if wnull else "x00", 0)
                v.set("id", rec_id(s.base_id, name))
                v.set("p6_shift", "%.1f" % shift)
                out.append(v)
                need[v.get("id")] = REC_TICKS
                row[name] = v.get("id")
            cases.append({"base_id": s.base_id, "town": s.town, "scenario": s.scenario, "cls": "REC", "seed": 0,
                          "split": s.split, "source": s.source, **row})
    ET.indent(out)
    d = root("v1")
    ET.ElementTree(out).write(d / "pairs.xml")
    (d / "need.json").write_text(json.dumps(need))
    c = pd.DataFrame(cases)
    c.to_csv(_res("q3") / "cases.csv", index=False)
    sel.to_csv(_res("q3") / "routes.csv", index=False)
    main = [c.loc[i, w] for i in c.index for w in p6.WORLDS if w in c and isinstance(c.loc[i, w], str) and c.loc[i, w]]
    rec = [c.loc[i, n] for i in c.index for n in REC if n in c and isinstance(c.loc[i, n], str) and c.loc[i, n]]
    (d / "ids_main.txt").write_text(",".join(main))
    (d / "ids_rec.txt").write_text(",".join(rec))
    by = sel.groupby(["scenario", "split"]).size().unstack(fill_value=0)
    log.info("v1: %d routes (%d new), %d main worlds, %d recovery worlds; test share %.2f\n%s", len(sel),
             int((sel.source != "v0").sum()), len(main), len(rec), (sel.split == "test").mean(), by)
    return sel, c


def ids(file: str, gen: str, head: int = 0):
    want = [x for x in Path(file).read_text().strip().split(",") if x]
    todo = [v for v in want if not (Path(gen) / "done" / (v + ".json")).exists()]
    print(",".join(todo[:head] if head else todo))


def smoke_ids() -> list[str]:
    """Q3 [A] 17:15 item 4: the v0 1W routes by id, the first of each class and Accident's second; odd ones (+1.0, -1.5),
    even ones (+1.5, -1.0)."""
    r = pd.read_csv(_res("q3") / "routes.csv", dtype={"base_id": str})
    r = r[(r.source == "v0") & (r.cls == "1W")].assign(n=lambda d: d.base_id.astype(int)).sort_values("n")
    pick = [g.base_id.iloc[0] for _, g in r.groupby("scenario", sort=True)]
    pick.append(r[r.scenario == "Accident"].base_id.iloc[1])
    out = []
    for i, b in enumerate(pick, 1):
        out += [rec_id(b, n) for n in (("L10", "R15") if i % 2 else ("L15", "R10"))]
    return out


def e1_ids(frac: float = 0.05, seed: int = 0) -> list[str]:
    w = [x for x in (root("v1") / "ids_main.txt").read_text().strip().split(",") if x]
    rng = np.random.default_rng(seed)
    return sorted(rng.choice(w, int(round(frac * len(w))), replace=False).tolist())


# ---------------------------------------------------------------- recovery smoke

def recovery(gen: Path, ids_: list[str]) -> pd.DataFrame:
    """Per shifted world: initial offset d0, |d| at 1 / 2 / 3 s after birth, first tick back within 0.3 m."""
    rows = []
    for rid in ids_:
        a = attempt(gen, rid)
        if a is None:
            continue
        W = P.load_world(a)
        lat = p6.lateral(W)
        k0 = int(lat.index.min())
        d = lat.d
        back = d.index[(np.abs(d) < 0.3) & (d.index > k0)]
        rows.append({"rid": rid, "shift": json.loads((a / "p6_shift.json").read_text())["shift_m"], "d0": round(float(d.iloc[0]), 3),
                     **{"d_%ds" % s: round(float(d.get(k0 + 20 * s, np.nan)), 3) for s in (1, 2, 3)},
                     "t_back_s": round((int(back[0]) - k0) * P.TICK, 2) if len(back) else None,
                     "v_3s": round(float(lat.v.get(k0 + 60, np.nan)), 2)})
    df = pd.DataFrame(rows)
    if len(df):
        df["back_by_3s"] = df.t_back_s.notna() & (df.t_back_s <= 3.0)
    return df


# ---------------------------------------------------------------- BLUE / SimLingo plan

def blue_plan(gen: Path, out: Path, need_file: Path | None = None, frames: str = "exam"):
    """The offline runner's plan: world id, attempt dir and the camera ticks to read (frames "exam": lane C's exam ticks
    for v0 re-record worlds; "all": every camera tick recorded)."""
    ks = {}
    if frames == "exam":
        fr = pd.read_parquet(data_dir() / "processed" / "carla_p6" / "nq3_exam_frames.parquet")
        fr["rid"] = fr.frame_name.str.split("-").str[0]
        ks = fr.groupby("rid").k.apply(lambda k: sorted(set(int(x) for x in k))).to_dict()
    plan = []
    for f in sorted((gen / "done").glob("*.json")):
        rid = f.stem
        a = attempt(gen, rid)
        if a is None or not (a / "blue_inputs.jsonl").exists():
            continue
        if frames == "exam":
            if rid not in ks:
                continue
            k = ks[rid]
        else:
            k = [json.loads(l)["k"] for l in open(a / "frames.jsonl") if json.loads(l).get("blue")]
        plan.append({"rid": rid, "adir": str(a), "ks": k})
    Path(out).write_text(json.dumps(plan))
    log.info("blue plan: %d worlds, %d frames -> %s", len(plan), sum(len(p["ks"]) for p in plan), out)


# ---------------------------------------------------------------- v1 post-processing (mechanical)

def _rec_index(g: Path, rid: str, base: str, name: str, town: str, meta: dict):
    a = attempt(g, rid)
    if a is None:
        return None
    r = P.world_rows(a, rid, town)
    if r is None:
        return None
    t, past, fut = r
    return t.assign(base_id=base, world=name, seed=0, source="p6v1", role="obs", **meta), past, fut


def v1_post(workers: int = 24):
    """Expert statistics of P6 v1 exactly as P6 v0's (jevdrive.p6._case / report) -> research/results/nq3/q3/, the
    recovery table, and the 5 Hz frame index of every finished v1 world -> processed/carla_p6_v1 (P6 v0's layout plus
    split and source; recovery worlds carry world = their recovery name) for lane C's Q2 v1."""
    from joblib import Parallel, delayed
    g, out = root("v1") / "gen", _res("q3")
    c = pd.read_csv(out / "cases.csv", dtype=str, keep_default_na=False).astype({"seed": int})
    main, rec = c[c.cls != "REC"], c[c.cls == "REC"]
    res = Parallel(workers)(delayed(p6._case)(g, r) for _, r in main.iterrows())
    pd.DataFrame([w for r in res for w in r[0]]).to_csv(out / "worlds.csv", index=False)
    pd.DataFrame([r[1] for r in res]).to_csv(out / "pairs_bypass.csv", index=False)
    pd.DataFrame([f for r in res for f in r[2]]).to_csv(out / "frame_modes.csv", index=False)
    pd.DataFrame([r[3] for r in res if r[3]]).to_csv(out / "negotiation.csv", index=False)
    p6.report(out)
    rid = [r[n] for _, r in rec.iterrows() for n in REC if r.get(n)]
    rv = recovery(g, rid)
    if len(rv):
        rv.to_csv(out / "recovery.csv", index=False)
    jobs = [(p6._world_index, (g, r[w], r.town, {"scenario": r.scenario, "cls": r.cls, "split": r.split, "src": r.source}))
            for _, r in main.iterrows() for w in p6.WORLDS if r[w]]
    jobs += [(_rec_index, (g, r[n], r.base_id, n, r.town, {"scenario": r.scenario, "cls": "REC", "split": r.split,
                                                           "src": r.source}))
             for _, r in rec.iterrows() for n in REC if r.get(n)]
    got = [x for x in Parallel(workers)(delayed(f)(*a) for f, a in jobs) if x is not None]
    t = pd.concat([x[0] for x in got], ignore_index=True)
    d = data_dir() / "processed" / "carla_p6_v1"
    d.mkdir(parents=True, exist_ok=True)
    t.to_parquet(d / "index.parquet", index=False)
    np.save(d / "past.npy", np.concatenate([x[1] for x in got]))
    np.save(d / "future.npy", np.concatenate([x[2] for x in got]))
    log.info("carla_p6_v1: %d frames from %d worlds; by split / world %s", len(t), len(got),
             t.groupby(["split", "world"]).size().to_dict())


# ---------------------------------------------------------------- Q1: the CARLA-rig examinees on the v0 exam

def _rig_preds(gen: Path, t: pd.DataFrame, ok: dict, blue_dirs: dict) -> tuple[dict, dict]:
    """(examinee -> (n, 20, 2) aligned to the P6 v0 index, notes). Re-recorded worlds are matched to the index by
    (world id, tick k); ok = {world: first tick at which its expert left the P6 v0 trajectory (inf if never)}, and
    frames from that tick on are left NaN (worlds missing from ok: all NaN)."""
    rid = t.frame_name.str.split("-").str[0]
    pos = pd.Series(np.arange(len(t)), index=pd.MultiIndex.from_arrays([rid, t.k.astype(int)]))
    n = len(t)
    wp, ts = np.full((n, 20, 2), np.nan, np.float32), np.full(n, np.nan, np.float32)
    for f in sorted((gen / "done").glob("*.json")):
        r = f.stem
        a = attempt(gen, r)
        if r not in ok or a is None or not (a / "bridgedrive.jsonl").exists():
            continue
        cls = np.asarray(json.loads((a / "nq3_summary.json").read_text())["shadows"][0]["target_speed_classes"], float)
        for line in open(a / "bridgedrive.jsonl"):
            q = json.loads(line)
            i = pos.get((r, int(q["k"])))
            if i is None or "pred_future_waypoints" not in q or int(q["k"]) >= ok[r]:
                continue
            wp[i, :8] = np.asarray(q["pred_future_waypoints"], np.float32)
            ts[i] = float(np.dot(q["pred_target_speed_distribution"], cls))
    preds, notes = {"BridgeDrive waypoint": wp}, {}
    tsf = np.full((n, 20, 2), np.nan, np.float32)
    tsf[..., 0] = ts[:, None] * (0.25 * np.arange(1, 21))[None]
    preds["BridgeDrive target speed"] = tsf
    notes["BridgeDrive target speed"] = "longitudinal only (route + target speed channel, the one it drives with)"
    for name, d in blue_dirs.items():
        a = np.full((n, 20, 2), np.nan, np.float32)
        for f in sorted(Path(d).glob("*.json")):
            z = json.loads(f.read_text())
            if z["rid"] not in ok:
                continue
            for q in z["frames"]:
                i = pos.get((z["rid"], int(q["k"])))
                if i is not None and int(q["k"]) < ok[z["rid"]]:
                    w = np.asarray(q["wps"], np.float32)
                    a[i, :len(w)] = w[:20]
        preds[name] = a
    return preds, notes


def judge_rig(gen: Path, e1_csv: Path, blue_dirs: dict, out: Path | None = None) -> pd.DataFrame:
    """Rule 7 (lane C's jevdrive.nq3_p6 judge, unchanged) for BridgeDrive (waypoint; target speed longitudinal only),
    BLUE and SimLingo (speed waypoints, 2.5 s), each lateral sign fixed against the expert's y(2 s) on x10 frames as
    lane C does for TFv6; frames from a world's first E1 divergence on are dropped ([A] 17:35)."""
    from . import nq3_p6 as J
    out = out or _res("q1")
    t = pd.read_parquet(data_dir() / "processed" / "carla_p6" / "index.parquet")
    fut = np.load(data_dir() / "processed" / "carla_p6" / "future.npy")
    e1 = pd.read_csv(e1_csv, dtype={"rid": str})
    e1 = e1[e1.cam_grid_same.astype(bool)]
    ok = dict(zip(e1.rid, e1.first_div_k.fillna(np.inf)))
    preds, notes = _rig_preds(gen, t, ok, blue_dirs)
    x10 = (t.world == "x10").to_numpy()
    for name, pr in preds.items():
        if "target speed" in name:
            continue
        sel = np.flatnonzero(~np.isnan(pr[:, 7, 1]) & x10)
        c = np.corrcoef(pr[sel, 7, 1], fut[sel, 7, 1])[0, 1] if len(sel) > 2 else np.nan
        if c < 0:
            pr[..., 1] *= -1
        notes[name] = f"y sign {'flipped' if c < 0 else 'kept'} (corr with the expert's y(2 s) on x10 frames {c:+.2f})"
    p = J.pairs("carla_p6")
    rows, per, wms = [], [], []
    for name, pr in preds.items():
        row, pc, s = J.judge_one(name, p, pr)
        if "target speed" in name:
            for k in ("bypass_flip", "lo", "hi", "gate_a", "shoulder_flip", "selective", "has_bypass", "mirror_borrow"):
                row[k] = np.nan
            row["verdict"] = "longitudinal only"
        else:
            wm = J.world_modes(s, pr)
            if len(wm):
                wms.append(J.mode_agreement(wm).assign(examinee=name))
        row["note"] = notes.get(name, "")
        row["e1_worlds_identical"] = int(np.isinf(list(ok.values())).sum())
        row["e1_worlds_diverged"] = int((~np.isinf(list(ok.values()))).sum())
        rows.append(row)
        per.append(pc)
    tab = pd.DataFrame(rows)
    tab.to_csv(out / "carla_rig_summary.csv", index=False)
    pd.concat(per).to_csv(out / "carla_rig_per_scenario.csv", index=False)
    if wms:
        wm = pd.concat(wms)
        wm.groupby(["examinee", "world"]).agree.agg(["mean", "size"]).round(3).reset_index().to_csv(
            out / "carla_rig_mode_agreement.csv", index=False)
    cols = [c for c in ["examinee", "n_frames", "routes", "tau_lat", "bypass_flip", "lo", "hi", "null_ff_oos", "gate_a",
                        "shoulder_flip", "shoulder_ref", "selective", "verdict", "stop_sub", "stop_lo", "stop_hi",
                        "neg_later_rate", "neg_agree_expert", "mirror_borrow"] if c in tab]
    (out / "carla_rig_summary.md").write_text(tab[cols].to_markdown(index=False, floatfmt=".3f") + "\n")
    log.info("\n%s", tab[cols].to_markdown(index=False, floatfmt=".3f"))
    return tab


# ---------------------------------------------------------------- status

def status(gen: Path, id_file: Path, t_start: float, est_h: float) -> str:
    want = [x for x in Path(id_file).read_text().strip().split(",") if x]
    done = [v for v in want if (gen / "done" / (v + ".json")).exists()]
    fails = 0
    ev = gen / "events.jsonl"
    if ev.exists():
        for line in open(ev):
            try:
                e = json.loads(line)
            except ValueError:
                continue
            if e.get("kind") == "route_end" and e.get("status") not in ("finished", None):
                fails += 1
    el = (time.time() - t_start) / 3600
    rate = len(done) / el if el > 0 else 0.0
    eta = (len(want) - len(done)) / rate if rate > 0 else float("nan")
    return (f"{len(done)} / {len(want)} worlds done ({len(done) / max(len(want), 1):.1%}), {fails} failed attempts; "
            f"elapsed {el:.2f} h of an estimated {est_h:.2f} h, {rate:.0f} worlds/h, ETA {eta:.2f} h")


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["configs", "need-v0", "build-v1", "ids", "check-det", "recovery", "blue-plan", "status",
                                    "pool", "smoke-ids", "e1-ids", "judge-rig", "v1-post"])
    ap.add_argument("--ref", default="")
    ap.add_argument("--e1", default="")
    ap.add_argument("--blue", default="", help="name=dir,name=dir of offline BLUE / SimLingo outputs")
    ap.add_argument("--gen", default="")
    ap.add_argument("--file", default="")
    ap.add_argument("--need", default="")
    ap.add_argument("--out", default="")
    ap.add_argument("--head", type=int, default=0)
    ap.add_argument("--frames", default="exam")
    ap.add_argument("--t0", type=float, default=0.0)
    ap.add_argument("--est-h", type=float, default=0.0)
    a = ap.parse_args()
    if a.cmd == "configs":
        configs()
    elif a.cmd == "need-v0":
        need_v0()
    elif a.cmd == "build-v1":
        build_v1()
    elif a.cmd == "pool":
        print(pool().groupby(["scenario", "source", "town"]).size().to_string())
    elif a.cmd == "ids":
        ids(a.file, a.gen, a.head)
    elif a.cmd == "v1-post":
        v1_post()
    elif a.cmd == "smoke-ids":
        print(",".join(smoke_ids()))
    elif a.cmd == "e1-ids":
        print(",".join(e1_ids()))
    elif a.cmd == "judge-rig":
        judge_rig(Path(a.gen), Path(a.e1), dict(x.split("=", 1) for x in a.blue.split(",") if x),
                  Path(a.out) if a.out else None)
    elif a.cmd == "check-det":
        df = check_det(Path(a.gen), Path(a.need) if a.need else None, a.file.split(",") if a.file else None,
                       Path(a.ref) if a.ref else None)
        if a.out:
            df.to_csv(a.out, index=False)
        ok = df.first_div_k.isna() & df.cam_grid_same
        print(df.to_string() if len(df) <= 40 else df.groupby("world").agg(n=("rid", "size"),
              identical=("first_div_k", lambda x: int(x.isna().sum()))).to_string())
        print(f"E1: {int(ok.sum())} / {len(df)} worlds identical to the reference up to need_k; without the flow worlds "
              f"{int(ok[~df.flow].sum())} / {int((~df.flow).sum())}")
    elif a.cmd == "recovery":
        df = recovery(Path(a.gen), a.file.split(","))
        if a.out:
            df.to_csv(a.out, index=False)
        print(df.to_string())
        if len(df):
            print(f"back within 0.3 m by 3 s: {int(df.back_by_3s.sum())} / {len(df)} = {df.back_by_3s.mean():.2f} (gate >= 0.80)")
    elif a.cmd == "blue-plan":
        blue_plan(Path(a.gen), Path(a.out), Path(a.need) if a.need else None, a.frames)
    elif a.cmd == "status":
        print(status(Path(a.gen), Path(a.file), a.t0, a.est_h))


if __name__ == "__main__":
    main()
