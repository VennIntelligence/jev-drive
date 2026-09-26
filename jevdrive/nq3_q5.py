"""Night queue 3, Q5: hack audit of the top-10 families (todos/2026-09-26-night-queue-3.md, Q5 and the [D] 16:50
entry, written before any Q5 number).

  req          navtest request from the frozen index (T2's request format) and the nuScenes request (T2's nusc.npz)
               with the scene / log grouping the swap arms need -> runs/nq3/q5/req/
  arms         ego-status arms per (set, model): base, S0 A0 C0 (the verdict arms), Sk Ak Ck (dataset constant),
               Ss As Cs (another frame of the same scene / log), ALL0, and H0 for WA-JEPA (history poses zeroed)
  verify       base arms against T2's stored nuScenes predictions (bitwise), WA-JEPA fp32 navtest head against T2's
               fp32 export (bitwise)
  exam-nusc    top10_t2_real.exam_nusc's metrics per arm, paired against base and CV, scene bootstrap
  nav-jobs     per-arm replay files for the devkit (v1.1 PDMS) and the job list
  nav-table    paired token bootstrap arm - base
  select       selectivity table (reactive flip - non-reactive false flip) from the stored exam tables
  frag-prep / frag-table   the rig perturbation (yaw +-0.5 deg, height +-5 cm) on 256 navtest tokens

Model-side runners: scripts/nq3_d/{drivor,wajepa}_arms.py; everything is chained by scripts/nq3_d/q5.sh.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .common import data_dir, get_logger

log = get_logger(__name__)
MODELS = ("drivor", "wajepa")
NAME = {"drivor": "DrivoR", "wajepa": "WA-JEPA"}
SETS = ("nusc", "navtest")
MAIN_ARMS = ("S0", "A0", "C0")
NAV_CAMS = ("CAM_L0", "CAM_F0", "CAM_R0", "CAM_B0")
B = 10_000
N_FRAG = 256
Z_GROUND = -0.36          # NAVSIM rear-axle frame: road surface (G0, navtrain GT box bottoms)
FRAG_ARMS = {"orig": None, "ident": (0.0, 0.0), "yaw+0.5": (0.5, 0.0), "yaw-0.5": (-0.5, 0.0), "z+5cm": (0.0, 0.05),
             "z-5cm": (0.0, -0.05)}
RESULTS = Path(__file__).resolve().parents[1] / "research" / "results" / "nq3" / "q5"


def run_dir(*p) -> Path:
    d = data_dir() / "runs" / "nq3" / "q5" / Path(*p)
    d.parent.mkdir(parents=True, exist_ok=True)
    return d


WJ_STRIDE = {"nusc": 4, "navtest": 10}     # WA-JEPA runs on every k-th request only ([D] 17:30 amendment)


def arm_names(model: str) -> list:
    if model == "wajepa":
        return ["base", "S0", "A0", "C0", "ALL0", "H0"]
    return ["base", "S0", "A0", "C0", "Sk", "Ak", "Ck", "Ss", "As", "Cs", "ALL0"]


def req_path(set_: str, model: str) -> Path:
    return run_dir("req", f"{set_}_wajepa.npz" if model == "wajepa" else f"{set_}.npz")


# ---------------------------------------------------------------- requests and arms

def req():
    from . import navsim_zs as Z, top10_t2_real as TR
    idx = Z.load_index("navtest")
    img = [[e["cams"][f][c]["path"] for f in range(4) for c in NAV_CAMS] for e in idx]
    np.savez(run_dir("req", "navtest.npz"), keys=np.array([e["token"] for e in idx]), img=np.array(img),
             hist=np.stack([e["pose"] for e in idx]).astype(np.float32),
             ego=np.stack([np.r_[e["vel"][-1], e["acc"][-1]] for e in idx]).astype(np.float32),
             cmd=np.array([int(np.argmax(e["cmd"][-1])) for e in idx]), group=np.array([e["log_name"] for e in idx]),
             t=np.array([e["timestamp"] for e in idx], np.int64))
    z = np.load(data_dir() / "runs/top10_t2/requests/nusc.npz")
    _, samples = TR.nusc_main()
    assert (z["keys"] == np.array([e["token"] for e in samples])).all()
    np.savez(run_dir("req", "nusc.npz"), **{k: z[k] for k in z.files}, group=np.array([e["scene"] for e in samples]),
             t=np.array([e["t0"] for e in samples], np.int64))
    # the fp32 navtest check: the first 16 tokens, base arm only (verify() compares with T2's fp32 export)
    n = np.load(run_dir("req", "navtest.npz"))
    np.savez(run_dir("req", "navcheck.npz"), **{k: n[k][:16] for k in n.files})
    np.savez(run_dir("req", "navcheck_arms.npz"), names=np.array(["base"]),
             ego8=np.concatenate([n["ego"][:16], np.eye(4)[n["cmd"][:16]]], 1)[None].astype(np.float32), hist=n["hist"][:16][None])
    log.info("requests: navtest %d, nusc %d", len(idx), len(samples))


def swap_index(group: np.ndarray, t: np.ndarray) -> tuple[np.ndarray, int]:
    """Partner of every row: same group, time order shifted cyclically by n // 2 (itself for groups of one)."""
    out = np.arange(len(group))
    single = 0
    for g in np.unique(group):
        r = np.flatnonzero(group == g)
        r = r[np.argsort(t[r], kind="stable")]
        if len(r) == 1:
            single += 1
            continue
        out[r] = np.roll(r, -(len(r) // 2))
    return out, single


def arms(set_: str, model: str) -> dict:
    z = np.load(run_dir("req", f"{set_}.npz"))
    if model == "wajepa":
        z = {k: z[k][::WJ_STRIDE[set_]] for k in z.files}
        np.savez(req_path(set_, model), **z)
    ego, hist = z["ego"].astype(np.float64), z["hist"].astype(np.float32)
    oh = np.eye(4)[z["cmd"]]
    base = np.concatenate([ego, oh], 1)
    sw, single = swap_index(z["group"], z["t"])
    const = {"S": ego[:, :2].mean(0), "A": ego[:, 2:4].mean(0), "C": np.eye(4)[np.bincount(z["cmd"], minlength=4).argmax()]}
    cols = {"S": slice(0, 2), "A": slice(2, 4), "C": slice(4, 8)}
    names, E, Hs = [], [], []
    for a in arm_names(model):
        e, h = base.copy(), hist.copy()
        if a == "ALL0":
            e[:] = 0
        elif a == "H0":
            h[:] = 0
        elif a != "base":
            var, how = a[0], a[1]
            c = cols[var]
            e[:, c] = 0 if how == "0" else (const[var] if how == "k" else base[sw, c])
        names.append(a)
        E.append(e)
        Hs.append(h)
    p = run_dir("req", f"{set_}_{model}_arms.npz")
    np.savez(p, names=np.array(names), ego8=np.stack(E).astype(np.float32), hist=np.stack(Hs))
    info = {"set": set_, "model": model, "arms": names, "swap_single_groups": single, "n": len(ego),
            "const": {k: np.asarray(v).tolist() for k, v in const.items()}}
    log.info("%s", json.dumps(info))
    return info


def arms_all():
    info = [arms(s, m) for s in SETS for m in MODELS]
    run_dir("req", "arms.json").write_text(json.dumps(info, indent=1))


# ---------------------------------------------------------------- equivalence checks

def verify():
    """Base arm == T2's stored nuScenes predictions (bitwise); the fp32 navtest head == T2's fp32 export (bitwise)."""
    import pickle
    res = {}
    for m in MODELS:
        p = np.load(run_dir("preds", f"nusc_{m}.npz"))
        t2 = np.load(data_dir() / "runs/top10_t2/preds" / f"nusc_{m}.npz")
        at = pd.Series(np.arange(len(t2["keys"])), index=t2["keys"])[p["keys"]].to_numpy()
        b = p["traj"][list(p["names"]).index("base")]
        d = np.abs(b - t2["traj"][at])
        if m == "wajepa":        # batched path: a numeric floor, reported, not a gate ([D] 17:30)
            disp = np.linalg.norm(b[..., :2] - t2["traj"][at][..., :2], axis=-1)
            res.update({"nusc_wajepa_batched_base_vs_t2_mean_disp_m": float(disp.mean()),
                        "nusc_wajepa_batched_base_vs_t2_p95_ade_m": float(np.percentile(disp.mean(1), 95)),
                        "nusc_wajepa_batched_base_vs_t2_maxdiff": float(d.max()), "nusc_wajepa_n": int(len(b))})
        else:
            res[f"nusc_{m}_base_vs_t2_max_abs"] = float(d.max())
    f = run_dir("preds", "navcheck_wajepa_fp32.npz")
    if f.exists():
        z = np.load(f)
        pk = sorted((data_dir() / "runs/top10_t2/navsim/wajepa").glob("*/trajectory_cache/navtest_trajectories.pkl"))[-1]
        tr = pickle.load(open(pk, "rb"))["trajectories"]
        ref = np.stack([tr[k] for k in z["keys"]])
        res["navtest_wajepa_fp32_vs_t2_export_max_abs"] = float(np.abs(z["traj"][0] - ref).max())
        res["navtest_wajepa_fp32_n"] = int(len(ref))
    run_dir("verify.json").write_text(json.dumps(res, indent=1))
    log.info("%s", json.dumps(res))
    bad = {k: v for k, v in res.items() if k.endswith("max_abs") and v != 0}
    assert not bad, f"equivalence check failed: {bad}"


# ---------------------------------------------------------------- nuScenes readout

_NUSC_SAMPLES = None


def _nusc_metrics(args):
    """VAD-style 1 / 2 / 3 s means of L2 and both collision rates for predictions `pts` on sample rows `rows`."""
    from . import nuscenes_zs as Z
    pts, rows = args
    smp = [_NUSC_SAMPLES[i] for i in rows]
    h = Z.horizons(Z.per_sample(pts.astype(np.float64), smp))
    return {"l2": (h["l2_1s"] + h["l2_2s"] + h["l2_3s"]) / 3,
            "col_vad": (h["col_vad_1s"] + h["col_vad_2s"] + h["col_vad_3s"]) / 3,
            "col_bevp": (h["col_bevp_1s"] + h["col_bevp_2s"] + h["col_bevp_3s"]) / 3}


def exam_nusc():
    from . import nuscenes_zs as Z, top10_t2_real as TR
    nz = TR._script("nusc_zs")
    idx, samples = TR.nusc_main()
    toks = [e["token"] for e in samples]
    src = nz.load_preds(idx)["cv"]
    cv = np.stack([src[t] for t in toks])
    scene_ids = np.unique([e["scene"] for e in samples], return_inverse=True)[1]

    def booter(sid):
        u, sid = np.unique(sid, return_inverse=True)
        n_sc = len(u)
        Bidx = np.random.default_rng(0).integers(0, n_sc, (B, n_sc))

        def boot(v):
            s, c = np.bincount(sid, v, n_sc), np.bincount(sid, None, n_sc)
            return s[Bidx].sum(1) / c[Bidx].sum(1)
        return boot

    t8 = 0.5 * np.arange(1, 9)
    jobs, meta = [("cv", "cv", cv, np.arange(len(samples)))], []
    for m in MODELS:
        z = np.load(run_dir("preds", f"nusc_{m}.npz"))
        rows_m = pd.Series(np.arange(len(toks)), index=toks)[z["keys"]].to_numpy()
        meta.append((m, list(z["names"]), rows_m))
        for a, traj in zip(z["names"], z["traj"]):
            pts = np.stack([Z.to_lidar_point(t8, tr[:, :2], tr[:, 2], idx["scenes"][samples[i]["scene"]]["lidar_xyz"],
                                             samples[i]["fut_t"])[0] for tr, i in zip(traj, rows_m)])
            jobs.append((m, str(a), pts, rows_m))
    from multiprocessing import Pool
    global _NUSC_SAMPLES
    _NUSC_SAMPLES = samples
    with Pool(min(16, len(jobs))) as p:        # per_sample's collision loop is ~50 ms / sample: one process per arm
        res = p.map(_nusc_metrics, [(pts, rows) for _, _, pts, rows in jobs])
    got = {(m, a): r for (m, a, _, _), r in zip(jobs, res)}
    mcv = got[("cv", "cv")]
    rows = []
    for m, names, rows_m in meta:
        M = {a: got[(m, a)] for a in names}
        cvm = {k: v[rows_m] for k, v in mcv.items()}
        sid = scene_ids[rows_m]
        boot = booter(sid)
        adv_b = cvm["l2"] - M["base"]["l2"]
        adv_b_ci = np.percentile(boot(adv_b), [2.5, 97.5])
        for a in names:
            r = {"model": NAME[m], "arm": a, "n": len(rows_m), "cv_l2": float(cvm["l2"].mean()), "l2": float(M[a]["l2"].mean()),
                 "col_vad": float(M[a]["col_vad"].mean()), "col_bevp": float(M[a]["col_bevp"].mean())}
            for k in ("l2", "col_vad", "col_bevp"):
                d = M[a][k] - M["base"][k]
                r[f"d_{k}_vs_base"] = float(d.mean())
                r[f"d_{k}_lo"], r[f"d_{k}_hi"] = np.percentile(boot(d), [2.5, 97.5])
            adv = cvm["l2"] - M[a]["l2"]
            r["adv_vs_cv"] = float(adv.mean())
            r["adv_lo"], r["adv_hi"] = np.percentile(boot(adv), [2.5, 97.5])
            r["adv_shrink"] = float(1 - adv.mean() / adv_b.mean())
            r["base_adv_ci_gt0"] = bool(adv_b_ci[0] > 0)
            rows.append(r)
    df = pd.DataFrame(rows)
    verdict = []
    for m in MODELS:
        g = df[df.model == NAME[m]].set_index("arm")
        if not g.loc["base", "base_adv_ci_gt0"]:
            v = "not applicable (no advantage over CV at base)"
        else:
            hit = [a for a in MAIN_ARMS if g.loc[a, "adv_shrink"] >= 0.5]
            v = ("nuScenes score mainly from the ego prior (" + ", ".join(hit) + ")") if hit else "not mainly ego prior"
        verdict.append({"model": NAME[m], "set": "nuScenes", "verdict": v})
    return df, pd.DataFrame(verdict)


# ---------------------------------------------------------------- NAVSIM

def nav_name(m: str, a: str) -> str:
    return f"nq3q5_{m}_{a.replace('+', 'p').replace('-', 'm').replace('.', '')}"


def nav_jobs(models=MODELS):
    lines = []
    for m in models:
        z = np.load(run_dir("preds", f"navtest_{m}.npz"))
        for a, traj in zip(z["names"], z["traj"]):
            p = run_dir("nav", f"{m}_{a}.npz")
            np.savez(p, tokens=z["keys"], poses=traj.astype(np.float32))
            tf = run_dir("nav", f"tokens_{m}.txt")
            tf.write_text("\n".join(z["keys"]) + "\n")
            lines.append(f"v1 navtest {nav_name(m, a)} {p} {tf if m == 'wajepa' else ''}".rstrip())
    run_dir("nav", f"jobs_{'_'.join(models)}.txt").write_text("\n".join(lines) + "\n")
    log.info("%d devkit jobs", len(lines))


def _scores(name: str) -> pd.Series | None:
    from .openloop_standing import _latest
    df = _latest("v1", "navtest", name)
    if df is None:
        return None
    df = df[df["token"].astype(str).str.fullmatch(r"[0-9a-f]{16,17}") & df["valid"].astype(bool)]
    return df.set_index("token")["score"].astype(float)


def nav_table():
    rows, verdict = [], []
    rng = np.random.default_rng(0)
    for m in MODELS:
        z = np.load(run_dir("preds", f"navtest_{m}.npz"))
        base = _scores(nav_name(m, "base"))
        for a in z["names"]:
            s = _scores(nav_name(m, a))
            x, y = s.align(base, join="inner")
            d = (x - y).to_numpy()
            bs = d[rng.integers(0, len(d), (B, len(d)))].mean(1)
            rows.append({"model": NAME[m], "arm": a, "n": len(d), "pdms": 100 * x.mean(), "base_pdms": 100 * y.mean(),
                         "delta": 100 * d.mean(), "lo": 100 * np.percentile(bs, 2.5), "hi": 100 * np.percentile(bs, 97.5)})
        g = pd.DataFrame(rows)
        g = g[g.model == NAME[m]].set_index("arm")
        hit = [a for a in MAIN_ARMS if -g.loc[a, "delta"] >= 5]
        verdict.append({"model": NAME[m], "set": "NAVSIM navtest",
                        "verdict": ("PDMS mainly from the ego prior (" + ", ".join(hit) + ")") if hit else "PDMS drop < 5 on every zero arm"})
    return pd.DataFrame(rows), pd.DataFrame(verdict)


# ---------------------------------------------------------------- selectivity

def select() -> pd.DataFrame:
    """Pooled reactive flip - non-reactive false flip from the stored exam tables ([D] 16:50 (5))."""
    R = Path(__file__).resolve().parents[1] / "research" / "results" / "top10-exams"
    D = data_dir()
    mc = sorted((D / "runs/reactivity/mc-carla_p5v1_ba").glob("*/flip_rates.csv"))
    mc = [p for p in mc if "20260925-233126" in str(p)][0]
    i3 = D / "runs/elicitation/i3-exam/20260926-012841/flip_rates.csv"
    rows = []

    def take(exam, fam, label, df, ex, flip="flip_rate", lo="flip_lo", ff="false_flip_nonreactive", null="false_flip_null_oos"):
        r = df[(df.examinee == ex) & (df.scope == "pooled")].iloc[0]
        rows.append({"exam": exam, "family": fam, "examinee": label, "flip": r[flip], "flip_lo": r[lo],
                     "false_flip_nonreactive": r[ff], "null_ff_oos": r[null]})
    t1p, t1i = pd.read_csv(R / "t1_p5_flip_rates.csv"), pd.read_csv(R / "t1_i3_flip_rates.csv")
    t2i = pd.read_csv(R / "t2_i3_flip_rates.csv")
    t2i = t2i[t2i.subset == "all"] if "subset" in t2i else t2i
    t2p = pd.read_csv(R / "t2_p5_scores.csv")
    t2p = t2p[(t2p.window == "per frame")].rename(columns={"flip": "flip_rate", "lo": "flip_lo"})
    t3p = pd.read_csv(R / "p5_t3_flip_rates.csv")
    mcp, i3e = pd.read_csv(mc), pd.read_csv(i3)
    for ex, fam in (("SparseDriveV2", "SD"), ("ZTRS", "HY")):
        take("P5 v1 BA", fam, ex, t1p, ex)
        take("I3", fam, ex, t1i, ex)
    for ex, fam in (("DrivoR", "DR"), ("WA-JEPA", "AF")):
        take("P5 v1 BA", fam, ex, t2p, ex, null="null_ff_oos")
        take("I3", fam, ex, t2i, ex)
    for ex, fam in (("BridgeDrive waypoint speed 2 s", "LEAD"), ("TFv6 waypoint speed 2 s", "LEAD"),
                    ("BLUE waypoint speed 2 s", "SL"), ("SimLingo waypoint speed 2 s", "SL")):
        take("P5 v1 BA", fam, ex, t3p, ex)
    for mdl in ("cinque", "lebowski"):
        take("P5 v1 BA", "openpilot", f"openpilot ridge_late ({mdl})", mcp, f"prior [{mdl}]")
        take("P5 v1 BA", "M-C", f"M-C pair ({mdl})", mcp, f"M-C pair [{mdl}]")
        take("I3", "openpilot", f"openpilot ridge_late ({mdl})", i3e, f"ridge_late op-{mdl} temporal")
        take("I3", "M-C", f"M-C pair ({mdl})", i3e, f"M-C pair [{mdl}]")
    df = pd.DataFrame(rows)
    df["selectivity_pp"] = 100 * (df.flip - df.false_flip_nonreactive)
    df["label"] = np.where(df.flip_lo <= df.null_ff_oos, "no reaction",
                           np.where(df.selectivity_pp < 10, "slows for any car", "selective"))
    return df


# ---------------------------------------------------------------- rig perturbation

def _frag_tokens(keys: np.ndarray) -> np.ndarray:
    return np.arange(0, len(keys), len(keys) // N_FRAG)[:N_FRAG]


def _maps(cam: dict, yaw_deg: float, dz: float):
    """(s, U, V) sampling the camera's own image for the camera turned by yaw_deg about ego z (about its own centre)
    or raised by dz: rotation exactly; height by ground-plane + infinity reprojection."""
    from . import navsim_rig as NR
    R, t = np.asarray(cam["R"], np.float64), np.asarray(cam["t"], np.float64)
    v = {"sensor2lidar_rotation": R, "sensor2lidar_translation": t, "intrinsics": np.asarray(cam["K"], np.float64),
         "distortion": np.asarray(cam["D"], np.float64)}
    src = NR.as_camgeom(v)
    src["width"], src["height"] = NR.W, NR.H
    a = np.radians(yaw_deg)
    Rz = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
    r = NR.rays({**v, "sensor2lidar_rotation": Rz @ R}).astype(np.float64)
    if dz:
        c = t + np.array([0.0, 0.0, dz])
        lam = np.where(r[..., 2] < -1e-6, (Z_GROUND - c[2]) / np.where(r[..., 2] < -1e-6, r[..., 2], -1), np.inf)
        P = c + lam[..., None] * r
        hit = np.isfinite(lam) & (np.linalg.norm(P[..., :2] - c[:2], axis=-1) <= 200)
        d = np.where(hit[..., None], P - t, r)
        r = d / np.linalg.norm(d, axis=-1, keepdims=True)
    u, w, ok = NR.project(r.astype(np.float32), src)
    return np.where(ok, 0, -1).astype(np.int8), u.astype(np.float32), w.astype(np.float32)


def _frag_worker(args):
    import cv2
    from PIL import Image
    from . import camgeom as G
    k, e, out = args
    paths = {}
    cache = {}
    for f in range(4):
        for cn in NAV_CAMS:
            cam = e["cams"][f][cn]
            im = np.asarray(Image.open(cam["path"]).convert("RGB"))
            for arm, pert in FRAG_ARMS.items():
                if pert is None:
                    continue
                key = (np.asarray(cam["R"]).tobytes(), np.asarray(cam["t"]).tobytes(), cn, arm)
                if key not in cache:
                    cache[key] = _maps(cam, *pert)
                s, U, V = cache[key]
                p = out / arm / f"{e['token']}_{f}_{cn}.jpg"
                if not p.exists():
                    rendered = G.render_np(s, U, V, [im])
                    cv2.imwrite(str(p), cv2.cvtColor(rendered, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 95])
                paths.setdefault(arm, []).append(str(p))
    return k, paths


def frag_prep(workers: int = 16):
    from multiprocessing import Pool
    from . import navsim_zs as Z
    idx = Z.load_index("navtest")
    z = np.load(run_dir("req", "navtest.npz"))
    sel = _frag_tokens(z["keys"])
    out = run_dir("frag", "img")
    for a in FRAG_ARMS:
        (out / a).mkdir(parents=True, exist_ok=True)
    jobs = [(k, idx[i], out) for k, i in enumerate(sel)]
    img = {a: [None] * len(sel) for a in FRAG_ARMS if FRAG_ARMS[a] is not None}
    with Pool(workers) as p:
        for k, paths in p.imap_unordered(_frag_worker, jobs):
            for a, ps in paths.items():
                img[a][k] = ps
    img["orig"] = [list(z["img"][i]) for i in sel]
    for a in FRAG_ARMS:
        np.savez(run_dir("frag", f"req_{a}.npz"), keys=z["keys"][sel], img=np.array(img[a]), hist=z["hist"][sel],
                 ego=z["ego"][sel], cmd=z["cmd"][sel])
    for m in MODELS:
        e8 = np.concatenate([z["ego"][sel], np.eye(4)[z["cmd"][sel]]], 1)[None].astype(np.float32)
        np.savez(run_dir("frag", f"arms_{m}.npz"), names=np.array(["base"]), ego8=e8, hist=z["hist"][sel][None])
    log.info("rig perturbation: %d tokens x %d arms", len(sel), len(FRAG_ARMS))


def frag_table() -> pd.DataFrame:
    rows = []
    for m in MODELS:
        P = {a: np.load(run_dir("frag", f"pred_{m}_{a}.npz"))["traj"][0] for a in FRAG_ARMS}
        for a, ref in [(a, "ident") for a in FRAG_ARMS if a not in ("ident", "orig")] + [("ident", "orig")]:
            d = np.linalg.norm(P[a][..., :2] - P[ref][..., :2], axis=-1)
            rows.append({"model": NAME[m], "arm": a, "vs": ref, "n": len(d),
                         "changed_rate": float((d.max(1) > 1e-3).mean()), "mean_disp_m": float(d.mean()),
                         "p95_disp_m": float(np.percentile(d.mean(1), 95)), "share_ade_gt_0.1m": float((d.mean(1) > 0.1).mean())})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- tables

def tables():
    RESULTS.mkdir(parents=True, exist_ok=True)
    out = {}
    nd, nv = exam_nusc()
    nd.round(4).to_csv(RESULTS / "ego_nusc.csv", index=False)
    vt, vv = nav_table()
    vt.round(3).to_csv(RESULTS / "ego_navtest.csv", index=False)
    v = pd.concat([nv, vv])
    v.to_csv(RESULTS / "ego_verdict.csv", index=False)
    s = select()
    s.round(4).to_csv(RESULTS / "selectivity.csv", index=False)
    f = frag_table()
    f.round(4).to_csv(RESULTS / "rig_perturbation.csv", index=False)
    for name, df in (("verdict", v), ("nusc", nd[["model", "arm", "l2", "d_l2_vs_base", "d_l2_lo", "d_l2_hi", "adv_vs_cv", "adv_shrink"]]),
                     ("navtest", vt), ("selectivity", s), ("rig", f)):
        out[name] = df
        log.info("%s\n%s", name, df.to_markdown(index=False, floatfmt=".3f"))
    for p in (run_dir("verify.json"), run_dir("req", "arms.json")):
        if p.exists():
            (RESULTS / p.name).write_text(p.read_text())


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=("req", "arms", "verify", "nav-jobs", "frag-prep", "tables", "select"))
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--model", default="drivor,wajepa")
    a = ap.parse_args()
    if a.step == "req":
        req()
    elif a.step == "arms":
        arms_all()
    elif a.step == "verify":
        verify()
    elif a.step == "nav-jobs":
        nav_jobs(tuple(a.model.split(",")))
    elif a.step == "frag-prep":
        frag_prep(a.workers)
    elif a.step == "select":
        print(select().to_markdown(index=False, floatfmt=".3f"))
    else:
        tables()


if __name__ == "__main__":
    main()
