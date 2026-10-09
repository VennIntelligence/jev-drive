"""op_parity nc-slow (plans/2026-10-09-nc-slow-prereg.md, lane NC1): is "where to slow down" decidable from what the driver has at inference time?

Labels = simulator scores of the same-path longitudinal scaling family (decision 196) on navtrain held-out fold-model plans; a selection head per
input arm is fitted on navtrain and read once on navtest (SH30-F-s0 / s1, scores of nc_tax's score_all.csv).

  extract --shard i | --seed s  (GPU, op-train)  one forward per row: plan, policy hidden state (select_4, mean), openpilot's native lead / lead_prob /
                                                 meta / action outputs. navtrain: the fold model that held the row's log out (G-leak); navtest: SH30 (G-plan)
  family                        (CPU)            held-out poses of every navtrain token in the v2_navtrain metric cache x {0.7 .. 1.1} -> poses_nt.npz
  gate                          (CPU)            G-id: a = 1.0 rows on the turn tokens == decision 190's held-out score rows
  build                         (CPU, torch)     feature streams (E, H, V, OP, G0, G1), labels, weights -> tab_train.npz / tab_test.npz / y_test.npz
  fit --arm A                   (GPU for M)      OOF on navtrain by cf5 fold, learner + threshold chosen on navtrain, navtest predictions (no navtest score read)
  xfit --arm A                  (CPU)            secondary: navtest-internal cross-fit by log (label-domain check; reads navtest scores)
  report                        (CPU)            gates, AUC, operating point, EPDMS translation, verdict -> $OUT/report/
Scoring is `python -m jevdrive.bench score-poses --traffic non_reactive --mcache v2_navtrain` (nc_slow_chain.sh). GT boxes enter only arms G0 / G1 (probe).
"""
import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

if hasattr(os, "sched_getaffinity"):                   # the pool pins the job (taskset): one OpenMP / BLAS thread per granted core, not the inherited count
    for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[_k] = str(len(os.sched_getaffinity(0)))

import numpy as np  # noqa: E402

REPO = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(REPO), str(REPO / "lib"), str(Path(__file__).parent)]
D = Path(os.environ.get("DATA_DIR", "/root/autodl-tmp/ujs"))
OUT = D / "runs/op_parity/nc_slow"
CR = D / "runs/op_parity/cache"
K, NSH, SEEDS = 5, 12, (0, 1)
SCALES = (0.7, 0.8, 0.9, 1.0, 1.1)                     # scored; MAIN family of decision 196
SLOW, ID = (0, 1, 2), 3                                # indices into SCALES: the selection set is SLOW + ID
SUB8 = ["NC", "DAC", "DDC", "TLC", "EP", "TTC", "LK", "HC"]
COLS = ["no_at_fault_collisions", "drivable_area_compliance", "driving_direction_compliance", "traffic_light_compliance", "ego_progress",
        "time_to_collision_within_bound", "lane_keeping", "history_comfort"]
ARMS = dict(E="e", H="e h", V="e v", N="e h v", OP="e op", OPH="e h v op", G0="e g0", G1="e g0 g1")
PRIV = ("G0", "G1")
ORACLE_N1, LINE_SHARE, LINE_EP = 1.13, 0.30, 0.3       # registered line: gain >= 0.3 x 1.13, CI lower bound > 0, EP loss < 0.3
T_IDXS = 10.0 * (np.arange(33) / 32.0) ** 2
FRONT, REAR, HALF_W = 4.049, -1.127, 1.1485
NPT, TOL_M = 15, 0.03
HGB = dict(max_iter=300, learning_rate=0.05, max_leaf_nodes=31, early_stopping=False, random_state=0)
M_CFG = dict(w=256, drop=0.1, wd=1e-2, lr=1e-3, epochs=30, batch=512, inits=5)


def akey(a):
    return f"a{round(100 * a):03d}"


def fold_of_log(log):
    return int(hashlib.sha256(f"cf{K}|{log}".encode()).hexdigest(), 16) % K


def save(path, **arrs):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.stem}.{os.getpid()}.npz")
    np.savez(tmp, **arrs)
    os.replace(tmp, path)


# ---------------------------------------------------------------- extract (GPU)
def run_rows(model, S, rows, batch=128):
    """One forward per store row -> plan mean (m, 33, 15) pieces, the policy's select_4 / mean (512 each, fp16) and the raw lead / lead_prob / meta / action slices."""
    import torch
    from jevdrive import op_adapt as A
    stash, orig = {}, model.net.run_batched

    def tap(feeds, want, **kw):
        o = orig(feeds, A.POLICY_OUT, **kw)
        stash.update(o)
        return {k: o[k] for k in want}
    model.net.run_batched = tap
    try:
        sl, m, dev = model.net.slices, len(rows), S.ego.device
        pi = np.arange(sl["plan"].start, sl["plan"].start + 495)
        R = dict(plan_pos=np.zeros((m, 33, 3), np.float32), plan_v=np.zeros((m, 33), np.float32), plan_a=np.zeros((m, 33), np.float32),
                 lead=np.zeros((m, 144), np.float32), lead_prob=np.zeros((m, 3), np.float32), meta=np.zeros((m, 55), np.float32),
                 action=np.zeros((m, 4), np.float32), select_4=np.zeros((m, 512), np.float16), mean=np.zeros((m, 512), np.float16))
        with torch.no_grad():
            for i in range(0, m, batch):
                r = torch.as_tensor(rows[i:i + batch], device=dev)
                o = model(S.front[r], S.ego[r], S.tc[r], S.side[r] if S.side is not None else None, None).float().cpu().numpy()
                j, p = slice(i, i + len(r)), o[:, pi].reshape(-1, 33, 15)
                R["plan_pos"][j], R["plan_v"][j], R["plan_a"][j] = p[:, :, 0:3], p[:, :, 3], p[:, :, 6]
                for k in ("lead", "lead_prob", "meta", "action"):
                    R[k][j] = o[:, sl[k]]
                R["select_4"][j] = stash["select_4"].reshape(len(r), -1).float().cpu().numpy()
                R["mean"][j] = stash["mean"].reshape(len(r), -1).float().cpu().numpy()
    finally:
        model.net.run_batched = orig
    return R


def cmd_extract(a):
    import torch
    from jevdrive.bench import navsim as N
    from jevdrive.bench.models import resolve
    from jevdrive.data import splits
    from jevdrive.run import Run
    import nt_labels as NL
    train = a.shard >= 0
    name = f"s{a.shard}" if train else f"navtest_s{a.seed}"
    out = OUT / "feat" / (a.tag + ("-smoke" if a.limit else "")) / f"{name}.npz"
    with Run("op_parity", f"nc_slow/extract-{name}", seed=0, config=vars(a)) as run:
        if out.exists():
            run.info("exists: %s", out)
            return
        N._pp_path()
        import pp_train as T
        dev = torch.device("cuda")
        data = NL.shard_data(a.shard) if train else "lb_navtest"
        run.use_split(splits.load("navsim/navtrain" if train else "navsim/navtest"))
        S = T.Store([data], dev, need_side=(N.cache_dir(data, "gimm") / "side.npy").exists(), frames="warp")
        names, logs, speed = S.tab["names"], S.tab["log"], S.tb["speed"]
        n = min(S.n, a.limit) if a.limit else S.n
        ld = lambda spec: N._load_ckpt(T, resolve(spec, check=True).ckpt, dev)  # noqa: E731
        if train:
            trs = [splits.load(f"navsim/op-parity-cf{K}f{j}-train") for j in range(K)]
            fold = np.array([fold_of_log(l) for l in logs[:n]])
            res, diff = None, np.zeros(n)
            for j in range(K):
                idx = np.flatnonzero(fold == j)
                if not len(idx):
                    continue
                run.use_split(trs[j])
                assert not trs[j].mask(logs[idx]).any(), f"fold {j}: a held-out log is in the training split"       # G-leak (a)
                model = ld(f"CF{K}f{j}-F-s0@warp")
                R = run_rows(model, S, idx)
                res = res or {k: np.zeros((n,) + v.shape[1:], v.dtype) for k, v in R.items()}
                for k, v in R.items():
                    res[k][idx] = v
                ref = np.load(NL.ol(a.shard, "plans", f"{NL.stem(f'CF{K}f{j}-F-s0')}.npz"))
                assert ref["names"].tolist() == names.tolist(), "stored plans rows != token cache rows"
                diff[idx] = np.abs(ref["plan_pos"][idx][:, :NPT] - R["plan_pos"][:, :NPT]).max((1, 2))
                del model
                torch.cuda.empty_cache()
            res["fold"] = fold
        else:
            res = run_rows(ld(f"SH30-F-s{a.seed}@warp"), S, np.arange(n))
            ref = np.load(D / "runs/op_parity/self_consist/infer" / f"SH30-F-s{a.seed}-warp__lb_navtest.npz")
            assert ref["names"][:n].tolist() == names[:n].tolist()
            diff = np.abs(ref["plan_pos"][:n, :NPT] - res["plan_pos"][:, :NPT]).max((1, 2))
        ok = speed[:n] >= 0.5
        gate = dict(name=name, n=int(n), rows_speed_ok=int(ok.sum()), max_diff_m=float(diff[ok].max()), frac_over_tol=float((diff[ok] > TOL_M).mean()))
        save(out, tokens=names[:n], log=logs[:n], diff=diff, gate=np.array(json.dumps(gate)), **res)
        run.summary.update(gate)
        run.info("gate %s", gate)
        assert gate["frac_over_tol"] < 1e-3, f"G-leak / G-plan: {gate}"


# ---------------------------------------------------------------- family / gate (CPU)
def cached_tokens():
    root = D / "runs/navsim/metric_cache/v2_navtrain"
    return {p.parent.name for p in root.glob("*/*/*/metric_cache.pkl")}


def cmd_family(a):
    import nt_labels as NL
    import turn_ceiling as TC
    from jevdrive.data import splits
    from jevdrive.run import Run
    with Run("op_parity", "nc_slow/family", config=vars(a)) as run:
        run.use_split(splits.load("navsim/navtrain"))
        dev = [splits.load(f"navsim/op-parity-cf{K}f{j}-dev") for j in range(K)]
        have, turn = cached_tokens(), set(NL.turn_tokens())
        n_rest_all = 0
        M = {k: [] for k in ("tokens", "log", "fold", "shard", "row", "turn", "P")}
        for i in range(NSH):
            tab = np.load(CR / NL.shard_data(i) / "tab.npz")
            nm, lg = tab["names"].astype(str), tab["log"].astype(str)
            fold = np.array([fold_of_log(l) for l in lg])
            for j in range(K):
                assert dev[j].mask(lg[fold == j]).all(), "hash fold != registered dev split"
            is_turn = np.isin(nm, list(turn))
            n_rest_all += int((~is_turn).sum())
            keep = np.flatnonzero(np.isin(nm, list(have)))
            P = np.zeros((len(nm), 8, 3), np.float32)
            for j in range(K):
                z = np.load(NL.N_pred(i, f"CF{K}f{j}-F-s0"))
                assert z["tokens"].tolist() == nm.tolist(), f"shard {i} fold {j}: export tokens != token cache rows"
                P[fold == j] = z["poses"][fold == j]
            for k, v in zip(M, (nm[keep], lg[keep], fold[keep], np.full(len(keep), i), keep, is_turn[keep], P[keep])):
                M[k].append(v)
        M = {k: np.concatenate(v) for k, v in M.items()}
        assert len(set(M["tokens"].tolist())) == len(M["tokens"]), "duplicate navtrain tokens"
        if a.limit:                                                        # smoke: the first `limit` turn tokens + as many rest tokens
            keep = np.sort(np.r_[np.flatnonzero(M["turn"])[:a.limit], np.flatnonzero(~M["turn"])[:a.limit]])
            M = {k: v[keep] for k, v in M.items()}
        P = M.pop("P")
        n_rest = int((~M["turn"]).sum())
        w = np.where(M["turn"], 1.0, n_rest_all / max(n_rest, 1))
        sfx = "_smoke" if a.limit else ""
        save(OUT / f"poses_nt{sfx}.npz", tokens=M["tokens"], **{akey(s): TC.transform(P, v=s) for s in SCALES})
        save(OUT / f"meta_nt{sfx}.npz", w=w, P=P, **M)
        (OUT / f"tokens_nt{sfx}.txt").write_text("\n".join(M["tokens"].tolist()) + "\n")
        run.summary.update(n=len(P), turn=int(M["turn"].sum()), rest=n_rest, rest_all=n_rest_all, w_rest=float(w.max()), turn_missing=len(turn - set(M["tokens"].tolist())) if not a.limit else -1)
        run.info(json.dumps(run.summary))


def read_scores(csv, tokens, keys):
    """score-poses CSV -> (score (n, len(keys)), subs (n, len(keys), 8)) in the given token / key order."""
    import pandas as pd
    df = pd.read_csv(csv)
    df = df[df.key.isin(keys)].set_index(["key", "token"])
    sc, sub = [], []
    for k in keys:
        g = df.loc[k].reindex(tokens)
        assert g.score.notna().all(), f"{csv}: key {k} misses tokens"
        sc.append(g.score.to_numpy(float))
        sub.append(g[COLS].to_numpy(float))
    return np.stack(sc, 1), np.stack(sub, 1)


def cmd_gate(a):
    import pandas as pd
    from jevdrive.run import Run
    with Run("op_parity", "nc_slow/gate", config=vars(a)) as run:
        sfx = "_smoke" if a.smoke else ""
        m = np.load(OUT / f"meta_nt{sfx}.npz")
        tok = m["tokens"][m["turn"]].astype(str)
        new = pd.read_csv(OUT / f"score_nt{sfx}.csv")
        new = new[new.key == akey(1.0)].set_index("token").loc[tok]
        ref = pd.read_csv(D / "runs/op_parity/sh30_crossfit/labels/score.csv")
        ref = ref[ref.key == "h"].set_index("token").loc[tok]
        d = {c: float(np.abs(new[c].to_numpy(float) - ref[c].to_numpy(float)).max()) for c in COLS + ["score"]}
        res = dict(n=len(tok), max_abs_diff=d, ok=bool(max(d.values()) < 1e-6))
        (OUT / f"gate_id{sfx}.json").write_text(json.dumps(res, indent=1))
        run.summary.update(res)
        run.info(json.dumps(res))
        if not res["ok"]:
            raise SystemExit("G-id failed")


# ---------------------------------------------------------------- feature streams
def e_feats(ego, P, v0):
    n = len(P)
    seg = np.linalg.norm(np.diff(np.concatenate([np.zeros((n, 1, 2)), P[:, :, :2]], 1), axis=1), axis=2)
    spd, arc = seg / 0.5, seg.sum(1)
    return np.concatenate([ego, P.reshape(n, -1), spd, arc[:, None], P[:, -1, 2:3], v0[:, None], (spd[:, 1:2] - spd[:, 0:1]) / 0.5,
                           (arc / np.maximum(4 * v0, 1.0))[:, None]], 1).astype(np.float32)


def op_feats(F, v0):
    """openpilot's native outputs as shipped through the parity path: raw lead (mean + log std), lead_prob, meta, action, the plan's velocity / acceleration
    channels, plus the plan-vs-own-lead clearances of decision 179 (lead hypothesis 0 at 0 / 2 / 4 s)."""
    ld = F["lead"][:, :72].reshape(-1, 3, 6, 4)
    px = np.stack([np.interp([0.0, 2.0, 4.0], T_IDXS, p) for p in F["plan_pos"][:, :, 0]])
    clr = ld[:, 0, :3, 0] - px
    return np.concatenate([F["lead"], F["lead_prob"], F["meta"], F["action"], F["plan_v"], F["plan_a"], clr, (ld[:, 0, 0, 0] / np.maximum(v0, 0.5))[:, None],
                           (v0 - ld[:, 0, 0, 2])[:, None]], 1).astype(np.float32)


def g0_feats(P, box, valid, cls, v0):
    """Ground-truth lead gap at t0 along the plan's path (privileged probe). box (n, 9, K, 5) x y yaw L W in the t0 rear-axle frame."""
    n, Ko = len(P), box.shape[2]
    last = P[:, -1]
    ext = last[:, :2] + 60.0 * np.stack([np.cos(last[:, 2]), np.sin(last[:, 2])], 1)
    Q = np.concatenate([np.tile([[[-30.0, 0.0]]], (n, 1, 1)), np.zeros((n, 1, 2)), P[:, :, :2], ext[:, None]], 1).astype(np.float64)   # (n, 11, 2)
    A, B = Q[:, :-1], Q[:, 1:]
    dv = B - A
    L = np.linalg.norm(dv, axis=2)                                             # (n, 10)
    cum = np.concatenate([np.full((n, 1), -30.0), -30.0 + np.cumsum(L, 1)[:, :-1]], 1)
    c = box[:, 0, :, :2].astype(np.float64)                                    # (n, K, 2)
    rel = c[:, :, None] - A[:, None]                                           # (n, K, 10, 2)
    Ls = np.maximum(L, 1e-6)[:, None]
    t = np.clip((rel * dv[:, None]).sum(-1) / Ls ** 2, 0, 1)
    dist = np.linalg.norm(rel - t[..., None] * dv[:, None], axis=-1)
    dist[np.broadcast_to(L[:, None] < 1e-3, dist.shape)] = 1e9
    j = dist.argmin(2)                                                         # (n, K)
    tk = lambda x: np.take_along_axis(x, j[..., None], 2)[..., 0]              # noqa: E731
    tx, ty = tk(np.broadcast_to((dv[..., 0] / np.maximum(L, 1e-6))[:, None], dist.shape)), tk(np.broadcast_to((dv[..., 1] / np.maximum(L, 1e-6))[:, None], dist.shape))
    s = tk(np.broadcast_to(cum[:, None], dist.shape)) + tk(t) * tk(np.broadcast_to(L[:, None], dist.shape))
    lat = tx * tk(rel[..., 1]) - ty * tk(rel[..., 0])
    dth = box[:, 0, :, 2] - np.arctan2(ty, tx)
    hl = 0.5 * (box[:, 0, :, 3] * np.abs(np.cos(dth)) + box[:, 0, :, 4] * np.abs(np.sin(dth)))
    hw = 0.5 * (box[:, 0, :, 3] * np.abs(np.sin(dth)) + box[:, 0, :, 4] * np.abs(np.cos(dth)))
    vel = (box[:, 1, :, :2] - box[:, 0, :, :2]) / 0.5
    vok = valid[:, 0] & valid[:, 1]
    vlon = np.where(vok, vel[..., 0] * tx + vel[..., 1] * ty, 0.0)
    vlat = np.where(vok, -vel[..., 0] * ty + vel[..., 1] * tx, 0.0)
    gap = s - hl - FRONT
    ahead = valid[:, 0] & (s > 1.5)
    arc = np.linalg.norm(np.diff(np.concatenate([np.zeros((n, 1, 2)), P[:, :, :2]], 1), axis=1), axis=2).sum(1)
    out = []
    for margin in (0.2, 1.2):                                                  # in the corridor / near it (cut-in candidates)
        g = np.where(ahead & (np.abs(lat) - hw < HALF_W + margin), gap, 1e9)
        o = np.argsort(g, 1)[:, :2]
        for r in range(2 if margin == 0.2 else 1):
            pick = lambda x: np.take_along_axis(x, o[:, r:r + 1], 1)[:, 0]     # noqa: E731
            gg = pick(g)
            has = gg < 1e8
            gg = np.where(has, np.clip(gg, -5, 60), 60.0)
            vl = np.where(has, pick(vlon), 0.0)
            close = np.where(has, v0 - vl, 0.0)
            out += [has, gg, vl, np.where(has, pick(vok), 0), close, gg / np.maximum(v0, 0.5), np.clip(gg / np.maximum(close, 0.1), -20, 20),
                    np.where(has, pick(lat), 0.0), np.where(has, pick(vlat), 0.0), np.where(has, pick(cls.astype(float)), -1.0), arc - gg]
        if margin == 0.2:
            out += [(g < d).sum(1) for d in (10.0, 20.0, 30.0)]
    eu = np.where(valid[:, 0], np.linalg.norm(c, axis=2), 1e9).min(1) if Ko else np.full(n, 60.0)
    out.append(np.minimum(eu, 60.0))
    return np.stack([np.asarray(x, np.float32) for x in out], 1)


def g1_feats(label_file, tokens, P):
    """Privileged: per selection scale, the scaled plan's minimum signed distance to the logged future boxes (lib/agent_hinge.py, 0.1 s steps)."""
    import torch
    import agent_hinge as AH
    import turn_ceiling as TC
    H = AH.AgentHinge(label_file, tokens, torch.device("cpu"), margin=0.0)
    out = []
    for i in list(SLOW) + [ID]:
        Q = torch.as_tensor(TC.transform(P, v=SCALES[i]), dtype=torch.float32)
        cols = []
        for b in range(0, len(Q), 4096):
            r = torch.arange(b, min(b + 4096, len(Q)))
            with torch.no_grad():
                dist, cnt, cor = H.distances(Q[r, :, 0], Q[r, :, 1], Q[r, :, 2], r)
            d1 = dist.masked_fill(~cnt, 60.0).clamp_max(60.0)
            d2 = dist.masked_fill(~(cnt & cor), 60.0).clamp_max(60.0)
            hit = (d1.amin(2) < 0).float()
            first = torch.where(hit.any(1), hit.argmax(1).float(), torch.full((len(r),), 41.0))
            cols.append(torch.stack([d1.amin((1, 2)), d2.amin((1, 2)), first], 1).numpy())
        out.append(np.concatenate(cols))
    return np.concatenate(out, 1).astype(np.float32)


def agent_rows(label_file, tokens):
    z = np.load(label_file)
    pos = {t: i for i, t in enumerate(z["tokens"].tolist())}
    idx = np.array([pos[t] for t in tokens.tolist()])
    return z["box"][idx], z["valid"][idx] & z["ok"][idx][:, None, None], z["cls"][idx]


def cmd_build(a):
    import nt_labels as NL
    from jevdrive.bench import compat
    from jevdrive.data import splits
    from jevdrive.run import Run
    with Run("op_parity", "nc_slow/build", config=vars(a)) as run:
        nt, ntr = splits.load("navsim/navtest"), splits.load("navsim/navtrain")
        run.use_split(nt), run.use_split(ntr)
        fd = OUT / "feat" / a.tag
        # ---- train side
        m = dict(np.load(OUT / "meta_nt.npz"))
        tok = m["tokens"].astype(str)
        assert ntr.mask(tok).all()
        sc, sub = read_scores(OUT / "score_nt.csv", tok, [akey(s) for s in SCALES])
        F, ego, v0 = {}, np.zeros((len(tok), 20), np.float32), np.zeros(len(tok), np.float32)
        V3, gates = np.zeros((len(tok), 32, 512), np.float16), []
        for i in range(NSH):
            sel = np.flatnonzero(m["shard"] == i)
            z, tab, rows = np.load(fd / f"s{i}.npz"), np.load(CR / NL.shard_data(i) / "tab.npz"), m["row"][sel]
            assert (z["tokens"][rows] == tok[sel]).all() and (z["fold"][rows] == m["fold"][sel]).all()
            gates.append(json.loads(str(z["gate"])))
            for k in ("plan_pos", "plan_v", "plan_a", "lead", "lead_prob", "meta", "action", "select_4", "mean"):
                F.setdefault(k, np.zeros((len(tok),) + z[k].shape[1:], z[k].dtype))[sel] = z[k][rows]
            ego[sel], v0[sel] = tab["ego"][rows], tab["speed"][rows]
            front = np.load(CR / f"{NL.shard_data(i)}@warp/front.npy", mmap_mode="r")
            o = np.argsort(rows)
            V3[sel[o]] = np.asarray(front[rows[o], -1])
        box, val, cls = agent_rows(D / "runs/op_parity/agent_labels/navtrain_all.npz", tok)
        P = m["P"]
        save(OUT / "tab_train.npz", tokens=tok, log=m["log"], fold=m["fold"], turn=m["turn"], w=m["w"], score=sc, sub=sub,
             e=e_feats(ego, P, v0), h=np.concatenate([F["select_4"], F["mean"]], 1), op=op_feats(F, v0), g0=g0_feats(P, box, val, cls, v0),
             g1=g1_feats(D / "runs/op_parity/agent_labels/navtrain_all.npz", tok, P), v0=v0)
        np.save(OUT / "v3_train.npy", V3)
        # ---- test side (features only here; the scores go to y_test.npz, which only xfit / report read)
        tab = np.load(CR / "lb_navtest/tab.npz")
        tok = tab["names"].astype(str)
        assert nt.mask(tok).all() and not set(tab["log"].tolist()) & set(m["log"].tolist()), "navtest / navtrain share a log"
        ego, v0 = tab["ego"], tab["speed"]
        box, val, cls = agent_rows(D / "runs/op_parity/agent_labels/navtest.npz", tok)
        T = {k: [] for k in ("e", "h", "op", "g0", "g1", "P")}
        ysc, ysub = [], []
        for s in SEEDS:
            z = np.load(fd / f"navtest_s{s}.npz")
            assert (z["tokens"] == tok).all()
            gates.append(json.loads(str(z["gate"])))
            pz = np.load(compat.pred_file(f"SH30-F-s{s}", "navtest"))
            pos = {t: i for i, t in enumerate(pz["tokens"].astype(str).tolist())}
            P = pz["poses"][[pos[t] for t in tok.tolist()]].astype(np.float32)
            for k, v in zip(T, (e_feats(ego, P, v0), np.concatenate([z["select_4"], z["mean"]], 1), op_feats(z, v0), g0_feats(P, box, val, cls, v0),
                                g1_feats(D / "runs/op_parity/agent_labels/navtest.npz", tok, P), P)):
                T[k].append(v)
            a_, b_ = read_scores(D / "runs/op_parity/nc_tax/score_all.csv", tok, [f"s{s}_{akey(x)}" for x in SCALES])
            ysc.append(a_), ysub.append(b_)
        save(OUT / "tab_test.npz", tokens=tok, log=tab["log"], v0=v0, **{k: np.stack(v) for k, v in T.items()})
        np.save(OUT / "v3_test.npy", np.ascontiguousarray(np.load(CR / "lb_navtest@warp/front.npy", mmap_mode="r")[:, -1]))
        save(OUT / "y_test.npz", tokens=tok, log=tab["log"], score=np.stack(ysc), sub=np.stack(ysub))
        tr = np.load(OUT / "tab_train.npz")
        g = tr["score"][:, list(SLOW)] - tr["score"][:, [ID]]
        w = tr["w"]
        res = dict(n_train=len(w), turn=int(tr["turn"].sum()), w_rest=float(w.max()), y_slow_rate_w=float((w * (g.max(1) > 1e-9)).sum() / w.sum()),
                   y_slow_n=int((g.max(1) > 1e-9).sum()), nc_fail_rate_w=float((w * (tr["sub"][:, ID, 0] < 1)).sum() / w.sum()),
                   nc_fail_n=int((tr["sub"][:, ID, 0] < 1).sum()), extract_gates=gates, ok=all(q["frac_over_tol"] < 1e-3 for q in gates))
        (OUT / "build.json").write_text(json.dumps(res, indent=1))
        run.summary.update({k: v for k, v in res.items() if k != "extract_gates"})
        run.info(json.dumps(run.summary))


# ---------------------------------------------------------------- learners and the selection rule
def choose_tau(m, real, w):
    """Threshold on the score m that maximises the weighted realised gain sum(w * real * [m > tau]) / sum(w); candidates = 200 quantiles of m and +inf."""
    best = (np.inf, 0.0)
    for t in np.unique(np.quantile(m, np.linspace(0, 1, 201))):
        g = float((w * real * (m > t)).sum() / w.sum())
        if g > best[1] + 1e-15:
            best = (float(t), g)
    return best


def policy(kind, pred, tau, afix=0):
    """-> (moved (n,) bool, scale index into SCALES (n,)). kind T / M: pred (n, 3) predicted gains; C: pred (n,) probability with a fixed scale."""
    if kind == "C":
        moved = pred > tau
        return moved, np.where(moved, SLOW[afix], ID)
    moved = pred.max(1) > tau
    return moved, np.where(moved, np.asarray(SLOW)[pred.argmax(1)], ID)


def tune(kind, pred, G, w):
    """(tau, afix, weighted OOF gain x 100) of a learner's OOF predictions; G (n, 3) true gains of the slower scales."""
    if kind == "C":
        c = [choose_tau(pred, G[:, j], w) + (j,) for j in range(3)]
        t, g, j = max(c, key=lambda x: x[1])
        return t, j, 100 * g
    t, g = choose_tau(pred.max(1), G[np.arange(len(G)), pred.argmax(1)], w)
    return t, 0, 100 * g


class Tab:
    """Design matrix of the tree learners: raw low-dimensional streams, whitened PCA-32 of the hidden state and of the mean vision token (fitted on train)."""

    def __init__(self, tr, V3, streams):
        from sklearn.decomposition import PCA
        self.streams, self.p = streams, {}
        if "h" in streams:
            self.p["h"] = PCA(32, whiten=True, random_state=0).fit(tr["h"].astype(np.float32))
        if "v" in streams:
            self.p["v"] = PCA(32, whiten=True, random_state=0).fit(V3.astype(np.float32).mean(1))

    def __call__(self, side, V3):
        c = [side[s].reshape(-1, side[s].shape[-1]) for s in self.streams if s in ("e", "op", "g0", "g1")]
        if "h" in self.p:
            c.append(self.p["h"].transform(side["h"].reshape(-1, 1024).astype(np.float32)))
        if "v" in self.p:
            v = self.p["v"].transform(V3.astype(np.float32).mean(1))
            c.append(np.tile(v, (len(c[0]) // len(v), 1)))
        return np.concatenate(c, 1).astype(np.float32)


def fit_T(X, G, w):
    from sklearn.ensemble import HistGradientBoostingRegressor as R
    ms = [R(**HGB).fit(X, G[:, j], sample_weight=w) for j in range(3)]
    return lambda Z: np.stack([m.predict(Z) for m in ms], 1)


def fit_C(X, G, w):
    from sklearn.ensemble import HistGradientBoostingClassifier as C
    m = C(**HGB).fit(X, (G.max(1) > 1e-9).astype(int), sample_weight=w)
    return lambda Z: m.predict_proba(Z)[:, 1]


def make_head(dims, tok):
    import torch
    import torch.nn as nn
    w, drop = M_CFG["w"], M_CFG["drop"]

    class Head(nn.Module):
        def __init__(s):
            super().__init__()
            s.enc = nn.ModuleDict({k: nn.Sequential(nn.Linear(d, w), nn.GELU()) for k, d in dims.items()})
            if tok:
                s.ln, s.proj, s.q = nn.LayerNorm(512), nn.Linear(512, 64), nn.Parameter(torch.randn(4, 64) * 0.1)
            s.out = nn.Sequential(nn.Linear(w * len(dims) + (256 if tok else 0), 2 * w), nn.GELU(), nn.Dropout(drop), nn.Linear(2 * w, 3))

        def forward(s, Z, tk):
            z = [s.enc[k](Z[k]) for k in s.enc]
            if tok:
                x = s.proj(s.ln(tk.float()))
                att = torch.softmax(torch.einsum("qd,bnd->bqn", s.q, x) / 8.0, -1)
                z.append(torch.einsum("bqn,bnd->bqd", att, x).flatten(1))
            return s.out(torch.cat(z, -1))
    return Head()


class MData:
    """Standardised dense streams + vision tokens of one side on the GPU."""

    def __init__(self, side, V3, streams, dev, mu=None, rows=None):
        import torch
        ks = [k for k in ("e", "h", "op") if k in streams]
        raw = {k: np.nan_to_num(side[k].reshape(-1, side[k].shape[-1]).astype(np.float32)) for k in ks}
        self.mu = mu or {k: (v[rows].mean(0), v[rows].std(0) + 1e-6) for k, v in raw.items()}
        self.Z = {k: torch.as_tensor(np.clip((v - self.mu[k][0]) / self.mu[k][1], -10, 10), device=dev) for k, v in raw.items()}
        self.V = V3 if "v" in streams else None
        self.n, self.nv = len(next(iter(raw.values()))), (len(V3) if V3 is not None else 0)

    def batch(self, ti):
        return {k: v[ti] for k, v in self.Z.items()}, (self.V[ti % self.nv] if self.V is not None else None)


def fit_M(Dtr, G, w, logs, tr, seed, dev):
    """One init of the decision-191 head on rows tr of Dtr; early stop on 20 % held-out logs by the weighted realised gain at tau = 0. -> predict(MData, n)."""
    import torch
    lg = np.random.default_rng(900 + seed).permutation(np.unique(logs[tr]))
    va_m = np.isin(logs[tr], lg[: max(1, len(lg) // 5)])
    va, trn = tr[va_m], tr[~va_m]
    ysd = float(G[trn].std()) + 1e-9
    Yt, Wt = torch.as_tensor(G / ysd, dtype=torch.float32, device=dev), torch.as_tensor(w, dtype=torch.float32, device=dev)
    torch.manual_seed(seed)
    net = make_head({k: v.shape[1] for k, v in Dtr.Z.items()}, Dtr.V is not None).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=M_CFG["lr"], weight_decay=M_CFG["wd"])

    def pred(Dm, idx):
        net.eval()
        with torch.no_grad():
            ti = torch.as_tensor(idx, device=dev)
            return torch.cat([net(*Dm.batch(ti[b:b + 2048])) for b in range(0, len(ti), 2048)]).cpu().numpy() * ysd
    best, best_g = None, -np.inf
    for ep in range(M_CFG["epochs"]):
        net.train()
        perm = np.random.default_rng(seed * 1000 + ep).permutation(trn)
        for b in range(0, len(perm), M_CFG["batch"]):
            ti = torch.as_tensor(perm[b:b + M_CFG["batch"]], device=dev)
            loss = (Wt[ti] * ((net(*Dtr.batch(ti)) - Yt[ti]) ** 2).mean(1)).sum() / Wt[ti].sum()
            opt.zero_grad(), loss.backward(), opt.step()
        pv = pred(Dtr, va)
        gv = float((w[va] * G[va][np.arange(len(va)), pv.argmax(1)] * (pv.max(1) > 0)).sum() / w[va].sum())
        if gv > best_g + 1e-12:
            best_g, best = gv, {k: v.detach().clone() for k, v in net.state_dict().items()}
    net.load_state_dict(best)
    return pred


def cmd_fit(a):
    from jevdrive.data import splits
    from jevdrive.run import Run
    streams = ARMS[a.arm].split()
    with Run("op_parity", f"nc_slow/fit-{a.arm}", seed=0, config=vars(a) | dict(hgb=HGB, m=M_CFG)) as run:
        if (OUT / "fit" / f"{a.arm}.npz").exists():
            run.info("exists: %s", a.arm)
            return
        run.use_split(splits.load("navsim/navtrain")), run.use_split(splits.load("navsim/navtest"))
        tr, te = dict(np.load(OUT / "tab_train.npz")), dict(np.load(OUT / "tab_test.npz"))      # tab_test holds no score
        hv = "h" in streams or "v" in streams
        V3tr = np.load(OUT / "v3_train.npy") if "v" in streams else None
        V3te = np.load(OUT / "v3_test.npy") if "v" in streams else None
        G = (tr["score"][:, list(SLOW)] - tr["score"][:, [ID]]).astype(np.float64)
        w, fold, logs, n = tr["w"].astype(np.float64), tr["fold"], tr["log"], len(tr["w"])
        nte = te["e"].shape[1]
        res, t0 = dict(arm=a.arm, streams=streams), time.time()
        subsets = {"": np.arange(n)}
        ul = np.random.default_rng(100).permutation(np.unique(logs))
        for fr in (0.25, 0.5):
            subsets[f"_c{round(100 * fr)}"] = np.flatnonzero(np.isin(logs, ul[: int(round(fr * len(ul)))]))
        tab = Tab(tr, V3tr, streams)
        Xtr, Xte = tab(tr, V3tr), tab(te, V3te)
        for kind, fitter in (("T", fit_T), ("C", fit_C)):
            oof = np.zeros((n, 3)) if kind == "T" else np.zeros(n)
            for k in range(K):
                oof[fold == k] = fitter(Xtr[fold != k], G[fold != k], w[fold != k])(Xtr[fold == k])
            tau, afix, g = tune(kind, oof, G, w)
            res[kind] = dict(oof=oof, tau=tau, afix=afix, oof_gain=g)
            for sfx, rows in subsets.items():
                res[kind][f"test{sfx}"] = fitter(Xtr[rows], G[rows], w[rows])(Xte).reshape((len(SEEDS), nte) + oof.shape[1:])
            run.info("%s %s: OOF gain %+.4f (tau %.4g, afix %d), %.0f s", a.arm, kind, g, tau, afix, time.time() - t0)
        if hv:
            import torch
            dev = torch.device("cuda")
            Vg = torch.from_numpy(V3tr).to(dev) if V3tr is not None else None
            Vt = torch.from_numpy(V3te).to(dev) if V3te is not None else None
            ens = lambda Dm, rows, idx, Dp: np.mean([fit_M(Dm, G, w, logs, rows, s, dev)(Dp, idx) for s in range(M_CFG["inits"])], 0)  # noqa: E731
            oof = np.zeros((n, 3))
            for k in range(K):
                Dk = MData(tr, Vg, streams, dev, rows=np.flatnonzero(fold != k))
                oof[fold == k] = ens(Dk, np.flatnonzero(fold != k), np.flatnonzero(fold == k), Dk)
                run.info("%s M: fold %d done, %.0f s", a.arm, k, time.time() - t0)
            tau, afix, g = tune("M", oof, G, w)
            res["M"] = dict(oof=oof, tau=tau, afix=afix, oof_gain=g)
            for sfx, rows in subsets.items():
                Dm = MData(tr, Vg, streams, dev, rows=rows)
                res["M"][f"test{sfx}"] = ens(Dm, rows, np.arange(len(SEEDS) * nte), MData(te, Vt, streams, dev, mu=Dm.mu)).reshape(len(SEEDS), nte, 3)
            run.info("%s M: OOF gain %+.4f (tau %.4g), %.0f s", a.arm, g, tau, time.time() - t0)
        kinds = [k for k in ("T", "C", "M") if k in res]
        res["chosen"] = max(kinds, key=lambda k: res[k]["oof_gain"])
        flat = {f"{k}_{q}": v for k in kinds for q, v in res[k].items()}
        save(OUT / "fit" / f"{a.arm}.npz", chosen=np.array(res["chosen"]), kinds=np.array(kinds), **flat)
        run.summary.update(arm=a.arm, chosen=res["chosen"], **{f"oof_gain_{k}": res[k]["oof_gain"] for k in kinds}, fit_s=time.time() - t0)
        run.info(json.dumps(run.summary))


def cmd_xfit(a):
    """Secondary read-out (i): navtest-internal cross-fit by log with the tree learners; nested OOF on the training folds picks the threshold."""
    from jevdrive.data import splits
    from jevdrive.run import Run
    streams = ARMS[a.arm].split()
    with Run("op_parity", f"nc_slow/xfit-{a.arm}", seed=0, config=vars(a)) as run:
        run.use_split(splits.load("navsim/navtest"))
        te, y = dict(np.load(OUT / "tab_test.npz")), np.load(OUT / "y_test.npz")
        V3 = np.load(OUT / "v3_test.npy") if "v" in streams else None
        nte = te["e"].shape[1]
        X = Tab({"h": te["h"].reshape(-1, 1024)}, V3, streams)(te, V3)
        G = (y["score"][:, :, list(SLOW)] - y["score"][:, :, [ID]]).reshape(-1, 3)
        logs = np.tile(te["log"], len(SEEDS))
        fo = np.array([int(hashlib.sha256(f"ncx|{l}".encode()).hexdigest(), 16) % K for l in logs])
        w = np.ones(len(G))
        out = {}
        for kind, fitter in (("T", fit_T), ("C", fit_C)):
            moved, sidx = np.zeros(len(G), bool), np.full(len(G), ID)
            for k in range(K):
                trn = np.flatnonzero(fo != k)
                oof = np.zeros((len(trn), 3)) if kind == "T" else np.zeros(len(trn))
                for q in range(K):
                    if q != k:
                        i_, o_ = fo[trn] != q, fo[trn] == q
                        oof[o_] = fitter(X[trn][i_], G[trn][i_], w[trn][i_])(X[trn][o_])
                tau, afix, _ = tune(kind, oof, G[trn], w[trn])
                moved[fo == k], sidx[fo == k] = policy(kind, fitter(X[trn], G[trn], w[trn])(X[fo == k]), tau, afix)
            out[f"{kind}_sidx"] = sidx.reshape(len(SEEDS), nte)
            run.info("%s xfit %s: moved %.3f", a.arm, kind, moved.mean())
        save(OUT / "xfit" / f"{a.arm}.npz", **out)


# ---------------------------------------------------------------- report
def auc(y, s):
    from scipy.stats import rankdata
    y = np.asarray(y, bool)
    n1, n0 = y.sum(), (~y).sum()
    return float((rankdata(s)[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)) if n1 and n0 else np.nan


def auc_ci(y, s, groups, w=None, B=1000):
    """AUC with a log-cluster percentile bootstrap. w: optional integer-like row weights are ignored (navtest is unweighted)."""
    import pandas as pd
    codes, uniq = pd.factorize(groups)
    by = [np.flatnonzero(codes == i) for i in range(len(uniq))]
    rng = np.random.default_rng(0)
    bs = []
    for _ in range(B):
        idx = np.concatenate([by[i] for i in rng.integers(len(uniq), size=len(uniq))])
        bs.append(auc(y[idx], s[idx]))
    lo, hi = np.nanquantile(bs, [0.025, 0.975])
    return auc(y, s), float(lo), float(hi)


def fmt(r, d=2, sign=True):
    f = f"{{:{'+' if sign else ''}.{d}f}}"
    return f"{f.format(r['mean'])} [{f.format(r['lo'])}, {f.format(r['hi'])}]"


def md(df, digits=3):
    def f(v):
        if isinstance(v, (bool, np.bool_)):
            return "yes" if v else ""
        if isinstance(v, (float, np.floating)):
            return "" if np.isnan(v) else f"{v:.{digits}f}"
        return str(v)
    cols = [str(c) for c in df.columns]
    return "\n".join(["| " + " | ".join(cols) + " |", "|" + "|".join([":--"] + ["--:"] * (len(cols) - 1)) + "|"]
                     + ["| " + " | ".join(f(v) for v in r) + " |" for r in df.itertuples(index=False)])


def n1_oracle(sc, sub, family):
    """Decision 196's board-level upper bound for one seed: a* only on the failing tokens with a clean recovery inside `family` (indices into SCALES)."""
    nc, ttc, base = sub[:, :, 0], sub[:, :, 5], sc[:, ID]
    fail_nc = nc[:, ID] < 1
    fail = fail_nc | (ttc[:, ID] < 1)
    crit = np.where(fail_nc[:, None], nc == 1, (ttc == 1) & (nc == 1))
    clean = crit & (sc > base[:, None] + 1e-9)
    clean[:, [i for i in range(len(SCALES)) if i not in family]] = False
    best = np.where(clean, sc, -1).max(1)
    d = np.where(fail & clean.any(1), best - base, 0.0)
    return d, fail & clean.any(1)


def cmd_report(a):
    import pandas as pd
    from jevdrive import stats
    from jevdrive.data import splits
    from jevdrive.run import Run
    with Run("op_parity", "nc_slow/report", config=vars(a)) as run:
        run.use_split(splits.load("navsim/navtest")), run.use_split(splits.load("navsim/navtrain"))
        RES = OUT / "report"
        RES.mkdir(parents=True, exist_ok=True)
        tr, y = np.load(OUT / "tab_train.npz"), np.load(OUT / "y_test.npz")
        tok, logs, SC, SUB = y["tokens"].astype(str), y["log"], y["score"], y["sub"]          # SC (2, n, 5), SUB (2, n, 5, 8)
        n = len(tok)
        Gte = SC[:, :, list(SLOW)] - SC[:, :, [ID]]
        yslow = Gte.max(2) > 1e-9
        L, summ = [], dict(build=json.loads((OUT / "build.json").read_text()), gate_id=json.loads((OUT / "gate_id.json").read_text()))
        # ---- oracles (G-oracle: the main family must reproduce decision 196's +1.13)
        orc = {}
        for nm, fam in (("N1 main 0.7-1.1", (0, 1, 2, 4)), ("slower only 0.7-0.9", SLOW)):
            d = [n1_oracle(SC[s], SUB[s], fam) for s in SEEDS]
            orc[nm] = dict(per_seed=[100 * float(x[0].mean()) for x in d], tokens=float(np.mean([x[1].sum() for x in d])))
            orc[nm]["mean"] = float(np.mean(orc[nm]["per_seed"]))
            if fam == SLOW:
                yfail = np.stack([x[1] for x in d])
                dslow = np.stack([x[0] for x in d])
        orc["any token, slower only (O_any)"] = dict(per_seed=[100 * float(np.maximum(Gte[s].max(1), 0).mean()) for s in SEEDS], tokens=float(yslow.sum(1).mean()))
        orc["any token, slower only (O_any)"]["mean"] = float(np.mean(orc["any token, slower only (O_any)"]["per_seed"]))
        summ["oracles"] = orc
        summ["g_oracle_ok"] = bool(abs(orc["N1 main 0.7-1.1"]["mean"] - ORACLE_N1) < 0.01)
        L.append("## Oracles on navtest (no-EC EPDMS points; G-oracle: the first row must be decision 196's +1.13)\n")
        L.append(pd.DataFrame([dict(oracle=k, gain=v["mean"], s0=v["per_seed"][0], s1=v["per_seed"][1], tokens=v["tokens"]) for k, v in orc.items()]).pipe(md, 3))
        if not summ["g_oracle_ok"]:
            raise SystemExit(f"G-oracle failed: {orc}")
        # ---- navtrain labels
        Gtr = tr["score"][:, list(SLOW)] - tr["score"][:, [ID]]
        wtr = tr["w"]
        ytr_slow = Gtr.max(1) > 1e-9
        dtr, ytr_fail = n1_oracle(tr["score"], tr["sub"], SLOW)
        wm = lambda x, m=None: float((wtr * x)[m].sum() / wtr[m].sum()) if m is not None else float((wtr * x).sum() / wtr.sum())  # noqa: E731
        lab = [dict(set=nm, n=int(m.sum()), **{"NC fail %": 100 * wm(tr["sub"][:, ID, 0] < 1, m), "NC or TTC fail %": 100 * wm((tr["sub"][:, ID, 0] < 1) | (tr["sub"][:, ID, 5] < 1), m),
                    "y_fail %": 100 * wm(ytr_fail, m), "y_slow %": 100 * wm(ytr_slow, m), "oracle slower-only (points)": 100 * wm(dtr, m),
                    "O_any (points)": 100 * wm(np.maximum(Gtr.max(1), 0), m)}) for nm, m in (("navtrain, weighted", np.ones(len(wtr), bool)), ("turn tokens", tr["turn"]), ("rest tokens", ~tr["turn"]))]
        lab.append(dict(set="navtest (seed mean)", n=n, **{"NC fail %": 100 * float((SUB[:, :, ID, 0] < 1).mean()), "NC or TTC fail %": 100 * float(((SUB[:, :, ID, 0] < 1) | (SUB[:, :, ID, 5] < 1)).mean()),
                                                          "y_fail %": 100 * float(yfail.mean()), "y_slow %": 100 * float(yslow.mean()), "oracle slower-only (points)": orc["slower only 0.7-0.9"]["mean"],
                                                          "O_any (points)": orc["any token, slower only (O_any)"]["mean"]}))
        summ["labels"] = lab
        L.append("\n## Labels (held-out fold-model plans on navtrain, SH30 on navtest)\n\n" + pd.DataFrame(lab).pipe(md, 3))
        # ---- per arm
        ft = pd.read_csv(OUT.parent / "nc_tax/report/navtest_fail_tokens.csv")
        pos = {t: i for i, t in enumerate(tok.tolist())}
        rows, extra, curve, picks = [], [], [], {}
        rng = np.random.default_rng(0)

        def read(sidx, tag):
            """Realised effect of a pick array (2, n) of indices into SCALES."""
            tk = lambda X: np.take_along_axis(X, sidx[:, :, None], 2)[:, :, 0]                  # noqa: E731
            d = tk(SC) - SC[:, :, ID]
            r = dict(arm=tag, moved=100 * float((sidx != ID).mean()))
            r["gain"] = stats.bootstrap(100 * d.mean(0), groups=logs)
            subd = {k: np.take_along_axis(SUB[..., j], sidx[:, :, None], 2)[:, :, 0] - SUB[:, :, ID, j] for j, k in enumerate(SUB8)}
            for k in ("NC", "TTC", "DAC", "EP"):
                r[k] = stats.bootstrap(100 * subd[k].mean(0), groups=logs)
            nc1, tt1 = SUB[:, :, ID, 0] + subd["NC"], SUB[:, :, ID, 5] + subd["TTC"]
            f0, f1 = (SUB[:, :, ID, 0] < 1) | (SUB[:, :, ID, 5] < 1), (nc1 < 1) | (tt1 < 1)
            r["fail"] = stats.bootstrap(100 * (f1.astype(float) - f0).mean(0), groups=logs)
            r.update(nc_fail_base=float((SUB[:, :, ID, 0] < 1).sum(1).mean()), nc_fail=float((nc1 < 1).sum(1).mean()), nc_new=float(((nc1 < 1) & (SUB[:, :, ID, 0] == 1)).sum(1).mean()),
                     nc_fixed=float(((nc1 == 1) & (SUB[:, :, ID, 0] < 1)).sum(1).mean()), fail_base=float(f0.sum(1).mean()), fail_after=float(f1.sum(1).mean()))
            mv = sidx != ID
            r["precision"] = 100 * float((d[mv] > 1e-9).mean()) if mv.any() else np.nan
            r["recall_fail"] = 100 * float((mv & yfail & (d > 1e-9)).sum() / max(yfail.sum(), 1))
            r["d"] = d
            return r

        for arm in ARMS:
            f = OUT / "fit" / f"{arm}.npz"
            if not f.exists():
                continue
            z = np.load(f)
            ch = str(z["chosen"])
            for kind in z["kinds"].tolist():
                pred, tau, afix = z[f"{kind}_test"], float(z[f"{kind}_tau"]), int(z[f"{kind}_afix"])
                moved, sidx = policy(kind, pred.reshape((-1,) + pred.shape[2:]), tau, afix)
                sidx = sidx.reshape(len(SEEDS), n)
                r = read(sidx, arm)
                sco = pred if kind == "C" else pred.max(2)
                oof = z[f"{kind}_oof"]
                so = oof if kind == "C" else oof.max(1)
                r.update(learner=kind, chosen=kind == ch, tau=tau, afix=SCALES[SLOW[afix]] if kind == "C" else np.nan, oof_gain=float(z[f"{kind}_oof_gain"]),
                         oof_auc_slow=auc(ytr_slow, so), oof_auc_fail=auc(ytr_fail, so), privileged=arm in PRIV)
                extra.append({k: v for k, v in r.items() if k != "d"})
                if kind != ch:
                    continue
                gl = np.tile(logs, len(SEEDS))
                r["auc_slow"], r["auc_fail"] = auc_ci(yslow.ravel(), sco.ravel(), gl), auc_ci(yfail.ravel(), sco.ravel(), gl)
                perm = [100 * float(np.mean([SC[s, np.arange(n), sidx[s][rng.permutation(n)]] - SC[s, :, ID] for s in SEEDS])) for _ in range(200)]
                r["perm_mean"], r["perm_hi"] = float(np.mean(perm)), float(np.quantile(perm, 0.975))
                r["gain_bonf"] = stats.bootstrap(100 * r["d"].mean(0), groups=logs, alpha=0.05 / 6)
                picks[arm] = sidx
                rows.append(r)
                for sfx, frac in (("_c25", 25), ("_c50", 50), ("", 100)):
                    pc = z[f"{kind}_test{sfx}"]
                    _, sx = policy(kind, pc.reshape((-1,) + pc.shape[2:]), tau, afix)
                    q = read(sx.reshape(len(SEEDS), n), arm)
                    curve.append(dict(arm=arm, learner=kind, logs_pct=frac, gain=fmt(q["gain"]), moved=q["moved"]))
        # stage-1 table
        tabl = []
        for r in rows:
            g = r["gain"]
            share = 100 * g["mean"] / ORACLE_N1
            passed = (not r["privileged"]) and g["mean"] >= LINE_SHARE * ORACLE_N1 and g["lo"] > 0 and -r["EP"]["mean"] < LINE_EP
            r["passed"], r["share"] = bool(passed), share
            tabl.append({"arm": r["arm"] + (" (privileged)" if r["privileged"] else ""), "learner": r["learner"], "AUC y_slow [95% CI]": "%.3f [%.3f, %.3f]" % r["auc_slow"],
                         "AUC y_fail [95% CI]": "%.3f [%.3f, %.3f]" % r["auc_fail"], "moved %": r["moved"], "precision %": r["precision"], "recall of y_fail %": r["recall_fail"],
                         "no-EC EPDMS gain [95% CI]": fmt(g, 3), "Bonferroni (m = 6) lower": r["gain_bonf"]["lo"], "share of +1.13 %": share, "NC": fmt(r["NC"]), "TTC": fmt(r["TTC"]),
                         "DAC": fmt(r["DAC"]), "EP": fmt(r["EP"]), "NC or TTC failing tokens, pp": fmt(r["fail"]), "NC failures (base -> after; fixed / new)":
                         f"{r['nc_fail_base']:.1f} -> {r['nc_fail']:.1f} ({r['nc_fixed']:.1f} / {r['nc_new']:.1f})", "permuted picks (mean; 97.5%)": f"{r['perm_mean']:+.3f}; {r['perm_hi']:+.3f}",
                         "navtrain OOF gain": r["oof_gain"], "OOF AUC y_slow / y_fail": f"{r['oof_auc_slow']:.3f} / {r['oof_auc_fail']:.3f}", "line": "pass" if passed else ("-" if r["privileged"] else "no")})
        T1 = pd.DataFrame(tabl)
        T1.to_csv(RES / "stage1.csv", index=False)
        L.append("\n## Stage 1: each input arm on navtest (12 146 tokens x SH30 s0 / s1; the learner chosen on navtrain OOF; points of the board mean)\n\n" + T1.pipe(md, 3))
        E = pd.DataFrame([{k: (fmt(v, 3) if isinstance(v, dict) else v) for k, v in r.items()} for r in extra])
        E.to_csv(RES / "all_learners.csv", index=False)
        L.append("\n## Every learner of every arm (navtest read for all; only the `chosen` one counts)\n\n" + E[["arm", "learner", "chosen", "oof_gain", "oof_auc_slow", "oof_auc_fail", "tau", "afix", "moved", "gain", "EP", "NC", "TTC", "precision", "recall_fail"]].pipe(md, 3))
        C = pd.DataFrame(curve)
        C.to_csv(RES / "curve.csv", index=False)
        L.append("\n## Learning curve (share of navtrain logs in the final fit; threshold from the full OOF)\n\n" + C.pipe(md, 2))
        # recall by N1 class / speed band for the chosen learners
        ft = ft[ft.token.isin(pos)]
        rc = []
        for arm, sidx in picks.items():
            d = np.take_along_axis(SC, sidx[:, :, None], 2)[:, :, 0] - SC[:, :, ID]
            hit = np.array([(sidx[s, pos[t]] != ID) and d[s, pos[t]] > 1e-9 for s, t in zip(ft.seed, ft.token)])
            for col in ("cls", "speed"):
                for k, g in ft.assign(hit=hit)[ft.set == "NC"].groupby(col):
                    rc.append(dict(arm=arm, by=col, cell=k, n=len(g) / len(SEEDS), recall_pct=100 * g.hit.mean()))
        RC = pd.DataFrame(rc)
        RC.to_csv(RES / "recall_by_class.csv", index=False)
        if len(RC):
            L.append("\n## Recall of decision 196's NC failures by class and t0 speed band (moved and scoring higher; %)\n\n"
                     + RC.pivot_table(index=["by", "cell", "n"], columns="arm", values="recall_pct").reset_index().pipe(md, 0))
        # secondary (i): navtest-internal cross-fit
        xr = []
        for arm in ARMS:
            f = OUT / "xfit" / f"{arm}.npz"
            if f.exists():
                z = np.load(f)
                for kind in ("T", "C"):
                    q = read(z[f"{kind}_sidx"], arm)
                    xr.append(dict(arm=arm, learner=kind, moved=q["moved"], gain=fmt(q["gain"], 3), EP=fmt(q["EP"]), NC=fmt(q["NC"]), TTC=fmt(q["TTC"]), precision=q["precision"], recall_fail=q["recall_fail"]))
        if xr:
            X = pd.DataFrame(xr)
            X.to_csv(RES / "xfit.csv", index=False)
            L.append("\n## Secondary: navtest-internal cross-fit by log (labels from navtest itself, 5 folds, nested threshold)\n\n" + X.pipe(md, 2))
        # verdict
        nonp = [r for r in rows if not r["privileged"]]
        best = max(nonp, key=lambda r: r["gain"]["mean"]) if nonp else None
        ver = dict(line=dict(gain=LINE_SHARE * ORACLE_N1, ep_loss=LINE_EP), passed=[r["arm"] for r in nonp if r["passed"]],
                   best_nonpriv=None if best is None else dict(arm=best["arm"], learner=best["learner"], gain=best["gain"], share=best["share"], EP=best["EP"], auc_fail=best["auc_fail"]),
                   privileged={r["arm"]: dict(gain=r["gain"], share=r["share"], EP=r["EP"], auc_fail=r["auc_fail"]) for r in rows if r["privileged"]})
        summ["verdict"] = ver
        summ["rows"] = [{k: v for k, v in r.items() if k != "d"} for r in rows]
        L.append(f"\n## Verdict\n\nRegistered line: gain >= {LINE_SHARE * ORACLE_N1:+.3f}, 95% CI lower bound > 0, EP loss < {LINE_EP}. Non-privileged arms passing: {ver['passed'] or 'none'}.")
        (RES / "tables.md").write_text("\n".join(L) + "\n")
        (RES / "summary.json").write_text(json.dumps(summ, indent=1, default=float))
        run.summary.update(passed=ver["passed"], best=None if best is None else best["arm"], best_gain=None if best is None else best["gain"]["mean"])
        run.info("\n".join(L))


def cmd_posthoc(a):
    """Not registered, descriptive (written after the stage-1 read): (1) where the gain of the registered operating point comes from (tokens whose DAC
    changes vs the rest); (2) the operating point with the EP loss capped at the registered 0.3: threshold picked on navtrain OOF under that cap
    (honest), and the best navtest threshold under the cap (optimistic bound for the arm)."""
    import pandas as pd
    from jevdrive import stats
    from jevdrive.run import Run
    with Run("op_parity", "nc_slow/posthoc", config=vars(a)) as run:
        tr, y = np.load(OUT / "tab_train.npz"), np.load(OUT / "y_test.npz")
        logs, SC, SUB, n = y["log"], y["score"], y["sub"], len(y["tokens"])
        Gtr, EPtr, wtr = tr["score"][:, list(SLOW)] - tr["score"][:, [ID]], tr["sub"][:, list(SLOW), 4] - tr["sub"][:, [ID], 4], tr["w"]
        Gte, EPte = SC[:, :, list(SLOW)] - SC[:, :, [ID]], SUB[:, :, list(SLOW), 4] - SUB[:, :, [ID], 4]
        rows = []
        for arm in ARMS:
            z = np.load(OUT / "fit" / f"{arm}.npz")
            kind = str(z["chosen"])
            pred, oof, tau, afix = z[f"{kind}_test"], z[f"{kind}_oof"], float(z[f"{kind}_tau"]), int(z[f"{kind}_afix"])
            _, sidx = policy(kind, pred.reshape((-1,) + pred.shape[2:]), tau, afix)
            sidx = sidx.reshape(len(SEEDS), n)
            tk = lambda X: np.take_along_axis(X, sidx[:, :, None], 2)[:, :, 0]                  # noqa: E731
            d = tk(SC) - SC[:, :, ID]
            dac = tk(SUB[..., 1]) != SUB[:, :, ID, 1]
            col = tk(SUB[..., 0]) != SUB[:, :, ID, 0]
            ttc = (tk(SUB[..., 5]) != SUB[:, :, ID, 5]) & ~col
            r = {"arm": arm, "learner": kind, "gain": fmt(stats.bootstrap(100 * d.mean(0), groups=logs), 3),
                 "from tokens whose DAC changes": fmt(stats.bootstrap(100 * (d * dac).mean(0), groups=logs), 3),
                 "from tokens whose NC changes (DAC same)": fmt(stats.bootstrap(100 * (d * (col & ~dac)).mean(0), groups=logs), 3),
                 "from tokens whose TTC changes (NC, DAC same)": fmt(stats.bootstrap(100 * (d * (ttc & ~dac)).mean(0), groups=logs), 3),
                 "rest (EP cost of moved tokens)": fmt(stats.bootstrap(100 * (d * ~(dac | col | ttc)).mean(0), groups=logs), 3)}
            # EP-capped operating points (T / M: argmax scale above a threshold; C: fixed scale)
            if kind == "C":
                it = np.zeros(pred.shape, int)
                m_tr, g_tr, e_tr, m_te, g_te, e_te = oof, Gtr[:, afix], EPtr[:, afix], pred, Gte[:, :, afix], EPte[:, :, afix]
            else:
                ia, it = oof.argmax(1), pred.argmax(2)
                m_tr, g_tr, e_tr = oof.max(1), Gtr[np.arange(len(oof)), ia], EPtr[np.arange(len(oof)), ia]
                m_te, g_te, e_te = pred.max(2), np.take_along_axis(Gte, it[..., None], 2)[..., 0], np.take_along_axis(EPte, it[..., None], 2)[..., 0]
            best = (np.inf, 0.0)
            for t in np.unique(np.quantile(m_tr, np.linspace(0, 1, 2001))):
                s_ = m_tr > t
                if -100 * (wtr * e_tr * s_).sum() / wtr.sum() <= LINE_EP:
                    g = 100 * (wtr * g_tr * s_).sum() / wtr.sum()
                    if g > best[1]:
                        best = (float(t), float(g))
            mv = m_te > best[0]
            r.update({"EP-capped tau (navtrain OOF)": best[0], "OOF gain at the cap": best[1], "navtest moved %": 100 * float(mv.mean()),
                      "navtest gain at the cap": fmt(stats.bootstrap(100 * (g_te * mv).mean(0), groups=logs), 3),
                      "navtest EP at the cap": fmt(stats.bootstrap(100 * (e_te * mv).mean(0), groups=logs), 2),
                      "share of +1.13 %": 100 * 100 * float((g_te * mv).mean()) / ORACLE_N1})
            opt = (0.0, 0.0)
            for t in np.unique(np.quantile(m_te, np.linspace(0, 1, 2001))):
                s_ = m_te > t
                if -100 * (e_te * s_).mean() <= LINE_EP and 100 * (g_te * s_).mean() > opt[0]:
                    opt = (100 * float((g_te * s_).mean()), 100 * float(s_.mean()))
            r.update({"navtest-tuned bound under the cap (gain; moved %)": f"{opt[0]:+.3f}; {opt[1]:.1f}"})
            # the capped point's effect on the collision gates
            sx = np.where(mv, (np.full_like(it, afix) if kind == "C" else it), -1)
            pick = np.where(sx >= 0, np.asarray(SLOW)[np.maximum(sx, 0)], ID)
            nc1 = np.take_along_axis(SUB[..., 0], pick[:, :, None], 2)[:, :, 0]
            tt1 = np.take_along_axis(SUB[..., 5], pick[:, :, None], 2)[:, :, 0]
            da1 = np.take_along_axis(SUB[..., 1], pick[:, :, None], 2)[:, :, 0]
            nc0, tt0 = SUB[:, :, ID, 0], SUB[:, :, ID, 5]
            f0, f1 = (nc0 < 1) | (tt0 < 1), (nc1 < 1) | (tt1 < 1)
            r.update({"capped: NC failures fixed / new": f"{((nc1 == 1) & (nc0 < 1)).sum(1).mean():.1f} / {((nc1 < 1) & (nc0 == 1)).sum(1).mean():.1f}",
                      "capped: NC or TTC failing tokens, pp": fmt(stats.bootstrap(100 * (f1.astype(float) - f0).mean(0), groups=logs)),
                      "capped: gain from tokens whose DAC changes": fmt(stats.bootstrap(100 * (g_te * mv * (da1 != SUB[:, :, ID, 1])).mean(0), groups=logs), 3)})
            rows.append(r)
        T = pd.DataFrame(rows)
        (OUT / "report").mkdir(exist_ok=True)
        T.to_csv(OUT / "report/posthoc.csv", index=False)
        c1 = list(T.columns[:7])
        txt = ("## Post hoc (not registered): where the registered operating point's gain comes from\n\n" + md(T[c1]) +
               "\n\n## Post hoc (not registered): operating point with the EP loss capped at 0.3\n\n" + md(T[["arm", "learner"] + list(T.columns[7:])]) + "\n")
        (OUT / "report/posthoc.md").write_text(txt)
        run.info(txt)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("extract")
    p.add_argument("--shard", type=int, default=-1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--tag", default="full")
    p = sp.add_parser("family")
    p.add_argument("--limit", type=int, default=0)
    p = sp.add_parser("gate")
    p.add_argument("--smoke", action="store_true")
    p = sp.add_parser("build")
    p.add_argument("--tag", default="full")
    for c in ("fit", "xfit"):
        p = sp.add_parser(c)
        p.add_argument("--arm", required=True, choices=list(ARMS))
    sp.add_parser("report")
    sp.add_parser("posthoc")
    a = ap.parse_args()
    {"extract": cmd_extract, "family": cmd_family, "gate": cmd_gate, "build": cmd_build, "fit": cmd_fit, "xfit": cmd_xfit, "report": cmd_report, "posthoc": cmd_posthoc}[a.cmd](a)
