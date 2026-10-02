"""op-adapt H trainer and port readouts (op-train venv). Plan: experiments/op_adapt_h/plans/2026-10-04-op-adapt-H-prereg.md.

  teacher   the original model on every unperturbed sample of nav / wod / carla -> $H/teacher/<dom>.npz (out = distilled heads, mu = plan)
  train     one arm (ARMS) -> $H/runs/<arm>-s<seed>/ (jevdrive.run.Run: log.txt, events.jsonl, tb/, STATUS, DONE / ERROR) with
            ckpt-final.pt in op_adapt_l's checkpoint format (so op_adapt_l's readouts load it) + dev.json
  dev       dev metrics of a checkpoint (or O) on the dev splits of the three pools -> <run>/dev.json or $H/dev/O.json
  probe     readout (a): the decision-92 history probe in the port on its own samples (pnav / pwod / pcarla), variants normal /
            rotL / rotR (10 deg/s) / repeat / single -> $H/probe/<model>.npz
  link      symlink a run into op_adapt_L/runs/H-<tag>/ so op_adapt_l_readout.py eval / read / navtest take it as a model name
  navhard   readout (c): navhard two-stage EPDMS of the port plans (r2's nav_plans + op_interp export + official v2 scorer)

  CUDA_VISIBLE_DEVICES=0 taskset -c 0-49 $DATA_DIR/envs/op-train/bin/python experiments/op_adapt_h/scripts/h_train.py train --arm pilot
"""
import sys as _sys, pathlib as _pl  # noqa: E401
_sys.path[:0] = [str(_pl.Path(__file__).resolve().parents[3] / _d) for _d in ("", "scripts", "experiments/op_adapt_r2/lib")]
import argparse, json, os, subprocess, sys, time  # noqa: E401,E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402
from dataclasses import replace  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

from jevdrive import op_adapt as A  # noqa: E402
from jevdrive import stats  # noqa: E402
from jevdrive.common import data_dir  # noqa: E402
from jevdrive.run import Run  # noqa: E402
from experiments.op_adapt_l.lib import op_adapt_l as L  # noqa: E402
from experiments.op_adapt_h.lib import op_adapt_h as H  # noqa: E402

ARMS = {
    "smoke": dict(steps=60, ckpt_every=10 ** 9, dom_w={"nav": 0.0, "wod": 0.5, "carla": 0.5}),
    "pilot": dict(),                                         # H + O pairs, U and D rows
    "pilot_ctl": dict(control=True),                         # U-only control: same rows unperturbed
}


def cfg_of(arm, seed=0, steps=0) -> H.HCfg:
    c = H.HCfg(name=arm, seed=seed, **ARMS[arm])
    return replace(c, steps=steps) if steps else c


def run_dir(tag):
    return H.hroot("runs", tag)


def ckpt_path(model):
    return None if model == "O" else run_dir(model) / "ckpt-final.pt"


# ---------------------------------------------------------------- teacher
def cmd_teacher(a):
    dev = torch.device("cuda")
    m = L.load_model(None, dev)
    didx = A.distill_index(m.net.slices)
    pi = A.plan_index(m.net.slices)
    for d in a.domains:
        S = H.Samples(d)
        p = H.hroot("teacher") / f"{d}.npz"
        if p.exists():
            print(d, "exists")
            continue
        out = np.zeros((S.n, len(didx)), np.float16)
        mu = np.zeros((S.n, 33, 15), np.float32)
        t0 = time.time()
        for i in range(0, S.n, a.batch):
            r = np.arange(i, min(i + a.batch, S.n))
            imgs = torch.from_numpy(np.asarray(S.imgs[r[0]:r[-1] + 1])).to(dev)
            with torch.no_grad():
                o = H.forward(m, imgs, torch.from_numpy(S.t["slot_valid"][r]).to(dev), torch.from_numpy(S.t["tc"][r]).to(dev))["outputs"].float()
            out[r] = o[:, didx].cpu().numpy()
            mu[r] = o[:, pi].reshape(-1, 33, 15).cpu().numpy()
        np.savez(p, out=out, mu=mu)
        print(f"teacher {d}: {S.n} in {time.time() - t0:.0f} s", flush=True)


# ---------------------------------------------------------------- variants (CPU workers) and the port forward on them
_S = {}


def _variant(job):
    """(dom, row, kind, args) -> (imgs, slot_valid) of one probe / dev variant."""
    dom, i, kind, arg = job
    S = _S.get(dom) or _S.setdefault(dom, H.Samples(dom))
    t = S.t
    imgs, sv = np.asarray(S.imgs[i]), t["slot_valid"][i].copy()
    if kind == "normal":
        return imgs, sv
    if kind == "rot":
        return H.hist_rot(imgs, t["img_t"][i], t["img_valid"][i], t["cam"][i], np.radians(arg)), sv
    if kind == "repeat":
        return H.hist_repeat(imgs, t["img_valid"][i]), sv
    if kind == "single":
        return H.hist_single(imgs, sv)
    if kind == "offset":
        dy, dpsi = arg
        return H.offset_imgs(imgs, t["img_t"][i], t["img_valid"][i], t["cam"][i], dy, np.radians(dpsi), float(t["v0"][i])), sv
    raise KeyError(kind)


def plans_of(models: dict, jobs: list, dev, ex, bs=48) -> dict:
    """Port plans (n, 33, 15) of every model on every job (variant images built once in the pool)."""
    out = {k: np.zeros((len(jobs), 33, 15), np.float32) for k in models}
    base = next(iter(models.values()))
    pi = A.plan_index(base.net.slices)
    tcs = {d: H.Samples(d).t["tc"] for d in {j[0] for j in jobs}}
    it = ex.map(_variant, jobs, chunksize=4)
    for i0 in range(0, len(jobs), bs):
        chunk = [next(it) for _ in range(min(bs, len(jobs) - i0))]
        imgs = torch.from_numpy(np.stack([c[0] for c in chunk])).to(dev)
        sv = torch.from_numpy(np.stack([c[1] for c in chunk])).to(dev)
        tc = torch.from_numpy(np.stack([tcs[j[0]][j[1]] for j in jobs[i0:i0 + len(chunk)]])).to(dev)
        tr = H.trunks(base.net, imgs)
        with torch.no_grad():
            for k, m in models.items():
                o = m(tr, sv, tc)["outputs"].float()
                out[k][i0:i0 + len(chunk)] = o[:, pi].reshape(-1, 33, 15).cpu().numpy()
    return out


def load_models(names, dev):
    return {n: L.load_model(ckpt_path(n), dev) for n in names}


# ---------------------------------------------------------------- dev metrics
DEV_OFF = [(0.0, 5.0), (0.0, -5.0), (0.5, 0.0), (-0.5, 0.0)]


def dev_metrics(models: dict, dev, ex, cap=400) -> dict:
    """Per model and domain on the dev splits: ADE / drift of the unperturbed plan, G at +-10 deg/s by speed bin, the repeat /
    single plan-length shifts, the offset recovery ratio (antisymmetric lateral response at 1 / 2 s over the target's)."""
    res = {k: {} for k in models}
    for d in H.DOMS:
        S = H.Samples(d)
        rows = S.rows("dev")
        rows = rows[np.random.default_rng(0).permutation(len(rows))[:cap]]
        kinds = [("normal", None), ("rot", 10.0), ("rot", -10.0), ("repeat", None), ("single", None)] + [("offset", o) for o in DEV_OFF]
        jobs = [(d, int(i), k, a) for k, a in kinds for i in rows]
        P = plans_of(models, jobs, dev, ex)
        n = len(rows)
        t = S.t
        cam = t["cam"][rows, 0]
        fut = t["fut20"][rows]
        b = t["bin"][rows]
        for k in models:
            pk = {j: P[k][j * n:(j + 1) * n] for j in range(len(kinds))}
            r = {}
            rear = H.plan_rear(pk[0], cam)
            r["ade4"] = float(np.linalg.norm(rear[:, :16] - fut[:, :16], axis=-1).mean())
            if S.tea is not None:
                r["drift_median"] = float(np.median(A.plan_drift(pk[0], S.tea["mu"][rows])))
            G = (H.psi3(pk[1]) - H.psi3(pk[2])) / 2
            for bn in ("stop", "low", "mid", "high"):
                m = b == bn
                if m.any():
                    r[f"G_{bn}"] = float(G[m].mean())
            x3 = lambda p: H.plan_rear(p, cam)[:, 11, 0]  # noqa: E731
            mov = t["v0"][rows] >= 0.5
            r["repeat_dx3_moving"] = float((x3(pk[3]) - x3(pk[0]))[mov].mean())
            r["single_dx3_stop"] = float((x3(pk[4]) - x3(pk[0]))[~mov].mean()) if (~mov).any() else float("nan")
            fast = t["v0"][rows] >= 1.0
            for name, (jp, jm, o) in {"psi": (5, 6, DEV_OFF[0]), "y": (7, 8, DEV_OFF[2])}.items():
                yp, ym = H.plan_rear(pk[jp], cam), H.plan_rear(pk[jm], cam)
                tp = np.stack([H.recover_target(f, o[0], np.radians(o[1]), v) for f, v in zip(fut, t["v0"][rows])])
                tm = np.stack([H.recover_target(f, -o[0], -np.radians(o[1]), v) for f, v in zip(fut, t["v0"][rows])])
                for ti, ts in ((3, "1s"), (7, "2s")):
                    num = (yp[:, ti, 1] - ym[:, ti, 1])[fast].mean()
                    den = (tp[:, ti, 1] - tm[:, ti, 1])[fast].mean()
                    r[f"recover_{name}_{ts}"] = float(num / den)
            res[k][d] = r
    return res


def cmd_dev(a):
    dev = torch.device("cuda")
    models = load_models(a.models, dev)
    with ProcessPoolExecutor(a.workers) as ex:
        r = dev_metrics(models, dev, ex)
    for k, v in r.items():
        p = (run_dir(k) / "dev.json") if k != "O" else (H.hroot("dev") / "O.json")
        p.write_text(json.dumps(v, indent=1))
    print(json.dumps(r, indent=1))


# ---------------------------------------------------------------- probe (readout a)
PROBE = ("pnav", "pwod", "pcarla")
PVARS = [("normal", None), ("rot", 10.0), ("rot", -10.0), ("repeat", None), ("single", None)]


def cmd_probe(a):
    dev = torch.device("cuda")
    models = load_models(a.models, dev)
    with ProcessPoolExecutor(a.workers) as ex:
        for d in PROBE:
            S = H.Samples(d)
            todo = {k: m for k, m in models.items() if not (H.hroot("probe", k) / f"{d}.npz").exists()}
            if not todo:
                continue
            jobs = [(d, i, k, v) for k, v in PVARS for i in range(S.n)]
            t0 = time.time()
            P = plans_of(todo, jobs, dev, ex)
            for k in todo:
                np.savez(H.hroot("probe", k) / f"{d}.npz", plan=P[k].reshape(len(PVARS), S.n, 33, 15), ids=S.t["id"],
                         variants=np.array([f"{v}|{x}" for v, x in PVARS]))
            print(f"probe {d}: {S.n} x {len(PVARS)} x {len(todo)} models in {time.time() - t0:.0f} s", flush=True)


def probe_table(model, ref="O") -> pd.DataFrame:
    """G (deg) and the repeat / single shifts per domain x speed bin for `model` and `ref`, and the paired difference with a
    cluster bootstrap (jevdrive.stats.paired, groups = log / sequence / route)."""
    rows = []
    for d in PROBE:
        S = H.Samples(d)
        za, zo = np.load(H.hroot("probe", model) / f"{d}.npz"), np.load(H.hroot("probe", ref) / f"{d}.npz")
        cam = S.t["cam"][:, 0]
        def m(z):
            p = z["plan"]
            G = (H.psi3(p[1]) - H.psi3(p[2])) / 2
            x3 = lambda q: H.plan_rear(q, cam)[:, 11, 0]  # noqa: E731
            y3 = lambda q: H.plan_rear(q, cam)[:, 11, 1]  # noqa: E731
            lat = lambda q: np.abs(y3(q) - S.t["fut20"][:, 11, 1])  # noqa: E731
            return {"G_deg": G, "repeat_dlat3": lat(p[3]) - lat(p[0]), "repeat_dx3": x3(p[3]) - x3(p[0]),
                    "single_dx3": x3(p[4]) - x3(p[0]), "lat3_normal": lat(p[0])}
        ma, mo = m(za), m(zo)
        for bn in ("stop", "low", "mid", "high", "moving"):
            sel = (S.t["bin"] != "stop") if bn == "moving" else (S.t["bin"] == bn)
            for met in ma:
                if met in ("repeat_dlat3", "repeat_dx3") and bn == "stop":
                    continue
                if not sel.any():
                    continue
                r = stats.paired(ma[met][sel], mo[met][sel], groups=S.t["cluster"][sel])
                rows.append(dict(domain=d, bin=bn, metric=met, n=int(sel.sum()), model=float(np.nanmean(ma[met][sel])),
                                 ref=float(np.nanmean(mo[met][sel])), delta=r["mean"], lo=r["lo"], hi=r["hi"]))
    return pd.DataFrame(rows)


def cmd_probe_table(a):
    df = probe_table(a.model, a.ref)
    out = H.hroot("probe", a.model) / f"table_vs_{a.ref}.csv"
    df.to_csv(out, index=False)
    with pd.option_context("display.width", 200, "display.max_rows", 500):
        print(df.round(3).to_string(index=False))


# ---------------------------------------------------------------- link into op_adapt_l, navhard
def cmd_link(a):
    for tag in a.models:
        dst = L.lroot("runs") / f"H-{tag}"
        dst.mkdir(parents=True, exist_ok=True)
        lk = dst / "ckpt-final.pt"
        if lk.is_symlink() or lk.exists():
            lk.unlink()
        lk.symlink_to(ckpt_path(tag))
        print(lk, "->", ckpt_path(tag))


def cmd_navhard(a):
    import op_adapt_r2_readout as RO
    import op_lb as OL
    dev = torch.device("cuda")
    models = {"port": L.load_model(None, dev)} | load_models(a.models, dev)
    split = "lb_navhard"
    names, out = RO.nav_plans(models, split, dev, a.batch)
    pdir = OL.root(split, "plans")
    stems = []
    for k in models:
        stem = "gimm@cinque_r2port" if k == "port" else f"gimm@cinque_H{k}"
        if k == "port" and (pdir / f"{stem}.npz").exists():
            stems.append(stem)
            continue
        mu = out[k]
        np.savez(pdir / f"{stem}.npz", names=np.array(names), plan_pos=mu[:, :, 0:3], plan_vel=mu[:, :, 3:6], plan_yaw=mu[:, :, 11],
                 plan_mu=mu, plan_std=out[f"{k}_std"], steps=31, info=json.dumps({"model": f"port {k}", "source": "op_adapt_h"}))
        stems.append(stem)
    env = dict(os.environ, OPI_ROOT="op_lb")
    jev = str(data_dir() / "envs" / "jevdrive" / "bin" / "python")
    repo = Path(__file__).resolve().parents[3]
    subprocess.check_call([jev, str(repo / "experiments/op_openloop/lib/op_interp.py"), "nav-export", "--data", split, "--adapters", "base",
                           "--plans", *stems], env=env)
    subprocess.check_call([str(repo / "experiments/op_openloop/archive/op_interp_score.sh"), a.cpus, split, "v2", "navhard_two_stage"], env=env)
    res = RO.read_nav_scores(split, "v2", "navhard_two_stage", [s.replace("@", "-") + "__base" for s in stems])
    o = res.get("gimm-cinque_r2port__base")
    for m in a.models:
        x = res.get(f"gimm-cinque_H{m}__base")
        (H.hroot("readout", m)).mkdir(parents=True, exist_ok=True)
        (H.hroot("readout", m) / "navhard.json").write_text(json.dumps({"port_O": o, "model": x}, indent=1, default=str))
    print(json.dumps({k: v.get("score") for k, v in res.items()}, indent=1))


# ---------------------------------------------------------------- train
def cmd_train(a):
    cfg = cfg_of(a.arm, a.seed, a.steps)
    tag = f"{a.arm}-s{a.seed}"
    d = run_dir(tag)
    from jevdrive.data import splits
    with Run("op_adapt_H", tag, resume=d, seed=cfg.seed, config=cfg.dump()) as run:
        for s in ("navsim/op-adapt-h-nav-train", "b2d/op-adapt-h-carla-train", "wod/r2-train"):
            run.use_split(splits.load(s))
        train(cfg, run, d, a)


def train(cfg: H.HCfg, run, d: Path, a):
    dev = torch.device("cuda")
    lc = cfg.lcfg()
    model = L.LModel(lc).to(dev)
    base, _ = model.trainable()
    opt = torch.optim.AdamW([{"params": base, "lr": cfg.lr}], weight_decay=cfg.wd)
    scaler = torch.amp.GradScaler()
    tstd = np.load(L.r2t() / "teacher" / "tstd.npy")
    lossf = H.Losses(model.net, cfg, tstd, dev)
    step, ck = 0, d / "ckpt.pt"
    if ck.exists() and not a.fresh:
        st = torch.load(ck, map_location="cpu", weights_only=False)
        model.load_state(st["model"]), opt.load_state_dict(st["opt"]), scaler.load_state_dict(st["scaler"])
        step = st["step"]
        run.info(f"resumed at step {step}")
    run.info(f"{cfg.name}: {cfg.steps} steps, roles {cfg.roles}, control {cfg.control}; trainable {sum(p.numel() for p in base) / 1e6:.1f} M")
    dl = torch.utils.data.DataLoader(H.Batcher(cfg), batch_size=None, sampler=range(step, 10 ** 9), num_workers=cfg.workers,
                                     prefetch_factor=3, pin_memory=True, persistent_workers=False)
    it = iter(dl)
    model.train()
    hist, wait, t_start, s0 = [], 0.0, time.time(), step
    torch.cuda.reset_peak_memory_stats()
    bar = run.tqdm(total=cfg.steps, initial=step, desc=cfg.name)
    kinds = {}
    while step < cfg.steps:
        tw = time.time()
        b = next(it)
        wait += time.time() - tw
        for k in b.get("kind", []):
            kinds[k] = kinds.get(k, 0) + 1
        bd = {k: v.to(dev, non_blocking=True) for k, v in b.items() if k != "kind"}
        o = H.forward(model, bd["imgs"], bd["valid"], bd["tc"])
        total, Ls = lossf(o, bd)
        frac = step / cfg.steps
        for g in opt.param_groups:
            g["lr"] = cfg.lr * min(1.0, (step + 1) / cfg.warmup) * 0.5 * (1 + np.cos(np.pi * frac))
        opt.zero_grad(set_to_none=True)
        if not torch.isfinite(total):
            raise FloatingPointError(f"non-finite loss at step {step}: { {k: float(v) for k, v in Ls.items()} }")
        scaler.scale(total).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(base, 1.0)
        scaler.step(opt)
        scaler.update()
        step += 1
        bar.update()
        hist.append({k: float(v) for k, v in Ls.items()} | {"total": float(total)})
        if step % 25 == 0 or step == cfg.steps:
            m = pd.DataFrame(hist[-25:]).mean()
            el = time.time() - t_start
            run.scalars({f"loss/{k}": v for k, v in m.items()}, step)
            run.scalars({"throughput/steps_per_s": (step - s0) / el, "throughput/data_wait_frac": wait / el,
                         "gpu/peak_gb": torch.cuda.max_memory_reserved() / 2 ** 30}, step)
            if step % 100 == 0:
                run.info(f"step {step}: " + ", ".join(f"{k} {v:.4f}" for k, v in m.items()) +
                         f"; {(step - s0) / el:.2f} it/s, wait {wait / el:.2f}, {torch.cuda.max_memory_reserved() / 2 ** 30:.1f} GB, kinds {kinds}")
        if step % cfg.ckpt_every == 0 and step < cfg.steps:
            torch.save({"model": model.state(), "opt": opt.state_dict(), "scaler": scaler.state_dict(), "step": step}, d / "ckpt.tmp.pt")
            (d / "ckpt.tmp.pt").replace(ck)
    bar.close()
    del it, dl
    torch.save({"model": model.state(), "cfg": lc.dump(), "hcfg": cfg.dump()}, d / "ckpt-final.pt")
    ck.unlink(missing_ok=True)
    run.summary.update(steps=step, train_s=time.time() - t_start, data_wait_frac=wait / (time.time() - t_start),
                       peak_gb=torch.cuda.max_memory_reserved() / 2 ** 30, kinds=kinds)
    if not a.no_eval:
        model.eval()
        with ProcessPoolExecutor(16) as ex:
            r = dev_metrics({cfg.name: model}, dev, ex)[cfg.name]
        (d / "dev.json").write_text(json.dumps(r, indent=1))
        run.info("dev " + json.dumps(r))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("teacher")
    p.add_argument("--domains", nargs="+", default=list(H.DOMS))
    p.add_argument("--batch", type=int, default=32)
    p = sp.add_parser("train")
    p.add_argument("--arm", required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--steps", type=int, default=0)
    p.add_argument("--fresh", action="store_true")
    p.add_argument("--no-eval", action="store_true")
    for name in ("dev", "probe"):
        p = sp.add_parser(name)
        p.add_argument("--models", nargs="+", required=True)
        p.add_argument("--workers", type=int, default=24)
    p = sp.add_parser("probe-table")
    p.add_argument("--model", required=True)
    p.add_argument("--ref", default="O")
    p = sp.add_parser("link")
    p.add_argument("--models", nargs="+", required=True)
    p = sp.add_parser("navhard")
    p.add_argument("--models", nargs="+", required=True)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--cpus", default="100-149")
    a = ap.parse_args()
    {"teacher": cmd_teacher, "train": cmd_train, "dev": cmd_dev, "probe": cmd_probe, "probe-table": cmd_probe_table, "link": cmd_link,
     "navhard": cmd_navhard}[a.cmd](a)
