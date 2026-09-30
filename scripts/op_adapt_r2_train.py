"""op-adapt round 2 trainer (op-train venv; todos/2026-09-29-op-adapt-r2-prereg.md §2, §3.3; lib jevdrive/op_adapt_r2.py).

  pack     flat trunk memmaps + sample tables per domain (R2/t/{trunk,samples}); source `c` = package C's index,
           `round1` = the round-1 caches directly (nus / wod / nav / p5 / wodval; tests and the equivalence check)
  teacher  the original model on every sample of the given domains -> R2/t/teacher/<domain>/{uid,out,mu,logstd,temporal}.npy,
           then the shared distillation scale tstd.npy and the A-bhv scale sd_dv.npy
  train    one run (arm, seed, lambda_s): fp16 + GradScaler, AdamW (stage 4 3e-5, new modules 1e-3, wd 0.01), warm-up +
           cosine over the steps, checkpoint every --ckpt-every steps and resume from it, dev eval every --eval-every
           steps; log.txt / events.jsonl / tb/ / STATUS, DONE or ERROR at the end
  select   the §2 lambda_s rule on the dev.json of the two A seed-0 runs -> R2/lambda_s.json
  equiv    round-1 numerics: with the r2 losses off, round-1 data and batch, the r2 code gives round 1's loss

  CUDA_VISIBLE_DEVICES=1 taskset -c 40-47 python scripts/op_adapt_r2_train.py train --arm A --seed 0 --lam-s 1
Run dir: R2/train-<arm>-s<seed>[-ls<lam>]/ (--run-dir to override).
"""
import argparse, json, queue, sys, threading, time, traceback
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from jevdrive import op_adapt as A  # noqa: E402
from jevdrive import op_adapt_r2 as R  # noqa: E402

AT = (0.275, 0.525)
TRAIN_DOMAINS = ("simC", "simK", "nus", "wod", "nav", "off")


class Log:
    """RunLog on a fixed directory (resume appends): log.txt, events.jsonl, tb/."""

    def __init__(self, d: Path):
        import logging
        self.dir = d
        d.mkdir(parents=True, exist_ok=True)
        self.log = logging.getLogger(f"r2.{d.name}")
        self.log.setLevel(logging.INFO)
        if not self.log.handlers:
            fmt = logging.Formatter("%(asctime)s %(message)s", "%m-%d %H:%M:%S")
            for h in (logging.FileHandler(d / "log.txt"), logging.StreamHandler(sys.stderr)):
                h.setFormatter(fmt)
                self.log.addHandler(h)
        self._ev = open(d / "events.jsonl", "a", buffering=1)
        try:
            from torch.utils.tensorboard import SummaryWriter
            self.tb = SummaryWriter(d / "tb", flush_secs=30)
        except ImportError:
            self.tb = None

    def info(self, m):
        self.log.info(m)

    def event(self, kind, **f):
        self._ev.write(json.dumps({"t": round(time.time(), 3), "kind": kind, **f}, default=float) + "\n")

    def scalar(self, tag, v, step):
        if self.tb:
            self.tb.add_scalar(tag, v, step)
        self.event("scalar", tag=tag, value=float(v), step=step)


# ---------------------------------------------------------------- pack
def cmd_pack(a):
    for dn in a.domains:
        t0 = time.time()
        if a.source == "c":
            ix = R.index_c(dn)
        else:
            import pandas as pd  # noqa: F811
            from jevdrive import op_adapt_data as D
            if dn == "nus":
                ix = R.index_nus()
            elif dn == "wod":
                ix = R.index_stream("wodtrain", pd.read_parquet(D.root() / "wod_train_labels.parquet"), frac_dev=0.05)
            elif dn == "wodval":
                ix = R.index_stream("wod")
            elif dn == "p5":
                ix = R.index_stream("p5", targets_only=False)
            elif dn == "nav":
                ix = R.index_nav()
            else:
                raise SystemExit(f"{dn}: no round-1 cache, use --source c")
        if a.limit:
            ix = ix[ix.cache.isin(sorted(ix.cache.unique())[: a.limit])]
        root = Path(a.root) if a.root else None
        R.pack(dn, ix, root, a.workers, copy=a.copy)
        if a.det:
            R.pack_det(dn, root)
        print(f"{dn}: {len(ix)} samples from {ix.cache.nunique()} files in {time.time() - t0:.0f} s")


# ---------------------------------------------------------------- teacher
@torch.no_grad()
def teacher_domain(model, d: R.Domain, out_dir: Path, dev, bs=256, didx=None):
    out_dir.mkdir(parents=True, exist_ok=True)
    n = len(d)
    sl = model.net.slices
    pi, ps = A.plan_index(sl), np.arange(sl["plan"].start + 495, sl["plan"].start + 990)
    di = torch.as_tensor(didx, device=dev)
    mm = {k: np.lib.format.open_memmap(out_dir / f"{k}.tmp.npy", "w+", dt, (n,) + sh) for k, dt, sh in (
        ("out", np.float16, (len(didx),)), ("mu", np.float32, (33, 15)), ("logstd", np.float32, (33, 15)),
        ("temporal", np.float16, (512,)))}
    from tqdm import tqdm
    for i in tqdm(range(0, n, bs), desc=f"teacher {d.name}", mininterval=30):
        r = np.arange(i, min(i + bs, n))
        x, v = d.gather(r)
        o = A.stage4_policy(model.net, torch.from_numpy(x).to(dev), AT, torch.from_numpy(d.tc[r]).to(dev),
                            torch.from_numpy(v).to(dev))
        out = o["outputs"].float()
        mm["out"][r] = out[:, di].half().cpu().numpy()                      # round 1: .half() of the fp32 cast
        mm["mu"][r] = out[:, pi].cpu().numpy().reshape(-1, 33, 15)
        mm["logstd"][r] = out[:, ps].cpu().numpy().reshape(-1, 33, 15)
        mm["temporal"][r] = o["select_4"].float().half().cpu().numpy()
    for k, m in mm.items():
        m.flush()
        del m
        (out_dir / f"{k}.tmp.npy").replace(out_dir / f"{k}.npy")
    np.save(out_dir / "uid.npy", d.uid)


def distill_rows(d: R.Domain, split="train") -> np.ndarray:
    m = (d.col("split") == split)
    if d.name.startswith("sim"):
        return np.flatnonzero(m & (d.col("sign") == -1))
    return np.flatnonzero(m & d.col("normal", False, bool))


def cmd_teacher(a):
    dev = torch.device("cuda")
    root = Path(a.root) if a.root else R.r2("t")
    model = R.Model(R.ARMS["O"]).to(dev).eval()
    didx = A.distill_index(model.net.slices)
    for dn in a.domains:
        t0 = time.time()
        teacher_domain(model, R.Domain(dn, root), root / "teacher" / dn, dev, a.batch, didx)
        print(f"teacher {dn}: {time.time() - t0:.0f} s")
    # the shared distillation scale: std of the original outputs over every train distillation row
    parts = []
    for dn in TRAIN_DOMAINS:
        if (root / "samples" / f"{dn}.parquet").exists() and (root / "teacher" / dn / "out.npy").exists():
            d = R.Domain(dn, root)
            rows = distill_rows(d)
            parts.append(np.asarray(R.load_teacher(dn, root)["out"][rows], np.float32))
    if parts:
        x = np.concatenate(parts)
        np.save(root / "teacher" / "tstd.npy", np.maximum(x.std(0, ddof=1), 1e-3).astype(np.float32))
        print(f"tstd over {len(x)} distillation rows")
    for dn in ("simC", "simK"):
        if (root / "samples" / f"{dn}.parquet").exists():
            d = R.Domain(dn, root)
            m = (d.col("split") == "train") & (d.col("sign") == 1)
            dv = R.dv_star(d)[m]
            np.save(root / "teacher" / f"sd_dv_{dn}.npy", np.nanstd(dv, 0).clip(1e-3).astype(np.float32))


# ---------------------------------------------------------------- train
def load_all(domains, root=None):
    doms, tea, sc = {}, {}, {}
    for dn in domains:
        r = root or R.r2("t")
        if (r / "samples" / f"{dn}.parquet").exists():
            doms[dn] = R.Domain(dn, r)
            tea[dn] = R.load_teacher(dn, r)
            sc[dn] = R.load_score(dn)
    return doms, tea, sc


def run_dir(cfg: R.RunCfg, a) -> Path:
    return Path(a.run_dir) if a.run_dir else R.r2(f"train-{cfg.tag}")


def write_status(d: Path, **kw):
    tmp = d / "STATUS.tmp"
    tmp.write_text(json.dumps({"t": time.strftime("%Y-%m-%d %H:%M:%S"), **kw}, default=float))
    tmp.replace(d / "STATUS")


def cmd_train(a):
    cfg = R.RunCfg(arm=a.arm, seed=a.seed, lam_s=a.lam_s, max_steps=a.max_steps, eval_every=a.eval_every,
                   ckpt_every=a.ckpt_every, batch=a.batch)
    d = run_dir(cfg, a)
    d.mkdir(parents=True, exist_ok=True)
    for f in ("DONE", "ERROR"):
        (d / f).unlink(missing_ok=True)
    log = Log(d)
    try:
        train(cfg, d, log, a)
        (d / "DONE").write_text(time.strftime("%Y-%m-%d %H:%M:%S\n"))
    except BaseException as e:                                      # noqa: BLE001  (KeyboardInterrupt included)
        (d / "ERROR").write_text(f"{type(e).__name__}: {e}\n{traceback.format_exc()}")
        log.info(f"ERROR {e}\n{traceback.format_exc()}")
        log.event("error", error=str(e))
        raise


def train(cfg: R.RunCfg, d: Path, log: Log, a):
    arm = R.ARMS[cfg.arm]
    if arm.name in ("O", "Z"):
        raise SystemExit(f"arm {arm.name} is not trained here")
    torch.manual_seed(cfg.seed)
    dev = torch.device("cuda")
    t0 = time.time()
    root = Path(a.root) if a.root else None
    doms, tea, sc = load_all(TRAIN_DOMAINS, root)
    det = R.det_adapter() if arm.det else None
    if arm.det and det is None:
        raise SystemExit("arm needs package D's adapter (jevdrive/op_adapt_det.py)")
    det_fn = R.DetSource(root) if arm.det else None
    model = R.Model(arm, det=det).to(dev)
    s4, new = model.trainable()
    groups = ([{"params": s4, "lr": cfg.lr}] if s4 else []) + [{"params": new, "lr": cfg.lr_new}]
    lr0 = [g["lr"] for g in groups]
    opt = torch.optim.AdamW(groups, weight_decay=cfg.wd)
    scaler = torch.amp.GradScaler()
    tstd = torch.as_tensor(np.load((root or R.r2("t")) / "teacher" / "tstd.npy"), device=dev)
    sd = [np.load(p) for p in sorted((root or R.r2("t")).glob("teacher/sd_dv_*.npy"))]
    sd_dv = np.mean(sd, 0) if sd else None
    mix = R.Mixer(cfg, doms, sc)
    asm = R.Assembler(doms, tea, sc, A.distill_index(model.net.slices), mix.pw)
    asm.det_fn = det_fn
    lossf = R.Losses(model.net, arm, cfg, tstd, dev, sd_dv)
    step = 0
    ck = d / "ckpt.pt"
    if ck.exists() and not a.fresh:
        st = torch.load(ck, map_location="cpu", weights_only=False)
        model.load_state(st["model"])
        opt.load_state_dict(st["opt"])
        scaler.load_state_dict(st["scaler"])
        step = st["step"]
        log.info(f"resumed from step {step}")
        log.event("resume", step=step)
    else:
        R.save_json(d / "config.json", R.cfg_dict(cfg) | {"mixer": mix.describe(), "domains": {k: len(v) for k, v in doms.items()},
                                                          "scores": {k: v is not None for k, v in sc.items()}})
        log.event("start", cfg=R.cfg_dict(cfg), mixer=mix.describe())
    log.info(f"{cfg.tag}: {cfg.steps} steps x {cfg.batch} labelled sequences; mixer {mix.describe()}; "
             f"data {time.time() - t0:.0f} s")
    o_probe = json.loads((R.r2("t") / "dev_o_probe.json").read_text()) if (R.r2("t") / "dev_o_probe.json").exists() else None

    def evaluate(tag):
        model.eval()
        r = R.dev_eval(model, doms, tea, sc, arm, dev, det_fn=det_fn, o_probe=o_probe)
        model.train()
        for k, v in r.items():
            if isinstance(v, float) and np.isfinite(v):
                log.scalar(f"dev/{k}", v, step)
        log.event("dev", step=step, tag=tag, **r)
        return r
    if step == 0:
        base = evaluate("before")
        R.save_json(d / "dev_before.json", base)
    # producers
    q, stop = queue.Queue(maxsize=a.prefetch), threading.Event()

    def producer(k):
        rng = np.random.default_rng([cfg.seed, k, step])
        try:
            while not stop.is_set():
                b = asm(mix.draw(rng))
                for x in ("trunk", "valid", "tc"):
                    b[x] = torch.from_numpy(b[x]).pin_memory()
                q.put(b)
        except BaseException as e:                                  # noqa: BLE001  surfaced in the main loop
            q.put(e)
    th = [threading.Thread(target=producer, args=(k,), daemon=True) for k in range(a.loaders)]
    for t in th:
        t.start()
    from tqdm import tqdm
    model.train()
    hist, nonfinite, tstart, s_start = [], 0, time.time(), step
    torch.cuda.reset_peak_memory_stats()
    bar = tqdm(total=cfg.steps, initial=step, desc=cfg.tag, mininterval=30)
    while step < cfg.steps:
        b = q.get()
        if isinstance(b, BaseException):
            raise RuntimeError("batch producer failed") from b
        b = R.to_dev(b, dev)
        o = model(b["trunk"], b["valid"], b["tc"], b.get("det_tok"), b.get("det_mask"), AT)
        total, L = lossf(o, b)
        if step == 0 and len(b["sc_rows"]):                         # §6: L_score exactly 0 where op is in Top at step 0
            with torch.no_grad():
                r_ = b["sc_rows"]
                dd = R.huber_d(R.plan_ch(lossf.plan(o)[r_]), b["cand"], b["tsig"][r_]).masked_fill(~b["top"], float("inf"))
                opk = [asm.op_k.get(x, -1) for x in b["doms"][r_.cpu().numpy()]]
                ontop = torch.tensor([bool(b["top"][i, k]) if k >= 0 else False for i, k in enumerate(opk)], device=dev)
                v = dd.amin(1)[ontop]
                log.event("score0", n_on_top=int(ontop.sum()), max_loss_on_top=float(v.max()) if len(v) else 0.0)
                log.info(f"step 0 L_score on op-in-Top slots: n {int(ontop.sum())}, max {float(v.max()) if len(v) else 0:.3g}")
        frac = step / cfg.steps
        for g, l0 in zip(opt.param_groups, lr0):
            g["lr"] = l0 * min(1.0, (step + 1) / cfg.warmup) * 0.5 * (1 + np.cos(np.pi * frac))
        opt.zero_grad(set_to_none=True)
        if not torch.isfinite(total):
            nonfinite += 1
            log.event("nonfinite", step=step, **{k: float(v) for k, v in L.items()})
            if nonfinite > 10:
                raise FloatingPointError(f"non-finite loss {nonfinite} times")
        scaler.scale(total).backward()
        scaler.unscale_(opt)
        if s4:
            torch.nn.utils.clip_grad_norm_(s4, 1.0)
        scaler.step(opt)
        scaler.update()
        step += 1
        bar.update()
        hist.append({k: float(v) for k, v in L.items()} | {"total": float(total), "n_seq": int(len(b["y"]))})
        if step % 50 == 0 or step == cfg.steps:
            m = pd.DataFrame(hist[-50:]).mean()
            el = time.time() - tstart
            sps = (step - s_start) * cfg.batch / el
            for k, v in m.items():
                if k != "n_seq":
                    log.scalar(f"loss/{k}", v, step)
            log.scalar("throughput/labelled_seq_per_s", sps, step)
            log.scalar("throughput/seq_per_s", (step - s_start) * m.n_seq / el, step)
            vram = torch.cuda.max_memory_reserved() / 2 ** 30
            log.scalar("gpu/peak_reserved_gb", vram, step)
            write_status(d, step=step, steps=cfg.steps, loss=float(m.total), sps=sps, vram_gb=vram,
                         eta_min=(cfg.steps - step) * cfg.batch / max(sps, 1e-6) / 60)
            if step % 500 == 0:
                log.info(f"step {step}: " + ", ".join(f"{k} {v:.4f}" for k, v in m.items() if k != "n_seq")
                         + f"; {sps:.0f} labelled seq/s, {vram:.1f} GB")
        if step % cfg.ckpt_every == 0 or step == cfg.steps:
            tmp = d / "ckpt.tmp.pt"
            torch.save({"model": model.state(), "opt": opt.state_dict(), "scaler": scaler.state_dict(), "step": step,
                        "cfg": R.cfg_dict(cfg)}, tmp)
            tmp.replace(ck)
        if step % cfg.eval_every == 0 and step < cfg.steps:
            evaluate("periodic")
    stop.set()
    bar.close()
    r = evaluate("final")
    r |= {"steps": step, "train_s": time.time() - tstart, "peak_reserved_gb": torch.cuda.max_memory_reserved() / 2 ** 30,
          "labelled_seq_per_s": (step - s_start) * cfg.batch / (time.time() - tstart), "nonfinite": nonfinite}
    R.save_json(d / "dev.json", r)
    torch.save({"model": model.state(), "cfg": R.cfg_dict(cfg), "dev": r}, d / "ckpt-final.pt")
    log.event("end", **r)
    log.info(f"done: {r}")


# ---------------------------------------------------------------- select
def cmd_select(a):
    devs = {}
    for p in a.runs:
        c = json.loads((Path(p) / "config.json").read_text())
        devs[c["lam_s"]] = json.loads((Path(p) / "dev.json").read_text())
    res = R.select_lambda(devs)
    R.save_json(R.r2() / "lambda_s.json", res | {"runs": a.runs})
    print(json.dumps(res, indent=1))


# ---------------------------------------------------------------- equivalence with round 1
def cmd_equiv(a):
    """Round-1 numerics: same scenes, same batch, r2 code path with the r2 losses off -> same loss and gradients."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("r1", Path(__file__).parent / "op_adapt_train.py")
    r1 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(r1)
    from jevdrive import op_adapt_data as D
    dev = torch.device("cuda")
    lab = pd.read_parquet(D.root() / "nusc_labels.parquet")
    tr_sc, _, _ = r1.split_scenes(lab)
    have = {p.stem for p in D.root("nusc").glob("*.npz")}
    tr_sc = [s for s in tr_sc if s in have][: a.scenes]
    out = R.r2("equiv")
    root = out / "t"
    st = r1.Store(tr_sc, lab)
    ix = R.index_nus(tr_sc)
    R.pack("nus", ix, root)
    dm = R.Domain("nus", root)
    assert len(dm) == len(st.slot) and (dm.col("token") == st.token).all(), "row order differs from round 1"
    torch.manual_seed(0)
    net = A.load("cinque", torch.float16, trainable=A.stage4_weights()).to(dev).eval()
    didx = A.distill_index(net.slices)
    tgt1, tplan1 = r1.teacher(net, st, dev, didx)
    model = R.Model(R.ARMS["A"]).to(dev).eval()
    teacher_domain(model, dm, root / "teacher" / "nus", dev, 256, didx)
    tea = R.load_teacher("nus", root)
    res = {"scenes": len(tr_sc), "samples": len(dm),
           "teacher_out_maxabs": float(np.abs(np.asarray(tea["out"], np.float32) - tgt1.float().numpy()).max()),
           "teacher_plan_maxabs": float(np.abs(np.asarray(tea["mu"]) - tplan1).max())}
    # one round-1 batch
    rng = np.random.default_rng(0)
    keys = np.flatnonzero(st.key)
    kpos = keys[st.wide_vru[keys] | (st.y[keys, 1] > 0)]
    kneg = keys[~(st.wide_vru[keys] | (st.y[keys, 1] > 0))]
    half = a.batch // 2
    npos = int(round(0.3 * half))
    idx = np.r_[np.r_[rng.choice(kpos, npos), rng.choice(kneg, half - npos)], rng.integers(0, len(st.slot), a.batch - half)]
    tstd1 = tgt1[torch.from_numpy(st.normal)].float().std(0).clamp_min(1e-3).to(dev)
    heads = A.AuxHeads().to(dev)
    net.train()
    x, v, tc = st.batch(idx, dev)
    o = A.stage4_policy(net, x, AT, tc, v)
    ha, hb = heads(o["select_4"], o["tokens"])
    y = torch.from_numpy(st.y[idx[:half]]).to(dev)
    la = r1.loss_aux(ha[:half], y) + r1.loss_aux(hb[:half], y)
    nm = torch.from_numpy(st.normal[idx]).to(dev)
    tt = tgt1[torch.from_numpy(idx)].to(dev).float()
    e = ((o["outputs"].float()[:, torch.as_tensor(didx, device=dev)] - tt) / tstd1).pow(2).mean(1)
    ld = (e * nm).sum() / nm.sum().clamp_min(1)
    loss1 = la + a.lam_d * ld
    loss1.backward()
    g1 = {k: p.grad.detach().clone() for k, p in net.params.items() if p.requires_grad}
    # the r2 path on the same rows
    cfg = R.RunCfg(arm="A", round1=True, lam_d=a.lam_d)
    model.heads.load_state_dict(heads.state_dict())
    model.train()
    tstd2 = torch.as_tensor(np.maximum(np.asarray(tea["out"], np.float32)[dm.col("normal", False, bool)].std(0, ddof=1), 1e-3),
                            device=dev)
    asm = R.Assembler({"nus": dm}, {"nus": tea}, {"nus": None}, didx)
    seg = R.Seg("nus", idx, np.r_[np.ones(half, bool), np.zeros(a.batch - half, bool)], dm.col("normal", False, bool)[idx],
                np.zeros(a.batch, bool))
    b = R.to_dev(asm([seg]), dev)
    o2 = model(b["trunk"], b["valid"], b["tc"], action_t=AT)
    loss2, L2 = R.Losses(model.net, R.ARMS["A"], cfg, tstd2, dev)(o2, b)
    loss2.backward()
    g2 = {k: p.grad for k, p in model.net.params.items() if p.requires_grad}
    gd = max(float((g1[k] - g2[k]).abs().max()) for k in g1)
    gs = max(float(g1[k].abs().max()) for k in g1)
    res |= {"loss_r1": float(loss1), "loss_r2": float(loss2), "aux_r1": float(la), "aux_r2": float(L2["aux"]),
            "distill_r1": float(ld), "distill_r2": float(L2["distill"]), "tstd_maxrel": float(((tstd1 - tstd2).abs() / tstd1).max()),
            "loss_absdiff": abs(float(loss1) - float(loss2)), "grad_maxabs_diff": gd, "grad_maxabs": gs}
    res["pass"] = res["loss_absdiff"] <= 1e-5 * max(1.0, abs(float(loss1))) and gd <= 1e-3 * gs
    R.save_json(out / "equiv.json", res)
    print(json.dumps(res, indent=1))


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("pack")
    p.add_argument("--domains", nargs="+", required=True)
    p.add_argument("--source", choices=("c", "round1"), default="c")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--limit", type=int, default=0, help="first n cache files (tests)")
    p.add_argument("--det", action="store_true", help="also lay package D's tokens onto the flat rows")
    p.add_argument("--copy", action="store_true", help="copy the rows into one flat memmap instead of mapping the caches")
    p.add_argument("--root", default="")
    p = sp.add_parser("teacher")
    p.add_argument("--domains", nargs="+", required=True)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--root", default="")
    p = sp.add_parser("train")
    p.add_argument("--arm", required=True, choices=[k for k in R.ARMS if k not in ("O", "Z")])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--lam-s", type=float, default=1.0)
    p.add_argument("--batch", type=int, default=R.BATCH)
    p.add_argument("--max-steps", type=int, default=0, help="0 = 1.2 M sequences; staged launch: 2000 / 10 %%")
    p.add_argument("--eval-every", type=int, default=1000)
    p.add_argument("--ckpt-every", type=int, default=500)
    p.add_argument("--loaders", type=int, default=4)
    p.add_argument("--prefetch", type=int, default=6)
    p.add_argument("--run-dir", default="")
    p.add_argument("--root", default="")
    p.add_argument("--fresh", action="store_true", help="ignore an existing ckpt.pt")
    p = sp.add_parser("select")
    p.add_argument("--runs", nargs=2, required=True)
    p = sp.add_parser("equiv")
    p.add_argument("--scenes", type=int, default=12)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--lam-d", type=float, default=10.0)
    a = ap.parse_args()
    {"pack": cmd_pack, "teacher": cmd_teacher, "train": cmd_train, "select": cmd_select, "equiv": cmd_equiv}[a.cmd](a)


if __name__ == "__main__":
    main()
