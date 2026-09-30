"""op-adapt L trainer (op-train venv; todos/2026-10-01-op-adapt-L-prereg.md; library jevdrive/op_adapt_l.py).

  teacher   the original model on every row of the full WOD val domain -> $L/t/teacher/wodval (r2's teacher_domain, same path)
  train     one run (config by name from jevdrive.op_adapt_l_arms or --cfg JSON): fp16 + GradScaler, AdamW, warm-up + cosine,
            checkpoint / resume, dev eval every --eval-every steps; log.txt / events.jsonl / tb/ / STATUS, DONE or ERROR
  selftest  numeric checks: step-0 identity with the original, intent adapter off = original, imitation loss finite

  CUDA_VISIBLE_DEVICES=0 taskset -c 8-19 python scripts/op_adapt_l_train.py train --arm sel_s4ia --seed 0
Run dir: $L/runs/<arm>-s<seed>/.
"""
import argparse, json, queue, sys, threading, time, traceback
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from jevdrive import op_adapt as A  # noqa: E402
from jevdrive import op_adapt_l as L  # noqa: E402
from jevdrive import op_adapt_l_arms as ARMS  # noqa: E402
from jevdrive import op_adapt_r2 as R  # noqa: E402


class Log:
    """RunLog on a fixed directory (resume appends): log.txt, events.jsonl, tb/."""

    def __init__(self, d: Path):
        import logging
        d.mkdir(parents=True, exist_ok=True)
        self.log = logging.getLogger(f"L.{d.name}")
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


def write_status(d: Path, **kw):
    tmp = d / "STATUS.tmp"
    tmp.write_text(json.dumps({"t": time.strftime("%Y-%m-%d %H:%M:%S"), **kw}, default=float))
    tmp.replace(d / "STATUS")


# ---------------------------------------------------------------- dev set and dev metrics
def dev_rows(D: L.Data, cap=1200, seed=0) -> dict:
    """Fixed dev row sets: WOD dev slices, contrast sets and other rows, nuScenes dev rows."""
    rng = np.random.default_rng(seed)
    pick = lambda r, n: np.sort(rng.choice(r, min(n, len(r)), replace=False))  # noqa: E731
    out = {}
    for s in L.SLICE3 + L.CONTRAST:
        out[s] = pick(D.rows("wod", "dev", s, need_future=True), cap)
    used = set()
    for r in out.values():
        used |= set(r.tolist())
    allr = D.rows("wod", "dev")
    ex = np.zeros(len(D.dom["wod"]), bool)
    for s in L.SLICE3:
        ex[D.rows("wod", "dev", s, need_future=True)] = True
    out["other"] = pick(allr[~ex[allr]], 2500)
    out["nus"] = pick(D.rows("nus", "dev"), 2000)
    return out


@torch.no_grad()
def dev_eval(model: L.LModel, D: L.Data, rows: dict, dev, extra=None) -> dict:
    """Capture / false-trigger / drift / slow / fast of the model against the original (teacher plans) on the dev row sets."""
    r, plans = {}, {}
    for k, rr in rows.items():
        plans[k] = L.fwd_rows(model, D, "nus" if k == "nus" else "wod", rr, dev)["plan"]
    tab, cam = D.tab["wod"], L.CAM_X["wod"]
    tea = lambda dn, rr: np.asarray(D.tea[dn]["mu"][rr], np.float32)  # noqa: E731
    grp = lambda rr: np.asarray(tab["seq"])[rr]  # noqa: E731
    for s in L.SLICE3 + L.CONTRAST:
        rr = rows[s]
        ma, mo = L.row_metrics(plans[s], tab, rr, cam), L.row_metrics(tea("wod", rr), tab, rr, cam)
        keys = {"start": ["cap_start"], "stop": ["cap_stop"], "turn_onset": ["cap_turn_onset"], "stay": ["false_start"],
                "control": ["false_stop", "false_turn"], "straight_int": ["false_turn"]}[s]
        for k in keys + ["lon3", "lat3", "slow", "fast"]:
            d = L.paired_delta(ma[k], mo[k], grp(rr), B=300)
            r[f"{s}/{k}"] = d.get("delta", float("nan"))
            r[f"{s}/{k}/adapt"], r[f"{s}/{k}/orig"] = d.get("adapt", float("nan")), d.get("orig", float("nan"))
    dr = []
    for k, dn in (("other", "wod"), ("nus", "nus")):
        rr = rows[k]
        d = A.plan_drift(plans[k], tea(dn, rr))
        r[f"drift_{k}_median"], r[f"drift_{k}_p95"] = float(np.median(d)), float(np.percentile(d, 95))
        dr.append(d)
        v0 = np.asarray(D.tab[dn]["v0"])[rr] if dn == "wod" else np.asarray(R.ego_speed(D.dom[dn], rr, D.tea[dn]), float)
        for who, pl in (("adapt", plans[k]), ("orig", tea(dn, rr))):
            r[f"{k}/slow/{who}"] = float(R.slow_flag(pl, v0).mean())
            r[f"{k}/fast/{who}"] = float((R.v_at(pl, 2.0) > v0 + np.maximum(1.0, 0.2 * v0)).mean())
        r[f"{k}/slow/delta_pp"] = 100 * (r[f"{k}/slow/adapt"] - r[f"{k}/slow/orig"])
        r[f"{k}/fast/delta_pp"] = 100 * (r[f"{k}/fast/adapt"] - r[f"{k}/fast/orig"])
    d = np.concatenate(dr)
    r["drift_median"], r["drift_p95"] = float(np.median(d)), float(np.percentile(d, 95))
    return r


def select_score(r: dict) -> dict:
    """The pre-registered selection rule on one dev dict (prereg 'Auswahl der Konfiguration'): a config qualifies when the
    dev drift median <= 0.10 m and p95 <= 0.50 m, every false-trigger delta <= +2 pp, and slow / fast deltas <= +2 pp; the
    score is the mean capture delta of the three slices."""
    cap = np.mean([r["start/cap_start"], r["stop/cap_stop"], r["turn_onset/cap_turn_onset"]])
    viol = {"drift_median": r["drift_median"] - 0.10, "drift_p95": r["drift_p95"] - 0.50,
            "false_start": 100 * r["stay/false_start"] - 2.0, "false_stop": 100 * r["control/false_stop"] - 2.0,
            "false_turn": 100 * max(r["control/false_turn"], r["straight_int/false_turn"]) - 2.0,
            "slow_pp": max(r["other/slow/delta_pp"], r["nus/slow/delta_pp"]) - 2.0,
            "fast_pp": max(r["other/fast/delta_pp"], r["nus/fast/delta_pp"]) - 2.0}
    return {"cap_gain": float(cap), "violations": {k: float(v) for k, v in viol.items()}, "qualifies": bool(all(v <= 0 for v in viol.values()))}


# ---------------------------------------------------------------- teacher
def cmd_teacher(a):
    import op_adapt_r2_train as T
    dev = torch.device("cuda")
    model = R.Model(R.ARMS["O"]).to(dev).eval()
    didx = A.distill_index(model.net.slices)
    root = L.lroot("t")
    d = R.Domain("wodval", root)
    t0 = time.time()
    T.teacher_domain(model, d, root / "teacher" / "wodval", dev, a.batch, didx)
    print(f"teacher wodval: {len(d)} rows in {time.time() - t0:.0f} s")


# ---------------------------------------------------------------- train
def cmd_train(a):
    cfg = ARMS.get(a.arm, seed=a.seed, steps=a.steps, eval_every=a.eval_every)
    d = Path(a.run_dir) if a.run_dir else L.lroot("runs", f"{a.arm}-s{a.seed}")
    d.mkdir(parents=True, exist_ok=True)
    for f in ("DONE", "ERROR"):
        (d / f).unlink(missing_ok=True)
    log = Log(d)
    try:
        train(cfg, d, log, a)
        (d / "DONE").write_text(time.strftime("%Y-%m-%d %H:%M:%S\n"))
    except BaseException as e:                                      # noqa: BLE001
        (d / "ERROR").write_text(f"{type(e).__name__}: {e}\n{traceback.format_exc()}")
        log.info(f"ERROR {e}\n{traceback.format_exc()}")
        log.event("error", error=str(e))
        raise


def train(cfg: L.LCfg, d: Path, log: Log, a):
    torch.manual_seed(cfg.seed)
    dev = torch.device("cuda")
    t0 = time.time()
    D = L.Data(("wod", "nus"))
    model = L.LModel(cfg).to(dev)
    base, new = model.trainable()
    groups = ([{"params": base, "lr": cfg.lr}] if base else []) + ([{"params": new, "lr": cfg.lr_new}] if new else [])
    lr0 = [g["lr"] for g in groups]
    opt = torch.optim.AdamW(groups, weight_decay=cfg.wd)
    scaler = torch.amp.GradScaler()
    mix = L.Mixer(cfg, D)
    lossf = L.Losses(model.net, cfg, D.tstd, dev)
    drows = dev_rows(D)
    step, ck = 0, d / "ckpt.pt"
    if ck.exists() and not a.fresh:
        st = torch.load(ck, map_location="cpu", weights_only=False)
        model.load_state(st["model"])
        opt.load_state_dict(st["opt"])
        scaler.load_state_dict(st["scaler"])
        step = st["step"]
        log.info(f"resumed from step {step}")
        log.event("resume", step=step)
    else:
        (d / "config.json").write_text(json.dumps({"cfg": cfg.dump(), "pools": mix.describe(),
                                                   "n_trainable": int(sum(p.numel() for p in base + new)),
                                                   "dev_rows": {k: int(len(v)) for k, v in drows.items()}}, indent=1))
        log.event("start", cfg=cfg.dump(), pools=mix.describe())
    log.info(f"{cfg.name}: {cfg.steps} steps x {cfg.batch}; pools {mix.describe()}; trainable "
             f"{sum(p.numel() for p in base + new) / 1e6:.2f} M; data {time.time() - t0:.0f} s")

    def evaluate(tag):
        model.eval()
        t1 = time.time()
        r = dev_eval(model, D, drows, dev)
        r["select"] = select_score(r)
        model.train()
        for k, v in r.items():
            if isinstance(v, float) and np.isfinite(v):
                log.scalar(f"dev/{k}", v, step)
        log.scalar("dev/cap_gain", r["select"]["cap_gain"], step)
        log.event("dev", step=step, tag=tag, eval_s=time.time() - t1, **{k: v for k, v in r.items() if k != "select"}, select=r["select"])
        log.info(f"dev[{tag}] step {step}: cap gain {r['select']['cap_gain']:+.3f} drift {r['drift_median']:.3f}/{r['drift_p95']:.3f} "
                 f"qualifies {r['select']['qualifies']} viol {{{', '.join(f'{k}: {v:+.2f}' for k, v in r['select']['violations'].items() if v > 0)}}} "
                 f"({time.time() - t1:.0f} s)")
        return r
    if step == 0 and not a.no_eval:
        (d / "dev_before.json").write_text(json.dumps(evaluate("before"), indent=1))
    q, stop = queue.Queue(maxsize=cfg.prefetch), threading.Event()

    def producer(k):
        rng = np.random.default_rng([cfg.seed, k, step])
        try:
            while not stop.is_set():
                b = L.assemble(D, mix.draw(rng))
                for x in ("trunk", "valid", "tc"):
                    b[x] = torch.from_numpy(b[x]).pin_memory()
                q.put(b)
        except BaseException as e:                                  # noqa: BLE001
            q.put(e)
    th = [threading.Thread(target=producer, args=(k,), daemon=True) for k in range(cfg.loaders)]
    for t in th:
        t.start()
    from tqdm import tqdm
    model.train()
    hist, nonfinite, tstart, s_start = [], 0, time.time(), step
    torch.cuda.reset_peak_memory_stats()
    bar = tqdm(total=cfg.steps, initial=step, desc=cfg.name, mininterval=30)
    while step < cfg.steps:
        b = q.get()
        if isinstance(b, BaseException):
            raise RuntimeError("batch producer failed") from b
        b = L.to_dev(b, dev)
        o = model(b["trunk"], b["valid"], b["tc"], b["intent"])
        total, Ls = lossf(o, b)
        frac = step / cfg.steps
        for g, l0 in zip(opt.param_groups, lr0):
            g["lr"] = l0 * min(1.0, (step + 1) / cfg.warmup) * 0.5 * (1 + np.cos(np.pi * frac))
        opt.zero_grad(set_to_none=True)
        if not torch.isfinite(total):
            nonfinite += 1
            log.event("nonfinite", step=step, **{k: float(v) for k, v in Ls.items()})
            if nonfinite > 10:
                raise FloatingPointError(f"non-finite loss {nonfinite} times")
        scaler.scale(total).backward()
        scaler.unscale_(opt)
        for g in opt.param_groups:
            torch.nn.utils.clip_grad_norm_(g["params"], 1.0)
        scaler.step(opt)
        scaler.update()
        step += 1
        bar.update()
        hist.append({k: float(v) for k, v in Ls.items()} | {"total": float(total)})
        if step % 50 == 0 or step == cfg.steps:
            m = pd.DataFrame(hist[-50:]).mean()
            el = time.time() - tstart
            sps = (step - s_start) * cfg.batch / el
            for k, v in m.items():
                log.scalar(f"loss/{k}", v, step)
            log.scalar("throughput/seq_per_s", sps, step)
            vram = torch.cuda.max_memory_reserved() / 2 ** 30
            log.scalar("gpu/peak_reserved_gb", vram, step)
            write_status(d, step=step, steps=cfg.steps, loss=float(m.total), sps=sps, vram_gb=vram,
                         eta_min=(cfg.steps - step) * cfg.batch / max(sps, 1e-6) / 60)
            if step % 250 == 0:
                log.info(f"step {step}: " + ", ".join(f"{k} {v:.4f}" for k, v in m.items()) + f"; {sps:.0f} seq/s, {vram:.1f} GB")
        if step % cfg.ckpt_every == 0 and step < cfg.steps:
            tmp = d / "ckpt.tmp.pt"
            torch.save({"model": model.state(), "opt": opt.state_dict(), "scaler": scaler.state_dict(), "step": step,
                        "cfg": cfg.dump()}, tmp)
            tmp.replace(ck)
        if step % cfg.eval_every == 0 and step < cfg.steps and not a.no_eval:
            evaluate("periodic")
    stop.set()
    bar.close()
    r = evaluate("final") if not a.no_eval else {}
    r |= {"steps": step, "train_s": time.time() - tstart, "peak_reserved_gb": torch.cuda.max_memory_reserved() / 2 ** 30,
          "seq_per_s": (step - s_start) * cfg.batch / (time.time() - tstart), "nonfinite": nonfinite}
    (d / "dev.json").write_text(json.dumps(r, indent=1, default=float))
    torch.save({"model": model.state(), "cfg": cfg.dump(), "dev": r}, d / "ckpt-final.pt")
    ck.unlink(missing_ok=True)                                  # resumable checkpoint (with optimizer state) is scratch
    log.event("end", **{k: v for k, v in r.items() if not isinstance(v, dict)})
    log.info(f"done: steps {step}, {r['seq_per_s']:.0f} seq/s, peak {r['peak_reserved_gb']:.1f} GB")


# ---------------------------------------------------------------- selftest
def cmd_selftest(a):
    dev = torch.device("cuda")
    D = L.Data(("wod", "nus"))
    rows = D.rows("wod", "dev", "start", need_future=True)[:64]
    O = L.load_model(None, dev)
    ref = L.fwd_rows(O, D, "wod", rows, dev)["plan"]
    tea = np.asarray(D.tea["wod"]["mu"][rows], np.float32)
    res = {"orig_vs_teacher_plan_maxabs": float(np.abs(ref - tea).max())}
    for name in ("sel_s4ia", "sel_polia", "sel_polid"):
        cfg = ARMS.get(name)
        m = L.LModel(cfg).to(dev).eval()
        p = L.fwd_rows(m, D, "wod", rows, dev)["plan"]
        res[f"{name}_step0_maxabs"] = float(np.abs(p - ref).max())
        res[f"{name}_trainable_M"] = sum(q.numel() for q in m.trainable()[0] + m.trainable()[1]) / 1e6
    # imitation loss on a batch and gradients reach the intended parameters
    cfg = ARMS.get("sel_s4ia")
    mix = L.Mixer(cfg, D)
    b = L.to_dev(L.assemble(D, mix.draw(np.random.default_rng(0))), dev)
    m = L.LModel(cfg).to(dev).train()
    lossf = L.Losses(m.net, cfg, D.tstd, dev)
    o = m(b["trunk"], b["valid"], b["tc"], b["intent"])
    total, Ls = lossf(o, b)
    total.backward()
    res |= {"loss": {k: float(v) for k, v in Ls.items()}, "total": float(total), "batch": len(b["role"]),
            "roles": np.bincount(b["role"].cpu().numpy()).tolist(),
            "adapter_grad_abs": float(m.adapter.E.grad.abs().max()),
            "s4_grad_abs": float(max(p.grad.abs().max() for p in m.trainable()[0] if p.grad is not None))}
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("teacher")
    p.add_argument("--batch", type=int, default=256)
    p = sp.add_parser("train")
    p.add_argument("--arm", required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--steps", type=int, default=0, help="0 = the arm's default")
    p.add_argument("--eval-every", type=int, default=0)
    p.add_argument("--run-dir", default="")
    p.add_argument("--fresh", action="store_true")
    p.add_argument("--no-eval", action="store_true")
    sp.add_parser("selftest")
    a = ap.parse_args()
    {"teacher": cmd_teacher, "train": cmd_train, "selftest": cmd_selftest}[a.cmd](a)
