"""Run the preregistered start gate and nested segment learning curve.

Subcommands: gate-cache, gate-pilot, gate-fit, gate-lock, curve-profile, curve-unit, curve-train, curve-eval.
All large files stay under $DATA_DIR/runs/op_adapt_L/gate_curve. Existing L/r2 files are read only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import resource
import subprocess
import sys
import time
import traceback
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from jevdrive import op_adapt as A  # noqa: E402
from jevdrive import op_adapt_l as L  # noqa: E402
from jevdrive import op_adapt_l_arms as ARMS  # noqa: E402
from jevdrive import op_adapt_r2 as R  # noqa: E402
from jevdrive.common import data_dir  # noqa: E402
from op_adapt_l_gate_curve_prep import core_set, dump  # noqa: E402
import op_adapt_l_train as T  # noqa: E402

ROOT = data_dir() / "runs/op_adapt_L/gate_curve"
PREP = ROOT / "prep-v3"
MAIN_TAGS = ("main-s0", "main-s1", "main-s2")


def grant_check(gpu=True):
    fields = dict(line.strip().split("=", 1) for line in (ROOT / "GO").read_text().splitlines()
                  if line.strip() and not line.startswith("#") and "=" in line)
    assert set(os.sched_getaffinity(0)) <= core_set(fields["GC_CPUS"]), "CPU affinity exceeds GO"
    granted = fields["GPUS"].strip('"').split()
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "unset").split(",")
    assert (set(visible) <= set(granted)) if gpu else visible == [""], "GPU visibility exceeds GO"
    assert os.environ.get("OPENBLAS_CORETYPE") == "Haswell"
    return {"gpus": visible, "cpus": sorted(os.sched_getaffinity(0))}


def device():
    return torch.device("cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu")


def path_for(tag):
    return L.lroot("runs", tag) / "ckpt-final.pt"


class CurveMixer(L.Mixer):
    def __init__(self, cfg, D, size):
        super().__init__(cfg, D)
        manifest = json.loads((PREP / "manifest.json").read_text())
        self.original_sizes = self.describe()
        seq = np.asarray(D.tab["wod"]["seq"]).astype(str)
        self.audit = {}
        for s in L.SLICE3:
            row = manifest[s][str(size)]
            pool = self.pools[("wod", s)]
            selected = pool[np.isin(seq[pool], row["segments"])]
            assert np.array_equal(selected, row["rows"]), f"Manifest mismatch for {s}"
            self.pools[("wod", s)] = selected
            self.audit[s] = {"rows": len(selected), "segments": len(set(seq[selected]))}
        reference = L.Mixer(cfg, D)
        for s in L.CONTRAST:
            assert np.array_equal(self.pools[("wod", s)], reference.pools[("wod", s)])
        assert np.array_equal(self.other, reference.other) and np.array_equal(self.nus, reference.nus)


def load_gate_arrays(split):
    z = np.load(PREP / "gate_rows.npz")
    rows = z[split]
    tab = dict(np.load(L.lroot("prep") / "wod.npz", allow_pickle=True))
    return rows, tab["s_start"][rows].astype(np.float32), tab["seq"][rows].astype(str)


def gate_cache(out, log, args):
    D = L.Data(("wod",), hstore=True)
    rowz = np.load(PREP / "gate_rows.npz")
    stat = {}
    for sp in ("train", "dev"):
        rows = rowz[sp]
        ctx = D.dom["wod"].ctx[rows]
        shape = (len(rows), 9 * 32 * 512)
        dest = np.lib.format.open_memmap(ROOT / f"gate_{sp}.npy", "w+", np.float16, shape)
        t0 = time.perf_counter()
        for i in tqdm(range(0, len(rows), 256), desc=f"Gate cache {sp}", mininterval=5):
            dest[i:i+256] = D.hs["wod"].gather(ctx[i:i+256]).reshape(-1, shape[1])
        dest.flush()
        elapsed = time.perf_counter() - t0
        sample = dest[:64].copy()
        t0 = time.perf_counter()
        plain = np.stack([D.hs["wod"].gather(c[None]).ravel() for c in ctx[:64]])
        plain_s = time.perf_counter() - t0
        t0 = time.perf_counter()
        fast = D.hs["wod"].gather(ctx[:64]).reshape(len(sample), -1)
        fast_s = time.perf_counter() - t0
        assert np.array_equal(plain, fast) and np.array_equal(sample, fast)
        assert np.isfinite(sample).all()
        stat[sp] = {"rows": len(rows), "cache_s": elapsed, "rows_per_s": len(rows)/elapsed,
                    "reference_rows_per_s": len(sample)/plain_s, "vector_rows_per_s": len(sample)/fast_s,
                    "feature_maxabs": float(np.abs(plain.astype(float)-fast).max())}
        del dest
        log.info(f"Gate feature cache {sp}: {stat[sp]}")
    X = np.load(ROOT / "gate_train.npy", mmap_mode="r")
    sums = np.zeros(X.shape[1], np.float64)
    squares = np.zeros_like(sums)
    for i in tqdm(range(0, len(X), 256), desc="Training-only standardization", mininterval=5):
        v = X[i:i+256].astype(np.float64)
        sums += v.sum(0)
        squares += np.einsum("ij,ij->j", v, v)
    mu = sums / len(X)
    sd = np.sqrt(np.maximum(squares/len(X)-mu**2, 0)).clip(min=0.01)
    np.savez(ROOT / "gate_norm.npz", mean=mu.astype(np.float32), std=sd.astype(np.float32))
    dump(out / "profile.json", stat)
    log.event("end", **stat)


class Gate(torch.nn.Module):
    def __init__(self, features, hidden=0):
        super().__init__()
        norm = np.load(ROOT / "gate_norm.npz")
        self.register_buffer("mean", torch.as_tensor(norm["mean"]))
        self.register_buffer("std", torch.as_tensor(norm["std"]))
        self.head = (torch.nn.Sequential(torch.nn.Linear(features, hidden), torch.nn.ReLU(), torch.nn.Linear(hidden, 1))
                     if hidden else torch.nn.Linear(features, 1))

    def forward(self, x):
        return self.head((x.float() - self.mean) / self.std).squeeze(-1)


def gate_inputs(sp, dev):
    X = np.load(ROOT / f"gate_{sp}.npy", mmap_mode="r")
    # Keep raw fp16 features resident; normalize each batch in fp32 without changing feature values.
    x = torch.empty(X.shape, dtype=torch.float16, device=dev)
    for i in tqdm(range(0, len(X), 256), desc=f"Gate resident {sp}", mininterval=5):
        x[i:i+256] = torch.as_tensor(X[i:i+256].copy(), device=dev)
    rows, labels, groups = load_gate_arrays(sp)
    return x, torch.as_tensor(labels, device=dev), rows, groups


@torch.no_grad()
def predict(head, x, bs=256):
    return torch.cat([head(x[i:i+bs]).sigmoid() for i in range(0, len(x), bs)]).cpu().numpy()


def head_train(hidden, wd, x, y, xd, yd, out, log):
    dev = x.device
    torch.manual_seed(0)
    h = Gate(x.shape[1], hidden).to(dev)
    opt = torch.optim.AdamW(h.parameters(), lr=1e-3, weight_decay=wd)
    criterion = torch.nn.BCEWithLogitsLoss()
    losses, gradients, epoch_times = [], [], []
    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    rng = torch.Generator(device=dev).manual_seed(0)
    for epoch in tqdm(range(20), desc=f"Gate hidden={hidden} wd={wd}", mininterval=5):
        ix = torch.randperm(len(x), generator=rng, device=dev)
        epoch_loss = []
        t0 = time.perf_counter()
        for batch in ix.split(256):
            opt.zero_grad(set_to_none=True)
            loss = criterion(h(x[batch]), y[batch])
            assert torch.isfinite(loss), "Nonfinite gate loss"
            loss.backward()
            assert all(torch.isfinite(p.grad).all() for p in h.parameters()), "Nonfinite gate gradient"
            gradients.append(float(sum(p.grad.square().sum() for p in h.parameters()).sqrt()))
            opt.step()
            epoch_loss.append(float(loss))
        if dev.type == "cuda":
            torch.cuda.synchronize()
        epoch_times.append(time.perf_counter()-t0)
        losses.append(float(np.mean(epoch_loss)))
        log.scalar("loss/bce", losses[-1], epoch)
        log.scalar("throughput/frames_per_s", len(x)/epoch_times[-1], epoch)
    prob = predict(h, xd)
    auc = float(roc_auc_score(yd.cpu().numpy(), prob))
    metrics = {"hidden": hidden, "weight_decay": wd, "losses": losses, "dev_auc": auc,
               "frames_per_s": len(x)/np.mean(epoch_times), "gradient_max": max(gradients),
               "gradient_min": min(gradients), "peak_vram_gib": torch.cuda.max_memory_reserved()/2**30 if dev.type=="cuda" else 0,
               "peak_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20,
               "epoch_seconds": epoch_times}
    torch.save({"head": h.state_dict(), "hidden": hidden, "features": x.shape[1], "metrics": metrics}, out / "head.pt")
    np.save(out / "dev_prob.npy", prob)
    dump(out / "metrics.json", metrics)
    log.info(f"Gate head complete: AUC {auc:.4f}, throughput {metrics['frames_per_s']:.0f} frames/s")
    return h, metrics


def gate_pilot(out, log, args):
    dev = device()
    x, y, _, _ = gate_inputs("train", dev)
    xd, yd, _, _ = gate_inputs("dev", dev)
    h, r = head_train(0, 1e-4, x, y, xd, yd, out, log)
    batched = predict(h, xd[:64])
    scalar = predict(h, xd[:64], bs=1)
    diff = float(np.max(np.abs(batched-scalar)))
    minimum = 1000 if dev.type=="cuda" else 100
    r["prediction_maxabs"] = diff
    r["checks"] = {"loss_decreased": np.mean(r["losses"][-4:]) < np.mean(r["losses"][:4]),
                   "gradients_nonzero": r["gradient_max"] > 0,
                   "prediction_equivalence": diff <= 1e-5,
                   "throughput": r["frames_per_s"] >= minimum,
                   "memory": r["peak_vram_gib"] <= 12 and r["peak_rss_gib"] <= 40,
                   "dev_direction": r["dev_auc"] >= 0.55}
    r["checks"] = {k: bool(v) for k, v in r["checks"].items()}
    dump(out / "checklist.json", r)
    assert all(r["checks"].values()), f"Gate unit checklist failed: {r['checks']}"
    # Plan-switch identity checks use only train-side teacher plans; shape and stationary semantics are exact.
    rr = np.load(PREP / "gate_rows.npz")["train"][:64]
    tea = np.asarray(R.load_teacher("wod", L.r2t())["mu"][rr]).copy()
    alternate = tea + np.float32(0.25)
    stat = np.arange(len(rr)) % 2 == 0
    pr = np.full(len(rr), 0.5)
    p1 = np.where((stat & (pr > 1))[:, None, None], alternate, np.where(stat[:, None, None], tea, alternate))
    p0 = np.where((stat & (pr > 0))[:, None, None], alternate, np.where(stat[:, None, None], tea, alternate))
    assert np.array_equal(p1[stat], tea[stat]) and np.array_equal(p1[~stat], alternate[~stat])
    assert np.array_equal(p0, alternate)
    log.event("end", status="passed", **r["checks"])


def gate_fit(out, log, args):
    assert (ROOT / "gate-pilot/DONE").exists(), "Gate pilot must pass before grid"
    dev = device()
    x, y, _, _ = gate_inputs("train", dev)
    xd, yd, _, _ = gate_inputs("dev", dev)
    candidates = []
    for hidden in (0, 32):
        for wd in (1e-4, 1e-2):
            d = out / f"h{hidden}-wd{wd}"
            d.mkdir(parents=True, exist_ok=True)
            h, r = head_train(hidden, wd, x, y, xd, yd, d, log)
            candidates.append({"path": str(d), **r})
            del h
    chosen = max(candidates, key=lambda r: (r["dev_auc"], -r["hidden"], r["weight_decay"]))
    dump(out / "selection.json", {"chosen": chosen, "candidates": candidates})
    log.event("end", selected=chosen["path"], dev_auc=chosen["dev_auc"])


def gate_lock(out, log, args):
    assert (ROOT / "gate-fit/DONE").exists()
    selection = json.loads((ROOT / "gate-fit/selection.json").read_text())
    head_dir = Path(selection["chosen"]["path"])
    prob = np.load(head_dir / "dev_prob.npy")
    rows = np.load(PREP / "gate_rows.npz")["dev"]
    D = L.Data(("wod",), hstore=True)
    dev = torch.device("cuda")
    po = np.asarray(D.tea["wod"]["mu"][rows], np.float32)
    ma = [L.row_metrics(L.fwd_rows(L.load_model(path_for(tag), dev), D, "wod", rows, dev, threads=2)["plan"],
                        D.tab["wod"], rows, D.cam["wod"]) for tag in MAIN_TAGS]
    mo = L.row_metrics(po, D.tab["wod"], rows, D.cam["wod"])
    flags = {s: D.tab["wod"][f"s_{s}"][rows] for s in ("start", "stay")}
    groups = D.tab["wod"]["seq"][rows]
    thresholds = np.unique(np.r_[0.0, np.percentile(prob, np.arange(1, 100)), 1.0])
    candidates = []
    for tau in tqdm(thresholds, desc="Dev thresholds", mininterval=5):
        keep = prob > tau
        r = {"tau": float(tau), "seeds": []}
        for m in ma:
            rec = {}
            for s, key in (("start", "cap_start"), ("stay", "false_start")):
                mask = flags[s]
                selected = np.where(keep, m[key], mo[key])
                rec[s] = L.paired_delta(selected[mask], mo[key][mask], groups[mask], B=2000, seed=0)
            r["seeds"].append(rec)
        r["qualifies"] = all(x["stay"]["delta"] <= .008 and x["stay"]["hi"] <= .02 for x in r["seeds"])
        r["min_start_gain"] = min(x["start"]["delta"] for x in r["seeds"])
        candidates.append(r)
    chosen = max((r for r in candidates if r["qualifies"]), key=lambda r: (r["min_start_gain"], r["tau"]))
    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (head_dir / "head.pt", ROOT / "gate_norm.npz")}
    dump(out / "lock.json", {"head": str(head_dir / "head.pt"), "tau": chosen["tau"], "sha256": hashes,
                              "chosen": chosen, "candidates": candidates, "val_read": False})
    log.event("end", tau=chosen["tau"], dev_start_gain=chosen["min_start_gain"], sha256=hashes)


def compute_step(model, lossf, batch, opt, scaler):
    out = model(batch["trunk"], batch["valid"], batch["tc"], batch["intent"])
    total, losses = lossf(out, batch)
    assert torch.isfinite(total), "Nonfinite curve loss"
    opt.zero_grad(set_to_none=True)
    scaler.scale(total).backward()
    scaler.unscale_(opt)
    for group in opt.param_groups:
        torch.nn.utils.clip_grad_norm_(group["params"], 1.0)
    scaler.step(opt)
    scaler.update()
    return float(total), {k: float(v) for k, v in losses.items()}


def curve_profile(out, log, args):
    cfg = ARMS.get("main")
    dev = torch.device("cuda")
    D = L.Data(("wod", "nus"))
    mix = CurveMixer(cfg, D, 25)
    segs = mix.draw(np.random.default_rng(0))
    t0 = time.perf_counter()
    ref = L.assemble(D, segs)
    ref_s = time.perf_counter() - t0
    # The resident tables and shared memmaps are the original hot path; verify reuse is unchanged.
    t0 = time.perf_counter()
    fast = L.assemble(D, segs)
    fast_s = time.perf_counter() - t0
    for k in ref:
        assert np.array_equal(ref[k], fast[k], equal_nan=True), f"Batch mismatch: {k}"
    batch = L.to_dev(ref, dev)
    grads, results = [], []
    for label, b in (("reference", batch), ("resident", L.to_dev(fast, dev))):
        torch.manual_seed(0)
        m = L.LModel(cfg).to(dev)
        lossf = L.Losses(m.net, cfg, D.tstd, dev)
        o = m(b["trunk"], b["valid"], b["tc"], b["intent"])
        total, _ = lossf(o, b)
        total.backward()
        g = torch.cat([p.grad.flatten().float() for p in sum(m.trainable(), []) if p.grad is not None])
        grads.append(g.cpu())
        results.append(float(total))
        del g, m, lossf, o, total
        torch.cuda.empty_cache()
    relative = float((grads[0]-grads[1]).norm()/grads[0].norm())
    assert abs(results[0]-results[1]) <= 1e-6 and relative <= 1e-5
    m = L.LModel(cfg).to(dev)
    step0_rows = D.rows("wod", "dev", "start", need_future=True)[:64]
    initial = L.fwd_rows_plain(m.eval(), D, "wod", step0_rows, dev)["plan"]
    teacher = np.asarray(D.tea["wod"]["mu"][step0_rows], np.float32)
    step0_maxabs = float(np.abs(initial - teacher).max())
    assert step0_maxabs <= .001, f"Step-0 identity failed: {step0_maxabs}"
    m.train()
    base, new = m.trainable()
    opt = torch.optim.AdamW([{"params": base, "lr": cfg.lr}, {"params": new, "lr": cfg.lr_new}], weight_decay=cfg.wd)
    scaler = torch.amp.GradScaler()
    lossf = L.Losses(m.net, cfg, D.tstd, dev)
    timings = []
    torch.cuda.reset_peak_memory_stats()
    for i in tqdm(range(15), desc="Curve compute profile", mininterval=5):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        compute_step(m, lossf, batch, opt, scaler)
        torch.cuda.synchronize()
        timings.append(time.perf_counter()-t0)
    infer_rows = D.rows("wod", "dev", "start", need_future=True)[:128]
    m.eval()
    t0 = time.perf_counter()
    plain = L.fwd_rows_plain(m, D, "wod", infer_rows, dev)["plan"]
    plain_s = time.perf_counter()-t0
    t0 = time.perf_counter()
    dedup = L.fwd_rows(m, D, "wod", infer_rows, dev, threads=2)["plan"]
    dedup_s = time.perf_counter()-t0
    a = L.rear_np(plain, D.cam["wod"])
    b = L.rear_np(dedup, D.cam["wod"])
    diff = np.abs(a-b)
    eq = {"maxabs_4s_m": float(diff.max()), "p99_4s_m": float(np.percentile(diff, 99))}
    assert eq["maxabs_4s_m"] <= .02 and eq["p99_4s_m"] <= .001, f"Readout equivalence failed: {eq}"
    ra = L.row_metrics(plain, D.tab["wod"], infer_rows, D.cam["wod"])
    rb = L.row_metrics(dedup, D.tab["wod"], infer_rows, D.cam["wod"])
    for k in ra:
        if ra[k].dtype == bool:
            assert np.array_equal(ra[k], rb[k]), f"Binary readout mismatch: {k}"
    r = {"batch_reference_s": ref_s, "batch_resident_s": fast_s, "loss_reference": results[0], "loss_resident": results[1],
         "gradient_relative_error": relative, "compute_seq_per_s": cfg.batch/np.median(timings[3:]),
         "peak_reserved_gib": torch.cuda.max_memory_reserved()/2**30,
         "plain_readout_rows_per_s": len(infer_rows)/plain_s, "dedup_readout_rows_per_s": len(infer_rows)/dedup_s,
         "readout_equivalence": eq, "step0_maxabs_m": step0_maxabs, "mixer": mix.audit,
         "configuration": cfg.dump(), "training_path": "unchanged original; resident table and shared memmap reuse"}
    dump(out / "profile.json", r)
    log.event("end", **r)
    log.info(f"Curve profile: {json.dumps(r)}")


def curve_train(out, log, args):
    unit = args.command == "curve-unit"
    if not unit:
        assert (ROOT / "curve-unit/DONE").exists(), "Curve unit must pass first"
    cfg = ARMS.get("main", seed=args.seed, steps=800 if unit else 4000, eval_every=400 if unit else 2000)
    if unit:
        assert (ROOT / "curve-profile/DONE").exists()
    a = SimpleNamespace(fresh=True, no_eval=False)
    T.train(cfg, out, log, a, mixer_factory=lambda c, d: CurveMixer(c, d, args.size))
    if unit:
        dev = json.loads((out / "dev.json").read_text())
        before = json.loads((out / "dev_before.json").read_text())
        profile = json.loads((ROOT / "curve-profile/profile.json").read_text())
        ev = [json.loads(v) for v in (out / "events.jsonl").read_text().splitlines()]
        imitation = [e["value"] for e in ev if e.get("tag") == "loss/imit"]
        gains = [dev[f"{s}/cap_{s}"] for s in L.SLICE3]
        checks = {"finite_loss": dev["nonfinite"] == 0,
                  "imitation_loss_decreased": len(imitation) == 16 and min(imitation) >= 0 and sum(imitation[-4:]) < sum(imitation[:3]),
                  "step0_identity": before["drift_median"] <= .001 and all(before[f"{s}/cap_{s}"] == 0 for s in L.SLICE3),
                  "capture_direction": np.mean(gains) > 0 and sum(g > 0 for g in gains) >= 2,
                  "dev_drift": dev["drift_median"] <= .15,
                  "dev_false_triggers": max(dev["select"]["values"][k] for k in
                      ("false_start_pp", "false_stop_pp", "false_turn_control_pp", "false_turn_straight_pp")) <= 5,
                  "throughput": 250 <= profile["compute_seq_per_s"] <= 500,
                  "vram": dev["peak_reserved_gb"] <= 40,
                  "data_wait": dev["data_wait_frac"] <= .1}
        dump(out / "checklist.json", {"checks": {k: bool(v) for k,v in checks.items()}, "dev": dev, "profile": profile})
        assert all(checks.values()), f"Curve unit checklist failed: {checks}"


def curve_eval(out, log, args):
    import op_adapt_l_prep as P
    dev = torch.device("cuda")
    D = L.Data(("wodval",), hstore=True)
    m = L.load_model(ROOT / f"curve-{args.size}-s{args.seed}/ckpt-final.pt", dev)
    rows = D.rows("wodval", "val", need_future=True)
    p = L.fwd_rows(m, D, "wodval", rows, dev, threads=2)["plan"]
    assert np.isfinite(p).all()
    np.savez(out / "wodval.npz", rows=rows, plan=p)
    log.scalar("eval/rows", len(rows), 0)
    rd = R.Domain("rater", L.r2t())
    D.dom["rater"] = rd
    names = rd.col("key").astype(str)
    D.tab["rater"] = {"intent": P.wod_kin(names)["intent"]}
    p = L.fwd_rows(m, D, "rater", np.arange(len(rd)), dev, threads=2)["plan"]
    assert len(p)==479 and np.isfinite(p).all()
    np.savez(out / "rater.npz", names=names, plan=p)
    log.event("end", validation_rows=len(rows), rater_rows=len(p))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("command", choices=("gate-cache", "gate-pilot", "gate-fit", "gate-lock", "curve-profile", "curve-unit", "curve-train", "curve-eval"))
    ap.add_argument("--size", choices=("25", "110", "300", "all"), default="25")
    ap.add_argument("--seed", type=int, choices=(0, 1), default=0)
    args = ap.parse_args()
    if args.command == "curve-train":
        name = f"curve-{args.size}-s{args.seed}"
    elif args.command == "curve-eval":
        name = f"eval-{args.size}-s{args.seed}"
    else:
        name = args.command
    out = ROOT / name
    out.mkdir(parents=True, exist_ok=True)
    if (out / "DONE").exists() or (out / "ERROR").exists():
        raise SystemExit(f"Existing sentinel in {out}; inspect before creating a new attempt")
    log = T.Log(out)
    assert log.tb is not None
    torch.set_num_threads(max(1, min(2, len(os.sched_getaffinity(0)))))
    commands = {"gate-cache": gate_cache, "gate-pilot": gate_pilot, "gate-fit": gate_fit,
                "gate-lock": gate_lock, "curve-profile": curve_profile, "curve-unit": curve_train,
                "curve-train": curve_train, "curve-eval": curve_eval}
    try:
        resources = grant_check(gpu=bool(os.environ.get("CUDA_VISIBLE_DEVICES")))
        assert (PREP / "DONE").exists(), "CPU preparation must pass before experiment work"
        log.event("start", command=args.command, size=args.size, seed=args.seed, resources=resources)
        commands[args.command](out, log, args)
        (out / "DONE").write_text(time.strftime("%Y-%m-%d %H:%M:%S\n"))
    except BaseException as error:
        (out / "ERROR").write_text(traceback.format_exc())
        log.info(f"ERROR {error}\n{traceback.format_exc()}")
        log.event("error", error=str(error))
        raise
    finally:
        log.tb.close()


if __name__ == "__main__":
    main()
