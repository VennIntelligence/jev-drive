"""vis_train arm R, "reverse order" (plans/2026-10-10-vis-train-prereg.md amendment 7): the trained branch FROZEN, the policy trained from the
shipped weights with the full SH30 recipe (pp_train.py: P2 + hinge 30 / 0.5, 10 000 steps x 128, cosine, protocol W, the SH30 row stream)
reading the branch tokens through the memory channel (pp_train --mem vtr_<name>: ParityAdapter use_side, n_cam = n_t = 1, memory dropped on
25% of the rows).

  R-0  VTR-0-s<seed>  bank vtr_0        Cinque's own frozen t0 tokens (arm A0's input): the control
  R-A  VTR-A-s<seed>  bank vtr_A-s<seed>  the branch of VT-A-s<seed> at its final checkpoint
  R-B  VTR-B-s<seed>  bank vtr_B-s<seed>  the branch of VT-B-s<seed> at its final checkpoint
Reference rows: SH30-F-s<seed> (no channel). The checkpoints are plain pp_train memory arms, so `jevdrive.bench` serves them on the parity path
(`:noside` masks the bank, `:mshuf` reads the bank row of a token of another log).

  bank   --kind vtr_0 --src cinque | --kind vtr_A-s0 --src A --seed 0   the bank of every data dir (N, 32, 512) fp16 in tab order ->
         runs/op_parity/mem/<kind>/<data>.npy + bank.json (source tag, its step). A source whose final checkpoint is missing is waited for
         while its training is alive, then read at its latest snapshot (the step is recorded).
  check  --tag T                dev ADE of a pp_train memory arm with the memory on / masked / taken from another log (the staged-launch check)
  gate   (repo venv)            waits for the bench reads it needs, then writes $OUT/chain/R/gate.json and the sentinel files the pool jobs
         are gated on: GATE_PASS | GATE_FAIL (arm A's memory is read: VT-A-s<seed> minus VT-A-s<seed>:noside >= 0.2 navtest EPDMS on the
         seed mean) and FALLBACK_A | FALLBACK_B (the arm that is continued: the lower > 45 deg off-road rate against A0 over its last
         three snapshots, seed mean; a tie goes to B). Cancels the pool jobs of the side that does not run (ids from --jobs).
The chain that submits everything: scripts/vt_r_chain.sh.
"""
import sys as _sys, pathlib as _pl  # noqa: E401
_R = _pl.Path(__file__).resolve().parents[3]
_sys.path[:0] = [str(_R), str(_R / "lib"), str(_R / "scripts"), str(_R / "experiments/op_adapt_r2/lib"), str(_R / "experiments/op_parity/scripts")]
import argparse, json, os, subprocess, time  # noqa: E401,E402

import numpy as np  # noqa: E402

from jevdrive.common import data_dir  # noqa: E402

FULL = tuple(f"navtrain_full.s{i}of12" for i in range(12))
DATAS = FULL + ("lb_navtest", "lb_navhard")
RUNS = data_dir() / "runs" / "op_parity" / "runs"
MEM = data_dir() / "runs" / "op_parity" / "mem"
CACHE = data_dir() / "runs" / "op_parity" / "cache"
OUT = data_dir() / "runs" / "vis_train"
CH = OUT / "chain" / "R"
DROP_LINE = 0.2                                  # registered: the masked memory costs >= 0.2 navtest EPDMS (prereg "判定线")
LAST3 = (40, 45, 50)                             # A / B: the last three snapshots (thousand steps; 50 = the final checkpoint)
A0_REG = (10, 20, 30, 40, 50)                    # A0's registered snapshots


def _atomic(f: _pl.Path, write):
    tmp = f.with_name(f".{f.name}.{os.getpid()}")
    write(tmp)
    os.replace(tmp, f)


# ---------------------------------------------------------------- bank
def source(a) -> tuple:
    """-> (tag, step) of the checkpoint the bank is computed from; ('cinque', 0) for the frozen t0 tokens."""
    import torch
    if a.src == "cinque":
        return "cinque", 0
    tag = f"VT-{a.src}-s{a.seed}"
    fin, t0 = RUNS / tag / "ckpt-final.pt", time.time()
    while not fin.exists():                      # alive: no stop-rule marker, no chain error, the wait budget not used up
        if (RUNS / tag / "STOP").exists() or (OUT / "chain" / f"{a.src}-s{a.seed}" / "ERROR").exists() or time.time() - t0 > 60 * a.wait_min:
            break
        time.sleep(60)
    if not fin.exists():
        snaps = sorted(RUNS.glob(f"{tag}-k[0-9]*/ckpt-final.pt"))
        if not snaps:
            raise SystemExit(f"{tag}: neither a final checkpoint nor a snapshot")
        tag = snaps[-1].parent.name
    return tag, int(torch.load(RUNS / tag / "ckpt-final.pt", map_location="cpu", weights_only=False)["step"])


def cmd_bank(a):
    import torch
    from jevdrive.run import Run
    assert a.kind.startswith("vtr_"), "bank kinds of this lane are vtr_<name>"
    tag, step = source(a)
    od, mf = MEM / a.kind, MEM / a.kind / "bank.json"
    meta = {"kind": a.kind, "src": tag, "step": step}
    if od.exists():                              # a kind that exists must be this bank, interrupted or finished
        old = json.loads(mf.read_text()) if mf.exists() else None
        if old is None or {k: old.get(k) for k in meta} != meta:
            raise SystemExit(f"{od} exists and is not the bank of {meta}: {old}")
    od.mkdir(parents=True, exist_ok=True)
    meta = (json.loads(mf.read_text()) if mf.exists() else meta | {"rows": {}})
    with Run("vis_train", f"bank-{a.kind}", config=vars(a) | {"src_tag": tag, "src_step": step}) as run:
        dev = torch.device("cuda" if tag != "cinque" else "cpu")
        if tag != "cinque":
            import vt as V
            m, _ = V.load_tag(tag, dev)
            assert m.k["vis"] == "branch" and m.nv == 1, f"{tag} (arm {m.arm}): not a one-view branch arm"
        for data in a.datas:
            f = od / f"{data}.npy"
            names = np.load(CACHE / data / "tab.npz")["names"]
            if f.exists() and meta["rows"].get(data) == len(names):
                continue
            front = np.load(CACHE / f"{data}@warp" / "front.npy", mmap_mode="r")      # (N, 8, 32, 512): Cinque's frozen slot tokens, protocol W
            assert len(front) == len(names)
            tmp = od / f".{data}.{os.getpid()}.npy"
            o = np.lib.format.open_memmap(tmp, "w+", np.float16, (len(names), 32, 512))
            if tag == "cinque":
                for i in run.tqdm(range(0, len(names), 4096), desc=data):
                    o[i:i + 4096] = front[i:i + 4096, -1]
            else:
                I = V.Infer(m, data, dev, host=True)
                assert (I.S.tab["names"] == names).all()
                for i in run.tqdm(range(0, I.S.n, 256), desc=data):
                    rn = np.arange(i, min(i + 256, I.S.n))
                    o[rn] = I.mem(rn).half().cpu().numpy()
                del I
            assert np.isfinite(o[:4096].astype(np.float32)).all()
            if data == "lb_navtest":             # how far the branch tokens are from the frozen ones (0 for vtr_0)
                d = o[:2048].astype(np.float32) - np.asarray(front[:2048, -1], np.float32)
                meta["navtest_vs_cinque"] = dict(mean_abs=float(np.abs(d).mean()), rms=float(np.sqrt((d ** 2).mean())),
                                                 ref_rms=float(np.sqrt((np.asarray(front[:2048, -1], np.float32) ** 2).mean())))
            o.flush()
            del o
            os.replace(tmp, f)
            meta["rows"][data] = len(names)
            _atomic(mf, lambda t: t.write_text(json.dumps(meta, indent=1)))
            run.info(f"{a.kind}: {data} {len(names)} rows from {tag} (step {step}) -> {f}")
        run.summary.update(meta | {"out": str(od)})


# ---------------------------------------------------------------- check
def cmd_check(a):
    """Dev rows of the run's own split: ADE to the log with the memory on / masked / of a row of another log."""
    import torch
    import pp_train as T
    from jevdrive.run import Run
    dev = torch.device("cuda")
    with Run("vis_train", f"rcheck-{a.tag}", config=vars(a)) as run:
        cfg = torch.load(RUNS / a.tag / "ckpt-final.pt", map_location="cpu", weights_only=False)["cfg"]
        model = T.load_pmodel(a.tag, dev)
        assert model.mem, f"{a.tag} (arm {model.arm}) is not a memory arm"
        S = T.Store(list(cfg["data"]), dev, need_side=False, frames=cfg["frames"], host=True, mem=model.mem)
        _, dv, sp = T.split_rows({"names": S.tab["names"], "log": S.tab["log"]}, cfg["split"])
        for x in sp:
            run.use_split(x)
        W = torch.as_tensor(T.R2.t_weights(T.T8), device=dev)
        pi = torch.as_tensor(S.pi, device=dev)
        lg, rng = S.tab["log"][dv], np.random.default_rng(0)
        other = np.array([rng.choice(dv[lg != x]) for x in lg])                  # a dev row of another log for every dev row
        acc = {"on": [], "masked": [], "shuf": []}
        with torch.no_grad():
            for i in range(0, len(dv), 128):
                r = torch.as_tensor(dv[i:i + 128], device=dev)
                ok = S.has_fut[r]
                for name in acc:
                    mem = S.mem[other[i:i + 128] if name == "shuf" else dv[i:i + 128]]
                    mask = torch.zeros(len(r), 1, dtype=torch.bool, device=dev) if name == "masked" else None
                    p = model(S.front[r], S.ego[r], S.tc[r], mem, mask).float()[:, pi].view(-1, 33, 15)
                    x, y, _ = T.rear(p, S.cam_x[r], W)
                    acc[name].append(torch.hypot(x - S.fut[r][..., 0], y - S.fut[r][..., 1]).mean(1)[ok])
        res = {"tag": a.tag, "arm": model.arm, "dev_rows": int(len(dv))} | {f"ade_{k}": float(torch.cat(v).mean()) for k, v in acc.items()}
        (OUT / "R").mkdir(parents=True, exist_ok=True)
        (OUT / "R" / f"check-{a.tag}.json").write_text(json.dumps(res, indent=1))
        run.info(json.dumps(res))
        run.summary.update(res)
        assert all(np.isfinite(v) for k, v in res.items() if k.startswith("ade_"))


# ---------------------------------------------------------------- gate
def units(spec):
    """The stored navtest units of a model (by token), or None while its read is not finished."""
    import pandas as pd
    from jevdrive import bench
    d = bench.run_dir(spec, "navtest")
    return pd.read_csv(d / "units.csv").set_index("token") if (d / "DONE").exists() and (d / "units.csv").exists() else None


def a0_step(k: int) -> int:
    return min(A0_REG, key=lambda s: (abs(s - k), s))                            # nearest registered snapshot, ties to the earlier (vt_read.ctrl_step)


def tag_at(arm: str, seed: int, k: int) -> str:
    return f"VT-{arm}-s{seed}" + ("" if k == 50 else f"-k{k:02d}")


def cmd_gate(a):
    from jevdrive import stats
    from jevdrive.bench import tables as BT
    CH.mkdir(parents=True, exist_ok=True)
    gate = [f"VT-A-s{s}{o}" for s in (0, 1) for o in ("", ":noside")]
    fall = sorted({tag_at(x, s, k) for x in "AB" for s in (0, 1) for k in LAST3} | {tag_at("A0", s, a0_step(k)) for s in (0, 1) for k in LAST3})
    jobs = lambda: [ln.split() for ln in _pl.Path(a.jobs).read_text().splitlines() if len(ln.split()) == 2] if a.jobs else []  # noqa: E731
    res = {"drop_line": DROP_LINE, "cancelled": []}

    def cancel(ids):
        for j in ids:
            subprocess.run([_sys.executable, "-m", "jevdrive.cl", "cancel", j], cwd=_R, capture_output=True)
        res["cancelled"] += ids

    def save():
        _atomic(CH / "gate.json", lambda t: t.write_text(json.dumps(res | {"time": time.strftime("%F %T %Z")}, indent=1)))

    t0 = time.time()
    while True:                                  # every read is a pool job of another chain: wait, never submit; each verdict is written when its reads are in
        U = {sp: units(sp) for sp in dict.fromkeys(gate + fall)}
        miss = [sp for sp, u in U.items() if u is None]
        (CH / "GATE_STATUS").write_text(f"{time.strftime('%F %T')} waiting for {len(miss)} navtest reads: {' '.join(miss)}\n")
        if "gate" not in res and all(U[sp] is not None for sp in gate):
            d, per = [], {}
            for s in (0, 1):
                on, off = U[f"VT-A-s{s}"], U[f"VT-A-s{s}:noside"].reindex(U[f"VT-A-s{s}"].index)
                assert off.score.notna().all() and (on.index == U["VT-A-s0"].index).all()
                per[s] = 100.0 * float(on.score.mean() - off.score.mean())
                d.append(100.0 * (on.score.to_numpy(float) - off.score.to_numpy(float)))
            r = stats.paired(np.mean(d, 0), np.zeros(len(d[0])), U["VT-A-s0"]["log"].astype(str).to_numpy())    # CI: information only, the line is on the mean
            drop = float(np.mean(list(per.values())))
            res["gate"] = dict(quantity="navtest EPDMS, VT-A-s<seed> minus VT-A-s<seed>:noside, seed mean", drop=drop, per_seed=per, ci=[r["lo"], r["hi"]],
                               passed=bool(drop >= DROP_LINE))
            if drop < DROP_LINE:
                cancel([j for j, n in jobs() if n.startswith("vtr-")])
            save()
            (CH / ("GATE_PASS" if drop >= DROP_LINE else "GATE_FAIL")).write_text(json.dumps(res["gate"]) + "\n")
        if "fallback" not in res and all(U[sp] is not None for sp in fall):
            turn = BT.navtest_strata().turn
            rate = lambda sp: 100.0 * float((U[sp].DAC[turn.reindex(U[sp].index).to_numpy() == ">45"] < 1).mean())  # noqa: E731
            diff = {x: {k: [rate(tag_at(x, s, k)) - rate(tag_at("A0", s, a0_step(k))) for s in (0, 1)] for k in LAST3} for x in "AB"}
            val = {x: round(float(np.mean([v for k in LAST3 for v in diff[x][k]])), 2) for x in "AB"}
            pick = "A" if val["A"] < val["B"] else "B"
            res["fallback"] = dict(quantity="> 45 deg off-road rate (pp), arm minus A0 (nearest registered snapshot), mean over k40 / k45 / final and both seeds, rounded to 0.01",
                                   value=val, per_snapshot_seed={x: {f"k{k}": v for k, v in diff[x].items()} for x in "AB"}, pick=pick)
            cancel([j for j, n in jobs() if f"-{'B' if pick == 'A' else 'A'}2-" in n])
            save()
            (CH / f"FALLBACK_{pick}").write_text(json.dumps(res["fallback"]) + "\n")
        if not miss or time.time() - t0 > 3600 * a.timeout_h:
            break
        time.sleep(120)
    res["missing"] = miss
    res.setdefault("gate", dict(passed=None, note="the reads of VT-A-s{0,1} and :noside did not all arrive: R is not released"))
    res.setdefault("fallback", dict(pick=None, note="the last-three-snapshot reads of A / B / A0 did not all arrive: no continuation is released"))
    save()
    print(json.dumps(res, indent=1), flush=True)
    (CH / "GATE_STATUS").write_text(f"{time.strftime('%F %T')} gate {res['gate'].get('passed')}, continuation {res['fallback'].get('pick')}, missing reads {miss}\n")
    if miss:
        raise SystemExit(f"timed out after {a.timeout_h} h waiting for {miss}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("bank")
    p.add_argument("--kind", required=True)
    p.add_argument("--src", required=True, choices=["cinque", "A", "B"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--datas", nargs="+", default=list(DATAS))
    p.add_argument("--wait-min", type=float, default=120, help="how long a missing final checkpoint is waited for before the latest snapshot is used")
    p = sp.add_parser("check")
    p.add_argument("--tag", required=True)
    p = sp.add_parser("gate")
    p.add_argument("--jobs", default="", help="the chain's jobs.txt (`<id> <name>` lines): the jobs of the side that does not run are cancelled")
    p.add_argument("--timeout-h", type=float, default=10)
    a = ap.parse_args()
    {"bank": cmd_bank, "check": cmd_check, "gate": cmd_gate}[a.cmd](a)
