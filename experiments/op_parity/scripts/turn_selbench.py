"""op_parity turn selector bench (plans/2026-10-08-turn-selector-bench-prereg.md): the decision-191 selector (arm N7, F19 x pc) as a bench model.

  refit    (op-train, GPU) re-fit N7 F19 x pc on the 28 323 navtrain turn tokens exactly as turn_selnt.py fit did (config = the one decision 191 selected,
           5 initialisations, seeds 200..204) and save the weights -> $DATA_DIR/runs/op_parity/turn_selbench/n7_f19_bundle.pt   (191 did not keep weights)
  repro    (op-train, GPU) G-repro: the bundle on 191's stored navtest turn-token features, gain read from the candidate score table (must be +2.75 +- 0.30)
  select   called by the bench stage `ts-select` (jevdrive.bench.navsim.select_stage): features of a bench from the base SH30 model, 19 candidates around its
           exported poses, own-edge margins, N7 picks, gate A / B / 0 -> the model's pred file (+ turn_selbench/select/<spec>__<bench>.npz: picks, gate, margins, predicted gains)
  report   (jev venv or op-train) pooled official readout of the bench runs: gate coverage, strata, decomposition, sub-scores, navhard -> results/turn_selector_bench/

Gates: A = every token, B = tokens whose own exported 4 s heading change is >= 20 deg (primary), 0 = forced identity (identity gate).
"""
import os
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_k, "1")
import sys as _sys, pathlib as _pl  # noqa: E401,E402
_R = _pl.Path(__file__).resolve().parents[3]
_sys.path[:0] = [str(_R), str(_R / "research"), str(_pl.Path(__file__).parent)]
import argparse, json, pickle, re, time  # noqa: E401,E402

import numpy as np  # noqa: E402

import turn_ceiling as TC  # noqa: E402
import turn_dewater as TD  # noqa: E402
import turn_selinput as TS  # noqa: E402
import turn_selnt as TN  # noqa: E402

D = TC.D
OUTB = D / "runs/op_parity/turn_selbench"
BUNDLE = OUTB / "n7_f19_bundle.pt"
FK0 = TN.FK[0]                                           # "F19 x pc"
NINIT, SEED0 = 5, 200
TURN_DEG = 20.0                                          # gate B: the model-side analogue of the |dyaw| >= 20 deg bucket
SELD = OUTB / "select"
RES = _R / "experiments/op_parity/results/turn_selector_bench"
FIG = _R / "experiments/op_parity/figs/turn_selector_bench"


# ---------------------------------------------------------------- refit
def cmd_refit(a):
    from jevdrive.data import splits
    from jevdrive.run import Run
    import torch
    with Run("op_parity", "turn_selbench/refit", seed=0, config=vars(a)) as run:
        run.use_split(splits.load("navsim/navtrain"))
        if BUNDLE.exists() and not a.force:
            run.info("exists: %s", BUNDLE)
            return
        with open(TN.OUT / "fit" / "full" / "N7.pkl", "rb") as fh:
            old = pickle.load(fh)
        cfg = tuple(old[FK0]["cfg"])
        run.info("N7 %s cfg selected by decision 191 on navtrain OOF: %s", FK0, cfg)
        tr = TN.load_train("full")
        allx = np.arange(tr["n"])
        t0 = time.time()
        models = []
        for s in range(NINIT):
            p = TN.fit_nn(tr, "N7", FK0, allx, SEED0 + s, cfg)
            models.append(p.bundle)
            run.info("init %d (seed %d): val gain %.4f, elapsed %.0f s", s, SEED0 + s, p.info["val_gain"], time.time() - t0)
        BUNDLE.parent.mkdir(parents=True, exist_ok=True)
        tmp = BUNDLE.with_suffix(".tmp")
        torch.save(dict(models=models, fk=FK0, cfg=cfg, n_train=tr["n"], seeds=list(range(SEED0, SEED0 + NINIT))), tmp)
        tmp.rename(BUNDLE)
        run.summary.update(cfg=str(cfg), fit_wall_s=time.time() - t0, n_train=tr["n"])


def infer(bundle, e, h, c, V3, dev=None, bs=2048):
    """Mean over the bundle's initialisations of the predicted gains of candidates 1..K-1, with 0 for the identity: (n, K).
    e (n, 35), h (n, 1024), c (n, 76) arrays as turn_selnt builds them, V3 (n, 32, 512) fp16 vision tokens."""
    import torch
    dev = dev or torch.device("cuda")
    ins = dict(e=e, h=h, c=c)
    Vg = torch.as_tensor(np.ascontiguousarray(V3)).to(dev)
    outs = []
    for b in bundle["models"]:
        w, p_drop, _ = b["cfg"]
        net = TN.make_head(b["spec"], b["dims"], w, p_drop, b["K"]).to(dev)
        net.load_state_dict({k: v.to(dev) for k, v in b["state"].items()})
        net.eval()
        Z = {k: torch.as_tensor(np.ascontiguousarray((v - b["mu"][k][0]) / b["mu"][k][1]), dtype=torch.float32, device=dev) for k, v in ins.items()}
        o = []
        with torch.no_grad():
            for i in range(0, len(e), bs):
                o.append(net(Vg[i:i + bs], Z["e"][i:i + bs], Z["h"][i:i + bs], Z["c"][i:i + bs]).cpu().numpy())
        outs.append(np.concatenate(o))
        del net
    p = np.mean(outs, 0)
    return np.concatenate([np.zeros((len(e), 1)), p], -1), np.stack(outs)


def load_bundle():
    import torch
    return torch.load(BUNDLE, map_location="cpu", weights_only=False)


# ---------------------------------------------------------------- G-repro
def cmd_repro(a):
    """The bundle on decision 191's stored test features (turn_selnt.load_test: HID, margins.npz, vision tokens), valued from the candidate score table."""
    from jevdrive import stats
    from jevdrive.data import splits
    from jevdrive.run import Run
    with Run("op_parity", "turn_selbench/repro", seed=0, config=vars(a)) as run:
        run.use_split(splits.load("navsim/navtest"))
        te = TN.load_test()
        bundle = load_bundle()
        with open(TN.OUT / "fit" / "full" / "N7.pkl", "rb") as fh:
            old = pickle.load(fh)[FK0]["test_pred"]
        C, tok, dyaw, log, X = TD.load()
        assert np.array_equal(tok, te["tokens"])
        F = TD.fams(C)
        Sc = TC.noec(TD.conv(X, C, "pc"))
        Y = (Sc[:, F["F19"]] - Sc[:, :1]).transpose(0, 2, 1)
        pred = np.zeros((te["S"], te["n"], 19))
        for s in range(te["S"]):
            e = np.concatenate([te["ego"][s], te["plan"][s]], -1)
            c = TN.sel(te, FK0, "C")[s].reshape(te["n"], -1)
            pred[s], _ = infer(bundle, e, te["Hraw"][s], c, te["V3"])
        pk, pk_old = TD.picks_of(pred), TD.picks_of(old)
        d = TD.take(Y, pk).mean(0)
        g = stats.paired(100 * d, np.zeros(len(d)), groups=log)
        d_old = TD.take(Y, pk_old).mean(0)
        g_old = stats.paired(100 * d_old, np.zeros(len(d_old)), groups=log)
        out = dict(gain=g["mean"], lo=g["lo"], hi=g["hi"], stored_gain=g_old["mean"], pick_agreement=float((pk == pk_old).mean()),
                   moved_new=float((pk != 0).mean()), moved_stored=float((pk_old != 0).mean()), window=[2.45, 3.05])
        out["ok"] = bool(2.45 <= g["mean"] <= 3.05 and g["lo"] > 0)
        OUTB.mkdir(parents=True, exist_ok=True)
        (OUTB / "gate_repro.json").write_text(json.dumps(out, indent=1))
        run.info("G-repro %s", out)
        run.summary.update(out)
        assert out["ok"], f"G-repro failed: {out}"


# ---------------------------------------------------------------- select stage
def select_file(m, bench):
    return SELD / f"{m.spec.replace(':', '_')}__{bench}.npz"


def seed_of(name):
    m = re.fullmatch(r"SH30-F-s(\d+)", name)
    return int(m.group(1))


def model_yaw_deg(P):
    """|heading at 4 s| of exported (n, 8, 3) poses in the t0 ego frame, degrees (the model-side version of the logged 4 s heading change)."""
    return np.abs(np.degrees(np.angle(np.exp(1j * np.asarray(P, np.float64)[:, -1, 2]))))


def curb_margins(P, ex, ey, cal):
    """Own-edge margins (n, 19, 4) of the F19 candidates around the exported poses P: turn_selinput._m_curb, clipped as in 191."""
    cands = TC.candidates()[:TN.NCAND]
    TS._D["ex"], TS._D["ey"], TS._D["cal"] = {0: ex}, {0: ey}, {0: cal}
    TS._D["poses"] = {}
    Pc, C = [], np.zeros((len(P), TN.NCAND, 4))
    for c, (_, o, k, v) in enumerate(cands):
        TS._D["poses"][TC.key(0, c)] = TC.transform(P, o, k, v)
        C[:, c] = TS._m_curb((0, c))[1]
    return np.clip(C, *TS.CLIP).astype(np.float32).astype(np.float64)


def features(m, bench, run=None):
    """Forward of the base SH30 model on every row of the bench: plan, hidden state, road edges; plus tab streams and vision tokens."""
    import torch
    from jevdrive.bench import navsim as N
    from jevdrive.bench.sets import NAVSIM
    import tsn_extract as TX
    N._pp_path()
    import pp_train as T
    data = NAVSIM[bench]["data"]
    dev = torch.device("cuda")
    S = T.Store([data], dev, need_side=(N.cache_dir(data, "gimm") / "side.npy").exists(), frames=m.frames)
    model = N._load_ckpt(T, m.ckpt, dev)
    assert getattr(model, "mem", None) is None and not m.opt, "plain base model only"
    names = S.tab["names"]
    P, h4, hm, re_ = TX.run_rows(model, S, np.arange(S.n), run)
    tab = np.load(N.cache_dir(data, "gimm") / "tab.npz")
    assert tab["names"].tolist() == names.tolist()
    front = np.load(N.cache_dir(data, m.frames) / "front.npy", mmap_mode="r")
    V3 = np.ascontiguousarray(np.asarray(front[:, -1]))
    del model, S
    torch.cuda.empty_cache()
    return dict(names=names, plan_pos=P, h4=h4, hm=hm, re=re_, ego=tab["ego"].astype(np.float64), cam=tab["cam"], V3=V3)


def select(spec, bench, run_dir):
    """Stage body: pred file of `SH30-F-s<k>@<frames>:ts{A,B,0}` from the base model's exported poses."""
    import shutil
    from dataclasses import replace
    from jevdrive.bench import navsim as N
    from jevdrive.bench import runner as R
    from jevdrive.bench.models import resolve
    from jevdrive.run import Run
    import sc_analyze as SC
    m = resolve(spec, check=True)
    base = replace(m, opt="")
    gate = m.opt[2:]                                     # A | B | 0
    s = seed_of(m.name)
    run_dir = _pl.Path(run_dir)
    out = N.pred_file(m, bench)
    with Run("op_parity", f"turn_selbench/select-{m.spec}-{bench}", seed=0, config=dict(spec=spec, bench=bench)) as run:
        R.status(run_dir, f"select: base features of {base.spec} on {bench}")
        z = np.load(N.pred_file(base, bench))
        f = features(base, bench, run)
        row = {t: i for i, t in enumerate(z["tokens"].tolist())}
        P = z["poses"][[row[t] for t in f["names"].tolist()]]
        n = len(P)
        cal = json.loads(TS.CAL.read_text())[f"SH30-F-s{s}"]
        ex, ey = SC.edges_ego(f["re"], f["cam"])
        C = curb_margins(P, ex, ey, (cal["s"], cal["b"]))
        e = np.concatenate([f["ego"], TD.plan_desc(P.astype(np.float64))], -1)
        h = np.concatenate([f["h4"], f["hm"]], 1).astype(np.float32)
        pred, each = infer(load_bundle(), e, h, C.reshape(n, -1), f["V3"])
        dyaw = model_yaw_deg(P)
        allowed = {"A": np.ones(n, bool), "B": dyaw >= TURN_DEG, "0": np.zeros(n, bool)}[gate]
        pk = np.where(allowed, TD.picks_of(pred), 0)
        cands = TC.candidates()
        poses = P.copy()
        for c in np.unique(pk[pk > 0]):
            i = pk == c
            poses[i] = TC.transform(P[i], *cands[c][1:])
        if gate == "0":
            assert np.array_equal(poses, P)
        arrs = {k: z[k] for k in z.files}
        arrs["tokens"], arrs["poses"] = f["names"].astype(arrs["tokens"].dtype), poses.astype(z["poses"].dtype)
        for k in z.files:                                # any per-token array of the base file follows the new token order
            if k not in ("tokens", "poses") and getattr(z[k], "shape", ())[:1] == (len(z["tokens"]),):
                arrs[k] = z[k][[row[t] for t in f["names"].tolist()]]
        SELD.mkdir(parents=True, exist_ok=True)
        np.savez(select_file(m, bench), tokens=f["names"], picks=pk, gate=allowed, dyaw_model=dyaw, pred=pred.astype(np.float32), C=C.astype(np.float32),
                 pred_each=each.astype(np.float32))
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_name(f".{out.stem}.{os.getpid()}.npz")
        np.savez(tmp, **arrs)
        os.replace(tmp, out)
        info = dict(spec=m.spec, bench=bench, n=n, gate_share=float(allowed.mean()), moved_share=float((pk != 0).mean()),
                    moved_of_gated=float((pk[allowed] != 0).mean()) if allowed.any() else 0.0)
        run.summary.update(info)
        run.info("select %s", info)
        R.status(run_dir, f"select done: gate {gate} passes {100 * info['gate_share']:.1f}% of tokens, moved {100 * info['moved_share']:.1f}%")


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("refit")
    p.add_argument("--force", action="store_true")
    sp.add_parser("repro")
    p = sp.add_parser("report")
    p.add_argument("--out", default=str(RES))
    p.add_argument("--figs", default=str(FIG))
    a = ap.parse_args()
    OUTB.mkdir(parents=True, exist_ok=True)
    {"refit": cmd_refit, "repro": cmd_repro, "report": lambda a: __import__("turn_selbench_report").report(a)}[a.cmd](a)


if __name__ == "__main__":
    main()
