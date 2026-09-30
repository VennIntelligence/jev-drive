"""Self-tests of the op-adapt r2 trainer / readout code (op-train venv) on synthetic data shaped like packages C / S / D
deliver them (tmp/2026-09-30-op-adapt-r2-build.md). Nothing here reads a registered readout set.

  units   CPU: sampler counts per arm, gates, losses, lambda_s rule, pack / gather round trip
  synth   GPU: a synthetic R2 root (C-format index + caches, S-format score tables, a stub D adapter), teacher, every
          arm for a few steps, checkpoint resume, dev eval, the stage checklist code
  OP_R2_ROOT=/tmp/r2test CUDA_VISIBLE_DEVICES=1 python scripts/op_adapt_r2_selftest.py synth
"""
import argparse, json, os, sys, tempfile, types
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

OK = []


def check(name, cond, info=""):
    OK.append((name, bool(cond)))
    print(f"[{'PASS' if cond else 'FAIL'}] {name} {info}")


# ---------------------------------------------------------------- units
def units():
    import torch
    from jevdrive import op_adapt_r2 as R
    check("split_counts", R.split_counts(26, (0.6, 0.4)) == [16, 10] and R.split_counts(12, (4, 3, 3)) == [5, 4, 3])
    # gates: (px, lat, top_eq) -> (g_dir, g_eq)
    px = np.array([600, 600, 300, 300, 50, 600, 600, 600])
    lat = np.array([1.0, 5.0, 1.0, 5.0, 1.0, 3.5, 5.0, np.nan])
    teq = np.array([1, 1, 1, 0, 0, 1, 0, 1], bool)
    gd, ge = R.dir_gates(px, lat, teq)
    check("dir_gates g_dir", list(gd) == [1, 0, 0, 0, 0, 0, 0, 0], gd)
    check("dir_gates g_eq (small-and-near empty, between-bands empty, top differs -> none)", list(ge) == [0, 1, 0, 0, 1, 0, 0, 0], ge)
    # t_weights exact on grid points and linear in between
    W = R.t_weights([A_T := float(R.A.T_IDXS[10]), 2.0])
    check("t_weights", abs(W[0] @ R.A.T_IDXS - A_T) < 1e-6 and abs(W[1] @ R.A.T_IDXS - 2.0) < 1e-5)
    # L_score: winner-take-all over Top, zero at a Top candidate, ignores non-Top
    mu = torch.zeros(2, 21, 4)
    cand = torch.stack([torch.zeros(2, 21, 4), torch.ones(2, 21, 4) * 3, -torch.ones(2, 21, 4)], 1)
    sig = torch.ones(2, 21, 4)
    top = torch.tensor([[True, True, False], [False, True, True]])
    l = R.loss_score(mu, cand, top, sig)
    exp = (0 + min(4 * (3 - 0.5), 4 * 0.5)) / 2
    check("loss_score WTA", abs(float(l) - exp) < 1e-5, (float(l), exp))
    # sigma normalisation
    check("huber_d sigma", abs(float(R.huber_d(mu[:1], cand[:1, 2:3], sig[:1] * 2)[0, 0]) - 4 * 0.125) < 1e-6)
    # L_dir: relu direction term only when v+ > v-, eq term uses stop-gradient on x-
    vp, vm = torch.tensor([[2., 2, 2, 2], [0, 0, 0, 0]]), torch.tensor([[1., 1, 1, 1], [1, 1, 1, 1]])
    mp = torch.zeros(2, 21, 4, requires_grad=True)
    mm = torch.ones(2, 21, 4, requires_grad=True)
    ld = R.loss_dir(vp, vm, torch.ones(2, 4), torch.tensor([True, True]), mp, mm, sig, torch.tensor([False, False]))
    check("loss_dir direction", abs(float(ld) - 0.5) < 1e-6, float(ld))
    ld2 = R.loss_dir(vp, vm, torch.ones(2, 4), torch.tensor([False, False]), mp, mm, sig, torch.tensor([True, False]))
    ld2.backward()
    check("loss_dir eq + stop-gradient", abs(float(ld2) - 4 * 0.5) < 1e-6 and mm.grad is None and mp.grad.abs().sum() > 0)
    check("loss_pair", abs(float(R.loss_pair(torch.tensor([2.0]), torch.tensor([1.0]))) - np.log(2)) < 1e-6)
    # lambda_s rule
    s = R.select_lambda({0.3: {"drift_median": 0.05, "null_slow_delta_pp": 1, "sjev_delta_simC": 0.01, "sjev_delta_simK": 0.01},
                         1.0: {"drift_median": 0.08, "null_slow_delta_pp": 1.5, "sjev_delta_simC": 0.03, "sjev_delta_simK": 0.02}})
    check("select_lambda best gain", s["lam_s"] == 1.0)
    s = R.select_lambda({0.3: {"drift_median": 0.05, "null_slow_delta_pp": 1, "sjev_delta_simC": 0.01},
                         1.0: {"drift_median": 0.12, "null_slow_delta_pp": 1, "sjev_delta_simC": 0.05}})
    check("select_lambda drift gate", s["lam_s"] == 0.3)
    s = R.select_lambda({0.3: {"drift_median": 0.2, "null_slow_delta_pp": 1}, 1.0: {"drift_median": 0.15, "null_slow_delta_pp": 5}})
    check("select_lambda fallback smaller drift", s["lam_s"] == 1.0)
    # batch composition per arm at batch 64
    exp = {"A": (8, 19, 13, 0), "A-real": (0, 38, 26, 0), "A-sim": (16, 0, 0, 0), "A-noC": (16, 19, 13, 0),     # v5: no offset share
           "A-bhv": (8, 19, 13, 0)}
    for arm, (sl, wo, nu, of) in exp.items():
        a = R.ARMS[arm]
        B = 64
        n_sim = (int(round(B * .5)) if a.real else B) if a.sim else 0
        n_slot = n_sim // (2 * len(a.sim)) if a.sim else 0
        n_real = B - n_sim if a.real else 0
        n_off = int(round(.1 * B)) if a.offset and n_real else 0
        w, n = R.split_counts(n_real - n_off, (.6, .4)) if n_real else (0, 0)
        check(f"composition {arm}", (n_slot, w, n, n_off) == (sl, wo, nu, of), (n_slot, w, n, n_off))
    check("score domains", R.ARMS["A-real"].score_domains == {"nus"} and R.ARMS["A-sim"].score_domains == {"simC", "simK"}
          and R.ARMS["A-bhv"].score_domains == set() and R.ARMS["A-noC"].score_domains == {"simK", "nus"})
    # pack / gather round trip on tiny caches (npz and npy)
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        rng = np.random.default_rng(0)
        T1 = rng.standard_normal((7, 1024, 8, 16)).astype(np.float16)
        T2 = rng.standard_normal((5, 1024, 8, 16)).astype(np.float16)
        np.savez(td / "a.npz", trunk=T1)
        np.save(td / "b.npy", T2)
        ix = pd.DataFrame({"cache": [str(td / "a.npz"), str(td / "a.npz"), str(td / "b.npy")], "row": [6, 3, 4],
                           "lctx": [np.array([-1] * 5 + [0, 2, 4, 6]), np.array([-1] * 8 + [3]), np.arange(-4, 5)],
                           "tc0": 1.0, "tc1": 0.0, "split": "train"})
        for nm, cp in (("x", False), ("y", True)):
            R.pack(nm, ix, td, copy=cp)
            d = R.Domain(nm, td)
            x, v = d.gather(np.array([0, 1, 2]))
            good = (np.array_equal(x[0, 5:], T1[[0, 2, 4, 6]]) and not x[0, :5].any() and np.array_equal(x[1, 8], T1[3])
                    and np.array_equal(x[2, 4:], T2[:5]) and v.sum() == 4 + 1 + 5)
            check(f"pack/gather round trip ({'flat copy' if cp else 'mapped caches'})", good)


# ---------------------------------------------------------------- synthetic R2 root
def synth_root(root: Path, seed=0):
    """C-format index + caches, as tmp/2026-09-30-op-adapt-r2-build.md §C."""
    rng = np.random.default_rng(seed)
    (root / "index").mkdir(parents=True, exist_ok=True)
    tr = lambda n: rng.standard_normal((n, 1024, 8, 16)).astype(np.float16)  # noqa: E731
    # sim: 12 pairs (8 train, 2 dev, 2 test), 4 streams x 24 slots per pair file
    (root / "cache-sim").mkdir(exist_ok=True)
    rows = {"C": [], "K": []}
    uid = {"C": 1_000_000_000, "K": 2_000_000_000}
    for p in range(16):
        f = root / "cache-sim" / f"p{p}.npy"
        np.save(f, tr(96))
        sp = "train" if p < 8 else "dev" if p < 10 else "test"
        for j in range(9, 24):
            px = float(rng.choice([0, 50, 300, 800, 3000]))
            for r, streams in (("C", (0, 1)), ("K", (2, 3))):
                for sign, st in zip((1, -1), streams):
                    corr = sign == 1 and px >= 68 and rng.random() < 0.6
                    rows[r].append({"uid": 0, "domain": f"sim{r}", "key": f"p{p}", "slot": j, "sign": sign, "split": sp,
                                    "file": str(f), "row": st * 24 + j, "ctx": list(range(st * 24 + j - 8, st * 24 + j + 1)),
                                    "tc0": 1.0, "tc1": 0.0, "token": "", "labeled": True, "ped_corr": corr, "ped_wide": corr,
                                    "vru_wide": sign == 1 and px > 0, "ped_dist": 15.0 if corr else np.nan, "dist_bin": 1 if corr else -1,
                                    "uncertain": False, "normal": sign == -1, "inst": p, "px_eq": px,
                                    "vis": sign == 1 and px >= 68, "ped_lat": float(rng.choice([1.0, 5.5, 3.5])),
                                    "dv_star_1": -rng.random() if sign == 1 else np.nan,
                                    "dv_star_2": -rng.random() if sign == 1 else np.nan, "dv_star_3": -rng.random() if sign == 1 else np.nan,
                                    "ego_speed": float(rng.uniform(0, 12))})
    for r in "CK":
        df = pd.DataFrame(rows[r])
        df["uid"] = uid[r] + np.arange(len(df))
        df["twin_uid"] = df.uid + np.where(df.sign == 1, 1, -1)
        df["other_uid"] = df.uid - uid[r] + uid["K" if r == "C" else "C"]
        df.to_parquet(root / "index" / f"sim{r}.parquet", index=False)
    # nus: 8 scenes x 40 slots, stride 2, keyframes every 5
    (root / "cache-nus").mkdir(exist_ok=True)
    nr = []
    for s in range(18):
        f = root / "cache-nus" / f"s{s}.npz"
        np.savez(f, trunk=tr(40))
        sp = "train" if s < 6 else "dev" if s < 8 else "val"
        for j in range(40):
            key = j % 5 == 0
            vru = key and rng.random() < 0.3
            corr = vru and rng.random() < 0.5
            loc = j + np.arange(-16, 1, 2)
            nr.append({"domain": "nus", "key": f"s{s}", "slot": j, "sign": 0, "split": sp, "file": str(f), "row": j,
                       "ctx": np.where(loc >= 0, loc, -1).tolist(), "tc0": 1.0, "tc1": 0.0, "token": f"t{s}_{j}" if key else "",
                       "labeled": key, "ped_corr": corr, "ped_wide": corr, "vru_wide": vru,
                       "ped_dist": float(rng.uniform(5, 30)) if corr else np.nan, "dist_bin": 1 if corr else -1,
                       "uncertain": False, "normal": key and not vru, "ego_speed": float(rng.uniform(0, 15))})
    df = pd.DataFrame(nr)
    df["uid"] = 3_000_000_000 + np.arange(len(df))
    df.to_parquet(root / "index" / "nus.parquet", index=False)
    # wod: 8 streams x 30 slots, stride 1; nav: 6 logs x 10 tokens
    (root / "cache-wod").mkdir(exist_ok=True)
    wr = []
    for s in range(18):
        f = root / "cache-wod" / f"w{s}.npz"
        np.savez(f, trunk=tr(30))
        for j in range(8, 30):
            corr = rng.random() < 0.15
            wr.append({"domain": "wod", "key": f"w{s}", "slot": j, "sign": 0, "split": "train" if s < 6 else "dev" if s < 8 else "val", "file": str(f),
                       "row": j, "ctx": list(range(j - 8, j + 1)), "tc0": 1.0, "tc1": 0.0, "token": "", "labeled": True,
                       "ped_corr": corr, "ped_wide": corr, "vru_wide": corr, "ped_dist": 12.0 if corr else np.nan,
                       "dist_bin": 1 if corr else -1, "uncertain": rng.random() < 0.1, "normal": not corr,
                       "part": "val" if s >= 8 else "train", "ego_speed": float(rng.uniform(0, 15))})
    df = pd.DataFrame(wr)
    df["uid"] = 4_000_000_000 + np.arange(len(df))
    df.to_parquet(root / "index" / "wod.parquet", index=False)
    (root / "cache-nav").mkdir(exist_ok=True)
    vr = []
    for s in range(6):
        f = root / "cache-nav" / f"l{s}.npz"
        np.savez(f, trunk=tr(20))
        for i in range(10):
            vr.append({"domain": "nav", "key": f"l{s}", "slot": i, "sign": 0, "split": "train" if s < 5 else "dev", "file": str(f),
                       "row": 10 + i, "ctx": [-1] + list(range(i, i + 8)), "tc0": 1.0, "tc1": 0.0, "token": f"n{s}_{i}",
                       "labeled": False, "ped_corr": False, "ped_wide": False, "vru_wide": False, "ped_dist": np.nan, "dist_bin": -1,
                       "uncertain": False, "normal": True})
    nav = pd.DataFrame(vr)
    nav["uid"] = 5_000_000_000 + np.arange(len(nav))
    nav.to_parquet(root / "index" / "nav.parquet", index=False)
    # off: 40 train + 8 dev, twins = nav tokens of the same split
    (root / "cache-off").mkdir(exist_ok=True)
    f = root / "cache-off" / "0.npy"
    np.save(f, tr(48 * 9))
    orows = []
    for i in range(48):
        sp = "train" if i < 40 else "dev"
        tw = nav[nav.split == sp].uid.to_numpy()
        orows.append({"domain": "off", "key": f"o{i}", "slot": 0, "sign": 0, "split": sp, "file": str(f), "row": 9 * i + 8,
                      "ctx": list(range(9 * i, 9 * i + 9)), "tc0": 1.0, "tc1": 0.0, "token": "", "labeled": False,
                      "ped_corr": False, "ped_wide": False, "vru_wide": False, "ped_dist": np.nan, "dist_bin": -1,
                      "uncertain": False, "normal": False, "twin_uid": int(tw[i % len(tw)]), "corner": -1 if sp == "train" else i % 4})
    df = pd.DataFrame(orows)
    df["uid"] = 6_000_000_000 + np.arange(len(df))
    df.to_parquet(root / "index" / "off.parquet", index=False)


CANDS = ("op", "op_L", "op_R", "op_slow", "op_stop", "hold", "brake_hard", "brake_mild", "shift_L", "shift_R", "shift_L_slow",
         "shift_R_slow", "nudge_L")


def synth_scores(root: Path, seed=0):
    """S-format score tables built around the teacher's own plan (op) with random scores."""
    from jevdrive import op_adapt_r2 as R
    import torch
    rng = np.random.default_rng(seed)
    (root / "score").mkdir(exist_ok=True)
    for dn in ("simC", "simK", "nus", "off"):
        d = R.Domain(dn, root / "t")
        t = R.load_teacher(dn, root / "t")
        rows = np.arange(len(d)) if dn != "nus" else np.flatnonzero(d.col("labelled", False, bool))
        names = CANDS + (("rej",) if dn == "off" else ())
        K = len(names)
        op = R.plan_ch(torch.from_numpy(np.asarray(t["mu"][rows], np.float32))).numpy()
        traj = np.repeat(op[:, None], K, 1) + rng.normal(0, 0.5, (len(rows), K, 21, 4)).astype(np.float32) * (np.arange(K) > 0)[None, :, None, None]
        S = rng.random((len(rows), K)).astype(np.float32)
        S[rng.random(len(rows)) < 0.6, 0] = 1.0
        top = S >= S.max(1, keepdims=True) - 0.1
        out = {"uid": d.uid[rows], "cands": np.array(names), "traj": traj.astype(np.float32), "S": S, "top": top,
               "valid": rng.random(len(rows)) < 0.9, "NC": np.ones((len(rows), K), np.uint8),
               "DAC": (rng.random((len(rows), K)) < 0.8).astype(np.uint8), "DDC": rng.choice([0.5, 1.0], (len(rows), K)).astype(np.float32)}
        if dn.startswith("sim"):
            out["ped_lat"] = d.col("ped_lat")[rows].astype(np.float32)
        np.savez(root / "score" / f"{dn}.npz", **out)


def stub_modules():
    """Stand-ins for package S's score_plans and package D's adapter, injected as the real module names."""
    import torch
    S = types.ModuleType("jevdrive.op_adapt_score")

    def score_plans(domain, uid, plans, frame="op"):
        rng = np.random.default_rng(len(uid))
        n = len(uid)
        return {"S": rng.random(n), "NC": np.ones(n), "DAC": (rng.random(n) < 0.8).astype(float), "DDC": np.ones(n)}
    S.score_plans = score_plans

    class Adapter(torch.nn.Module):
        def __init__(self, f=16):
            super().__init__()
            self.proj = torch.nn.Linear(f, 512)
            torch.nn.init.zeros_(self.proj.weight), torch.nn.init.zeros_(self.proj.bias)
            self.gate = torch.nn.Parameter(torch.zeros(()))

        def forward(self, H, tok, mask):
            add = self.proj(tok.float().mean(2))[:, :, None]          # (B, 9, 1, 512)
            return H + (self.gate * add).to(H.dtype)

    def load_tok(cache_file):
        p = Path(cache_file)
        n = len(np.load(p, mmap_mode="r")) if p.suffix == ".npy" else len(np.load(p)["trunk"])
        rng = np.random.default_rng(abs(hash(p.name)) % 2 ** 31)
        return rng.standard_normal((n, 8, 16)).astype(np.float16), rng.random((n, 8)) < 0.7
    DD = types.ModuleType("jevdrive.op_adapt_det")
    DD.DetAdapter, DD.load_tok = Adapter, load_tok
    sys.modules["jevdrive.op_adapt_score"] = S
    sys.modules["jevdrive.op_adapt_det"] = DD
    import jevdrive
    jevdrive.op_adapt_score, jevdrive.op_adapt_det = S, DD


def synth(a):
    root = Path(os.environ["OP_R2_ROOT"])
    stub_modules()
    import op_adapt_r2_train as T
    from jevdrive import op_adapt_r2 as R
    if not (root / "t" / "teacher" / "tstd.npy").exists():
        synth_root(root)
        ns = types.SimpleNamespace
        T.cmd_pack(ns(domains=["simC", "simK", "nus", "wod", "nav", "off"], source="c", workers=4, limit=0, root="", det=True, copy=False))
        T.cmd_teacher(ns(domains=["simC", "simK", "nus", "wod", "nav", "off"], batch=64, root=""))
        synth_scores(root)
    doms = {dn: R.Domain(dn) for dn in ("simC", "simK", "nus", "off")}
    ds = R.DetSource()
    c = doms["simC"].ctx[:5]
    tk, mk = ds("simC", c)
    src = pd.read_parquet(root / "t/trunk/simC.src.parquet")
    f, rr = src.file[c[0, -1]], src.row[c[0, -1]]
    ref = sys.modules["jevdrive.op_adapt_det"].load_tok(f)
    check("pack_det: tokens follow the flat trunk rows", np.array_equal(tk[0, -1], ref[0][rr]) and np.array_equal(mk[0, -1], ref[1][rr]))
    check("pack: sim ctx rows point at the same trunk as the C index",
          np.array_equal(doms["simC"].gather(np.array([3]))[0][0, -1],
                         np.load(pd.read_parquet(root / "index/simC.parquet").file[3], mmap_mode="r")[pd.read_parquet(root / "index/simC.parquet").row[3]]))
    runs = {}
    for arm in a.arms:
        lam = 1.0
        ns = types.SimpleNamespace(arm=arm, seed=0, lam_s=lam, max_steps=a.steps, eval_every=a.steps, ckpt_every=2, batch=a.batch,
                                   loaders=2, prefetch=2, run_dir=str(root / f"run-{arm}"), root="", fresh=True)
        try:
            T.cmd_train(ns)
            ok = (root / f"run-{arm}" / "DONE").exists()
        except Exception as e:  # noqa: BLE001
            ok = False
            print(arm, "failed:", e)
        runs[arm] = root / f"run-{arm}"
        ev = [json.loads(l) for l in open(runs[arm] / "events.jsonl")]
        losses = [e for e in ev if e["kind"] == "scalar" and e["tag"].startswith("loss/")]
        fin = all(np.isfinite(e["value"]) for e in losses)
        keys = sorted({e["tag"] for e in losses})
        check(f"train {arm}", ok and fin, keys)
        s0 = [e for e in ev if e["kind"] == "score0"]
        if s0:
            check(f"step-0 L_score on op-in-Top slots ~ 0 ({arm})", s0[0]["max_loss_on_top"] < 1e-3, s0[0])
    # resume: continue A to 2 * steps from its checkpoint
    ns = types.SimpleNamespace(arm="A", seed=0, lam_s=1.0, max_steps=2 * a.steps, eval_every=100, ckpt_every=2, batch=a.batch,
                               loaders=2, prefetch=2, run_dir=str(root / "run-A"), root="", fresh=False)
    T.cmd_train(ns)
    ev = [json.loads(l) for l in open(root / "run-A" / "events.jsonl")]
    check("resume", any(e["kind"] == "resume" and e["step"] == a.steps for e in ev)
          and json.loads((root / "run-A" / "dev.json").read_text())["steps"] == 2 * a.steps)
    dv = json.loads((root / "run-A" / "dev.json").read_text())
    need = ("drift_median", "null_slow_delta_pp", "ks_v2_normal", "aux_auc_nus", "pair_acc_simC", "sjev_delta_simC",
            "xminus_slow_nearest_simC_adapt", "off_train_op_fail", "off_dev_pass_adapt", "twin_drift_median")
    check("dev eval fields", all(k in dv for k in need), [k for k in need if k not in dv])
    if "D" in runs:
        check("D gate present", "det_gate_abs" in json.loads((runs["D"] / "dev.json").read_text()))
    json.dump({"runs": {k: str(v) for k, v in runs.items()}}, open(root / "selftest_runs.json", "w"))


def readout(a):
    """Readout code on the synthetic root (after `synth`): O pass, model pass, every readout part that needs no
    registered set; the P5 / WOD-log / RFS / navsim parts need the real sets and are exercised by M1 (phase 2)."""
    root = Path(os.environ["OP_R2_ROOT"])
    stub_modules()
    import op_adapt_r2_readout as RO
    link = root / "train-A-s0-ls1"
    if not link.exists():
        link.symlink_to(root / "run-A")
    ns = types.SimpleNamespace
    RO.cmd_eval(ns(model="O", sets=["nusval", "wodval", "cosC", "cosK", "offdev"]))
    RO.cmd_eval(ns(model="A-s0-ls1", sets=["nusval", "wodval", "cosC", "cosK", "offdev"]))
    for m in ("O", "A-s0-ls1"):
        RO.cmd_read(ns(model=m, parts=["R-nus", "R-wod", "S-cos", "B-real", "B-score", "shortcut"]))
        sm = json.loads((RO.rdir(m) / "summary.json").read_text())
        errs = {k: v["error"] for k, v in sm.items() if isinstance(v, dict) and "error" in v}
        check(f"readout parts run ({m})", not errs, errs)
        check(f"readout R-nus / S-cos / B-real present ({m})", all(k in sm and sm[k] for k in ("R-nus", "S-cos", "B-real")))
    sm = json.loads((RO.rdir("A-s0-ls1") / "summary.json").read_text())
    check("B-score off corners", "corner0" in sm["B-score"]["offdev"], list(sm["B-score"]["offdev"]))
    p = {k: True for k in ("R-nus", "R-wod", "S-p5", "S-p5-all", "S-cos", "B-p5", "B-real", "N-drift", "N-ade", "N-lead", "N-nav",
                           "N-rfs", "N-cutin", "B-score-p5", "B-score-nus")}
    check("outcome P", RO.outcome(p) == ["P"])
    check("outcome P-rep", RO.outcome(p | {"B-real": False}) == ["P-rep", "sim-dominant"])
    check("outcome P-size", RO.outcome(p | {"S-p5-all": False})[0].startswith("P-size") if not RO.outcome(p | {"S-p5-all": False})[0] == "P"
          else RO.outcome(p | {"S-p5-all": False})[1].startswith("P-size"), RO.outcome(p | {"S-p5-all": False}))
    check("outcome real-only", RO.outcome(p | {"S-p5": False, "S-p5-all": False, "S-cos": False, "B-p5": False}) == ["real-only"])
    check("outcome sim-dominant", RO.outcome(p | {"R-wod": False}) == ["sim-dominant"])
    check("outcome none", RO.outcome({k: False for k in p}) == ["none"])
    # lane: checklists evaluate on the synthetic runs without raising; the packer runs a job on a card
    import op_adapt_r2_lane as LN
    c1 = LN.verdict(LN.check_stage1(root / "run-A"))
    check("stage-1 checklist evaluates", len(c1["items"]) == 6, [(i["item"][:30], i["pass"]) for i in c1["items"]])
    runs = {arm: root / f"run-{arm}" for arm in ("A", "D", "A-real", "A-sim", "A-noC", "A-noK", "D-only", "A-bhv")}
    c10 = LN.verdict(LN.check_stage10(runs))
    check("stage-10 checklist evaluates", len(c10["items"]) > 40, f"{len(c10['items'])} items, failed {len(c10['failed'])}")
    gpu = int(os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0])
    lane = root / "lane"
    (lane / "logs").mkdir(parents=True, exist_ok=True)
    done = LN.Packer([gpu], 78.0, "40-41", lane, print).run(
        [{"tag": "t1", "argv": [sys.executable, "-c", "print('ok')"], "vram_gb": 1.0}], poll=2)
    check("packer runs a job", done["t1"]["rc"] == 0, done)


def real(a):
    """The real package-C indexes: pack (mapped caches, no copy) into OP_R2_ROOT/t, sampler pools and composition of
    every arm, and the batch gather throughput (no teacher, no model, no readout set is read)."""
    import time
    from concurrent.futures import ThreadPoolExecutor
    from jevdrive import op_adapt_r2 as R
    import op_adapt_r2_train as T
    src = Path(os.environ["OP_R2_SRC"])                      # the real R2 root whose index/ is read
    ns = types.SimpleNamespace
    os.environ["OP_R2_ROOT"] = str(src)                     # index_c reads <R2>/index
    doms = [d for d in ("simC", "simK", "nus", "wod", "nav", "off", "p5") if (src / "index" / f"{d}.parquet").exists()]
    out = Path(a.out)
    for d in doms:
        t0 = time.time()
        T.cmd_pack(ns(domains=[d], source="c", workers=8, limit=0, root=str(out), det=False, copy=False))
        print(f"pack {d}: {time.time() - t0:.0f} s")
    D = {d: R.Domain(d, out) for d in doms}
    for arm in ("A", "D", "A-real", "A-sim", "A-noC", "A-noK", "A-bhv"):
        need = {"simC", "simK", "nus", "wod", "nav"} | ({"off"} if R.ARMS[arm].offset else set())
        if not need <= set(D):
            print(f"{arm}: missing domains {need - set(D)}")
            continue
        mx = R.Mixer(R.RunCfg(arm=arm), D)
        print(arm, json.dumps(mx.describe(), default=str))
        check(f"mixer {arm} pools non-empty", all(v > 0 for k, v in mx.describe()["pool_sizes"].items() if k != "off" or R.ARMS[arm].offset))
    if {"simC", "simK", "nus", "wod", "nav"} <= set(D):
        mx = R.Mixer(R.RunCfg(arm="A"), D)
        rng = np.random.default_rng(0)
        t0 = time.time()
        n = 0
        with ThreadPoolExecutor(4) as ex:
            for segs in [mx.draw(rng) for _ in range(20)]:
                xs = list(ex.map(lambda s: D[s.dom].gather(s.rows)[0], segs))
                n += sum(len(x) for x in xs)
        dt = time.time() - t0
        print(f"gather: {n} sequences in {dt:.1f} s = {n / dt:.0f} seq/s ({20 / dt:.1f} batches/s), cold page cache")
        check("gather throughput >= 1 batch / s (cold)", 20 / dt >= 1.0, f"{20 / dt:.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=("units", "synth", "readout", "real"))
    ap.add_argument("--out", default="/tmp/r2real")
    ap.add_argument("--arms", nargs="+", default=["A", "D", "A-real", "A-sim", "A-noC", "A-noK", "D-only", "A-bhv"])
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--batch", type=int, default=16)
    a = ap.parse_args()
    {"units": units, "synth": synth, "readout": readout, "real": real}[a.what](*(() if a.what == "units" else (a,)))
    bad = [n for n, ok in OK if not ok]
    print(f"{len(OK) - len(bad)}/{len(OK)} passed" + (f"; FAILED: {bad}" if bad else ""))
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
