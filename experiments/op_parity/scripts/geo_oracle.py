"""op_parity geo-oracle (plans/2026-10-09-geo-oracle-prereg.md, decision 197): the ceiling of a perfect geometry branch. Ground-truth drivable
SDF and ground-truth agent occupancy become 32 x 512 adapter memory tokens (the decision-160 channel, pp_train --mem) through a small geometry
tokenizer. PRIVILEGED INPUTS: every arm that reads these banks is an oracle probe, never a method and never a reportable driver. No WA-JEPA
weights or features anywhere. Training is pp_train.py --mem geo_*; navtest scores come from jevdrive.bench.

  tok    (op-train, GPU, pool)  --kind s | a | b: train the tokenizer of the kind (rasters -> conv -> 32 x 512 tokens -> a throw-away thin plan
                                head; imitation + drivable hinge (s, b) + agent hinge (a, b)) on navtrain shards outside the pilot's s2-s4, then
                                write runs/op_parity/mem/geo_<kind>/<data>.npy (N, 32, 512) fp16 for s2-s4 and navtest. kind b also writes
                                geo_x: the geo_b rows permuted inside each data dir, always to another log (the shuffled control).
                                Pre-registered checks (exit 1): dev ADE with true tokens <= 0.9 x with shuffled tokens; bank RMS within 25%.
                                --weights F: save the tokenizer weights to F and stop before the banks (the warm start of geo_e2e.py).
  split  (op-train, CPU)        register the tokenizer's fit rows as navsim/op-parity-geotok-train (run once, before the tok jobs)
  gate   (venv, CPU)            seed-0 clear-negative gate: GB - H0 EPDMS < +0.3 and CI high < +0.7 -> exit 2
  report (op-train, CPU, pool)  tables, channel-read evidence, the registered verdict -> $DATA_DIR/runs/op_parity/geo_oracle/report[-<name>]/
"""
import sys as _sys, pathlib as _pl  # noqa: E401
_R = _pl.Path(__file__).resolve().parents[3]
_sys.path[:0] = [str(_R), str(_R / "lib"), str(_R / "research"), str(_pl.Path(__file__).parent), str(_R / "experiments/op_probe/scripts")]
import argparse, glob, json, os  # noqa: E401,E402

import numpy as np  # noqa: E402

D = _pl.Path(os.environ.get("DATA_DIR", "/root/autodl-tmp/ujs"))
OUT = D / "runs" / "op_parity" / "geo_oracle"
MEM = D / "runs" / "op_parity" / "mem"
CR = D / "runs" / "op_parity" / "cache"
K_SH = 12
PILOT = tuple(f"navtrain_full.s{i}of{K_SH}" for i in (2, 3, 4))
TEST = "lb_navtest"
SDF_LAB = {"navtrain": "runs/op_probe/labels/navtrain_all.npz", "navtest": "runs/op_probe/labels/navtest.npz"}
AG_LAB = {"navtrain": "runs/op_parity/agent_labels/navtrain_all.npz", "navtest": "runs/op_parity/agent_labels/navtest.npz"}
CH = {"s": 1, "a": 3, "b": 4}
CLIP, SCALE, V_SCALE, V_REACH = 6.0, 3.0, 10.0, 2.0      # SDF channels: clip(+-6 m) / 3; velocity / 10 on cells within 2 m of the nearest box
NTOK, DTOK, GRID = 32, 512, (8, 4)                       # 32 tokens x 512, 8 strips of 8 m along x by 4 of 12 m along y
HINGE = dict(lam=30.0, margin=0.5)                       # the SH30 drivable hinge
AGENT = dict(lam=10.0, margin=0.5)                       # the decision-158 agent hinge
STEM = {"H0": "GH0", "GS": "GOS", "GA": "GOA", "GB": "GOB", "GX": "GOX"}
KIND = {"GS": "geo_s", "GA": "geo_a", "GB": "geo_b", "GX": "geo_x"}
NB = 4000
LINES = ("combined arm (GB) navtest >= +0.7 and CI low > +0.3: geometry is enough; "
         "< +0.3: negative, not geometry; between: partial")


# ---------------------------------------------------------------- tokenizer
class Src:
    """One label domain (navtrain: all 12 shards in shard order; navtest) on the device: SDF, agent boxes, ego, logged future."""

    def __init__(self, dom, dev):
        import torch
        from agent_hinge import AgentHinge
        from drivable_hinge import Hinge
        datas = [f"navtrain_full.s{i}of{K_SH}" for i in range(K_SH)] if dom == "navtrain" else [TEST]
        tabs = [np.load(CR / d / "tab.npz") for d in datas]
        self.names = np.concatenate([t["names"] for t in tabs])
        self.log = np.concatenate([t["log"] for t in tabs])
        self.data = np.concatenate([[d] * len(t["names"]) for d, t in zip(datas, tabs)])
        fut = np.concatenate([t["fut"] for t in tabs])
        self.has_fut = ~np.isnan(fut[:, 0, 0])
        self.fut = torch.as_tensor(np.nan_to_num(fut), dtype=torch.float32, device=dev)
        self.ego = torch.as_tensor(np.concatenate([t["ego"] for t in tabs]), dtype=torch.float32, device=dev)
        self.H = Hinge(D / SDF_LAB[dom], self.names, dev, HINGE["margin"])
        self.H.sdf[~self.H.ok] = 0                                                # rows without a label: an all-zero raster (Hinge leaves them empty)
        self.A = AgentHinge(D / AG_LAB[dom], self.names, dev, AGENT["margin"])
        self.ok = self.H.ok.cpu().numpy() & self.A.ok.cpu().numpy()
        from drivable_hinge import NH, NW, X0, Y0
        self.gx = (X0 + (torch.arange(NH, device=dev) + 0.5) * 0.5)[None, :, None, None]
        self.gy = (Y0 + (torch.arange(NW, device=dev) + 0.5) * 0.5)[None, None, :, None]

    def raster(self, rows, kind):
        """rows (B,) long -> (B, C, 128, 96) input raster of the kind (s: drivable SDF; a: agent distance, vx, vy; b: both)."""
        import torch
        from agent_hinge import _box_sdf
        ch = []
        if kind in ("s", "b"):
            ch.append(self.H.sdf[rows].float().clamp(-CLIP, CLIP) / SCALE)
        if kind in ("a", "b"):
            b, v = self.A.box[rows], self.A.val[rows]                              # (B, 9, K, 5) x, y, yaw, length, width; (B, 9, K)
            b0, v0 = b[:, 0], v[:, 0]
            vel = (b[:, 1, :, :2] - b0[:, :, :2]) / 0.5 * (v0 & v[:, 1])[..., None]  # (B, K, 2), ego axes at t0
            e = lambda x: x[:, None, None, :]  # noqa: E731
            d = _box_sdf(self.gx, self.gy, e(b0[..., 0]), e(b0[..., 1]), e(b0[..., 2]), e(b0[..., 3] / 2), e(b0[..., 4] / 2))
            d = d.masked_fill(~e(v0), 1e3)
            dm, ix = d.min(-1)                                                    # (B, 128, 96)
            near = (dm < V_REACH).float()
            vv = [torch.gather(vel[..., c][:, None, None, :].expand(-1, d.shape[1], d.shape[2], -1), 3, ix[..., None])[..., 0] * near / V_SCALE
                  for c in (0, 1)]
            ch.append(torch.stack([dm.clamp(-CLIP, CLIP) / SCALE, *vv], 1))
        return torch.cat(ch, 1)


def build_net(c_in):
    import torch
    import torch.nn as nn

    class GeoTok(nn.Module):
        def __init__(self):
            super().__init__()
            w, L = (c_in, 32, 64, 128, 256), []
            for i in range(4):                                                    # 128 x 96 -> 8 x 6
                L += [nn.Conv2d(w[i], w[i + 1], 3, 2, 1), nn.GroupNorm(8, w[i + 1]), nn.GELU()]
            L += [nn.Conv2d(256, 256, 3, 1, 1), nn.GroupNorm(8, 256), nn.GELU(), nn.Conv2d(256, DTOK, 1), nn.AdaptiveAvgPool2d(GRID)]
            self.enc = nn.Sequential(*L)
            self.head = nn.Sequential(nn.Dropout(0.1), nn.Linear(NTOK * DTOK + 20, 1024), nn.GELU(), nn.Linear(1024, 1024), nn.GELU(), nn.Linear(1024, 24))

        def tokens(self, x):
            return self.enc(x).flatten(2).transpose(1, 2)                         # (B, 32, 512), slot = x strip * 4 + y strip

        def plan(self, tok, ego):
            return self.head(torch.cat([tok.flatten(1), ego], 1)).view(-1, 8, 3)

    return GeoTok()


def derange(logs, rng):
    from turn_oracle import derange as _d
    return _d(logs, rng)


def fit_mask():
    """Tokenizer fit rows over the 12 navtrain shards in shard order (numpy only): outside the pilot's s2-s4, drivable + agent labels ok, a logged
    future, log not in navsim/op-parity-full-dev."""
    from jevdrive.data import splits
    datas = [f"navtrain_full.s{i}of{K_SH}" for i in range(K_SH)]
    tabs = [np.load(CR / d / "tab.npz") for d in datas]
    names, log = np.concatenate([t["names"] for t in tabs]), np.concatenate([t["log"] for t in tabs])
    data = np.concatenate([[d] * len(t["names"]) for d, t in zip(datas, tabs)])
    has_fut = ~np.isnan(np.concatenate([t["fut"][:, 0, 0] for t in tabs]))
    ok = np.ones(len(names), bool)
    for f in (SDF_LAB["navtrain"], AG_LAB["navtrain"]):
        z = np.load(D / f)
        pos = {t: i for i, t in enumerate(z["tokens"].tolist())}
        ix = np.array([pos.get(t, -1) for t in names.tolist()])
        ok &= (ix >= 0) & z["ok"][np.maximum(ix, 0)]
    dv = splits.load("navsim/op-parity-full-dev").mask(names)
    return ~np.isin(data, PILOT) & ok & has_fut & ~np.isin(log, np.unique(log[dv])), names


def cmd_split(a):
    """Register navsim/op-parity-geotok-train (idempotent)."""
    from jevdrive.data import splits
    fit, names = fit_mask()
    s = splits.define("navsim", "op-parity-geotok-train", names[fit].tolist(), unit="token",
                      origin=f"navtrain shards 0, 1, 5-11 of {K_SH} (pp_prep caches) with drivable + agent labels and a logged future, logs of "
                             "navsim/op-parity-full-dev removed; token-disjoint from navsim/op-parity-s234-{train,dev}",
                      used_by=["experiments/op_parity"], status="frozen", notes="geo-oracle tokenizer fit rows (decision 197)")
    for n in ("navsim/op-parity-s234-train", "navsim/op-parity-s234-dev"):
        splits.check_disjoint(s, splits.load(n))
    print("registered", s.id, len(names[fit]))


def cmd_tok(a):
    import time
    import torch
    import torch.nn.functional as F
    from jevdrive.data import splits
    from jevdrive.run import Run
    dev = torch.device("cuda")
    k = a.kind
    OUT.mkdir(parents=True, exist_ok=True)
    with Run("op_parity", f"geo-oracle-tok-{k}" + ("-smoke" if a.smoke else ""), seed=0, config=vars(a)) as run:
        S, T = Src("navtrain", dev), Src("navtest", dev)
        s_tr, s_dv, s_fd, s_te = (splits.load(n) for n in ("navsim/op-parity-s234-train", "navsim/op-parity-s234-dev", "navsim/op-parity-full-dev",
                                                          "navsim/navtest"))
        fit, _names = fit_mask()
        assert (_names == S.names).all()
        s_tk = splits.load("navsim/op-parity-geotok-train")                       # registered by `split` (one writer, before the tokenizer jobs)
        assert (fit == s_tk.mask(S.names)).all() and not (fit & ~(S.ok & S.has_fut)).any(), "fit rows differ from the registered split"
        for s in (s_tk, s_tr, s_dv, s_te):
            run.use_split(s)
        fit_r = torch.as_tensor(np.flatnonzero(fit), device=dev)
        ev_ok = S.ok & S.has_fut
        sets = {"tok_train": np.flatnonzero(fit)[:: max(1, int(fit.sum()) // 4000)], "pilot_train": np.flatnonzero(s_tr.mask(S.names) & ev_ok),
                "dev": np.flatnonzero(s_dv.mask(S.names) & ev_ok)}
        sets["pilot_train"] = sets["pilot_train"][:: max(1, len(sets["pilot_train"]) // 4000)]
        run.info(f"kind {k}: fit {int(fit.sum())} rows ({s_tk.id}); eval sets " + ", ".join(f"{n} {len(r)}" for n, r in sets.items()))
        torch.manual_seed(0)
        net = build_net(CH[k]).to(dev).to(memory_format=torch.channels_last)
        opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-2)
        steps, wu = (30 if a.smoke else a.steps), (5 if a.smoke else 300)
        g = torch.Generator(device=dev).manual_seed(0)
        t0, hist = time.time(), []
        for step in range(steps):
            for q in opt.param_groups:
                q["lr"] = 1e-3 * min(1.0, (step + 1) / wu) * 0.5 * (1 + np.cos(np.pi * step / steps))
            r = fit_r[torch.randint(0, len(fit_r), (a.batch,), device=dev, generator=g)]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                tok = net.tokens(S.raster(r, k).contiguous(memory_format=torch.channels_last))
            P = net.plan(tok.float(), S.ego[r])
            Y = S.fut[r]
            Ls = {"imit": F.huber_loss(P[..., :2], Y[..., :2], delta=1.0) + 3.0 * F.huber_loss(P[..., 2], Y[..., 2], delta=0.1)}
            loss = Ls["imit"]
            if k in ("s", "b"):
                Ls["hinge"] = S.H(P[..., 0], P[..., 1], P[..., 2], r)
                loss = loss + HINGE["lam"] * Ls["hinge"]
            if k in ("a", "b"):
                Ls["agent"] = S.A(P[..., 0], P[..., 1], P[..., 2], r)
                loss = loss + AGENT["lam"] * Ls["agent"]
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss at step {step}")
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            hist.append({n: float(v) for n, v in Ls.items()})
            if (step + 1) % 100 == 0 or step + 1 == steps:
                m = {n: float(np.mean([h[n] for h in hist])) for n in hist[-1]}
                hist = []
                run.scalars({f"loss/{n}": v for n, v in m.items()}, step + 1)
                run.info(f"step {step + 1}: " + ", ".join(f"{n} {v:.4f}" for n, v in m.items()) + f"; {(step + 1) / (time.time() - t0):.1f} it/s")
                run.status(f"tokenizer {k}: step {step + 1}/{steps}")
        net.eval()

        @torch.no_grad()
        def tokenize(src, rows, bs=512):
            out = []
            for i in range(0, len(rows), bs):
                r = torch.as_tensor(rows[i:i + bs], device=dev)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    out.append(net.tokens(src.raster(r, k).contiguous(memory_format=torch.channels_last)).float())
            return torch.cat(out)

        @torch.no_grad()
        def head_eval(rows, shuffle=False):
            tok = tokenize(S, rows)
            if shuffle:
                tok = tok[torch.as_tensor(derange(S.log[rows], np.random.default_rng(1)), device=dev)]
            r = torch.as_tensor(rows, device=dev)
            P = torch.cat([net.plan(tok[i:i + 1024], S.ego[r[i:i + 1024]]) for i in range(0, len(r), 1024)])
            ade = (P[..., :2] - S.fut[r][..., :2]).norm(dim=-1).mean()
            out_ = (S.H.margins(P[..., 0], P[..., 1], P[..., 2], r).amin(1) < 0).float().mean()
            hit = (S.A.per_step(P[..., 0], P[..., 1], P[..., 2], r).amax(1) > AGENT["margin"]).float().mean()   # ego box overlaps a counted object box
            return dict(ade=float(ade), footprint_out=float(out_), agent_overlap=float(hit), n=len(rows))

        st = {"kind": k, "channels": CH[k], "params_enc": sum(p.numel() for p in net.enc.parameters()), "fit_rows": int(fit.sum()),
              "split": s_tk.id, "train_s": time.time() - t0, "steps": steps}
        st["head"] = {n: head_eval(r) for n, r in sets.items()}
        st["head"]["dev_shuffled_tokens"] = head_eval(sets["dev"], shuffle=True)
        ratio = st["head"]["dev"]["ade"] / st["head"]["dev_shuffled_tokens"]["ade"]
        st["check_dev_ade_ratio_true_over_shuffled"] = ratio
        run.info("head: " + json.dumps(st["head"]))
        if a.smoke:
            tokenize(S, sets["dev"][:64]), tokenize(T, np.arange(64))
            run.info("smoke: tokenizer, losses and eval run; no bank written")
            return
        if a.weights:                                                             # geo-e2e warm start (geo_e2e.py): weights + stats only, no bank is touched
            assert ratio <= 0.9, f"tokenizer {k}: dev ADE true / shuffled {ratio:.3f} > 0.9"
            _pl.Path(a.weights).parent.mkdir(parents=True, exist_ok=True)
            torch.save(net.state_dict(), a.weights)
            _pl.Path(a.weights).with_suffix(".json").write_text(json.dumps(st, indent=1))
            run.summary.update(weights=a.weights, dev_ade=st["head"]["dev"]["ade"], dev_ade_shuffled=st["head"]["dev_shuffled_tokens"]["ade"])
            return
        # ---- banks (tab order of each data dir; rows without labels are tokenized from an all-zero SDF / an empty agent set)
        rms = {}
        for d in (*PILOT, TEST):
            src = T if d == TEST else S
            rows = np.arange(len(src.names)) if d == TEST else np.flatnonzero(S.data == d)
            assert (src.names[rows] == np.load(CR / d / "tab.npz")["names"]).all()
            bank = tokenize(src, rows).cpu().numpy().astype(np.float16)
            assert bank.shape == (len(rows), NTOK, DTOK) and np.isfinite(bank).all()
            _save(f"geo_{k}", d, bank)
            rms[d] = float(np.sqrt((bank[:: max(1, len(bank) // 2000)].astype(np.float32) ** 2).mean()))
            st[f"label_coverage/{d}"] = float(src.ok[rows].mean())
            if k == "b":
                lg = src.log[rows]
                perm = derange(lg, np.random.default_rng(0))
                assert not (lg[perm] == lg).any()
                _save("geo_x", d, bank[perm])
                np.save(MEM / "geo_x" / f"{d}.perm.npy", perm)
                st[f"shuffle_token_l2_ratio/{d}"] = float(np.linalg.norm((bank[perm][:2000] - bank[:2000]).astype(np.float32).reshape(2000, -1), axis=1).mean()
                                                          / np.linalg.norm(bank[:2000].astype(np.float32).reshape(2000, -1), axis=1).mean())
            run.info(f"bank geo_{k}/{d}: {bank.shape}, rms {rms[d]:.3f}, label coverage {st[f'label_coverage/{d}']:.4f}")
        st["bank_rms"] = rms
        st["check_bank_rms_spread"] = max(rms.values()) / min(rms.values()) - 1
        ok = bool(ratio <= 0.9 and st["check_bank_rms_spread"] < 0.25)
        st["checks_pass"] = ok
        (OUT / f"tok_{k}.json").write_text(json.dumps(st, indent=1))
        run.summary.update(checks_pass=ok, dev_ade=st["head"]["dev"]["ade"], dev_ade_shuffled=st["head"]["dev_shuffled_tokens"]["ade"],
                           pilot_train_ade=st["head"]["pilot_train"]["ade"], bank_rms_spread=st["check_bank_rms_spread"])
        if not ok:
            raise SystemExit(f"tokenizer {k}: pre-registered check failed (dev ADE true / shuffled {ratio:.3f} > 0.9 or bank RMS spread "
                             f"{st['check_bank_rms_spread']:.3f} >= 0.25)")


def _save(kind, data, x):
    d = MEM / kind
    d.mkdir(parents=True, exist_ok=True)
    tmp = d / f".{data}.tmp.npy"
    np.save(tmp, x)
    os.replace(tmp, d / f"{data}.npy")


# ---------------------------------------------------------------- gate / report
def spec(arm, seed):
    """'GB' / 'GB:off' -> bench model spec (':off' = memory masked at test)."""
    off = arm.endswith(":off")
    a = arm[:-4] if off else arm
    return f"{STEM[a]}-F-s{seed}" + (":noside" if off else "")


def frames(seeds, arms):
    """Per-token seed-mean frame of every arm (x 100): EPDMS, sub-scores, failure flags. navtest tab order. No replay needed."""
    import pandas as pd
    from jevdrive.bench import tables as BT
    tab = np.load(CR / TEST / "tab.npz")
    tok, log = tab["names"], tab["log"]
    dpsi = np.abs(np.degrees(np.arctan2(np.sin(tab["fut"][:, 7, 2]), np.cos(tab["fut"][:, 7, 2]))))
    sets = {"all": np.ones(len(tok), bool), "< 5 deg": dpsi < 5, "5-20 deg": (dpsi >= 5) & (dpsi <= 20), "20-45 deg": (dpsi > 20) & (dpsi <= 45),
            "> 45 deg": dpsi > 45, "> 20 deg": dpsi > 20}
    assert (int(sets["> 20 deg"].sum()), int(sets["> 45 deg"].sum())) == (3154, 1517), "turn strata differ from decision 160's T20 / T45"
    F = {}
    for arm in arms:
        fs = []
        for s in seeds:
            u = BT.load("navtest", spec(arm, s))[0]
            assert u is not None, f"no navtest result for {spec(arm, s)}"
            u = u.reindex(tok)
            assert not u.score.isna().any(), f"missing navtest tokens in {spec(arm, s)}"
            f = pd.DataFrame({"EPDMS": u.score.to_numpy(float)}, index=tok)
            for c in ("NC", "DAC", "DDC", "TLC", "EP", "TTC", "LK", "HC", "EC"):
                if c in u:
                    f[c] = u[c].to_numpy(float)
            f["DAC fail %"] = (u.DAC < 1).to_numpy(float)
            f["NC fail %"] = (u.NC < 1).to_numpy(float)
            f["TTC fail %"] = (u.TTC < 1).to_numpy(float)
            f["NC+TTC fail %"] = ((u.NC < 1) | (u.TTC < 1)).to_numpy(float)
            fs.append(f * 100)
        F[arm] = sum(fs) / len(fs)
    return F, sets, tok, log


def paired(A, B, col, m, log):
    from jevdrive import stats
    r = stats.paired(A[col].to_numpy()[m], B[col].to_numpy()[m], groups=log[m], n_boot=NB)
    return dict(arm=r["mean_a"], ref=r["mean_b"], diff=r["mean"], lo=r["lo"], hi=r["hi"])


def dev_ade(arm, seed):
    fs = sorted(glob.glob(str(D / "runs" / "op_parity" / f"train-{STEM[arm]}-F-s{seed}" / "*" / "DONE")))
    return json.loads(open(fs[-1]).read()).get("dev_ade") if fs else None


def cmd_gate(a):
    F, sets, _, log = frames([0], ["H0", "GB"])
    r = paired(F["GB"], F["H0"], "EPDMS", sets["all"], log)
    neg = bool(r["diff"] < 0.3 and r["hi"] < 0.7)
    res = dict(rule="clear negative iff seed-0 GB - H0 navtest EPDMS < +0.3 and CI high < +0.7", GB_minus_H0=r, clear_negative=neg)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "gate-s0.json").write_text(json.dumps(res, indent=1, default=float))
    print(json.dumps(res, indent=1, default=float))
    raise SystemExit(2 if neg else 0)


def cmd_report(a):
    import pandas as pd
    import turn_oracle as TO
    from jevdrive.run import Run
    out = OUT / ("report" + (f"-{a.name}" if a.name else ""))
    out.mkdir(parents=True, exist_ok=True)
    arms = ["H0", "GS", "GA", "GB", "GX", "GS:off", "GA:off", "GB:off"]
    with Run("op_parity", f"geo-oracle-{out.name}", seed=0, config=vars(a)) as run:
        F, sets, tok, log = frames(a.seeds, arms)
        # turn anatomy (inside cut / cannot make the turn) from the four_dirs replay, decision 166's read-outs
        Dt = TO.Data(a.replays)
        assert (Dt.tok == tok).all()
        for arm in arms:
            fs = [Dt.one(spec(arm, s))[0] for s in a.seeds]
            g = sum(fs) / len(fs)
            for c in ("inside-cut %", "cannot-make-turn %", "other DAC fail %", "raw-plan departure %"):
                F[arm][c] = g[c].to_numpy()
            assert np.allclose(g["DAC fail %"].to_numpy(), F[arm]["DAC fail %"].to_numpy())
        sets["sharp R < 15 m"] = Dt.sets["sharp R < 15 m"]
        TABLE = ([("EPDMS", s) for s in ("all", "< 5 deg", "5-20 deg", "20-45 deg", "> 45 deg")] +
                 [("DAC fail %", s) for s in ("all", "> 20 deg", "20-45 deg", "> 45 deg")] +
                 [("inside-cut %", "> 45 deg"), ("cannot-make-turn %", "> 45 deg"), ("other DAC fail %", "> 45 deg"), ("raw-plan departure %", "> 45 deg"),
                  ("inside-cut %", "> 20 deg"), ("cannot-make-turn %", "> 20 deg"), ("NC+TTC fail %", "all"), ("NC fail %", "all"), ("TTC fail %", "all"),
                  ("NC+TTC fail %", "< 5 deg"), ("NC+TTC fail %", "> 20 deg")] +
                 [(c, "all") for c in ("NC", "DAC", "DDC", "TLC", "EP", "TTC", "LK", "HC", "EC") if c in F["H0"]] +
                 [(c, s) for s in ("< 5 deg", "> 20 deg", "> 45 deg") for c in ("NC", "DAC", "EP", "TTC", "LK") if c in F["H0"]])
        refs = {"H0": arms, "GX": ["GB", "GS", "GA"], "GB": ["GB:off"], "GS": ["GS:off"], "GA": ["GA:off"]}
        rows = []
        for col, sn in TABLE:
            m = sets[sn]
            for arm in arms:
                r = dict(metric=col, stratum=sn, n=int(m.sum()), arm=arm, value=float(F[arm][col].to_numpy()[m].mean()))
                for ref, who in refs.items():
                    if arm in who and arm != ref:
                        p = paired(F[arm], F[ref], col, m, log)
                        r |= {f"diff_vs_{ref}": p["diff"], f"lo_vs_{ref}": p["lo"], f"hi_vs_{ref}": p["hi"]}
                rows.append(r)
        T = pd.DataFrame(rows)
        T.to_csv(out / "arms.csv", index=False)
        get = lambda arm, col, sn, ref: T[(T.arm == arm) & (T.metric == col) & (T.stratum == sn)].iloc[0][[f"diff_vs_{ref}", f"lo_vs_{ref}", f"hi_vs_{ref}"]].tolist()  # noqa: E731
        g = get("GB", "EPDMS", "all", "H0")
        ex = lambda v: bool(v[1] > 0 or v[2] < 0)  # noqa: E731
        read_a, read_b = get("GB", "EPDMS", "all", "GX"), get("GB:off", "EPDMS", "all", "GB")
        channel_read = ex(read_a) or ex(read_b)
        line = "geometry is enough" if (g[0] >= 0.7 and g[1] > 0.3) else "negative" if g[0] < 0.3 else "partial"
        if line == "negative" and not channel_read:
            line = "negative, but the channel was not read (decision 166's state; not a ceiling of geometry)"
        ade = {arm: [dev_ade(arm, s) for s in a.seeds] for arm in ("H0", "GS", "GA", "GB", "GX")}
        vd = dict(registered_lines=LINES, verdict=line, seeds=list(a.seeds), GB_minus_H0_EPDMS=g, channel_read=channel_read,
                  channel_evidence=dict(GB_minus_GX_EPDMS=read_a, GBoff_minus_GB_EPDMS=read_b, GB_minus_GX_T20_DAC_fail=get("GB", "DAC fail %", "> 20 deg", "GX"),
                                        GSoff_minus_GS_EPDMS=get("GS:off", "EPDMS", "all", "GS"), GAoff_minus_GA_EPDMS=get("GA:off", "EPDMS", "all", "GA"),
                                        dev_ade_m=ade),
                  per_seed_EPDMS={arm: [float(frames([s], [arm])[0][arm]["EPDMS"].mean()) for s in a.seeds] for arm in arms},
                  reference=dict(d160_MW_minus_H0="+0.95 [+0.47, +1.37] (P2H lambda-10 baseline, WA-Cf tokens; not rerun)"),
                  tokenizer={k: json.loads((OUT / f"tok_{k}.json").read_text()) for k in CH if (OUT / f"tok_{k}.json").exists()},
                  replay_checks=Dt.check)
        (out / "verdict.json").write_text(json.dumps(vd, indent=1, default=float))
        tk = pd.DataFrame({"token": tok, "log": log})
        for arm in arms:
            for c in ("EPDMS", "DAC fail %", "NC+TTC fail %", "inside-cut %", "cannot-make-turn %"):
                tk[f"{arm} | {c}"] = F[arm][c].to_numpy()
        tk.to_csv(out / "tokens.csv.gz", index=False, float_format="%.4g")
        L = [f"seeds {a.seeds}; values are per-token seed means x 100; diffs with 95% CI (log-cluster paired bootstrap, B {NB}, 12 146 navtest tokens)\n"]
        for ref, who in refs.items():
            cols = [ref] + [w for w in who if w != ref]
            L.append(f"\n**diff vs {ref}**\n\n| metric | stratum | n | " + " | ".join(cols) + " |\n|:--|:--|--:|" + ":--|" * len(cols))
            for col, sn in TABLE:
                q = T[(T.metric == col) & (T.stratum == sn)].set_index("arm")
                cells = [f"{q.loc[c, 'value']:.2f}" + ("" if c == ref else f" ({q.loc[c, f'diff_vs_{ref}']:+.2f} [{q.loc[c, f'lo_vs_{ref}']:+.2f}, {q.loc[c, f'hi_vs_{ref}']:+.2f}])")
                         for c in cols]
                L.append(f"| {col} | {sn} | {q.n.iloc[0]} | " + " | ".join(cells) + " |")
        L.append("\n```json\n" + json.dumps({k: v for k, v in vd.items() if k not in ("replay_checks", "tokenizer")}, indent=1, default=float) + "\n```")
        (out / "tables.md").write_text("\n".join(L) + "\n")
        run.info("\n".join(L))
        run.summary.update(verdict=line, GB_minus_H0=g, channel_read=channel_read, out=str(out))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("tok")
    p.add_argument("--kind", required=True, choices=list(CH))
    p.add_argument("--steps", type=int, default=6000)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--smoke", action="store_true", help="30 steps, the evals and a 64-row tokenize; no bank, no stats file")
    p.add_argument("--weights", default="", help="save the trained tokenizer's state dict here (+ .json stats) and stop: no bank is written")
    sp.add_parser("gate")
    sp.add_parser("split")
    p = sp.add_parser("report")
    p.add_argument("--name", default="")
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1])
    p.add_argument("--replays", nargs="+", default=["geo_s0", "geo_s1"])
    a = ap.parse_args()
    {"tok": cmd_tok, "gate": cmd_gate, "report": cmd_report, "split": cmd_split}[a.cmd](a)
