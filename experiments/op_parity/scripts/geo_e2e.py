"""op_parity geo-e2e (plans/2026-10-09-geo-e2e-prereg.md, decision 200): the follow-up of the geo-oracle (decision 197). The geometry tokenizer is
no longer trained once and frozen: it sits inside the adapter training (pp_train.py --mem-e2e), so the plan imitation and hinge gradients flow
through the adapter memory channel into the tokenizer. PRIVILEGED INPUTS: the true drivable SDF, the true agent boxes and (arm JP) the logged
future path enter as memory; every arm here is an oracle probe, never a method and never a reportable driver. No WA-JEPA weights or features.

  OnlineMem / finish   used by pp_train.py --mem-e2e <kind>: memory tokens computed from the rasters of the batch rows by a trainable tokenizer
                       (geo_oracle.build_net); after training the navtest bank of the tag is written to runs/op_parity/mem/ge_<tag>/lb_navtest.npy
                       (what jevdrive.bench reads), with dev diagnostics (ADE and raw-footprint departures with memory on / masked / mismatched).
       kinds: b  true drivable SDF + true agent occupancy (4 channels, geo_oracle's raster)
              x  the same rasters permuted across logs (geo_oracle's geo_x permutation of every data dir): the shuffled-content control
              p  distance to the LOGGED FUTURE PATH (1 channel; leaks the label): the positive control of the set-up, not a geometry arm
  report (op-train, CPU, pool)   tables, the registered verdict and the channel-read rule -> $DATA_DIR/runs/op_parity/geo_e2e/report/
"""
import sys as _sys, pathlib as _pl  # noqa: E401
_R = _pl.Path(__file__).resolve().parents[3]
_sys.path[:0] = [str(_R), str(_R / "lib"), str(_R / "research"), str(_pl.Path(__file__).parent), str(_R / "experiments/op_probe/scripts")]
import argparse, json  # noqa: E401,E402

import numpy as np  # noqa: E402

import geo_oracle as G  # noqa: E402

OUT = G.D / "runs" / "op_parity" / "geo_e2e"
CH = {"b": 4, "x": 4, "p": 1}
STEM = {"JB": "GEB", "JX": "GEX", "JW": "GEW", "JP": "GEP"}     # JB joint tokenizer (primary), JX shuffled control, JW warm start, JP path field
G.STEM.update(STEM)
ARMS = ["H0", "JB", "JX", "JW", "JP", "JB:off", "JW:off", "GB", "GX"]   # GB / GX: decision 197's frozen-tokenizer arms (reference rows)
REFS = {"H0": ARMS, "JX": ["JB", "JW"], "JB": ["JB:off", "JW"], "JW": ["JW:off"], "GX": ["GB"], "GB": ["JB", "JW"]}
LINES = ("combined arm (JB) navtest >= +0.7 and CI low > +0.3: geometry is enough; < +0.3: negative, not geometry; between: partial. "
         "Channel read iff JB - JX EPDMS CI above 0 AND JB memory-masked - JB EPDMS CI below 0. "
         "Set-up absorbs iff JP - H0 EPDMS CI above 0 and JP dev ADE <= 0.8 x H0's")


# ---------------------------------------------------------------- online memory (pp_train --mem-e2e)
def path_field(fut, gx, gy):
    """fut (B, 8, 3) logged future poses (rear-axle frame at t0) -> (B, 1, 128, 96): distance of every cell centre to the polyline through the
    origin and the 8 poses, clip(6 m) / 3 (the scaling of the SDF channels). The spatial path only: no timing, no speed."""
    import torch
    P = torch.cat([fut.new_zeros(len(fut), 1, 2), fut[..., :2]], 1)
    a, d = P[:, None, None, :-1], (P[:, 1:] - P[:, :-1])[:, None, None]             # (B, 1, 1, 8, 2)
    cx, cy = gx.reshape(1, -1, 1, 1) - a[..., 0], gy.reshape(1, 1, -1, 1) - a[..., 1]
    t = ((cx * d[..., 0] + cy * d[..., 1]) / d.pow(2).sum(-1).clamp_min(1e-6)).clamp(0, 1)
    dist = torch.hypot(cx - t * d[..., 0], cy - t * d[..., 1]).amin(-1)
    return (dist.clamp(max=G.CLIP) / G.SCALE)[:, None]


class OnlineMem:
    """Memory tokens (B, 32, 512) fp32 of store rows, computed by the trainable tokenizer. H: drivable_hinge.Hinge over the rows (its SDF raster),
    A: agent_hinge.AgentHinge over the rows (boxes), fut (N, 8, 3); perm (N,) long: row r reads the raster of row perm[r] (kind x)."""

    def __init__(self, kind, H, A, fut, dev, perm=None, net=None, init=""):
        import torch
        from drivable_hinge import NH, NW, X0, Y0
        self.kind, self.H, self.A, self.fut = kind, H, A, fut
        if H is not None:
            H.sdf[~H.ok] = 0                                                      # rows without a label: an all-zero raster (Hinge leaves them empty)
        self.gx = (X0 + (torch.arange(NH, device=dev) + 0.5) * 0.5)[None, :, None, None]
        self.gy = (Y0 + (torch.arange(NW, device=dev) + 0.5) * 0.5)[None, None, :, None]
        self.perm = None if perm is None else torch.as_tensor(np.asarray(perm), device=dev)
        self.net = net if net is not None else G.build_net(CH[kind]).to(dev)
        if init:
            self.net.load_state_dict(torch.load(init, map_location="cpu"))
        self.params = list(self.net.enc.parameters())                             # the thin head of build_net is not used here

    def raster(self, rows):
        r = rows if self.perm is None else self.perm[rows]
        return path_field(self.fut[r], self.gx, self.gy) if self.kind == "p" else G.Src.raster(self, r, "b")

    def __getitem__(self, rows):
        return self.net.tokens(self.raster(rows))


def attach(cfg, S, hinge, dev):
    """The OnlineMem of a pp_train store (rows = the concatenated data dirs)."""
    import torch
    from agent_hinge import AgentHinge
    k = cfg.mem_e2e
    assert hinge is not None and hasattr(hinge, "sdf"), "--mem-e2e reads the SDF raster of the footprint hinge (needs --hinge-lam > 0, no --hinge-replay)"
    A = AgentHinge(G.D / cfg.agent_labels, S.tab["names"], dev, G.AGENT["margin"]) if k in ("b", "x") else None
    perm = None
    if k == "x":                                                                  # geo_x's fixed derangement of every data dir, as global store rows
        ps, off = [], 0
        for d in cfg.data:
            p = np.load(G.MEM / "geo_x" / f"{d}.perm.npy")
            ps.append(p + off)
            off += len(p)
        perm = np.concatenate(ps)
        assert len(perm) == S.n and not (S.tab["log"][perm] == S.tab["log"]).any()
    return OnlineMem(k, hinge, A, S.fut, dev, perm=perm, init=cfg.mem_init)


def finish(om, model, S, dv_rows, W, rear, hinge, run, tag, ckpt_dir):
    """After training: tokenizer weights, dev diagnostics, the navtest bank of the tag (tab order; kind x: geo_x's navtest permutation)."""
    import torch
    dev = S.ego.device
    torch.save(om.net.state_dict(), ckpt_dir / "geotok.pt")
    model.eval()
    with torch.no_grad():
        pi = torch.as_tensor(S.pi, device=dev)
        r = torch.as_tensor(dv_rows, device=dev)
        mis = r[torch.as_tensor(G.derange(S.tab["log"][dv_rows], np.random.default_rng(1)), device=dev)]

        def read(mode):
            ade, out, rms = [], [], []
            for i in range(0, len(r), 128):
                q = r[i:i + 128]
                tok = om[mis[i:i + 128] if mode == "mismatched" else q]
                tok = tok.half().float() if mode == "fp16" else tok
                mask = torch.zeros(len(q), 1, dtype=torch.bool, device=dev) if mode == "masked" else None
                p = model(S.front[q], S.ego[q], S.tc[q], tok, mask).float()[:, pi].view(-1, 33, 15)
                x, y, psi = rear(p, S.cam_x[q], W)
                ade.append(torch.hypot(x - S.fut[q][..., 0], y - S.fut[q][..., 1]).mean(1)[S.has_fut[q]])
                out.append((hinge.margins(x, y, psi, q).amin(1) < 0)[hinge.ok[q]].float())
                rms.append(tok.pow(2).mean((1, 2)))
            return float(torch.cat(ade).mean()), float(torch.cat(out).mean()), float(torch.cat(rms).mean().sqrt())
        dg = {}
        for mode in ("on", "masked", "mismatched", "fp16"):
            dg[f"e2e_dev_ade_{mode}"], dg[f"e2e_dev_out_{mode}"], rms = read(mode)
            if mode == "on":
                dg["e2e_token_rms"] = rms
        T = G.Src("navtest", dev)
        assert om.kind != "p" or bool(T.has_fut.all()), "the path field needs a logged future on every navtest row"
        perm = np.load(G.MEM / "geo_x" / f"{G.TEST}.perm.npy") if om.kind == "x" else None
        te = OnlineMem(om.kind, T.H, T.A, T.fut, dev, perm=perm, net=om.net)
        n = len(T.names)
        assert (T.names == np.load(G.CR / G.TEST / "tab.npz")["names"]).all() and (perm is None or not (T.log[perm] == T.log).any())
        bank = torch.cat([te[torch.arange(i, min(i + 512, n), device=dev)].half() for i in range(0, n, 512)]).cpu().numpy()
        assert bank.shape == (n, G.NTOK, G.DTOK) and np.isfinite(bank).all()
        G._save(f"ge_{tag}", G.TEST, bank)
        dg["e2e_bank_rms"] = float(np.sqrt((bank[::6].astype(np.float32) ** 2).mean()))
    run.info(f"geo-e2e {om.kind}: " + json.dumps(dg) + f"; navtest bank {bank.shape} -> {G.MEM / f'ge_{tag}'}")
    run.summary.update(dg)


# ---------------------------------------------------------------- report
def done(arm, seed):
    import glob
    fs = sorted(glob.glob(str(G.D / "runs" / "op_parity" / f"train-{G.STEM[arm]}-F-s{seed}" / "*" / "DONE")))
    return json.loads(open(fs[-1]).read()) if fs else {}


def cmd_report(a):
    import pandas as pd
    import turn_oracle as TO
    from jevdrive.run import Run
    out = OUT / "report"
    out.mkdir(parents=True, exist_ok=True)
    with Run("op_parity", "geo-e2e-report", seed=0, config=vars(a)) as run:
        F, sets, tok, log = G.frames(a.seeds, ARMS)
        per_seed = {s: G.frames([s], ARMS)[0] for s in a.seeds}
        Dt = TO.Data(a.replays)
        assert (Dt.tok == tok).all()
        for arm in ARMS:
            fs = [Dt.one(G.spec(arm, s))[0] for s in a.seeds]
            g = sum(fs) / len(fs)
            for c in ("inside-cut %", "cannot-make-turn %", "other DAC fail %", "raw-plan departure %"):
                F[arm][c] = g[c].to_numpy()
            assert np.allclose(g["DAC fail %"].to_numpy(), F[arm]["DAC fail %"].to_numpy())
        SUB = [c for c in ("NC", "DAC", "DDC", "TLC", "EP", "TTC", "LK", "HC", "EC") if c in F["H0"]]
        BK = ("< 5 deg", "5-20 deg", "20-45 deg", "> 45 deg")
        TABLE = ([("EPDMS", s) for s in ("all", *BK)] + [("DAC fail %", s) for s in ("all", "> 20 deg", "20-45 deg", "> 45 deg")] +
                 [("inside-cut %", "> 45 deg"), ("cannot-make-turn %", "> 45 deg"), ("other DAC fail %", "> 45 deg"), ("raw-plan departure %", "> 45 deg"),
                  ("inside-cut %", "> 20 deg"), ("cannot-make-turn %", "> 20 deg"), ("NC+TTC fail %", "all"), ("NC fail %", "all"), ("TTC fail %", "all"),
                  ("NC+TTC fail %", "< 5 deg"), ("NC+TTC fail %", "> 20 deg")] +
                 [(c, "all") for c in SUB] + [(c, s) for s in BK for c in ("NC", "DAC", "EP", "TTC", "LK") if c in F["H0"]])
        rows = []
        for col, sn in TABLE:
            m = sets[sn]
            for arm in ARMS:
                r = dict(metric=col, stratum=sn, n=int(m.sum()), arm=arm, value=float(F[arm][col].to_numpy()[m].mean()))
                for ref, who in REFS.items():
                    if arm in who and arm != ref:
                        p = G.paired(F[arm], F[ref], col, m, log)
                        r |= {f"diff_vs_{ref}": p["diff"], f"lo_vs_{ref}": p["lo"], f"hi_vs_{ref}": p["hi"]}
                rows.append(r)
        T = pd.DataFrame(rows)
        T.to_csv(out / "arms.csv", index=False)
        get = lambda arm, col, sn, ref: T[(T.arm == arm) & (T.metric == col) & (T.stratum == sn)].iloc[0][[f"diff_vs_{ref}", f"lo_vs_{ref}", f"hi_vs_{ref}"]].tolist()  # noqa: E731
        dg = {arm: [done(arm, s) for s in a.seeds] for arm in ("H0", "JB", "JX", "JW", "JP", "GB", "GX")}
        ade = {arm: [d.get("dev_ade") for d in v] for arm, v in dg.items()}

        def judge(arm):                                                           # the registered lines and the channel-read rule, for JB (and JW as a variant)
            g, va, vb = get(arm, "EPDMS", "all", "H0"), get(arm, "EPDMS", "all", "JX"), get(f"{arm}:off", "EPDMS", "all", arm)
            line = "geometry is enough" if (g[0] >= 0.7 and g[1] > 0.3) else "negative" if g[0] < 0.3 else "partial"
            return dict(minus_H0_EPDMS=g, line=line, minus_JX_EPDMS=va, masked_minus_on_EPDMS=vb, channel_read=bool(va[1] > 0 and vb[2] < 0),
                        per_seed_minus_H0=[float(per_seed[s][arm]["EPDMS"].mean() - per_seed[s]["H0"]["EPDMS"].mean()) for s in a.seeds],
                        per_seed_minus_JX=[float(per_seed[s][arm]["EPDMS"].mean() - per_seed[s]["JX"]["EPDMS"].mean()) for s in a.seeds])
        jb, jw, jp = judge("JB"), judge("JW"), get("JP", "EPDMS", "all", "H0")
        absorbs = bool(jp[1] > 0 and np.mean(ade["JP"]) <= 0.8 * np.mean(ade["H0"]))
        verdict = jb["line"]
        if verdict == "negative":
            verdict += (" and the channel was read: geometry is not what this policy lacks (the task-book sense)" if jb["channel_read"] else
                        ", the channel was not read although the same set-up absorbs a path field: at pilot scale true geometry offers this policy "
                        "nothing it takes up" if absorbs else ", the channel was not read and the set-up was not shown to absorb anything: ceiling unmeasured")
        elif not jb["channel_read"]:
            verdict += " on the number, but not judged on the line: the channel-read rule is not met, the gain is not attributable to the geometry content"
        vd = dict(registered_lines=LINES, verdict=verdict, seeds=list(a.seeds), JB=jb, JW_variant=jw,
                  JP_positive_control=dict(minus_H0_EPDMS=jp, dev_ade=ade["JP"], absorbs=absorbs),
                  frozen_reference=dict(GB_minus_H0_EPDMS=get("GB", "EPDMS", "all", "H0"), GB_minus_GX_EPDMS=get("GB", "EPDMS", "all", "GX"),
                                        JB_minus_GB_EPDMS=get("JB", "EPDMS", "all", "GB"), JW_minus_GB_EPDMS=get("JW", "EPDMS", "all", "GB")),
                  dev_ade_m=ade, per_seed_EPDMS={arm: [float(per_seed[s][arm]["EPDMS"].mean()) for s in a.seeds] for arm in ARMS},
                  e2e_dev={arm: [{k: v for k, v in d.items() if k.startswith("e2e_")} for d in dg[arm]] for arm in ("JB", "JX", "JW", "JP")},
                  warm_start_tokenizer=json.loads((OUT / "tok_b_init.json").read_text()).get("head") if (OUT / "tok_b_init.json").exists() else None,
                  replay_checks=Dt.check)
        (out / "verdict.json").write_text(json.dumps(vd, indent=1, default=float))
        tk = pd.DataFrame({"token": tok, "log": log})
        for arm in ARMS:
            for c in ("EPDMS", "DAC fail %", "NC+TTC fail %", "inside-cut %", "cannot-make-turn %"):
                tk[f"{arm} | {c}"] = F[arm][c].to_numpy()
        tk.to_csv(out / "tokens.csv.gz", index=False, float_format="%.4g")
        L = [f"seeds {a.seeds}; values are per-token seed means x 100; diffs with 95% CI (log-cluster paired bootstrap, B {G.NB}, 12 146 navtest tokens). "
             "H0 no memory (GH0), JB joint tokenizer on true SDF + agents, JX the same on rasters shuffled across logs, JW JB from the pre-trained "
             "tokenizer, JP joint tokenizer on the logged-path field (label leak, positive control), :off memory masked at test, GB / GX decision 197's "
             "frozen-tokenizer arms.\n"]
        for ref, who in REFS.items():
            cols = [ref] + [w for w in who if w != ref]
            L.append(f"\n**diff vs {ref}**\n\n| metric | stratum | n | " + " | ".join(cols) + " |\n|:--|:--|--:|" + ":--|" * len(cols))
            for col, sn in TABLE:
                q = T[(T.metric == col) & (T.stratum == sn)].set_index("arm")
                cells = [f"{q.loc[c, 'value']:.2f}" + ("" if c == ref else f" ({q.loc[c, f'diff_vs_{ref}']:+.2f} [{q.loc[c, f'lo_vs_{ref}']:+.2f}, {q.loc[c, f'hi_vs_{ref}']:+.2f}])")
                         for c in cols]
                L.append(f"| {col} | {sn} | {q.n.iloc[0]} | " + " | ".join(cells) + " |")
        L.append("\n```json\n" + json.dumps({k: v for k, v in vd.items() if k != "replay_checks"}, indent=1, default=float) + "\n```")
        (out / "tables.md").write_text("\n".join(L) + "\n")
        run.info("\n".join(L))
        run.summary.update(verdict=verdict, JB_minus_H0=jb["minus_H0_EPDMS"], channel_read=jb["channel_read"], absorbs=absorbs, out=str(out))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("report")
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1])
    p.add_argument("--replays", nargs="+", default=["geo_s0", "geo_e2e"])
    a = ap.parse_args()
    {"report": cmd_report}[a.cmd](a)
