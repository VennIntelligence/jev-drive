#!/usr/bin/env python3
"""BODY1 arm 4.2, everything that is fixed before a closed-loop score is read (plans/2026-10-10-body1-prereg.md, section 4.2 + Amendment 3).

  cand     (GPU + CPU) own-plan decisions of body1-hold-logs (on-log + ot1 + yr1 states, both student seeds): the plan and its lateral-ramp
           candidates (serve_body.ramps, the serving code) scored by the two full-scale `step` heads, and their true labels by lib/sweep.py
           -> $DATA_DIR/runs/body1/replan/hold/cand.npz
  gate     (CPU) the offline gate on cand.npz: the registered choice rule (serve_body.pick, memoryless: one decision at a time) over the
           threshold grid, the selection, and the tables of the chosen setting -> replan/hold/{grid.csv, chosen.json, by_*.csv, shift.csv}
  replay   (GPU, JEV_REPLAN=1) plumbing on COL1's serialized driver messages (the G2 replay harness; navtest logs: nothing is chosen here):
           the plan before the shift against SWV1's stored replay (bit for bit), the hook's plan logits against the gate's stored predictions,
           served = ramps(plan)[choice], side memory respected, latency, and one descriptive count of the registered rule's choices
           -> replan/replay/on.json, on_decisions.csv. (Switch off: `bd1_stop.py replay` without JEV_STOP / JEV_REPLAN.)

Truth of one (state, query), from lib/sweep.py:   agent = a_hit or a_rear (any contact with an object box, also one the label excludes as a
rear-end by a faster object: a lateral move into a faster neighbour must count);   boundary = minimum drivable margin < -0.20 m (Amendment 1
item 2), not judged for states already outside at t = 0;   contact = agent or boundary.
Decision sets (own plan):   clean = no agent contact and margin >= -0.20 m;   true contact = a_hit or boundary (rear-only plans are in neither set).
  (a) resolve rate = share of the flagged true contacts whose served candidate has no contact
  (b) harm rate    = share of the clean decisions whose served candidate has a contact                      LINE: <= 0.2 % and 5 x harm <= resolved
  (c) served |a| on clean decisions                    (d) the same per user class, the > 45 deg bucket, state family and ego speed
Grid: flag thresholds FA / FB = the 98 / 99 / 99.5th percentile of the clean decisions' plan logit (FB also "off": boundary flags do not
trigger), clear thresholds CA / CB = the 50 / 80 / 90 / 95th percentile of the same distributions (one percentile for both heads).
Selection, written before the grid was computed: among the settings that meet the line (b), the largest (resolved - 5 x harm); ties -> fewer
clean decisions re-planned. No setting meets it -> the arm stops here.
"""
import sys as _sys, pathlib as _pl  # noqa: E401
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1] / "lib"))
import argparse  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import pickle  # noqa: E402
import tempfile  # noqa: E402

import numpy as np  # noqa: E402

import b1 as B  # noqa: E402

PF, PFB, PC = (98.0, 99.0, 99.5), (None, 98.0, 99.0, 99.5), (50.0, 80.0, 90.0, 95.0)
HARM_LINE, FACTOR = 0.002, 5
KEYS = ("a_hit", "a_rear", "a_clr", "b_margin", "b_t0", "b_cov")


def out_dir(name):
    d = B.root() / "replan" / name
    d.mkdir(parents=True, exist_ok=True)
    return d


# ---------------------------------------------------------------- candidates: scores and labels
def _lab(u):
    import sweep as SW
    q, off, gi = u
    L = B.labels()
    z = SW.labels(q, off, np.asarray(L["box"][gi]), np.asarray(L["valid"][gi]), np.asarray(L["cls"][gi]), np.asarray(L["sdf"][gi]))
    return {k: z[k] for k in KEYS}


def cmd_cand(a):
    import torch
    import bd1_gate as G
    import contact_head as C
    import serve_body as SB
    from jevdrive import par
    from jevdrive.common import n_cpus
    from jevdrive.run import Run
    dev = torch.device("cuda")
    with Run("body1", "replan/cand", config=vars(a)) as run:
        R, sp = G.hold_rows()
        for s in sp:
            run.use_split(s)
        n = len(R["gi"]) if not a.limit else a.limit
        own = R["q"][:n, list(C.OWN)].astype(np.float64)                                    # (n, 2, 8, 3)
        cand, ok = SB.ramps(own)                                                            # (n, 2, m, 8, 3), (n, 2, m)
        Q = np.concatenate([own[:, :, None], cand], 2).astype(np.float32)                   # (n, 2, 1 + m, 8, 3)
        m1 = Q.shape[2]
        Qf = Q.reshape(n, 2 * m1, 8, 3)
        off = R["off"][:n]
        # labels (CPU, parallel over chunks of states)
        units = [(Qf[i:i + 64], off[i:i + 64], R["gi"][i:i + 64]) for i in range(0, n, 64)]
        res = par.pmap(_lab, units, run=run, workers=min(n_cpus(), 48))
        res.raise_if_failed()
        lab = {k: np.concatenate([r[k] for r in res.values]).reshape(n, 2, m1) for k in KEYS}
        assert np.array_equal(lab["a_hit"][:, :, 0], R["a_hit"][:n, :2]) and np.allclose(lab["b_margin"][:, :, 0], R["b_margin"][:n, :2]), "the plan's labels differ from the stored rows"
        # scores (GPU)
        V = C.load_tokens(R["src"], len(R["gi"]), dev)[:n]
        Qp = torch.as_tensor(Qf).to(dev)
        Z = []
        for i, c in enumerate(SB.CKPTS):
            net, ck = C.load_ckpt(C.sroot() / c, dev)
            out, _ = C.predict(net, V, torch.as_tensor((R["ego"][:n] - ck["emu"]) / ck["esd"]).to(dev), Qp, batch=256)
            ref = np.load(C.sroot() / "pred" / f"step-s{i}.npz")["hold_out"][:n, :2, :2]
            d = np.abs(out.reshape(n, 2, m1, -1)[:, :, 0, :2] - ref).max()
            run.info("seed %d: plan logits against the gate's stored predictions, max |difference| %.4f", i, d)
            assert d < 0.05, d
            Z.append(out.reshape(n, 2, m1, -1)[..., :2].astype(np.float32))
            del net
        Z = np.mean(Z, 0)
        f = out_dir("hold") / ("cand.npz" if not a.limit else f"cand-first{a.limit}.npz")
        np.savez(f, za=Z[..., 0], zb=Z[..., 1], ok=ok, A=SB.A_RP, gi=R["gi"][:n], log=R["log"][:n], fam=np.array(R["fams"])[R["fam"][:n]], qok=R["qok"][:n, :2],
                 arc4=np.hypot(np.diff(np.concatenate([np.zeros((n, 2, 1, 2)), own[..., :2]], 2)[..., 0], axis=-1), np.diff(np.concatenate([np.zeros((n, 2, 1, 2)), own[..., :2]], 2)[..., 1], axis=-1)).sum(-1), **lab)
        run.summary.update(states=n, decisions=2 * n, candidates=int(ok.sum()), out=str(f))
        run.info(str(run.summary))


# ---------------------------------------------------------------- the gate
def truth(z):
    """cand.npz -> dict of bool arrays: agent (n, 2, 1 + m), bound, contact; clean, pos (n, 2) of the own plan; judged (n, 2)."""
    bjudge = ~z["b_t0"][:, :, :1] & (z["b_margin"][:, :, :1] < 90)
    agent = z["a_hit"] | z["a_rear"]
    bound = bjudge & (z["b_margin"] < -0.20) & (z["b_margin"] < 90)
    con = agent | bound
    own_b = bound[:, :, 0]
    return dict(agent=agent, bound=bound, contact=con, clean=z["qok"] & ~agent[:, :, 0] & ~own_b, pos=z["qok"] & (z["a_hit"][:, :, 0] | own_b),
                pos_a=z["qok"] & z["a_hit"][:, :, 0], pos_b=z["qok"] & own_b & ~z["a_hit"][:, :, 0])


def serve(z, thr):
    """The registered rule, one decision at a time (no memory) -> choice (n, 2) index into A (-1 = plan), flag_a, flag_b, has_clear."""
    import serve_body as SB
    n = len(z["za"])
    j, fa, fb, has = SB.pick(z["za"].reshape(2 * n, -1), z["zb"].reshape(2 * n, -1), z["ok"].reshape(2 * n, -1), thr, 0, z["A"])
    return j.reshape(n, 2), fa.reshape(n, 2), fb.reshape(n, 2), has.reshape(n, 2)


def tally(z, T, thr, m=None):
    """Counts of one setting on the decisions m (n, 2) (default all)."""
    j, fa, fb, has = serve(z, thr)
    m = np.ones_like(fa) if m is None else m
    fl, rp = fa | fb, j >= 0
    sc = np.take_along_axis(T["contact"], (j + 1)[..., None], 2)[..., 0]                    # contact of what is served (index 0 = the plan)
    sa = np.take_along_axis(T["agent"], (j + 1)[..., None], 2)[..., 0]
    cl, po = T["clean"] & m, T["pos"] & m
    harm, res = cl & rp & sc, po & fl & rp & ~sc
    absa = np.where(rp, np.abs(z["A"])[np.maximum(j, 0)], 0.0)
    r = dict(clean=int(cl.sum()), pos=int(po.sum()), pos_agent=int((T["pos_a"] & m).sum()), pos_boundary=int((T["pos_b"] & m).sum()),
             clean_flagged=int((cl & fl).sum()), clean_replanned=int((cl & rp).sum()), harm=int(harm.sum()), harm_agent=int((harm & sa).sum()), harm_boundary=int((harm & ~sa).sum()),
             pos_flagged=int((po & fl).sum()), pos_replanned=int((po & rp).sum()), resolved=int(res.sum()), resolved_agent=int((res & T["pos_a"]).sum()), resolved_boundary=int((res & T["pos_b"]).sum()),
             pos_flagged_none_clear=int((po & fl & ~rp).sum()), clean_flagged_none_clear=int((cl & fl & ~rp).sum()))
    r |= dict(harm_rate=r["harm"] / max(r["clean"], 1), resolve_rate=r["resolved"] / max(r["pos_flagged"], 1), resolved_of_all_pos=r["resolved"] / max(r["pos"], 1),
              clean_replanned_rate=r["clean_replanned"] / max(r["clean"], 1), clean_flag_rate=r["clean_flagged"] / max(r["clean"], 1),
              clean_shift_mean_m=float(absa[cl & rp].mean()) if (cl & rp).any() else 0.0)
    return r, (j, fl, rp, sc, absa)


def cmd_gate(a):
    import pandas as pd
    import contact_head as C
    import serve_body as SB
    from jevdrive.data import splits
    from jevdrive.run import Run
    with Run("body1", "replan/gate", config=vars(a)) as run:
        run.use_split(splits.load(B.HOLD))
        o = out_dir("hold")
        z = dict(np.load(o / "cand.npz"))
        T = truth(z)
        za0, zb0, cl = z["za"][:, :, 0], z["zb"][:, :, 0], T["clean"]
        pa, pb = (lambda p: float(np.percentile(za0[cl], p))), (lambda p: float(np.percentile(zb0[cl], p)))
        rows = []
        for fa in PF:
            for fb in PFB:
                for c in PC:
                    thr = (pa(fa), pb(fb) if fb else np.inf, pa(c), pb(c))
                    r, _ = tally(z, T, thr)
                    rows.append(dict(pF_agent=fa, pF_boundary=fb or "off", pC=c, FA=thr[0], FB=thr[1], CA=thr[2], CB=thr[3], **r,
                                     line_b=bool(r["harm_rate"] <= HARM_LINE and FACTOR * r["harm"] <= r["resolved"]), objective=r["resolved"] - FACTOR * r["harm"]))
        G = pd.DataFrame(rows)
        G.to_csv(o / "grid.csv", index=False)
        show = ["pF_agent", "pF_boundary", "pC", "clean_flagged", "clean_replanned", "harm", "harm_rate", "pos_flagged", "resolved", "resolve_rate", "resolved_of_all_pos", "clean_shift_mean_m", "line_b", "objective"]
        run.info("clean %d, true contacts %d (agent %d, boundary only %d)\n%s", r["clean"], r["pos"], r["pos_agent"], r["pos_boundary"], G[show].round(4).to_string(index=False))
        ok = G[G.line_b]
        if not len(ok):
            (o / "chosen.json").write_text(json.dumps(dict(passed=False)))
            raise SystemExit("offline gate FAILED: no setting meets line (b)")
        best = ok.sort_values(["objective", "clean_replanned"], ascending=[False, True]).iloc[0]
        thr = (float(best.FA), float(best.FB), float(best.CA), float(best.CB))
        if a.thr:                                                                           # tables of an explicit setting (the registered constants)
            thr = tuple(float(x) for x in a.thr)
        r, (j, fl, rp, sc, absa) = tally(z, T, thr)
        # (d) per class, > 45 deg, state family, ego speed
        D = pd.read_parquet(C.sroot() / "report" / "own.parquet")
        n = len(z["gi"])
        assert len(D) == 2 * n
        cls, gt45, speed = (D[k].to_numpy().reshape(n, 2) for k in ("cls", "gt45", "speed"))
        fam = np.repeat(z["fam"][:, None], 2, 1)
        subs = [("pooled", np.ones((n, 2), bool))] + [(f"class {c}" if c else "other", cls == c) for c in (1, 2, 3, 0)] + [("> 45 deg", gt45)] + \
               [(f"states {f}", fam == f) for f in ("log", "ot1", "yr1")] + [(f"speed {lo}-{hi} m/s", (speed >= lo) & (speed < hi)) for lo, hi in ((0, 1), (1, 3), (3, 6), (6, 10), (10, 99))]
        by = pd.DataFrame([dict(subset=s, **tally(z, T, thr, m)[0]) for s, m in subs])
        by.to_csv(o / "by_subset.csv", index=False)
        # (c) served shift on clean decisions and on true contacts; what an unresolved flagged contact was served
        sh = []
        for name, m in (("clean", T["clean"]), ("true contact", T["pos"])):
            for v in (0.0, *np.unique(np.abs(z["A"]))):
                sh.append(dict(decisions=name, shift_m=v, n=int((m & (absa == v)).sum()), share=float((m & (absa == v)).sum() / m.sum()), served_contact=int((m & (absa == v) & sc).sum())))
        sh = pd.DataFrame(sh)
        sh.to_csv(o / "shift.csv", index=False)
        side = np.where(rp, np.sign(z["A"])[np.maximum(j, 0)], 0)
        # precision of the flagged decisions without a clear candidate (what a fallback stop would act on): agent-flagged only
        fa_ = za0 >= thr[0]
        nc = fa_ & ~rp & z["qok"]
        extra = dict(none_clear_agent_flag=int(nc.sum()), none_clear_agent_flag_true_agent=int((nc & z["a_hit"][:, :, 0]).sum()),
                     replanned_left=int((side > 0).sum()), replanned_right=int((side < 0).sum()),
                     pos_unflagged=int((T["pos"] & ~fl).sum()), pos_flagged_served_contact=int((T["pos"] & fl & rp & sc).sum()),
                     candidates_per_decision_mean=float(z["ok"].sum(-1).mean()), decisions_without_candidates=int((z["ok"].sum(-1) == 0).sum()),
                     clean_harm_strict_margin0=int((T["clean"] & rp & (np.take_along_axis(z["b_margin"], (j + 1)[..., None], 2)[..., 0] < 0) & ~sc).sum()),
                     logs=len(set(z["log"].tolist())), logs_with_harm=len(set(z["log"][(T["clean"] & rp & sc).any(1)].tolist())), logs_with_resolved=len(set(z["log"][(T["pos"] & fl & rp & ~sc).any(1)].tolist())))
        ch = dict(passed=True, thr=dict(FA=thr[0], FB=thr[1], CA=thr[2], CB=thr[3]), selected=best[["pF_agent", "pF_boundary", "pC"]].to_dict(), explicit=bool(a.thr), **r, **extra)
        (o / "chosen.json").write_text(json.dumps(ch, indent=1, default=float))
        run.info("chosen %s\n%s\n%s\n%s", json.dumps(ch, default=float), by.round(4).to_string(index=False), sh.round(4).to_string(index=False), SB.A_RP)
        run.summary.update(ch)


# ---------------------------------------------------------------- replay: plumbing with the switch on
class Ctx:
    def abort(self, code, msg):
        raise RuntimeError(f"{code}: {msg}")


def cmd_replay(a):
    import csv
    alp = B.REPO / "experiments" / "alpasim"
    _sys.path[:0] = [str(alp / "scripts"), str(alp / "lib")]
    import torch
    import sh30_driver as D
    from jevdrive.run import Run
    SB = D.BD
    assert SB is not None and SB.RP, "run with JEV_REPLAN=1"
    LD, stored_dir = B.root().parent / "alpasim/col1/lead", B.root().parent / "alpasim/swv1/replay"
    with Run("body1", "replan/replay-on", config=vars(a)) as run:
        spec = json.loads((LD / "spec.json").read_text())
        pb, ctx, res, caught, dec = D.egodriver_pb2, Ctx(), {}, [], []
        SB.load("cuda")
        apply0 = SB.apply

        def apply(lock, o, v0, *x):
            info = apply0(lock, o, v0, *x)
            caught.append((info, np.asarray(o.get("poses_plan", o["poses"])).copy(), np.asarray(o["poses"]).copy()))
            return info
        SB.apply = apply
        pred = [np.load(B.root() / "s0" / "pred" / f"step-s{i}.npz") for i in (0, 1)]
        for g in a.groups:
            core = D.C.Core(spec[g]["tag"], "cuda", "backwarp")
            drv = D.Driver(core, _pl.Path(tempfile.mkdtemp()), 0, False)
            stored = pickle.load(open(stored_dir / f"{g}.pkl", "rb"))
            scenes = spec[g]["scenes"][:a.limit] if a.limit else spec[g]["scenes"]
            d_pose, recs, n0 = [], [], len(caught)
            for scene in run.tqdm(scenes, desc=g):
                uuid, k = None, 0
                for kind, raw in pickle.load(open(LD / "msgs" / g / f"{scene}.pkl", "rb")):
                    if kind == "driver_session_request":
                        req = pb.DriveSessionRequest.FromString(raw)
                        uuid = req.session_uuid
                        drv.start_session(req, ctx)
                    elif kind == "driver_camera_image":
                        drv.submit_image_observation(pb.RolloutCameraImage.FromString(raw), ctx)
                    elif kind == "driver_ego_trajectory":
                        drv.submit_egomotion_observation(pb.RolloutEgoTrajectory.FromString(raw), ctx)
                    elif kind == "route_request":
                        drv.submit_route(pb.RouteRequest.FromString(raw), ctx)
                    elif kind == "driver_request":
                        drv.drive(pb.DriveRequest.FromString(raw), ctx)
                        d_pose.append(float(np.abs(caught[-1][1] - stored[scene][k]["poses"]).max()))
                        recs.append((scene, k))
                        k += 1
                drv.sessions.pop(uuid, None)
            C_ = caught[n0:]
            assert len(C_) == len(recs)
            z = np.load(B.root() / "g2" / "dump" / f"{g}.npz")
            key = {(str(sc), int(kk)): i for i, (sc, kk) in enumerate(zip(z["scene"], z["k"]))}
            idx = np.array([key[x] for x in recs])
            ref = np.mean([p[f"g2_{g}"][idx, :2] for p in pred], 0)                         # (n, 2) stored two-seed mean agent / boundary logits of the plan
            got = np.array([[c[0]["za"][0], c[0]["zb"][0]] for c in C_])
            bad, flips, prev = 0, 0, {}
            for (scene, k), (info, pl, sv) in zip(recs, C_):
                cand, ok = SB.ramps(pl)
                if info["reason"] == "replan":
                    j = int(np.flatnonzero(SB.A_RP == info["a"])[0])
                    FA, FB, CA, CB = SB.THR_RP
                    good = ok[j] and info["za"][j + 1] < CA and info["zb"][j + 1] < CB and np.allclose(sv, cand[j], atol=1e-5) and k > 0 and (info["side"] in (0, int(np.sign(info["a"]))))
                    # no eligible clear candidate of a smaller size
                    sm = [i for i in range(len(SB.A_RP)) if abs(SB.A_RP[i]) < abs(info["a"]) - 1e-9 and ok[i] and info["za"][i + 1] < CA and info["zb"][i + 1] < CB and info["side"] in (0, int(np.sign(SB.A_RP[i])))]
                    bad += int(not good or bool(sm))
                    p = prev.get(scene)
                    flips += int(p is not None and p[1] != np.sign(info["a"]) and k - p[0] <= SB.IDLE)
                    prev[scene] = (k, np.sign(info["a"]))
                else:
                    bad += int(not np.array_equal(pl, sv))
                dec.append(dict(group=g, scene=scene, k=k, reason=info["reason"], flag=info["flag"], a=info["a"], side=info["side"], za_plan=info["za"][0], zb_plan=info["zb"][0], v0=info["v0"],
                                n_cand=int(sum(info["ok"])), ms=info["ms"]))
            rs = np.array([c[0]["reason"] for c in C_])
            ms = np.array([c[0]["ms"] for c in C_])
            sc_rp = {s for (s, _), c in zip(recs, C_) if c[0]["reason"] == "replan"}
            r = dict(n=len(recs), scenes=len(scenes), max_abs_plan_diff_m=float(np.max(d_pose)), logit_max_abs_diff=float(np.abs(got - ref).max()), logit_mean_abs_diff=float(np.abs(got - ref).mean()),
                     reasons={x: int((rs == x).sum()) for x in ("cold", "clean", "replan", "none_clear")}, replan_rate=float((rs == "replan").mean()), scenes_replanned=len(sc_rp),
                     flag_a=int(sum("a" in c[0]["flag"] for c in C_)), flag_b=int(sum("b" in c[0]["flag"] for c in C_)), checks_failed=bad, side_flips_within_idle=flips,
                     shift_abs_mean=float(np.mean([abs(c[0]["a"]) for c in C_ if c[0]["reason"] == "replan"] or [0])), hook_ms_p50=float(np.median(ms)), hook_ms_p90=float(np.quantile(ms, 0.9)), hook_ms_max=float(ms.max()))
            run.info("%s: %s", g, r)
            res[g] = r
            del core, drv
            torch.cuda.empty_cache()
        o = out_dir("replay")
        (o / "on.json").write_text(json.dumps(res, indent=1))
        with (o / "on_decisions.csv").open("w", newline="") as fh:
            w = csv.DictWriter(fh, list(dec[0]))
            w.writeheader(), w.writerows(dec)
        run.summary.update(res)
        if any(v["max_abs_plan_diff_m"] > 0 or v["checks_failed"] or v["side_flips_within_idle"] for v in res.values()):
            raise SystemExit(f"replay gate FAILED: {res}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("cand")
    p.add_argument("--limit", type=int, default=0)
    p = sub.add_parser("gate")
    p.add_argument("--thr", nargs=4, help="FA FB CA CB: tables of this setting instead of the selected one")
    p = sub.add_parser("replay")
    p.add_argument("--groups", nargs="*", default=["P2H10-F-s0", "P2H10-F-s1", "ctrl"]), p.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    {"cand": cmd_cand, "gate": cmd_gate, "replay": cmd_replay}[a.cmd](a)


if __name__ == "__main__":
    main()
