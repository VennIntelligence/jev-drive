#!/usr/bin/env python3
"""LONG-PROBE PAI rows (prereg section 10, descriptive): LOWDIAG's stored PAI rollouts (seed 0, the 32 scenes whose driver messages were
kept: $DATA_DIR/runs/lowboard_diag/pai/x/ab_*_s0/msgs) replayed open loop through the real PAI driver class, as
experiments/lowboard_diag/scripts/lbd_pai_replay.py does (serving switches off, the logged state at every decision), here storing what the
ladder's heads read: the view_39 tokens of the newest 8 valid slots, the ego vector with the fed command, the served plan, and the
student's native outputs (lead, lead_prob, meta, action; select_4 / mean) from the same forward.

  replay  (GPU, pool) -> read/pai/dump/<scene>.npz
  score   (GPU, pool) the ladder's B / C / A logits (rungs blind / vis / nat, both seeds) -> read/pai/scores.npz
  report  (CPU) per LOWDIAG unit with an overrun (L5 -> B), a fast turn entry (L2 -> C) or a lead contact (L1 -> A; both are lateral in
          fact by lowboard_diag/results/pai.md): per rung (blind / vis / nat) the time before the event of the first decision
          flagged at the hold-log P1 2 % threshold of that label, and the flag rate on decisions of the other replayed scenes
          -> read/pai/lead.csv, flags.csv
  env ALPASIM_SRC=$DATA_DIR/third_party/alpasim SH30_TAG=P2H10-F-s0 $DATA_DIR/envs/op-train/bin/python experiments/long_contact/scripts/lc_pai.py replay
"""
import argparse
import os
import pickle
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve()
sys.path[:0] = [str(HERE.parents[1] / "lib"), str(HERE.parent), str(HERE.parents[2] / "alpasim" / "lib"), str(HERE.parents[2] / "alpasim" / "scripts")]
import lc as LC  # noqa: E402
import b1 as B  # noqa: E402

X = B.data_dir() / "runs/lowboard_diag/pai/x"
UNITS = B.REPO / "experiments/lowboard_diag/results/pai/units.csv"
RUNGS = ("blind", "vis", "nat")


def out_dir():
    return LC.root() / "read" / "pai"


class Ctx:
    def abort(self, code, msg):
        raise RuntimeError(f"{code}: {msg}")


def cmd_replay(a):
    import torch
    import pai_driver as D
    from jevdrive import op_adapt as AA
    from jevdrive.run import Run
    import lc_native as NA
    with Run("long_contact", "read/pai-replay", config=vars(a)) as run:
        C, PC, I = D.C, D.PC, D.I
        PA, Z = C.PA, C.Z
        core = C.Core(a.tag, "cuda")
        A = core.A
        drv = D.Driver(core, Path(tempfile.mkdtemp()), "backwarp", 1, 0, False)
        sl, pb = core.model.net.slices, D.egodriver_pb2
        cols = np.concatenate([np.arange(sl[k].start, sl[k].stop) for k in NA.SLICES])
        cap, orig = {}, PC.plan

        def plan(core_, cur, valid, P, V, acc, cmd, cam_t, lht=False):
            cap.update(cur=cur, valid=valid, P=P, V=V, acc=acc, cmd=cmd, cam_t=cam_t, lht=lht)
            cap["o"] = orig(core_, cur, valid, P, V, acc, cmd, cam_t, lht)
            return cap["o"]
        PC.plan = plan
        stash, rb = {}, core.model.net.run_batched

        def tap(feeds, want, **kw):
            o = rb(feeds, AA.POLICY_OUT, **kw)
            stash.update(o)
            return {k: o[k] for k in want}

        def extra():
            cur, valid, cam_t = cap["cur"], cap["valid"], np.asarray(cap["cam_t"], np.float64)
            prev = np.concatenate([np.zeros((1,) + C.FRAME, np.uint8), cur[:-1]])
            with torch.no_grad():
                p, c = (torch.from_numpy(x[valid]).to(core.dev) for x in (prev, cur))
                H = core.model.net.run_batched(A.vision_feeds(p, c), ["view_39"])["view_39"].reshape(1, int(valid.sum()), *A.H_SHAPE)
                tc = torch.tensor([[0.0, 1.0] if cap["lht"] else [1.0, 0.0]], device=core.dev)
                ego = PA.ego_features(cap["P"], cap["V"], np.tile(np.asarray(cap["acc"], np.float32), (4, 1)), np.asarray(cap["cmd"], np.float32))
                core.model.net.run_batched = tap
                try:
                    o = core.model(H, torch.from_numpy(ego[None]).to(core.dev), tc).float()[0]
                finally:
                    core.model.net.run_batched = rb
                o = o.cpu().numpy()
                mu = o[sl["plan"].start:sl["plan"].start + 495].reshape(33, 15)
                poses = I.to_rear(mu[:, 0:3], mu[:, 11], I.T_IDXS, cam_t[:2], Z.T_OUT, "lever")
                h = H[0].cpu().numpy().astype(np.float16)
                n = min(8, len(h))
                tok = np.zeros((8,) + h.shape[1:], np.float16)
                tok[8 - n:] = h[-n:]
                return dict(tok=tok, valid=np.arange(8) >= 8 - n, ego=ego.astype(np.float32), poses=np.asarray(poses, np.float32), nat=o[cols].astype(np.float32),
                            hid=torch.cat([stash["select_4"].reshape(1, -1), stash["mean"].reshape(1, -1)], 1).float().cpu().numpy()[0].astype(np.float16))
        od = out_dir() / "dump"
        od.mkdir(parents=True, exist_ok=True)
        ctx = Ctx()
        files = sorted(f for s in sorted(X.glob("ab_*_s0")) for f in (s / "msgs").glob("*.pkl"))
        for f in run.tqdm(files, desc="scenes"):
            if (od / f"{f.stem}.npz").exists():
                continue
            R, k, uuid = {n: [] for n in ("now", "t0", "cmd", "run", "tok", "valid", "ego", "poses", "nat", "hid")}, 0, None
            for kind, raw in pickle.load(open(f, "rb")):
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
                    req = pb.DriveRequest.FromString(raw)
                    cap.clear()
                    try:
                        drv.drive(req, ctx)
                    except RuntimeError as e:
                        run.info("%s %d: drive failed: %s", f.stem, k, e)
                    if "o" in cap:
                        x = extra()
                        R["now"].append(int(req.time_now_us)), R["t0"].append(int(drv.sessions[uuid].anchor.timestamp_us))
                        R["cmd"].append(int(np.argmax(cap["cmd"]))), R["run"].append(cap["o"]["poses"])
                        for n in ("tok", "valid", "ego", "poses", "nat", "hid"):
                            R[n].append(x[n])
                    k += 1
            drv.sessions.pop(uuid, None)
            arr = {n: np.array(v) for n, v in R.items()}
            np.savez(od / f"{f.stem}.npz", stack=f.parent.parent.name, **arr)
            d = float(np.abs(arr["run"] - arr["poses"]).max()) if len(arr["run"]) else np.nan
            run.info("%s: %d decisions, served plan against this forward max |diff| %.4f", f.stem, len(arr["now"]), d)


def cmd_score(a):
    import torch
    import lc_head as H
    from jevdrive.run import Run
    dev = torch.device("cuda")
    with Run("long_contact", "read/pai-score", config=vars(a)) as run:
        res = {}
        for f in sorted((out_dir() / "dump").glob("*.npz")):
            z = np.load(f)
            if not len(z["now"]):
                continue
            V, vm = torch.as_tensor(z["tok"]).to(dev), torch.as_tensor(z["valid"]).to(dev)
            Qp = torch.as_tensor(z["poses"][:, None].astype(np.float32)).to(dev)
            arr = {}
            for rung in RUNGS:
                for s in (0, 1):
                    rdir = Path((LC.root() / "train" / "runs" / f"full-{rung}-s{s}.dir").read_text().strip())
                    for lab in LC.LONG:
                        net, ck = H.load_ckpt(rdir / "ckpt.pt", dev, lab)
                        D = dict(V=V, Qp=Qp, E=torch.as_tensor((z["ego"] - ck["emu"]) / ck["esd"]).to(dev))
                        if net.nat:
                            D["nat"] = torch.as_tensor(np.repeat(((z["nat"] - ck["nmu"]) / ck["nsd"])[:, None], 2, 1)).to(dev, torch.float16)
                            D["hid"] = torch.as_tensor(np.repeat(z["hid"][:, None], 2, 1)).to(dev, torch.float16)
                        arr[f"{rung}-s{s}|{lab}"] = H.predict(net, D, valid=vm)[:, 0, H.IDX[lab]].astype(np.float32)
            res[f.stem] = arr
        np.savez(out_dir() / "scores.npz", **{f"{sc}|{k}": v for sc, arr in res.items() for k, v in arr.items()})
        run.summary.update(scenes=len(res))


def cmd_report(a):
    import json
    import pandas as pd
    import col1_lib as CL
    from jevdrive.run import Run
    with Run("long_contact", "read/pai-report", config=vars(a)):
        thr = json.loads((LC.root() / "read" / "report" / "thresholds.json").read_text())
        S = np.load(out_dir() / "scores.npz")
        U = pd.read_csv(UNITS)
        T0 = {}
        for d in sorted(X.glob("ab_*_s0")):
            if (d / "logs.pkl").exists():
                for s, o in CL.load(d / "logs.pkl").items():
                    T0[s[7:15]] = float(o["actors"]["EGO"][0, 0])
        lead, flags = [], []
        for f in sorted((out_dir() / "dump").glob("*.npz")):
            z = np.load(f)
            scn = f.stem[7:15]
            t = (z["t0"].astype(float) - T0.get(scn, np.nan)) * 1e-6
            u = U[U.scene == scn]
            raw = str(u.raw.iloc[0]) if len(u) else ""
            tev = float(u.t_event.iloc[0]) if len(u) and np.isfinite(u.t_event.iloc[0]) else np.nan
            for lab, flag in (("B", "L5"), ("C", "L2"), ("A", "L1")):           # L1 = LOWDIAG's lead-contact units (lateral in fact, pai.md)
                for rung in RUNGS:
                    s = 0.5 * (S[f"{f.stem}|{rung}-s0|{lab}"] + S[f"{f.stem}|{rung}-s1|{lab}"])
                    th = thr[lab][rung]
                    if flag in raw.split("+") and np.isfinite(tev):
                        w = (t <= tev) & (t >= tev - 8.0)
                        fl = t[w & (s >= th)]
                        lead.append(dict(label=lab, scene=scn, flags=raw, t_event=tev, rung=rung, decisions_8s=int(w.sum()), flagged_8s=int((s[w] >= th).sum()),
                                         first_flag_s_before_event=float(tev - fl.min()) if len(fl) else np.nan, max_logit_4s=float(s[(t <= tev) & (t >= tev - 4)].max())
                                         if ((t <= tev) & (t >= tev - 4)).any() else np.nan))
                    else:
                        flags.append(dict(label=lab, scene=scn, flags=raw, rung=rung, decisions=len(s), flagged=int((s >= th).sum())))
        L, Fl = pd.DataFrame(lead), pd.DataFrame(flags)
        L.to_csv(out_dir() / "lead.csv", index=False)
        Fl.to_csv(out_dir() / "flags.csv", index=False)
        with pd.option_context("display.width", 220):
            print(L.to_string(index=False))
            print(Fl.groupby(["label", "rung"])[["decisions", "flagged"]].sum().assign(rate=lambda x: x.flagged / x.decisions).to_string())


if __name__ == "__main__":
    from jevdrive.run import cli_args
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("replay")
    p.add_argument("--tag", default=os.environ.get("SH30_TAG", "P2H10-F-s0"))
    cli_args(p)
    cli_args(sp.add_parser("score"))
    cli_args(sp.add_parser("report"))
    a = ap.parse_args()
    {"replay": cmd_replay, "score": cmd_score, "report": cmd_report}[a.cmd](a)
