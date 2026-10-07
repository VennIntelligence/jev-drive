"""Why WP2 under-launches from standstill on WOD-E2E val, and the recipe fix (plans/2026-10-08-wod-launch-prereg.md, results/wod_launch.md).

  label   (jevdrive env, one GPU) zero-shot Qwen3-VL-4B context labels of the 479 rater frames from the FRONT camera JPEGs
          (processed/wod_zeroshot/t2_jpg): light / stop sign / lead (two frames, t0 - 1 s and t0) / crossing; option scoring on the first
          token after a forced `ANSWER:` -> results/wod_launch/context_vlm.csv
  bias    (op-train env, CPU) intent_bias files of the ego-channel interventions on WP2 (rater frames) -> $DATA_DIR/runs/op_parity/wod/launch/
          bias-<tag>_<var>.npz; served by scripts/wod_zeroshot_openpilot.py --set rater --tag lx-<tag>_<var> ... --bias ...
  tok     (op-train env, one GPU) the training-protocol token path (image pair (f - 2, f), 9 slots at 0.2 s; wod_parity.py check) for shipped /
          WP1 / WP2: every even val frame at slot >= 9 with a logged future (part V) and the r2-dev rows + r2-train standstill rows of the
          train cache (part D) -> $DATA_DIR/runs/op_parity/wod/launch/tok_{val,dev,train_stop}.npz
  report / figs: scripts/wod_launch_report.py (jevdrive env, CPU)
"""
import sys as _sys, pathlib as _pl  # noqa: E401
_R = _pl.Path(__file__).resolve().parents[3]
_sys.path[:0] = [str(_R), str(_R / "lib"), str(_R / "scripts"), str(_R / "experiments/op_adapt_r2/lib"), str(_pl.Path(__file__).parent)]
import argparse, json, time  # noqa: E401,E402

import numpy as np  # noqa: E402

from jevdrive.common import data_dir  # noqa: E402

OUT = _R / "experiments/op_parity/results/wod_launch"
VARS = ("main", "zero", "biasmean", "biasresid", "cmd0", "acc0", "vx1", "vx3", "cv1", "cv3")
T_HIST = np.array([-1.5, -1.0, -0.5, 0.0])
TOK_ARMS = ("P0", "WP1-full-s0", "WP2-full-s0", "WP2-full-s1")
V_STOP = 0.5

ONE = "The image was taken by the front camera of a car (the ego vehicle)."
TWO = ("The two images were taken by the front camera of a car (the ego vehicle): image 1 one second before image 2. "
       "Answer for the moment of image 2.")
QS = {
    "light": (ONE, "Traffic light status controlling the ego vehicle lane ahead. Options:\n"
              "  red_or_yellow_for_ego: a red or yellow traffic light controls the ego lane ahead\n"
              "  green_for_ego: a green traffic light controls the ego lane ahead\n"
              "  no_light_for_ego: no traffic light controls the ego lane ahead (none visible, or only lights for other lanes / directions)",
              ["red_or_yellow_for_ego", "green_for_ego", "no_light_for_ego"], 1),
    "stop_sign": (ONE, "Stop sign controlling the ego vehicle lane ahead. Options:\n"
                  "  stop_sign_for_ego: a stop sign (or a painted STOP on the road) controls the ego lane just ahead\n"
                  "  no_stop_sign: no stop sign controls the ego lane ahead",
                  ["stop_sign_for_ego", "no_stop_sign"], 1),
    "lead": (TWO, "Vehicle directly ahead of the ego vehicle in the ego lane, within about 20 metres. Options:\n"
             "  none_ahead: no vehicle is directly ahead in the ego lane within about 20 metres\n"
             "  stopped_lead: a vehicle is directly ahead and it is standing still (same position in both images)\n"
             "  moving_lead: a vehicle is directly ahead and it is moving away or driving (its position changes between the images)",
             ["none_ahead", "stopped_lead", "moving_lead"], 2),
    "cross": (ONE, "Pedestrians, cyclists or crossing vehicles in the ego vehicle's path. Options:\n"
              "  crossing: a pedestrian, a cyclist or a vehicle is crossing or standing in the road space directly in front of the ego vehicle\n"
              "  clear: nothing is crossing or blocking the road space directly in front of the ego vehicle (a vehicle queued ahead in the "
              "ego lane does not count)",
              ["crossing", "clear"], 1),
}
TAIL = "End your reply with one line of the form `ANSWER: <option>`, <option> being one of: %s. Reply with that line only."


def ldir():
    d = data_dir() / "runs/op_parity/wod/launch"
    d.mkdir(parents=True, exist_ok=True)
    return d


# ---------------------------------------------------------------- label
def cmd_label(a):
    import glob
    import pandas as pd
    import torch
    from PIL import Image
    from transformers import AutoModelForImageTextToText, AutoProcessor
    from jevdrive import wod_zeroshot as Z
    from jevdrive.run import Run
    with Run("op_parity", "wod-launch-label", config=vars(a)) as run:
        p = glob.glob(str(data_dir() / "cache/huggingface/hub/models--Qwen--Qwen3-VL-4B-Instruct/snapshots/*"))[0]
        proc = AutoProcessor.from_pretrained(p)
        m = AutoModelForImageTextToText.from_pretrained(p, dtype=torch.bfloat16).to("cuda").eval()
        tok = proc.tokenizer
        pre = tok("ANSWER:", add_special_tokens=False).input_ids
        first = {}
        for q, (_, _, opts, _) in QS.items():
            first[q] = [tok("ANSWER: " + o, add_special_tokens=False).input_ids[len(pre)] for o in opts]
            assert len(set(first[q])) == len(opts), (q, first[q])
        names = Z.load_sets()["rater"]["name"].astype(str)[: a.limit or None]
        t2 = Z.root("t2_jpg")
        rows, t0 = [], time.time()
        with torch.no_grad():
            for i, n in enumerate(names):
                s, f = n.rsplit("-", 1)
                im0 = Image.open(t2 / n / "1.jpg").convert("RGB")
                prev = t2 / f"{s}-{int(f) - 10:03d}" / "1.jpg"
                row = {"name": n}
                for q, (head, body, opts, k) in QS.items():
                    ims = [im0] if k == 1 else [Image.open(prev).convert("RGB") if prev.exists() else im0, im0]
                    msgs = [{"role": "user", "content": [{"type": "image", "image": im} for im in ims]
                             + [{"type": "text", "text": f"{head}\n{body}\n" + TAIL % ", ".join(opts)}]}]
                    x = proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False) + "ANSWER:"
                    x = proc(text=[x], images=ims, return_tensors="pt").to("cuda")
                    lg = m(**x, use_cache=False).logits[0, -1].float()[first[q]]
                    pr = torch.softmax(lg, -1).cpu().numpy()
                    row[q] = opts[int(pr.argmax())]
                    row |= {f"{q}_p_{o}": round(float(v), 4) for o, v in zip(opts, pr)}
                    if q == "lead":
                        row["lead_two_frames"] = prev.exists()
                rows.append(row)
                if (i + 1) % 50 == 0 or i + 1 == len(names):
                    run.info(f"[{i + 1}/{len(names)}] {(time.time() - t0) / (i + 1):.2f} s/frame")
        OUT.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(rows)
        df.to_csv(OUT / ("context_vlm.csv" if not a.limit else f"context_vlm_first{a.limit}.csv"), index=False)
        run.info("counts: %s", {q: df[q].value_counts().to_dict() for q in QS})


def cmd_label2(a):
    """Post hoc (added after the hand check: the ego's own stop sign is outside the FRONT crop on most stop-controlled junctions): one
    question on the FRONT + FRONT_RIGHT pair -> results/wod_launch/context_extra.csv (column stop_ctrl)."""
    import glob
    import pandas as pd
    import torch
    from PIL import Image
    from transformers import AutoModelForImageTextToText, AutoProcessor
    from jevdrive import wod_zeroshot as Z
    from jevdrive.run import Run
    opts = ["stop_controlled", "not_stop_controlled"]
    text = ("The two images were taken at the same instant by a car (the ego vehicle): image 1 is its front camera, image 2 its front-right camera.\n"
            "Is the ego vehicle at, or just before, a junction where a stop sign controls the ego's direction? Evidence: a stop sign facing the ego "
            "(usually on the right kerb, image 2), a painted STOP on the ego lane, or the backs of stop signs for the other directions at an all-way "
            "stop. A junction with traffic lights is not stop-controlled. Options:\n"
            "  stop_controlled: a stop sign controls the ego's direction at the junction just ahead\n"
            "  not_stop_controlled: no stop sign controls the ego here\n" + TAIL % ", ".join(opts))
    with Run("op_parity", "wod-launch-label2", config=vars(a)) as run:
        p = glob.glob(str(data_dir() / "cache/huggingface/hub/models--Qwen--Qwen3-VL-4B-Instruct/snapshots/*"))[0]
        proc = AutoProcessor.from_pretrained(p)
        m = AutoModelForImageTextToText.from_pretrained(p, dtype=torch.bfloat16).to("cuda").eval()
        tok = proc.tokenizer
        pre = tok("ANSWER:", add_special_tokens=False).input_ids
        first = [tok("ANSWER: " + o, add_special_tokens=False).input_ids[len(pre)] for o in opts]
        assert len(set(first)) == 2, first
        names = Z.load_sets()["rater"]["name"].astype(str)[: a.limit or None]
        t2 = Z.root("t2_jpg")
        rows = []
        with torch.no_grad():
            for i, n in enumerate(names):
                ims = [Image.open(t2 / n / f"{c}.jpg").convert("RGB") for c in (1, 3)]
                msgs = [{"role": "user", "content": [{"type": "image", "image": im} for im in ims] + [{"type": "text", "text": text}]}]
                x = proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False) + "ANSWER:"
                x = proc(text=[x], images=ims, return_tensors="pt").to("cuda")
                pr = torch.softmax(m(**x, use_cache=False).logits[0, -1].float()[first], -1).cpu().numpy()
                rows.append({"name": n, "stop_ctrl": opts[int(pr.argmax())], "stop_ctrl_p": round(float(pr[0]), 4)})
                if (i + 1) % 100 == 0:
                    run.info(f"[{i + 1}/{len(names)}]")
        df = pd.DataFrame(rows)
        df.to_csv(OUT / ("context_extra.csv" if not a.limit else f"context_extra_first{a.limit}.csv"), index=False)
        run.info("counts: %s", df.stop_ctrl.value_counts().to_dict())


# ---------------------------------------------------------------- bias (ego-channel interventions on WP2)
def edit(e, var):
    """Main-mapping ego features (n, 20) -> the variant's (layout: pp_wod_diag; 4 = vx / 10, 6:8 = ax, ay / 3, 8:20 = 4 poses x / 10, y / 10, yaw)."""
    e = e.copy()
    if var == "cmd0":
        e[:, 1:4] = 0
    elif var == "acc0":
        e[:, 6:8] = 0
    elif var in ("vx1", "vx3", "cv1", "cv3"):
        e[:, 4] += float(var[2]) / 10.0
        if var.startswith("cv"):                                       # pose history made consistent with the nudged speed (straight, constant speed)
            e[:, 8:20:3] = e[:, 4:5] * T_HIST[None]
            e[:, 9:20:3] = 0
            e[:, 10:20:3] = 0
    return e


def cmd_bias(a):
    import torch
    import pp_hugsim as H
    from jevdrive import wod_zeroshot as Z
    from pp_wod import wod_ego
    r = Z.load_sets()["rater"]
    names = r["name"].astype(str)
    ego, _ = wod_ego(r["past"].astype(np.float64), r["intent"])
    for tag in a.tags:
        m = H.pmodel(tag, torch.device("cpu"))
        assert m.adapter is not None and not m.adapter.use_side, tag

        def run(e):
            with torch.no_grad():
                return np.concatenate([m.adapter(torch.from_numpy(np.ascontiguousarray(e[i:i + 512], np.float32)), None, None).numpy()
                                       for i in range(0, len(e), 512)])
        main = run(ego)
        st = {}
        for var in a.vars:
            e = edit(ego, var)
            b = {"main": main, "zero": np.zeros_like(main), "biasmean": np.broadcast_to(main.mean(0), main.shape),
                 "biasresid": main - main.mean(0)}.get(var)
            if b is None:
                b = run(e)
            b = np.ascontiguousarray(b).astype(np.float16)
            st[var] = {"rms": float(np.sqrt(np.mean(b.astype(np.float32) ** 2))), "rms_diff_main": float(np.sqrt(np.mean((b.astype(np.float32) - main) ** 2)))}
            np.savez(ldir() / f"bias-{tag}_{var}.npz", names=names, bias=b, ego=e)
            print(f"{tag}_{var}: {st[var]}", flush=True)
        v0 = np.linalg.norm(r["past"][:, -1, 2:4], axis=-1)
        stop = v0 < V_STOP
        st["standstill"] = {"n": int(stop.sum()), "rms_main": float(np.sqrt(np.mean(main[stop] ** 2))),
                            "rms_of_mean": float(np.sqrt(np.mean(main[stop].mean(0) ** 2))),
                            "rms_around_mean": float(np.sqrt(np.mean((main[stop] - main[stop].mean(0)) ** 2))),
                            "rms_all_mean": float(np.sqrt(np.mean(main.mean(0) ** 2))),
                            "rms_stopmean_minus_allmean": float(np.sqrt(np.mean((main[stop].mean(0) - main.mean(0)) ** 2)))}
        (ldir() / f"bias_stats-{tag}.json").write_text(json.dumps(st, indent=1))
        print(json.dumps(st["standstill"]), flush=True)


def cmd_gbias(a):
    """Serving bias of a --stop-gate arm (rater + extra frames): pp_wod.py bias with the rows fed a speed below the run's gate zeroed, as in
    training -> $DATA_DIR/runs/op_parity/wod/bias-<tag>.npz (the file wod_parity_chain's serve step looks for)."""
    import torch
    import pp_hugsim as H
    import pp_train as T
    from jevdrive import wod_zeroshot as Z
    from pp_wod import wod_ego
    S = Z.load_sets()
    names = np.concatenate([S[k]["name"].astype(str) for k in ("rater", "extra")])
    ego, _ = wod_ego(np.concatenate([S[k]["past"] for k in ("rater", "extra")]), np.concatenate([S[k]["intent"] for k in ("rater", "extra")]))
    for tag in a.tags:
        gate = float(torch.load(T.proot("runs", tag) / "ckpt-final.pt", map_location="cpu", weights_only=False)["cfg"]["stop_gate"])
        assert gate > 0, f"{tag} was not trained with --stop-gate"
        m = H.pmodel(tag, torch.device("cpu"))
        with torch.no_grad():
            b = np.concatenate([m.adapter(torch.from_numpy(ego[i:i + 256]), None, None).to(torch.float16).numpy() for i in range(0, len(ego), 256)])
        off = ego[:, 4] * 10.0 < gate
        b[off] = 0
        np.savez(data_dir() / "runs/op_parity/wod" / f"bias-{tag}.npz", names=names, bias=b, ego=ego, gated=off)
        print(f"{tag}: gate {gate} m/s, {int(off.sum())} of {len(off)} frames gated off, bias rms {float(np.sqrt(np.mean(b.astype(np.float32) ** 2))):.4f}", flush=True)


# ---------------------------------------------------------------- tok (training-protocol token path)
def to_wod(plan, dev_xy):
    """(n, 33, 15) plan means, (n, 2) camera x, y on the vehicle -> (n, 20, 2) WOD rear-axle waypoints at 0.25 .. 5 s (jevdrive.wod_zeroshot.openpilot_to_wod)."""
    from jevdrive import wod_zeroshot as Z
    from jevdrive.openpilot.model import T_IDXS
    plan = np.asarray(plan, np.float64)
    p = np.stack([plan[..., 0], -plan[..., 1]], -1)
    psi = -plan[..., 11]
    d = np.asarray(dev_xy, np.float64)[:, None]
    Rd = np.stack([np.cos(psi) * d[..., 0] - np.sin(psi) * d[..., 1], np.sin(psi) * d[..., 0] + np.cos(psi) * d[..., 1]], -1)
    rear = d + p - Rd
    out = np.empty((len(plan), len(Z.T_FUT), 2), np.float32)
    for k in range(2):
        out[..., k] = np.stack([np.interp(Z.T_FUT, T_IDXS, r) for r in rear[..., k]]) if len(plan) else 0
    return out


def cmd_tok(a):
    import torch
    import pp_train as T
    import pp_wod as PW
    import wod_parity as WP
    from jevdrive import op_adapt as A
    from jevdrive import waymo as W
    from jevdrive.data import splits
    from jevdrive.run import Run
    dev = torch.device("cuda")
    with Run("op_parity", "wod-launch-tok", config=vars(a)) as run:
        tr, dv, val = splits.load("wod/r2-train"), splits.load("wod/r2-dev"), splits.load("wod/val")
        for s in (tr, dv, val):
            run.use_split(s)
        models = {t: T.load_pmodel(t, dev) for t in TOK_ARMS}
        net = models["P0"].net
        pi = torch.as_tensor(A.plan_index(net.slices), device=dev)
        sx = net.slices["plan"].start + 495 + 15 * np.arange(33)               # raw spread of plan x at the 33 grid points
        df = W.load_index()
        key = dict(zip(W.frame_names(df), range(len(df))))
        past_all, fut_all = W.load_ego()
        intent_all = df.intent.to_numpy()
        cal = WP.calib()

        def run_models(H, ego):
            """H (B, 9, 32, 512) tokens, ego (B, 20) -> {arm: plan (B, 33, 15)}, shipped's raw plan-x spread (B, 33)."""
            tc = torch.tensor([[1.0, 0.0]], device=dev).expand(len(H), 2)
            out, sp = {}, None
            for t, m in models.items():
                o = m(H, ego, tc).float()
                out[t] = o[:, pi].view(-1, 33, 15).cpu().numpy()
                if t == "P0":
                    sp = o[:, sx].cpu().numpy()
            return out, sp

        # ---- part V: val streams
        streams = json.loads((WP.SRC / "wod_val_plan.json").read_text())["streams"]
        if a.limit:
            streams = streams[: a.limit]
        seq = df.sequence.astype(str).to_numpy()
        names_all = W.frame_names(df)
        isval = df.split.astype(str).to_numpy() == "val"
        off, Tn, rows = WP.stream_rows(streams, set(names_all[isval & val.mask(seq)]), need_future=True)
        idx = np.array([r[3] for r in rows])
        ego, _ = PW.wod_ego(past_all[idx], intent_all[idx])
        dxy = np.array([np.array(cal[s]["1"]["extrinsic"]).reshape(4, 4)[:2, 3] for s in seq[idx]])
        by = {}
        for r, (si, j, _, _) in enumerate(rows):
            by.setdefault(si, []).append((r, j))
        plans = {t: np.zeros((len(rows), 20, 2), np.float32) for t in TOK_ARMS}
        spread = np.zeros((len(rows), 33), np.float32)
        pooled = np.zeros((len(rows), 2, 512), np.float16)
        t0 = time.time()
        with torch.no_grad():
            for n_done, si in enumerate(sorted(by)):
                z = np.load(WP.SRC / "wodval" / f"{streams[si]['key']}.npz")
                x = torch.from_numpy(z["trunk"]).to(dev)
                tk = torch.cat([net.run_batched({A.TRUNK_OUT: x[i:i + 256, None].to(net.dtype)}, ["view_39"])["view_39"].reshape(-1, *A.H_SHAPE)
                                for i in range(0, len(x), 256)])
                r, j = map(np.array, zip(*by[si]))
                H = tk[torch.as_tensor(j[:, None] - np.arange(WP.NCTX - 1, -1, -1)[None], device=dev)]
                o, sp = run_models(H, torch.from_numpy(ego[r]).to(dev))
                for t in TOK_ARMS:
                    plans[t][r] = to_wod(o[t], dxy[r])
                spread[r] = sp
                pooled[r] = H[:, [8, 3]].float().mean(2).cpu().numpy()
                if (n_done + 1) % 100 == 0:
                    run.info(f"val streams {n_done + 1}/{len(by)}, {time.time() - t0:.0f} s")
        np.savez(ldir() / ("tok_val.npz" if not a.limit else "tok_val_smoke.npz"), names=np.array([r[2] for r in rows]), seq=seq[idx], past=past_all[idx],
                 fut=fut_all[idx], intent=intent_all[idx], spread_x_shipped=spread, pooled=pooled, **{f"plan_{t}": p for t, p in plans.items()})
        run.info(f"part V: {len(rows)} val rows of {len(by)} streams")

        # ---- part D: r2-dev rows (plans) and r2-train standstill rows (labels + pooled tokens for the probe)
        cd = WP.CACHE / "wod_r2"
        tab = dict(np.load(cd / "tab.npz"))
        fi = np.load(cd / "front_idx.npy")
        ticks = np.load(cd / "ticks.npy", mmap_mode="r")
        isdev, istr = dv.mask(tab["log"]), tr.mask(tab["log"])
        for part, sel in (("dev", np.flatnonzero(isdev)), ("train_stop", np.flatnonzero(istr & (tab["speed"] < 1.0)))):
            if a.limit:
                sel = sel[:: max(1, len(sel) // 512)]
            gi = np.array([key[n] for n in tab["names"][sel]])
            dxy = tab["cam"][sel][:, :2]
            plans = {t: np.zeros((len(sel), 20, 2), np.float32) for t in TOK_ARMS}
            spread = np.zeros((len(sel), 33), np.float32)
            pooled = np.zeros((len(sel), 2, 512), np.float16)
            with torch.no_grad():
                for i in range(0, len(sel), 256):
                    r = sel[i:i + 256]
                    f = fi[r]
                    if part == "dev":
                        u, inv = np.unique(f, return_inverse=True)
                        H = torch.from_numpy(ticks[u]).to(dev)[torch.as_tensor(inv.reshape(f.shape), device=dev)]
                        o, sp = run_models(H, torch.from_numpy(tab["ego"][r]).to(dev))
                        for t in TOK_ARMS:
                            plans[t][i:i + 256] = to_wod(o[t], dxy[i:i + 256])
                        spread[i:i + 256] = sp
                        pooled[i:i + 256] = H[:, [8, 3]].float().mean(2).cpu().numpy()
                    else:
                        u, inv = np.unique(f[:, [8, 3]], return_inverse=True)
                        pooled[i:i + 256] = torch.from_numpy(ticks[u]).to(dev).float().mean(1)[torch.as_tensor(inv.reshape(-1, 2), device=dev)].cpu().numpy()
                    if (i // 256) % 40 == 0:
                        run.info(f"{part} {i}/{len(sel)}")
            extra = {f"plan_{t}": p for t, p in plans.items()} | {"spread_x_shipped": spread} if part == "dev" else {}
            np.savez(ldir() / (f"tok_{part}.npz" if not a.limit else f"tok_{part}_smoke.npz"), names=tab["names"][sel], seq=tab["log"][sel],
                     past=past_all[gi], fut=fut_all[gi], intent=tab["intent"][sel], speed=tab["speed"][sel], pooled=pooled, **extra)
            run.info(f"part D {part}: {len(sel)} rows")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sp_ = ap.add_subparsers(dest="cmd", required=True)
    p = sp_.add_parser("label")
    p.add_argument("--limit", type=int, default=0)
    p = sp_.add_parser("label2")
    p.add_argument("--limit", type=int, default=0)
    p = sp_.add_parser("bias")
    p.add_argument("--tags", nargs="+", required=True)
    p.add_argument("--vars", nargs="+", default=list(VARS))
    p = sp_.add_parser("gbias")
    p.add_argument("--tags", nargs="+", required=True)
    p = sp_.add_parser("tok")
    p.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    {"label": cmd_label, "label2": cmd_label2, "bias": cmd_bias, "gbias": cmd_gbias, "tok": cmd_tok}[a.cmd](a)
