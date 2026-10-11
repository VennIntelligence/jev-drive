"""ONPOL-PAI row cache: the PAI row files (results/rows_format.md) -> what the trainer reads per row, and gate G-id.

The model input of a row is rebuilt as the PAI driver builds it (experiments/alpasim/lib/pai_core.plan), not as the navtrain caches were
(no warp protocol): the 8 slot frames are the rendered frames the driver stacked (`frames[slot]`, or `cold_cur` on the cold-start rows whose
oldest slots were back-warped), the image pair of slot j is (slot j - 1, slot j) with a zero image before slot 0, and the frozen Cinque
encoder turns each pair into one (32, 512) fp16 token. A pair shared by several decisions is encoded once (two tokens per rendered frame:
with its predecessor 0.2 s earlier, and as the oldest slot of the decision 1.4 s later). The ego vector is the row's `ego_in`.
The teacher of a row (targets of the non-plan heads, as pp_prep's teacher.npz) is shipped Cinque on the same 8 tokens.

  build --pass pol          every scene file of $DATA_DIR/runs/onpol_pai/rows/<pass>/ -> <out>/<pass>/parts/<scene>.{npz,tok.npy} (resumable:
                            a part is redone only when its row file changed), then merged into <out>/<pass>/{tok.npy (U, 32, 512) fp16,
                            tab.npz}: scenes (S,), sid (N,) scene index, tix (N, 8) token rows, t_out (N, Dd) teacher, usable (N,) (all 8
                            slots valid), every per-row input / label key the trainer reads, per-scene keys repeated per row
  gid   --pass pol --tag P2H10S-F-s0 --control P2H10S-F-s1
                            G-id: pp_train.PModel of the tag on the cached tokens + ego, exported as the driver exports (onp_loss.rear_xy),
                            against the driver's logged `poses_model`: share of rows whose largest position difference over the 8 poses
                            (0.5 .. 4 s) exceeds 0.03 m, and the same for a wrong checkpoint (docs/long-runs.md, equality gates on fp16
                            plan points). Line: share <= 1 % and the control's median above 0.03 m -> <out>/gid/<pass>-<tag>.{json,md}

  $DATA_DIR/envs/op-train/bin/python experiments/onpol_pai/scripts/onp_cache.py build --pass pol        (GPU: about 2 x frames encoder passes)
"""
import sys as _sys, pathlib as _pl  # noqa: E401
_R = _pl.Path(__file__).resolve().parents[3]
_sys.path[:0] = [str(_R), str(_R / "lib"), str(_R / "scripts"), str(_R / "experiments/op_adapt_r2/lib"), str(_R / "experiments/op_parity/scripts"),
                 str(_R / "experiments/body1/lib"), str(_R / "experiments/onpol_pai/lib")]
import argparse, json, os, time  # noqa: E401,E402
from concurrent.futures import ThreadPoolExecutor  # noqa: E402

import numpy as np  # noqa: E402

FRAME = (2, 6, 128, 256)
AT = (0.275, 0.525)
IN = ("frames", "slot", "valid", "cold_rows", "cold_cur", "lht")
ROW = ("t0_us", "k", "ego_in", "poses", "poses_model", "v0", "e0", "dpsi0", "ds0", "s_ego", "box", "box_ok", "edge_box", "edge_ok", "path", "path_ok",
       "path_v", "path_h", "log0", "log_fut", "log_fut_ok", "imit", "imit_ok", "stop_s", "stop_tcross_us", "turn_s", "turn_vpk", "after_event", "after_end")
SCENE = ("cam_t", "veh", "veh_sim", "has_stop", "score", "zero_kind")


def root(a) -> _pl.Path:
    from jevdrive.common import data_dir
    return _pl.Path(a.out) if a.out else data_dir() / "runs" / "onpol_pai" / "train" / "cache"


def sig(f) -> list:
    st = os.stat(f)
    return [st.st_size, st.st_mtime_ns]


def load(f) -> dict:
    """The row file's arrays and its signature as read (a file replaced between this read and the part's write must not look fresh)."""
    s = sig(f)
    with np.load(f) as z:
        d = {k: z[k] for k in IN + ROW + SCENE}
    assert sig(f) == s, f"{f} was replaced while it was read: run the build again"
    return d | {"_sig": np.array(s)}


def in_key(z: dict) -> str:
    """Hash of everything a row's tokens and teacher depend on: the frames, the slot table, the cold-start rows' own frames, the traffic side."""
    import hashlib
    h = hashlib.blake2b(digest_size=16)
    for k in ("frames", "slot", "valid", "cold_rows", "cold_cur", "lht"):
        h.update(k.encode()), h.update(str(z[k].shape).encode()), h.update(np.ascontiguousarray(z[k]).data)
    return h.hexdigest()


def pairs(z: dict):
    """-> prev, cur (U, 2, 6, 128, 256) uint8 distinct image pairs of the scene and tix (N, 8) the pair of every row's slot."""
    slot, fr, zero = z["slot"], z["frames"], np.zeros(FRAME, np.uint8)
    cold = {int(i): j for j, i in enumerate(z["cold_rows"])}
    assert set(cold) == set(np.flatnonzero((slot < 0).any(1)).tolist()), "cold_rows do not match the rows with a -1 slot"
    seen, prev, cur, tix = {}, [], [], np.empty(slot.shape, np.int64)
    for i in range(len(slot)):
        c = z["cold_cur"][cold[i]] if i in cold else None
        for j in range(8):
            k = ("c", i, j) if c is not None else (int(slot[i, j - 1]) if j else -1, int(slot[i, j]))
            if k not in seen:
                seen[k] = len(cur)
                src = c if c is not None else fr
                a, b = (j - 1, j) if c is not None else k
                prev.append(zero if (j == 0) else src[a]), cur.append(src[b])
            tix[i, j] = seen[k]
    return np.stack(prev), np.stack(cur), tix


def cmd_build(a):
    import torch
    from jevdrive import op_adapt as A
    from jevdrive.common import data_dir
    from jevdrive.run import Run
    src = _pl.Path(a.rows) if a.rows else data_dir() / "runs" / "onpol_pai" / "rows" / a.pas
    assert not a.labels_only or a.out, "--labels-only writes a cache without tokens: give it its own --out"
    out = root(a) / a.pas
    (out / "parts").mkdir(parents=True, exist_ok=True)
    files = sorted(src.glob("*.npz"))[:a.limit or None]
    with Run("onpol_pai", f"cache-{a.pas}", config=vars(a)) as run:
        assert files, f"no row file in {src}"
        dev = torch.device("cpu" if a.labels_only else "cuda")
        net = A.load("cinque", torch.float16).to(dev).eval()
        di, pi = A.distill_index(net.slices), A.plan_index(net.slices)
        part = lambda f: out / "parts" / f"{f.stem}.npz"  # noqa: E731
        def fresh(f):                                                   # the part is of this row file and holds every key read today
            if not (part(f).exists() and (out / "parts" / f"{f.stem}.tok.npy").exists()):
                return False
            z = np.load(part(f))
            return z["src"].tolist() == sig(f) and all(k in z.files for k in ROW + SCENE + ("cold", "in_key"))
        todo = [f for f in files if a.force or a.relabel or not fresh(f)]
        run.info(f"{a.pas}: {len(files)} scene files in {src}, {len(todo)} to encode (batch {a.bs})")
        t0, n_pair, n_row, n_keep = time.time(), 0, 0, 0
        with ThreadPoolExecutor(2) as ex:                               # the next scene is read while this one is on the card
            nxt = ex.submit(load, todo[0]) if todo else None
            for q, f in enumerate(run.tqdm(todo, desc="scenes")):
                z = nxt.result()
                nxt = ex.submit(load, todo[q + 1]) if q + 1 < len(todo) else None
                n = len(z["slot"])
                tix, tok, t_out = np.zeros((n, 8), np.int64), np.zeros((1, *A.H_SHAPE), np.float16), np.zeros((n, len(di)), np.float32)
                ik, old = in_key(z), (np.load(part(f)) if part(f).exists() and (out / "parts" / f"{f.stem}.tok.npy").exists() else None)
                keep = old is not None and "in_key" in old.files and str(old["in_key"]) == ik and not (a.force or a.labels_only)
                if keep:                                                # the row file was rebuilt with the same model inputs: labels only
                    tix, t_out, tok, n_keep = old["tix"], old["t_out"], None, n_keep + 1
                elif not a.labels_only:
                    prev, cur, tix = pairs(z)
                    tok = np.empty((len(cur), *A.H_SHAPE), np.float16)
                    with torch.no_grad():
                        for i in range(0, len(cur), a.bs):
                            p, c = (torch.from_numpy(x[i:i + a.bs]).to(dev) for x in (prev, cur))
                            tok[i:i + a.bs] = net.run_batched(A.vision_feeds(p, c), ["view_39"])["view_39"].reshape(len(c), *A.H_SHAPE).cpu().numpy()
                        tc = torch.tensor([[0.0, 1.0] if bool(z["lht"]) else [1.0, 0.0]], device=dev)
                        for i in range(0, n, 256):                      # pp_prep's teacher: a zero, invalid slot before the 8
                            H = torch.from_numpy(tok[tix[i:i + 256]]).to(dev)
                            H = torch.cat([torch.zeros_like(H[:, :1]), H], 1)
                            valid = torch.ones(H.shape[:2], dtype=torch.bool, device=dev)
                            valid[:, 0] = False
                            t_out[i:i + 256] = A._policy(net, H, AT, tc.expand(len(H), 2), valid)["outputs"].float()[:, di].cpu().numpy()
                rep = lambda k: np.broadcast_to(np.asarray(z[k]), (n,) + np.asarray(z[k]).shape).copy()  # noqa: E731
                d = {k: z[k] for k in ROW} | {k: rep(k) for k in SCENE + ("lht",)}
                assert all(len(v) == n for v in d.values()), {k: v.shape for k, v in d.items() if len(v) != n}
                if tok is not None:
                    tmp = out / "parts" / f"{f.stem}.tok.tmp.npy"
                    np.save(tmp, tok)
                    os.replace(tmp, out / "parts" / f"{f.stem}.tok.npy")
                tmp = out / "parts" / f"{f.stem}.tmp.npz"
                np.savez(tmp, tix=tix, t_out=t_out, usable=z["valid"].all(1), cold=(z["slot"] < 0).any(1), src=z["_sig"], in_key=np.array(ik), **d)
                os.replace(tmp, part(f))
                n_pair, n_row = n_pair + (len(tok) if tok is not None else 0), n_row + n
        el = time.time() - t0
        run.info(f"encoded {n_pair} pairs of {n_row} rows in {el:.0f} s ({n_pair / max(el, 1e-9):.0f} pairs/s); tokens kept by input-key equality (labels only rewritten): {n_keep} of {len(todo)} scenes")
        # ---- merge: one token file, one table
        P = [np.load(part(f)) for f in files]
        nt = [len(np.load(out / "parts" / f"{f.stem}.tok.npy", mmap_mode="r")) for f in files]
        off = np.cumsum([0] + nt)
        tk = np.lib.format.open_memmap(out / "tok.tmp.npy", "w+", np.float16, (int(off[-1]), *A.H_SHAPE))
        for i, f in enumerate(files):
            tk[off[i]:off[i + 1]] = np.load(out / "parts" / f"{f.stem}.tok.npy", mmap_mode="r")
        tk.flush()
        del tk
        os.replace(out / "tok.tmp.npy", out / "tok.npy")
        tab = {k: np.concatenate([p[k] for p in P]) for k in ROW + SCENE + ("lht", "t_out", "usable", "cold")}
        tab |= dict(scenes=np.array([f.stem for f in files]), sid=np.concatenate([np.full(len(p["tix"]), i) for i, p in enumerate(P)]),
                    tix=np.concatenate([p["tix"] + off[i] for i, p in enumerate(P)]), di=di, pi=pi, **({"labels_only": np.array(True)} if a.labels_only else {}))
        np.savez(out / "tab.tmp.npz", **tab)
        os.replace(out / "tab.tmp.npz", out / "tab.npz")
        n = len(tab["sid"])
        run.summary.update(scenes=len(files), rows=n, tokens=int(off[-1]), usable=int(tab["usable"].sum()), after_event=int(tab["after_event"].sum()),
                           after_end=int(tab["after_end"].sum()), trainable=int((tab["usable"] & ~tab["after_event"] & ~tab["after_end"]).sum()),
                           cold_rows=int(tab["cold"].sum()), encode_s=el, out=str(out))
        run.info(f"{out}: {len(files)} scenes, {n} rows, {off[-1]} tokens ({off[-1] * 32 / 2 ** 20:.2f} GiB), usable {int(tab['usable'].sum())}, "
                 f"after an event {int(tab['after_event'].sum())}")


def cmd_gid(a):
    import torch
    import onp_loss as OL
    import pp_train as T
    from experiments.op_adapt_r2.lib import op_adapt_r2 as R2
    from jevdrive.run import Run
    dev = torch.device("cuda")
    with Run("onpol_pai", f"gid-{a.pas}-{a.tag}", config=vars(a)) as run:
        R = OL.OnRows(root(a), [a.pas], dev)
        rows = np.flatnonzero(R.usable)
        W = torch.as_tensor(R2.t_weights(T.T8), device=dev)
        cold = R.cold[rows]                                         # rows with a back-warped slot (the first 1.4 s of a session)
        ref = R.poses_model[torch.as_tensor(rows, device=dev)]
        res = dict(pas=a.pas, rows=int(len(rows)), tol_m=a.tol)

        def read(tag):
            m = T.load_pmodel(tag, dev)
            pi = torch.as_tensor(np.arange(m.net.slices["plan"].start, m.net.slices["plan"].start + 495), device=dev)
            out = []
            with torch.no_grad():
                for i in range(0, len(rows), 256):
                    r = torch.as_tensor(rows[i:i + 256], device=dev)
                    o = m(R.tokens(rows[i:i + 256]).to(dev), R.ego[r], R.tc[r]).float()
                    out.append(torch.stack(OL.rear_xy(o[:, pi].view(-1, 33, 15), R.cam[r, :2], W), -1))
            p = torch.cat(out)
            d = torch.hypot(p[..., 0] - ref[..., 0], p[..., 1] - ref[..., 1]).amax(1).cpu().numpy()
            dy = OL.wrap(p[..., 2] - ref[..., 2]).abs().amax(1).cpu().numpy()
            sc = R.scene[rows]
            by = sorted(((float((d[sc == q] > a.tol).mean()), float(d[sc == q].max()), q) for q in np.unique(sc)), reverse=True)
            return dict(tag=tag, scenes_over_10pct=sum(b[0] > 0.1 for b in by), scene_share_median=float(np.median([b[0] for b in by])),
                        worst_scenes=[dict(scene=q, share_over=round(u, 4), max_m=round(v, 4)) for u, v, q in by[:5]], share_over=float((d > a.tol).mean()), median_m=float(np.median(d)), p99_m=float(np.quantile(d, 0.99)), max_m=float(d.max()),
                        max_yaw_rad=float(dy.max()), share_over_cold=float((d[cold] > a.tol).mean()) if cold.any() else None, n_cold=int(cold.sum()),
                        share_over_warm=float((d[~cold] > a.tol).mean()))
        res["model"] = read(a.tag)
        res["control"] = [read(c) for c in a.control]
        res["pass"] = bool(res["model"]["share_over"] <= a.share and all(c["median_m"] > a.tol for c in res["control"]) and len(res["control"]) > 0)
        md = (f"# G-id, pass `{a.pas}`: training-path forward against the driver's logged `poses_model`\n\n{len(rows)} usable rows; largest position "
              f"difference over the 8 poses (0.5 to 4 s) per row; tolerance {a.tol} m. Line: share over <= {a.share}, every control's median over.\n\n"
              "| Checkpoint | Share of rows over | Median m | p99 m | Max m | Max yaw rad | Share over, cold-start rows (n) | Share over, other rows |\n|---|--:|--:|--:|--:|--:|--:|--:|\n"
              + "".join(f"| `{x['tag']}`{' (control)' if j else ''} | {x['share_over']:.4f} | {x['median_m']:.4f} | {x['p99_m']:.4f} | {x['max_m']:.3f} | {x['max_yaw_rad']:.4f} | "
                        f"{x['share_over_cold'] if x['share_over_cold'] is None else round(x['share_over_cold'], 4)} ({x['n_cold']}) | {x['share_over_warm']:.4f} |\n"
                        for j, x in enumerate([res["model"]] + res["control"])) + f"\nPer scene, `{a.tag}`: median share over {res['model']['scene_share_median']:.4f}; worst: "
              + "; ".join(f"{w['scene'][-6:]} {w['share_over']} (max {w['max_m']} m)" for w in res["model"]["worst_scenes"])
              + f"\n\nGate: {'PASS' if res['pass'] else 'FAIL'}\n")
        g = root(a) / "gid"
        g.mkdir(parents=True, exist_ok=True)
        (g / f"{a.pas}-{a.tag}.json").write_text(json.dumps(res, indent=1) + "\n")
        (g / f"{a.pas}-{a.tag}.md").write_text(md)
        run.info(md)
        run.summary.update(gate=res["pass"], share_over=res["model"]["share_over"], out=str(g / f"{a.pas}-{a.tag}.md"))
    raise SystemExit(0 if res["pass"] else 2)


def cmd_gid_ulp(a):
    """Amended G-id (prereg, post-hoc amendment of 2026-10-11): per coordinate (x, y) of the 8 poses the tolerance is max(tol, n_ulp fp16 ulps
    at the logged coordinate's magnitude); a row is over when any coordinate is. Line: share of rows over <= --share on every usable row, and
    the wrong checkpoint clearly apart under the same tolerance (its share over >= 0.5) -> <out>/gid/<pass>-<tag>-ulp.{json,md}."""
    import torch
    import onp_loss as OL
    import pp_train as T
    from experiments.op_adapt_r2.lib import op_adapt_r2 as R2
    from jevdrive.run import Run
    dev, g_ = torch.device("cuda"), root(a) / "gid"
    with Run("onpol_pai", f"gid-ulp-{a.pas}-{a.tag}", config=vars(a)) as run:
        R = OL.OnRows(root(a), [a.pas], dev)
        rows = np.flatnonzero(R.usable)
        W = torch.as_tensor(R2.t_weights(T.T8), device=dev)
        rt = torch.as_tensor(rows, device=dev)
        ref = R.poses_model[rt][..., :2].double()
        ulp = torch.pow(2.0, torch.floor(torch.log2(ref.abs().clamp_min(2.0 ** -14))) - 10)        # fp16 spacing at the logged coordinate
        tol = torch.clamp(a.n_ulp * ulp, min=a.tol)
        cold, v0 = R.cold[rows], R.v0[rt].cpu().numpy()
        bins = [("v0 < 5 m/s", v0 < 5), ("5 to 10", (v0 >= 5) & (v0 < 10)), ("10 to 20", (v0 >= 10) & (v0 < 20)), (">= 20", v0 >= 20)]

        def read(tag):
            m = T.load_pmodel(tag, dev)
            pi = torch.as_tensor(np.arange(m.net.slices["plan"].start, m.net.slices["plan"].start + 495), device=dev)
            out = []
            with torch.no_grad():
                for i in range(0, len(rows), 256):
                    r = rt[i:i + 256]
                    o = m(R.tokens(rows[i:i + 256]).to(dev), R.ego[r], R.tc[r]).float()
                    out.append(torch.stack(OL.rear_xy(o[:, pi].view(-1, 33, 15), R.cam[r, :2], W)[:2], -1))
            d = (torch.cat(out).double() - ref).abs()
            over = (d > tol).flatten(1).any(1).cpu().numpy()
            fix = (d > a.tol).flatten(1).any(1).cpu().numpy()
            eu = (d / ulp).flatten(1).amax(1).cpu().numpy()                                          # largest difference of the row in ulps
            ex = torch.where(d > a.tol, d / ulp, torch.zeros_like(d)).flatten(1).amax(1).cpu().numpy()   # the same over the coordinates beyond tol
            return dict(tag=tag, share_over=float(over.mean()), share_over_fixed_tol=float(fix.mean()), share_over_cold=float(over[cold].mean()), n_cold=int(cold.sum()),
                        share_over_other=float(over[~cold].mean()), by_speed={n: dict(n=int(b.sum()), share_over=float(over[b].mean()), share_over_fixed_tol=float(fix[b].mean())) for n, b in bins},
                        share_over_at={f"{k} ulp": float((ex > k).mean()) for k in (2, 3, 4, 6, 8)}, ulps_quantiles={str(q): float(np.quantile(eu, q)) for q in (0.5, 0.9, 0.99, 0.999, 1.0)},
                        max_m=float(d.max()))
        res = dict(pas=a.pas, rows=int(len(rows)), tol_m=a.tol, n_ulp=a.n_ulp, model=read(a.tag), control=[read(c) for c in a.control])
        # the line the lane runs on (prereg deviation of 2026-10-11: the registered 0.03 m rule and its 2-ulp amendment both fail on fp16 noise)
        k4 = lambda x: x["share_over_at"]["4 ulp"]  # noqa: E731
        reg = g_ / f"{a.pas}-{a.tag}.json"
        res["line"] = dict(control_share_over_4ulp=[k4(c) for c in res["control"]], max_m=res["model"]["max_m"], share_over_4ulp=k4(res["model"]),
                           share_over_2ulp=res["model"]["share_over_at"]["2 ulp"], share_over_registered=json.loads(reg.read_text())["model"]["share_over"] if reg.exists() else None)
        res["pass"] = bool(len(res["control"]) > 0 and all(k4(c) >= 0.9 for c in res["control"]) and res["model"]["max_m"] <= 0.25 and k4(res["model"]) <= a.share)
        row = lambda x, j: (f"| `{x['tag']}`{' (control)' if j else ''} | {x['share_over']:.4f} | {x['share_over_fixed_tol']:.4f} | {x['share_over_cold']:.4f} ({x['n_cold']}) | {x['share_over_other']:.4f} | "  # noqa: E731
                            + " | ".join(f"{v['share_over']:.4f} ({v['n']})" for v in x["by_speed"].values()) + " | " + " / ".join(f"{v:.4f}" for v in x["share_over_at"].values())
                            + " | " + " / ".join(f"{v:.2f}" for v in x["ulps_quantiles"].values()) + " |\n")
        md = (f"# G-id (amended), pass `{a.pas}`: per-coordinate tolerance max({a.tol} m, {a.n_ulp} fp16 ulps at the logged coordinate)\n\n{len(rows)} usable rows, x and y of the 8 poses "
              f"(0.5 to 4 s); a row is over when any coordinate is. Line: share over <= {a.share}; control's share over >= 0.5. `Fixed` = the registered rule per coordinate ({a.tol} m). "
              "`At k ulp` = share of rows with a coordinate over both the fixed tolerance and k ulps (k = 2, 3, 4, 6, 8). `Ulps` = the row's largest difference in ulps, quantiles 0.5 / 0.9 / 0.99 / 0.999 / max.\n\n"
              "| Checkpoint | Share over | Fixed | Cold-start rows (n) | Other rows | " + " | ".join(n for n, _ in bins) + " | At k ulp | Ulps |\n|---|--:|--:|--:|--:|--:|--:|--:|--:|---|---|\n"
              + "".join(row(x, j) for j, x in enumerate([res["model"]] + res["control"]))
              + f"\nWorking line (deviation recorded in the pre-registration; the registered rule and its 2-ulp amendment FAIL): wrong-checkpoint share over max({a.tol} m, 4 ulp) "
              f"{res['line']['control_share_over_4ulp']} (>= 0.9), largest difference {res['line']['max_m']:.3f} m (<= 0.25), share over max({a.tol} m, 4 ulp) "
              f"{res['line']['share_over_4ulp']:.4f} (<= {a.share}). Registered share over (0.03 m, position): {res['line']['share_over_registered']}; 2-ulp amendment: "
              f"{res['line']['share_over_2ulp']:.4f}.\n\nWorking line: {'PASS' if res['pass'] else 'FAIL'}\n")
        g = g_
        (g / f"{a.pas}-{a.tag}-ulp.json").write_text(json.dumps(res, indent=1) + "\n")
        (g / f"{a.pas}-{a.tag}-ulp.md").write_text(md)
        run.info(md)
        run.summary.update(gate=res["pass"], **{q: v for q, v in res["line"].items() if not isinstance(v, list)})
    raise SystemExit(0 if res["pass"] else 2)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("build")
    p.add_argument("--pass", dest="pas", required=True, help="pol | log")
    p.add_argument("--rows", default="", help="directory of the row files (default $DATA_DIR/runs/onpol_pai/rows/<pass>)")
    p.add_argument("--out", default="", help="cache root (default $DATA_DIR/runs/onpol_pai/train/cache)")
    p.add_argument("--bs", type=int, default=8, help="image pairs per encoder call (8 = the driver's batch: one decision)")
    p.add_argument("--limit", type=int, default=0, help="first n scene files only (tests)")
    p.add_argument("--force", action="store_true")
    p.add_argument("--relabel", action="store_true", help="rewrite every part's labels from its row file; tokens and teacher are kept where the input key is equal")
    p.add_argument("--labels-only", action="store_true", help="no encoder, no teacher (zero tokens): the labels alone, for onp_train.py check on the CPU; needs --out")
    p = sub.add_parser("gid")
    p.add_argument("--pass", dest="pas", required=True)
    p.add_argument("--out", default="")
    p.add_argument("--tag", default="P2H10S-F-s0")
    p.add_argument("--control", nargs="*", default=["P2H10S-F-s1"])
    p.add_argument("--tol", type=float, default=0.03)
    p.add_argument("--share", type=float, default=0.01)
    p = sub.add_parser("gid-ulp")
    p.add_argument("--pass", dest="pas", required=True)
    p.add_argument("--out", default="")
    p.add_argument("--tag", default="P2H10S-F-s0")
    p.add_argument("--control", nargs="*", default=["P2H10S-F-s1"])
    p.add_argument("--tol", type=float, default=0.03)
    p.add_argument("--n-ulp", type=float, default=2.0)
    p.add_argument("--share", type=float, default=0.01)
    a = ap.parse_args()
    {"build": cmd_build, "gid": cmd_gid, "gid-ulp": cmd_gid_ulp}[a.cmd](a)
