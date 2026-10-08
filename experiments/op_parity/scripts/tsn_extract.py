"""op_parity turn selector navtrain (plans/2026-10-08-turn-selector-navtrain-prereg.md): fold-model features of the navtrain turn tokens.

  python tsn_extract.py train   --shard i [--tag full|smoke] [--control]
  python tsn_extract.py navtest [--tag full|smoke]

train: every navtrain turn token of shard i is run through the fold model that held its log out (CF5f{j}-F-s0, j = sha256("cf5|" + log) % 5), in
the same loop, batch and fp16 path as sc_infer.py / tsi_extract.py; one forward gives the plan, the policy's `select_4` and `mean` (512 each)
and the road-edge head. Gate G-leak: j agrees with fold_of_token.csv, the log is not in the fold's training split, and plan_pos equals the
fold model's stored plan (bench/ol/<shard>/plans) (max |d| < 0.03 m over the first 15 points, fp16 rounding) on every row with speed >= 0.5 m/s. --control also runs the next fold's model on up to 100
rows per fold: at least half of those rows must exceed the tolerance (the check has teeth; the fold models share init and recipe, so their plans are close, median 0.07 m).
navtest: the 5 fold models on the navtest turn tokens (hidden state + plan only), for the G-hidden alignment gate against SH30.
Output: $DATA_DIR/runs/op_parity/turn_selnt/feat/<tag>/{s<i>.npz, navtest.npz}. GPU job: submit through the pool (jevdrive.cl submit).
"""
import sys as _sys, pathlib as _pl  # noqa: E401
_R = _pl.Path(__file__).resolve().parents[3]
_sys.path[:0] = [str(_R), str(_pl.Path(__file__).parent)]
import argparse, hashlib, json  # noqa: E401,E402

import numpy as np  # noqa: E402

from jevdrive.run import Run  # noqa: E402

K = 5
NSH = 12


def fold_of_log(log):
    return int(hashlib.sha256(f"cf{K}|{log}".encode()).hexdigest(), 16) % K


def run_rows(model, S, rows, run, batch=128):
    """-> plan_pos (m, 33, 3), select_4 (m, 512) fp16, mean (m, 512) fp16, road_edges mu (m, 2, 33, 2) for the given store rows."""
    import torch
    from jevdrive import op_adapt as A
    stash, orig = {}, model.net.run_batched

    def tap(feeds, want, **kw):
        o = orig(feeds, A.POLICY_OUT, **kw)
        stash.update(o)
        return {k: o[k] for k in want}
    model.net.run_batched = tap
    try:
        sl = model.net.slices
        pi = np.arange(sl["plan"].start, sl["plan"].start + 495)
        rs = sl["road_edges"]
        half = (rs.stop - rs.start) // 2
        assert half == 132, (rs, half)
        m = len(rows)
        P, h4, hm, re = np.zeros((m, 33, 3), np.float32), np.zeros((m, 512), np.float16), np.zeros((m, 512), np.float16), np.zeros((m, 2, 33, 2), np.float32)
        dev = S.ego.device
        with torch.no_grad():
            for i in range(0, m, batch):
                r = torch.as_tensor(rows[i:i + batch], device=dev)
                side = S.side[r] if S.side is not None else None
                o = model(S.front[r], S.ego[r], S.tc[r], side, None).float().cpu().numpy()
                j = slice(i, i + len(r))
                P[j] = o[:, pi].reshape(-1, 33, 15)[:, :, 0:3]
                re[j] = o[:, rs][:, :half].reshape(-1, 2, 33, 2)
                h4[j] = stash["select_4"].reshape(len(r), -1).float().cpu().numpy()
                hm[j] = stash["mean"].reshape(len(r), -1).float().cpu().numpy()
    finally:
        model.net.run_batched = orig
    return P, h4, hm, re


NPT = 15            # plan points compared: the first 15 (<= 37 m ahead) cover the 4 s horizon; the far points reach 190 m, where one fp16 ulp is 0.125 m
TOL_M = 0.03        # two fp16 ulps at 32 m; the same model through a different batch composition differs by <= 0.0156 m on these points


def plan_diff(ref, P):
    """Per-row max |ref - P| (m) over the first NPT plan points."""
    return np.abs(np.asarray(ref, np.float32)[:, :NPT] - P[:, :NPT]).max((1, 2))


def load_model(T, N, resolve, j, dev):
    m = resolve(f"CF{K}f{j}-F-s0@warp", check=True)
    model = N._load_ckpt(T, m.ckpt, dev)
    assert getattr(model, "mem", None) is None and not m.opt, "plain models only"
    return m, model


def out_dir(tag):
    from jevdrive.bench.models import data_dir
    return data_dir() / "runs/op_parity/turn_selnt/feat" / tag


def tokens_of(tag):
    """(tokens, fold, log) of the turn tokens to extract: all 28 323, or the smoke subset (60 per fold from shards 0-2, fixed seed)."""
    import pandas as pd
    from jevdrive.bench.models import data_dir
    df = pd.read_csv(data_dir() / "runs/op_parity/sh30_crossfit/labels/fold_of_token.csv")
    if tag == "smoke":
        f = data_dir() / "runs/op_parity/turn_selnt/smoke_tokens.txt"
        keep = set(f.read_text().split())
        df = df[df.token.isin(keep)]
    return df


def cmd_train(a):
    import torch
    from jevdrive.bench import navsim as N
    from jevdrive.bench.models import data_dir, resolve
    from jevdrive.data import splits
    import nt_labels as NL
    out = out_dir(a.tag) / f"s{a.shard}.npz"
    with Run("op_parity", f"turn_selnt/extract-{a.tag}-s{a.shard}", seed=0, config=vars(a)) as run:
        if out.exists():
            run.info("exists: %s", out)
            return
        N._pp_path()
        import pp_train as T
        dev = torch.device("cuda")
        df = tokens_of(a.tag)
        trs = [splits.load(f"navsim/op-parity-cf{K}f{j}-train") for j in range(K)]
        for s in trs:
            run.use_split(s)
        data = NL.shard_data(a.shard)
        S = T.Store([data], dev, need_side=(N.cache_dir(data, "gimm") / "side.npy").exists(), frames="warp")
        names = S.tab["names"]
        row = {t: i for i, t in enumerate(names.tolist())}
        sel = df[df.token.isin(row)]
        rows_all = np.array([row[t] for t in sel.token])
        fold = sel.fold.to_numpy()
        logs = sel.log.to_numpy()
        # G-leak (a): the fold of the csv is the hash fold, and the log is not in that fold's training split
        assert all(fold_of_log(l) == f for l, f in zip(logs, fold)), "fold_of_token.csv disagrees with sha256 fold"
        for j in range(K):
            m = fold == j
            assert not trs[j].mask(logs[m]).any(), f"fold {j}: a held-out log is in the training split"
        n = len(rows_all)
        res = dict(tokens=sel.token.to_numpy().astype(str), fold=fold, row=rows_all, select_4=np.zeros((n, 512), np.float16), mean=np.zeros((n, 512), np.float16),
                   road_edges=np.zeros((n, 2, 33, 2), np.float32), plan_pos=np.zeros((n, 33, 3), np.float32), model_fold=np.full(n, -1))
        diff, ctrl = np.zeros(n), []
        speed = S.tb["speed"][rows_all]
        for j in range(K):
            idx = np.flatnonzero(fold == j)
            if not len(idx):
                continue
            mj, model = load_model(T, N, resolve, j, dev)
            P, h4, hm, re = run_rows(model, S, rows_all[idx], run)
            res["plan_pos"][idx], res["select_4"][idx], res["mean"][idx], res["road_edges"][idx], res["model_fold"][idx] = P, h4, hm, re, j
            ref = np.load(NL.ol(a.shard, "plans", f"{NL.stem(mj.name)}.npz"))
            assert ref["names"].tolist() == names.tolist(), "stored plans rows != token cache rows"
            d = plan_diff(ref["plan_pos"][rows_all[idx]], P)
            diff[idx] = d
            ok = speed[idx] >= 0.5
            run.info("fold %d: %d rows, max |dplan_pos| vs stored %.3g m over the first 15 points (%d rows with speed >= 0.5)", j, len(idx), d[ok].max() if ok.any() else 0, ok.sum())
            if a.control:
                jj = (j + 1) % K
                _, wrong = load_model(T, N, resolve, jj, dev)
                c = idx[:100]
                Pw, *_ = run_rows(wrong, S, rows_all[c], run)
                ctrl.append(plan_diff(ref["plan_pos"][rows_all[c]], Pw))
                del wrong
            del model
            torch.cuda.empty_cache()
        ok = speed >= 0.5
        gate = dict(shard=a.shard, n=int(n), rows_speed_ok=int(ok.sum()), max_diff_m=float(diff[ok].max()) if ok.any() else 0.0, median_diff_m=float(np.median(diff[ok])) if ok.any() else 0.0,
                    leak_fold_ok=True)
        if ctrl:
            c = np.concatenate(ctrl)
            gate.update(ctrl_median_diff_m=float(np.median(c)), ctrl_frac_over_tol=float((c > TOL_M).mean()), ctrl_n=int(len(c)))
        res["diff"] = diff
        res["gate"] = np.array(json.dumps(gate))
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savez(out.with_suffix(".tmp.npz"), **res)
        out.with_suffix(".tmp.npz").rename(out)
        run.summary.update(gate)
        run.info("gate %s", gate)
        assert gate["max_diff_m"] < TOL_M, f"G-leak (b): extraction differs from the stored fold-model plan by {gate['max_diff_m']} m"


def cmd_navtest(a):
    import torch
    from jevdrive.bench import navsim as N
    from jevdrive.bench.models import resolve
    import turn_ceiling as TC
    out = out_dir(a.tag) / "navtest.npz"
    with Run("op_parity", f"turn_selnt/extract-{a.tag}-navtest", seed=0, config=vars(a)) as run:
        if out.exists():
            return
        N._pp_path()
        import pp_train as T
        dev = torch.device("cuda")
        tok, _ = TC.bucket_tokens()
        S = T.Store(["lb_navtest"], dev, need_side=(N.cache_dir("lb_navtest", "gimm") / "side.npy").exists(), frames="warp")
        row = {t: i for i, t in enumerate(S.tab["names"].tolist())}
        rows = np.array([row[t] for t in tok])
        res = dict(tokens=np.array(tok))
        for j in range(K):
            _, model = load_model(T, N, resolve, j, dev)
            P, h4, hm, re = run_rows(model, S, rows, run)
            res[f"select_4_{j}"], res[f"mean_{j}"], res[f"plan_pos_{j}"] = h4, hm, P
            del model
            torch.cuda.empty_cache()
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savez(out.with_suffix(".tmp.npz"), **res)
        out.with_suffix(".tmp.npz").rename(out)


def cmd_smoke_tokens(a):
    """60 turn tokens per fold from shards 0-2 (fixed seed) -> $DATA_DIR/runs/op_parity/turn_selnt/smoke_tokens.txt."""
    import pandas as pd
    import nt_labels as NL
    from jevdrive.bench.models import data_dir
    df = pd.read_csv(data_dir() / "runs/op_parity/sh30_crossfit/labels/fold_of_token.csv")
    inshard = set()
    for i in range(3):
        inshard |= set(np.load(data_dir() / f"runs/op_parity/cache/{NL.shard_data(i)}/tab.npz")["names"].tolist())
    df = df[df.token.isin(inshard)]
    rng = np.random.default_rng(0)
    keep = np.concatenate([rng.permutation(df.token[df.fold == j].to_numpy())[:60] for j in range(K)])
    f = data_dir() / "runs/op_parity/turn_selnt/smoke_tokens.txt"
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text("\n".join(keep.tolist()) + "\n")
    print(len(keep), "tokens ->", f)


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("train")
    p.add_argument("--shard", type=int, required=True)
    p.add_argument("--tag", default="full")
    p.add_argument("--control", action="store_true")
    sp.add_parser("smoke-tokens")
    p = sp.add_parser("navtest")
    p.add_argument("--tag", default="full")
    a = ap.parse_args()
    {"train": cmd_train, "navtest": cmd_navtest, "smoke-tokens": cmd_smoke_tokens}[a.cmd](a)


if __name__ == "__main__":
    main()
