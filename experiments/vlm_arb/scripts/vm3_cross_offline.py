"""vm3 cross-traffic release check, offline: can zero-shot Qwen3-VL-4B (one forward pass, first-token option scoring after
the forced `ANSWER:`, both cameras, r1153, exactly like the light / stop-sign questions of vlm_qwen_server.py) tell
"a vehicle is closing on the ego's path from the side" on the saved closed-loop frames of the vmerge-family runs?

  python vm3_cross_offline.py label                 privileged-truth labels of every saved frame pair (CPU)
  python vm3_cross_offline.py score [--limit N]     P(positive) of every candidate prompt on the selected frames (GPU, resumable)
  python vm3_cross_offline.py report                dev choice, test numbers, collision episodes -> results/vm3_cross_offline.{md,csv}

Env: $DATA_DIR/envs/jevdrive/bin/python, CUDA_VISIBLE_DEVICES=2. Outputs: $DATA_DIR/runs/vlm_arb/vm3_cross/.

Pre-registered (written before any model answer was read):
- label: positive = some vehicle (not the ego, speed > 1 m/s, heading differs from the ego's by > 30 deg) whose constant-velocity
  centre track over the next 3.0 s (0.1 s steps) is inside the ego corridor (ego frame: 0-20 m ahead of the front bumper,
  |lateral| <= 2.5 m); snapshot = the last privileged row <= t_q. ttc = first time inside.
- main set: ego v < 0.5 m/s inside a junction window (state rules contain R2 or R3, or 0 <= junc_dist <= 10 m).
- split: routes sorted by id, alternate routes dev / test (DEV below); one prompt chosen by dev AUC on main-set frames.
- threshold: on dev main-set negatives, the P(positive) at which dev FPR = 0.10 (margin under the 0.15 line); also argmax.
- line: usable as a release check iff on the test main set, at the dev threshold, recall >= 0.6 and FPR <= 0.15.
"""
import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DATA = Path(os.environ.get("DATA_DIR", "/root/autodl-tmp/ujs"))
ARMS = DATA / "runs/vlm_arb/arms"
OUT = DATA / "runs/vlm_arb/vm3_cross"
RES_DIR = HERE.parent / "results"
UNIT_GLOBS = ["v2-vmerge-*", "v2-vmerge2-*", "v2-vmj-*", "v2-vmnobyp-*", "v2-vmnocusum-*", "v2-vmnor1-*", "v2-dbg*-vm*"]
HORIZON, DT, V_MIN, HEAD_MIN, LONG, LAT = 3.0, 0.1, 1.0, np.deg2rad(30), 20.0, 2.5
N_SECONDARY = 1500                                     # random other frames scored for the secondary all-frames readout
EPISODES = [("v2-vmerge-s1", "27297"), ("v2-vmerge-s1", "17280"), ("v2-vmerge2-s1", "17280"), ("v2-vmerge2-s2", "17280"),
            ("v2-vmerge2-s3", "17280")]

CAMS2 = ("The two images were taken at the same instant by the front cameras of a car (the ego vehicle): "
         "image 1 is the wide-angle camera, image 2 is the narrow road camera.")
TAIL = "End your reply with one line of the form `ANSWER: <option>`, <option> being one of: %s. Reply with that line only."


def mk(q, opts):
    return (CAMS2 + "\n" + q + "\n" + TAIL % ", ".join(o for o, _ in opts), [o for o, _ in opts])


# option 0 = positive (vehicle closing on the ego path)
PROMPTS = {
    "cross": mk("Cross traffic in front of the ego vehicle. Options:\n"
                "  vehicle_crossing: a moving vehicle is crossing, turning into, or about to enter the road directly in front of the ego vehicle\n"
                "  path_clear: no vehicle is crossing or about to cross in front of the ego vehicle (vehicles ahead driving away in the ego "
                "vehicle direction do not count)",
                [("vehicle_crossing", ""), ("path_clear", "")]),
    "go": mk("The ego vehicle is stopped at a junction. Can it start moving now without a vehicle coming from the left, the right or "
             "turning across its path? Options:\n"
             "  wait_for_vehicle: a vehicle is approaching from the side or turning across the ego vehicle path and will pass in front of it\n"
             "  safe_to_go: no vehicle is about to pass in front of the ego vehicle",
             [("wait_for_vehicle", ""), ("safe_to_go", "")]),
    "side": mk("Vehicles approaching the ego vehicle path from the side. Options:\n"
               "  side_vehicle_approaching: a moving vehicle coming from the left or the right, or an oncoming vehicle turning, will cross the "
               "lane directly ahead of the ego vehicle within the next few seconds\n"
               "  no_side_vehicle: no vehicle will cross the lane directly ahead of the ego vehicle",
               [("side_vehicle_approaching", ""), ("no_side_vehicle", "")]),
}


def jl(p):
    return [json.loads(x) for x in open(p)] if p.exists() else []


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def threat(ego, actors):
    """(positive, ttc s, n candidate vehicles, closest id) from one privileged snapshot."""
    ex, ey = ego["xyz"][:2]
    c, s = np.cos(ego["yaw"]), np.sin(ego["yaw"])
    front = ego["extent"][0]
    ts = np.arange(0, HORIZON + 1e-6, DT)
    best = (np.inf, None)
    for a in actors:
        if not a["type"].startswith("vehicle.") or a.get("id") == ego.get("id"):
            continue
        vx, vy = a["velocity"][:2]
        if np.hypot(vx, vy) <= V_MIN or abs(wrap(a["yaw"] - ego["yaw"])) <= HEAD_MIN:
            continue
        px, py = a["xyz"][0] + vx * ts - ex, a["xyz"][1] + vy * ts - ey
        lo, la = c * px + s * py - front, -s * px + c * py
        ins = (lo >= 0) & (lo <= LONG) & (np.abs(la) <= LAT)
        if ins.any() and ts[ins.argmax()] < best[0]:
            best = (float(ts[ins.argmax()]), a.get("id"))
    return np.isfinite(best[0]), best[0] if np.isfinite(best[0]) else np.nan, best[1]


def label_attempt(adir):
    adir = Path(adir)
    unit, route = adir.parts[-4], adir.parts[-2]
    P = jl(adir / "privileged.jsonl")
    D = jl(adir / "vlm_decisions.jsonl")
    if not P or not D:
        return [], []
    pt = np.array([p["t"] for p in P])
    S = [d for d in D if d.get("k") == "s"]
    st = np.array([d["t"] for d in S]) if S else np.zeros(0)
    # contacts: world clock -> scenario clock through the frame numbers
    pf = np.array([p["frame"] for p in P])
    cont = []
    for c in jl(adir / "contacts.jsonl"):
        k = int(np.abs(pf - c["frame"]).argmin())
        cont.append(dict(unit=unit, route=route, id=c["id"], type=c["type"], t_world=c["t"],
                         t=round(float(pt[k] + (c["frame"] - pf[k]) * 0.05), 3), impulse=c.get("impulse")))
    rows = []
    for d in D:
        if d.get("k") != "a" or not d.get("ans", {}).get("ok"):
            continue
        tq = d["t_q"]
        f = {k: adir / "vlm_frames" / ("%08.2f_%s.jpg" % (tq, k)) for k in ("wide", "road")}
        if not all(p.exists() for p in f.values()):
            continue
        i = int(np.searchsorted(pt, tq + 1e-4) - 1)
        if i < 0:
            continue
        p = P[i]
        pos, ttc, aid = threat(p["ego"], p["actors"])
        j = int(np.searchsorted(st, tq + 1e-4) - 1) if len(st) else -1
        s = S[j] if j >= 0 else {}
        rules, jd = s.get("rules", []), s.get("junc_dist")
        jwin = ("R2" in rules or "R3" in rules or (jd is not None and 0 <= jd <= 10))
        rows.append(dict(id="%s/%s/%.2f" % (unit, route, tq), unit=unit, route=route, t=round(tq, 3), dt_snap=round(tq - pt[i], 3),
                         v=round(p["ego"]["v"], 3), rules="+".join(rules), junc_dist=jd, jwin=jwin, pos=int(pos), ttc=ttc, threat_id=aid,
                         n_act=len(p["actors"]), wide=str(f["wide"]), road=str(f["road"])))
    return rows, cont


def attempts():
    out = []
    for g in UNIT_GLOBS:
        for u in sorted(ARMS.glob(g)):
            out += sorted(u.glob("attempts/*/*/"))
    return sorted(set(str(a) for a in out if (a / "vlm_frames").is_dir()))


def label(a):
    OUT.mkdir(parents=True, exist_ok=True)
    rows, cont = [], []
    with ProcessPoolExecutor(min(48, os.cpu_count())) as ex:
        for r, c in ex.map(label_attempt, attempts()):
            rows += r
            cont += c
    df = pd.DataFrame(rows).drop_duplicates("id").sort_values("id").reset_index(drop=True)
    df["main"] = (df.v < 0.5) & df.jwin
    routes = sorted(df.route.unique(), key=int)
    dev = set(routes[0::2])
    df["part"] = np.where(df.route.isin(dev), "dev", "test")
    df.to_csv(OUT / "labels.csv", index=False)
    pd.DataFrame(cont).to_csv(OUT / "contacts.csv", index=False)
    print("frames %d, positive %.3f; main %d, positive %.3f; dev routes %s" % (
        len(df), df.pos.mean(), df.main.sum(), df[df.main].pos.mean(), sorted(dev, key=int)))
    print(df[df.main].groupby(["part", "route"]).pos.agg(["size", "sum"]).to_string())


def selection(df, cont):
    """Frames to score: the main set, every frame of the collision episodes within 5 s before a contact, a fixed random
    subset of the rest (secondary readout)."""
    sel = df.main.copy()
    for u, r in EPISODES:
        c = cont[cont.unit.str.startswith(u + "-") & (cont.route.astype(str) == r)]
        for tc in c.t:
            sel |= df.unit.str.startswith(u + "-") & (df.route.astype(str) == r) & (df.t >= tc - 5) & (df.t <= tc + 0.5)
    rest = df.index[~sel]
    rng = np.random.default_rng(0)
    sel.loc[rng.choice(rest, min(N_SECONDARY, len(rest)), replace=False)] = True
    return sel


def score(a):
    sys.path.insert(0, str(HERE))
    from vlm_thin_common import RES, SCORE_PREFIX, Thin
    df = pd.read_csv(OUT / "labels.csv")
    cont = pd.read_csv(OUT / "contacts.csv")
    df = df[selection(df, cont)].reset_index(drop=True)
    # order: main dev first, then main test, episodes, secondary
    df["o"] = np.where(df.main, np.where(df.part == "dev", 0, 1), 2)
    df = df.sort_values(["o", "id"]).reset_index(drop=True)
    if a.limit:
        df = df.head(a.limit)
    outp = OUT / "scores.csv"
    done = set(pd.read_csv(outp).id) if outp.exists() else set()
    todo = df[~df.id.isin(done)]
    print("selected %d, done %d, todo %d" % (len(df), len(done), len(todo)), flush=True)
    th = Thin()
    th.restore()
    tok, torch = th.tok, th.torch
    pre = tok(SCORE_PREFIX, add_special_tokens=False).input_ids
    Q = {}
    for k, (prompt, opts) in PROMPTS.items():
        th.set_prompt(prompt)
        first = [tok(SCORE_PREFIX + " " + o, add_special_tokens=False).input_ids[len(pre)] for o in opts]
        assert len(set(first)) == len(opts), (k, first)
        Q[k] = (th.pieces, th.m.lm_head.weight[first].float())
    res = RES[a.res]
    f = open(outp, "a")
    if not done:
        f.write("id," + ",".join("p_" + k for k in PROMPTS) + "\n")
    t0, B = time.time(), a.batch
    rows = list(todo.itertuples())
    with torch.no_grad():
        for b in range(0, len(rows), B):
            chunk = rows[b:b + B]
            jp = [[Path(r.wide).read_bytes(), Path(r.road).read_bytes()] for r in chunk]
            ps = []
            for k, (pieces, W) in Q.items():
                th.pieces = pieces
                xs = [th.prep(j, res) for j in jp]
                if len(set(tuple(x["n_img"]) for x in xs)) > 1:
                    lg = torch.cat([run1(th, [x]) @ W.T for x in xs])
                else:
                    lg = run1(th, xs) @ W.T
                ps.append(torch.softmax(lg, -1)[:, 0].cpu().numpy())
            for i, r in enumerate(chunk):
                f.write(r.id + "," + ",".join("%.5f" % p[i] for p in ps) + "\n")
            f.flush()
            n = b + len(chunk)
            if n % (20 * B) < B or n == len(rows):
                el = time.time() - t0
                print("%s %d / %d, %.0f ms per frame (all prompts), eta %.1f min, max mem %.1f GB" % (
                    time.strftime("%H:%M:%S"), n, len(rows), 1e3 * el / n, el / n * (len(rows) - n) / 60,
                    torch.cuda.max_memory_allocated() / 2**30), flush=True)
    (OUT / "SCORE_DONE").write_text(time.strftime("%F %T"))


def run1(th, xs):
    t = th.torch
    cat = lambda k: t.cat([x[k] for x in xs])      # noqa: E731
    out = th.m.model(input_ids=cat("input_ids"), attention_mask=cat("attention_mask"), pixel_values=cat("pixel_values"),
                     image_grid_thw=cat("image_grid_thw"), mm_token_type_ids=cat("mm_token_type_ids"), use_cache=False)
    return out.last_hidden_state[:, -1].float()


# ---------------------------------------------------------------------------------------------------- report
def auc(y, s):
    y, s = np.asarray(y, bool), np.asarray(s, float)
    if y.all() or (~y).all():
        return float("nan")
    r = pd.Series(s).rank().to_numpy()
    n1 = y.sum()
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * (~y).sum()))


def boot_auc(d, col, n=1000):
    rng = np.random.default_rng(0)
    rs = d.route.unique()
    g = {r: d[d.route == r] for r in rs}
    v = []
    for _ in range(n):
        x = pd.concat([g[r] for r in rng.choice(rs, len(rs))])
        v.append(auc(x.pos, x[col]))
    v = np.array(v)
    v = v[np.isfinite(v)]
    return np.percentile(v, [2.5, 97.5]) if len(v) else (np.nan, np.nan)


def rates(d, col, thr):
    yhat = d[col] >= thr
    P, N = d.pos == 1, d.pos == 0
    return dict(n=len(d), n_pos=int(P.sum()), n_neg=int(N.sum()), recall=float((yhat & P).sum() / max(P.sum(), 1)),
                fpr=float((yhat & N).sum() / max(N.sum(), 1)))


def report(a):
    lab = pd.read_csv(OUT / "labels.csv")
    sc = pd.read_csv(OUT / "scores.csv").drop_duplicates("id")
    cont = pd.read_csv(OUT / "contacts.csv")
    d = lab.merge(sc, on="id")
    cols = ["p_" + k for k in PROMPTS]
    M = d[d.main]
    dev, test = M[M.part == "dev"], M[M.part == "test"]
    L = ["# vm3 cross-traffic release check: zero-shot Qwen3-VL-4B, offline on the vmerge-family frames", "",
         "Script: `experiments/vlm_arb/scripts/vm3_cross_offline.py` (label / score / report; pre-registration in its docstring). "
         "Frames: every saved light-request frame pair (wide + road) of the vmerge, vmerge2, vmj, vmnobyp, vmnocusum, vmnor1 and dbg vm* "
         "units. Scoring = the server's path (forced `ANSWER:`, first-token logits of the two options, softmax, r1153, whole model); "
         "P = softmax probability of the positive option.", "",
         "Label (privileged.jsonl, last snapshot <= t_q): positive = a vehicle (not ego, > 1 m/s, heading > 30 deg off the ego's) whose "
         "constant-velocity centre track within 3.0 s is inside the ego corridor (0-20 m ahead of the front bumper, |lat| <= 2.5 m). "
         "Main set = ego v < 0.5 m/s and junction window (rules R2 / R3, or 0 <= junc_dist <= 10 m). Routes alternate dev / test by id.", "",
         "## Label prevalence", "",
         "| set | frames | positive | prevalence |", "|:--|--:|--:|--:|"]
    for name, x in [("all saved frames", lab), ("main (junction stop)", lab[lab.main]), ("main dev", lab[lab.main & (lab.part == "dev")]),
                    ("main test", lab[lab.main & (lab.part == "test")])]:
        L.append("| %s | %d | %d | %.3f |" % (name, len(x), x.pos.sum(), x.pos.mean()))
    L += ["", "Positive ttc (main set): median %.1f s, %d of %d at ttc 0 (already in the corridor)." % (
        lab[lab.main & (lab.pos == 1)].ttc.median(), int((lab[lab.main & (lab.pos == 1)].ttc == 0).sum()), int(lab[lab.main].pos.sum())), ""]
    L += ["## Dev: prompt choice (main set, %d frames, %d positive, %d routes)" % (len(dev), dev.pos.sum(), dev.route.nunique()), "",
          "| prompt | dev AUC [route bootstrap 95%] | recall / FPR at argmax |", "|:--|--:|--:|"]
    best, ba = None, -1
    for c in cols:
        A = auc(dev.pos, dev[c])
        lo, hi = boot_auc(dev, c)
        r = rates(dev, c, 0.5)
        L.append("| %s | %.3f [%.3f, %.3f] | %.2f / %.2f |" % (c[2:], A, lo, hi, r["recall"], r["fpr"]))
        if A > ba:
            best, ba = c, A
    neg = dev[dev.pos == 0][best].to_numpy()
    thr = float(np.quantile(neg, 0.90)) if len(neg) else 0.5
    k = best[2:]
    L += ["", "Chosen: **%s** (highest dev AUC). Dev threshold (dev FPR = 0.10): P(positive) >= %.4f." % (k, thr), "",
          "Prompt (exact string):", "", "```", PROMPTS[k][0], "```", "", "Options: positive `%s`, negative `%s`." % tuple(PROMPTS[k][1]), ""]
    L += ["## Test (main set, the chosen prompt only)", "", "| set | n | n pos | AUC [95%] | recall @dev thr | FPR @dev thr | recall @argmax | FPR @argmax |",
          "|:--|--:|--:|--:|--:|--:|--:|--:|"]
    rows = []
    for name, x in [("test main", test), ("dev main", dev), ("all scored frames (secondary)", d),
                    ("all scored, ego moving (secondary)", d[d.v >= 0.5])]:
        lo, hi = boot_auc(x, best)
        r1, r2 = rates(x, best, thr), rates(x, best, 0.5)
        L.append("| %s | %d | %d | %.3f [%.3f, %.3f] | %.2f | %.2f | %.2f | %.2f |" % (
            name, len(x), r1["n_pos"], auc(x.pos, x[best]), lo, hi, r1["recall"], r1["fpr"], r2["recall"], r2["fpr"]))
        rows.append(dict(set=name, prompt=k, thr=thr, auc=auc(x.pos, x[best]), auc_lo=lo, auc_hi=hi, **{"thr_" + q: v for q, v in r1.items()},
                         **{"argmax_" + q: v for q, v in r2.items() if q in ("recall", "fpr")}))
    for c in cols:
        if c != best:
            L.append("")
            L.append("Unchosen `%s` on test main (for the record, not used): AUC %.3f." % (c[2:], auc(test.pos, test[c])))
    rt = rates(test, best, thr)
    ok = rt["recall"] >= 0.6 and rt["fpr"] <= 0.15
    L += ["", "**Pre-registered line** (test main, dev threshold: recall >= 0.6 and FPR <= 0.15): recall %.2f, FPR %.2f (n pos %d, n neg %d) -> **%s**." % (
        rt["recall"], rt["fpr"], rt["n_pos"], rt["n_neg"], "usable" if ok else "not usable"), ""]
    # recall by ttc on test main
    tp = test[test.pos == 1]
    if len(tp):
        L += ["Test main positives by time-to-corridor: " + ", ".join(
            "%s: %d/%d" % (lab_, int((g[best] >= thr).sum()), len(g)) for lab_, g in tp.groupby(pd.cut(tp.ttc, [-0.01, 0.5, 1.5, 3.01],
                                                                                                         labels=["<=0.5 s", "0.5-1.5 s", "1.5-3 s"]), observed=False)), ""]
    # collision episodes
    L += ["## Collision episodes: answers in the 3 s before contact", "",
          "Contact time = scenario clock (contacts.jsonl frame -> privileged frame). P = P(%s); flag = P >= dev threshold." % PROMPTS[k][1][0], "",
          "| unit | route | contact t | other | frame t | ego v | label (ttc) | P | flag |", "|:--|:--|--:|:--|--:|--:|:--|--:|:--|"]
    ep_rows = []
    for u, r in EPISODES:
        c = cont[cont.unit.str.startswith(u + "-") & (cont.route.astype(str) == r)].sort_values("t")
        if not len(c):
            L.append("| %s | %s | no contact | | | | | | |" % (u, r))
            continue
        first = c.groupby("id").t.min().sort_values()
        for aid, tc in first.items():
            typ = c[c.id == aid].type.iloc[0].replace("vehicle.", "")
            x = d[d.unit.str.startswith(u + "-") & (d.route.astype(str) == r) & (d.t >= tc - 3) & (d.t <= tc)].sort_values("t")
            if not len(x):
                L.append("| %s | %s | %.2f | %s | no frame | | | | |" % (u, r, tc, typ))
            for e in x.itertuples():
                L.append("| %s | %s | %.2f | %s | %.2f | %.2f | %s | %.3f | %s |" % (
                    u, r, tc, typ, e.t, e.v, ("pos (%.1f s)" % e.ttc) if e.pos else "neg", getattr(e, best), "STOP" if getattr(e, best) >= thr else "-"))
                ep_rows.append(dict(unit=u, route=r, t_contact=tc, other=typ, t=e.t, v=e.v, pos=e.pos, ttc=e.ttc, p=getattr(e, best)))
    pd.DataFrame(rows).to_csv(RES_DIR / "vm3_cross_offline.csv", index=False)
    pd.DataFrame(ep_rows).to_csv(RES_DIR / "vm3_cross_episodes.csv", index=False)
    (RES_DIR / "vm3_cross_offline.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["label", "score", "report"])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--res", default="r1153")
    a = ap.parse_args()
    {"label": label, "score": score, "report": report}[a.cmd](a)
