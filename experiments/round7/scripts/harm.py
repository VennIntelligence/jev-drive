"""Harmonizer open-loop check (plans/2026-10-21-harmonizer-check.md): the same PAI human states rendered OFF, OFF again and ON (NuRec
DiffusionHarmonizer), encoded by the normal cache path; token distances and plan deltas against the OFF / OFF re-render floor.

  select   the H rows of pai_rows (10-20, $DATA_DIR/runs/pai_rows/rows/H), train scenes of scenes.tsv (held False) -> <out>/states.tsv
           (scene, k, stratum) and <out>/chunks/cNNN.txt (c000 = one scene, the smoke; the rest CHUNK scenes each). Strata from the human's
           logged 4 s future (log_fut) and v0, first match wins: launch, stop, turn, curve, straight; 60 a stratum, at most 6 a scene per
           stratum, at least 5 decisions apart. Seeded (0): the selection is fixed before anything is rendered
  compare  <cache root>/<variant>/{tab.npz, tok.npy} (onp_cache build) of OFFA, OFFB, ON + the row files -> <out>/cmp.json, <out>/cmp.md:
           frame bytes, newest-pair token cosine distance / relative L2, plan deltas (rear-axle x = longitudinal, y = lateral) at 0.5 / 2 / 4 s
           per checkpoint (each variant's own tokens and ego, the onp_cache gid export), per stratum and pooled; verdict per the plan
  report   cmp.json (+ cost.json) -> the tables of results/harmonizer_openloop.md (stdout); runs anywhere (no torch)
  png      side-by-side OFF-A | ON | 4 x |diff| of the driver camera's newest frame (the model input, 512 x 256) for one state per stratum + the
           largest NTV0 4 s lateral delta -> <out>/png/*.png (palette PNG)

  $DATA_DIR/envs/op-train/bin/python experiments/round7/scripts/harm.py select --out $DATA_DIR/runs/round7/harm
"""
import sys as _sys, pathlib as _pl  # noqa: E401
_R = _pl.Path(__file__).resolve().parents[3]
_sys.path[:0] = [str(_R), str(_R / "lib"), str(_R / "scripts"), str(_R / "experiments/op_adapt_r2/lib"), str(_R / "experiments/op_parity/scripts"),
                 str(_R / "experiments/body1/lib"), str(_R / "experiments/onpol_pai/lib")]   # = onp_cache.py's
import argparse, csv, json  # noqa: E401,E402

import numpy as np  # noqa: E402

STRATA = ("launch", "stop", "turn", "curve", "straight")
VARIANTS = ("OFFA", "OFFB", "ON")
H_IDX = {"0.5": 0, "2": 3, "4": 7}                   # T8 = 0.5 .. 4.0 s
ABS_LINE = {"0.5": 0.05, "2": 0.15, "4": 0.3}         # the plan's fallback when the floor is degenerate (< 0.01 m)
KEYS = ("k", "v0", "log_fut", "log_fut_ok", "valid", "slot", "after_event", "after_end")


def strata(z) -> np.ndarray:
    """Per row: index into STRATA, -1 = none / not usable."""
    lf, v0 = z["log_fut"].astype(np.float64), z["v0"].astype(np.float64)
    ok = z["valid"].all(1) & ~(z["slot"] < 0).any(1) & ~z["after_event"] & ~z["after_end"] & z["log_fut_ok"].all(1)
    yaw4, d4, lat4 = np.abs(lf[:, 7, 2]), np.hypot(lf[:, 7, 0], lf[:, 7, 1]), np.abs(lf[:, 7, 1])
    vend = np.hypot(*(lf[:, 7, :2] - lf[:, 6, :2]).T) / 0.5
    rules = [(v0 < 0.3) & (d4 > 2.0),
             (v0 > 3) & (vend < 1.0) & (yaw4 < np.radians(15)),
             (yaw4 > np.radians(35)) & (v0 > 1),
             (yaw4 > np.radians(6)) & (yaw4 < np.radians(20)) & (v0 > 5) & (vend > 0.7 * v0),
             (yaw4 < np.radians(1.7)) & (lat4 < 0.5) & (v0 > 5) & (vend > 0.7 * v0)]
    s = np.full(len(v0), -1)
    for i, r in reversed(list(enumerate(rules))):
        s[r & ok] = i
    return s


def spaced(ks, rng, n, gap=5):
    out = []
    for k in rng.permutation(ks):
        if all(abs(int(k) - j) >= gap for j in out):
            out.append(int(k))
        if len(out) == n:
            break
    return sorted(out)


def cmd_select(a):
    rng = np.random.default_rng(0)
    out = _pl.Path(a.out)
    (out / "chunks").mkdir(parents=True, exist_ok=True)
    tsv = list(csv.DictReader(open(_R / "experiments/pai_rows/results/scenes.tsv"), delimiter="\t"))
    train = [r["scene_id"] for r in tsv if r["held"] == "False" and (_pl.Path(a.rows) / f"{r['scene_id']}.npz").exists()]
    elig = {}                                                       # scene -> {stratum: [k]}
    for s in train:
        with np.load(_pl.Path(a.rows) / f"{s}.npz") as z:
            d = {q: z[q] for q in KEYS}
        st = strata(d)
        elig[s] = {i: d["k"][st == i].tolist() for i in range(len(STRATA)) if (st == i).sum()}
    picked, chosen = [], []                                         # rarest stratum first; scenes already chosen are drawn from first
    have = {i: sum(len(v.get(i, [])) > 0 for v in elig.values()) for i in range(len(STRATA))}
    for i in sorted(range(len(STRATA)), key=lambda i: have[i]):
        n = sum(1 for p in picked if p[2] == i)
        pool = [s for s in chosen if i in elig[s]] + [s for s in rng.permutation(sorted(elig)) if i in elig[s] and s not in chosen]
        for s in pool:
            if n >= a.n:
                break
            ks = spaced(elig[s][i], rng, min(a.per_scene, a.n - n))
            taken = {p[1] for p in picked if p[0] == s}
            ks = [k for k in ks if all(abs(k - j) >= 5 for j in taken)]
            if len(ks) < min(3, a.n - n):                           # a scene must give >= 3 of a stratum to join for it
                continue
            picked += [(s, k, i) for k in ks]
            n += len(ks)
            if s not in chosen:
                chosen.append(s)
    picked.sort()
    with open(out / "states.tsv", "w") as f:
        f.write("scene\tk\tstratum\n" + "".join(f"{s}\t{k}\t{STRATA[i]}\n" for s, k, i in picked))
    sc = sorted(chosen)
    rng.shuffle(sc)
    chunks = [sc[:1]] + [sc[1 + j:1 + j + a.chunk] for j in range(0, len(sc) - 1, a.chunk)]
    for j, c in enumerate(chunks):
        (out / "chunks" / f"c{j:03d}.txt").write_text("\n".join(c) + "\n")
    cnt = {STRATA[i]: sum(1 for p in picked if p[2] == i) for i in range(len(STRATA))}
    json.dump(dict(states=len(picked), scenes=len(sc), chunks=len(chunks), per_stratum=cnt, eligible_scenes={STRATA[i]: have[i] for i in have},
                   train_scenes=len(train)), open(out / "select.json", "w"), indent=1)
    print(json.dumps(cnt), len(picked), "states,", len(sc), "scenes,", len(chunks), "chunks")


def q(x, p):
    return float(np.quantile(x, p)) if len(x) else float("nan")


def boot_med(x, rng, n=2000):
    x = np.asarray(x)
    m = np.median(x[rng.integers(0, len(x), (n, len(x)))], 1)
    return [float(np.quantile(m, 0.025)), float(np.quantile(m, 0.975))]


def cmd_compare(a):
    import torch
    import onp_loss as OL
    import pp_train as T
    from experiments.op_adapt_r2.lib import op_adapt_r2 as R2
    dev, out, rng = torch.device("cuda"), _pl.Path(a.out), np.random.default_rng(0)
    st = list(csv.DictReader(open(out / "states.tsv"), delimiter="\t"))
    key = [(r["scene"], int(r["k"])) for r in st]
    lab = np.array([STRATA.index(r["stratum"]) for r in st])
    R, rows = {}, {}
    for v in VARIANTS:
        R[v] = OL.OnRows(_pl.Path(a.cache), [v], dev)
        ix = {(s, int(k)): i for i, (s, k) in enumerate(zip(R[v].scene, R[v].k))}
        rows[v] = np.array([ix.get(x, -1) for x in key])
    have = np.all([rows[v] >= 0 for v in VARIANTS], 0)
    # route-memory checkpoints: plans from rf_recov.py fwd on the pools on1_HARM<V> (rf_pai_pool: the driver's route messages -> rmem features)
    from jevdrive.common import data_dir
    fp = {}
    for tag in [t for t in a.tags if t not in ("P0", "served")]:
        fp[tag] = {}
        for v in VARIANTS:
            nm = np.load(data_dir() / "runs/op_parity/cache" / f"on1_HARM{v}" / "tab.npz")["names"].astype(str)
            P = np.full((len(nm), 8, 3), np.nan, np.float32)
            for rs in ("train-rec", "dev-rec"):
                f = out / "fwd" / f"HARM{v}-{rs}" / f"plans_{tag}.npz"
                if f.exists():
                    z = np.load(f)
                    P[z["rows"]] = z["poses"]
            ix = {x: i for i, x in enumerate(nm)}
            j = np.array([ix.get(f"on:harm:{s}:{k}", -1) for s, k in key])
            fp[tag][v] = np.where((j >= 0)[:, None, None], P[np.maximum(j, 0)], np.nan)
            have &= np.isfinite(fp[tag][v]).all((1, 2))
    res = dict(states=len(key), matched=int(have.sum()), missing=[f"{s} {k}" for (s, k), h in zip(key, have) if not h][:20])
    key, lab = [x for x, h in zip(key, have) if h], lab[have]
    for v in VARIANTS:
        rows[v] = rows[v][have]
        for tag in fp:
            fp[tag][v] = fp[tag][v][have]
    n = len(key)
    # ---- frame bytes (the model input), newest slot
    fb = {"OFFB": [], "ON": []}
    for s in sorted({x[0] for x in key}):
        Z = {v: np.load(_pl.Path(a.rows) / v / f"{s}.npz") for v in VARIANTS}
        F = {v: (Z[v]["frames"], Z[v]["slot"], Z[v]["k"]) for v in VARIANTS}
        for (s2, k) in key:
            if s2 != s:
                continue
            img = {}
            for v, (fr, sl, kk) in F.items():
                img[v] = fr[sl[np.flatnonzero(kk == k)[0], 7]].astype(np.int16)
            for v in fb:
                d = np.abs(img[v] - img["OFFA"])
                fb[v].append((float(d.mean()), float((d > 0).mean())))
    res["frame"] = {v: dict(mean_abs_grey=float(np.mean([x[0] for x in fb[v]])), share_bytes_diff=float(np.mean([x[1] for x in fb[v]])),
                            states_identical=int(sum(x[1] == 0 for x in fb[v]))) for v in fb}
    # ---- ego inputs
    e = {v: R[v].ego[torch.as_tensor(rows[v], device=dev)].cpu().numpy() for v in VARIANTS}
    res["ego_maxabs"] = {v: float(np.abs(e[v] - e["OFFA"]).max()) for v in ("OFFB", "ON")}
    # ---- tokens: newest pair (slot 7) and the mean over the 8 slots
    tk = {v: R[v].tokens(rows[v]).float().numpy().reshape(n, 8, -1) for v in VARIANTS}
    def tdist(b):
        A, B = tk["OFFA"], tk[b]
        cos = 1 - (A * B).sum(-1) / (np.linalg.norm(A, axis=-1) * np.linalg.norm(B, axis=-1) + 1e-12)
        rl2 = np.linalg.norm(A - B, axis=-1) / (np.linalg.norm(A, axis=-1) + 1e-12)
        return cos, rl2
    tok = {}
    for b in ("OFFB", "ON"):
        cos, rl2 = tdist(b)
        tok[b] = dict(cos7=cos[:, 7], rl27=rl2[:, 7], cos_all=cos.mean(1), rl2_all=rl2.mean(1))
    # ---- plans
    W = torch.as_tensor(R2.t_weights(T.T8), device=dev)
    plans = {}
    for tag in a.tags:
        if tag in fp:
            plans[tag] = fp[tag]
            continue
        if tag == "served":                                            # NTV0-L18-s0 as the driver served it in each render (drive.jsonl poses_model)
            plans[tag] = {v: R[v].poses_model[torch.as_tensor(rows[v], device=dev)].cpu().numpy() for v in VARIANTS}
            continue
        m = T.load_pmodel(tag, dev)                                    # P0: shipped Cinque, no route input
        pi = torch.as_tensor(np.arange(m.net.slices["plan"].start, m.net.slices["plan"].start + 495), device=dev)
        plans[tag] = {}
        with torch.no_grad():
            for v in VARIANTS:
                r = torch.as_tensor(rows[v], device=dev)
                o = m(R[v].tokens(rows[v]).to(dev), R[v].ego[r], R[v].tc[r]).float()
                plans[tag][v] = torch.stack(OL.rear_xy(o[:, pi].view(-1, 33, 15), R[v].cam[r, :2], W), -1).cpu().numpy()   # (n, 8, 3)
        del m
    if "served" in plans and "NTV0-L18-s0" in plans:                   # the fwd path against what the driver served, per render
        res["fwd_vs_served_max_m"] = {v: float(np.hypot(*(plans["NTV0-L18-s0"][v][..., :2] - plans["served"][v][..., :2]).transpose(2, 0, 1)).max()) for v in VARIANTS}
    groups = [("all", np.ones(n, bool))] + [(s, lab == i) for i, s in enumerate(STRATA)]
    res["n"] = {g: int(mk.sum()) for g, mk in groups}
    res["tokens"] = {g: {b: {k: dict(med=q(x[mk], 0.5), p90=q(x[mk], 0.9)) for k, x in tok[b].items()} for b in tok} for g, mk in groups}
    res["plans"], res["verdict"] = {}, {}
    for tag in a.tags:
        P = plans[tag]
        D = {b: {"lat": np.abs(P[b][..., 1] - P["OFFA"][..., 1]), "lon": np.abs(P[b][..., 0] - P["OFFA"][..., 0])} for b in ("OFFB", "ON")}
        res["plans"][tag] = {g: {b: {ax: {h: dict(med=q(D[b][ax][mk, i], 0.5), p90=q(D[b][ax][mk, i], 0.9)) for h, i in H_IDX.items()} for ax in ("lat", "lon")}
                                 for b in D} for g, mk in groups}
        res["plans"][tag]["ci_all_on_4s"] = {ax: boot_med(D["ON"][ax][:, 7], rng) for ax in ("lat", "lon")}
        res["plans"][tag]["ci_all_floor_4s"] = {ax: boot_med(D["OFFB"][ax][:, 7], rng) for ax in ("lat", "lon")}
        # verdict (plan): pooled, 6 cells, ON median <= 2 x floor median (or the absolute line when the floor median < 0.01 m), and lateral 4 s < 0.2 m
        cells, ok = [], True
        for ax in ("lat", "lon"):
            for h in H_IDX:
                on, fl = res["plans"][tag]["all"]["ON"][ax][h]["med"], res["plans"][tag]["all"]["OFFB"][ax][h]["med"]
                line = ABS_LINE[h] if fl < 0.01 else 2 * fl
                cells.append(dict(ax=ax, h=h, on=on, floor=fl, line=line, rule="abs" if fl < 0.01 else "2x floor", ok=bool(on <= line)))
                ok &= on <= line
        lat4 = res["plans"][tag]["all"]["ON"]["lat"]["4"]["med"]
        flags = []
        for s in STRATA:
            c = res["plans"][tag][s]
            if c["ON"]["lat"]["4"]["med"] >= 0.2:
                flags.append(f"{s}: lateral 4 s median {c['ON']['lat']['4']['med']:.3f} m >= 0.2")
            fl, on = c["OFFB"]["lon"]["4"]["med"], c["ON"]["lon"]["4"]["med"]
            if on > max(2 * fl, 0.3 if fl >= 0.01 else ABS_LINE["4"]) and on > 0.3:
                flags.append(f"{s}: longitudinal 4 s median {on:.3f} m > 2 x floor {fl:.3f} and > 0.3")
        res["verdict"][tag] = dict(cells=cells, cells_ok=bool(ok), lat4_med=lat4, lat4_ok=bool(lat4 < 0.2), small=bool(ok and lat4 < 0.2), stratum_flags=flags)
    judged = [t for t in a.tags if t not in ("P0", "served")]
    res["overall"] = "small" if all(res["verdict"][t]["small"] for t in judged) else "closed-loop paired read"
    # per-state deltas for png choice and later looks
    np.savez(out / "cmp_states.npz", scene=np.array([x[0] for x in key]), k=np.array([x[1] for x in key]), stratum=lab,
             **{f"{t}_{v}": plans[t][v] for t in a.tags for v in VARIANTS}, **{f"tok_{b}_{k}": x for b in tok for k, x in tok[b].items()})
    (out / "cmp.json").write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps(dict(overall=res["overall"], frame=res["frame"], ego=res["ego_maxabs"], verdict={t: (v["small"], v["lat4_med"]) for t, v in res["verdict"].items()})))


def cmd_png(a):
    import torch
    from PIL import Image
    from jevdrive.op_interp import to_rgb
    out = _pl.Path(a.out)
    z = np.load(out / "cmp_states.npz")
    tag = a.tag
    lat4 = np.abs(z[f"{tag}_ON"][:, 7, 1] - z[f"{tag}_OFFA"][:, 7, 1])
    rng = np.random.default_rng(0)
    pick = [int(rng.choice(np.flatnonzero(z["stratum"] == i))) for i in range(len(STRATA)) if (z["stratum"] == i).any()]
    pick.append(int(np.argmax(np.where(np.isin(np.arange(len(lat4)), pick), -1, lat4))))
    (out / "png").mkdir(exist_ok=True)
    lst = []
    for j, i in enumerate(pick):
        s, k = str(z["scene"][i]), int(z["k"][i])
        im = {}
        for v in ("OFFA", "ON"):
            with np.load(_pl.Path(a.rows) / v / f"{s}.npz") as r:
                row = np.flatnonzero(r["k"] == k)[0]
                im[v] = to_rgb(torch.from_numpy(r["frames"][r["slot"][row, 7]][0])).permute(1, 2, 0).numpy()
        d = np.clip(4 * np.abs(im["ON"] - im["OFFA"]), 0, 1)
        img = Image.fromarray((255 * np.concatenate([im["OFFA"], im["ON"], d], 1)).astype(np.uint8)).quantize(128)
        name = f"{j}_{STRATA[z['stratum'][i]]}_{s[7:15]}_k{k}.png"
        img.save(out / "png" / name, optimize=True)
        lst.append(dict(png=name, scene=s, k=k, stratum=STRATA[z["stratum"][i]], lat4_delta_m=float(lat4[i]), largest=j == len(pick) - 1))
    (out / "png" / "list.json").write_text(json.dumps(lst, indent=1) + "\n")
    print(json.dumps(lst))


def cmd_report(a):
    r = json.load(open(a.cmp))
    f = lambda x: f"{x:.3f}"  # noqa: E731
    o = [f"States matched in all three renders: {r['matched']} / {r['states']}. Overall verdict: **{r['overall']}**.\n",
         "| Render pair | Mean abs grey diff (newest frame, model input) | Share of bytes differing | States bit-identical |\n|---|--:|--:|--:|\n"
         + "".join(f"| {b} vs OFFA | {v['mean_abs_grey']:.3f} | {v['share_bytes_diff']:.4f} | {v['states_identical']} |\n" for b, v in r["frame"].items()),
         f"Ego input max abs difference vs OFFA: OFFB {r['ego_maxabs']['OFFB']:.2e}, ON {r['ego_maxabs']['ON']:.2e}. "
         f"NTV0 fwd path vs the plan the driver served, max per render: {r.get('fwd_vs_served_max_m')}.\n"]
    for tag, V in r["verdict"].items():
        P = r["plans"][tag]
        o.append(f"### {tag}{' (reference, not judged)' if tag in ('P0', 'served') else ''}: {'small' if V['small'] else 'NOT small'}\n")
        o.append("Plan |delta| median (p90) in m, ON vs OFFA against the floor OFFB vs OFFA. Lat = rear-axle y, lon = rear-axle x.\n")
        o.append("| Stratum | n | " + " | ".join(f"{ax} {h} s ON / floor" for ax in ("lat", "lon") for h in H_IDX) + " |\n|---|--:|" + "--:|" * 6 + "\n")
        for g in ("all",) + STRATA:
            o.append(f"| {g} | {r['n'][g]} | " + " | ".join(f"{f(P[g]['ON'][ax][h]['med'])} ({f(P[g]['ON'][ax][h]['p90'])}) / {f(P[g]['OFFB'][ax][h]['med'])} ({f(P[g]['OFFB'][ax][h]['p90'])})"
                                                          for ax in ("lat", "lon") for h in H_IDX) + " |\n")
        o.append(f"\nPooled 4 s median, 95 % bootstrap CI: ON lat {P['ci_all_on_4s']['lat']}, lon {P['ci_all_on_4s']['lon']}; floor lat {P['ci_all_floor_4s']['lat']}, lon {P['ci_all_floor_4s']['lon']}.\n")
        o.append("Verdict cells: " + "; ".join(f"{c['ax']} {c['h']} s {f(c['on'])} vs line {f(c['line'])} ({c['rule']}) {'ok' if c['ok'] else 'OVER'}" for c in V["cells"])
                 + f"; lateral 4 s median {f(V['lat4_med'])} {'< 0.2 ok' if V['lat4_ok'] else '>= 0.2 OVER'}. Stratum flags: {'; '.join(V['stratum_flags']) or 'none'}.\n")
    o.append("### Tokens (newest pair, 32 x 512)\n\n| Stratum | n | 1 - cos ON | 1 - cos floor | rel L2 ON | rel L2 floor | rel L2 ON, mean of 8 slots | rel L2 floor, mean of 8 |\n|---|--:|--:|--:|--:|--:|--:|--:|\n")
    for g in ("all",) + STRATA:
        t = r["tokens"][g]
        o.append(f"| {g} | {r['n'][g]} | {t['ON']['cos7']['med']:.4f} ({t['ON']['cos7']['p90']:.4f}) | {t['OFFB']['cos7']['med']:.4f} ({t['OFFB']['cos7']['p90']:.4f}) | "
                 f"{t['ON']['rl27']['med']:.4f} ({t['ON']['rl27']['p90']:.4f}) | {t['OFFB']['rl27']['med']:.4f} ({t['OFFB']['rl27']['p90']:.4f}) | "
                 f"{t['ON']['rl2_all']['med']:.4f} | {t['OFFB']['rl2_all']['med']:.4f} |\n")
    print("\n".join(o))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("select")
    p.add_argument("--out", required=True)
    p.add_argument("--rows", default=str(_pl.Path(__import__("os").environ.get("DATA_DIR", "/root/autodl-tmp/ujs")) / "runs/pai_rows/rows/H"))
    p.add_argument("--n", type=int, default=60)
    p.add_argument("--per-scene", type=int, default=6)
    p.add_argument("--chunk", type=int, default=9)
    p = sub.add_parser("compare")
    p.add_argument("--out", required=True)
    p.add_argument("--cache", required=True)
    p.add_argument("--rows", required=True, help="dir with <variant>/<scene>.npz")
    p.add_argument("--tags", nargs="+", default=["NTV0-L18-s0", "R5LC-L18-s0", "P0", "served"], help="route checkpoints read from <out>/fwd; P0 forward here; served = the driver's logged NTV0 plan")
    p = sub.add_parser("report")
    p.add_argument("--cmp", required=True)
    p = sub.add_parser("png")
    p.add_argument("--out", required=True)
    p.add_argument("--rows", required=True)
    p.add_argument("--tag", default="NTV0-L18-s0")
    a = ap.parse_args()
    {"select": cmd_select, "compare": cmd_compare, "report": cmd_report, "png": cmd_png}[a.cmd](a)
