"""Real vs render analysis for HUGSIM Waymo / PandaSet / KITTI-360: the pre-registered readouts of rvr_analyze.py (gap vs the adjacent-frame
floor with scene-cluster bootstrap CIs, history-yaw gains G and launch gain L real vs render, by speed bin), per dataset and pooled with the
nuScenes numbers kept separate. Reads $DATA_DIR/runs/real_vs_render/<ds>/<scene>/{stream,probe}.npz, writes stats_<ds>.json.
Project venv, CPU: .venv/bin/python experiments/leaderboard_audit/scripts/rvr_ds_analyze.py [waymo pandaset kitti360]"""
import json, os, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rvr_analyze as A  # noqa: E402

D = Path(os.environ.get("DATA_DIR", "/root/autodl-tmp/ujs"))
RUN = D / "runs/real_vs_render"
A.FIRST = 30                       # frames >= 3 s at 10 Hz
f3 = A.f3


def load(ds):
    S = {}
    for d in sorted((RUN / ds).iterdir()):
        if not (d / "stream.npz").exists():
            continue
        z = np.load(d / "stream.npz")
        info = json.loads(str(z["info"]))
        S[d.name] = dict(v=z["v"], **{a: A.heads(z[a], info["heads_slices"]) for a in info["arms"]})
    return S


def probes(ds):
    P = {d.name: np.load(d / "probe.npz") for d in sorted((RUN / ds).iterdir()) if (d / "probe.npz").exists()}
    if not P:
        return {}
    nm = list(next(iter(P.values()))["names"])
    ix = {n: nm.index(n) for n in nm}

    def gains(z, arm):
        r = z[arm]
        return dict(G1=(r[:, ix["g1L"]] - r[:, ix["g1R"]]) / 2, G10=(r[:, ix["g10L"]] - r[:, ix["g10R"]]) / 2,
                    L=(r[:, ix["launchL"]] - r[:, ix["launchR"]]) / 2, H0=r[:, ix["normal"]])
    pr = {}
    for g in ("G1", "G10", "L"):
        for bn, lo, hi in (("stop", -1, 0.5), ("low", 0.5, 3), ("mid", 3, 99), ("all", -1, 99)):
            grp = []
            for z in P.values():
                a, b = gains(z, "real")[g], gains(z, "render")[g]
                m = (z["v"] >= lo) & (z["v"] < hi)
                if m.any():
                    grp.append((a[m], b[m]))
            if not grp:
                continue
            est = A.boot(lambda gs: np.mean(np.concatenate([b for _, b in gs])) / np.mean(np.concatenate([a for a, _ in gs])), grp)
            pr[f"{g}|{bn}"] = dict(real=float(np.mean(np.concatenate([a for a, _ in grp]))), render=float(np.mean(np.concatenate([b for _, b in grp]))),
                                   ratio=est, n=int(sum(len(a) for a, _ in grp)), scenes=len(grp),
                                   scenes_up=int(sum(np.mean(b) > np.mean(a) for a, b in grp)))
    pr["normal_heading_absdiff_deg"] = A.boot(lambda gs: np.median(np.concatenate(gs)), [np.abs(gains(z, "render")["H0"] - gains(z, "real")["H0"]) for z in P.values()])
    return pr


def main(dss):
    for ds in dss:
        S = load(ds)
        if not S:
            continue
        RR, RS = A.pairs(S, "real", "real", 1), A.pairs(S, "real", "render")
        gap = {}
        for m in A.METRICS:
            r = A.ratio_stat(RS[m], RR[m])
            gap[m] = dict(rr=float(np.median(RR[m][0])), rs=float(np.median(RS[m][0])), ratio=r, n=int(len(RS[m][0])), meaningful=bool(r[1] > 1))
        n_yes = sum(gap[m]["meaningful"] for m in A.PRIMARY)
        sh = {}
        for m in ("plan_v0", "plan_x4", "lane_w10", "road_z", "lane_p", "plan_lat4", "yaw3"):
            g = []
            for x in S.values():
                k = np.arange(A.FIRST, len(x["v"]))
                d = x["render"][m][k] - x["real"][m][k]
                d = d.mean(1) if m != "lane_p" else d[:, 1:3].mean(1)
                g.append(d[np.isfinite(d)])
            sh[m] = A.boot(lambda gs: np.median(np.concatenate(gs)), [x for x in g if len(x)])
        info = json.loads(str(np.load(sorted((RUN / ds).glob("*/stream.npz"))[0])["info"]))
        out = dict(n_scenes=len(S), scenes=list(S), verdict_gap="yes" if n_yes >= 2 else "partly" if n_yes == 1 else "no", gap=gap,
                   shift_render_minus_real=sh, probes=probes(ds), coverage=info.get("coverage"))
        (RUN / f"stats_{ds}.json").write_text(json.dumps(out, indent=1, default=float))
        print(f"\n== {ds}: {len(S)} scenes, verdict {out['verdict_gap']}")
        print("| readout | floor RR | RS | RS / RR [95% CI] | n | meaningful |")
        for m, r in gap.items():
            print(f"| {m} | {r['rr']:.3f} | {r['rs']:.3f} | {r['ratio'][0]:.2f} {f3(r['ratio'])} | {r['n']} | {r['meaningful']} |")
        print("signed shifts render - real:", {m: (round(r[0], 3), f3(r)) for m, r in sh.items()})
        for k, v in out["probes"].items():
            print(k, v)


if __name__ == "__main__":
    main(sys.argv[1:] or ["waymo", "pandaset", "kitti360"])
