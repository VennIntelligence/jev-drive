#!/usr/bin/env python3
"""ROUND9 POOL-DESIGN (plans/2026-10-23-pool-design.md): offline evidence for the S-row pool of a clean single-run recipe (NTV1).
CPU only, stored files; no training, no closed loop, no pool jobs. Box, repo .venv. Outputs: $DATA_DIR/runs/round9/pool_design/
-> experiments/round9/results/pool_design/ (json + md).

  nuplan   the nuPlan S pool on1_JT, part by part (rb0 = RB0-F-s0's own rollouts, the only part a clean chain may use; rtb1 / y / ys = other
           lineages), from the S audit's per-row file (runs/s_rows_audit/rows_audit.parquet: NTV0-L18-s0 forward terms, RB0-F-s0 forward
           agent / road, the stored served plan's agent / road, the recovery path's agent / road, source scene outcome):
           composition (speed band, standing, decision index, log-replay decisions, situation, source outcome), activity and learnability
           under the roller itself (RB0-F) and under the trained model (NTV0), hinge mass, lateral-term coverage, draws per live row
  pai      speed bands and standing share of the PAI on-policy states we have (on1_PAID: R5LC-L18-s0 under ACC + GAP on the 719 training
           scenes, 10 Hz decisions) and of the PAI human rows on the same scenes (on1_PAIH), against the PAI 281 test states (behavior_stats
           steps of RB0-F-s0 / NTV0-L18-s0 reads, 0.5 s steps) and the 281 human stations; near-duplicate structure of 10 Hz rows
  road     road class of the 719 training scenes (behavior_stats.scene_facts on pool_map.py's maps_train.pkl, same rules as Q1) against
           the 281 exam (behavior_stats out/stations.parquet)
  report   all of the above -> pool_design.json + pool_design.md
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

DATA = Path(os.environ.get("DATA_DIR", "/root/autodl-tmp/ujs"))
O = DATA / "runs/round9/pool_design"
AUDIT = DATA / "runs/s_rows_audit/rows_audit.parquet"
CACHE = DATA / "runs/op_parity/cache"
BS = DATA / "runs/pai_av/behavior_stats"
BANDS = [0, 0.5, 3, 8, 13, 20, 1e9]
BLAB = ["<0.5", "0.5-3", "3-8", "8-13", "13-20", ">20"]
TERMS = ["agent", "road", "lat_onc", "lat_soft", "lat_side"]
STEPS, K_S = 9500, 32                                    # NTV0's S stage: 9 500 steps x 32 S rows a batch


def band(v):
    return pd.cut(np.asarray(v, float), BANDS, right=False, labels=BLAB)


def shares(x, labels=BLAB):
    s = pd.Series(x).value_counts(normalize=True)
    return {k: round(float(s.get(k, 0.0)), 4) for k in labels}


# ---------------------------------------------------------------- nuplan
def cmd_nuplan(a):
    d = pd.read_parquet(AUDIT)
    live = d.w > 0
    act_n = (d[TERMS] > 0).any(axis=1)                                  # NTV0 forward, any of the five trainer terms
    unl_n = d.off_t0 | ((d.agent > 0) & (d.fut_agent > 0)) | ((d.road > 0) & (d.fut_road > 0))
    act_r = (d.rb0_agent > 0) | (d.rb0_road > 0)                        # RB0-F-s0 forward (agent / road only)
    unl_r = d.off_t0 | ((d.rb0_agent > 0) & (d.fut_agent > 0)) | ((d.rb0_road > 0) & (d.fut_road > 0))
    act_s = (d.plan_agent > 0) | (d.plan_road > 0)                      # the stored served plan of the policy that rolled the row
    mass = d[TERMS].sum(axis=1)
    d = d.assign(live=live, act_n=act_n, la_n=act_n & ~unl_n, act_r=act_r, la_r=act_r & ~unl_r, act_s=act_s, mass=mass,
                 band=band(d.speed), logrep=d.k <= 1)
    out = {}
    for part, g in list(d.groupby("pool")) + [("all", d)]:
        L = g[g.live]
        sc = g.groupby("scene").score.first()
        out[part] = dict(
            roller=sorted(g.tag.unique().tolist()), rows=len(g), live=int(L.shape[0]), scenes=int(g.scene.nunique()),
            rows_per_scene=round(len(g) / g.scene.nunique(), 1), draws_per_live_row_k32=round(STEPS * K_S / max(len(L), 1), 1),
            speed=shares(L.band), standing=round(float((L.speed < 0.5).mean()), 4), logreplay_k01=round(float(L.logrep.mean()), 4),
            kband=shares(L.kb.astype(str), ["k0-5", "k6-11", "k12-17"]), sit=shares(L.sit, ["free/other", "lead<20m", "turn>30deg", "standing"]),
            outcome_rows=shares(L.outcome, ["full", "partial", "zero"]),
            source_scene=dict(mean=round(float(sc.mean()), 4), zero=round(float((sc == 0).mean()), 4), full=round(float((sc >= 1).mean()), 4)),
            act_roller_served=round(float(L.act_s.mean()), 4), act_rb0=round(float(L.act_r.mean()), 4), la_rb0=round(float(L.la_r.mean()), 4),
            act_ntv0=round(float(L.act_n.mean()), 4), la_ntv0=round(float(L.la_n.mean()), 4),
            act_ntv0_by_term={t: round(float((L[t] > 0).mean()), 4) for t in TERMS},
            mass_share_of_pool=round(float(L.mass.sum() / d[d.live].mass.sum()), 4),
            la_mass_share=round(float(L.mass[L.la_n].sum() / max(L.mass.sum(), 1e-9)), 4),
            la_ntv0_above13=round(float((L.la_n & (L.speed >= 13)).mean()), 5),
            lat_rasters=bool((g.lat_onc > 0).any() or (g.lat_soft > 0).any()),
            la_by_speed={b: round(float(L.la_n[L.band == b].mean()), 4) if (L.band == b).any() else None for b in BLAB},
        )
    # the clean part: what each hygiene filter removes (live rows)
    r = d[(d.pool == "rb0") & d.live]
    filt = dict(standing=r.speed < 0.5, off_t0=r.off_t0, logreplay=r.logrep, unlearnable_ntv0=r.act_n & ~r.la_n)
    out["rb0_filters"] = {k: dict(rows=int(m.sum()), share=round(float(m.mean()), 4), mass_share=round(float(r.mass[m].sum() / r.mass.sum()), 4))
                          for k, m in filt.items()}
    keep = ~(filt["standing"] | filt["off_t0"])
    out["rb0_kept"] = dict(rows=int(keep.sum()), draws_per_row_k32=round(STEPS * K_S / keep.sum(), 1), la_ntv0=round(float(r.la_n[keep].mean()), 4),
                           act_ntv0=round(float(r.act_n[keep].mean()), 4))
    # RB0-F's own states versus the other policies' states, read with RB0-F itself and with NTV0: is the clean part poorer per row?
    out["rb0_vs_rest"] = {f"{m}_{p}": round(float(d[d.live & ((d.pool == "rb0") == (p == "rb0"))][m].mean()), 4)
                          for m in ("act_r", "la_r", "act_n", "la_n") for p in ("rb0", "rest")}
    # source outcome of RB0-F on the train scenes (all 18 decisions), by situation of the scene's rows
    ps = pd.read_csv(DATA / "runs/s_rows_audit/outcome/per_scene.csv")
    p0 = ps[ps.label == "RB0-F-s0"]
    out["rb0_rollouts"] = dict(scenes=int(len(p0)), mean=round(float(p0.score.mean()), 4), zero=int((p0.score == 0).sum()),
                               full=int((p0.score >= 1).sum()), atfault=int(p0.collision_at_fault.fillna(0).astype(bool).sum()),
                               offroad=int(p0.offroad.fillna(0).astype(bool).sum()),
                               corridor=int(p0.left_corridor_laterally.fillna(0).astype(bool).sum()))
    # route polyline the rb0 rows store (rm_build's input): how far ahead a recomputed command / window can look
    z = np.load(CACHE / "on1_L18RB0/route.npz")
    sl = np.nan_to_num(z["seglen"][z["ok"]]).sum(1)
    px = z["poly"][z["ok"]][:, :, 0]
    out["rb0_route"] = dict(ok=round(float(z["ok"].mean()), 4), length_m_p5_p50_p95=[round(float(x), 1) for x in np.percentile(sl, [5, 50, 95])],
                            x_far_m_p50=round(float(np.nanmedian(np.nanmax(px, 1))), 1), points=int(z["poly"].shape[1]))
    json.dump(out, open(O / "nuplan.json", "w"), indent=1)
    print(json.dumps({k: out[k] for k in ("rb0", "all", "rb0_filters", "rb0_kept", "rb0_vs_rest", "rb0_rollouts", "rb0_route")}, indent=1))


# ---------------------------------------------------------------- pai
def _tab(p):
    t = np.load(CACHE / p / "tab.npz", allow_pickle=True)
    return pd.DataFrame(dict(name=t["names"], log=t["log"], speed=t["speed"], src=t["src"] if "src" in t.files else ""))


def cmd_pai(a):
    out = {}
    for p in ("on1_PAID", "on1_PAIH"):
        t = _tab(p)
        k = t.name.str.rsplit(":", n=1).str[1].astype(int)
        t = t.assign(k=k, band=band(t.speed))
        out[p] = dict(rows=len(t), scenes=int(t.log.nunique()), rows_per_scene=round(len(t) / t.log.nunique(), 1), speed=shares(t.band),
                      standing=round(float((t.speed < 0.5).mean()), 4), src=t.src.value_counts().to_dict(),
                      above13=round(float((t.speed >= 13).mean()), 4), above20=round(float((t.speed >= 20).mean()), 4))
        if p == "on1_PAID":                                             # near-duplicates: speed / k structure of consecutive rows
            t = t.sort_values(["log", "k"])
            dk = t.groupby("log").k.diff()
            dv = t.groupby("log").speed.diff().abs()
            out[p]["dk_mode"] = int(dk.mode().iloc[0])
            out[p]["dv_lag1_median"] = round(float(dv[dk == 1].median()), 3)
            s5 = t.groupby("log").speed.diff(5).abs()
            out[p]["dv_lag5_median"] = round(float(s5[t.groupby("log").k.diff(5) == 5].median()), 3)
            out[p]["rows_2hz"] = int((t.k % 5 == 0).sum())
    # PAI 281 test states: 0.5 s steps of stored reads (kept window, after the 2 s handoff), class from the human path at that station
    for r in ("RB0-F-s0", "NTV0-L18-s0"):
        s = pd.read_parquet(BS / f"out/steps_{r}.parquet")
        out[f"test281_{r}"] = dict(steps=len(s), speed=shares(band(s.v)), standing=round(float((s.v < 0.5).mean()), 4),
                                   above13=round(float((s.v >= 13).mean()), 4), above20=round(float((s.v >= 20).mean()), 4),
                                   cls=shares(s.cls, ["urban", "arterial", "highway", "ramp"]))
    st = pd.read_parquet(BS / "out/stations.parquet")                   # 1 m stations of the 281 human paths (distance-weighted)
    out["test281_human_stations"] = dict(stations=len(st), speed=shares(band(st.v)), cls=shares(st.cls, ["urban", "arterial", "highway", "ramp"]),
                                         fallback=round(float(st.fb.mean()), 4))
    json.dump(out, open(O / "pai.json", "w"), indent=1)
    print(json.dumps(out, indent=1))


# ---------------------------------------------------------------- road
def _facts(u):
    s, m = u
    import behavior_stats as B
    try:
        f = B.scene_facts(m, m.get("lon", 0.0) if np.isfinite(m.get("lon", np.nan)) else 0.0)
        return s, dict(cls=f["cls"], fb=f["fb"], v=f["vq"], scene_cls=f["scene_cls"]), ""
    except Exception as ex:  # noqa: BLE001
        return s, None, repr(ex)[:300]


def cmd_road(a):
    sys.path.insert(0, str(BS / "code"))
    from concurrent.futures import ProcessPoolExecutor
    M = pickle.load(open(O / "maps_train.pkl", "rb"))["scenes"]
    with ProcessPoolExecutor(a.workers) as ex:
        res = list(ex.map(_facts, sorted(M.items()), chunksize=2))
    F = {s: f for s, f, e in res if f is not None}
    errs = [(s, e) for s, f, e in res if e]
    st = pd.concat([pd.DataFrame(dict(scene=s, cls=f["cls"], fb=f["fb"], v=f["v"])) for s, f in F.items()])
    st.to_parquet(O / "stations_train.parquet")
    sc = pd.Series({s: f["scene_cls"] for s, f in F.items()})
    te = pd.read_parquet(BS / "out/stations.parquet")
    tsc = pd.read_csv(BS / "out/scenes_map.csv").set_index("scene").scene_cls
    C = ["urban", "arterial", "highway", "ramp"]
    out = dict(train=dict(scenes=len(F), errors=len(errs), lon_missing=int(sum(not np.isfinite(m.get("lon", np.nan)) for m in M.values())),
                          km=round(len(st) / 1000, 1), stations=shares(st.cls, C), fallback=round(float(st.fb.mean()), 4),
                          fallback_by_cls={c: round(float(st.fb[st.cls == c].mean()), 4) for c in C if (st.cls == c).any()},
                          scene_majority=sc.value_counts().to_dict(), speed=shares(band(st.v)),
                          v_median_by_cls={c: round(float(st.v[st.cls == c].median()), 2) for c in C if (st.cls == c).any()}),
               test281=dict(scenes=int(te.scene.nunique()), km=round(len(te) / 1000, 1), stations=shares(te.cls, C),
                            fallback=round(float(te.fb.mean()), 4), scene_majority=tsc.value_counts().to_dict(), speed=shares(band(te.v)),
                            v_median_by_cls={c: round(float(te.v[te.cls == c].median()), 2) for c in C if (te.cls == c).any()}),
               errors=errs[:10])
    json.dump(out, open(O / "road.json", "w"), indent=1)
    print(json.dumps(out, indent=1))


# ---------------------------------------------------------------- report
def _row(name, x, keys):
    return f"| {name} | " + " | ".join(f"{100 * x.get(k, 0):.1f}" for k in keys) + " |"


def cmd_report(a):
    N = json.load(open(O / "nuplan.json"))
    P = json.load(open(O / "pai.json"))
    R = json.load(open(O / "road.json")) if (O / "road.json").exists() else None
    L = ["# POOL-DESIGN offline evidence (CPU, stored files)", "",
         "## nuPlan S pool on1_JT by part (live rows = drawn in NTV0; activity on each row's input with the named checkpoint's eval plan)", "",
         "| part | roller | rows / live / scenes | draws per live row (32 x 9 500) | standing | log-replay k<=1 | > 13 m/s | act RB0 / LA RB0 | act NTV0 / LA NTV0 | share of pool hinge mass | LA share of own mass | onc / soft rasters |",
         "|:--|:--|:--|--:|--:|--:|--:|:--|:--|--:|--:|:--|"]
    for p in ("rb0", "rtb1", "y", "ys", "all"):
        x = N[p]
        L.append(f"| {p} | {', '.join(x['roller'])} | {x['rows']} / {x['live']} / {x['scenes']} | {x['draws_per_live_row_k32']} | "
                 f"{100 * x['standing']:.1f} % | {100 * x['logreplay_k01']:.1f} % | {100 * (x['speed']['13-20'] + x['speed']['>20']):.2f} % | "
                 f"{100 * x['act_rb0']:.1f} / {100 * x['la_rb0']:.1f} % | {100 * x['act_ntv0']:.1f} / {100 * x['la_ntv0']:.1f} % | "
                 f"{100 * x['mass_share_of_pool']:.1f} % | {100 * x['la_mass_share']:.1f} % | {'yes' if x['lat_rasters'] else 'no'} |")
    L += ["", "Speed bands of live rows (%):", "", "| part | " + " | ".join(BLAB) + " |", "|:--|" + "--:|" * len(BLAB)]
    L += [_row(p, N[p]["speed"], BLAB) for p in ("rb0", "rtb1", "y", "ys", "all")]
    L += ["", f"rb0 filters (live rows): `{json.dumps(N['rb0_filters'])}`", f"rb0 kept after standing + off_t0: `{json.dumps(N['rb0_kept'])}`",
          f"RB0-F own states vs other parts: `{json.dumps(N['rb0_vs_rest'])}`", f"RB0-F rollouts on the train scenes: `{json.dumps(N['rb0_rollouts'])}`",
          f"rb0 stored route: `{json.dumps(N['rb0_route'])}`", "",
          "## PAI states: speed bands (%)", "", "| source | rows / steps | " + " | ".join(BLAB) + " | > 13 | > 20 |", "|:--|--:|" + "--:|" * (len(BLAB) + 2)]
    for k, nm in (("on1_PAID", "PAID: R5LC own states, ACC + GAP, 719 train scenes, 10 Hz"), ("on1_PAIH", "PAIH: human states, train scenes"),
                  ("test281_RB0-F-s0", "281 exam, RB0-F-s0 read (0.5 s steps)"), ("test281_NTV0-L18-s0", "281 exam, NTV0-L18-s0 read")):
        x = P[k]
        n = x.get("rows", x.get("steps"))
        L.append(f"| {nm} | {n} | " + " | ".join(f"{100 * x['speed'][b]:.1f}" for b in BLAB) + f" | {100 * x['above13']:.1f} | {100 * x['above20']:.1f} |")
    for nm, x in (("nuPlan rb0 (live)", N["rb0"]), ("nuPlan on1_JT (live)", N["all"])):
        L.append(f"| {nm} | {x['live']} | " + " | ".join(f"{100 * x['speed'][b]:.1f}" for b in BLAB) +
                 f" | {100 * (x['speed']['13-20'] + x['speed']['>20']):.1f} | {100 * x['speed']['>20']:.1f} |")
    L += ["", f"PAID near-duplicates: `{json.dumps({k: P['on1_PAID'][k] for k in ('dk_mode', 'dv_lag1_median', 'dv_lag5_median', 'rows_2hz')})}`",
          f"281 exam human stations: `{json.dumps(P['test281_human_stations'])}`"]
    if R:
        C = ["urban", "arterial", "highway", "ramp"]
        L += ["", "## Road class (behavior_stats Q1 rules; distance-weighted 1 m stations of the human path)", "",
              "| set | scenes | km | " + " | ".join(C) + " | fallback (human-speed class) | scene majority |", "|:--|--:|--:|" + "--:|" * 5 + ":--|"]
        for k in ("train", "test281"):
            x = R[k]
            L.append(f"| {k} | {x['scenes']} | {x['km']} | " + " | ".join(f"{100 * x['stations'][c]:.1f}" for c in C) +
                     f" | {100 * x['fallback']:.1f} | {x['scene_majority']} |")
        L += ["", f"median human speed by class: train `{R['train']['v_median_by_cls']}`, 281 `{R['test281']['v_median_by_cls']}`",
              f"train fallback by class `{R['train']['fallback_by_cls']}`; errors {R['train']['errors']}, lon missing {R['train']['lon_missing']}"]
    (O / "pool_design.md").write_text("\n".join(L) + "\n")
    json.dump(dict(nuplan=N, pai=P, road=R), open(O / "pool_design.json", "w"), indent=1)
    print("\n".join(L))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["nuplan", "pai", "road", "report"])
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()
    O.mkdir(parents=True, exist_ok=True)
    dict(nuplan=cmd_nuplan, pai=cmd_pai, road=cmd_road, report=cmd_report)[a.cmd](a)


if __name__ == "__main__":
    main()
