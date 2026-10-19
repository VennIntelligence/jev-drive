#!/usr/bin/env python3
"""TV-AUDIT: the full NAVSIM trainval CAM_F0 download (2026-10-19) before any use (research/next-round/trainval-use.md). No training.
Box, repo root, navsim2 venv (nuPlan devkit for the stop-line labels); every job through the pool.

  integrity [--workers N] [--decode-every K]   every trainval log pickle: frames, CAM_F0 referenced / on disk / empty / bad JPEG markers
                     (SOI at the head, EOI in the last 64 bytes), a full PIL decode of every K-th image; orphan dirs / files; archive state
                     markers -> results/tv_integrity.{json,md}
  leak               trainval logs (nuPlan log segments) against every evaluation set (navtest / navsim test logs, navhard_two_stage, held 460,
                     public 1491 / their logs, arena windows, wait set v2, waitcap_test, PAI 281) by exact segment and by drive (date_veh,
                     the nuPlan db); token overlap with navtest / navhard; if clean, splits navsim/tv-clean-train-logs and
                     navsim/tv-clean-hold-logs (crc32(log) % 10 == 0, si.py's hold rule; navtrain dev logs excluded) -> results/tv_leak.{json,md}
  census [--workers N]   every eligible trainval frame (wc.log_rows: full 4 s future, route, 2 Hz timing) labelled by cap_ledger
                     led.log_cells: lib/self_iter classes, wait_cap cells (W_L .. F_L), d331 cells, the first stop line ahead (type, front
                     bumper distance, light state at t0, route connector) -> runs/self_iter/tv_census/frames.npz + results/tv_census.{json,md}:
                     counts in navtrain tokens vs the clean non-navtrain frames, with the df.py standing-reason rule and the two cues of
                     decision 372 (standing at a red line with no lead, lead start)
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import zlib
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path[:0] = [str(REPO), str(REPO / "lib"), str(REPO / "experiments/wait_cap/scripts")]
import self_iter as SI  # noqa: E402

DATA = Path(os.environ.get("DATA_DIR", "/root/autodl-tmp/ujs"))
NAV = DATA / "datasets/navsim"
TV, BLOB = NAV / "navsim_logs/trainval", NAV / "sensor_blobs/trainval"
OUT = DATA / "runs/self_iter/tv_census"
RES = REPO / "experiments/self_iter/results"
CR = DATA / "runs/op_parity/cache"
V_STAND, D_STAND, LINE_MIN, LINE_MAX = 0.3, 1.0, -2.0, 10.0      # df.py's standing rule and stop-line window (front bumper)


def write(name, rec, md):
    RES.mkdir(parents=True, exist_ok=True)
    (RES / f"{name}.json").write_text(json.dumps(rec, indent=1, default=float))
    (RES / f"{name}.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


def held_out(log):
    return zlib.crc32(log.encode()) % 10 == 0


def drive(seg):
    return "_".join(seg.split("_")[:2])


# ---------------------------------------------------------------- integrity
def _integ_job(job):
    from PIL import Image
    p, every = job
    fr = pickle.load(open(p, "rb"))
    paths = [f["cams"]["CAM_F0"]["data_path"] for f in fr]
    r = dict(log=p.stem, frames=len(fr), missing=0, empty=0, bad_marker=0, decode_tried=0, decode_bad=0, bad_sample=[], bytes=0)
    for i, q in enumerate(paths):
        f = BLOB / q
        try:
            n = f.stat().st_size
        except FileNotFoundError:
            r["missing"] += 1
            continue
        r["bytes"] += n
        if n == 0:
            r["empty"] += 1
            continue
        with open(f, "rb") as h:
            head = h.read(2)
            h.seek(max(0, n - 64))
            tail = h.read()
        if head != b"\xff\xd8" or b"\xff\xd9" not in tail:
            r["bad_marker"] += 1
            r["bad_sample"].append(q)
        if i % every == 0:
            r["decode_tried"] += 1
            try:
                with Image.open(f) as im:
                    im.load()
            except Exception:  # noqa: BLE001
                r["decode_bad"] += 1
                r["bad_sample"].append(q)
    d = BLOB / p.stem / "CAM_F0"
    on = {x.name for x in d.iterdir()} if d.is_dir() else set()
    r["files_on_disk"], r["orphans"] = len(on), len(on - {Path(q).name for q in paths})
    r["bad_sample"] = r["bad_sample"][:5]
    return r


def cmd_integrity(a):
    from jevdrive import par
    from jevdrive.run import Run
    logs = sorted(TV.glob("*.pkl"))
    dirs = {x.name for x in BLOB.iterdir() if x.is_dir()}
    st = NAV / "_archives/state"
    done = sorted(x.name for x in st.glob("openscene_sensor_trainval_camera_*.tgz.done"))
    parts = sorted(x.name for x in (NAV / "_archives").glob("*.part.d"))
    with Run("self_iter", "tv-integrity", config=vars(a)) as run:
        res = par.pmap(_integ_job, [(p, a.decode_every) for p in logs], run=run, workers=a.workers, threads=False, desc="logs")
        res.raise_if_failed()
        R = res.values
        tot = {k: int(sum(r[k] for r in R)) for k in ("frames", "missing", "empty", "bad_marker", "decode_tried", "decode_bad", "files_on_disk", "orphans", "bytes")}
        nums = sorted(int(x.split("_")[-1].split(".")[0]) for x in done)
        rec = dict(logs=len(logs), blob_dirs=len(dirs), dirs_without_pickle=sorted(dirs - {p.stem for p in logs})[:20],
                   pickles_without_dir=sorted({p.stem for p in logs} - dirs)[:20], archives_done=len(done),
                   archive_numbers_missing=sorted(set(range(max(nums) + 1)) - set(nums)), stale_part_dirs=len(parts), **tot,
                   logs_incomplete=sorted(r["log"] for r in R if r["missing"] or r["empty"] or r["bad_marker"] or r["decode_bad"])[:50],
                   bad_sample=[q for r in R for q in r["bad_sample"]][:20])
        run.summary |= {k: v for k, v in rec.items() if not isinstance(v, list)}
    md = [f"# TV-AUDIT integrity (trainval CAM_F0, {rec['logs']} log pickles)", "",
          f"- archives: {rec['archives_done']} trainval camera `.done` markers (0..{max(nums)}, missing numbers {rec['archive_numbers_missing'] or 'none'}); "
          f"{rec['stale_part_dirs']} stale `.part.d` chunk dirs of finished archives left in `_archives/`",
          f"- logs: {rec['logs']} pickles, {rec['blob_dirs']} blob dirs; dirs without a pickle {len(rec['dirs_without_pickle'])}, pickles without a dir {len(rec['pickles_without_dir'])}",
          f"- frames {tot['frames']}; CAM_F0 referenced and missing {tot['missing']}, empty {tot['empty']}, bad JPEG markers {tot['bad_marker']}; "
          f"full decode {tot['decode_tried']} (every {a.decode_every}-th) bad {tot['decode_bad']}",
          f"- CAM_F0 files on disk {tot['files_on_disk']} ({tot['bytes'] / 1e9:.1f} GB referenced), orphans (on disk, not in a pickle) {tot['orphans']}",
          f"- incomplete logs: {len(rec['logs_incomplete'])} {rec['logs_incomplete'][:10]}"]
    write("tv_integrity", rec, md)


# ---------------------------------------------------------------- leak
def _navhard():
    """navhard_two_stage: the real log segment and original scene token of every synthetic scene (synthetic_scenes_attributes.csv; the
    synthetic pickles carry hashed log names)."""
    import csv
    R = list(csv.DictReader(open(NAV / "navhard_two_stage/synthetic_scenes_attributes.csv")))
    return {r["log_name"] for r in R}, {r["corresponding_original_scene_token"] for r in R} | {r["previous_original_scene_initial_token"] for r in R}


def cmd_leak(a):
    import gzip
    import csv
    from jevdrive.data import splits
    tv_logs = sorted(p.stem for p in TV.glob("*.pkl"))
    tv_tok = set()
    for p in TV.glob("*.pkl"):
        tv_tok |= {f["token"] for f in pickle.load(open(p, "rb"))}
    seg = lambda s: s.rsplit("-", 1)[0]  # noqa: E731   alpasim scene id = <log segment>-<token>
    test_logs = {p.stem for p in (NAV / "navsim_logs/test").glob("*.pkl")}
    ar = list(csv.DictReader(gzip.open(DATA / "runs/alpasim_proxy/arena/set/set.csv.gz", "rt")))
    sets = {
        "navtest (navsim test logs)": test_logs,
        "navhard_two_stage": _navhard()[0],
        "held460": {seg(s) for s in splits.load("alpasim_nuplan/onpol-held").members},
        "public1491": {seg(s) for s in splits.load("alpasim_nuplan/onpol-all").members},
        "arena (all windows)": {r["log"] for r in ar},
        "arena held windows": {r["log"] for r in ar if r["split"] != "train"},
        "waitset v2": {seg(s) for s in splits.load("alpasim_nuplan/waitset").members},
        "waitcap_test": set(np.load(DATA / "runs/wait_cap/select/rows.npz")["log"].astype(str).tolist()),
    }
    L, D = set(tv_logs), {drive(l) for l in tv_logs}
    rows = {}
    for k, S in sets.items():
        rows[k] = dict(segments=len(S), drives=len({drive(s) for s in S}), seg_overlap=sorted(L & S), drive_overlap=sorted(D & {drive(s) for s in S}))
    tok = {"navtest": set(splits.load("navsim/navtest").members), "navhard_two_stage": set(splits.load("navsim/navhard_two_stage").members),
           "navhard original scene tokens": _navhard()[1]}
    tok_ov = {k: len(tv_tok & v) for k, v in tok.items()}
    pai = sum(1 for f in (DATA / "runs/alpasim/pai_box/tokyo/r281/rsg/chunks").glob("*.txt") for s in open(f) if s.strip())
    clean = all(not v["seg_overlap"] and not v["drive_overlap"] for v in rows.values()) and not any(tok_ov.values())
    dv = splits.load("navsim/op-parity-full-dev")
    devlogs = set()
    for i in range(12):
        z = np.load(CR / f"navtrain_full.s{i}of12/tab.npz")
        devlogs |= set(z["log"][dv.mask(z["names"])].astype(str).tolist())
    rec = dict(tv_logs=len(tv_logs), tv_drives=len(D), tv_tokens=len(tv_tok), sets={k: dict(v, seg_overlap=v["seg_overlap"][:10], drive_overlap=v["drive_overlap"][:10],
               n_seg_overlap=len(v["seg_overlap"]), n_drive_overlap=len(v["drive_overlap"])) for k, v in rows.items()},
               token_overlap=tok_ov, pai281_scenes=pai, clean=clean, navtrain_dev_logs=len(devlogs & L))
    if clean:
        ok = [l for l in tv_logs if l not in devlogs]
        org = ("experiments/self_iter/scripts/tv_audit.py leak (2026-10-19): every NAVSIM trainval log (navsim_logs/trainval, full CAM_F0 download) "
               "with no segment or drive (date_veh) overlap with navtest / navhard_two_stage / alpasim_nuplan onpol-all (held 460 + public 1491) / "
               "arena / waitset / waitcap_test, minus the logs holding a navtrain dev token (navsim/op-parity-full-dev)")
        tr = splits.define("navsim", "tv-clean-train-logs", [l for l in ok if not held_out(l)], unit="log", origin=org + "; crc32(log) % 10 != 0",
                           status="frozen", used_by=["experiments/self_iter"], notes="trainval rows for any use; navtrain tokens inside are still navtrain")
        ho = splits.define("navsim", "tv-clean-hold-logs", [l for l in ok if held_out(l)], unit="log", origin=org + "; crc32(log) % 10 == 0 (si.py hold rule)",
                           status="frozen", used_by=["experiments/self_iter"], notes="never trained; log-disjoint validation for probes / imitation")
        splits.check_disjoint(tr, ho)
        rec |= dict(split_train=tr.id, split_hold=ho.id, n_train_logs=tr.n, n_hold_logs=ho.n)
    md = [f"# TV-AUDIT leak ({len(tv_logs)} trainval log segments, {len(D)} drives, {len(tv_tok)} tokens)", "",
          "| evaluation set | segments | drives | segment overlap | drive overlap |", "|:--|--:|--:|--:|--:|"]
    md += [f"| {k} | {v['segments']} | {v['drives']} | {len(v['seg_overlap'])} | {len(v['drive_overlap'])} |" for k, v in rows.items()]
    md += ["", f"token overlap: {tok_ov}; PAI 281 ({pai} scenes) is another dataset (no nuPlan log); navtrain dev logs inside trainval {len(devlogs & L)} (excluded from the splits)",
           f"clean: {clean}" + (f"; registered {rec['split_train']} ({rec['n_train_logs']} logs), {rec['split_hold']} ({rec['n_hold_logs']} logs)" if clean else "")]
    write("tv_leak", rec, md)


# ---------------------------------------------------------------- census
_LED = None


def _led():
    import importlib.util
    spec = importlib.util.spec_from_file_location("led", REPO / "experiments/cap_ledger/scripts/led.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _census_job(p):
    import wc
    global _LED
    if _LED is None:
        _LED = _led()
    fr = pickle.load(open(p, "rb"))
    ok = [(BLOB / f["cams"]["CAM_F0"]["data_path"]).exists() for f in fr]
    out = []
    for r in _LED.log_cells(p):
        i = r["frame"]
        ty, dF, st0, conn = (np.asarray(r[k]) for k in ("ln_type", "ln_d", "ln_st0", "ln_conn"))
        dF = dF - _LED.FRONT
        q = int(np.argmax(ty > 0)) if (ty > 0).any() else -1
        lc = _LED.cells_of(r)
        out.append((r["token"], r["log"], i, r["v0"], r["d2"], r["d4"], r["a4"], r["vend"], float(np.degrees(abs(r["dpsi4"]))), r["lead_gap"], r["lead_v"],
                    r["lead_any"], wc.cell_of(r) or "", sum(1 << j for j, k in enumerate(_LED.CELLS) if lc[k]), int(ty[q]) if q >= 0 else 0,
                    float(dF[q]) if q >= 0 else np.inf, int(st0[q]) if q >= 0 else -1, bool(conn[q]) if q >= 0 else False,
                    bool(r["junc"]), bool(all(ok[i - 3:i + 1])), r["fut"]))
    return out


COLS = ("token", "log", "frame", "v0", "d2", "d4", "a4", "vend", "dpsi4", "lead_gap", "lead_v", "lead_any", "cell", "bits", "ln_type", "ln_dF", "ln_st0",
        "ln_conn", "junc", "cam4")


def episodes(log, frame, m):
    """Distinct events: runs of consecutive (2 Hz) frames of one log inside mask m."""
    o = np.lexsort((frame, log))
    l, f, mm = log[o], frame[o], m[o]
    idx = np.flatnonzero(mm)
    if not len(idx):
        return 0
    new = np.r_[True, (l[idx][1:] != l[idx][:-1]) | (f[idx][1:] - f[idx][:-1] > 1)]
    return int(new.sum())


def cmd_census(a):
    from jevdrive import par
    from jevdrive.data import splits
    from jevdrive.run import Run
    os.environ.setdefault("NUPLAN_MAPS_ROOT", str(NAV / "maps"))
    os.environ.setdefault("NUPLAN_MAP_VERSION", "nuplan-maps-v1.0")
    led = _led()
    tr, ho = splits.load("navsim/tv-clean-train-logs"), splits.load("navsim/tv-clean-hold-logs")
    with Run("self_iter", "tv-census", config=vars(a)) as run:
        run.use_split(tr), run.use_split(ho)
        res = par.pmap(_census_job, sorted(TV.glob("*.pkl")), run=run, workers=a.workers, desc="logs")
        res.raise_if_failed()
        rows = [x for v in res.values for x in v]
        C = {k: np.array([x[j] for x in rows]) for j, k in enumerate(COLS)}
        fut = np.stack([x[-1] for x in rows]).astype(np.float32)
        OUT.mkdir(parents=True, exist_ok=True)
        np.savez(OUT / "frames.npz", fut=fut, cells=np.array(led.CELLS), **C)
        tok, log = C["token"].astype(str), C["log"].astype(str)
        nav = np.isin(tok, list(splits.load("navsim/navtrain").members))
        used = np.isin(tok, list(splits.load("navsim/selfiter_tv").members))
        intr, inho = np.isin(log, list(tr.members)), np.isin(log, list(ho.members))
        cls = SI.classes(fut, C["v0"])
        cell = C["cell"].astype(str)
        stand = (C["v0"] < V_STAND) & (C["d4"] < D_STAND)
        near = (C["ln_dF"] >= LINE_MIN) & (C["ln_dF"] <= LINE_MAX)
        red = near & (C["ln_type"] == 2) & (C["ln_st0"] == 1) & C["ln_conn"]
        sign = near & (C["ln_type"] == 1)
        lead_stop = cell == "W_L"                                         # stopped lead <= 15 m (df.py rlead)
        nolead_line = C["lead_any"] > C["ln_dF"] + 2.0                    # no in-path vehicle before the line (led.cells_of rule)
        bit = lambda k: (C["bits"] >> led.CELLS.index(k)) & 1 == 1  # noqa: E731
        launch = np.isin(cell, ("G_L", "G_N"))
        M = {
            "all eligible": np.ones(len(tok), bool),
            "class stopped": cls == "stopped", "class starting": cls == "starting", "class turning": cls == "turning",
            "class braking": cls == "braking", "class cruising": cls == "cruising",
            "standing (v0 < 0.3, 4 s < 1 m)": stand,
            "standing, reason visible (stopped lead <= 15 m | stop sign | red light, line -2..10 m)": stand & (lead_stop | sign | red),
            "standing, no visible reason": stand & ~(lead_stop | sign | red),
            "standing, stopped lead <= 15 m (W_L)": stand & lead_stop,
            "standing at red line (-2..10 m), any lead": stand & red,
            "standing at red line, no stopped lead <= 15 m": stand & red & ~lead_stop,
            "standing at red line, no vehicle before the line (d372 cue 1)": stand & red & nolead_line,
            "standing at red line, no vehicle before the line, line 0..5 m": stand & red & nolead_line & (C["ln_dF"] >= 0) & (C["ln_dF"] <= 5),
            "standing at stop sign line": stand & sign,
            "launch (G_L | G_N)": launch,
            "lead start (G_L: human launches, lead <= 15 m, v < 3; d372 cue 2)": cell == "G_L",
            "lead start, lead stopped at t0 (G_L, lead v < 0.5)": (cell == "G_L") & (C["lead_v"] < 0.5),
            "launch, no lead (G_N)": cell == "G_N",
            "launch at a light line (G_* with a light line -2..10 m)": launch & near & (C["ln_type"] == 2),
            "stop behind lead (S_L)": cell == "S_L", "stop at red (d331 S_R)": bit("S_R"), "go on green (d331 GO_G)": bit("GO_G"),
            "yield pedestrian (Y_P)": bit("Y_P"), "yield vehicle (Y_V)": bit("Y_V"), "follow moving lead (F_L)": cell == "F_L", "junction": C["junc"],
        }
        S = {"navtrain tokens": nav, "trainval clean train logs, not navtrain, CAM_F0 history": intr & ~nav & C["cam4"],
             "  of which already in selfiter_tv (NR1TV / JT rows)": intr & ~nav & C["cam4"] & used,
             "  of which new": intr & ~nav & C["cam4"] & ~used,
             "trainval clean hold logs, not navtrain, CAM_F0 history": inho & ~nav & C["cam4"]}
        tab = {s: {k: dict(frames=int((m & ms).sum()), events=episodes(log, C["frame"], m & ms), logs=len(set(log[m & ms].tolist()))) for k, m in M.items()}
               for s, ms in S.items()}
        rec = dict(frames=len(tok), logs=len(set(log.tolist())), cam4=int(C["cam4"].sum()), navtrain_labelled=int(nav.sum()), table=tab,
                   rules=dict(stand=f"v0 < {V_STAND} and 4 s displacement < {D_STAND} m", line=f"first stop line ahead on the hindsight route, "
                              f"front bumper {LINE_MIN}..{LINE_MAX} m", red="light line, red at t0, on the route connector",
                              no_vehicle_before_line="lead_any > line distance + 2 m", events="runs of consecutive 2 Hz frames of one log"))
        run.summary |= dict(frames=len(tok), cam4=int(C["cam4"].sum()))
    ks = list(S)
    md = [f"# TV-AUDIT census ({rec['frames']} eligible trainval frames, {rec['logs']} logs, {rec['cam4']} with 4 history CAM_F0)", "",
          "Cells: frames / events (runs of consecutive frames) / logs. Rules: " + "; ".join(f"{k}: {v}" for k, v in rec["rules"].items()), "",
          "| class | " + " | ".join(ks) + " |", "|:--|" + "--:|" * len(ks)]
    for k in M:
        md.append(f"| {k} | " + " | ".join(f"{tab[s][k]['frames']} / {tab[s][k]['events']} / {tab[s][k]['logs']}" for s in ks) + " |")
    write("tv_census", rec, md)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("integrity")
    p.add_argument("--workers", type=int, default=None)
    p.add_argument("--decode-every", type=int, default=25)
    sp.add_parser("leak")
    p = sp.add_parser("census")
    p.add_argument("--workers", type=int, default=None)
    a = ap.parse_args()
    dict(integrity=cmd_integrity, leak=cmd_leak, census=cmd_census)[a.cmd](a)


if __name__ == "__main__":
    main()
