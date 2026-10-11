#!/usr/bin/env python
"""Tables of experiments/hugsim/results/serving_trace.md (plans/2026-10-11-serving-trace-prereg.md). CPU, reads logs only.

  clock   from existing rollout logs (<run>/bench/zs/<scenario>/zs_steps.jsonl): simulator step, fed speed / simulator speed, the
          pose-history key time implied by the fed poses, plan read-out time, plan length against speed, executed distance against the plan
  slots   from a traced run (agent opt `trace`: zs_trace.jsonl) and the queue probe (serving_trace_probe.py: probe.json): for every policy
          slot of a decision the simulator steps of the two frames behind it and their time offsets

  .venv/bin/python experiments/hugsim/scripts/serving_trace_report.py --runs $B/P2H10-F-s0_spec-rr1 $B/P2H10-F-s0_spec-rr2 \
      --trace $B/P2H10-F-s0_spec-c85e22e1821d5 --probe <probe.json> --out <dir>          (B = $DATA_DIR/runs/bench/hugsim)
"""
import sys as _sys, pathlib as _pl  # noqa: E401
_R = _pl.Path(__file__).resolve().parents[3]
_sys.path[:0] = [str(_R)]
import argparse, csv, json, zlib  # noqa: E401,E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

from jevdrive.run import Run, cli_args  # noqa: E402

T_IDXS = np.array([10.0 * (i / 32) ** 2 for i in range(33)])
LOGGED = [4, 8, 12, 16, 20, 24, 32]                 # model_pos rows of zs_steps.jsonl
SIM_DT, MODEL_DT, REAR = 0.25, 0.05, 1.7347         # simulator step, 20 Hz model step, rear-axle offset of the parity poses (m)
ZERO_CRC = zlib.crc32(np.zeros((2, 6, 128, 256), np.uint8).tobytes())


def steps_of(f):
    return [r for r in map(json.loads, open(f)) if "step" in r]


def q(x, p=(5, 50, 95)):
    x = np.asarray(x, float)
    return dict(n=int(len(x)), **{f"p{k}": float(np.percentile(x, k)) if len(x) else float("nan") for k in p})


def clock(run_dirs, v_min, warm):
    """Per-decision ratios, pooled over the scenarios of the runs."""
    R = {k: [] for k in ("dt", "vfed_over_v", "key_t_oldest", "plan_t_dil", "plan_t_raw", "d1_over_v", "exec_over_plan",
                         "arc_model", "arc_sim", "mv0_over_vfed", "reps")}
    n_scen = 0
    for d in run_dirs:
        for f in sorted(Path(d).glob("bench/zs/*/zs_steps.jsonl")):
            S = steps_of(f)
            if len(S) < 12:
                continue
            n_scen += 1
            t = np.array([s["t"] for s in S], float)
            v = np.array([s["v"] for s in S], float)
            pos = np.array([s["pos"] for s in S], float)
            th = np.array([s["theta"] for s in S], float)
            rear = pos - REAR * np.stack([np.sin(th), np.cos(th)], -1)
            R["dt"] += np.diff(t).tolist()
            R["reps"] += [s["reps"] for s in S[1:] if "reps" in s]
            for i, s in enumerate(S):
                if "model_pos" not in s or s.get("oracle"):
                    continue
                ego = (s.get("parity") or {}).get("ego")
                ok = i >= warm and v[i] > v_min
                if ego is not None and v[i] > 0.5:
                    R["vfed_over_v"].append(10 * ego[4] / v[i])
                if ego is not None and ok and t[i] >= 2.0 and np.ptp(th[max(0, i - 8):i + 1]) < 0.02:
                    # fed oldest pose x (model -1.5 s): which simulator time lies that far behind the car on the logged track?
                    x_old = -10 * ego[8]
                    back = np.linalg.norm(rear[i] - rear[max(0, i - 11):i + 1], axis=1)[::-1]   # distance to the pose k steps back, k = 0..11
                    if back[-1] > x_old > 0 and np.all(np.diff(back) > 0):
                        R["key_t_oldest"].append(float(np.interp(x_old, back, SIM_DT * np.arange(len(back)))))
                mp = np.r_[[[0.0, 0.0]], np.asarray(s["model_pos"], float)]
                tm = np.r_[0.0, T_IDXS[LOGGED]]
                arc = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(mp, axis=0), axis=1))]
                plan = np.asarray(s.get("raw_plan", s["plan"]), float)
                clean = not any(k in s for k in ("raw_plan", "stop", "plan_before_ol", "plan_before_lm", "plan_before_rr", "plan_before_ll"))
                if ok and clean and len(plan) >= 6 and mp[5, 0] > 1.0 and np.all(np.diff(mp[:, 0]) > 0):
                    # the model time at which the model's forward position equals the forward position of plan point k (sent for 0.5 k s)
                    for k in (2, 4):
                        tau = float(np.interp(plan[k - 1, 1], mp[:, 0], tm))
                        R["plan_t_dil"].append(tau / (0.5 * k / 1.25))
                        R["plan_t_raw"].append(tau / (0.5 * k))
                if ok and clean:
                    R["d1_over_v"].append(float(np.linalg.norm(plan[0])) / (0.5 * v[i]))
                    if i + 2 < len(S):
                        R["exec_over_plan"].append(float(np.linalg.norm(pos[i + 2] - pos[i])) / max(float(np.linalg.norm(plan[0])), 1e-6))
                    R["arc_sim"].append(float(np.linalg.norm(np.diff(np.r_[[[0.0, 0.0]], plan[:6]], axis=0), axis=1).sum()) / (3.0 * v[i]))
                    vfed = 1.25 * v[i]
                    R["arc_model"].append(float(arc[5]) / (tm[5] * vfed))                  # arc to model 3.906 s over fed speed
                    R["mv0_over_vfed"].append(float(s["model_v"][0]) / vfed)
    return n_scen, {k: q(x) for k, x in R.items()}


def slots(trace_dir, probe):
    """Per decision and policy slot: simulator step of the current / previous frame of the slot's image pair."""
    fb = sorted(probe["read_feat_steps_back"])                                # 20 Hz steps before the new row, the past policy slots
    ib = probe["read_img_steps_back"]
    assert len(ib) == 1, ib
    ib = ib[0]                                                                # the pair is (frame ib steps back, new frame)
    rows, checks = [], dict(decisions=0, model_steps=0, n_ok=0, img_ok=0, img_n=0, feat_rows_ok=0, feat_rows_n=0, scenarios=0, still_pairs=0)
    for f in sorted(Path(trace_dir).glob("bench/zs/*/zs_trace.jsonl")):
        D = [json.loads(x) for x in open(f)]
        if not D:
            continue
        checks["scenarios"] += 1
        owner, feat = {}, {}                                                  # model step -> simulator step; model step -> CRC of its feature row
        N = 0
        for d in D:
            for r in d["srv"]:
                N += 1
                checks["model_steps"] += 1
                checks["n_ok"] += r["n"] == N
                owner[N], feat[N] = d["step"], r["feat"]
                want = [D[owner[N - 4 + i]]["img2_crc"] if N - 4 + i >= 1 else ZERO_CRC for i in range(5)]
                checks["img_n"] += 5
                checks["img_ok"] += sum(a == b for a, b in zip(r["img"], want))
            fq = d["srv"][-1]["feat_q"]
            for b in range(len(fq)):                                          # row 127 - b was appended b steps before the last step
                if N - b >= 1:
                    checks["feat_rows_n"] += 1
                    checks["feat_rows_ok"] += fq[len(fq) - 1 - b] == feat[N - b]
            checks["decisions"] += 1
            n, t0 = d["step"], d["t"]
            checks["still_pairs"] += n > 0 and d["img2_crc"] == D[n - 1]["img2_crc"]
            for j, b in enumerate([0] + fb):                                  # slot 0 = the current frame's tokens
                cur, prev = N - b, N - b - ib
                sc = owner.get(cur)
                sp = owner.get(prev)
                rows.append(dict(scenario=f.parent.name, step=n, t=t0, slot=j, model_offset_s=-MODEL_DT * b,
                                 cur_step=sc, prev_step=sp, cur_offset_s=None if sc is None else D[sc]["t"] - t0,
                                 prev_offset_s=None if sp is None else D[sp]["t"] - t0,
                                 kind="empty" if sc is None else "zero-prev" if sp is None else "repeat" if sc == sp else "render-pair"))
    return rows, checks


def slot_table(rows, warm):
    out = []
    for j in sorted({r["slot"] for r in rows}):
        W = [r for r in rows if r["slot"] == j and r["step"] >= warm]
        co = np.array([r["cur_offset_s"] for r in W], float)
        po = np.array([r["prev_offset_s"] for r in W], float)
        out.append(dict(slot=j, model_offset_s=W[0]["model_offset_s"], n=len(W), cur_offset_s_min=co.min(), cur_offset_s_max=co.max(),
                        prev_offset_s_min=po.min(), prev_offset_s_max=po.max(), render_pair=sum(r["kind"] == "render-pair" for r in W)))
    return out


def early_table(rows, upto):
    out = []
    for n in range(upto):
        E = [r for r in rows if r["step"] == n]
        if E:
            k = len({r["scenario"] for r in E})
            out.append(dict(step=n, **{c: sum(r["kind"] == c for r in E) / k for c in ("render-pair", "repeat", "zero-prev", "empty")}))
    return out


def write_csv(path, rows):
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="*", default=[], help="bench HUGSIM run dirs for the clock table")
    ap.add_argument("--ref-runs", nargs="*", default=[], help="reference arm run dirs (same clock table, own block)")
    ap.add_argument("--trace", default="", help="bench HUGSIM run dir of the traced run")
    ap.add_argument("--probe", default="", help="probe.json of serving_trace_probe.py")
    ap.add_argument("--out", required=True)
    ap.add_argument("--v-min", type=float, default=3.0)
    ap.add_argument("--warm", type=int, default=9, help="first decision whose 8 past slots all come from driven frames")
    cli_args(ap)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    with Run("hugsim_serving_trace", "report", seed=a.seed, config=vars(a), resume=a.resume) as run:
        res = {}
        for name, dirs in (("clock", a.runs), ("clock_ref", a.ref_runs), ("clock_trace", [a.trace] if a.trace else [])):
            if dirs:
                n, tab = clock(dirs, a.v_min, a.warm)
                res[name] = dict(runs=[Path(d).name for d in dirs], scenarios=n, v_min=a.v_min, warm=a.warm, table=tab)
                write_csv(out / f"{name}.csv", [dict(metric=k, **v) for k, v in tab.items()])
                run.info("%s (%d scenario runs): %s", name, n, json.dumps({k: [round(v[p], 4) for p in ("p5", "p50", "p95")] + [v["n"]] for k, v in tab.items()}))
        if a.trace and a.probe:
            probe = json.loads(Path(a.probe).read_text())
            rows, checks = slots(a.trace, probe)
            st, et = slot_table(rows, a.warm), early_table(rows, a.warm)
            write_csv(out / "slots_warm.csv", st)
            write_csv(out / "slots_early.csv", et)
            res.update(slot_checks=checks, slots_warm=st, slots_early=et, probe={k: probe[k] for k in probe if k not in ("img_q", "feat_q")})
            run.info("slot checks: %s", json.dumps(checks))
            for r in st:
                run.info("slot %s", json.dumps(r))
        (out / "summary.json").write_text(json.dumps(res, indent=1))
        run.summary["out"] = str(out)


if __name__ == "__main__":
    main()
