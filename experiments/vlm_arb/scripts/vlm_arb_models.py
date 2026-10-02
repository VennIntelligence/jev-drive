"""Model comparison: the vision-language models on the box's disk on the four Phase A questions, same frames.

  $DATA_DIR/envs/jevdrive/bin/python experiments/vlm_arb/scripts/vlm_arb_models.py [--cap 0] [--units ...]
      [--only NAME ... --no-report | --score-only]

The lane (plan deviation D15) runs one job per model (--only NAME --no-report, each alone on a card) and then one
--score-only job that writes the table from the cached replies; --cap 0 = every answered request with frames.

Frames and truth: the saved requests of the new-format shadow units (vlm_decisions.jsonl + vlm_frames/, finished
routes only). Sample (seed 0, the same for every model): every request with an ego light within 50 m, every one with
a stop sign within 25 m, every one with truth static_block, plus a random draw of the rest, about --cap in total (if
the three strata alone exceed the cap, the light-only stratum is thinned at random and the output says so).

Models (MODELS; verified on the box on 2026-10-02, nothing is discovered at run time or downloaded):
  openjev        answers already in the logs (System One endpoint, constrained answers), not re-queried
  every other    one fixed multiple-choice chat prompt built from QUESTIONS_SCHEMA (vlm_arb_models_worker.PROMPT), both
                 frames, reply parsed as strict JSON or option letters. One subprocess per model, sequential, in the
                 env that loads it, on one GPU (the first of CUDA_VISIBLE_DEVICES, else the card with most free memory).
                 DiffusionGemma direct chat uses the running vLLM server on port 8000.
Latency: a worker stamps its rows load = 0 when no CARLA server ran on the box at its start; the latency column
uses those rows when there are at least 50 and says so, else all rows, labelled as measured under load.
Scoring = vlm_arb_phase_a.evaluate (same labels, route-cluster bootstrap). A reply with no parseable option for a
question is no answer for that question: it stays in the denominator (a miss for a recall, not a false alarm) and is
counted under "parse failures". Requests that raised are left out and counted.

Writes results/models.md, results/models.json; raw replies in results/models/<name>.jsonl (a rerun skips finished
requests). Exit 0 when the table was written, 2 when nothing could be scored.
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vlm_arb_common import DATA, RUN, fmt, write_json  # noqa: E402
from vlm_arb_models_worker import OPTIONS, PROMPT, parse_reply  # noqa: E402
from vlm_arb_phase_a import STRATA, SUBSETS, evaluate  # noqa: E402

WORKER = Path(__file__).resolve().parent / "vlm_arb_models_worker.py"
HUB = DATA / "cache/huggingface/hub"
UNITS = ["v2-dbg-shadow-s0-light", "v2-drive-s1-dev", "v2-drive-s1-tgt", "v2-drive-s0-tgt"]


def snap(repo):
    """Snapshot directory of a HF cache repo (first one), as a string; the path itself if none exists."""
    d = sorted((HUB / ("models--" + repo.replace("/", "--")) / "snapshots").glob("*"))
    return str(d[0]) if d else str(HUB / ("models--" + repo.replace("/", "--")))


def env(name):
    return str(DATA / "envs" / name / "bin/python")


# name, model id, loader kind (vlm_arb_models_worker.LOADERS), local path, env python
MODELS = [
    dict(name="dgemma-chat", id="nvidia/diffusiongemma-26B-A4B-it-NVFP4, plain chat on the running vLLM server",
         kind="openai", path="http://127.0.0.1:8000/v1/chat/completions|dgemma", py=sys.executable),
    dict(name="qwen3-vl-4b", id="Qwen/Qwen3-VL-4B-Instruct", kind="hf_chat", path=snap("Qwen/Qwen3-VL-4B-Instruct"),
         py=env("jevdrive")),
    dict(name="cosmos-reason1-7b", id="nvidia/Cosmos-Reason1-7B (Qwen2.5-VL-7B)", kind="hf_chat",
         path=str(DATA / "models/cosmos/Cosmos-Reason1-7B"), py=env("jevdrive")),
    dict(name="qwen-drive-1.0-4b", id="Qwen/Qwen-Drive-1.0-4B, VQA mode (VLM alone)", kind="qwen_drive",
         path=str(DATA / "models/Qwen-Drive-1.0-4B"), py=env("qwen-drive")),
    dict(name="internvl2-1b", id="OpenGVLab/InternVL2-1B (SimLingo's base VLM)", kind="internvl",
         path=snap("OpenGVLab/InternVL2-1B"), py=env("simlingo")),
]
# On disk but no existing env can run them as a chat model.
NOT_RUNNABLE = [
    dict(name="cosmos-reason2-8b", id="nvidia/Cosmos-Reason2-8B", path=snap("nvidia/Cosmos-Reason2-8B"),
         why="only config and tokenizer files are in the HF cache (12 MB), the weight shards were never downloaded"),
    dict(name="autovla", id="Zewei-Zhou/AutoVLA AutoVLA_PDMS_89.ckpt (Qwen2.5-VL-3B + action tokens)",
         path=snap("Zewei-Zhou/AutoVLA"),
         why="the checkpoint needs the base model directory models/Qwen2.5-VL-3B-Instruct (config, processor), which "
             "is no longer on the disk"),
    dict(name="alpamayo-1.5-10b", id="nvidia/Alpamayo-1.5-10B, VQA mode (third_party/alpamayo1.5/.venv)",
         path=snap("nvidia/Alpamayo-1.5-10B"),
         why="its loader fetches a processor from the HF hub that is not in the cache (offline: LocalEntryNotFoundError "
             "in Alpamayo1_5._build_processor; its VQA helper also needs Qwen/Qwen3-VL-2B-Instruct, not on disk)"),
    dict(name="simlingo", id="RenzKa/simlingo epoch=013.ckpt (InternVL2-1B driving policy)", path=snap("RenzKa/simlingo"),
         why="a driving-policy checkpoint that is fed through its own CARLA agent pipeline, no plain two-image chat "
             "entry point; its base VLM InternVL2-1B is run instead"),
]
COLS = [("red_recall_0-50", "ego red recall 0-50 m"), ("red_recall_0-20", "... 0-20 m"), ("red_recall_20-50", "... 20-50 m"),
        ("red_as_green_0-50", "ego red answered green"),
        ("other_fp_new", "other direction red: answered red"), ("nolight_fp", "no light: answered red"),
        ("sign_recall", "stop sign <= 25 m: yes"), ("sign_fp", "no stop sign: yes"),
        ("block_recall", "static block recall"), ("block_recall_cones", "... cones"),
        ("block_recall_vehicle", "... stopped vehicle"), ("block_fp_clear", "clear answered block")]


def load_frames(units, partial):
    """One row per answered request with both frames on disk: truth, the logged OpenJev answer, frame paths."""
    rows = []
    for u in units:
        udir = RUN / "arms" / u
        done = {p.stem: json.loads(p.read_text()).get("attempt", 1) for p in sorted(udir.glob("done/*.json"))}
        if partial:                                   # smoke tests: also the latest attempt of unfinished routes
            for r in sorted(udir.glob("attempts/*")):
                n = [int(p.name) for p in r.iterdir() if p.name.isdigit()]
                if r.name not in done and n:
                    done[r.name] = max(n)
        for rid, att in sorted(done.items()):
            a = udir / "attempts" / rid / str(att)
            if not (a / "vlm_decisions.jsonl").exists():
                continue
            for line in open(a / "vlm_decisions.jsonl"):
                try:
                    d = json.loads(line)
                except ValueError:                    # a line still being written
                    continue
                if d.get("k") != "a" or not d["ans"].get("ok"):
                    continue
                f = {c: a / "vlm_frames" / ("%08.2f_%s.jpg" % (d["t_q"], c)) for c in ("wide", "road")}
                if not all(p.exists() for p in f.values()):
                    continue
                g, ans, tl = d["gt"], d["ans"], d["gt"].get("tl")
                others = [x for x in g.get("lights", []) if x[0] != g.get("tl_id")]
                rows.append(dict(
                    id="%s/%s/%.2f" % (u, rid, d["t_q"]), src="new", unit=u, route=rid, t=d["t_q"],
                    tl=-1 if tl is None else tl, tl_dist=g["tl_dist"] if tl is not None else np.nan,
                    other_red=float(any(x[1] == 2 for x in others)), any_light=float(bool(g.get("lights"))),
                    other_green=float(any(x[1] == 0 for x in others)),
                    stop_dist=np.nan if g.get("stop_dist") is None else g["stop_dist"], has_sign_label=True,
                    block=g["block"], side=g["side"], wide=str(f["wide"]), road=str(f["road"]),
                    oj_light=ans.get("Q_light", ""), oj_sign=ans.get("Q_sign", ""), oj_block=ans.get("Q_block", ""),
                    oj_side=ans.get("Q_side", ""), oj_lat=ans.get("latency_ms", np.nan)))
    return pd.DataFrame(rows).sort_values("id").reset_index(drop=True) if rows else pd.DataFrame()


def take_sample(df, cap, seed=0):
    """Stratified sample; returns (rows, composition)."""
    rng = np.random.default_rng(seed)
    cap = cap if cap > 0 else len(df)                 # 0: every request
    block, sign = (df.block == "static_block").to_numpy(), (df.stop_dist <= 25).to_numpy()
    light = ((df.tl >= 0) & (df.tl_dist < 50)).to_numpy() & ~block & ~sign          # light-only stratum
    rest = ~(block | sign | light)
    n_rest = int(min(rest.sum(), max(cap - (block | sign | light).sum(), round(0.2 * cap))))
    n_light = int(min(light.sum(), max(cap - n_rest - (block | sign).sum(), 0)))
    pick = lambda mask, n: rng.choice(np.flatnonzero(mask), n, replace=False)   # noqa: E731
    idx = np.sort(np.concatenate([np.flatnonzero(block | sign), pick(light, n_light), pick(rest, n_rest)]))
    s = df.iloc[idx].reset_index(drop=True)
    comp = dict(available=len(df), sampled=len(s), cap=cap, seed=seed,
                static_block=dict(available=int(block.sum()), taken=int(block.sum())),
                stop_sign_25m=dict(available=int(sign.sum()), taken=int(sign.sum())),
                ego_light_50m_only=dict(available=int(light.sum()), taken=n_light),
                rest=dict(available=int(rest.sum()), taken=n_rest),
                routes=int(s.route.nunique()), units={u: int(n) for u, n in s.unit.value_counts().items()},
                ego_red_or_yellow_50m=int((s.tl.isin([1, 2]) & (s.tl_dist < 50)).sum()),
                ego_green_50m=int(((s.tl == 0) & (s.tl_dist < 50)).sum()))
    return s, comp


def box_quiet():
    """No CARLA server holds a card: latencies measured now are not under the closed-loop batch."""
    out = subprocess.run(["nvidia-smi", "--query-compute-apps=name", "--format=csv,noheader"], capture_output=True, text=True).stdout
    return "CarlaUE4" not in out


def pick_gpu():
    v = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if v:
        return v.split(",")[0]
    out = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True, check=True).stdout
    return max((int(x.split(",")[1]), x.split(",")[0].strip()) for x in out.strip().splitlines())[1]


def run_worker(m, sample_path, mdir, gpu, n):
    """Run one model's worker unless every sampled request is cached; returns a failure text or ''."""
    out = mdir / (m["name"] + ".jsonl")
    ids = {json.loads(x)["id"] for x in open(sample_path)}
    have = {r["id"] for r in map(json.loads, open(out)) if not r.get("err")} if out.exists() else set()
    if ids <= have:
        return ""
    if not Path(m["py"]).exists():
        return "env python missing: " + m["py"]
    if m["kind"] != "openai" and not Path(m["path"]).exists():
        return "model path missing: " + m["path"]
    e = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", OMP_NUM_THREADS="8",
             TOKENIZERS_PARALLELISM="false")
    cmd = [m["py"], str(WORKER), "--kind", m["kind"], "--path", m["path"], "--sample", str(sample_path), "--out", str(out),
           "--load", str(int(not box_quiet()))]
    t0 = time.time()
    with open(mdir / (m["name"] + ".log"), "a") as log:
        log.write("\n==> %s  GPU %s  %s\n" % (time.strftime("%F %T"), gpu, " ".join(cmd)))
        log.flush()
        try:
            rc = subprocess.run(cmd, env=e, stdout=log, stderr=subprocess.STDOUT, timeout=1200 + 30 * n).returncode
        except subprocess.TimeoutExpired:
            return "timed out after %d s" % (1200 + 30 * n)
    print("%s: worker exit %d after %.0f s" % (m["name"], rc, time.time() - t0), flush=True)
    meta = mdir / (m["name"] + ".meta.json")
    err = json.loads(meta.read_text()).get("error", "") if meta.exists() else ""
    return err or ("" if rc == 0 else "worker exit %d, see %s.log" % (rc, m["name"]))


def score(s, light, sign, block, side, lat, ok):
    df = s.assign(a_light=light, a_sign=sign, a_block=block, a_side=side, lat=lat, ok=ok)
    if not df.ok.any():
        return None
    R, _ = evaluate(df.reset_index(drop=True))
    return R


def score_model(s, m, mdir):
    """Readouts of one chat model from its cached replies; (R or None, counts)."""
    out = mdir / (m["name"] + ".jsonl")
    rep = {r["id"]: r for r in map(json.loads, open(out))} if out.exists() else {}
    got = [rep.get(i) for i in s.id]
    ok = np.array([bool(r) and not r.get("err") for r in got])
    ans = [parse_reply(r.get("raw", "")) if o else dict.fromkeys(OPTIONS) for r, o in zip(got, ok)]
    col = lambda q: [a[q] or "" for a in ans]   # noqa: E731
    lat = np.array([r["lat_ms"] if o else np.nan for r, o in zip(got, ok)], float)
    cnt = dict(requests=int(ok.sum()), missing=int(sum(r is None for r in got)),
               request_errors=int(sum(bool(r) and bool(r.get("err")) for r in got)),
               parse_fail_any=int(sum(o and None in a.values() for a, o in zip(ans, ok))),
               parse_fail={q: int(sum(o and a[q] is None for a, o in zip(ans, ok))) for q in OPTIONS})
    bad = [r["raw"][:200] for r, a, o in zip(got, ans, ok) if o and None in a.values()][:3]
    if bad:
        cnt["unparsed_examples"] = bad
    meta = mdir / (m["name"] + ".meta.json")
    cnt.update({k: v for k, v in (json.loads(meta.read_text()) if meta.exists() else {}).items() if k != "error"})
    R = score(s, col("Q_light"), col("Q_sign"), col("Q_block"), col("Q_side"), lat, ok)
    quiet = np.array([o and r.get("load") == 0 for r, o in zip(got, ok)], bool)
    if R is not None:
        use = quiet if quiet.sum() >= 50 else ok
        R["latency_ms"].update(n=int(use.sum()), p50=float(np.percentile(lat[use], 50)), p95=float(np.percentile(lat[use], 95)),
                               p99=float(np.percentile(lat[use], 99)), basis="quiet box" if use is quiet else "under load")
    return R, cnt


def cell(r):
    return "n/a" if r["n"] == 0 else "%s n=%d" % (fmt(r, True), r["n"])


def report(rows, comp, units):
    c = comp
    L = ["# Model comparison on the Phase A questions: %d requests, %d routes" % (c["sampled"], c["routes"]), "",
         "**Diagnostic read on few routes**: the frames come from %d routes of the shadow units (%s), intervals are "
         "route-cluster bootstraps (2000 resamples, seed 0) and are wide or degenerate where a readout has frames from "
         "one or two routes. It ranks nothing with confidence; it shows which models are worth a real test." % (
             c["routes"], ", ".join("%s %d" % kv for kv in sorted(c["units"].items()))), "",
         "Sample (seed %d, cap %d, of %d answered requests with frames): static block %d of %d, stop sign within 25 m "
         "%d of %d, ego light within 50 m (not in the two strata before) %d of %d, rest %d of %d. In the sample: ego "
         "red or yellow within 50 m %d, ego green within 50 m %d." % (
             c["seed"], c["cap"], c["available"], c["static_block"]["taken"], c["static_block"]["available"],
             c["stop_sign_25m"]["taken"], c["stop_sign_25m"]["available"], c["ego_light_50m_only"]["taken"],
             c["ego_light_50m_only"]["available"], c["rest"]["taken"], c["rest"]["available"],
             c["ego_red_or_yellow_50m"], c["ego_green_50m"]), "",
         "Registered lines (Phase A): red recall >= 80%, other-direction red answered red <= 10%, no light answered "
         "red <= 2%, stop-sign recall >= 70%, stop-sign false alarm <= 5%, static-block recall >= 80%, latency p95 "
         "<= 600 ms. Cells: estimate [95% CI] n = requests in the denominator.", "",
         "| model | " + " | ".join(t for _, t in COLS) + " | parse failures | latency p50 / p95 ms (basis, n) | peak VRAM | note |",
         "|:--|" + ":--|" * len(COLS) + ":--|:--|:--|:--|"]
    for r in rows:
        R, cnt = r.get("readouts"), r.get("counts", {})
        if R is None:
            L.append("| %s | " % r["name"] + " | ".join([""] * len(COLS)) + " | | | | %s |" % r["note"])
            continue
        pf = "logged answers" if r["name"] == "openjev" else "%d of %d (%s)" % (
            cnt["parse_fail_any"], cnt["requests"], ", ".join("%s %d" % (q[2:], n) for q, n in cnt["parse_fail"].items()))
        vram = "%.1f GB" % (cnt["peak_vram_mb"] / 1024) if cnt.get("peak_vram_mb") else "shared server"
        la = R["latency_ms"]
        L.append("| %s | " % r["name"] + " | ".join(cell(R[k]) for k, _ in COLS) + " | %s | %.0f / %.0f (%s, n=%d) | %s | %s |" % (
            pf, la["p50"], la["p95"], la.get("basis", "in the drive, under load"), la["n"], vram, r["note"]))
    L += ["", "Ego red or yellow within 50 m, split by whether a light of another approach is green at that moment "
          "(every light within 60 m is logged), with route 27043 on its own. Cells: answered red / answered green, "
          "estimate [95% route-cluster CI] n. Listed, no line.", "",
          "| model | " + " | ".join("%s, %s" % (a, b) for _, a in STRATA for _, b in SUBSETS) + " |",
          "|:--|" + ":--|" * (len(STRATA) * len(SUBSETS))]
    for r in rows:
        R = r.get("readouts")
        if R is not None:
            L.append("| %s | " % r["name"] + " | ".join(
                "%s / %s" % (cell(R["red_recall_%s%s" % (t, u)]), cell(R["red_as_green_%s%s" % (t, u)]))
                for t, _ in STRATA for u, _ in SUBSETS) + " |")
    L += ["", "Models:", ""] + ["- `%s`: %s%s" % (r["name"], r["id"], "" if r.get("readouts") else ". " + r["note"])
                               for r in rows]
    L += ["", "How to read it:", "",
          "- `openjev` is the lane's System One endpoint (DiffusionGemma with constrained answers and its own prompt); "
          "its answers and latency are the ones logged during the drive, restricted to this sample. Every other row is "
          "the plain chat prompt below, one request at a time, greedy decoding (Qwen-Drive: its released VQA decode "
          "parameters; DiffusionGemma chat: server defaults, the server rejects a temperature).",
          "- Latency is wall-clock per request including JPEG decode and image preprocessing. Basis `quiet box`: only "
          "the requests of a worker that started with no CARLA server on the box, the model alone on its card "
          "(`dgemma-chat` alone on the vLLM server); `under load`: all requests, some or all measured while the "
          "closed-loop batch ran. `openjev` is the latency logged in the drive, under the batch's load. Peak VRAM is "
          "torch's reserved peak of the worker process.",
          "- Cones / stopped vehicle: static-block recall on the routes whose scenario places cones "
          "(ConstructionObstacle*) or a stopped vehicle (Accident*, ParkedObstacle*, HazardAtSideLane*, VehicleOpensDoor*).",
          "- A question with no parseable option is no answer: it lowers recalls and cannot be a false alarm, so read "
          "the false-alarm columns together with the parse-failure column.",
          "- Requests that raised or are missing are left out of every denominator (see models.json: counts).",
          "", "Prompt (identical for every chat model):", "", "```", PROMPT, "```", "",
          "Units read: %s. Raw replies: `results/models/<name>.jsonl`; numbers: `results/models.json`." % ", ".join(units)]
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--units", nargs="+", default=UNITS, help="shadow units under runs/vlm_arb/arms (missing ones are skipped)")
    ap.add_argument("--cap", type=int, default=1500, help="sample size; 0 = every answered request with frames")
    ap.add_argument("--no-report", action="store_true", help="run the workers only (one lane job per model)")
    ap.add_argument("--score-only", action="store_true", help="no worker: the table from the cached replies")
    ap.add_argument("--tag", default="", help="suffix of the output names (models<tag>.md, models<tag>/), for smoke tests")
    ap.add_argument("--partial", action="store_true", help="also read unfinished routes (smoke tests)")
    ap.add_argument("--only", nargs="+", help="run only these models; the others are scored from their cache")
    a = ap.parse_args()
    units = [u for u in a.units if (RUN / "arms" / u).exists()]
    df = load_frames(units, a.partial)
    if not len(df):
        print("no answered requests with frames in %s" % a.units)
        return 2
    s, comp = take_sample(df, a.cap)
    out = RUN / "results"
    mdir = out / ("models" + a.tag)
    mdir.mkdir(parents=True, exist_ok=True)
    sample_path = mdir / "sample.jsonl"
    tmp = mdir / ("sample.%d.tmp" % os.getpid())       # atomic: model jobs run side by side on the same sample
    tmp.write_text("".join(json.dumps(dict(id=r.id, wide=r.wide, road=r.road)) + "\n" for r in s.itertuples()))
    tmp.replace(sample_path)
    gpu = pick_gpu()
    print("sample: %s\nGPU %s" % (json.dumps(comp), gpu), flush=True)

    rows = [dict(name="openjev", id="OpenJev System One on nvidia/diffusiongemma-26B-A4B-it-NVFP4 (answers from the logs)",
                 note="", counts=dict(requests=len(s)),
                 readouts=score(s, s.oj_light, s.oj_sign, s.oj_block, s.oj_side, s.oj_lat, np.ones(len(s), bool)))]
    for m in MODELS:
        skip = a.score_only or (a.only and m["name"] not in a.only)
        fail = "" if skip else run_worker(m, sample_path, mdir, gpu, len(s))
        if a.no_report:
            continue
        meta = mdir / (m["name"] + ".meta.json")
        fail = fail or (json.loads(meta.read_text()).get("error", "") if meta.exists() else "")
        R, cnt = score_model(s, m, mdir)
        note = ("failed: " + fail) if fail else ""
        if R is None and not fail:
            note = "not run (no cached replies)"
        elif R is not None and cnt["requests"] < len(s):
            note = (note + "; " if note else "") + "partial: %d of %d requests" % (cnt["requests"], len(s))
        rows.append(dict(name=m["name"], id=m["id"], path=m["path"], env=m["py"], kind=m["kind"], note=note, counts=cnt,
                         readouts=R))
    if a.no_report:
        return 0
    rows += [dict(name=m["name"], id=m["id"], path=m["path"], note="not runnable: " + m["why"], readouts=None)
             for m in NOT_RUNNABLE]
    md = report(rows, comp, units)
    (out / ("models%s.md" % a.tag)).write_text(md)
    write_json(out / ("models%s.json" % a.tag), dict(sample=comp, units=units, prompt=PROMPT, models=rows))
    print(md)
    return 0 if any(r["readouts"] is not None for r in rows) else 2


if __name__ == "__main__":
    sys.exit(main())
