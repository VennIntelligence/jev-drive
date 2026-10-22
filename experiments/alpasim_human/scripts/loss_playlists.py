"""Loss playlists for the exam panel: per candidate checkpoint and track, the scenes where its stored read loses points, sorted by the
points weighted for the board (research/round9/loss-playlists.md says how). Mac, repo root, any python3 with pandas (tmp/pcs_local/.venv):

  # held 460 @18 reads of every tag on the box (route_mem/read/<tag>/c*/run) -> one row per tag and scene: tag, scene, score, why, n
  ssh autodl 'cd /root/autodl-tmp/ujs/jev-drive && python3 - held' < experiments/alpasim_human/scripts/loss_playlists.py > tmp/lossplay/held460_all.tsv
  ssh autodl 'cd /root/autodl-tmp/ujs/jev-drive && python3 - arena' < experiments/alpasim_human/scripts/loss_playlists.py > tmp/lossplay/arena_3.tsv
  tmp/pcs_local/.venv/bin/python experiments/alpasim_human/scripts/loss_playlists.py build --dir tmp/lossplay
        -> tmp/lossplay/spec.json (playlists: name, track, stored_tag, ids, notes), tmp/lossplay/rows.csv, research/round9/loss-playlists.md
  scp tmp/lossplay/spec.json ujs@100.108.238.8:/data/tmp/lossplay_spec.json
  ssh ujs@100.108.238.8 'cd ~/mycode/jev-drive && python3 experiments/alpasim_human/scripts/loss_playlists.py make /data/tmp/lossplay_spec.json'
        (Tokyo: Backend.new_batch(..., start=False) per playlist; starts nothing, takes no lock)

PAI (281, ACC + GAP, Harmonizer off): score = the stored pai_fix facts read (results/stored_scores.tsv, "PAI 281" ACC+GAP); value = a x (1 - score),
a = the paimix 2PL item discrimination of decision 434 / 436 (experiments/round7/results/loss_map.csv; the 6 scenes without a fit take the
median a). Recoverable = another stored ACC + GAP 281 read of ours scores 1 (stored_scores.tsv other tags, loss_map best_ag), else any serve
(loss_map best_any), else garage 0014 (loss_map garage).
nuPlan (held 460 @18, native): the panel replays public scenes, so the list is the held 460 public scenes (onpol held split) the tag loses in
its held 460 @18 read; value = a_bm x (1 - score), a_bm = the frozen board-mix discrimination (pcs_local items_boardmix, decisions 382 / 387)
of the arena held window on the same asset whose start is closest to the public scene start (|offset| in the note; median a when the asset
has no window); easy = garage 0014 scores 1 on that window. Recoverable = any held 460 @18 read of ours (204 tags) scores 1.
"""
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
PAI = {"R8K-L18-s2": "R8K-s2", "R6DAG-L18-s1": "R6DAG-s1", "NTV0-L18-s0": "NTV0-s0 (submitted p1021a)"}
NU = {"STANDFMXH-L18-s1": "B2-s1 (candidate)", "JTN-L18-s1": "JTN-s1 (submitted native)", "NPB70-L18-s0": "NPB70-s0"}
TOPN = 30


def box_held():
    import glob
    sys.path.insert(0, "experiments/alpasim_human/lib")
    from alpasim_exam import why
    for td in sorted(glob.glob("/root/autodl-tmp/ujs/runs/route_mem/read/*")):
        per = {}
        for c in glob.glob(td + "/c*/run/aggregate/results-summary.json") + glob.glob(td + "/c*/run/sim/aggregate/results-summary.json"):
            for r in json.load(open(c))["rollouts"]:
                per.setdefault(r["clipgt_id"], []).append(r)
        for s, v in per.items():
            lo = min(v, key=lambda r: r["score"] or 0)
            print("\t".join([Path(td).name, s, "%.4f" % (sum((r["score"] or 0) for r in v) / len(v)), why(lo).replace("\t", " "), str(len(v))]))


def box_arena():
    import glob
    sys.path.insert(0, "experiments/alpasim_human/lib")
    from alpasim_exam import why
    for t in NU:
        seen = set()
        for f in sorted(glob.glob("/root/autodl-tmp/ujs/runs/defect_fix/jt/arena/%s@off/c*/run/aggregate/results-summary.json" % t)):
            for r in json.load(open(f))["rollouts"]:
                if r["clipgt_id"] not in seen:
                    seen.add(r["clipgt_id"])
                    print("\t".join([t, r["clipgt_id"], "%.4f" % (r["score"] or 0), why(r).replace("\t", " ")]))


def build(d):
    import pandas as pd
    d = Path(d)
    ST = pd.read_csv(REPO / "experiments/alpasim_human/results/stored_scores.tsv", sep="\t")
    LM = pd.read_csv(REPO / "experiments/round7/results/loss_map.csv").set_index("scene")
    amed = LM.a.median()
    pai = ST[(ST.track == "pai") & (ST.config == "ACC+GAP") & ST.read.str.startswith("PAI 281")]
    best_other = lambda t, s: pai[(pai.tag != t) & (pai.scene_id == s)].sort_values("score").tail(1)  # noqa: E731
    rows, lists = [], []
    for t, nm in PAI.items():
        m = pai[pai.tag == t].set_index("scene_id")
        L = []
        for s, r in m.iterrows():
            if r.score >= 1:
                continue
            a = LM.a.get(s, float("nan"))
            a = amed if a != a else a
            bo = best_other(t, s)
            if len(bo) and bo.score.iloc[0] >= 1:
                rec = "%s 1.0" % bo.tag.iloc[0].replace("-L18", "")
            elif LM.best_ag.get(s, 0) >= 1:
                rec = "%s 1.0" % LM.best_ag_read.get(s)
            elif LM.best_any.get(s, 0) >= 1:
                rec = "other serve 1.0"
            elif LM.garage.get(s, 0) >= 1:
                rec = "garage 1.0"
            else:
                rec = ""
            fc = r.why.split(" (")[0]
            L.append(dict(track="pai", tag=t, scene=s, label=s[7:15], score=r.score, lost=1 - r.score, a=a, value=a * (1 - r.score),
                          cls=fc, why=r.why, stratum=LM.stratum.get(s, ""), recoverable=rec))
        L.sort(key=lambda x: (-x["value"], -x["lost"]))
        name = "PAI %s losses by value" % nm.split(" ")[0]
        for i, x in enumerate(L, 1):
            x.update(playlist=name, pos=i)
        lists.append(dict(name=name, track="pai", stored_tag=t, ids=[x["scene"] for x in L],
                          notes={x["scene"]: "#%d value %.2f (a %.2f x lost %.2f) %s%s" % (x["pos"], x["value"], x["a"], x["lost"], x["cls"],
                                                                                     "; fixed by " + x["recoverable"] if x["recoverable"] else "")
                                 for x in L}))
        rows += L
    # nuPlan
    H = pd.read_csv(d / "held460_all.tsv", sep="\t", names=["tag", "scene", "score", "why", "n"])
    held = set((REPO / "experiments/onpol_nuplan/results/split/held.txt").read_text().split()) if \
        (REPO / "experiments/onpol_nuplan/results/split/held.txt").is_file() else None
    S = pd.read_csv(REPO / "experiments/alpasim_proxy/results/arena/set.csv.gz").drop_duplicates("id")
    S = S[S.split == "held"]
    IB = pd.read_csv(REPO / "experiments/alpasim_proxy/results/pcs_local/items_boardmix.csv.gz", index_col=0)
    S = S[S.id.isin(IB.index)].copy()
    S["a"], S["garage"] = IB.a.reindex(S.id).values, IB.garage.reindex(S.id).values
    near = S.assign(ao=S.pub_off_s.abs()).sort_values("ao").drop_duplicates("src").set_index("src")
    bmed = IB.a.median()
    best = H.groupby("scene").apply(lambda g: g.sort_values("score").iloc[-1][["tag", "score"]], include_groups=False)
    A = pd.read_csv(d / "arena_3.tsv", sep="\t", names=["tag", "id", "score", "why"]) if (d / "arena_3.tsv").is_file() else None
    for t, nm in NU.items():
        m = H[H.tag == t].set_index("scene")
        if held is not None:
            m = m[m.index.isin(held)]
        aw = None
        if A is not None:
            aw = A[A.tag == t].merge(S[["id", "src", "a", "garage"]], on="id")
            aw["val"] = aw.a * (1 - aw.score)
        L = []
        for s, r in m.iterrows():
            if r.score >= 1 or str(r.why).startswith("failed: Waypoint"):    # route check fails before driving: every tag, not a drive
                continue
            nr = near.loc[s] if s in near.index else None
            a = float(nr.a) if nr is not None else bmed
            others = H[(H.scene == s) & (H.tag != t) & (H.score >= 1)]
            pref = [x for x in others.tag if x in ("STANDFMXH-L18-s0", "STANDFMXH-L18-s1", "JTN-L18-s1", "NTV0-L18-s0", "R8K-L18-s2",
                                                  "R6DAG-L18-s1", "NPB70-L18-s0", "RSG-L18-s0")]
            rec = ("%s 1.0" % (pref or list(others.tag))[0].replace("-L18", "") + (" (+%d reads)" % (len(others) - 1) if len(others) > 1 else "")) \
                if len(others) else ""
            ar = ""
            if aw is not None:
                w = aw[aw.src == s]
                ar = "arena: %d windows, %d lost, value %.2f" % (len(w), (w.score < 1).sum(), w.val.sum()) if len(w) else ""
            L.append(dict(track="nuplan", tag=t, scene=s, label=s[-12:], score=r.score, lost=1 - r.score, a=a, value=a * (1 - r.score),
                          cls=r.why, why=r.why, stratum=nr.stratum if nr is not None else "",
                          easy=(nr is not None and nr.garage >= 1), near_off=float(nr.pub_off_s) if nr is not None else None,
                          recoverable=rec, arena=ar))
        L.sort(key=lambda x: (-x["value"], -x["lost"]))
        name = "nuPlan %s losses by value" % nm.split(" ")[0]
        for i, x in enumerate(L, 1):
            x.update(playlist=name, pos=i)
        lists.append(dict(name=name, track="nuplan", stored_tag=t, ids=[x["scene"] for x in L],
                          notes={x["scene"]: "#%d value %.2f (a %.2f x lost %.2f)%s %s%s" % (
                              x["pos"], x["value"], x["a"], x["lost"], " easy" if x["easy"] else "", x["stratum"],
                              "; fixed by " + x["recoverable"] if x["recoverable"] else "") for x in L}))
        rows += L
    # combined top recoverable per track (the panel's batches are one track each)
    for tr, tags, lab in (("pai", PAI, "PAI"), ("nuplan", NU, "nuPlan")):
        best_row = {}
        for x in rows:
            if x["track"] == tr and x["recoverable"] and x["tag"] in tags:
                if x["scene"] not in best_row or x["value"] > best_row[x["scene"]]["value"]:
                    best_row[x["scene"]] = x
        top = sorted(best_row.values(), key=lambda x: -x["value"])[:TOPN]
        lists.append(dict(name="%s top %d recoverable losses (all candidates)" % (lab, TOPN), track=tr, stored_tag=None,
                          ids=[x["scene"] for x in top], stored_tags={x["scene"]: x["tag"] for x in top},
                          notes={x["scene"]: "#%d %s value %.2f (score %.2f) %s; fixed by %s" % (i, x["tag"].replace("-L18", ""), x["value"], x["score"],
                                                                                        x["cls"], x["recoverable"]) for i, x in enumerate(top, 1)}))
    (d / "spec.json").write_text(json.dumps(dict(playlists=lists), indent=1))
    pd.DataFrame(rows).to_csv(d / "rows.csv", index=False)
    for p in lists:
        print("%-55s %s %3d  stored=%s" % (p["name"], p["track"], len(p["ids"]), p["stored_tag"]))


def make(spec, prefix="", replace=False):
    """Tokyo: create the playlists whose name starts with `prefix` (no loop start). Existing playlists of the same name are skipped, or with
    `replace` (--replace) rebuilt in place when nothing was driven in them yet (exam.json rewritten, run dir kept)."""
    sys.path.insert(0, str(REPO / "experiments/alpasim_human/lib"))
    sys.path.insert(0, str(REPO / "lib"))
    import alpasim_exam as AE
    b = AE.AlpaSimExam()
    have = {AE.read_json(Path(r, "exam.json"), {}).get("label") for r in b.runs()}
    for p in json.load(open(spec))["playlists"]:
        if not p["name"].startswith(prefix):
            continue
        if p["name"] in have:
            run = next(r for r in b.runs() if AE.read_json(Path(r, "exam.json"), {}).get("label") == p["name"])
            if not replace or any(Path(run, "done").glob("*.json")):
                print("skip (exists%s): %s" % (", driven" * replace, p["name"]))
                continue
            ex = json.load(open(Path(run, "exam.json")))
            ex = {k: v for k, v in ex.items() if k not in ("stored_tag", "stored_tags", "notes")}
            ex.update(ids=p["ids"], notes=p["notes"], **{k: p[k] for k in ("stored_tag", "stored_tags") if p.get(k)})
            Path(run, "exam.json").write_text(json.dumps(ex, indent=1))
            print("replaced %s (%d scenes)" % (p["name"], len(p["ids"])))
            continue
        extra = dict(notes=p["notes"], **{k: p[k] for k in ("stored_tag", "stored_tags") if p.get(k)})
        print(b.new_batch("custom", ids=p["ids"], name=p["name"], start=False, extra=extra))


if __name__ == "__main__":
    a = sys.argv[1:]
    if a[:1] == ["held"]:
        box_held()
    elif a[:1] == ["arena"]:
        box_arena()
    elif a[:1] == ["build"]:
        build(a[a.index("--dir") + 1])
    elif a[:1] == ["make"]:
        make(a[1], next((x for x in a[2:] if not x.startswith("--")), ""), "--replace" in a)
    else:
        print(__doc__)
