#!/usr/bin/env python3
"""PAI exam split (user decision 2026-10-13: PAI becomes the exam). Over the scenes every reference read covers: hard = scored zero in any
reference; control = a seeded random half of the never-zero scenes; trainable = the other never-zero half. A zero caused by infrastructure
(results-summary failure_reason other than offroad / left_corridor_laterally / collision_at_fault, e.g. an RPC crash before any drive call)
does not count as a zero; such scenes are excluded from all three lists and listed. Plain python3.

v2 rule (--freeze DIR, the committed v1 lists): membership only ever moves into hard. hard = frozen hard + every frozen control / trainable
scene that is now zero in a reference; the rest keep their frozen list; a scene not in the frozen lists goes to hard if zero, else to control
or trainable by a stable hash (sha1 of "<seed>:<scene>", even first byte = control).

Hard carve-out (user decision 2026-10-13; --hard-train-frac F --hash-seed H): hard splits by the stable hash u = first 8 bytes of
sha1("<H>:<scene>") / 2^64 into hard_train (u < F, training-eligible) and hard_exam (never trained, never used for selection). Per scene, so
a scene that moves into hard later lands by the same rule. --tally name=file counts each list's members among the scenes of a file.

  pai_exam_split.py --freeze experiments/alpasim/results/pai_exam/v1 --hard-train-frac 0.5 --hash-seed 1013 --refs ... --out ... \
      --tally turn48=<route-follow 2.4 list> turn46=<logged-turn list>
  pai_exam_split.py --refs ybase_g20=/data/runs/route_night/r281/ybase_g20 ys0_g20=... rsg=... ys1=/data/runs/onpol_long/pai281/ys1 rbsg2=... \
      --out experiments/alpasim/results/pai_exam [--seed 0]
"""
import argparse, glob, hashlib, json, os, random
from pathlib import Path

KINDS = ("offroad", "left_corridor_laterally", "collision_at_fault")


def unit(seed, scene):
    """Stable hash of a scene in [0, 1): first 8 bytes of sha1("<seed>:<scene>") / 2^64."""
    return int.from_bytes(hashlib.sha1(f"{seed}:{scene}".encode()).digest()[:8], "big") / 2.0 ** 64


def read(root):
    out = {}
    for f in glob.glob(os.path.join(root, "runs", "*", "sim", "aggregate", "results-summary.json")):
        for r in json.load(open(f))["rollouts"]:
            fr = r.get("failure_reason") or ""
            out[r["clipgt_id"]] = "infra" if (not r["score"] and fr not in KINDS) else ("zero:" + fr if not r["score"] else "ok")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refs", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--freeze", default="", help="dir of the frozen lists (v2 rule)")
    ap.add_argument("--hard-train-frac", type=float, default=0.0, help="carve hard into hard_train / hard_exam (0 = off)")
    ap.add_argument("--hash-seed", default="1013")
    ap.add_argument("--tally", nargs="*", default=[], help="name=file of scene ids: members per list")
    a = ap.parse_args()
    R = {k: read(v) for k, v in (x.split("=", 1) for x in a.refs)}
    cover = sorted(set.intersection(*[set(d) for d in R.values()]))
    infra = sorted(s for s in cover if any(d[s] == "infra" for d in R.values()))
    ok = [s for s in cover if s not in infra]
    hard = sorted(s for s in ok if any(d[s].startswith("zero") for d in R.values()))
    never = sorted(s for s in ok if s not in hard)
    if a.freeze:
        F = {n: set(Path(a.freeze, f"{n}.txt").read_text().split()) for n in ("hard", "control", "trainable")}
        hard = sorted(set(hard) | (F["hard"] & set(ok)))
        side = lambda s: "control" if hashlib.sha1(f"{a.seed}:{s}".encode()).digest()[0] % 2 == 0 else "trainable"  # noqa: E731
        put = {s: next((n for n in ("control", "trainable") if s in F[n]), None) or side(s) for s in ok if s not in hard}
        control, trainable = sorted(s for s, n in put.items() if n == "control"), sorted(s for s, n in put.items() if n == "trainable")
    else:
        rnd = random.Random(a.seed)
        sh = never[:]
        rnd.shuffle(sh)
        control, trainable = sorted(sh[:len(sh) // 2]), sorted(sh[len(sh) // 2:])
    L = dict(hard=hard, control=control, trainable=trainable, infra_excluded=infra)
    if a.hard_train_frac > 0:
        L["hard_train"] = [s for s in hard if unit(a.hash_seed, s) < a.hard_train_frac]
        L["hard_exam"] = [s for s in hard if unit(a.hash_seed, s) >= a.hard_train_frac]
    o = Path(a.out)
    o.mkdir(parents=True, exist_ok=True)
    for n, l in L.items():
        (o / f"{n}.txt").write_text("".join(s + "\n" for s in l))
    per = {k: dict(zeros=sum(d[s].startswith("zero") for s in cover), infra=sum(d[s] == "infra" for s in cover),
                   **{x: sum(d[s] == "zero:" + x for s in cover) for x in KINDS}) for k, d in R.items()}
    rec = dict(refs=dict(x.split("=", 1) for x in a.refs), seed=a.seed, freeze=a.freeze, covered=len(cover), infra_excluded=len(infra), hard=len(hard),
               control=len(control), trainable=len(trainable), per_ref=per)
    if a.hard_train_frac > 0:
        rec.update(hard_train_frac=a.hard_train_frac, hash_seed=a.hash_seed, hash_rule='first 8 bytes of sha1("<hash_seed>:<scene>") / 2^64 < frac',
                   hard_train=len(L["hard_train"]), hard_exam=len(L["hard_exam"]))
    if a.tally:
        T = {k: set(Path(v).read_text().split()) for k, v in (x.split("=", 1) for x in a.tally)}
        rec["tally"] = {k: dict(n=len(t), **{n: len(t & set(l)) for n, l in L.items()}) for k, t in T.items()}
    (o / "split.json").write_text(json.dumps(rec, indent=1))
    print(json.dumps(rec, indent=1))


if __name__ == "__main__":
    main()
