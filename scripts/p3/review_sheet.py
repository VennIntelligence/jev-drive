"""P3 human review sheet: every exam item rendered the same way, one Markdown page per item type (user 2026-09-28).

Item types: PI = pedestrian insertion (cross-scene donor, walk-in rules, grounded standard; TTR 3 s shown; the 2 / 4 s and null variants are
the same donor and placement rule), PD = pedestrian deletion (registered target), VD = vehicle deletion (event vehicle).
Every clip: front camera, the log on the left and the edited frame on the right (PI: donor inserted; PD / VD: object
deleted), both 480 x 320, 16 frames at 5 fps over [anchor - 2 s, anchor + 1 s], a caption bar with the item id, category,
t - anchor, view gap and box PSNR. Animated WebP, a few hundred KB each.
Box PSNR: PI = the donor at its original place (x+ against the log, insertion rule); PD / VD = x+ against the log inside the
bounding box of the pixels the deletion changes (|x+ - x-| > 15), mean over frames where that region has >= 100 px.

  on the box:  $DATA_DIR/envs/jevdrive/bin/python scripts/p3/review_sheet.py render --out $DATA_DIR/runs/nq4/p3/review
  on the Mac:  python3 scripts/p3/review_sheet.py md --items <pulled items.json> [--pending ...]
"""
import argparse
import csv
import json
import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
W, H, BAR, N, STEP5 = 480, 320, 40, 16, 2          # panel size, caption bar, frames, 10 Hz -> 5 Hz stride


def _font(size):
    from PIL import ImageFont
    for p in ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
              str(Path(os.environ.get("DATA_DIR", "/")) / "envs/drivestudio/lib/python3.10/site-packages/matplotlib/mpl-data/fonts/ttf/DejaVuSans.ttf")):
        if Path(p).exists():
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def del_psnr(real_files, plus_files, minus_files):
    import numpy as np
    from PIL import Image
    vals = []
    for r, p, m in zip(real_files, plus_files, minus_files):
        R, P, M = (np.asarray(Image.open(f), np.float64) for f in (r, p, m))
        mask = np.abs(P - M).max(-1) > 15
        if mask.sum() < 100:
            continue
        ys, xs = np.nonzero(mask)
        e = (P[ys.min():ys.max() + 1, xs.min():xs.max() + 1] - R[ys.min():ys.max() + 1, xs.min():xs.max() + 1]) / 255
        vals.append(-10 * np.log10(max((e ** 2).mean(), 1e-10)))
    return float(np.mean(vals)) if vals else float("nan")


def clip(out: Path, iid: str, left, right, rel, caption):
    import numpy as np
    from PIL import Image, ImageDraw
    f1, f2 = _font(15), _font(13)
    frames = []
    for lf, rf, tr in zip(left, right, rel):
        G = Image.new("RGB", (2 * W, H + BAR), (18, 18, 18))
        for k, (f, lab) in enumerate(((lf, "log"), (rf, "edited"))):
            G.paste(Image.open(f).convert("RGB").resize((W, H), Image.LANCZOS), (k * W, 0))
            ImageDraw.Draw(G).text((k * W + 6, 4), lab, fill=(255, 255, 255), font=f1, stroke_width=2, stroke_fill=(0, 0, 0))
        d = ImageDraw.Draw(G)
        d.text((8, H + 3), f"{iid}   t - anchor {tr:+.1f} s", fill=(240, 240, 240), font=f1)
        d.text((8, H + 22), caption, fill=(200, 200, 200), font=f2)
        frames.append(G)
    p = out / "figs" / f"{iid}.webp"
    p.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(p, save_all=True, append_images=frames[1:], duration=200, loop=0, quality=62, method=6)
    return p.stat().st_size


def window(ticks, anchor_tick, step, pre_s=2.0):
    """Frames at 5 Hz from anchor - pre_s to anchor + 1 s (ticks are 20 Hz file indices = 2 x the 10 Hz frame)."""
    want = [anchor_tick + 4 * i for i in range(-int(round(pre_s * 5)), 6)]
    have = set(ticks)
    return [t for t in want if t in have]


def render(a):
    data = Path(os.environ["DATA_DIR"])
    out = a.out
    items = []
    # item category of the registered pedestrian deletion targets (research/results/nq4/p3/filter/items_A.csv)
    A = {(r["seg"], int(r["f0"])): r for r in csv.DictReader(open(REPO / "research/results/nq4/p3/filter/items_A.csv"))}
    # PD
    for sd in sorted((data / "processed/nq4_p3/scenes").glob("p3_[0-9][0-9][0-9]")):
        if not (sd / "meta.json").exists():
            continue
        m = json.loads((sd / "meta.json").read_text())
        ticks = [2 * t for t in m["frames"]]
        win = window(ticks, 2 * m["f0"], 1)
        fs = {w: [sd / w / "cams/front" / f"{t:07d}.jpg" for t in win] for w in ("real", "plus", "minus")}
        r = A.get((m["segment"], m["f0"]))
        cat = ("must-react" if r["survives"] == "True" else "react frame, driver did not slow" if int(r["n_react"]) else
               "in-lane conflict, not seen / behind lead" if int(r["n_conflict"]) else "no must-react frame (null-type)") if r else "?"
        iid = f"PD-{m['scene']:03d}"
        ps = del_psnr(fs["real"], fs["plus"], fs["minus"])
        cap = f"pedestrian deletion ({len(m['delete_tracks'])} deleted)  |  {cat}  |  view gap -  |  deleted-region PSNR {ps:.1f} dB"
        size = clip(out, iid, fs["real"], fs["minus"], [(t - 2 * m["f0"]) / 20 for t in win], cap)
        items.append({"id": iid, "type": "PD", "scene": m["key"], "segment": m["segment"], "category": cat, "view_gap": None,
                      "psnr": round(ps, 1), "n_deleted": len(m["delete_tracks"]), "bytes": size})
    # PI: cross-scene donors under the walk-in rules (runs/nq4/p3/xinsert/items; user review 2026-09-28 16:30). The
    # same-scene items of runs/nq4/p3/insert_batch were built before those rules and are retired (they appeared mid-clip)
    X = data / "runs/nq4/p3/xinsert"
    old_pi = {json.loads(p.read_text())["scene"] for p in (data / "runs/nq4/p3/insert_batch").glob("p3_*/meta.json")
              if json.loads(p.read_text()).get("chosen")}
    for md in sorted((X / "items").glob("p3_[0-9][0-9][0-9]/meta.json")):
        m = json.loads(md.read_text())
        pl = m["plan"]
        v3 = pl["variants"].get("ins3") or {}
        if pl.get("rules") != "walk-in v2 (2026-09-28 16:30)" or "traj" not in v3:
            continue                                          # rendered before the walk-in rules: not an exam item
        src = md.parent
        win = window([2 * t for t in m["frames"]], 2 * m["t_star"], 1, pre_s=pl.get("lead", 30) / 10)   # the whole walk-in
        iid = f"PI-{int(m['scene'][3:]):03d}"
        dn = pl["donor"]
        cap = (f"pedestrian insertion, cross-scene donor p3_{dn['scene']:03d}/{dn['node']}, walk-in from the kerb ({v3['walk_s']:.1f} s at "
               f"{v3['speed']:.1f} m/s), TTR 3 s  |  view gap {pl['view_gap']:.0f} deg  |  donor PSNR {dn['psnr']:.1f} dB")
        size = clip(out, iid, [src / "real/cams/front" / f"{t:07d}.jpg" for t in win], [src / "ins3/cams/front" / f"{t:07d}.jpg" for t in win],
                    [(t - 2 * m["t_star"]) / 20 for t in win], cap)
        changed = m["scene"] in old_pi
        items.append({"id": iid, "type": "PI", "scene": m["scene"], "segment": m["target_segment"],
                      "category": "must-react (TTR 2 / 3 / 4 s) + null" + ("; changed 2026-09-28: walk-in rules, cross-scene donor" if changed else ""),
                      "view_gap": pl["view_gap"], "psnr": dn["psnr"], "donor": f"p3_{dn['scene']:03d}/{dn['node']}",
                      "walk_s": v3["walk_s"], "vis_out_s": v3["vis_out_s"], "gain": m.get("gain_rgb"),
                      "shadow_frac": m.get("shadow_on_fraction", {}).get("ins3"), "changed": changed, "bytes": size})
    pi_none = [json.loads(p.read_text())["scene"] for p in sorted((X / "plan").glob("p3_*.json")) if not json.loads(p.read_text())["item"]]
    # VD
    V = data / "runs/nq4/p3veh"
    for sd in sorted((data / "processed/nq4_p3_veh/scenes").glob("p3_v*")):
        if not (sd / "meta.json").exists():
            continue
        m = json.loads((sd / "meta.json").read_text())
        ticks = [2 * t for t in m["frames"]]
        win = window(ticks, 2 * m["f0"], 1)
        fs = {w: [sd / w / "cams/front" / f"{t:07d}.jpg" for t in win] for w in ("real", "plus", "minus")}
        ps = del_psnr(fs["real"], fs["plus"], fs["minus"])
        iid = f"VD-{m['scene']:03d}"
        cap = f"vehicle deletion  |  {m.get('cat', '?')}, TTC {m.get('ttc', float('nan')):.1f} s  |  view gap -  |  deleted-region PSNR {ps:.1f} dB"
        size = clip(out, iid, fs["real"], fs["minus"], [(t - 2 * m["f0"]) / 20 for t in win], cap)
        items.append({"id": iid, "type": "VD", "scene": m["key"], "segment": m["segment"], "category": m.get("cat", "?"),
                      "view_gap": None, "psnr": round(ps, 1), "bytes": size})
    n_veh = len(json.loads((V / "scenes.json").read_text())["segments"]) if (V / "scenes.json").exists() else 0
    (out / "items.json").write_text(json.dumps({"items": items, "pi_no_donor": pi_none, "veh_total": n_veh}, indent=1))
    print(json.dumps({"items": len(items), "MB": round(sum(i["bytes"] for i in items) / 1e6, 1)}))


PAGES = {"PI": ("ped-insert", "行人插入（跨场景供体、从路边走进车道、贴地标准构造，展示到达时间 3 s 的版本）"),
         "PD": ("ped-delete", "行人删除（登记的目标事件）"),
         "VD": ("veh-delete", "车辆删除（事件车辆）")}


def _existing_verdicts(rdir: Path) -> dict:
    """id -> verdict text already on disk, keyed by item id, so a regenerate never wipes a human's keep / drop / note."""
    verdicts = {}
    for slug, _ in PAGES.values():
        p = rdir / f"{slug}.md"
        if not p.exists():
            continue
        for line in p.read_text().splitlines():
            if not line.startswith("| !["):
                continue
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            if len(cells) < 6:
                continue
            iid = cells[1].split("<br>")[0].strip()
            verdict = cells[5].strip()
            if iid and verdict:
                verdicts[iid] = verdict
    return verdicts


def md(a):
    j = json.loads(Path(a.items).read_text())
    items = j["items"]
    rdir = REPO / "research/p3-review"
    rdir.mkdir(parents=True, exist_ok=True)
    verdicts = _existing_verdicts(rdir)
    rows = {}
    for t, (slug, title) in PAGES.items():
        its = [i for i in items if i["type"] == t]
        rows[t] = len(its)
        lines = [f"# P3 人工复核：{title}", "", "[返回总表](../p3-review-sheet.md)", "",
                 "每段视频：前相机，左边是录像，右边是编辑后的画面；锚点前 2 s 到后 1 s，5 帧每秒循环。最后一列留给你填：keep / drop / 备注。", "",
                 "| 视频 | 编号 | 类别 | 视角差 | 框内 PSNR (dB) | 结论（keep / drop / 备注） |", "|:--|:--|:--|--:|--:|:--|"]
        for i in its:
            vg = "—" if i["view_gap"] is None else f"{i['view_gap']:.0f}°"
            v = verdicts.get(i["id"], "")
            lines.append(f"| ![{i['id']}](../figs/p3/review/{i['id']}.webp) | {i['id']}<br>{i['scene']} | {i['category']} | {vg} | {i['psnr']:.1f} | {v} |")
        (rdir / f"{slug}.md").write_text("\n".join(lines) + "\n")
    total = len(items)
    mb = sum(i["bytes"] for i in items) / 1e6
    idx = [f"# P3 考题人工复核总表", "",
           f"**现有 {total} 道题可看**（{a.stamp}）：行人插入 {rows['PI']}、行人删除 {rows['PD']}、车辆删除 {rows['VD']}；视频合计 {mb:.1f} MB。"
           f"用户原先预期约 60 道，这里按实际数量列。", "", a.pending, "",
           "所有题用同一个脚本、同一种版式渲染（`scripts/p3/review_sheet.py`）：前相机，左录像、右编辑后，锚点前 2 s 到后 1 s，每道题一段约 0.3 MB 的循环 WebP。"
           "请在各分页最后一列填 keep / drop / 备注。背景与规则见 [p3-exam-filter.md](p3-exam-filter.md)。", "",
           "| 分页 | 题数 | 说明 |", "|:--|--:|:--|"]
    notes = {"PI": f"同一供体另有到达时间 2 s、4 s 和车道外对照三个版本，版式相同不重复列。2026-09-28 按用户对 p3_003 的意见改为「从路边走进来」的轨迹（整段都在、先在车道外被看见 ≥ 1.5 s、不穿过任何车），类别栏标了 changed 的是按新规则重做的题；找不到合规走法的场景不出题（目前 {len(j['pi_no_donor'])} 个）",
             "PD": "类别按登记目标事件的过滤结果：must-react = 有该反应帧且司机减速；null-type = 窗口内没有该反应帧，只能当删除型对照",
             "VD": "类别是事件类型（车道内静止车、路口横穿、加塞、对向）；全部是按登记车辆规则选出的「必须反应」事件"}
    for t, (slug, title) in PAGES.items():
        idx.append(f"| [{title}](p3-review/{slug}.md) | {rows[t]} | {notes[t]} |")
    (REPO / "research/p3-review-sheet.md").write_text("\n".join(idx) + "\n")


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("render")
    p.add_argument("--out", type=Path, required=True)
    p = sp.add_parser("md")
    p.add_argument("--items", required=True)
    p.add_argument("--stamp", required=True)
    p.add_argument("--pending", default="")
    a = ap.parse_args()
    {"render": render, "md": md}[a.cmd](a)


if __name__ == "__main__":
    main()
